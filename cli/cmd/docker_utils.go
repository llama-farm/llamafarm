package cmd

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/user"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/image"
	"github.com/docker/docker/api/types/network"
	"github.com/docker/docker/client"
	"github.com/docker/go-connections/nat"
)

// versionPattern matches semantic versions with or without leading "v"
// Examples: v1.0.0, v1.0.0-rc1, v2.0.0-beta.1+build.123, 1.0.0, 1.0.0-alpha
var versionPattern = regexp.MustCompile(`^v?\d+\.\d+\.\d+.*`)

// getCurrentUserGroup returns the current user:group string for Docker user mapping
func getCurrentUserGroup() string {
	currentUser, err := user.Current()
	if err != nil {
		logDebug(fmt.Sprintf("Failed to get current user, using default: %v", err))
		return ""
	}
	return fmt.Sprintf("%s:%s", currentUser.Uid, currentUser.Gid)
}

// knownComponents lists the valid component names for image URLs
var knownComponents = map[string]bool{
	"server":   true,
	"designer": true,
	"rag":      true,
	"runtime":  true,
	"models":   true,
}

// DockerPullProgress represents the progress of a Docker pull operation
type DockerPullProgress struct {
	ID      string
	Status  string
	Current int64
	Total   int64
}

// DockerSDKProgress represents the JSON progress structure from Docker SDK
type DockerSDKProgress struct {
	ID             string `json:"id"`
	Status         string `json:"status"`
	Error          string `json:"error,omitempty"`
	Progress       string `json:"progress,omitempty"`
	ProgressDetail struct {
		Current int64 `json:"current"`
		Total   int64 `json:"total"`
	} `json:"progressDetail,omitempty"`
}

// ProgressTracker tracks overall pull progress across all layers
type ProgressTracker struct {
	layers        map[string]*DockerPullProgress
	totalBytes    int64
	doneBytes     int64
	lastUpdate    time.Time
	lastDoneBytes int64 // Track previous doneBytes for rate calculation
	startTime     time.Time
}

// NewProgressTracker creates a new progress tracker
func NewProgressTracker() *ProgressTracker {
	now := time.Now()
	return &ProgressTracker{
		layers:        make(map[string]*DockerPullProgress),
		lastUpdate:    now,
		startTime:     now,
		lastDoneBytes: 0,
	}
}

// Update updates the progress tracker with new layer information
func (pt *ProgressTracker) Update(progress *DockerPullProgress) {
	if progress.ID == "" {
		return
	}

	// Store the layer progress
	pt.layers[progress.ID] = progress

	// Recalculate totals
	pt.recalculate()
	pt.lastUpdate = time.Now()
}

// recalculate recalculates total and done bytes across all layers for transfer rate calculation
func (pt *ProgressTracker) recalculate() {
	pt.totalBytes = 0
	pt.doneBytes = 0

	for _, layer := range pt.layers {
		if layer.Total > 0 {
			pt.totalBytes += layer.Total

			// Handle different layer states for transfer rate calculation
			switch layer.Status {
			case "Download complete", "Verifying Checksum", "Extracting", "Pull complete":
				// Layer download is complete, count full size towards transfer rate
				pt.doneBytes += layer.Total
			case "Downloading":
				// Layer is still downloading, use current progress for transfer rate
				pt.doneBytes += layer.Current
			default:
				// For other statuses, use current progress if available
				if layer.Current > 0 {
					pt.doneBytes += layer.Current
				}
			}
		}
	}
}

// GetProgress returns the overall progress percentage (0-100) based on layer completion
func (pt *ProgressTracker) GetProgress() float64 {
	if len(pt.layers) == 0 {
		return 0.0
	}

	completedLayers := 0
	totalLayers := len(pt.layers)

	for _, layer := range pt.layers {
		switch layer.Status {
		case "Download complete", "Verifying Checksum", "Extracting", "Pull complete":
			completedLayers++
		}
	}

	// Calculate progress based on completed layers
	progress := float64(completedLayers) / float64(totalLayers) * 100

	// Ensure progress stays within bounds
	if progress > 100.0 {
		progress = 100.0
	} else if progress < 0.0 {
		progress = 0.0
	}

	return progress
}

// GetTransferRate returns the transfer rate in bytes per second
func (pt *ProgressTracker) GetTransferRate() float64 {
	// Calculate rate based on total progress over total time elapsed
	elapsed := time.Since(pt.startTime).Seconds()
	if elapsed < 1.0 { // Avoid division by very small numbers
		return 0
	}
	return float64(pt.doneBytes) / elapsed
}

// FormatTransferRate formats transfer rate in human-readable format
func (pt *ProgressTracker) FormatTransferRate() string {
	rate := pt.GetTransferRate()

	// Cap unrealistic rates to prevent display issues
	maxReasonableRate := float64(1024 * 1024 * 1024) // 1 GB/s max
	if rate > maxReasonableRate {
		rate = maxReasonableRate
	}

	if rate < 1024 {
		return fmt.Sprintf("%.1f B/s", rate)
	} else if rate < 1024*1024 {
		return fmt.Sprintf("%.1f KB/s", rate/1024)
	} else if rate < 1024*1024*1024 {
		return fmt.Sprintf("%.1f MB/s", rate/(1024*1024))
	} else {
		return fmt.Sprintf("%.1f GB/s", rate/(1024*1024*1024))
	}
}

// DisplayProgress displays a single-line progress update
func (pt *ProgressTracker) DisplayProgress(imageName string) {
	progress := pt.GetProgress()
	rate := pt.FormatTransferRate()

	// Count layers in different states for more informative display
	downloading := 0
	extracting := 0
	complete := 0
	total := len(pt.layers)

	for _, layer := range pt.layers {
		switch layer.Status {
		case "Downloading":
			downloading++
		case "Extracting":
			extracting++
		case "Download complete", "Pull complete":
			complete++
		}
	}

	// Use \r to overwrite the current line
	if total > 1 {
		OutputProgress("\rPulling %s: %.1f%% (%s) [%d/%d layers]    ",
			imageName, progress, rate, complete+extracting, total)
	} else {
		OutputProgress("\rPulling %s: %.1f%% (%s)    ", imageName, progress, rate)
	}
}

// createDockerClient creates a new Docker client with API version negotiation
func createDockerClient() (*client.Client, error) {
	cli, err := client.NewClientWithOpts(client.FromEnv, client.WithAPIVersionNegotiation())
	if err != nil {
		return nil, fmt.Errorf("failed to create Docker client: %v", err)
	}
	return cli, nil
}

// promptUserConfirmation prompts the user for a yes/no confirmation
func promptUserConfirmation(message string) bool {
	// Check if running in CI or non-interactive environment
	if isNonInteractiveEnvironment() {
		fmt.Fprintf(os.Stderr, "%s (skipping in non-interactive environment)\n", message)
		return false
	}

	fmt.Fprintf(os.Stderr, "%s (y/N): ", message)
	scanner := bufio.NewScanner(os.Stdin)
	if scanner.Scan() {
		response := strings.ToLower(strings.TrimSpace(scanner.Text()))
		return response == "y" || response == "yes"
	}
	return false
}

// isNonInteractiveEnvironment checks if we're running in a CI or non-interactive environment
func isNonInteractiveEnvironment() bool {
	// Check common CI environment variables
	ciEnvVars := []string{
		"CI", "CONTINUOUS_INTEGRATION", "BUILD_NUMBER", "JENKINS_URL",
		"GITHUB_ACTIONS", "GITLAB_CI", "CIRCLECI", "TRAVIS", "BUILDKITE",
		"AZURE_PIPELINES", "TEAMCITY_VERSION", "BAMBOO_BUILD_NUMBER",
	}

	for _, envVar := range ciEnvVars {
		if os.Getenv(envVar) != "" {
			return true
		}
	}

	// Check if stdin is not a terminal (piped input)
	if fileInfo, err := os.Stdin.Stat(); err == nil {
		return (fileInfo.Mode() & os.ModeCharDevice) == 0
	}

	return false
}

// checkWindowsVirtualization checks if virtualization is supported and enabled on Windows
func checkWindowsVirtualization() error {
	// Check hardware virtualization support using systeminfo
	cmd := exec.Command("systeminfo")
	output, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("failed to run systeminfo command: %v", err)
	}

	outputStr := string(output)

	// Check for Hyper-V requirements section
	if !strings.Contains(outputStr, "Hyper-V Requirements") {
		return fmt.Errorf("unable to determine virtualization support. Your system may not support Hyper-V")
	}

	// Check if virtualization is enabled in firmware
	if !strings.Contains(outputStr, "Virtualization Enabled in Firmware: Yes") {
		return fmt.Errorf("hardware virtualization is not enabled in BIOS/UEFI. Please enable Intel VT-x or AMD-V in your system settings")
	}

	// Check if SLAT is supported (required for Hyper-V)
	if !strings.Contains(outputStr, "Second Level Address Translation: Yes") {
		return fmt.Errorf("Second Level Address Translation (SLAT) is not supported. This is required for Docker Desktop")
	}

	// Check if Data Execution Prevention is available
	if !strings.Contains(outputStr, "Data Execution Prevention Available: Yes") {
		return fmt.Errorf("Data Execution Prevention (DEP) is not available. This is required for Docker Desktop")
	}

	return nil
}

// checkWindowsContainerSupport checks if Windows has the necessary features for Docker
func checkWindowsContainerSupport() error {
	// Check if WSL2 is available (preferred backend for Docker Desktop)
	cmd := exec.Command("wsl", "--status")
	if err := cmd.Run(); err == nil {
		fmt.Fprintln(os.Stderr, "WSL2 is available and will be used as Docker backend.")
		return nil
	}

	// If WSL2 is not available, check Hyper-V
	cmd = exec.Command("powershell", "-Command", "Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-All | Select-Object State")
	output, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("unable to check Windows features. please ensure you have administrator privileges")
	}

	if strings.Contains(string(output), "Enabled") {
		fmt.Fprintln(os.Stderr, "Hyper-V is enabled and will be used as Docker backend.")
		return nil
	}

	return fmt.Errorf("neither WSL2 nor Hyper-V is properly configured. Docker Desktop requires one of these to be enabled")
}

// installDockerOnWindows attempts to install Docker Desktop using winget
func installDockerOnWindows() error {
	fmt.Fprintln(os.Stderr, "Installing Docker Desktop using winget...")
	cmd := exec.Command("winget", "install", "docker.dockerdesktop")
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to install Docker Desktop: %v", err)
	}

	fmt.Fprintln(os.Stderr, "Docker Desktop installation completed. You may need to restart your terminal or log out/in for changes to take effect.")
	return nil
}

// ensureDockerAvailable checks whether docker is available by creating a client
func ensureDockerAvailable() error {
	cli, err := createDockerClient()
	if err != nil {
		// On Windows, offer to install Docker automatically
		if runtime.GOOS == "windows" {
			fmt.Fprintln(os.Stderr, "Docker is not available.")

			// First check if the system supports virtualization
			fmt.Fprintln(os.Stderr, "Checking virtualization support...")
			if virtErr := checkWindowsVirtualization(); virtErr != nil {
				return fmt.Errorf("Docker Desktop requires virtualization support: %v", virtErr)
			}

			// Check if container support (WSL2 or Hyper-V) is available
			fmt.Fprintln(os.Stderr, "Checking container backend support...")
			if containerErr := checkWindowsContainerSupport(); containerErr != nil {
				fmt.Fprintf(os.Stderr, "Warning: %v\n", containerErr)
				fmt.Fprintln(os.Stderr, "Docker Desktop may prompt you to enable WSL2 or Hyper-V during installation.")
			}

			if promptUserConfirmation("Would you like me to install Docker Desktop using winget?") {
				if installErr := installDockerOnWindows(); installErr != nil {
					return fmt.Errorf("failed to install Docker: %v", installErr)
				}

				// Wait a moment for installation to complete
				fmt.Fprintln(os.Stderr, "Waiting for Docker installation to complete...")
				time.Sleep(3 * time.Second)

				// Retry Docker availability check
				fmt.Fprintln(os.Stderr, "Retrying Docker connection...")
				return ensureDockerAvailableRetry()
			} else {
				return fmt.Errorf("Docker is not available. You can install Docker Desktop using:\n  winget install docker.dockerdesktop\n\nOr download from https://docker.com/get-started: %v", err)
			}
		} else {
			return fmt.Errorf("Docker is not available. Please install Docker and try again: %v", err)
		}
	}
	defer cli.Close()

	// Try to ping the Docker daemon
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	_, err = cli.Ping(ctx)
	if err != nil {
		return fmt.Errorf("docker daemon is not running: %v", err)
	}
	return nil
}

// ensureDockerAvailableRetry is a helper function for retrying Docker availability without prompting for installation
func ensureDockerAvailableRetry() error {
	cli, err := createDockerClient()
	if err != nil {
		return fmt.Errorf("Docker is still not available after installation. You may need to restart your terminal or start Docker Desktop manually: %v", err)
	}
	defer cli.Close()

	// Try to ping the Docker daemon
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	_, err = cli.Ping(ctx)
	if err != nil {
		return fmt.Errorf("Docker is installed but the daemon is not running. Please start Docker Desktop and try again: %v", err)
	}

	fmt.Fprintln(os.Stderr, "Docker is now available!")
	return nil
}

// imageExists checks if a Docker image exists locally
func imageExists(imageName string) bool {
	ctx := context.Background()
	cli, err := createDockerClient()
	if err != nil {
		if debug {
			logDebug(fmt.Sprintf("Failed to create Docker client for image check: %v", err))
		}
		return false
	}
	defer cli.Close()

	// List images and check if our image exists
	images, err := cli.ImageList(ctx, image.ListOptions{})
	if err != nil {
		if debug {
			logDebug(fmt.Sprintf("Failed to list images: %v", err))
		}
		return false
	}

	for _, img := range images {
		for _, tag := range img.RepoTags {
			if tag == imageName {
				return true
			}
		}
	}

	return false
}

// pullImage pulls a docker image using the Docker SDK with progress tracking
// Only pulls if the image doesn't exist locally
func pullImage(imageName string) error {
	return pullImageWithForce(imageName, false)
}

// PullImageForce pulls a docker image using the Docker SDK with progress tracking
// Always pulls even if image exists locally (public function for external use)
func PullImageForce(imageName string) error {
	return pullImageWithForce(imageName, false)
}

// pullImageWithForce pulls a docker image using the Docker SDK with progress tracking
// If force is true, pulls even if image exists locally
func pullImageWithForce(imageName string, force bool) error {
	// Check if image exists locally first (unless forcing)
	if !force && imageExists(imageName) {
		if debug {
			logDebug(fmt.Sprintf("Image %s already exists locally, skipping pull", imageName))
		}
		return nil
	}

	ctx := context.Background()
	cli, err := createDockerClient()
	if err != nil {
		return err
	}
	defer cli.Close()

	// Extract image name for display (remove registry/tag parts for brevity)
	imageParts := strings.Split(imageName, "/")
	displayName := imageParts[len(imageParts)-1]
	if tagIdx := strings.Index(displayName, ":"); tagIdx > 0 {
		displayName = displayName[:tagIdx]
	}

	OutputProgress("Pulling image: %s\n", imageName)

	// Pull the image
	out, err := cli.ImagePull(ctx, imageName, image.PullOptions{})
	if err != nil {
		return fmt.Errorf("failed to pull image: %v", err)
	}
	defer out.Close()

	// Create progress tracker
	tracker := NewProgressTracker()
	lastProgressTime := time.Now()

	// Channel to collect all output for debug logging
	var allOutput strings.Builder

	// Track whether we've seen any progress information
	hasProgress := false

	// Read and process the JSON stream
	decoder := json.NewDecoder(out)
	for {
		var progress DockerSDKProgress
		if err := decoder.Decode(&progress); err != nil {
			if err == io.EOF {
				break
			}
			// Log decode errors but continue
			if debug {
				logDebug(fmt.Sprintf("Error decoding progress: %v", err))
			}
			continue
		}

		// Log all progress for debugging
		if debug {
			progressJSON, _ := json.Marshal(progress)
			logDebug(string(progressJSON) + "\n")
		}

		// Handle errors in the progress stream
		if progress.Error != "" {
			return fmt.Errorf("docker pull failed: %s", progress.Error)
		}

		// Convert SDK progress to our internal format
		if progress.ID != "" {
			hasProgress = true
			dockerProgress := &DockerPullProgress{
				ID:      progress.ID,
				Status:  progress.Status,
				Current: progress.ProgressDetail.Current,
				Total:   progress.ProgressDetail.Total,
			}

			// For layers without progress details but with status updates,
			// preserve the previous total if we had one
			if dockerProgress.Total == 0 {
				if existingLayer, exists := tracker.layers[progress.ID]; exists && existingLayer.Total > 0 {
					dockerProgress.Total = existingLayer.Total
				}
			}

			tracker.Update(dockerProgress)

			// Throttle display updates to avoid overwhelming the terminal and reduce fluctuations
			if time.Since(lastProgressTime) > 500*time.Millisecond {
				tracker.DisplayProgress(displayName)
				lastProgressTime = time.Now()
			}
		} else if !hasProgress {
			// If no progress info yet, show basic status for certain statuses
			if strings.Contains(progress.Status, "Downloading") ||
				strings.Contains(progress.Status, "Extracting") ||
				strings.Contains(progress.Status, "Pull complete") {
				OutputProgress("\rPulling %s...    ", displayName)
			}
		}
	}

	// Clear the progress line and show completion
	OutputProgress("\r%s\r", strings.Repeat(" ", 80))
	OutputSuccess("✓ Pulled %s successfully\n", displayName)

	// Log all output to debug if enabled
	if debug {
		output := allOutput.String()
		if len(output) > 0 {
			logDebug(fmt.Sprintf("docker pull output: %s", output))
		}
	}

	return nil
}

func containerExists(name string) bool {
	ctx := context.Background()
	cli, err := createDockerClient()
	if err != nil {
		if debug {
			logDebug(fmt.Sprintf("Failed to create Docker client: %v", err))
		}
		return false
	}
	defer cli.Close()

	containers, err := cli.ContainerList(ctx, container.ListOptions{All: true})
	if err != nil {
		if debug {
			logDebug(fmt.Sprintf("Failed to list containers: %v", err))
		}
		return false
	}

	for _, c := range containers {
		for _, containerName := range c.Names {
			// Container names from the API include leading slash
			cleanName := strings.TrimPrefix(containerName, "/")
			if cleanName == name {
				return true
			}
		}
	}
	return false
}

func isContainerRunning(name string) bool {
	ctx := context.Background()
	cli, err := createDockerClient()
	if err != nil {
		if debug {
			logDebug(fmt.Sprintf("Failed to create Docker client: %v", err))
		}
		return false
	}
	defer cli.Close()

	containers, err := cli.ContainerList(ctx, container.ListOptions{})
	if err != nil {
		if debug {
			logDebug(fmt.Sprintf("Failed to list running containers: %v", err))
		}
		return false
	}

	for _, c := range containers {
		for _, containerName := range c.Names {
			// Container names from the API include leading slash
			cleanName := strings.TrimPrefix(containerName, "/")
			if cleanName == name {
				return true
			}
		}
	}
	return false
}

// resolveImageTag determines the appropriate Docker image tag based on version and environment variables
func resolveImageTag(component string, defaultTag string) string {
	// Check for component-specific environment variable first
	componentEnvVar := fmt.Sprintf("LF_%s_IMAGE_TAG", strings.ToUpper(component))
	if tag := strings.TrimSpace(os.Getenv(componentEnvVar)); tag != "" {
		return tag
	}

	// Check for global override
	if tag := strings.TrimSpace(os.Getenv("LF_IMAGE_TAG")); tag != "" {
		return tag
	}

	// Use version-based logic
	version := strings.TrimSpace(Version)
	if version == "" {
		return defaultTag
	}

	// Handle version patterns: vX.X.X, X.X.X, with optional suffixes
	if versionPattern.MatchString(version) {
		// Ensure version has "v" prefix for Docker tag consistency
		if !strings.HasPrefix(version, "v") {
			return "v" + version
		}
		return version
	}

	// Handle dev versions
	if version == "dev" {
		return "latest"
	}

	// Fallback to default
	return defaultTag
}

// getImageURL constructs the full Docker image URL for a given component
func getImageURL(component string) (string, error) {
	if !knownComponents[component] {
		return "", fmt.Errorf("unknown component '%s'; valid components are: %s",
			component, getKnownComponentsList())
	}

	// If LF_CONTAINER_{component} is set, return that directly
	containerEnvVar := fmt.Sprintf("LF_CONTAINER_%s", strings.ToUpper(component))
	if val := strings.TrimSpace(os.Getenv(containerEnvVar)); val != "" {
		return val, nil
	}

	baseURL := "ghcr.io/llama-farm/llamafarm"
	tag := resolveImageTag(component, "latest")
	return fmt.Sprintf("%s/%s:%s", baseURL, component, tag), nil
}

// getKnownComponentsList returns a comma-separated list of known components
func getKnownComponentsList() string {
	components := make([]string, 0, len(knownComponents))
	for component := range knownComponents {
		components = append(components, component)
	}
	return strings.Join(components, ", ")
}

// ---- Generic container utilities ----

// ensureHostDockerInternal adds host.docker.internal:host-gateway mapping for Linux containers
// This provides the equivalent of --add-host=host.docker.internal:host-gateway
func ensureHostDockerInternal(addHosts []string) []string {
	// On Linux, Docker doesn't automatically provide host.docker.internal
	// so we need to add it manually using host-gateway
	if runtime.GOOS == "linux" {
		// Check if host.docker.internal mapping already exists
		hasHostDockerInternal := false
		for _, host := range addHosts {
			if strings.Contains(host, "host.docker.internal") {
				hasHostDockerInternal = true
				break
			}
		}

		// Add the mapping if it doesn't exist
		if !hasHostDockerInternal {
			if addHosts == nil {
				addHosts = make([]string, 0, 1)
			}
			addHosts = append(addHosts, "host.docker.internal:host-gateway")
		}
	}
	return addHosts
}

type PortSpec struct {
	Container int
	Protocol  string
}

type PortMapping struct {
	Host      int
	Container int
	Protocol  string
}

type ContainerRunSpec struct {
	Name           string
	Image          string
	DynamicPublish bool
	StaticPorts    []PortMapping
	Env            map[string]string
	Volumes        []string
	AddHosts       []string
	Labels         map[string]string
	Workdir        string
	Entrypoint     []string
	Cmd            []string
	User           string // Docker user specification (e.g., "1000:1000")
}

type PortResolutionPolicy struct {
	PreferredHostPort int
	Forced            bool
}

func removeContainer(name string) error {
	if !containerExists(name) {
		return nil
	}

	ctx := context.Background()
	cli, err := createDockerClient()
	if err != nil {
		return fmt.Errorf("failed to create Docker client: %v", err)
	}
	defer cli.Close()

	// Find the container by name
	containers, err := cli.ContainerList(ctx, container.ListOptions{All: true})
	if err != nil {
		return fmt.Errorf("failed to list containers: %v", err)
	}

	var containerID string
	for _, c := range containers {
		for _, containerName := range c.Names {
			cleanName := strings.TrimPrefix(containerName, "/")
			if cleanName == name {
				containerID = c.ID
				break
			}
		}
		if containerID != "" {
			break
		}
	}

	if containerID == "" {
		return nil // Container not found, nothing to remove
	}

	// Remove the container (force=true to stop and remove)
	removeOptions := container.RemoveOptions{
		Force: true,
	}

	if err := cli.ContainerRemove(ctx, containerID, removeOptions); err != nil {
		return fmt.Errorf("failed to remove container %s: %v", name, err)
	}

	if debug {
		logDebug(fmt.Sprintf("Successfully removed container: %s (ID: %s)", name, containerID))
	}
	return nil
}

func isHostPortAvailable(port int) bool {
	l, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return false
	}
	_ = l.Close()
	return true
}

// StartContainerDetachedWithPolicy starts a container with either static port mapping
// or dynamic published ports based on availability and the provided policy.
// Returns a map of containerPort->hostPort that were published.
func StartContainerDetachedWithPolicy(spec ContainerRunSpec, policy *PortResolutionPolicy) (map[int]int, error) {
	if err := ensureDockerAvailable(); err != nil {
		return nil, err
	}
	if strings.TrimSpace(spec.Name) == "" || strings.TrimSpace(spec.Image) == "" {
		return nil, errors.New("container name and image are required")
	}

	// Remove stale container if exists and not running
	if containerExists(spec.Name) && !isContainerRunning(spec.Name) {
		fmt.Fprintln(os.Stderr, "Removing existing container to refresh image/args...")
		if err := removeContainer(spec.Name); err != nil {
			return nil, fmt.Errorf("failed to remove existing container %s: %w", spec.Name, err)
		}
	}

	// If already running, do nothing and return current published ports
	if isContainerRunning(spec.Name) {
		ports, _ := GetPublishedPorts(spec.Name)
		resolved := make(map[int]int)
		for key, val := range ports {
			// key like "80/tcp"; extract container port
			parts := strings.Split(key, "/")
			if len(parts) > 0 {
				if cp, err := strconv.Atoi(parts[0]); err == nil {
					if hp, err2 := strconv.Atoi(val); err2 == nil {
						resolved[cp] = hp
					}
				}
			}
		}
		return resolved, nil
	}

	// Pull image best-effort (captured)
	_ = pullImage(spec.Image)

	ctx := context.Background()
	cli, err := createDockerClient()
	if err != nil {
		return nil, fmt.Errorf("failed to create Docker client: %v", err)
	}
	defer cli.Close()

	// Prepare container configuration
	config := &container.Config{
		Image:      spec.Image,
		Env:        make([]string, 0, len(spec.Env)),
		Labels:     spec.Labels,
		WorkingDir: spec.Workdir,
		Cmd:        spec.Cmd,
		User:       spec.User,
	}

	// Add environment variables
	for k, v := range spec.Env {
		config.Env = append(config.Env, fmt.Sprintf("%s=%s", k, v))
	}

	// Set entrypoint if specified
	if len(spec.Entrypoint) > 0 {
		config.Entrypoint = spec.Entrypoint
	}

	// Prepare host configuration
	hostConfig := &container.HostConfig{
		Binds:      spec.Volumes,
		ExtraHosts: ensureHostDockerInternal(spec.AddHosts),
		AutoRemove: false,
	}

	// Handle port configuration
	exposedPorts := make(nat.PortSet)
	portBindings := make(nat.PortMap)

	useDynamic := false
	if policy != nil && policy.PreferredHostPort > 0 && len(spec.StaticPorts) > 0 {
		if isHostPortAvailable(policy.PreferredHostPort) {
			for _, pm := range spec.StaticPorts {
				hostPort := policy.PreferredHostPort
				if pm.Host > 0 {
					hostPort = pm.Host
				}
				protocol := pm.Protocol
				if protocol == "" {
					protocol = "tcp"
				}

				port, err := nat.NewPort(protocol, strconv.Itoa(pm.Container))
				if err != nil {
					return nil, fmt.Errorf("invalid port specification: %v", err)
				}
				exposedPorts[port] = struct{}{}
				portBindings[port] = []nat.PortBinding{
					{
						HostPort: strconv.Itoa(hostPort),
					},
				}
			}
		} else {
			if policy.Forced {
				return nil, fmt.Errorf("port %d is already in use", policy.PreferredHostPort)
			}
			useDynamic = true
		}
	} else {
		useDynamic = true
	}

	if useDynamic {
		hostConfig.PublishAllPorts = true
		// For dynamic ports, we need to expose the static ports if any
		for _, pm := range spec.StaticPorts {
			protocol := pm.Protocol
			if protocol == "" {
				protocol = "tcp"
			}
			port, err := nat.NewPort(protocol, strconv.Itoa(pm.Container))
			if err != nil {
				return nil, fmt.Errorf("invalid port specification: %v", err)
			}
			exposedPorts[port] = struct{}{}
		}
	}

	config.ExposedPorts = exposedPorts
	hostConfig.PortBindings = portBindings

	// Create the container
	resp, err := cli.ContainerCreate(ctx, config, hostConfig, nil, nil, spec.Name)
	if err != nil {
		return nil, fmt.Errorf("failed to create container: %v", err)
	}

	// Start the container
	if err := cli.ContainerStart(ctx, resp.ID, container.StartOptions{}); err != nil {
		return nil, fmt.Errorf("failed to start container: %v", err)
	}

	if debug {
		logDebug(fmt.Sprintf("Successfully started container: %s (ID: %s)", spec.Name, resp.ID))
	}

	// Resolve published ports
	published, err := GetPublishedPorts(spec.Name)
	if err != nil {
		return nil, err
	}
	resolved := make(map[int]int)
	for key, val := range published {
		parts := strings.Split(key, "/")
		if len(parts) > 0 {
			if cp, err := strconv.Atoi(parts[0]); err == nil {
				if hp, err2 := strconv.Atoi(val); err2 == nil {
					resolved[cp] = hp
				}
			}
		}
	}
	return resolved, nil
}

// GetPublishedPorts returns a map like "80/tcp" -> "49154"
func GetPublishedPorts(name string) (map[string]string, error) {
	ctx := context.Background()
	cli, err := createDockerClient()
	if err != nil {
		return nil, fmt.Errorf("failed to create Docker client: %v", err)
	}
	defer cli.Close()

	// Find the container by name
	containers, err := cli.ContainerList(ctx, container.ListOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to list containers: %v", err)
	}

	var containerID string
	for _, c := range containers {
		for _, containerName := range c.Names {
			cleanName := strings.TrimPrefix(containerName, "/")
			if cleanName == name {
				containerID = c.ID
				break
			}
		}
		if containerID != "" {
			break
		}
	}

	if containerID == "" {
		return nil, fmt.Errorf("container %s not found", name)
	}

	// Get container details
	containerJSON, err := cli.ContainerInspect(ctx, containerID)
	if err != nil {
		return nil, fmt.Errorf("failed to inspect container %s: %v", name, err)
	}

	res := make(map[string]string)
	for port, bindings := range containerJSON.NetworkSettings.Ports {
		if len(bindings) > 0 {
			// Take the first binding if multiple exist
			res[string(port)] = bindings[0].HostPort
		}
	}

	if debug {
		logDebug(fmt.Sprintf("Published ports for %s: %v", name, res))
	}
	return res, nil
}

func WaitForReadiness(ctx context.Context, check func() error, interval time.Duration) error {
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-t.C:
			if err := check(); err == nil {
				return nil
			}
		}
	}
}

func HTTPGetReady(url string) func() error {
	return func() error {
		req, err := http.NewRequest(http.MethodGet, url, nil)
		if err != nil {
			return err
		}
		client := &http.Client{Timeout: 1500 * time.Millisecond}
		resp, err := client.Do(req)
		if err != nil {
			return err
		}
		defer resp.Body.Close()
		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			return nil
		}
		return fmt.Errorf("status %d", resp.StatusCode)
	}
}

// ---- Network Management with Docker SDK ----

// NetworkManager handles Docker network creation and management using Docker SDK
type NetworkManager struct {
	networkName string
}

// NewNetworkManager creates a new network manager instance
func NewNetworkManager() *NetworkManager {
	// Create network name with timestamp for uniqueness
	timestamp := time.Now().Format("20060102-150405")
	networkName := fmt.Sprintf("llamafarm-%s", timestamp)

	return &NetworkManager{
		networkName: networkName,
	}
}

// GetNetworkName returns the current network name
func (nm *NetworkManager) GetNetworkName() string {
	return nm.networkName
}

// EnsureNetwork creates the Docker network if it doesn't exist using Docker SDK
func (nm *NetworkManager) EnsureNetwork() error {
	ctx := context.Background()
	cli, err := createDockerClient()
	if err != nil {
		return err
	}
	defer cli.Close()

	// Check if network already exists
	networks, err := cli.NetworkList(ctx, network.ListOptions{})
	if err != nil {
		return fmt.Errorf("failed to list networks: %v", err)
	}

	for _, net := range networks {
		if net.Name == nm.networkName {
			if debug {
				logDebug(fmt.Sprintf("Network %s already exists", nm.networkName))
			}
			return nil
		}
	}

	// Create the network
	if debug {
		logDebug(fmt.Sprintf("Creating Docker network: %s", nm.networkName))
	}

	_, err = cli.NetworkCreate(ctx, nm.networkName, network.CreateOptions{
		Driver: "bridge",
		Labels: map[string]string{
			"llamafarm.managed": "true",
		},
	})

	if err != nil {
		return fmt.Errorf("failed to create Docker network: %v", err)
	}

	return nil
}

// CleanupNetwork removes the Docker network if it exists using Docker SDK
func (nm *NetworkManager) CleanupNetwork() error {
	ctx := context.Background()
	cli, err := createDockerClient()
	if err != nil {
		return err
	}
	defer cli.Close()

	// Find the network
	networks, err := cli.NetworkList(ctx, network.ListOptions{})
	if err != nil {
		return fmt.Errorf("failed to list networks: %v", err)
	}

	var networkID string
	for _, net := range networks {
		if net.Name == nm.networkName {
			networkID = net.ID
			break
		}
	}

	if networkID == "" {
		return nil // Network doesn't exist
	}

	if debug {
		logDebug(fmt.Sprintf("Removing Docker network: %s", nm.networkName))
	}

	err = cli.NetworkRemove(ctx, networkID)
	if err != nil {
		return fmt.Errorf("failed to remove Docker network: %v", err)
	}

	return nil
}

// ---- Enhanced Container Management ----

// IsContainerRunning checks if a container with the given name is currently running using Docker SDK
func IsContainerRunning(name string) bool {
	return isContainerRunning(name)
}

// StopAndRemoveContainer stops and removes a container using Docker SDK
func StopAndRemoveContainer(name string) error {
	return removeContainer(name)
}

// GetContainerLogs returns recent logs from a container using Docker SDK
func GetContainerLogs(name string, lines int) (string, error) {
	ctx := context.Background()
	cli, err := createDockerClient()
	if err != nil {
		return "", fmt.Errorf("failed to create Docker client: %v", err)
	}
	defer cli.Close()

	// Find the container by name
	containers, err := cli.ContainerList(ctx, container.ListOptions{All: true})
	if err != nil {
		return "", fmt.Errorf("failed to list containers: %v", err)
	}

	var containerID string
	for _, c := range containers {
		for _, containerName := range c.Names {
			cleanName := strings.TrimPrefix(containerName, "/")
			if cleanName == name {
				containerID = c.ID
				break
			}
		}
		if containerID != "" {
			break
		}
	}

	if containerID == "" {
		return "", fmt.Errorf("container %s not found", name)
	}

	// Get container logs
	logOptions := container.LogsOptions{
		ShowStdout: true,
		ShowStderr: true,
		Tail:       fmt.Sprintf("%d", lines),
	}

	logReader, err := cli.ContainerLogs(ctx, containerID, logOptions)
	if err != nil {
		return "", fmt.Errorf("failed to get container logs: %v", err)
	}
	defer logReader.Close()

	logs, err := io.ReadAll(logReader)
	if err != nil {
		return "", fmt.Errorf("failed to read container logs: %v", err)
	}

	return string(logs), nil
}

// StartContainerWithNetwork starts a container with network support using Docker SDK
func StartContainerWithNetwork(spec ContainerRunSpec, networkName string, policy *PortResolutionPolicy) (map[int]int, error) {
	if err := ensureDockerAvailable(); err != nil {
		return nil, err
	}
	if strings.TrimSpace(spec.Name) == "" || strings.TrimSpace(spec.Image) == "" {
		return nil, errors.New("container name and image are required")
	}

	// Remove stale container if exists and not running
	if containerExists(spec.Name) && !isContainerRunning(spec.Name) {
		fmt.Fprintln(os.Stderr, "Removing existing container to refresh image/args...")
		if err := removeContainer(spec.Name); err != nil {
			return nil, fmt.Errorf("failed to remove existing container %s: %w", spec.Name, err)
		}
	}

	// If already running, do nothing and return current published ports
	if isContainerRunning(spec.Name) {
		ports, _ := GetPublishedPorts(spec.Name)
		resolved := make(map[int]int)
		for key, val := range ports {
			// key like "80/tcp"; extract container port
			parts := strings.Split(key, "/")
			if len(parts) > 0 {
				if cp, err := strconv.Atoi(parts[0]); err == nil {
					if hp, err2 := strconv.Atoi(val); err2 == nil {
						resolved[cp] = hp
					}
				}
			}
		}
		return resolved, nil
	}

	// Pull image best-effort
	_ = pullImage(spec.Image)

	ctx := context.Background()
	cli, err := createDockerClient()
	if err != nil {
		return nil, fmt.Errorf("failed to create Docker client: %v", err)
	}
	defer cli.Close()

	// Prepare container configuration
	config := &container.Config{
		Image:      spec.Image,
		Env:        make([]string, 0, len(spec.Env)),
		Labels:     spec.Labels,
		WorkingDir: spec.Workdir,
		Cmd:        spec.Cmd,
		User:       spec.User,
	}

	// Add environment variables
	for k, v := range spec.Env {
		config.Env = append(config.Env, fmt.Sprintf("%s=%s", k, v))
	}

	// Set entrypoint if specified
	if len(spec.Entrypoint) > 0 {
		config.Entrypoint = spec.Entrypoint
	}

	// Prepare host configuration
	hostConfig := &container.HostConfig{
		Binds:      spec.Volumes,
		ExtraHosts: ensureHostDockerInternal(spec.AddHosts),
		AutoRemove: false,
	}

	// Handle port configuration
	exposedPorts := make(nat.PortSet)
	portBindings := make(nat.PortMap)

	useDynamic := false
	if policy != nil && policy.PreferredHostPort > 0 && len(spec.StaticPorts) > 0 {
		if isHostPortAvailable(policy.PreferredHostPort) {
			for _, pm := range spec.StaticPorts {
				hostPort := policy.PreferredHostPort
				if pm.Host > 0 {
					hostPort = pm.Host
				}
				protocol := pm.Protocol
				if protocol == "" {
					protocol = "tcp"
				}

				port, err := nat.NewPort(protocol, strconv.Itoa(pm.Container))
				if err != nil {
					return nil, fmt.Errorf("invalid port specification: %v", err)
				}
				exposedPorts[port] = struct{}{}
				portBindings[port] = []nat.PortBinding{
					{
						HostPort: strconv.Itoa(hostPort),
					},
				}
			}
		} else {
			if policy.Forced {
				return nil, fmt.Errorf("port %d is already in use", policy.PreferredHostPort)
			}
			useDynamic = true
		}
	} else {
		useDynamic = true
	}

	if useDynamic {
		hostConfig.PublishAllPorts = true
		// For dynamic ports, we need to expose the static ports if any
		for _, pm := range spec.StaticPorts {
			protocol := pm.Protocol
			if protocol == "" {
				protocol = "tcp"
			}
			port, err := nat.NewPort(protocol, strconv.Itoa(pm.Container))
			if err != nil {
				return nil, fmt.Errorf("invalid port specification: %v", err)
			}
			exposedPorts[port] = struct{}{}
		}
	}

	config.ExposedPorts = exposedPorts
	hostConfig.PortBindings = portBindings

	// Create the container
	resp, err := cli.ContainerCreate(ctx, config, hostConfig, nil, nil, spec.Name)
	if err != nil {
		return nil, fmt.Errorf("failed to create container: %v", err)
	}

	// Connect to network if specified
	if networkName != "" {
		if err := cli.NetworkConnect(ctx, networkName, resp.ID, nil); err != nil {
			return nil, fmt.Errorf("failed to connect container to network %s: %v", networkName, err)
		}
	}

	// Start the container
	if err := cli.ContainerStart(ctx, resp.ID, container.StartOptions{}); err != nil {
		return nil, fmt.Errorf("failed to start container: %v", err)
	}

	if debug {
		logDebug(fmt.Sprintf("Successfully started container: %s (ID: %s)", spec.Name, resp.ID))
	}

	// Resolve published ports
	published, err := GetPublishedPorts(spec.Name)
	if err != nil {
		return nil, err
	}
	resolved := make(map[int]int)
	for key, val := range published {
		parts := strings.Split(key, "/")
		if len(parts) > 0 {
			if cp, err := strconv.Atoi(parts[0]); err == nil {
				if hp, err2 := strconv.Atoi(val); err2 == nil {
					resolved[cp] = hp
				}
			}
		}
	}
	return resolved, nil
}
