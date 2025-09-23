package cmd

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// versionPattern matches semantic versions with or without leading "v"
// Examples: v1.0.0, v1.0.0-rc1, v2.0.0-beta.1+build.123, 1.0.0, 1.0.0-alpha
var versionPattern = regexp.MustCompile(`^v?\d+\.\d+\.\d+.*`)

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

// ProgressTracker tracks overall pull progress across all layers
type ProgressTracker struct {
	layers     map[string]*DockerPullProgress
	totalBytes int64
	doneBytes  int64
	lastUpdate time.Time
}

// NewProgressTracker creates a new progress tracker
func NewProgressTracker() *ProgressTracker {
	return &ProgressTracker{
		layers:     make(map[string]*DockerPullProgress),
		lastUpdate: time.Now(),
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

// recalculate recalculates total and done bytes across all layers
func (pt *ProgressTracker) recalculate() {
	pt.totalBytes = 0
	pt.doneBytes = 0

	for _, layer := range pt.layers {
		if layer.Total > 0 {
			pt.totalBytes += layer.Total
			pt.doneBytes += layer.Current
		}
	}
}

// GetProgress returns the overall progress percentage (0-100)
func (pt *ProgressTracker) GetProgress() float64 {
	if pt.totalBytes == 0 {
		return 0
	}
	return float64(pt.doneBytes) / float64(pt.totalBytes) * 100
}

// GetTransferRate returns the transfer rate in bytes per second
func (pt *ProgressTracker) GetTransferRate() float64 {
	elapsed := time.Since(pt.lastUpdate).Seconds()
	if elapsed == 0 {
		return 0
	}
	return float64(pt.doneBytes) / elapsed
}

// FormatTransferRate formats transfer rate in human-readable format
func (pt *ProgressTracker) FormatTransferRate() string {
	rate := pt.GetTransferRate()
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

	// Use \r to overwrite the current line
	fmt.Fprintf(os.Stderr, "\rPulling %s: %.1f%% (%s)    ", imageName, progress, rate)
}

// ensureDockerAvailable checks whether docker is available on PATH
func ensureDockerAvailable() error {
	if err := exec.Command("docker", "--version").Run(); err != nil {
		return errors.New("docker is not available. Please install Docker and try again")
	}
	return nil
}

// parseDockerProgress parses a Docker progress line and extracts progress information
// Examples of Docker progress lines (default format):
// "a1b2c3d4e5f6: Downloading [==============>                                    ]  123.4MB/456.7MB"
// "a1b2c3d4e5f6: Extracting  [============================>                      ]  234.5MB/456.7MB"
// "Downloading from ghcr.io/llama-farm/llamafarm/server"
func parseDockerProgress(line string) *DockerPullProgress {
	// First try the progress bar format (with layer ID)
	progressRegex := regexp.MustCompile(`^([a-f0-9]{12}):\s+(\w+)\s+\[.*?\]\s+([0-9.]+)([KMGT]?B)/([0-9.]+)([KMGT]?B)`)
	matches := progressRegex.FindStringSubmatch(line)

	if len(matches) == 7 {
		layerID := matches[1]
		status := matches[2]
		currentStr := matches[3]
		currentUnit := matches[4]
		totalStr := matches[5]
		totalUnit := matches[6]

		// Convert sizes to bytes
		current := parseSize(currentStr, currentUnit)
		total := parseSize(totalStr, totalUnit)

		if current >= 0 && total >= 0 {
			return &DockerPullProgress{
				ID:      layerID,
				Status:  status,
				Current: current,
				Total:   total,
			}
		}
	}

	// Try alternative format without progress bar but with size info
	// "a1b2c3d4e5f6: Downloading 123.4MB/456.7MB"
	simpleRegex := regexp.MustCompile(`^([a-f0-9]{12}):\s+(\w+)\s+([0-9.]+)([KMGT]?B)/([0-9.]+)([KMGT]?B)`)
	matches = simpleRegex.FindStringSubmatch(line)

	if len(matches) == 7 {
		layerID := matches[1]
		status := matches[2]
		currentStr := matches[3]
		currentUnit := matches[4]
		totalStr := matches[5]
		totalUnit := matches[6]

		// Convert sizes to bytes
		current := parseSize(currentStr, currentUnit)
		total := parseSize(totalStr, totalUnit)

		if current >= 0 && total >= 0 {
			return &DockerPullProgress{
				ID:      layerID,
				Status:  status,
				Current: current,
				Total:   total,
			}
		}
	}

	return nil
}

// parseSize converts a size string with unit to bytes
func parseSize(sizeStr, unit string) int64 {
	size, err := strconv.ParseFloat(sizeStr, 64)
	if err != nil {
		return -1
	}

	switch unit {
	case "B":
		return int64(size)
	case "KB":
		return int64(size * 1024)
	case "MB":
		return int64(size * 1024 * 1024)
	case "GB":
		return int64(size * 1024 * 1024 * 1024)
	case "TB":
		return int64(size * 1024 * 1024 * 1024 * 1024)
	default:
		return int64(size) // Assume bytes if no unit
	}
}

// pullImage pulls a docker image with progress tracking, capturing output to avoid breaking TUIs
func pullImage(image string) error {
	// Extract image name for display (remove registry/tag parts for brevity)
	imageParts := strings.Split(image, "/")
	displayName := imageParts[len(imageParts)-1]
	if tagIdx := strings.Index(displayName, ":"); tagIdx > 0 {
		displayName = displayName[:tagIdx]
	}

	fmt.Fprintf(os.Stderr, "Pulling image: %s\n", image)

	// Use standard docker pull command (no --progress flag for compatibility)
	pullCmd := exec.Command("docker", "pull", image)

	// Create pipes to capture stdout and stderr separately
	stdout, err := pullCmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("failed to create stdout pipe: %v", err)
	}
	stderr, err := pullCmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("failed to create stderr pipe: %v", err)
	}

	// Start the command
	if err := pullCmd.Start(); err != nil {
		return fmt.Errorf("failed to start docker pull: %v", err)
	}

	// Create progress tracker
	tracker := NewProgressTracker()
	lastProgressTime := time.Now()

	// Channel to collect all output for debug logging
	var allOutput strings.Builder

	// Track whether we've seen any progress information
	hasProgress := false

	// Read and process stdout (progress output)
	go func() {
		scanner := bufio.NewScanner(stdout)
		for scanner.Scan() {
			line := scanner.Text()
			allOutput.WriteString(line + "\n")

			// Try to parse progress information
			if progress := parseDockerProgress(line); progress != nil {
				hasProgress = true
				tracker.Update(progress)

				// Throttle display updates to avoid overwhelming the terminal
				if time.Since(lastProgressTime) > 100*time.Millisecond {
					tracker.DisplayProgress(displayName)
					lastProgressTime = time.Now()
				}
			} else if !hasProgress {
				// If no progress info yet, show basic status for certain lines
				if strings.Contains(line, "Downloading") || strings.Contains(line, "Extracting") || strings.Contains(line, "Pull complete") {
					fmt.Fprintf(os.Stderr, "\rPulling %s...    ", displayName)
				}
			}
		}
	}()

	// Read and process stderr (also capture for debug)
	go func() {
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			line := scanner.Text()
			allOutput.WriteString(line + "\n")
		}
	}()

	// Wait for command to complete
	if err := pullCmd.Wait(); err != nil {
		// Clear the progress line before showing error
		fmt.Fprintf(os.Stderr, "\r%s\r", strings.Repeat(" ", 80))
		return fmt.Errorf("docker pull failed: %v", err)
	}

	// Clear the progress line and show completion
	fmt.Fprintf(os.Stderr, "\r%s\r", strings.Repeat(" ", 80))
	fmt.Fprintf(os.Stderr, "✓ Pulled %s successfully\n", displayName)

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
	cmd := exec.Command("docker", "ps", "-a", "--format", "{{.Names}}")
	out, err := cmd.Output()
	if err != nil {
		return false
	}
	for _, line := range strings.Split(string(out), "\n") {
		if strings.TrimSpace(line) == name {
			return true
		}
	}
	return false
}

func isContainerRunning(name string) bool {
	cmd := exec.Command("docker", "ps", "--format", "{{.Names}}")
	out, err := cmd.Output()
	if err != nil {
		return false
	}
	for _, line := range strings.Split(string(out), "\n") {
		if strings.TrimSpace(line) == name {
			return true
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
}

type PortResolutionPolicy struct {
	PreferredHostPort int
	Forced            bool
}

func removeContainer(name string) error {
	if !containerExists(name) {
		return nil
	}
	rmCmd := exec.Command("docker", "rm", "-f", name)
	out, err := rmCmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("docker rm failed: %v\n%s", err, string(out))
	}
	if debug && len(out) > 0 {
		logDebug(fmt.Sprintf("docker rm output: %s", string(out)))
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

	runArgs := []string{"run", "-d", "--name", spec.Name}

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
				runArgs = append(runArgs, "-p", fmt.Sprintf("%d:%d/%s", hostPort, pm.Container, protocol))
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
		runArgs = append(runArgs, "-P")
	}

	for k, v := range spec.Env {
		runArgs = append(runArgs, "-e", fmt.Sprintf("%s=%s", k, v))
	}
	for _, v := range spec.Volumes {
		runArgs = append(runArgs, "-v", v)
	}
	for _, h := range spec.AddHosts {
		runArgs = append(runArgs, "--add-host", h)
	}
	for k, v := range spec.Labels {
		runArgs = append(runArgs, "--label", fmt.Sprintf("%s=%s", k, v))
	}
	if strings.TrimSpace(spec.Workdir) != "" {
		runArgs = append(runArgs, "-w", spec.Workdir)
	}
	if len(spec.Entrypoint) > 0 {
		runArgs = append(runArgs, "--entrypoint", strings.Join(spec.Entrypoint, " "))
	}

	runArgs = append(runArgs, spec.Image)
	runArgs = append(runArgs, spec.Cmd...)

	runCmd := exec.Command("docker", runArgs...)
	runOut, err := runCmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("failed to start docker container: %v\n%s", err, string(runOut))
	}
	if debug && len(runOut) > 0 {
		logDebug(fmt.Sprintf("docker run output: %s", string(runOut)))
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
	cmd := exec.Command("docker", "port", name)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("docker port failed: %v\n%s", err, string(out))
	}
	res := make(map[string]string)
	s := bufio.NewScanner(strings.NewReader(string(out)))
	for s.Scan() {
		line := strings.TrimSpace(s.Text())
		// Example: "80/tcp -> 0.0.0.0:49154" or "80/tcp -> :::49154"
		parts := strings.Split(line, " -> ")
		if len(parts) != 2 {
			continue
		}
		key := strings.TrimSpace(parts[0])
		host := strings.TrimSpace(parts[1])
		idx := strings.LastIndex(host, ":")
		if idx > -1 && idx+1 < len(host) {
			res[key] = host[idx+1:]
		}
	}
	if debug && len(out) > 0 {
		logDebug(fmt.Sprintf("docker port output: %s", string(out)))
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
