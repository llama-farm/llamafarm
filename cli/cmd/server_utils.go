package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"strings"
	"time"
)

var containerName = "llamafarm-server"

type Component struct {
	Name      string                 `json:"name"`
	Status    string                 `json:"status"`
	Message   string                 `json:"message"`
	LatencyMs int                    `json:"latency_ms"`
	Details   map[string]interface{} `json:"details,omitempty"`
	Runtime   map[string]interface{} `json:"runtime,omitempty"`
}
type HealthPayload struct {
	Status     string      `json:"status"`
	Summary    string      `json:"summary"`
	Components []Component `json:"components"`
	Seeds      []Component `json:"seeds"`
	Timestamp  int64       `json:"timestamp"`
}

// HealthError wraps a non-healthy /health response.
type HealthError struct {
	Status     string
	HealthResp HealthPayload
}

func (e *HealthError) Error() string {
	return fmt.Sprintf("server unhealthy: %s", e.Status)
}

// ensureServerAvailable verifies the server at serverURL is reachable.
// If not reachable and the host is localhost, it attempts to start the
// server via Docker, then waits for readiness. Returns an error if it
// ultimately cannot ensure availability.
func ensureServerAvailable(serverURL string, printStatus bool) *HealthPayload {
	if serverURL == "" {
		serverURL = "http://localhost:8000"
	}

	if hr, err := checkServerHealth(serverURL); err == nil {
		return hr
	} else {
		// If we already got a health payload, render a clean, readable error
		url := strings.TrimRight(serverURL, "/") + "/health/liveness"
		if perr := pingURL(url); perr == nil {
			// The server is reachable, but not healthy
			if herr, ok := err.(*HealthError); ok {
				if printStatus || herr.Status == "unhealthy" {
					prettyPrintHealth(os.Stderr, herr.HealthResp)
				}
				if herr.Status == "unhealthy" {
					os.Exit(1)
				} else {
					return &herr.HealthResp
				}
			}
		}
	}

	// Only attempt auto-start when pointing to localhost
	if !isLocalhost(serverURL) {
		fmt.Fprintf(os.Stderr, "❌ Could not contact server %s\n", serverURL)
		os.Exit(1)
	}

	if err := startLocalServerViaDocker(serverURL); err != nil {
		fmt.Fprintf(os.Stderr, "❌ Could not start local server: %v\n", err)
		os.Exit(1)
	}

	// Poll for readiness
	timeout := serverStartTimeout
	if timeout <= 0 {
		timeout = 45 * time.Second
	}
	deadline := time.Now().Add(timeout)
	var lastError error = nil

	fmt.Fprintf(os.Stderr, "Waiting for server to become ready...\n")
	for {
		if hr, err := checkServerHealth(serverURL); err == nil {
			return hr
		} else {
			lastError = err
			if time.Now().After(deadline) {
				break
			}
			duration := 1 * time.Second
			time.Sleep(duration)
		}
	}
	fmt.Fprintf(os.Stderr, "Server did not become ready at %s within timeout\n", serverURL)
	if herr, ok := lastError.(*HealthError); ok {
		// Render once on each failed poll tick to aid diagnosis
		if printStatus || herr.Status == "unhealthy" {
			prettyPrintHealth(os.Stderr, herr.HealthResp)
		}
		if herr.Status == "unhealthy" {
			os.Exit(1)
		}
	} else {
		fmt.Fprintf(os.Stderr, "%v\n", lastError)
		os.Exit(1)
	}
	return nil
}

// checkServerHealth requires /health to be healthy.
func checkServerHealth(serverURL string) (*HealthPayload, error) {
	base := strings.TrimRight(serverURL, "/")
	healthURL := base + "/health"

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, healthURL, nil)
	if err != nil {
		return nil, err
	}
	resp, err := (&http.Client{Timeout: 2 * time.Second}).Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		var payload HealthPayload
		if err := json.Unmarshal(body, &payload); err != nil {
			return nil, fmt.Errorf("invalid health payload: %v", err)
		}
		if strings.EqualFold(payload.Status, "healthy") {
			return &payload, nil
		}
		return nil, &HealthError{Status: payload.Status, HealthResp: payload}
	}
	return nil, fmt.Errorf("unexpected health status %d", resp.StatusCode)
}

func isLocalhost(serverURL string) bool {
	u, err := url.Parse(serverURL)
	if err != nil {
		return false
	}
	host := strings.ToLower(u.Hostname())
	return host == "localhost" || host == "127.0.0.1" || host == "::1"
}

// startLocalServerViaDocker pulls and runs the LlamaFarm server container if needed.
// It uses a fixed container name and maps the serverURL port to container port 8000.
func startLocalServerViaDocker(serverURL string) error {
	// Ensure Docker is available
	if err := ensureDockerAvailable(); err != nil {
		return err
	}

	port := resolvePort(serverURL, 8000)

	// Get the dynamic image URL using our version-aware resolution
	image, err := getImageURL("server")
	if err != nil {
		return fmt.Errorf("failed to resolve server image URL: %v", err)
	}

	// If a container with this name exists and is running, nothing to do
	if isContainerRunning(containerName) {
		return nil
	}

	fmt.Fprintln(os.Stderr, "Starting local LlamaFarm server via Docker...")

	// If a container with this name exists, remove it to ensure we always use the latest image
	if containerExists(containerName) {
		fmt.Fprintln(os.Stderr, "Removing existing LlamaFarm server container to ensure latest image and arguments...")
		rmCmd := exec.Command("docker", "rm", "-f", containerName)
		rmCmd.Stdout = os.Stdout
		rmCmd.Stderr = os.Stderr
		if err := rmCmd.Run(); err != nil {
			return fmt.Errorf("failed to remove existing container %s: %v", containerName, err)
		}
	}

	// Pull latest image (best effort)
	_ = pullImage(image)

	// Run new container
	runArgs := []string{
		"run",
		"-d",
		"--name", containerName,
		"-p", fmt.Sprintf("%d:8000", port),
		"-v", fmt.Sprintf("%s:%s", os.ExpandEnv("$HOME/.llamafarm"), "/var/lib/llamafarm"),
	}

	// Mount effective working directory into the container at the same path
	if cwd := getEffectiveCWD(); strings.TrimSpace(cwd) != "" {
		runArgs = append(runArgs, "-v", fmt.Sprintf("%s:%s", cwd, cwd))
	} else {
		fmt.Fprintln(os.Stderr, "Warning: could not determine current directory; continuing without volume mount")
	}

	// Pass through or configure Ollama access inside the container
	if isLocalhost(ollamaHost) {
		port := resolvePort(ollamaHost, 11434)
		runArgs = append(runArgs, "--add-host", "host.docker.internal:host-gateway")
		runArgs = append(runArgs, "-e", fmt.Sprintf("OLLAMA_HOST=http://host.docker.internal:%d", port))
	} else {
		runArgs = append(runArgs, "-e", fmt.Sprintf("OLLAMA_HOST=%s", ollamaHost))
	}

	if v, ok := os.LookupEnv("OLLAMA_PORT"); ok && strings.TrimSpace(v) != "" {
		runArgs = append(runArgs, "-e", fmt.Sprintf("OLLAMA_PORT=%s", v))
	}

	// Image last
	runArgs = append(runArgs, image)
	runCmd := exec.Command("docker", runArgs...)
	runOut, err := runCmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to start docker container: %v\n%s", err, string(runOut))
	}
	return nil
}

func resolvePort(serverURL string, defaultPort int) int {
	u, err := url.Parse(serverURL)
	if err != nil {
		return defaultPort
	}
	if p := u.Port(); p != "" {
		if portNum, err := net.LookupPort("tcp", p); err == nil {
			return portNum
		}
	}
	// If URL scheme implies a default port, prefer it
	if u.Scheme == "https" {
		return 443
	}
	if u.Scheme == "http" {
		return 80
	}
	return defaultPort
}

// prettyPrintHealth decodes a /health payload and renders a concise, readable summary
func prettyPrintHealth(w io.Writer, hr HealthPayload) {
	prefix := "❌"
	switch hr.Status {
	case "degraded":
		prefix = "⚠️"
	case "healthy":
		prefix = "✅"
	}

	fmt.Fprintf(w, "%s Server is %s\n", prefix, hr.Status)
	if strings.TrimSpace(hr.Summary) != "" {
		fmt.Fprintf(w, "Summary: %s\n", hr.Summary)
	}
	if len(hr.Components) > 0 {
		fmt.Fprintln(w, "Components:")
		for _, c := range hr.Components {
			icon := iconForStatus(c.Status)
			fmt.Fprintf(w, "  %s %-20s %-10s %s (latency: %dms)\n", icon, c.Name, c.Status, c.Message, c.LatencyMs)
			for k, v := range c.Details {
				fmt.Fprintf(w, "      %s: %v\n", k, v)
			}
		}
	}
	if len(hr.Seeds) > 0 {
		fmt.Fprintln(w, "Seeds:")
		for _, s := range hr.Seeds {
			icon := iconForStatus(s.Status)
			fmt.Fprintf(w, "  %s %-20s %-10s %s (latency: %dms)\n", icon, s.Name, s.Status, s.Message, s.LatencyMs)
			for k, v := range s.Runtime {
				fmt.Fprintf(w, "      %s: %v\n", k, v)
			}
		}
	}
}

// prettyPrintHealthProblems prints only the non-healthy components and seeds from a HealthPayload.
// It is intended for concise error reporting.
func prettyPrintHealthProblems(w io.Writer, hr HealthPayload) {
	// Check components
	for _, c := range hr.Components {
		if c.Status != "healthy" {
			icon := iconForStatus(c.Status)
			fmt.Fprintf(w, "  %s %-20s %-10s %s (latency: %dms)\n", icon, c.Name, c.Status, c.Message, c.LatencyMs)
			for k, v := range c.Details {
				fmt.Fprintf(w, "      %s: %v\n", k, v)
			}
		}
	}

	// Check seeds
	for _, s := range hr.Seeds {
		if s.Status != "healthy" {
			icon := iconForStatus(s.Status)
			fmt.Fprintf(w, "  %s %-20s %-10s %s (latency: %dms)\n", icon, s.Name, s.Status, s.Message, s.LatencyMs)
			for k, v := range s.Runtime {
				fmt.Fprintf(w, "      %s: %v\n", k, v)
			}
		}
	}
}

func iconForStatus(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	switch s {
	case "healthy":
		return "✅"
	case "degraded":
		return "⚠️"
	case "unhealthy":
		return "❌"
	default:
		return "❓"
	}
}

func pingURL(base string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, base, nil)
	if err != nil {
		return err
	}
	resp, err := (&http.Client{Timeout: 2 * time.Second}).Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, resp.Body)
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return nil
	}
	return fmt.Errorf("status %d", resp.StatusCode)
}

// MultiContainerOrchestrator manages the startup sequence of server and RAG containers
type MultiContainerOrchestrator struct {
	networkManager *NetworkManager
	ragManager     *RAGContainerManager
}

// NewMultiContainerOrchestrator creates a new orchestrator for multi-container setup
func NewMultiContainerOrchestrator() *MultiContainerOrchestrator {
	networkManager := NewNetworkManager()
	ragManager := NewRAGContainerManager(networkManager)

	return &MultiContainerOrchestrator{
		networkManager: networkManager,
		ragManager:     ragManager,
	}
}

// startLocalServerWithNetwork starts the server container connected to the custom network
func (mco *MultiContainerOrchestrator) startLocalServerWithNetwork(serverURL string) error {
	// Ensure network exists
	if err := mco.networkManager.EnsureNetwork(); err != nil {
		return fmt.Errorf("failed to ensure network: %v", err)
	}

	networkName := mco.networkManager.GetNetworkName()

	// Parse server URL to get port
	port := resolvePort(serverURL, 8000)

	// Remove existing container
	if err := removeContainer(containerName); err != nil {
		return fmt.Errorf("failed to remove existing server container: %v", err)
	}

	// Get server image
	image, err := getImageURL("server")
	if err != nil {
		return fmt.Errorf("failed to get server image URL: %v", err)
	}

	// Pull the image
	if err := pullImage(image); err != nil {
		return fmt.Errorf("failed to pull server image: %v", err)
	}

	// Get Ollama host for container configuration
	ollamaHost := os.Getenv("OLLAMA_HOST")
	if ollamaHost == "" {
		ollamaHost = "http://localhost:11434"
	}

	// Build docker run command with network
	runArgs := []string{
		"run",
		"-d",
		"--name", containerName,
		"--network", networkName,
		"-p", fmt.Sprintf("%d:8000", port),
		"-v", fmt.Sprintf("%s:%s", os.ExpandEnv("$HOME/.llamafarm"), "/var/lib/llamafarm"),
		"--label", "llamafarm.component=server",
		"--label", "llamafarm.managed=true",
	}

	// Mount effective working directory into the container at the same path
	if cwd := getEffectiveCWD(); strings.TrimSpace(cwd) != "" {
		runArgs = append(runArgs, "-v", fmt.Sprintf("%s:%s", cwd, cwd))
	} else {
		fmt.Fprintln(os.Stderr, "Warning: could not determine current directory; continuing without volume mount")
	}

	// Pass through or configure Ollama access inside the container
	if isLocalhost(ollamaHost) {
		ollamaPort := resolvePort(ollamaHost, 11434)
		runArgs = append(runArgs, "--add-host", "host.docker.internal:host-gateway")
		runArgs = append(runArgs, "-e", fmt.Sprintf("OLLAMA_HOST=http://host.docker.internal:%d", ollamaPort))
	} else {
		runArgs = append(runArgs, "-e", fmt.Sprintf("OLLAMA_HOST=%s", ollamaHost))
	}

	if v, ok := os.LookupEnv("OLLAMA_PORT"); ok && strings.TrimSpace(v) != "" {
		runArgs = append(runArgs, "-e", fmt.Sprintf("OLLAMA_PORT=%s", v))
	}

	// Image last
	runArgs = append(runArgs, image)

	logDebug(fmt.Sprintf("Starting server container with network: %s", networkName))
	runCmd := exec.Command("docker", runArgs...)
	runOut, err := runCmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to start server container: %v\n%s", err, string(runOut))
	}

	return nil
}

// EnsureMultiContainerStack ensures both server and RAG containers are running
func (mco *MultiContainerOrchestrator) EnsureMultiContainerStack(serverURL string, printStatus bool) *HealthPayload {
	// First, try to check if server is already available
	if hr, err := checkServerHealth(serverURL); err == nil {
		// Server is healthy, check if RAG is also running
		if mco.ragManager.IsRAGContainerRunning() {
			return hr
		}
		// Server is healthy but RAG is not running - start RAG
		logDebug("Server is healthy but RAG container is not running, starting RAG...")
	}

	// Start server with network
	if err := mco.startLocalServerWithNetwork(serverURL); err != nil {
		fmt.Fprintf(os.Stderr, "❌ Could not start server container: %v\n", err)
		os.Exit(1)
	}

	// Wait for server to be ready
	fmt.Fprintf(os.Stderr, "Waiting for server to become ready...\n")
	timeout := serverStartTimeout
	if timeout <= 0 {
		timeout = 45 * time.Second
	}

	deadline := time.Now().Add(timeout)
	var lastError error

	for time.Now().Before(deadline) {
		if hr, err := checkServerHealth(serverURL); err == nil {
			// Server is ready, now start RAG container
			fmt.Fprintf(os.Stderr, "Server ready, starting RAG service...\n")

			if err := mco.ragManager.StartRAGContainer(); err != nil {
				fmt.Fprintf(os.Stderr, "❌ Could not start RAG container: %v\n", err)
				os.Exit(1)
			}

			// Wait for RAG to be ready
			if err := mco.ragManager.WaitForRAGContainer(30 * time.Second); err != nil {
				fmt.Fprintf(os.Stderr, "⚠️  RAG container may not be fully ready: %v\n", err)
				// Don't exit - server can still function without RAG initially
			} else {
				fmt.Fprintf(os.Stderr, "✅ Multi-container stack is ready\n")
			}

			return hr
		} else {
			lastError = err
			time.Sleep(2 * time.Second)
		}
	}

	fmt.Fprintf(os.Stderr, "❌ Server did not become healthy within %v: %v\n", timeout, lastError)
	os.Exit(1)
	return nil
}

// StopMultiContainerStack stops both server and RAG containers
func (mco *MultiContainerOrchestrator) StopMultiContainerStack() error {
	var errors []string

	// Stop RAG container first
	if err := mco.ragManager.StopRAGContainer(); err != nil {
		errors = append(errors, fmt.Sprintf("RAG container: %v", err))
	}

	// Stop server container
	if err := removeContainer(containerName); err != nil {
		errors = append(errors, fmt.Sprintf("server container: %v", err))
	}

	if len(errors) > 0 {
		return fmt.Errorf("errors stopping containers: %s", strings.Join(errors, "; "))
	}

	return nil
}
