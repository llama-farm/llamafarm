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
	"path/filepath"
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
		// Server is healthy, check RAG in background and use it
		go handleRAGServiceInBackground(serverURL, hr)
		return hr
	} else if herr, ok := err.(*HealthError); ok {
		// Server responded but is not healthy (degraded, unhealthy, etc.)
		// This means the server is running, so we should use it rather than start a new one
		if printStatus || herr.Status == "unhealthy" {
			prettyPrintHealth(os.Stderr, herr.HealthResp)
		}
		if herr.Status == "unhealthy" {
			// Check if unhealthy is only due to RAG being down
			if isUnhealthyOnlyDueToRAG(&herr.HealthResp) {
				OutputWarning("Server is unhealthy due to RAG component, attempting to start RAG service...")
				// Try to start RAG service
				if isLocalhost(serverURL) {
					orchestrator := NewContainerOrchestrator()
					go orchestrator.startRAGContainerAsync(serverURL)
					// Return the current health status - RAG will be started in background
					return &herr.HealthResp
				} else {
					OutputError("Server is unhealthy due to RAG component, but cannot start RAG on remote server")
					os.Exit(1)
				}
			} else {
				// Server is unhealthy for other reasons
				os.Exit(1)
			}
		} else {
			// Server is degraded but running - use it
			return &herr.HealthResp
		}
	}

	// Only attempt auto-start when pointing to localhost
	if !isLocalhost(serverURL) {
		OutputError("Could not contact server %s", serverURL)
		os.Exit(1)
	}

	if err := startLocalServerViaDocker(serverURL); err != nil {
		OutputError("Could not start local server: %v", err)
		os.Exit(1)
	}

	// Poll for readiness
	timeout := serverStartTimeout
	if timeout <= 0 {
		timeout = 45 * time.Second
	}
	deadline := time.Now().Add(timeout)
	var lastError error = nil

	OutputProgress("Waiting for server to become ready...\n")
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
	OutputError("Server did not become ready at %s within timeout", serverURL)
	if herr, ok := lastError.(*HealthError); ok {
		// Render once on each failed poll tick to aid diagnosis
		if printStatus || herr.Status == "unhealthy" {
			prettyPrintHealth(os.Stderr, herr.HealthResp)
		}
		if herr.Status == "unhealthy" {
			os.Exit(1)
		}
	} else {
		OutputError("%v", lastError)
		os.Exit(1)
	}
	return nil
}

// checkServerHealth requires /health to be healthy.
func checkServerHealth(serverURL string) (*HealthPayload, error) {
	base := strings.TrimRight(serverURL, "/")
	healthURL := base + "/health"

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, healthURL, nil)
	if err != nil {
		return nil, err
	}
	resp, err := (&http.Client{Timeout: 5 * time.Second}).Do(req)
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
	if IsContainerRunning(containerName) {
		return nil
	}

	OutputProgress("Starting '%s' as '%s' via Docker...\n", image, containerName)

	// Get Ollama host for container configuration
	ollamaHostVar := os.Getenv("OLLAMA_HOST")
	if ollamaHostVar == "" {
		ollamaHostVar = ollamaHost
	}
	if ollamaHostVar == "" {
		ollamaHostVar = "http://localhost:11434"
	}

	// Prepare container specification
	homeDir, _ := os.UserHomeDir()
	spec := ContainerRunSpec{
		Name:  containerName,
		Image: image,
		StaticPorts: []PortMapping{
			{Host: port, Container: 8000, Protocol: "tcp"},
		},
		Env: make(map[string]string),
		Volumes: []string{
			fmt.Sprintf("%s:%s", convertToDockerPath(filepath.Join(homeDir, ".llamafarm")), "/var/lib/llamafarm"),
		},
		Labels: map[string]string{
			"llamafarm.component": "server",
			"llamafarm.managed":   "true",
		},
		User: getCurrentUserGroup(),
	}

	// Mount effective working directory into the container at the same path
	if err := setupWorkdirVolumeMount(&spec); err != nil {
		return fmt.Errorf("failed to configure working directory volume: %v", err)
	}

	// Pass through or configure Ollama access inside the container
	if isLocalhost(ollamaHostVar) {
		ollamaPort := resolvePort(ollamaHostVar, 11434)
		spec.AddHosts = []string{"host.docker.internal:host-gateway"}
		spec.Env["OLLAMA_HOST"] = fmt.Sprintf("http://host.docker.internal:%d", ollamaPort)
	} else {
		spec.Env["OLLAMA_HOST"] = ollamaHostVar
	}

	if v, ok := os.LookupEnv("OLLAMA_PORT"); ok && strings.TrimSpace(v) != "" {
		spec.Env["OLLAMA_PORT"] = v
	}

	// Use the Docker SDK-based container starter
	_, err = StartContainerDetachedWithPolicy(spec, &PortResolutionPolicy{
		PreferredHostPort: port,
		Forced:            true,
	})

	return err
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
		for _, c := range hr.Components {
			if c.Status == "healthy" {
				continue
			}

			icon := iconForStatus(c.Status)
			fmt.Fprintf(w, "  %s %-20s %-10s %s (latency: %dms)\n", icon, c.Name, c.Status, c.Message, c.LatencyMs)
			for k, v := range c.Details {
				fmt.Fprintf(w, "      %s: %v\n", k, v)
			}
		}
	}
	if len(hr.Seeds) > 0 {
		var builder strings.Builder
		for _, s := range hr.Seeds {
			if s.Status == "healthy" {
				continue
			}

			icon := iconForStatus(s.Status)
			builder.WriteString(fmt.Sprintf("  %s %-20s %-10s %s (latency: %dms)\n", icon, s.Name, s.Status, s.Message, s.LatencyMs))
			for k, v := range s.Runtime {
				builder.WriteString(fmt.Sprintf("      %s: %v\n", k, v))
			}
		}
		if builder.Len() > 0 {
			fmt.Fprintln(w, "Seeds:")
		}
		fmt.Fprintln(w, builder.String())
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

// handleRAGServiceInBackground checks if RAG service is unhealthy and starts it in background if needed
func handleRAGServiceInBackground(serverURL string, hr *HealthPayload) {
	if hr == nil {
		return
	}

	// Check if RAG service is unhealthy
	ragComponent := findRAGComponent(hr)
	if ragComponent == nil || strings.EqualFold(ragComponent.Status, "healthy") {
		return // RAG is fine, nothing to do
	}

	// RAG is unhealthy, start it in background if we're on localhost
	if !isLocalhost(serverURL) {
		OutputDebug("RAG service is unhealthy on remote server, cannot auto-start")
		return
	}

	OutputDebug("RAG service is unhealthy, starting in background...")
	orchestrator := NewContainerOrchestrator()
	orchestrator.startRAGContainerAsync(serverURL)
}

// findRAGComponent finds the RAG component in the health response
func findRAGComponent(hr *HealthPayload) *Component {
	if hr == nil {
		return nil
	}

	// Check components first
	for _, component := range hr.Components {
		name := strings.ToLower(component.Name)
		if name == "rag-service" || strings.HasPrefix(name, "rag-") || strings.HasSuffix(name, "-rag") {
			return &component
		}
	}

	// Check seeds as well
	for _, seed := range hr.Seeds {
		name := strings.ToLower(seed.Name)
		if name == "rag-service" || strings.HasPrefix(name, "rag-") || strings.HasSuffix(name, "-rag") {
			return &seed
		}
	}

	return nil
}

// isUnhealthyOnlyDueToRAG checks if the server is unhealthy solely because of RAG components
// Returns true if all non-RAG components are healthy and only RAG components are unhealthy
func isUnhealthyOnlyDueToRAG(hr *HealthPayload) bool {
	if hr == nil {
		return false
	}

	hasUnhealthyRAG := false
	hasUnhealthyNonRAG := false

	// Check components
	for _, component := range hr.Components {
		name := strings.ToLower(component.Name)
		isRAGComponent := name == "rag-service" || strings.HasPrefix(name, "rag-") || strings.HasSuffix(name, "-rag")
		isHealthy := strings.EqualFold(component.Status, "healthy")

		if isRAGComponent && !isHealthy {
			hasUnhealthyRAG = true
		} else if !isRAGComponent && !isHealthy {
			hasUnhealthyNonRAG = true
		}
	}

	// Check seeds as well
	for _, seed := range hr.Seeds {
		name := strings.ToLower(seed.Name)
		isRAGComponent := name == "rag-service" || strings.HasPrefix(name, "rag-") || strings.HasSuffix(name, "-rag")
		isHealthy := strings.EqualFold(seed.Status, "healthy")

		if isRAGComponent && !isHealthy {
			hasUnhealthyRAG = true
		} else if !isRAGComponent && !isHealthy {
			hasUnhealthyNonRAG = true
		}
	}

	// Server is unhealthy only due to RAG if:
	// 1. There are unhealthy RAG components, AND
	// 2. There are NO unhealthy non-RAG components
	return hasUnhealthyRAG && !hasUnhealthyNonRAG
}

// convertToDockerPath normalizes a host path to use forward slashes which Docker accepts across platforms.
func convertToDockerPath(hostPath string) string {
	return filepath.ToSlash(hostPath)
}

// validateDockerVolumePath checks if a path can be safely mounted as a Docker volume
func validateDockerVolumePath(hostPath string) error {
	if strings.TrimSpace(hostPath) == "" {
		return fmt.Errorf("empty path")
	}

	// Check if the path is accessible
	if _, err := os.Stat(hostPath); err != nil {
		return fmt.Errorf("path not accessible: %v", err)
	}

	return nil
}

// setupWorkdirVolumeMount safely sets up the working directory volume mount
// Returns an error if the working directory cannot be determined or validated
func setupWorkdirVolumeMount(spec *ContainerRunSpec) error {
	cwd := getEffectiveCWD()
	if strings.TrimSpace(cwd) == "" {
		return fmt.Errorf("could not determine current directory")
	}

	// Simple validation - check if path is accessible
	if err := validateDockerVolumePath(cwd); err != nil {
		return fmt.Errorf("working directory not accessible (%s): %v", cwd, err)
	}

	// Convert to Docker-compatible path
	dockerPath := convertToDockerPath(cwd)
	volumeMount := fmt.Sprintf("%s:%s", dockerPath, dockerPath)
	spec.Volumes = append(spec.Volumes, volumeMount)

	OutputDebug("Mounting volume: %s\n", volumeMount)

	return nil
}
