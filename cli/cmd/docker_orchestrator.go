package cmd

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"
)

// ContainerOrchestrator manages the startup sequence and lifecycle of multiple containers
type ContainerOrchestrator struct {
	serverContainerName string
	ragContainerName    string
	networkManager      *NetworkManager
}

// NewContainerOrchestrator creates a new orchestrator for multi-container setup
func NewContainerOrchestrator() *ContainerOrchestrator {
	return &ContainerOrchestrator{
		serverContainerName: "llamafarm-server",
		ragContainerName:    "llamafarm-rag",
		networkManager:      NewNetworkManager(),
	}
}

// EnsureMultiContainerStack ensures both server and RAG containers are running
func (co *ContainerOrchestrator) EnsureMultiContainerStack(serverURL string, printStatus bool) *HealthPayload {
	// First, try to check if server is already available
	if hr, err := checkServerHealth(serverURL); err == nil {
		// Server is healthy, check if RAG is also healthy according to server
		if co.isRAGHealthyViaServer(hr) {
			return hr
		}
		// Server is healthy but RAG is not healthy - start RAG in background
		logDebug("Server is healthy but RAG is not healthy according to server health check, starting RAG...")
		go co.startRAGContainerAsync(serverURL)
		return hr
	} else if herr, ok := err.(*HealthError); ok {
		// Server is running but not healthy (degraded/unhealthy)
		// Use the existing server rather than starting a new container
		logDebug(fmt.Sprintf("Server is running but %s, using existing server", herr.Status))
		if herr.Status == "unhealthy" {
			// Check if unhealthy is only due to RAG being down
			if isUnhealthyOnlyDueToRAG(&herr.HealthResp) {
				OutputWarning("Server is unhealthy due to RAG component, starting RAG service...")
				go co.startRAGContainerAsync(serverURL)
				return &herr.HealthResp
			} else {
				// Server is unhealthy for other reasons
				OutputError("Server is unhealthy: %v\n", herr)
				os.Exit(1)
			}
		}
		// Server is degraded but running - start RAG if needed and use it
		if !co.isRAGHealthyViaServer(&herr.HealthResp) {
			logDebug("Server is degraded and RAG is not healthy, starting RAG...")
			go co.startRAGContainerAsync(serverURL)
		}
		return &herr.HealthResp
	}

	// Start server with network
	if err := co.startServerContainer(serverURL); err != nil {
		OutputError("Could not start server container: %v\n", err)
		os.Exit(1)
	}

	// Wait for server to be ready
	OutputProgress("Waiting for server to become ready...\n")
	timeout := serverStartTimeout
	if timeout <= 0 {
		timeout = 45 * time.Second
	}

	deadline := time.Now().Add(timeout)
	var lastError error

	for time.Now().Before(deadline) {
		if hr, err := checkServerHealth(serverURL); err == nil {
			// Server is healthy, now start RAG container in background
			go co.startRAGContainerAsync(serverURL)

			OutputSuccess("Server is ready\n")
			return hr
		} else if herr, ok := err.(*HealthError); ok {
			// Server is responding but not healthy (degraded)
			// If it's degraded (not unhealthy), consider it ready enough to use
			if herr.Status != "unhealthy" {
				logDebug(fmt.Sprintf("Server is %s but usable, starting RAG...", herr.Status))
				go co.startRAGContainerAsync(serverURL)
				OutputSuccess("Server is ready (degraded)\n")
				return &herr.HealthResp
			}
			// If unhealthy, check if it's only due to RAG
			if isUnhealthyOnlyDueToRAG(&herr.HealthResp) {
				logDebug("Server is unhealthy due to RAG, starting RAG and treating as ready")
				go co.startRAGContainerAsync(serverURL)
				OutputSuccess("Server is ready (unhealthy due to RAG, starting RAG service)\n")
				return &herr.HealthResp
			}
			// If unhealthy for other reasons, keep waiting
			lastError = err
			time.Sleep(2 * time.Second)
		} else {
			lastError = err
			time.Sleep(2 * time.Second)
		}
	}

	OutputError("Server did not become healthy within %v: %v\n", timeout, lastError)
	os.Exit(1)
	return nil
}

// StopMultiContainerStack stops both server and RAG containers
func (co *ContainerOrchestrator) StopMultiContainerStack() error {
	var errors []string

	// Stop RAG container first
	if err := StopAndRemoveContainer(co.ragContainerName); err != nil {
		errors = append(errors, fmt.Sprintf("RAG container: %v", err))
	}

	// Stop server container
	if err := StopAndRemoveContainer(co.serverContainerName); err != nil {
		errors = append(errors, fmt.Sprintf("server container: %v", err))
	}

	if len(errors) > 0 {
		return fmt.Errorf("errors stopping containers: %s", strings.Join(errors, "; "))
	}

	return nil
}

// startServerContainer starts the server container connected to the custom network
func (co *ContainerOrchestrator) startServerContainer(serverURL string) error {
	// Ensure network exists
	if err := co.networkManager.EnsureNetwork(); err != nil {
		return fmt.Errorf("failed to ensure network: %v", err)
	}

	networkName := co.networkManager.GetNetworkName()
	port := resolvePort(serverURL, 8000)

	// Get server image
	image, err := getImageURL("server")
	if err != nil {
		return fmt.Errorf("failed to get server image URL: %v", err)
	}

	// Get Ollama host for container configuration
	ollamaHost := os.Getenv("OLLAMA_HOST")
	if ollamaHost == "" {
		ollamaHost = "http://localhost:11434"
	}

	// Prepare container specification
	spec := ContainerRunSpec{
		Name:  co.serverContainerName,
		Image: image,
		StaticPorts: []PortMapping{
			{Host: port, Container: 8000, Protocol: "tcp"},
		},
		Env: make(map[string]string),
		Volumes: []string{
			fmt.Sprintf("%s:%s", os.ExpandEnv("$HOME/.llamafarm"), "/var/lib/llamafarm"),
		},
		Labels: map[string]string{
			"llamafarm.component": "server",
			"llamafarm.managed":   "true",
		},
	}

	// Mount effective working directory into the container at the same path
	if cwd := getEffectiveCWD(); strings.TrimSpace(cwd) != "" {
		spec.Volumes = append(spec.Volumes, fmt.Sprintf("%s:%s", cwd, cwd))
	} else {
		fmt.Fprintln(os.Stderr, "Warning: could not determine current directory; continuing without volume mount")
	}

	// Pass through or configure Ollama access inside the container
	if isLocalhost(ollamaHost) {
		ollamaPort := resolvePort(ollamaHost, 11434)
		spec.AddHosts = []string{"host.docker.internal:host-gateway"}
		spec.Env["OLLAMA_HOST"] = fmt.Sprintf("http://host.docker.internal:%d", ollamaPort)
	} else {
		spec.Env["OLLAMA_HOST"] = ollamaHost
	}

	if v, ok := os.LookupEnv("OLLAMA_PORT"); ok && strings.TrimSpace(v) != "" {
		spec.Env["OLLAMA_PORT"] = v
	}

	logDebug(fmt.Sprintf("Starting server container with network: %s", networkName))

	// Use the new Docker SDK-based container starter with network support
	_, err = StartContainerWithNetwork(spec, networkName, &PortResolutionPolicy{
		PreferredHostPort: port,
		Forced:            true,
	})

	return err
}

// startRAGContainer starts the RAG container and connects it to the network
func (co *ContainerOrchestrator) startRAGContainer() error {
	// Ensure network exists
	if err := co.networkManager.EnsureNetwork(); err != nil {
		return fmt.Errorf("failed to ensure network: %v", err)
	}

	networkName := co.networkManager.GetNetworkName()

	// Get RAG image URL
	imageURL, err := getImageURL("rag")
	if err != nil {
		return fmt.Errorf("failed to get RAG image URL: %v", err)
	}

	// Get data directory
	dataDir := os.Getenv("LF_DATA_DIR")
	if dataDir == "" {
		homeDir, err := os.UserHomeDir()
		if err != nil {
			return fmt.Errorf("could not determine home directory: %v", err)
		}
		dataDir = fmt.Sprintf("%s/.llamafarm/data", homeDir)
	}

	// Create data directory if it doesn't exist
	if err := os.MkdirAll(dataDir, 0755); err != nil {
		return fmt.Errorf("failed to create data directory: %v", err)
	}

	// Prepare container specification
	spec := ContainerRunSpec{
		Name:  co.ragContainerName,
		Image: imageURL,
		Env: map[string]string{
			"LF_DATA_DIR": "/var/lib/llamafarm", // Container path
		},
		Volumes: []string{
			fmt.Sprintf("%s:/var/lib/llamafarm", dataDir), // Volume mount
		},
		Labels: map[string]string{
			"llamafarm.component": "rag",
			"llamafarm.managed":   "true",
		},
	}

	logDebug(fmt.Sprintf("Starting RAG container with network: %s", networkName))

	// Use the new Docker SDK-based container starter with network support
	_, err = StartContainerWithNetwork(spec, networkName, nil)
	if err != nil {
		return fmt.Errorf("failed to start RAG container: %v", err)
	}

	logDebug("RAG container started successfully")
	return nil
}

// startRAGContainerAsync starts the RAG container in the background without blocking
func (co *ContainerOrchestrator) startRAGContainerAsync(serverURL string) {
	OutputProgress("Starting RAG service in background...\n")

	// Start the RAG container
	if err := co.startRAGContainer(); err != nil {
		OutputWarning("Could not start RAG container: %v\n", err)
		logDebug(fmt.Sprintf("RAG container startup failed: %v", err))
		return
	}

	// Wait for RAG to be ready in background - prioritize server health check
	go func() {
		if err := co.waitForRAGReadiness(10*time.Second, serverURL); err != nil {
			OutputWarning("RAG service may not be fully ready: %v\n", err)
			logDebug(fmt.Sprintf("RAG readiness check failed: %v", err))
		} else {
			OutputSuccess("RAG service is ready\n")
			logDebug("RAG service is fully ready")
		}
	}()
}

// GetRAGContainerLogs returns recent logs from the RAG container
func (co *ContainerOrchestrator) GetRAGContainerLogs(lines int) (string, error) {
	return GetContainerLogs(co.ragContainerName, lines)
}

// IsRAGContainerRunning checks if the RAG container is currently running
func (co *ContainerOrchestrator) IsRAGContainerRunning() bool {
	return IsContainerRunning(co.ragContainerName)
}

// isRAGHealthyViaServer checks if RAG is healthy according to the server's health check
func (co *ContainerOrchestrator) isRAGHealthyViaServer(hr *HealthPayload) bool {
	if hr == nil {
		return false
	}

	// Check components for RAG health
	for _, component := range hr.Components {
		if strings.Contains(strings.ToLower(component.Name), "rag") {
			return strings.EqualFold(component.Status, "healthy")
		}
	}

	// Check seeds for RAG health as fallback
	for _, seed := range hr.Seeds {
		if strings.Contains(strings.ToLower(seed.Name), "rag") {
			return strings.EqualFold(seed.Status, "healthy")
		}
	}

	return false
}

// waitForRAGReadiness waits for RAG to be ready by checking server health first, then container status
func (co *ContainerOrchestrator) waitForRAGReadiness(timeout time.Duration, serverURL string) error {
	logDebug("Waiting for RAG service to be ready...")

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	return WaitForReadiness(ctx, func() error {
		// First, try checking server health (preferred method)
		if hr, err := checkServerHealth(serverURL); err == nil {
			if co.isRAGHealthyViaServer(hr) {
				return nil
			}
			// Server is healthy but RAG is not ready according to health check
			return fmt.Errorf("RAG not ready according to server health check")
		}

		// If server health check fails, fall back to container status check
		if !IsContainerRunning(co.ragContainerName) {
			return fmt.Errorf("RAG container is not running")
		}

		// Check container logs as final fallback
		logs, err := GetContainerLogs(co.ragContainerName, 10)
		if err != nil {
			return fmt.Errorf("could not check RAG container logs: %v", err)
		}

		// Look for successful startup indicators
		if strings.Contains(logs, "Starting RAG Celery worker service") {
			return nil
		}

		// Check for error conditions
		if strings.Contains(logs, "Error") || strings.Contains(logs, "Failed") {
			return fmt.Errorf("RAG container failed to start properly. Check logs: docker logs %s", co.ragContainerName)
		}

		return fmt.Errorf("RAG service not ready yet")
	}, 3*time.Second)
}
