package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/docker/docker/api/types/container"
	"github.com/spf13/cobra"
)

// ServiceInfo represents the status of a single service
type ServiceInfo struct {
	Name          string            `json:"name"`
	ContainerName string            `json:"container_name,omitempty"`
	State         string            `json:"state"` // "running", "stopped", "not_found"
	ContainerID   string            `json:"container_id,omitempty"`
	PID           int               `json:"pid,omitempty"`
	Image         string            `json:"image,omitempty"`
	Ports         map[string]string `json:"ports,omitempty"`
	Health        *Component        `json:"health,omitempty"`
	Uptime        string            `json:"uptime,omitempty"`
	LogFile       string            `json:"log_file,omitempty"`
	Orchestration string            `json:"orchestration"` // "docker" or "native"
}

// ServicesStatusOutput represents the complete status output
type ServicesStatusOutput struct {
	Services      []ServiceInfo `json:"services"`
	DockerRunning bool          `json:"docker_running,omitempty"`
	Orchestration string        `json:"orchestration"` // "docker" or "native"
	Timestamp     int64         `json:"timestamp"`
}

// servicesCmd is the parent command for service management
var servicesCmd = &cobra.Command{
	Use:   "services",
	Short: "Manage LlamaFarm services",
	Long:  "Commands for managing and inspecting LlamaFarm services (server, rag, etc.)",
}

// servicesStatusCmd displays the status of all services
var servicesStatusCmd = &cobra.Command{
	Use:   "status",
	Short: "Check status of all LlamaFarm services",
	Long: `Display the current status of all LlamaFarm services without starting them.

This command automatically detects the orchestration mode (native or Docker) and shows:
  - Process/container running state
  - PID (native) or container ID (Docker)
  - Port mappings (Docker)
  - Health status (if service is running)
  - Log file location (native) or image information (Docker)
  - Uptime

The orchestration mode is determined by the LF_ORCHESTRATION_MODE environment variable:
  - "native" (default): Check native processes
  - "docker": Check Docker containers
  - "auto": Auto-detect (prefers native)

This is a read-only operation that never auto-starts services.`,
	Run: runServicesStatus,
}

// servicesStartCmd starts LlamaFarm services
var servicesStartCmd = &cobra.Command{
	Use:   "start [service-name]",
	Short: "Start LlamaFarm services",
	Long: `Start LlamaFarm services using the configured orchestration mode.

Without arguments, starts all services (server, rag, universal-runtime).
With a service name, starts only that specific service.

Available services:
  - server: The main FastAPI server
  - rag: The RAG/Celery worker
  - universal-runtime: The universal runtime server

The orchestration mode is determined by the LF_ORCHESTRATION_MODE environment variable:
  - "native" (default): Start services as native processes
  - "docker": Start services as Docker containers
  - "auto": Auto-detect (prefers native)

Examples:
  lf services start                    # Start all services
  lf services start server              # Start only the server
  LF_ORCHESTRATION_MODE=docker lf services start  # Start all services with Docker`,
	Args: cobra.MaximumNArgs(1),
	Run:  runServicesStart,
}

// servicesStopCmd stops LlamaFarm services
var servicesStopCmd = &cobra.Command{
	Use:   "stop [service-name]",
	Short: "Stop LlamaFarm services",
	Long: `Stop LlamaFarm services using the configured orchestration mode.

Without arguments, stops all services (server, rag, universal-runtime).
With a service name, stops only that specific service.

Available services:
  - server: The main FastAPI server
  - rag: The RAG/Celery worker
  - universal-runtime: The universal runtime server

The orchestration mode is determined by the LF_ORCHESTRATION_MODE environment variable:
  - "native" (default): Stop native processes
  - "docker": Stop Docker containers
  - "auto": Auto-detect (prefers native)

Examples:
  lf services stop                     # Stop all services
  lf services stop server               # Stop only the server
  LF_ORCHESTRATION_MODE=docker lf services stop  # Stop all Docker containers`,
	Args: cobra.MaximumNArgs(1),
	Run:  runServicesStop,
}

func init() {
	rootCmd.AddCommand(servicesCmd)
	servicesCmd.AddCommand(servicesStatusCmd)
	servicesCmd.AddCommand(servicesStartCmd)
	servicesCmd.AddCommand(servicesStopCmd)

	// Add --json flag for machine-readable output
	servicesStatusCmd.Flags().Bool("json", false, "Output status in JSON format")
}

// runServicesStatus is the main entry point for the services status command
func runServicesStatus(cmd *cobra.Command, args []string) {
	// Determine orchestration mode
	orchestrationMode := determineOrchestrationMode()

	var statuses []ServiceInfo
	var orchestrationType string
	dockerAvailable := false

	// Get server URL for health checks
	serverURLToUse := serverURL
	if serverURLToUse == "" {
		serverURLToUse = "http://localhost:8000"
	}

	jsonOutput, _ := cmd.Flags().GetBool("json")

	// Check services based on orchestration mode
	if orchestrationMode == OrchestrationDocker {
		// Docker mode - check Docker containers
		orchestrationType = "docker"

		// Check if Docker is available
		if err := ensureDockerAvailable(); err != nil {
			dockerAvailable = false

			if jsonOutput {
				output := ServicesStatusOutput{
					Services:      []ServiceInfo{},
					DockerRunning: false,
					Orchestration: orchestrationType,
					Timestamp:     time.Now().Unix(),
				}
				json.NewEncoder(os.Stdout).Encode(output)
			} else {
				OutputError("Docker is not available: %v\n", err)
				fmt.Fprintf(os.Stderr, "\nPlease ensure Docker is installed and running.\n")
				fmt.Fprintf(os.Stderr, "Visit https://docs.docker.com/get-docker/ for installation instructions.\n")
			}
			os.Exit(1)
		}

		dockerAvailable = true

		// Check each service defined in ServiceGraph using Docker
		for serviceName := range ServiceGraph {
			status := checkServiceStatusDocker(serviceName, serverURLToUse)
			statuses = append(statuses, status)
		}
	} else {
		// Native mode - check native processes
		orchestrationType = "native"

		// Check each service defined in ServiceGraph using native process manager
		for serviceName := range ServiceGraph {
			status := checkServiceStatusNative(serviceName, serverURLToUse)
			statuses = append(statuses, status)
		}
	}

	// Build output structure
	output := ServicesStatusOutput{
		Services:      statuses,
		DockerRunning: dockerAvailable,
		Orchestration: orchestrationType,
		Timestamp:     time.Now().Unix(),
	}

	// Format output based on --json flag
	if jsonOutput {
		encoder := json.NewEncoder(os.Stdout)
		encoder.SetIndent("", "  ")
		if err := encoder.Encode(output); err != nil {
			OutputError("Failed to encode JSON output: %v\n", err)
			os.Exit(1)
		}
	} else {
		formatServicesStatus(&output)
	}
}

// checkServiceStatusDocker checks the status of a single service using Docker
func checkServiceStatusDocker(serviceName string, serverURL string) ServiceInfo {
	// Get the service definition
	_, exists := ServiceGraph[serviceName]
	if !exists {
		return ServiceInfo{
			Name:          serviceName,
			ContainerName: fmt.Sprintf("llamafarm-%s", serviceName),
			State:         "unknown",
			Orchestration: "docker",
		}
	}

	// Determine container name
	containerName := fmt.Sprintf("llamafarm-%s", serviceName)

	status := ServiceInfo{
		Name:          serviceName,
		ContainerName: containerName,
		State:         "not_found",
		Ports:         make(map[string]string),
		Orchestration: "docker",
	}

	// Check if container exists
	if !containerExists(containerName) {
		return status
	}

	// Check if container is running
	if !isContainerRunning(containerName) {
		status.State = "stopped"
		// Try to get container details even if stopped
		if containerID, image, _, err := getContainerDetails(containerName); err == nil {
			status.ContainerID = containerID
			status.Image = image
		}
		return status
	}

	// Container is running - get full details
	status.State = "running"

	// Get container details
	containerID, image, uptime, err := getContainerDetails(containerName)
	if err != nil {
		logDebug(fmt.Sprintf("Failed to get container details for %s: %v", containerName, err))
	} else {
		status.ContainerID = containerID
		status.Image = image
		status.Uptime = uptime
	}

	// Get port mappings
	if ports, err := GetPublishedPorts(containerName); err == nil {
		status.Ports = ports
	}

	// Get health status if service is running
	status.Health = getServiceHealth(serviceName, serverURL)

	return status
}

// checkServiceStatusNative checks the status of a single service using native processes
func checkServiceStatusNative(serviceName string, serverURL string) ServiceInfo {
	// Get the service definition
	_, exists := ServiceGraph[serviceName]
	if !exists {
		return ServiceInfo{
			Name:          serviceName,
			State:         "unknown",
			Orchestration: "native",
		}
	}

	status := ServiceInfo{
		Name:          serviceName,
		State:         "not_found",
		Ports:         make(map[string]string),
		Orchestration: "native",
	}

	// Get the expected log file path (even if process isn't tracked by current session)
	homeDir, err := os.UserHomeDir()
	if err == nil {
		logFile := filepath.Join(homeDir, ".llamafarm", "logs", fmt.Sprintf("%s.log", serviceName))
		if _, err := os.Stat(logFile); err == nil {
			status.LogFile = logFile
		}
	}

	// Check for PID file first - this is the primary method for native service discovery
	pid, pidFileExists := readPIDFile(serviceName)
	if pidFileExists {
		logDebug(fmt.Sprintf("Found PID file for %s with PID %d", serviceName, pid))

		processRunning := isProcessRunning(pid)
		logDebug(fmt.Sprintf("Process %d running check: %v", pid, processRunning))

		if processRunning {
			// Process is running based on PID file
			status.State = "running"
			status.PID = pid

			// Try to get start time from /proc (Linux) or ps (Unix-like)
			if startTime := getProcessStartTime(pid); !startTime.IsZero() {
				duration := time.Since(startTime)
				status.Uptime = formatDuration(duration)
			}

			// Get health status if service is running
			status.Health = getServiceHealth(serviceName, serverURL)
			return status
		} else {
			// Process not running, but be conservative - only clean up if we're very sure
			logDebug(fmt.Sprintf("Process %d not found, marking as stopped (PID file will be cleaned up by service or stop command)", pid))
			status.State = "stopped"
			// Don't automatically clean up PID file here - let the service or explicit stop command do it
			// This prevents race conditions where we check while the process is starting up
			return status
		}
	}

	// Fallback: Check if we have a global native orchestrator with active processes
	// This handles processes started by the current CLI session
	if globalNativeOrchestrator != nil {
		processMgr := globalNativeOrchestrator.GetProcessManager()
		if processMgr != nil {
			// Check process status from the current orchestrator
			processStatus, err := processMgr.GetProcessStatus(serviceName)
			if err == nil {
				// Process is tracked by current orchestrator
				if processStatus == "running" {
					status.State = "running"

					// Get process info
					processMgr.mu.RLock()
					if proc, exists := processMgr.processes[serviceName]; exists {
						if proc.Cmd != nil && proc.Cmd.Process != nil {
							status.PID = proc.Cmd.Process.Pid
						}
						status.LogFile = proc.LogFile

						// Calculate uptime
						if !proc.StartTime.IsZero() {
							duration := time.Since(proc.StartTime)
							status.Uptime = formatDuration(duration)
						}
					}
					processMgr.mu.RUnlock()

					// Get health status if service is running
					status.Health = getServiceHealth(serviceName, serverURL)
				} else {
					status.State = "stopped"
				}
				return status
			}
		}
	}

	// Final fallback: service is not found
	status.State = "not_found"
	return status
}

// getServiceHealth retrieves health information for a service
func getServiceHealth(serviceName string, serverURL string) *Component {
	if serviceName == "server" {
		// For server, check its own health endpoint
		if hr, err := checkServerHealth(serverURL); err == nil {
			// Find server component in health response
			for _, comp := range hr.Components {
				compName := strings.ToLower(comp.Name)
				if strings.Contains(compName, "server") || comp.Name == "api" {
					return &comp
				}
			}
			// If no specific server component found, use overall health
			return &Component{
				Name:    "server",
				Status:  hr.Status,
				Message: hr.Summary,
			}
		}
	} else if serviceName == "rag" {
		// For RAG, check via server's health endpoint
		if hr, err := checkServerHealth(serverURL); err == nil {
			if ragComp := findRAGComponent(hr); ragComp != nil {
				return ragComp
			}
		}
	}
	return nil
}

// runServicesStart is the main entry point for the services start command
func runServicesStart(cmd *cobra.Command, args []string) {
	// Determine orchestration mode
	orchestrationMode := determineOrchestrationMode()

	// Get server URL for operations
	serverURLToUse := serverURL
	if serverURLToUse == "" {
		serverURLToUse = "http://localhost:8000"
	}

	// Determine which services to start
	var servicesToStart []string
	if len(args) > 0 {
		// Specific service requested
		serviceName := args[0]

		// Validate service name
		if _, exists := ServiceGraph[serviceName]; !exists {
			OutputError("Unknown service: %s\n", serviceName)
			fmt.Fprintf(os.Stderr, "\nAvailable services:\n")
			for name := range ServiceGraph {
				fmt.Fprintf(os.Stderr, "  - %s\n", name)
			}
			os.Exit(1)
		}

		servicesToStart = []string{serviceName}
	} else {
		// Start all services
		for serviceName := range ServiceGraph {
			servicesToStart = append(servicesToStart, serviceName)
		}
	}

	// Start services based on orchestration mode
	if orchestrationMode == OrchestrationDocker {
		startServicesDocker(servicesToStart, serverURLToUse)
	} else {
		startServicesNative(servicesToStart, serverURLToUse)
	}

	// Show final status
	fmt.Println()
	OutputSuccess("Service start complete. Checking status...\n")
	fmt.Println()

	// Re-run status check to show final state
	runServicesStatus(cmd, []string{})
}

// startServicesDocker starts services using Docker
func startServicesDocker(serviceNames []string, serverURL string) {
	OutputProgress("Starting services with Docker orchestration...\n")

	// Check if Docker is available
	if err := ensureDockerAvailable(); err != nil {
		OutputError("Docker is not available: %v\n", err)
		fmt.Fprintf(os.Stderr, "\nPlease ensure Docker is installed and running.\n")
		fmt.Fprintf(os.Stderr, "Visit https://docs.docker.com/get-docker/ for installation instructions.\n")
		os.Exit(1)
	}

	// Start each service
	for _, serviceName := range serviceNames {
		OutputProgress("Starting %s...\n", serviceName)

		containerName := fmt.Sprintf("llamafarm-%s", serviceName)

		// Check if container already exists and is running
		if containerExists(containerName) && isContainerRunning(containerName) {
			OutputProgress("%s is already running\n", serviceName)
			continue
		}

		// Start the container
		if err := startDockerContainer(serviceName); err != nil {
			OutputError("Failed to start %s: %v\n", serviceName, err)
			continue
		}

		OutputSuccess("%s started successfully\n", serviceName)

		// Wait a moment for the service to initialize
		time.Sleep(2 * time.Second)
	}
}

// startServicesNative starts services using native processes
func startServicesNative(serviceNames []string, serverURL string) {
	OutputProgress("Starting services with native orchestration...\n")

	// Ensure native environment is set up
	orchestrator, err := ensureNativeEnvironment(serverURL)
	if err != nil {
		OutputError("Failed to set up native environment: %v\n", err)
		os.Exit(1)
	}

	// Start each service
	for _, serviceName := range serviceNames {
		OutputProgress("Starting %s...\n", serviceName)

		// Check if already running
		if orchestrator.processMgr.IsProcessHealthy(serviceName) {
			OutputProgress("%s is already running\n", serviceName)
			continue
		}

		// Start the service
		var startErr error
		switch serviceName {
		case "server":
			startErr = orchestrator.StartServerNative()
		case "rag":
			startErr = orchestrator.StartRAGNative()
		case "universal-runtime":
			startErr = orchestrator.StartUniversalRuntimeNative()
		default:
			OutputError("Unknown service: %s\n", serviceName)
			continue
		}

		if startErr != nil {
			OutputError("Failed to start %s: %v\n", serviceName, startErr)
			continue
		}

		OutputSuccess("%s started successfully\n", serviceName)
	}
}

// startDockerContainer starts a specific Docker container
func startDockerContainer(serviceName string) error {
	ctx := context.Background()
	cli, err := createDockerClient()
	if err != nil {
		return fmt.Errorf("failed to create Docker client: %w", err)
	}
	defer cli.Close()

	containerName := fmt.Sprintf("llamafarm-%s", serviceName)

	// Check if container exists
	if !containerExists(containerName) {
		return fmt.Errorf("container %s does not exist. Please run 'lf dev' to create it", containerName)
	}

	// Start the container
	if err := cli.ContainerStart(ctx, containerName, container.StartOptions{}); err != nil {
		return fmt.Errorf("failed to start container: %w", err)
	}

	return nil
}

// runServicesStop is the main entry point for the services stop command
func runServicesStop(cmd *cobra.Command, args []string) {
	// Determine orchestration mode
	orchestrationMode := determineOrchestrationMode()

	// Get server URL for operations
	serverURLToUse := serverURL
	if serverURLToUse == "" {
		serverURLToUse = "http://localhost:8000"
	}

	// Determine which services to stop
	var servicesToStop []string
	if len(args) > 0 {
		// Specific service requested
		serviceName := args[0]

		// Validate service name
		if _, exists := ServiceGraph[serviceName]; !exists {
			OutputError("Unknown service: %s\n", serviceName)
			fmt.Fprintf(os.Stderr, "\nAvailable services:\n")
			for name := range ServiceGraph {
				fmt.Fprintf(os.Stderr, "  - %s\n", name)
			}
			os.Exit(1)
		}

		servicesToStop = []string{serviceName}
	} else {
		// Stop all services
		for serviceName := range ServiceGraph {
			servicesToStop = append(servicesToStop, serviceName)
		}
	}

	// Stop services based on orchestration mode
	if orchestrationMode == OrchestrationDocker {
		stopServicesDocker(servicesToStop, serverURLToUse)
	} else {
		stopServicesNative(servicesToStop, serverURLToUse)
	}

	// Show final status
	fmt.Println()
	OutputSuccess("Service stop complete. Checking status...\n")
	fmt.Println()

	// Re-run status check to show final state
	runServicesStatus(cmd, []string{})
}

// stopServicesDocker stops services using Docker
func stopServicesDocker(serviceNames []string, serverURL string) {
	OutputProgress("Stopping services with Docker orchestration...\n")

	// Check if Docker is available
	if err := ensureDockerAvailable(); err != nil {
		OutputError("Docker is not available: %v\n", err)
		fmt.Fprintf(os.Stderr, "\nPlease ensure Docker is installed and running.\n")
		fmt.Fprintf(os.Stderr, "Visit https://docs.docker.com/get-docker/ for installation instructions.\n")
		os.Exit(1)
	}

	// Stop each service
	for _, serviceName := range serviceNames {
		OutputProgress("Stopping %s...\n", serviceName)

		containerName := fmt.Sprintf("llamafarm-%s", serviceName)

		// Check if container exists
		if !containerExists(containerName) {
			OutputProgress("%s container does not exist\n", serviceName)
			continue
		}

		// Check if container is already stopped
		if !isContainerRunning(containerName) {
			OutputProgress("%s is already stopped\n", serviceName)
			continue
		}

		// Stop the container
		if err := stopDockerContainer(serviceName); err != nil {
			OutputError("Failed to stop %s: %v\n", serviceName, err)
			continue
		}

		OutputSuccess("%s stopped successfully\n", serviceName)
	}
}

// stopServicesNative stops services using native processes
func stopServicesNative(serviceNames []string, serverURL string) {
	OutputProgress("Stopping services with native orchestration...\n")

	// Check if we have an active orchestrator
	if globalNativeOrchestrator == nil {
		// No active orchestrator - try to find and stop processes by checking ports/PIDs
		OutputProgress("No active orchestrator found. Attempting to stop processes via system signals...\n")
		stopServicesNativeBySystem(serviceNames)
		return
	}

	processMgr := globalNativeOrchestrator.GetProcessManager()
	if processMgr == nil {
		OutputProgress("No process manager available. Attempting to stop processes via system signals...\n")
		stopServicesNativeBySystem(serviceNames)
		return
	}

	// Stop each service using the process manager
	for _, serviceName := range serviceNames {
		OutputProgress("Stopping %s...\n", serviceName)

		// Check if process is tracked
		if !processMgr.IsProcessHealthy(serviceName) {
			OutputProgress("%s is not running\n", serviceName)
			continue
		}

		// Stop the process
		if err := processMgr.StopProcess(serviceName); err != nil {
			OutputError("Failed to stop %s: %v\n", serviceName, err)
			continue
		}

		OutputSuccess("%s stopped successfully\n", serviceName)
	}
}

// stopServicesNativeBySystem stops native services by using PID files
func stopServicesNativeBySystem(serviceNames []string) {
	for _, serviceName := range serviceNames {
		OutputProgress("Stopping %s...\n", serviceName)

		// Try to read PID file
		pid, pidFileExists := readPIDFile(serviceName)
		if !pidFileExists {
			OutputProgress("%s is not running (no PID file found)\n", serviceName)
			continue
		}

		// Check if process is actually running
		if !isProcessRunning(pid) {
			OutputProgress("%s is not running (stale PID file)\n", serviceName)
			cleanupPIDFile(serviceName)
			continue
		}

		// Find the process
		process, err := os.FindProcess(pid)
		if err != nil {
			OutputError("Failed to find process %d for %s: %v\n", pid, serviceName, err)
			cleanupPIDFile(serviceName)
			continue
		}

		// Try graceful shutdown first (SIGTERM)
		if err := process.Signal(os.Interrupt); err != nil {
			// If graceful shutdown fails, try SIGKILL
			if err := process.Kill(); err != nil {
				OutputError("Failed to stop %s (PID %d): %v\n", serviceName, pid, err)
				continue
			}
		}

		// Wait a moment for the process to stop
		time.Sleep(1 * time.Second)

		// Verify it stopped
		if !isProcessRunning(pid) {
			OutputSuccess("%s stopped successfully\n", serviceName)
			cleanupPIDFile(serviceName)
		} else {
			OutputError("Process %d for %s did not stop after signal\n", pid, serviceName)
		}
	}
}

// stopDockerContainer stops a specific Docker container
func stopDockerContainer(serviceName string) error {
	ctx := context.Background()
	cli, err := createDockerClient()
	if err != nil {
		return fmt.Errorf("failed to create Docker client: %w", err)
	}
	defer cli.Close()

	containerName := fmt.Sprintf("llamafarm-%s", serviceName)

	// Stop the container with a timeout
	timeout := 10 // seconds
	if err := cli.ContainerStop(ctx, containerName, container.StopOptions{
		Timeout: &timeout,
	}); err != nil {
		return fmt.Errorf("failed to stop container: %w", err)
	}

	return nil
}

// getContainerDetails retrieves detailed information about a container
func getContainerDetails(containerName string) (containerID, image, uptime string, err error) {
	ctx := context.Background()
	cli, err := createDockerClient()
	if err != nil {
		return "", "", "", fmt.Errorf("failed to create Docker client: %v", err)
	}
	defer cli.Close()

	// Find the container by name
	containers, err := cli.ContainerList(ctx, container.ListOptions{All: true})
	if err != nil {
		return "", "", "", fmt.Errorf("failed to list containers: %v", err)
	}

	var foundContainerID string
	for _, c := range containers {
		for _, name := range c.Names {
			cleanName := strings.TrimPrefix(name, "/")
			if cleanName == containerName {
				foundContainerID = c.ID
				break
			}
		}
		if foundContainerID != "" {
			break
		}
	}

	if foundContainerID == "" {
		return "", "", "", fmt.Errorf("container %s not found", containerName)
	}

	// Inspect the container for detailed information
	containerJSON, err := cli.ContainerInspect(ctx, foundContainerID)
	if err != nil {
		return "", "", "", fmt.Errorf("failed to inspect container: %v", err)
	}

	// Extract short container ID (first 12 characters)
	shortID := foundContainerID
	if len(shortID) > 12 {
		shortID = shortID[:12]
	}

	// Extract image name
	imageName := containerJSON.Config.Image

	// Calculate uptime if container is running
	uptimeStr := ""
	if containerJSON.State.Running {
		startTime, err := time.Parse(time.RFC3339Nano, containerJSON.State.StartedAt)
		if err == nil {
			duration := time.Since(startTime)
			uptimeStr = formatDuration(duration)
		}
	}

	return shortID, imageName, uptimeStr, nil
}

// formatDuration formats a duration in a human-readable format
func formatDuration(d time.Duration) string {
	if d < time.Minute {
		return fmt.Sprintf("%ds", int(d.Seconds()))
	} else if d < time.Hour {
		return fmt.Sprintf("%dm", int(d.Minutes()))
	} else if d < 24*time.Hour {
		hours := int(d.Hours())
		minutes := int(d.Minutes()) % 60
		if minutes > 0 {
			return fmt.Sprintf("%dh%dm", hours, minutes)
		}
		return fmt.Sprintf("%dh", hours)
	} else {
		days := int(d.Hours()) / 24
		hours := int(d.Hours()) % 24
		if hours > 0 {
			return fmt.Sprintf("%dd%dh", days, hours)
		}
		return fmt.Sprintf("%dd", days)
	}
}

// formatServicesStatus formats the status output in a human-readable format
func formatServicesStatus(output *ServicesStatusOutput) {
	fmt.Println()
	fmt.Println("LlamaFarm Services Status")
	fmt.Println("=========================")
	fmt.Printf("Orchestration: %s\n", output.Orchestration)
	fmt.Println()

	// Track if any services are running
	anyRunning := false
	allRunning := true

	for _, service := range output.Services {
		// Service name header
		fmt.Printf("Service: %s\n", service.Name)

		// Show orchestration-specific info
		if service.Orchestration == "docker" && service.ContainerName != "" {
			fmt.Printf("  Container: %s\n", service.ContainerName)
		}

		// State with icon
		stateIcon := getStateIcon(service.State)
		fmt.Printf("  State: %s %s\n", stateIcon, service.State)

		if service.State == "running" {
			anyRunning = true

			// PID (for native processes)
			if service.PID > 0 {
				fmt.Printf("  PID: %d\n", service.PID)
			}

			// Container ID (for Docker)
			if service.ContainerID != "" {
				fmt.Printf("  Container ID: %s\n", service.ContainerID)
			}

			// Image (for Docker)
			if service.Image != "" {
				fmt.Printf("  Image: %s\n", service.Image)
			}

			// Log file (for native processes)
			if service.LogFile != "" {
				fmt.Printf("  Log File: %s\n", service.LogFile)
			}

			// Uptime
			if service.Uptime != "" {
				fmt.Printf("  Uptime: %s\n", service.Uptime)
			}

			// Ports
			if len(service.Ports) > 0 {
				fmt.Printf("  Ports:\n")
				for containerPort, hostPort := range service.Ports {
					fmt.Printf("    %s -> %s\n", containerPort, hostPort)
				}
			}

			// Health status
			if service.Health != nil {
				healthIcon := getHealthIcon(service.Health.Status)
				fmt.Printf("  Health: %s %s", healthIcon, service.Health.Status)
				if service.Health.Message != "" {
					fmt.Printf(" - %s", service.Health.Message)
				}
				fmt.Println()
				if service.Health.LatencyMs > 0 {
					fmt.Printf("  Latency: %dms\n", service.Health.LatencyMs)
				}
			}
		} else {
			allRunning = false

			// Show container ID and image even if stopped (Docker)
			if service.ContainerID != "" {
				fmt.Printf("  Container ID: %s\n", service.ContainerID)
			}
			if service.Image != "" {
				fmt.Printf("  Image: %s\n", service.Image)
			}

			// Show log file even if stopped (native)
			if service.LogFile != "" {
				fmt.Printf("  Log File: %s\n", service.LogFile)
			}
		}

		fmt.Println()
	}

	// Summary and helpful messages
	if !anyRunning {
		fmt.Println("⚠️  No services are currently running")
		fmt.Println()
		fmt.Println("To start services:")
		if output.Orchestration == "docker" {
			fmt.Println("  lf services start  (or set LF_ORCHESTRATION_MODE=native to use native processes)")
		} else {
			fmt.Println("  lf services start")
		}
		fmt.Println()
	} else if !allRunning {
		fmt.Println("⚠️  Some services are not running")
		fmt.Println()
		fmt.Println("To start all services:")
		fmt.Println("  lf services start")
		fmt.Println()
	} else {
		fmt.Println("✅ All services are running")
		fmt.Println()
	}
}

// getStateIcon returns an icon for the container state
func getStateIcon(state string) string {
	switch state {
	case "running":
		return "✓"
	case "stopped":
		return "✗"
	case "not_found":
		return "○"
	default:
		return "?"
	}
}

// getHealthIcon returns an icon for the health status
func getHealthIcon(status string) string {
	status = strings.ToLower(strings.TrimSpace(status))
	switch status {
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

// isPortInUse checks if a TCP port is in use on localhost
func isPortInUse(port string) bool {
	timeout := 500 * time.Millisecond
	conn, err := net.DialTimeout("tcp", "localhost:"+port, timeout)
	if err != nil {
		return false
	}
	conn.Close()
	return true
}

// readPIDFile reads the PID from a service's PID file
func readPIDFile(serviceName string) (int, bool) {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return 0, false
	}

	pidFile := filepath.Join(homeDir, ".llamafarm", "pids", fmt.Sprintf("%s.pid", serviceName))
	data, err := os.ReadFile(pidFile)
	if err != nil {
		return 0, false
	}

	var pid int
	_, err = fmt.Sscanf(string(data), "%d", &pid)
	if err != nil {
		return 0, false
	}

	return pid, true
}

// isProcessRunning checks if a process with the given PID is running
func isProcessRunning(pid int) bool {
	if pid <= 0 {
		return false
	}

	process, err := os.FindProcess(pid)
	if err != nil {
		return false
	} else {
		err := process.Signal(syscall.Signal(0))
		if err == nil {
			return true
		}
	}

	return false
}

// getProcessStartTime gets the start time of a process (best effort)
func getProcessStartTime(pid int) time.Time {
	// This is platform-specific and best effort
	// On Linux, we can read from /proc
	// On macOS/BSD, we can use ps
	// On Windows, this is more complex

	if runtime.GOOS == "linux" {
		// Try to read from /proc
		statFile := fmt.Sprintf("/proc/%d/stat", pid)
		data, err := os.ReadFile(statFile)
		if err != nil {
			return time.Time{}
		}

		// Parse the stat file to get start time
		// Field 22 is start time in clock ticks since boot
		fields := strings.Fields(string(data))
		if len(fields) < 22 {
			return time.Time{}
		}

		// This is complex to convert accurately, so we'll skip it for now
		// The uptime will just not be available in this case
		return time.Time{}
	}

	// For other platforms, we could use ps, but it's complex and may not be reliable
	// Return zero time to indicate we don't have the start time
	return time.Time{}
}

// cleanupPIDFile removes a stale PID file
func cleanupPIDFile(serviceName string) {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return
	}

	pidFile := filepath.Join(homeDir, ".llamafarm", "pids", fmt.Sprintf("%s.pid", serviceName))
	os.Remove(pidFile) // Ignore errors
}
