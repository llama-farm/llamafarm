package cmd

import (
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/spf13/cobra"
)

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

// ensureSourceVersion checks and updates source code version if needed
// This should be called at the start of all services commands to ensure
// source code matches the CLI version
func ensureSourceVersion() error {
	// Create UV manager
	uvMgr, err := NewUVManager()
	if err != nil {
		return fmt.Errorf("failed to create UV manager: %w", err)
	}

	// Create Python environment manager
	pythonMgr, err := NewPythonEnvManager(uvMgr)
	if err != nil {
		return fmt.Errorf("failed to create Python environment manager: %w", err)
	}

	// Create source manager and ensure source version matches CLI
	srcMgr, err := NewSourceManager(pythonMgr)
	if err != nil {
		return fmt.Errorf("failed to create source manager: %w", err)
	}

	// This will check version and download/update if needed
	if err := srcMgr.EnsureSource(); err != nil {
		return fmt.Errorf("failed to ensure source code version: %w", err)
	}

	return nil
}

// runServicesStatus is the main entry point for the services status command
func runServicesStatus(cmd *cobra.Command, args []string) {
	// Ensure source code version matches CLI version
	if err := ensureSourceVersion(); err != nil {
		OutputWarning("Warning: %v\n", err)
		// Continue anyway - user might just want to check status
	}

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
		manager := NewDockerServiceManager(serverURLToUse)

		// Check if Docker is available
		if err := manager.IsAvailable(); err != nil {
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

		// Check each service using Docker manager
		for serviceName := range ServiceGraph {
			status := manager.CheckStatus(serviceName, serverURLToUse)
			statuses = append(statuses, status)
		}
	} else {
		// Native mode - check native processes
		orchestrationType = "native"

		// Check each service using native status checker
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

// runServicesStart is the main entry point for the services start command
func runServicesStart(cmd *cobra.Command, args []string) {
	// Ensure source code version matches CLI version
	if err := ensureSourceVersion(); err != nil {
		OutputError("Failed to ensure source code version: %v\n", err)
		os.Exit(1)
	}

	// Determine orchestration mode
	orchestrationMode := determineOrchestrationMode()

	// Get server URL for operations
	serverURLToUse := serverURL
	if serverURLToUse == "" {
		serverURLToUse = "http://localhost:8000"
	}

	// Determine which services to start
	servicesToStart := getServicesToManage(args)

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

// runServicesStop is the main entry point for the services stop command
func runServicesStop(cmd *cobra.Command, args []string) {
	// Ensure source code version matches CLI version
	// (Important even for stop - ensures source is ready if user restarts)
	if err := ensureSourceVersion(); err != nil {
		OutputWarning("Warning: %v\n", err)
		// Continue anyway - stopping services doesn't require source to be perfect
	}

	// Determine orchestration mode
	orchestrationMode := determineOrchestrationMode()

	// Get server URL for operations
	serverURLToUse := serverURL
	if serverURLToUse == "" {
		serverURLToUse = "http://localhost:8000"
	}

	// Determine which services to stop
	servicesToStop := getServicesToManage(args)

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

// getServicesToManage determines which services to manage based on command arguments
func getServicesToManage(args []string) []string {
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

		return []string{serviceName}
	}

	// Start all services
	var services []string
	for serviceName := range ServiceGraph {
		services = append(services, serviceName)
	}
	return services
}
