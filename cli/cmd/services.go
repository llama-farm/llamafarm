package cmd

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/llamafarm/cli/cmd/orchestrator"
	"github.com/llamafarm/cli/cmd/utils"
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

// runServicesStatus is the main entry point for the services status command
func runServicesStatus(cmd *cobra.Command, args []string) {
	var statuses []ServiceInfo

	// Get server URL for health checks
	serverURLToUse := serverURL
	if serverURLToUse == "" {
		serverURLToUse = "http://localhost:14345"
	}

	jsonOutput, _ := cmd.Flags().GetBool("json")

	sm, err := orchestrator.NewServiceManager(serverURLToUse)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize service manager: %v\n", err)
		os.Exit(1)
	}

	// Check each service using native status checker
	statusInfos, err := sm.GetServicesStatus()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to get services status: %v\n", err)
		os.Exit(1)
	}

	// Convert orchestrator.ServiceStatusInfo to cmd.ServiceInfo
	statuses = make([]ServiceInfo, 0, len(statusInfos))
	for _, info := range statusInfos {
		serviceInfo := ServiceInfo{
			Name:    info.Name,
			State:   info.State,
			PID:     info.PID,
			LogFile: info.LogFile,
			Health:  info.Health,
		}

		// Format uptime as a string if the service is running
		if info.Uptime > 0 {
			serviceInfo.Uptime = formatUptime(info.Uptime)
		}

		statuses = append(statuses, serviceInfo)
	}

	// Build output structure
	output := ServicesStatusOutput{
		Services:  statuses,
		Timestamp: time.Now().Unix(),
	}

	// Format output based on --json flag
	if jsonOutput {
		encoder := json.NewEncoder(os.Stdout)
		encoder.SetIndent("", "  ")
		if err := encoder.Encode(output); err != nil {
			utils.OutputError("Failed to encode JSON output: %v\n", err)
			os.Exit(1)
		}
	} else {
		formatServicesStatus(&output)
	}
}

// runServicesStart is the main entry point for the services start command
func runServicesStart(cmd *cobra.Command, args []string) {
	// Get server URL for operations
	serverURLToUse := serverURL
	if serverURLToUse == "" {
		serverURLToUse = "http://localhost:14345"
	}

	// Determine which services to start
	servicesToStart := getServicesToManage(args)

	// Start services based on orchestration mode
	var failedServices []string
	sm, err := orchestrator.NewServiceManager(serverURLToUse)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize service manager: %v\n", err)
		os.Exit(1)
	}
	utils.OutputInfo("Starting services (%s). This may take a minute...", strings.Join(servicesToStart, ", "))
	if err := sm.EnsureServices(servicesToStart...); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to start services: %v\n", err)
		os.Exit(1)
	}

	// Only show status if more than one service was requested
	if len(servicesToStart) > 1 {
		utils.OutputSuccess("Service start complete. Checking status...\n")

		// Re-run status check to show final state
		runServicesStatus(cmd, []string{})
	}

	// If any services failed, exit with error code
	if len(failedServices) > 0 {
		fmt.Println()
		utils.OutputError("Failed to start the following services: %v\n", failedServices)
		os.Exit(1)
	}
}

// runServicesStop is the main entry point for the services stop command
func runServicesStop(cmd *cobra.Command, args []string) {
	// Get server URL for operations
	serverURLToUse := serverURL
	if serverURLToUse == "" {
		serverURLToUse = "http://localhost:14345"
	}

	// Determine which services to stop
	servicesToStop := getServicesToManage(args)

	sm, err := orchestrator.NewServiceManager(serverURLToUse)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize service manager: %v\n", err)
		os.Exit(1)
	}
	if err := sm.StopServices(servicesToStop...); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to stop services: %v\n", err)
		os.Exit(1)
	}

	// Show final status
	fmt.Println()
	utils.OutputSuccess("Service stop complete. Checking status...\n")
	fmt.Println()

	// Re-run status check to show final state
	runServicesStatus(cmd, []string{})
}

// formatUptime formats a duration into a human-readable uptime string
func formatUptime(d time.Duration) string {
	if d < time.Minute {
		return fmt.Sprintf("%ds", int(d.Seconds()))
	} else if d < time.Hour {
		return fmt.Sprintf("%dm", int(d.Minutes()))
	} else if d < 24*time.Hour {
		hours := int(d.Hours())
		minutes := int(d.Minutes()) % 60
		return fmt.Sprintf("%dh%dm", hours, minutes)
	} else {
		days := int(d.Hours()) / 24
		hours := int(d.Hours()) % 24
		return fmt.Sprintf("%dd%dh", days, hours)
	}
}

// getServicesToManage determines which services to manage based on command arguments
func getServicesToManage(args []string) []string {
	if len(args) > 0 {
		// Support comma-separated service names and return a list
		serviceNames := strings.Split(args[0], ",")
		for i := range serviceNames {
			serviceNames[i] = strings.TrimSpace(serviceNames[i])
		}

		// Filter out empty service names to prevent invalid arguments
		filteredServiceNames := make([]string, 0, len(serviceNames))
		for _, name := range serviceNames {
			if name != "" {
				filteredServiceNames = append(filteredServiceNames, name)
			}
		}

		return filteredServiceNames
	}

	// Start all services
	var services []string
	for serviceName := range orchestrator.ServiceGraph {
		services = append(services, serviceName)
	}
	return services
}
