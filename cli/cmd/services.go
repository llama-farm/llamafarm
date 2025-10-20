package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/docker/docker/api/types/container"
	"github.com/spf13/cobra"
)

// ServiceInfo represents the status of a single service
type ServiceInfo struct {
	Name          string            `json:"name"`
	ContainerName string            `json:"container_name"`
	State         string            `json:"state"` // "running", "stopped", "not_found"
	ContainerID   string            `json:"container_id,omitempty"`
	Image         string            `json:"image,omitempty"`
	Ports         map[string]string `json:"ports,omitempty"`
	Health        *Component        `json:"health,omitempty"`
	Uptime        string            `json:"uptime,omitempty"`
}

// ServicesStatusOutput represents the complete status output
type ServicesStatusOutput struct {
	Services      []ServiceInfo `json:"services"`
	DockerRunning bool          `json:"docker_running"`
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

This command shows:
  - Container running state
  - Port mappings
  - Health status (if service is running)
  - Container ID and image information
  - Uptime

This is a read-only operation that never auto-starts services.`,
	Run: runServicesStatus,
}

func init() {
	rootCmd.AddCommand(servicesCmd)
	servicesCmd.AddCommand(servicesStatusCmd)

	// Add --json flag for machine-readable output
	servicesStatusCmd.Flags().Bool("json", false, "Output status in JSON format")
}

// runServicesStatus is the main entry point for the services status command
func runServicesStatus(cmd *cobra.Command, args []string) {
	// Check if Docker is available
	dockerAvailable := true
	if err := ensureDockerAvailable(); err != nil {
		dockerAvailable = false
		jsonOutput, _ := cmd.Flags().GetBool("json")

		if jsonOutput {
			output := ServicesStatusOutput{
				Services:      []ServiceInfo{},
				DockerRunning: false,
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

	// Collect status for each service in the ServiceGraph
	var statuses []ServiceInfo

	// Get server URL for health checks
	serverURLToUse := serverURL
	if serverURLToUse == "" {
		serverURLToUse = "http://localhost:8000"
	}

	// Check each service defined in ServiceGraph
	for serviceName := range ServiceGraph {
		status := checkServiceStatus(serviceName, serverURLToUse)
		statuses = append(statuses, status)
	}

	// Build output structure
	output := ServicesStatusOutput{
		Services:      statuses,
		DockerRunning: dockerAvailable,
		Timestamp:     time.Now().Unix(),
	}

	// Format output based on --json flag
	jsonOutput, _ := cmd.Flags().GetBool("json")
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

// checkServiceStatus checks the status of a single service
func checkServiceStatus(serviceName string, serverURL string) ServiceInfo {
	// Get the service definition
	_, exists := ServiceGraph[serviceName]
	if !exists {
		return ServiceInfo{
			Name:          serviceName,
			ContainerName: fmt.Sprintf("llamafarm-%s", serviceName),
			State:         "unknown",
		}
	}

	// Determine container name
	containerName := fmt.Sprintf("llamafarm-%s", serviceName)

	status := ServiceInfo{
		Name:          serviceName,
		ContainerName: containerName,
		State:         "not_found",
		Ports:         make(map[string]string),
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
	if serviceName == "server" {
		// For server, check its own health endpoint
		if hr, err := checkServerHealth(serverURL); err == nil {
			// Find server component in health response
			for _, comp := range hr.Components {
				compName := strings.ToLower(comp.Name)
				if strings.Contains(compName, "server") || comp.Name == "api" {
					status.Health = &comp
					break
				}
			}
			// If no specific server component found, use overall health
			if status.Health == nil {
				status.Health = &Component{
					Name:    "server",
					Status:  hr.Status,
					Message: hr.Summary,
				}
			}
		}
	} else if serviceName == "rag" {
		// For RAG, check via server's health endpoint
		if hr, err := checkServerHealth(serverURL); err == nil {
			if ragComp := findRAGComponent(hr); ragComp != nil {
				status.Health = ragComp
			}
		}
	}

	return status
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
	fmt.Println()

	// Track if any services are running
	anyRunning := false
	allRunning := true

	for _, service := range output.Services {
		// Service name header
		fmt.Printf("Service: %s\n", service.Name)
		fmt.Printf("  Container: %s\n", service.ContainerName)

		// State with icon
		stateIcon := getStateIcon(service.State)
		fmt.Printf("  State: %s %s\n", stateIcon, service.State)

		if service.State == "running" {
			anyRunning = true

			// Container ID
			if service.ContainerID != "" {
				fmt.Printf("  Container ID: %s\n", service.ContainerID)
			}

			// Image
			if service.Image != "" {
				fmt.Printf("  Image: %s\n", service.Image)
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

			// Show container ID and image even if stopped
			if service.ContainerID != "" {
				fmt.Printf("  Container ID: %s\n", service.ContainerID)
			}
			if service.Image != "" {
				fmt.Printf("  Image: %s\n", service.Image)
			}
		}

		fmt.Println()
	}

	// Summary and helpful messages
	if !anyRunning {
		fmt.Println("⚠️  No services are currently running")
		fmt.Println()
		fmt.Println("To start services:")
		fmt.Println("  lf dev")
		fmt.Println()
	} else if !allRunning {
		fmt.Println("⚠️  Some services are not running")
		fmt.Println()
		fmt.Println("To start all services:")
		fmt.Println("  lf dev")
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
