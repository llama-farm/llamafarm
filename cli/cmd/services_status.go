package cmd

import (
	"fmt"
	"strings"
)

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
