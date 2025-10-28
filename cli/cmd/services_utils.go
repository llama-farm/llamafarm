package cmd

import (
	"fmt"
	"net"
	"strings"
	"time"
)

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

// waitForCondition polls a condition function until it returns true or timeout is reached
// This replaces fixed sleeps with active polling for better responsiveness
func waitForCondition(condition func() bool, timeout time.Duration, pollInterval time.Duration) bool {
	deadline := time.Now().Add(timeout)

	for time.Now().Before(deadline) {
		if condition() {
			return true
		}
		time.Sleep(pollInterval)
	}

	// Final check after timeout
	return condition()
}

// waitForServiceReady waits for a service to be ready by checking its health
func waitForServiceReady(serviceName string, serverURL string, timeout time.Duration) bool {
	return waitForCondition(func() bool {
		health := getServiceHealth(serviceName, serverURL)
		return health != nil && strings.ToLower(health.Status) == "healthy"
	}, timeout, 500*time.Millisecond)
}
