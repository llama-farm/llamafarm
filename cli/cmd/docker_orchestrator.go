package cmd

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
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

// startRAGContainer starts the RAG container connected to the custom network
func (co *ContainerOrchestrator) startRAGContainer() error {
	// Ensure network exists
	if err := co.networkManager.EnsureNetwork(); err != nil {
		return fmt.Errorf("failed to ensure network: %v", err)
	}

	networkName := co.networkManager.GetNetworkName()

	// Get RAG image
	image, err := getImageURL("rag")
	if err != nil {
		return fmt.Errorf("failed to get RAG image URL: %v", err)
	}

	// Prepare container specification
	homeDir, _ := os.UserHomeDir()
	spec := ContainerRunSpec{
		Name:  co.ragContainerName,
		Image: image,
		StaticPorts: []PortMapping{
			{Host: 0, Container: 8001, Protocol: "tcp"}, // Dynamic port for RAG
		},
		Env: make(map[string]string),
		Volumes: []string{
			func() string {
				homeLlamaFarmPath := filepath.Join(homeDir, ".llamafarm")
				dockerPath := convertToDockerPath(homeLlamaFarmPath)
				volumeMount := fmt.Sprintf("%s:%s", dockerPath, "/var/lib/llamafarm")

				// Debug logging for RAG home directory volume mount
				logDebug(fmt.Sprintf("RAG home volume mount: %s", volumeMount))

				return volumeMount
			}(),
		},
		Labels: map[string]string{
			"llamafarm.component": "rag",
			"llamafarm.managed":   "true",
		},
		User: getCurrentUserGroup(),
	}

	// Mount effective working directory into the container at the same path
	if err := setupWorkdirVolumeMount(&spec); err != nil {
		return fmt.Errorf("failed to configure working directory volume: %v", err)
	}

	logDebug(fmt.Sprintf("Starting RAG container with network: %s", networkName))

	// Use the new Docker SDK-based container starter with network support
	_, err = StartContainerWithNetwork(spec, networkName, &PortResolutionPolicy{
		PreferredHostPort: 0, // Use dynamic ports for RAG
		Forced:            false,
	})

	return err
}

// waitForRAGReadiness waits for the RAG container to become ready
func (co *ContainerOrchestrator) waitForRAGReadiness(timeout time.Duration, serverURL string) error {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	return WaitForReadiness(ctx, func() error {
		// Check RAG health via server health endpoint
		hr, err := checkServerHealth(serverURL)
		if err != nil {
			return err
		}

		ragComponent := findRAGComponent(hr)
		if ragComponent == nil {
			return fmt.Errorf("RAG component not found in health response")
		}

		if !strings.EqualFold(ragComponent.Status, "healthy") {
			return fmt.Errorf("RAG component status: %s", ragComponent.Status)
		}

		return nil
	}, 2*time.Second)
}
