package cmd

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// ContainerOrchestrator manages the startup sequence and lifecycle of multiple containers
type ContainerOrchestrator struct {
	serverContainerName   string
	ragContainerName      string
	chromadbContainerName string
	networkManager        *NetworkManager
}

// NewContainerOrchestrator creates a new orchestrator for multi-container setup
func NewContainerOrchestrator() *ContainerOrchestrator {
	return &ContainerOrchestrator{
		serverContainerName:   "llamafarm-server",
		ragContainerName:      "llamafarm-rag",
		chromadbContainerName: "llamafarm-chromadb",
		networkManager:        NewNetworkManager(),
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
		Env: map[string]string{
			"CHROMADB_HOST": co.chromadbContainerName, // Container name resolves via Docker network
			"CHROMADB_PORT": "8000",                   // Internal container port
		},
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

// startChromaDBContainer starts the ChromaDB server container
func (co *ContainerOrchestrator) startChromaDBContainer() error {
	// Ensure network exists
	if err := co.networkManager.EnsureNetwork(); err != nil {
		return fmt.Errorf("failed to ensure network: %v", err)
	}

	networkName := co.networkManager.GetNetworkName()

	// Use official ChromaDB image (same version as docker-compose)
	image := "chromadb/chroma:1.0.20"

	// Prepare container specification
	spec := ContainerRunSpec{
		Name:  co.chromadbContainerName,
		Image: image,
		StaticPorts: []PortMapping{
			{Host: 8001, Container: 8000, Protocol: "tcp"}, // Expose on 8001
		},
		Env: map[string]string{
			"PERSIST_DIRECTORY":       "/chroma/data",
			"IS_PERSISTENT":           "TRUE",
			"CHROMA_SERVER_HOST":      "0.0.0.0",
			"CHROMA_SERVER_HTTP_PORT": "8000",
		},
		Volumes: []string{
			"llamafarm-chromadb-data:/chroma/data", // Named volume for persistence
		},
		Labels: map[string]string{
			"llamafarm.component": "chromadb",
			"llamafarm.managed":   "true",
			"llamafarm.issue":     "279",
		},
	}

	logDebug(fmt.Sprintf("Starting ChromaDB container with network: %s", networkName))

	// Start container with network support
	_, err := StartContainerWithNetwork(spec, networkName, &PortResolutionPolicy{
		PreferredHostPort: 8001,
		Forced:            true, // Force port 8001 for consistency
	})

	return err
}

// waitForChromaDBReadiness waits for ChromaDB to become ready
func (co *ContainerOrchestrator) waitForChromaDBReadiness(timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	return WaitForReadiness(ctx, func() error {
		// Check if ChromaDB is accessible
		client := &http.Client{Timeout: 2 * time.Second}
		resp, err := client.Get("http://localhost:8001/api/v1/heartbeat")
		if err != nil {
			return fmt.Errorf("chromadb not reachable: %v", err)
		}
		defer resp.Body.Close()
		return nil
	}, 2*time.Second)
}
