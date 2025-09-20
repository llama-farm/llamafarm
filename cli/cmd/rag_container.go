package cmd

import (
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"
)

var ragContainerName = "llamafarm-rag"

// RAGContainerManager handles RAG container lifecycle management
type RAGContainerManager struct {
	networkManager *NetworkManager
}

// NewRAGContainerManager creates a new RAG container manager
func NewRAGContainerManager(networkManager *NetworkManager) *RAGContainerManager {
	return &RAGContainerManager{
		networkManager: networkManager,
	}
}

// IsRAGContainerRunning checks if the RAG container is currently running
func (rcm *RAGContainerManager) IsRAGContainerRunning() bool {
	return isContainerRunning(ragContainerName)
}

// StartRAGContainer starts the RAG container and connects it to the network
func (rcm *RAGContainerManager) StartRAGContainer() error {
	// Ensure network exists
	if err := rcm.networkManager.EnsureNetwork(); err != nil {
		return fmt.Errorf("failed to ensure network: %v", err)
	}

	networkName := rcm.networkManager.GetNetworkName()

	// Remove existing container if it exists
	if err := removeContainer(ragContainerName); err != nil {
		return fmt.Errorf("failed to remove existing RAG container: %v", err)
	}

	// Get RAG image URL
	imageURL, err := getImageURL("rag")
	if err != nil {
		return fmt.Errorf("failed to get RAG image URL: %v", err)
	}

	// Pull the image
	logDebug(fmt.Sprintf("Pulling RAG image: %s", imageURL))
	if err := pullImage(imageURL); err != nil {
		return fmt.Errorf("failed to pull RAG image: %v", err)
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

	// Build docker run command
	args := []string{
		"run",
		"-d", // detached
		"--name", ragContainerName,
		"--network", networkName,
		"--restart", "unless-stopped",
		"-e", fmt.Sprintf("LF_DATA_DIR=%s", "/var/lib/llamafarm"), // Container path
		"-v", fmt.Sprintf("%s:/var/lib/llamafarm", dataDir), // Volume mount
		"--label", "llamafarm.component=rag",
		"--label", "llamafarm.managed=true",
		imageURL,
	}

	logDebug(fmt.Sprintf("Starting RAG container with command: docker %s", fmt.Sprintf("%v", args)))

	cmd := exec.Command("docker", args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to start RAG container: %v\n%s", err, string(out))
	}

	if debug && len(out) > 0 {
		logDebug(fmt.Sprintf("docker run output: %s", string(out)))
	}

	logDebug("RAG container started successfully")
	return nil
}

// StopRAGContainer stops and removes the RAG container
func (rcm *RAGContainerManager) StopRAGContainer() error {
	if !containerExists(ragContainerName) {
		logDebug("RAG container does not exist")
		return nil
	}

	logDebug("Stopping RAG container")

	// Stop the container first
	stopCmd := exec.Command("docker", "stop", ragContainerName)
	if out, err := stopCmd.CombinedOutput(); err != nil {
		logDebug(fmt.Sprintf("Warning: failed to stop RAG container: %v\n%s", err, string(out)))
	}

	// Remove the container
	return removeContainer(ragContainerName)
}

// WaitForRAGContainer waits for the RAG container to be ready by checking its logs
func (rcm *RAGContainerManager) WaitForRAGContainer(timeout time.Duration) error {
	if !rcm.IsRAGContainerRunning() {
		return fmt.Errorf("RAG container is not running")
	}

	logDebug("Waiting for RAG container to be ready...")

	deadline := time.Now().Add(timeout)

	for time.Now().Before(deadline) {
		// Check container logs for startup completion
		cmd := exec.Command("docker", "logs", "--tail", "10", ragContainerName)
		out, err := cmd.Output()
		if err != nil {
			logDebug(fmt.Sprintf("Could not check RAG container logs: %v", err))
			time.Sleep(2 * time.Second)
			continue
		}

		logs := string(out)

		// Look for successful startup indicators
		if strings.Contains(logs, "Starting RAG Celery worker service") {
			// Give it a moment to fully initialize
			time.Sleep(3 * time.Second)
			logDebug("RAG container is ready")
			return nil
		}

		// Check for error conditions
		if strings.Contains(logs, "Error") || strings.Contains(logs, "Failed") {
			return fmt.Errorf("RAG container failed to start properly. Check logs: docker logs %s", ragContainerName)
		}

		time.Sleep(2 * time.Second)
	}

	return fmt.Errorf("RAG container did not become ready within %v", timeout)
}

// GetRAGContainerLogs returns recent logs from the RAG container
func (rcm *RAGContainerManager) GetRAGContainerLogs(lines int) (string, error) {
	if !containerExists(ragContainerName) {
		return "", fmt.Errorf("RAG container does not exist")
	}

	cmd := exec.Command("docker", "logs", "--tail", fmt.Sprintf("%d", lines), ragContainerName)
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("failed to get RAG container logs: %v", err)
	}

	return string(out), nil
}
