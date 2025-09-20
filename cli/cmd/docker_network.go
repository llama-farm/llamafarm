package cmd

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// NetworkManager handles Docker network creation and management for multi-container setup
type NetworkManager struct {
	networkName string
	configDir   string
}

// NewNetworkManager creates a new network manager instance
func NewNetworkManager() *NetworkManager {
	// Get or create llamafarm config directory
	homeDir, err := os.UserHomeDir()
	if err != nil {
		logDebug("Could not get home directory, using /tmp")
		homeDir = "/tmp"
	}

	configDir := filepath.Join(homeDir, ".llamafarm")
	if err := os.MkdirAll(configDir, 0755); err != nil {
		logDebug(fmt.Sprintf("Could not create config directory: %v", err))
	}

	return &NetworkManager{
		configDir: configDir,
	}
}

// getNetworkName returns the current network name, creating one if needed
func (nm *NetworkManager) getNetworkName() string {
	if nm.networkName != "" {
		return nm.networkName
	}

	// Try to load existing network name from file
	networkFile := filepath.Join(nm.configDir, "network")
	if data, err := os.ReadFile(networkFile); err == nil {
		networkName := strings.TrimSpace(string(data))
		if networkName != "" && nm.networkExists(networkName) {
			nm.networkName = networkName
			return nm.networkName
		}
	}

	// Create new network name with timestamp
	timestamp := time.Now().Format("20060102-150405")
	nm.networkName = fmt.Sprintf("llamafarm-%s", timestamp)

	// Save network name to file
	if err := os.WriteFile(networkFile, []byte(nm.networkName), 0644); err != nil {
		logDebug(fmt.Sprintf("Could not save network name: %v", err))
	}

	return nm.networkName
}

// networkExists checks if a Docker network exists
func (nm *NetworkManager) networkExists(networkName string) bool {
	cmd := exec.Command("docker", "network", "ls", "--format", "{{.Name}}")
	out, err := cmd.Output()
	if err != nil {
		return false
	}

	for _, line := range strings.Split(string(out), "\n") {
		if strings.TrimSpace(line) == networkName {
			return true
		}
	}
	return false
}

// EnsureNetwork creates the Docker network if it doesn't exist
func (nm *NetworkManager) EnsureNetwork() error {
	networkName := nm.getNetworkName()

	if nm.networkExists(networkName) {
		logDebug(fmt.Sprintf("Network %s already exists", networkName))
		return nil
	}

	logDebug(fmt.Sprintf("Creating Docker network: %s", networkName))
	cmd := exec.Command("docker", "network", "create",
		"--driver", "bridge",
		"--label", "llamafarm.managed=true",
		networkName)

	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to create Docker network: %v\n%s", err, string(out))
	}

	if debug && len(out) > 0 {
		logDebug(fmt.Sprintf("docker network create output: %s", string(out)))
	}

	return nil
}

// GetNetworkName returns the current network name (creates if needed)
func (nm *NetworkManager) GetNetworkName() string {
	return nm.getNetworkName()
}

// CleanupNetwork removes the Docker network if it exists
func (nm *NetworkManager) CleanupNetwork() error {
	networkName := nm.getNetworkName()

	if !nm.networkExists(networkName) {
		return nil
	}

	logDebug(fmt.Sprintf("Removing Docker network: %s", networkName))
	cmd := exec.Command("docker", "network", "rm", networkName)

	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to remove Docker network: %v\n%s", err, string(out))
	}

	if debug && len(out) > 0 {
		logDebug(fmt.Sprintf("docker network rm output: %s", string(out)))
	}

	// Remove network file
	networkFile := filepath.Join(nm.configDir, "network")
	os.Remove(networkFile)

	return nil
}

// ListContainersOnNetwork returns containers connected to the network
func (nm *NetworkManager) ListContainersOnNetwork() ([]string, error) {
	networkName := nm.getNetworkName()

	if !nm.networkExists(networkName) {
		return []string{}, nil
	}

	cmd := exec.Command("docker", "network", "inspect", networkName,
		"--format", "{{range .Containers}}{{.Name}} {{end}}")

	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("failed to inspect network: %v", err)
	}

	containerNames := strings.Fields(strings.TrimSpace(string(out)))
	return containerNames, nil
}
