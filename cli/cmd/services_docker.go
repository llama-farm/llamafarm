package cmd

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/docker/docker/api/types/container"
)

// DockerServiceManager handles Docker-specific service operations
type DockerServiceManager struct {
	serverURL string
}

// NewDockerServiceManager creates a new Docker service manager
func NewDockerServiceManager(serverURL string) *DockerServiceManager {
	return &DockerServiceManager{serverURL: serverURL}
}

// IsAvailable checks if Docker is available
func (d *DockerServiceManager) IsAvailable() error {
	return ensureDockerAvailable()
}

// CheckStatus checks the status of a single service using Docker
func (d *DockerServiceManager) CheckStatus(serviceName string, serverURL string) ServiceInfo {
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

// StartService starts a Docker container for a service
func (d *DockerServiceManager) StartService(serviceName string) error {
	containerName := fmt.Sprintf("llamafarm-%s", serviceName)

	// Check if container already exists and is running
	if containerExists(containerName) && isContainerRunning(containerName) {
		return nil // Already running
	}

	// Start the container
	if err := d.startDockerContainer(serviceName); err != nil {
		return fmt.Errorf("failed to start container: %w", err)
	}

	// Wait for service to be ready with active polling instead of fixed sleep
	waitForServiceReady(serviceName, d.serverURL, 10*time.Second)

	return nil
}

// StopService stops a Docker container for a service
func (d *DockerServiceManager) StopService(serviceName string) error {
	containerName := fmt.Sprintf("llamafarm-%s", serviceName)

	// Check if container exists
	if !containerExists(containerName) {
		return fmt.Errorf("container %s does not exist", containerName)
	}

	// Check if container is already stopped
	if !isContainerRunning(containerName) {
		return nil // Already stopped
	}

	// Stop the container
	return d.stopDockerContainer(serviceName)
}

// startDockerContainer starts a specific Docker container
func (d *DockerServiceManager) startDockerContainer(serviceName string) error {
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

// stopDockerContainer stops a specific Docker container
func (d *DockerServiceManager) stopDockerContainer(serviceName string) error {
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

// startServicesDocker starts multiple services using Docker
func startServicesDocker(serviceNames []string, serverURL string) {
	OutputProgress("Starting services with Docker orchestration...\n")

	manager := NewDockerServiceManager(serverURL)

	// Check if Docker is available
	if err := manager.IsAvailable(); err != nil {
		OutputError("Docker is not available: %v\n", err)
		fmt.Fprintf(os.Stderr, "\nPlease ensure Docker is installed and running.\n")
		fmt.Fprintf(os.Stderr, "Visit https://docs.docker.com/get-docker/ for installation instructions.\n")
		os.Exit(1)
	}

	// Start each service
	for _, serviceName := range serviceNames {
		OutputProgress("Starting %s...\n", serviceName)

		containerName := fmt.Sprintf("llamafarm-%s", serviceName)

		// Check if already running
		if containerExists(containerName) && isContainerRunning(containerName) {
			OutputProgress("%s is already running\n", serviceName)
			continue
		}

		// Start the service
		if err := manager.StartService(serviceName); err != nil {
			OutputError("Failed to start %s: %v\n", serviceName, err)
			continue
		}

		OutputSuccess("%s started successfully\n", serviceName)
	}
}

// stopServicesDocker stops multiple services using Docker
func stopServicesDocker(serviceNames []string, serverURL string) {
	OutputProgress("Stopping services with Docker orchestration...\n")

	manager := NewDockerServiceManager(serverURL)

	// Check if Docker is available
	if err := manager.IsAvailable(); err != nil {
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

		// Check if already stopped
		if !isContainerRunning(containerName) {
			OutputProgress("%s is already stopped\n", serviceName)
			continue
		}

		// Stop the service
		if err := manager.StopService(serviceName); err != nil {
			OutputError("Failed to stop %s: %v\n", serviceName, err)
			continue
		}

		OutputSuccess("%s stopped successfully\n", serviceName)
	}
}
