package cmd

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strings"
)

// ensureDockerAvailable checks whether docker is available on PATH
func ensureDockerAvailable() error {
	if err := exec.Command("docker", "--version").Run(); err != nil {
		return errors.New("docker is not available. Please install Docker and try again")
	}
	return nil
}

// pullImage pulls a docker image, streaming output to the current stdio
func pullImage(image string) error {
	pullCmd := exec.Command("docker", "pull", image)
	pullCmd.Stdout = os.Stdout
	pullCmd.Stderr = os.Stderr
	return pullCmd.Run()
}

func containerExists(name string) bool {
	cmd := exec.Command("docker", "ps", "-a", "--format", "{{.Names}}")
	out, err := cmd.Output()
	if err != nil {
		return false
	}
	for _, line := range strings.Split(string(out), "\n") {
		if strings.TrimSpace(line) == name {
			return true
		}
	}
	return false
}

func isContainerRunning(name string) bool {
	cmd := exec.Command("docker", "ps", "--format", "{{.Names}}")
	out, err := cmd.Output()
	if err != nil {
		return false
	}
	for _, line := range strings.Split(string(out), "\n") {
		if strings.TrimSpace(line) == name {
			return true
		}
	}
	return false
}

// resolveImageTag determines the appropriate Docker image tag based on version and environment variables
func resolveImageTag(component string, defaultTag string) string {
	// Check for component-specific environment variable first
	componentEnvVar := fmt.Sprintf("LF_%s_IMAGE_TAG", strings.ToUpper(component))
	if tag := strings.TrimSpace(os.Getenv(componentEnvVar)); tag != "" {
		return tag
	}

	// Check for global override
	if tag := strings.TrimSpace(os.Getenv("LF_IMAGE_TAG")); tag != "" {
		return tag
	}

	// Use version-based logic
	version := strings.TrimSpace(Version)
	if version == "" {
		return defaultTag
	}

	// Handle version patterns: vX.X.X, vX.X.X-suffix, etc.
	versionPattern := regexp.MustCompile(`^v\d+\.\d+\.\d+.*`)
	if versionPattern.MatchString(version) {
		return version
	}

	// Handle dev versions
	if version == "dev" || version == "vdev" {
		return "latest"
	}

	// Fallback to default
	return defaultTag
}

// getImageURL constructs the full Docker image URL for a given component
func getImageURL(component string) string {
	baseURL := "ghcr.io/llama-farm/llamafarm"
	tag := resolveImageTag(component, "latest")
	return fmt.Sprintf("%s/%s:%s", baseURL, component, tag)
}
