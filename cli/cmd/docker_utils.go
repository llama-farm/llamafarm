package cmd

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strings"
)

// versionPattern matches semantic versions with or without leading "v"
// Examples: v1.0.0, v1.0.0-rc1, v2.0.0-beta.1+build.123, 1.0.0, 1.0.0-alpha
var versionPattern = regexp.MustCompile(`^v?\d+\.\d+\.\d+.*`)

// knownComponents lists the valid component names for image URLs
var knownComponents = map[string]bool{
	"server":   true,
	"designer": true,
	"rag":      true,
	"runtime":  true,
	"models":   true,
}

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

	// Handle version patterns: vX.X.X, X.X.X, with optional suffixes
	if versionPattern.MatchString(version) {
		// Ensure version has "v" prefix for Docker tag consistency
		if !strings.HasPrefix(version, "v") {
			return "v" + version
		}
		return version
	}

	// Handle dev versions
	if version == "dev" {
		return "latest"
	}

	// Fallback to default
	return defaultTag
}

// getImageURL constructs the full Docker image URL for a given component
func getImageURL(component string) (string, error) {
	if !knownComponents[component] {
		return "", fmt.Errorf("unknown component '%s'; valid components are: %s",
			component, getKnownComponentsList())
	}

	baseURL := "ghcr.io/llama-farm/llamafarm"
	tag := resolveImageTag(component, "latest")
	return fmt.Sprintf("%s/%s:%s", baseURL, component, tag), nil
}

// getKnownComponentsList returns a comma-separated list of known components
func getKnownComponentsList() string {
	components := make([]string, 0, len(knownComponents))
	for component := range knownComponents {
		components = append(components, component)
	}
	return strings.Join(components, ", ")
}
