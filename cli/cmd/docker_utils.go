package cmd

import (
	"errors"
	"os"
	"os/exec"
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
