package cmd

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"strings"
	"time"
)

// ensureServerAvailable verifies the server at serverURL is reachable.
// If not reachable and the host is localhost, it attempts to start the
// server via Docker, then waits for readiness. Returns an error if it
// ultimately cannot ensure availability.
func ensureServerAvailable(serverURL string) error {
	if serverURL == "" {
		serverURL = "http://localhost:8000"
	}

	if err := checkServerHealth(serverURL); err == nil {
		return nil
	}

	// Only attempt auto-start when pointing to localhost
	if !isLocalhost(serverURL) {
		return fmt.Errorf("server %s is not reachable", serverURL)
	}

	if err := startLocalServerViaDocker(serverURL); err != nil {
		return err
	}

	// Poll for readiness
	timeout := serverStartTimeout
	if timeout <= 0 {
		timeout = 45 * time.Second
	}
	deadline := time.Now().Add(timeout)
	for {
		if err := checkServerHealth(serverURL); err == nil {
			return nil
		}
		if time.Now().After(deadline) {
			break
		}
		time.Sleep(1 * time.Second)
	}
	return fmt.Errorf("server did not become ready at %s within timeout", serverURL)
}

// checkServerHealth pings the /info endpoint with a short timeout.
func checkServerHealth(serverURL string) error {
	base := strings.TrimRight(serverURL, "/")
	infoURL := base + "/info"

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, infoURL, nil)
	if err != nil {
		return err
	}

	resp, err := (&http.Client{Timeout: 2 * time.Second}).Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, resp.Body)
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return nil
	}
	return fmt.Errorf("unexpected health status %d", resp.StatusCode)
}

func isLocalhost(serverURL string) bool {
	u, err := url.Parse(serverURL)
	if err != nil {
		return false
	}
	host := strings.ToLower(u.Hostname())
	return host == "localhost" || host == "127.0.0.1" || host == "::1"
}

// startLocalServerViaDocker pulls and runs the LlamaFarm server container if needed.
// It uses a fixed container name and maps the serverURL port to container port 8000.
func startLocalServerViaDocker(serverURL string) error {
	// Ensure Docker is available
	if err := ensureDockerAvailable(); err != nil {
		return err
	}

	port := resolvePort(serverURL, 8000)
	containerName := "llamafarm-server"
	image := "ghcr.io/llama-farm/llamafarm/server:latest"

	// If a container with this name exists and is running, nothing to do
	if isContainerRunning(containerName) {
		return nil
	}

	fmt.Fprintln(os.Stderr, "Starting local LlamaFarm server via Docker...")

	// Try to start existing stopped container first
	if containerExists(containerName) {
		startCmd := exec.Command("docker", "start", containerName)
		startCmd.Stdout = os.Stdout
		startCmd.Stderr = os.Stderr
		if err := startCmd.Run(); err == nil {
			return nil
		}
	}

	// Pull latest image (best effort)
	_ = pullImage(image)

	// Run new container
	runArgs := []string{
		"run",
		"-d",
		"--name", containerName,
		"-p", fmt.Sprintf("%d:8000", port),
	}

	// Mount effective working directory into the container at the same path
	if cwd := getEffectiveCWD(); strings.TrimSpace(cwd) != "" {
		runArgs = append(runArgs, "-v", fmt.Sprintf("%s:%s", cwd, cwd))
	} else {
		fmt.Fprintln(os.Stderr, "Warning: could not determine current directory; continuing without volume mount")
	}

	// Pass through or configure Ollama access inside the container
	if v, ok := os.LookupEnv("OLLAMA_HOST"); ok && strings.TrimSpace(v) != "" {
		runArgs = append(runArgs, "-e", fmt.Sprintf("OLLAMA_HOST=%s", v))
	} else if isHostOllamaAvailable() {
		// Ensure the container can resolve host.docker.internal on Linux
		runArgs = append(runArgs, "--add-host", "host.docker.internal:host-gateway")
		runArgs = append(runArgs, "-e", "OLLAMA_HOST=http://host.docker.internal:11434/v1")
	}
	// Also pass through OLLAMA_HOST/OLLAMA_PORT if explicitly set by the user
	if v, ok := os.LookupEnv("OLLAMA_HOST"); ok && strings.TrimSpace(v) != "" {
		runArgs = append(runArgs, "-e", fmt.Sprintf("OLLAMA_HOST=%s", v))
	}
	if v, ok := os.LookupEnv("OLLAMA_PORT"); ok && strings.TrimSpace(v) != "" {
		runArgs = append(runArgs, "-e", fmt.Sprintf("OLLAMA_PORT=%s", v))
	}

	// Image last
	runArgs = append(runArgs, image)
	runCmd := exec.Command("docker", runArgs...)
	runOut, err := runCmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to start docker container: %v\n%s", err, string(runOut))
	}
	return nil
}

func resolvePort(serverURL string, defaultPort int) int {
	u, err := url.Parse(serverURL)
	if err != nil {
		return defaultPort
	}
	if p := u.Port(); p != "" {
		if portNum, err := net.LookupPort("tcp", p); err == nil {
			return portNum
		}
	}
	// If URL scheme implies a default port, prefer it
	if u.Scheme == "https" {
		return 443
	}
	if u.Scheme == "http" {
		return 80
	}
	return defaultPort
}

// isHostOllamaAvailable returns true if an Ollama server on the host appears reachable
// at the default localhost port 11434 or via OLLAMA_HOST.
func isHostOllamaAvailable() bool {
	// Respect explicit OLLAMA_HOST
	if baseURL, ok := os.LookupEnv("OLLAMA_HOST"); ok && strings.TrimSpace(baseURL) != "" {
		return pingURL(strings.TrimSpace(baseURL)) == nil
	}
	// Try default local port
	return pingURL("http://localhost:11434") == nil
}

func pingURL(base string) error {
	u := strings.TrimRight(base, "/") + "/api/tags"
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return err
	}
	resp, err := (&http.Client{Timeout: 2 * time.Second}).Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, resp.Body)
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return nil
	}
	return fmt.Errorf("status %d", resp.StatusCode)
}
