package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"regexp"
	"strings"
	"syscall"
	"time"
)

var (
	// inferenceReadyPrinted ensures we only print the final ready + models once
	inferenceReadyPrinted bool
)

// ensureServerAvailable verifies the server at serverURL is reachable.
// If not reachable and the host is localhost, it attempts to start the
// server via Docker, then waits for readiness. Returns an error if it
// ultimately cannot ensure availability.
func ensureInferenceRuntimeAvailable() error {
	if err := checkInferenceRuntimeHealth(); err == nil {
		return nil
	}

	if err := startLocalInferenceRuntimeViaDocker(); err != nil {
		return err
	}

	// Poll for readiness
	timeout := 5 * time.Minute
	if v := os.Getenv("LF_INFERENCE_START_TIMEOUT"); strings.TrimSpace(v) != "" {
		if d, err := time.ParseDuration(v); err == nil {
			timeout = d
		}
	}
	deadline := time.Now().Add(timeout)
	for {
		if err := checkInferenceRuntimeHealth(); err == nil {
			return nil
		}
		if time.Now().After(deadline) {
			break
		}
		time.Sleep(1 * time.Second)
	}
	return fmt.Errorf("inference runtime did not become ready at within timeout")
}

// checkServerHealth pings using a fast TCP dial to the host:port.
func checkInferenceRuntimeHealth() error { return checkInferenceRuntimeHealthCore(true) }

// Quiet variant used while streaming logs
func checkInferenceRuntimeHealthQuiet() error { return checkInferenceRuntimeHealthCore(false) }

// checkInferenceRuntimeHealthCore performs the HTTP check; when noisy is true it prints
// a one-time ready banner and the available models list.
func checkInferenceRuntimeHealthCore(noisy bool) error {
	port := _port()
	baseURL := fmt.Sprintf("http://localhost:%s", port)
	infoURL := strings.TrimRight(baseURL, "/") + "/api/tags"

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

	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		if noisy && !inferenceReadyPrinted {
			inferenceReadyPrinted = true
			fmt.Fprintln(os.Stderr, "✅ Inference runtime is ready")
			if body, err := io.ReadAll(resp.Body); err == nil && len(body) > 0 {
				var parsed struct {
					Models []struct {
						Name string `json:"name"`
					} `json:"models"`
				}
				if json.Unmarshal(body, &parsed) == nil && len(parsed.Models) > 0 {
					fmt.Fprintln(os.Stderr, "🧠 Available models:")
					for _, m := range parsed.Models {
						fmt.Fprintf(os.Stderr, " - %s\n", m.Name)
					}
				}
			}
		} else {
			// Drain body quietly
			io.Copy(io.Discard, resp.Body)
		}
		return nil
	}
	// Drain body for non-2xx as well
	io.Copy(io.Discard, resp.Body)
	return fmt.Errorf("unexpected health status for inference runtime %d", resp.StatusCode)
}

func startLocalInferenceRuntimeViaDocker() error {
	// Ensure Docker is available
	if err := ensureDockerAvailable(); err != nil {
		return err
	}

	port := _port()

	// Determine model to serve
	model := _model()

	// Make the model name safe for a docker container name: only [a-zA-Z0-9][a-zA-Z0-9_.-] allowed
	safeModel := model
	// Replace any character not in [a-zA-Z0-9_.-] with '_'
	safeModel = regexp.MustCompile(`[^a-zA-Z0-9_.-]`).ReplaceAllString(safeModel, "_")
	containerName := fmt.Sprintf("lf-ramalama-serve-%s", safeModel)
	image := "quay.io/ramalama/ramalama:latest"

	// If a container with this name exists and is running, attach logs until healthy
	if isContainerRunning(containerName) {
		fmt.Fprintln(os.Stderr, "⏳ Waiting for inference runtime to become healthy...")
		streamLogsUntilHealthy(containerName)
		return nil
	}

	fmt.Fprintln(os.Stderr, "🚀 Starting inference runtime via Docker...")

	// Try to start existing stopped container first
	if containerExists(containerName) {
		fmt.Fprintln(os.Stderr, "▶️  Starting existing container...")
		startCmd := exec.Command("docker", "start", containerName)
		startCmd.Stdout = os.Stdout
		startCmd.Stderr = os.Stderr
		if err := startCmd.Run(); err == nil {
			// Stream logs until healthy, then return
			streamLogsUntilHealthy(containerName)
			return nil
		}
	}

	// Pull latest image (best effort)
	_ = pullImage(image)

	// Determine a persistent cache directory on host
	hostCacheDir := strings.TrimSpace(os.Getenv("RAMALAMA_STORE"))
	if hostCacheDir == "" {
		if home, err := os.UserHomeDir(); err == nil && strings.TrimSpace(home) != "" {
			hostCacheDir = filepath.Join(home, ".local", "share", "ramalama")
		} else {
			hostCacheDir = "/var/lib/ramalama"
		}
	}
	// Ensure directory exists (best effort)
	_ = os.MkdirAll(hostCacheDir, 0o755)

	// Run new container: mount docker socket and expose port 8081
	runArgs := []string{
		"run",
		"-d",
		"-t",
		"--name", containerName,
		"-v", "/var/run/docker.sock:/var/run/docker.sock",
		"-v", fmt.Sprintf("%s:%s", hostCacheDir, "/var/lib/ramalama"),
		"-p", fmt.Sprintf("%s:%s", port, port),
		image,
		"ramalama", "serve", "--webui", "off", "-p", port, model,
	}

	logDebug("inference runtime docker command: " + strings.Join(runArgs, " "))

	runCmd := exec.Command("docker", runArgs...)
	runOut, err := runCmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to start ramalama container: %v\n%s", err, string(runOut))
	}
	// After creating the container, stream logs until healthy, then return
	streamLogsUntilHealthy(containerName)
	return nil
}

// streamLogsUntilHealthy tails the container logs until the inference runtime passes
// the health check, then stops tailing. This function blocks until either the
// runtime becomes healthy or the default/startup timeout elapses.
func streamLogsUntilHealthy(containerName string) {
	// Prefer attaching to the container TTY to preserve CR-based progress
	attach := exec.Command("docker", "attach", "--no-stdin", "--sig-proxy=false", containerName)
	stdout, _ := attach.StdoutPipe()
	stderr, _ := attach.StderrPipe()
	var streamer *exec.Cmd
	if err := attach.Start(); err == nil {
		streamer = attach
	} else {
		// Fallback to docker logs if attach is not available
		cmd := exec.Command("docker", "logs", "--tail", "10", "-f", containerName)
		stdout, _ = cmd.StdoutPipe()
		stderr, _ = cmd.StderrPipe()
		if err := cmd.Start(); err == nil {
			streamer = cmd
		} else {
			fmt.Fprintln(os.Stderr, "⚠️  Could not stream container output:", err)
			waitUntilHealthy()
			return
		}
	}

	// Handle SIGINT/SIGTERM to allow Ctrl+C to abort cleanly
	stopCh := make(chan struct{})
	printedLines := 0
	cleanup := func() {
		close(stopCh)
		if streamer != nil && streamer.Process != nil {
			_ = streamer.Process.Kill()
			_ = streamer.Wait()
		}
		// Clear current (possibly CR-updating) line
		fmt.Fprint(os.Stderr, "\r\x1b[2K")
		// Collapse previously printed newline lines
		for i := 0; i < printedLines; i++ {
			fmt.Fprint(os.Stderr, "\x1b[1A\x1b[2K")
		}
	}

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sigCh
		cleanup()
		os.Exit(130)
	}()

	// Render function that handles both CR (in-place) and NL (new line)
	const gray = "\x1b[90m"
	const reset = "\x1b[0m"
	copyStream := func(r io.Reader) {
		buf := make([]byte, 0, 8192)
		line := make([]byte, 0, 8192)
		for {
			select {
			case <-stopCh:
				return
			default:
				n, err := r.Read(buf[:cap(buf)])
				if n > 0 {
					chunk := buf[:n]
					for len(chunk) > 0 {
						i := 0
						for i < len(chunk) && chunk[i] != '\n' && chunk[i] != '\r' {
							i++
						}
						line = append(line, chunk[:i]...)
						if i < len(chunk) {
							// boundary at chunk[i]
							if chunk[i] == '\n' {
								// real newline: commit the line
								fmt.Fprintf(os.Stderr, "%s%s%s\n", gray, string(line), reset)
								printedLines++
							} else {
								// carriage return: update in place
								fmt.Fprint(os.Stderr, "\r\x1b[2K")
								fmt.Fprintf(os.Stderr, "%s%s%s", gray, string(line), reset)
							}
							line = line[:0]
							i++
						}
						chunk = chunk[i:]
					}
				}
				if err != nil {
					if err == io.EOF {
						return
					}
					time.Sleep(50 * time.Millisecond)
				}
			}
		}
	}

	go copyStream(stdout)
	go copyStream(stderr)

	// Wait until healthy or timeout (quiet)
	waitUntilHealthy()

	// Normal path
	cleanup()
}

func waitUntilHealthy() {
	// Respect the same startup timeout as ensureInferenceRuntimeAvailable
	timeout := 5 * time.Minute
	if v := os.Getenv("LF_INFERENCE_START_TIMEOUT"); strings.TrimSpace(v) != "" {
		if d, err := time.ParseDuration(v); err == nil {
			timeout = d
		}
	}
	deadline := time.Now().Add(timeout)
	for {
		if err := checkInferenceRuntimeHealthQuiet(); err == nil {
			return
		}
		if time.Now().After(deadline) {
			fmt.Fprintln(os.Stderr, "⚠️  Timed out waiting for inference runtime to become healthy")
			return
		}
		time.Sleep(1 * time.Second)
	}
}

func _port() string {
	port := strings.TrimSpace(os.Getenv("LF_INFERENCE_PORT"))
	if port == "" {
		port = "11434"
	}
	return port
}

func _model() string {
	model := strings.TrimSpace(os.Getenv("LF_INFERENCE_MODEL"))
	if model == "" {
		model = "qwen3:8b"
	}
	return model
}
