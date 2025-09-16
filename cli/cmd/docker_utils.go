package cmd

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"
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

// pullImage pulls a docker image, capturing output to avoid breaking TUIs
func pullImage(image string) error {
	pullCmd := exec.Command("docker", "pull", image)
	out, err := pullCmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("docker pull failed: %v\n%s", err, string(out))
	}
	if debug && len(out) > 0 {
		logDebug(fmt.Sprintf("docker pull output: %s", string(out)))
	}
	return nil
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

// ---- Generic container utilities ----

type PortSpec struct {
	Container int
	Protocol  string
}

type PortMapping struct {
	Host      int
	Container int
	Protocol  string
}

type ContainerRunSpec struct {
	Name           string
	Image          string
	DynamicPublish bool
	StaticPorts    []PortMapping
	Env            map[string]string
	Volumes        []string
	AddHosts       []string
	Labels         map[string]string
	Workdir        string
	Entrypoint     []string
	Cmd            []string
}

type PortResolutionPolicy struct {
	PreferredHostPort int
	Forced            bool
}

func removeContainer(name string) error {
	if !containerExists(name) {
		return nil
	}
	rmCmd := exec.Command("docker", "rm", "-f", name)
	out, err := rmCmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("docker rm failed: %v\n%s", err, string(out))
	}
	if debug && len(out) > 0 {
		logDebug(fmt.Sprintf("docker rm output: %s", string(out)))
	}
	return nil
}

func isHostPortAvailable(port int) bool {
	l, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return false
	}
	_ = l.Close()
	return true
}

// StartContainerDetachedWithPolicy starts a container with either static port mapping
// or dynamic published ports based on availability and the provided policy.
// Returns a map of containerPort->hostPort that were published.
func StartContainerDetachedWithPolicy(spec ContainerRunSpec, policy *PortResolutionPolicy) (map[int]int, error) {
	if err := ensureDockerAvailable(); err != nil {
		return nil, err
	}
	if strings.TrimSpace(spec.Name) == "" || strings.TrimSpace(spec.Image) == "" {
		return nil, errors.New("container name and image are required")
	}

	// Remove stale container if exists and not running
	if containerExists(spec.Name) && !isContainerRunning(spec.Name) {
		fmt.Fprintln(os.Stderr, "Removing existing container to refresh image/args...")
		if err := removeContainer(spec.Name); err != nil {
			return nil, fmt.Errorf("failed to remove existing container %s: %w", spec.Name, err)
		}
	}

	// If already running, do nothing and return current published ports
	if isContainerRunning(spec.Name) {
		ports, _ := GetPublishedPorts(spec.Name)
		resolved := make(map[int]int)
		for key, val := range ports {
			// key like "80/tcp"; extract container port
			parts := strings.Split(key, "/")
			if len(parts) > 0 {
				if cp, err := strconv.Atoi(parts[0]); err == nil {
					if hp, err2 := strconv.Atoi(val); err2 == nil {
						resolved[cp] = hp
					}
				}
			}
		}
		return resolved, nil
	}

	// Pull image best-effort (captured)
	_ = pullImage(spec.Image)

	runArgs := []string{"run", "-d", "--name", spec.Name}

	useDynamic := false
	if policy != nil && policy.PreferredHostPort > 0 && len(spec.StaticPorts) > 0 {
		if isHostPortAvailable(policy.PreferredHostPort) {
			for _, pm := range spec.StaticPorts {
				hostPort := policy.PreferredHostPort
				if pm.Host > 0 {
					hostPort = pm.Host
				}
				protocol := pm.Protocol
				if protocol == "" {
					protocol = "tcp"
				}
				runArgs = append(runArgs, "-p", fmt.Sprintf("%d:%d/%s", hostPort, pm.Container, protocol))
			}
		} else {
			if policy.Forced {
				return nil, fmt.Errorf("port %d is already in use", policy.PreferredHostPort)
			}
			useDynamic = true
		}
	} else {
		useDynamic = true
	}

	if useDynamic {
		runArgs = append(runArgs, "-P")
	}

	for k, v := range spec.Env {
		runArgs = append(runArgs, "-e", fmt.Sprintf("%s=%s", k, v))
	}
	for _, v := range spec.Volumes {
		runArgs = append(runArgs, "-v", v)
	}
	for _, h := range spec.AddHosts {
		runArgs = append(runArgs, "--add-host", h)
	}
	for k, v := range spec.Labels {
		runArgs = append(runArgs, "--label", fmt.Sprintf("%s=%s", k, v))
	}
	if strings.TrimSpace(spec.Workdir) != "" {
		runArgs = append(runArgs, "-w", spec.Workdir)
	}
	if len(spec.Entrypoint) > 0 {
		runArgs = append(runArgs, "--entrypoint", strings.Join(spec.Entrypoint, " "))
	}

	runArgs = append(runArgs, spec.Image)
	runArgs = append(runArgs, spec.Cmd...)

	runCmd := exec.Command("docker", runArgs...)
	runOut, err := runCmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("failed to start docker container: %v\n%s", err, string(runOut))
	}
	if debug && len(runOut) > 0 {
		logDebug(fmt.Sprintf("docker run output: %s", string(runOut)))
	}

	// Resolve published ports
	published, err := GetPublishedPorts(spec.Name)
	if err != nil {
		return nil, err
	}
	resolved := make(map[int]int)
	for key, val := range published {
		parts := strings.Split(key, "/")
		if len(parts) > 0 {
			if cp, err := strconv.Atoi(parts[0]); err == nil {
				if hp, err2 := strconv.Atoi(val); err2 == nil {
					resolved[cp] = hp
				}
			}
		}
	}
	return resolved, nil
}

// GetPublishedPorts returns a map like "80/tcp" -> "49154"
func GetPublishedPorts(name string) (map[string]string, error) {
	cmd := exec.Command("docker", "port", name)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("docker port failed: %v\n%s", err, string(out))
	}
	res := make(map[string]string)
	s := bufio.NewScanner(strings.NewReader(string(out)))
	for s.Scan() {
		line := strings.TrimSpace(s.Text())
		// Example: "80/tcp -> 0.0.0.0:49154" or "80/tcp -> :::49154"
		parts := strings.Split(line, " -> ")
		if len(parts) != 2 {
			continue
		}
		key := strings.TrimSpace(parts[0])
		host := strings.TrimSpace(parts[1])
		idx := strings.LastIndex(host, ":")
		if idx > -1 && idx+1 < len(host) {
			res[key] = host[idx+1:]
		}
	}
	if debug && len(out) > 0 {
		logDebug(fmt.Sprintf("docker port output: %s", string(out)))
	}
	return res, nil
}

func WaitForReadiness(ctx context.Context, check func() error, interval time.Duration) error {
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-t.C:
			if err := check(); err == nil {
				return nil
			}
		}
	}
}

func HTTPGetReady(url string) func() error {
	return func() error {
		req, err := http.NewRequest(http.MethodGet, url, nil)
		if err != nil {
			return err
		}
		client := &http.Client{Timeout: 1500 * time.Millisecond}
		resp, err := client.Do(req)
		if err != nil {
			return err
		}
		defer resp.Body.Close()
		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			return nil
		}
		return fmt.Errorf("status %d", resp.StatusCode)
	}
}
