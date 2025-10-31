package cmd

import (
	"context"
	"fmt"
	"strconv"
	"time"
)

type DesignerLaunchOptions struct {
	PreferredPort int
	Forced        bool
}

func StartDesignerInBackground(ctx context.Context, opts DesignerLaunchOptions) (string, error) {
	// Check orchestration mode - if native, designer is served by server
	orchestrationMode := determineOrchestrationMode()
	if orchestrationMode == OrchestrationNative {
		// Designer is served by server at root URL
		serverURLToUse := serverURL
		if serverURLToUse == "" {
			serverURLToUse = "http://localhost:8000"
		}
		// Return server URL (designer is at root)
		return serverURLToUse, nil
	}

	// Docker mode - use existing Docker container logic
	if err := ensureDockerAvailable(); err != nil {
		return "", err
	}

	image, err := getImageURL("designer")
	if err != nil {
		return "", err
	}

	preferred := opts.PreferredPort
	if preferred <= 0 {
		preferred = 7724
	}

	spec := ContainerRunSpec{
		Name:           "llamafarm-designer",
		Image:          image,
		DynamicPublish: false,
		StaticPorts:    []PortMapping{{Host: preferred, Container: 80, Protocol: "tcp"}},
		Env:            map[string]string{},
		Volumes:        nil,
		AddHosts:       nil,
		Labels:         map[string]string{"component": "designer"},
	}

	policy := &PortResolutionPolicy{PreferredHostPort: preferred, Forced: opts.Forced}
	resolved, err := StartContainerDetachedWithPolicy(spec, policy)
	if err != nil {
		return "", err
	}

	hostPort, ok := resolved[80]
	if !ok {
		// Fallback: query docker port and parse 80/tcp
		ports, perr := GetPublishedPorts(spec.Name)
		if perr != nil {
			return "", fmt.Errorf("designer started but could not resolve published port: %v", perr)
		}
		if p, ok2 := ports["80/tcp"]; ok2 {
			if hp, convErr := strconv.Atoi(p); convErr == nil {
				hostPort = hp
			}
		}
	}
	if hostPort <= 0 {
		return "", fmt.Errorf("could not determine designer host port")
	}

	url := fmt.Sprintf("http://localhost:%d", hostPort)

	// Wait briefly for readiness
	readyCtx, cancel := context.WithTimeout(ctx, 45*time.Second)
	defer cancel()
	check := HTTPGetReady(url)
	if err := WaitForReadiness(readyCtx, check, 1*time.Second); err != nil {
		logDebug(fmt.Sprintf("designer readiness wait timed out or failed: %v", err))
		// Return the URL anyway so the UI can still attempt to open it
	}
	return url, nil
}
