package orchestrator

import (
	"fmt"
	"net"
	"net/url"
	"time"
)

// isPortInUse checks if a TCP port is in use on localhost
func isPortInUse(port string) bool {
	timeout := 500 * time.Millisecond
	conn, err := net.DialTimeout("tcp", "localhost:"+port, timeout)
	if err != nil {
		return false
	}
	conn.Close()
	return true
}

// findAvailablePort finds an available port starting from the given port
func findAvailablePort(startPort int, maxAttempts int) (int, error) {
	for i := 0; i < maxAttempts; i++ {
		port := startPort + i
		portStr := fmt.Sprintf("%d", port)
		if !isPortInUse(portStr) {
			return port, nil
		}
	}
	return 0, fmt.Errorf("no available port found in range %d-%d", startPort, startPort+maxAttempts-1)
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
