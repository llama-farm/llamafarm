package cmd

import "github.com/llamafarm/cli/cmd/orchestrator"

// ServiceInfo represents the status of a single service
type ServiceInfo struct {
	Name    string                        `json:"name"`
	State   string                        `json:"state"` // "running", "stopped", "not_found"
	PID     int                           `json:"pid,omitempty"`
	Ports   map[string]string             `json:"ports,omitempty"`
	Health  *orchestrator.ComponentHealth `json:"health,omitempty"`
	Uptime  string                        `json:"uptime,omitempty"`
	LogFile string                        `json:"log_file,omitempty"`
}

// ServicesStatusOutput represents the complete status output
type ServicesStatusOutput struct {
	Services      []ServiceInfo `json:"services"`
	DockerRunning bool          `json:"docker_running,omitempty"`
	Orchestration string        `json:"orchestration"` // "docker" or "native"
	Timestamp     int64         `json:"timestamp"`
}
