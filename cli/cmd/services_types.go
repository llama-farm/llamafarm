package cmd

// ServiceInfo represents the status of a single service
type ServiceInfo struct {
	Name          string            `json:"name"`
	ContainerName string            `json:"container_name,omitempty"`
	State         string            `json:"state"` // "running", "stopped", "not_found"
	ContainerID   string            `json:"container_id,omitempty"`
	PID           int               `json:"pid,omitempty"`
	Image         string            `json:"image,omitempty"`
	Ports         map[string]string `json:"ports,omitempty"`
	Health        *Component        `json:"health,omitempty"`
	Uptime        string            `json:"uptime,omitempty"`
	LogFile       string            `json:"log_file,omitempty"`
	Orchestration string            `json:"orchestration"` // "docker" or "native"
}

// ServicesStatusOutput represents the complete status output
type ServicesStatusOutput struct {
	Services      []ServiceInfo `json:"services"`
	DockerRunning bool          `json:"docker_running,omitempty"`
	Orchestration string        `json:"orchestration"` // "docker" or "native"
	Timestamp     int64         `json:"timestamp"`
}
