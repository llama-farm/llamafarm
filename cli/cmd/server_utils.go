package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

var containerName = "llamafarm-server"

// Service Orchestration Types

type ServiceRequirement int

const (
	ServiceIgnored  ServiceRequirement = iota // Don't start, don't check
	ServiceOptional                           // Start async, don't wait, don't fail if unhealthy
	ServiceRequired                           // Start and wait, fail if can't become healthy
)

type ServiceStatus int

const (
	StatusUnknown ServiceStatus = iota
	StatusStarting
	StatusHealthy
	StatusDegraded
	StatusUnhealthy
	StatusFailed
)

type ServiceState struct {
	Name    string
	Status  ServiceStatus
	Message string
	Error   error
	Health  *Component // From health payload
}

type OrchestrationMode int

const (
	OrchestrationAuto   OrchestrationMode = iota // Auto-detect (prefer native)
	OrchestrationDocker                          // Force Docker mode
	OrchestrationNative                          // Force Native mode
)

type ServiceOrchestrationConfig struct {
	ServerURL         string
	PrintStatus       bool
	ServiceNeeds      map[string]ServiceRequirement
	DefaultTimeout    time.Duration
	ServiceTimeouts   map[string]time.Duration
	OrchestrationMode OrchestrationMode
}

func (config *ServiceOrchestrationConfig) isLocalhost() bool {
	return isLocalhost(config.ServerURL)
}

type OrchestrationResult struct {
	ServerHealth *HealthPayload
	Services     map[string]*ServiceState

	// Channels for async monitoring
	ServerReady chan *ServiceState
	RAGReady    chan *ServiceState
	Done        chan struct{}
}

type ServiceDefinition struct {
	Name            string
	Dependencies    []string
	CanStartLocally bool
	DefaultTimeout  time.Duration

	// Service-specific functions
	CheckHealth func(serverURL string) (*Component, error)
	StartLocal  func(serverURL string) error
	WaitReady   func(serverURL string, timeout time.Duration) error
}

type ServiceOrchestrator struct {
	config       *ServiceOrchestrationConfig
	serviceGraph map[string]*ServiceDefinition
	results      map[string]*ServiceState

	// Channels for coordination
	serviceReady map[string]chan *ServiceState
	done         chan struct{}
	mu           sync.RWMutex
}

type Component struct {
	Name      string                 `json:"name"`
	Status    string                 `json:"status"`
	Message   string                 `json:"message"`
	LatencyMs int                    `json:"latency_ms"`
	Details   map[string]interface{} `json:"details,omitempty"`
	Runtime   map[string]interface{} `json:"runtime,omitempty"`
}
type HealthPayload struct {
	Status     string      `json:"status"`
	Summary    string      `json:"summary"`
	Components []Component `json:"components"`
	Seeds      []Component `json:"seeds"`
	Timestamp  int64       `json:"timestamp"`
}

// HealthError wraps a non-healthy /health response.
type HealthError struct {
	Status     string
	HealthResp HealthPayload
}

func (e *HealthError) Error() string {
	return fmt.Sprintf("server unhealthy: %s", e.Status)
}

// Service Graph Definition

var ServiceGraph = map[string]*ServiceDefinition{
	"universal-runtime": {
		Name:            "universal-runtime",
		Dependencies:    []string{}, // No dependencies - starts in parallel with server
		CanStartLocally: true,
		DefaultTimeout:  30 * time.Second,
		CheckHealth:     checkUniversalRuntimeHealthForService,
		StartLocal:      startUniversalRuntimeContainerForService,
		WaitReady:       waitForUniversalRuntimeReadyForService,
	},
	"server": {
		Name:            "server",
		Dependencies:    []string{}, // No dependencies - starts in parallel with universal runtime
		CanStartLocally: true,
		DefaultTimeout:  45 * time.Second,
		CheckHealth:     checkServerHealthForService,
		StartLocal:      startServerContainerForService,
		WaitReady:       waitForServerReadyForService,
	},
	"rag": {
		Name:            "rag",
		Dependencies:    []string{"server"}, // Depends on server
		CanStartLocally: true,
		DefaultTimeout:  30 * time.Second,
		CheckHealth:     checkRAGHealthForService,
		StartLocal:      startRAGContainerForService,
		WaitReady:       waitForRAGReadyForService,
	},
}

// Service-specific health check functions

func checkServerHealthForService(serverURL string) (*Component, error) {
	hr, err := checkServerHealth(serverURL)
	if err != nil {
		return nil, err
	}
	// Find server component in health response
	for _, comp := range hr.Components {
		if strings.Contains(strings.ToLower(comp.Name), "server") || comp.Name == "api" {
			return &comp, nil
		}
	}
	// If no specific server component, create one from overall health
	return &Component{
		Name:    "server",
		Status:  hr.Status,
		Message: hr.Summary,
	}, nil
}

func checkRAGHealthForService(serverURL string) (*Component, error) {
	hr, err := checkServerHealth(serverURL)
	if err != nil {
		return nil, err
	}
	ragComponent := findRAGComponent(hr)
	if ragComponent == nil {
		return nil, fmt.Errorf("RAG component not found in health response")
	}
	return ragComponent, nil
}

func checkUniversalRuntimeHealthForService(serverURL string) (*Component, error) {
	// Universal runtime runs on its own port (11540 by default)
	port := os.Getenv("TRANSFORMERS_PORT")
	if port == "" {
		port = "11540"
	}
	host := os.Getenv("TRANSFORMERS_HOST")
	if host == "" {
		host = "127.0.0.1"
	}

	runtimeURL := fmt.Sprintf("http://%s:%s", host, port)

	// Check health endpoint
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(runtimeURL + "/health")
	if err != nil {
		return nil, fmt.Errorf("universal runtime not reachable: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("universal runtime unhealthy: status %d", resp.StatusCode)
	}

	return &Component{
		Name:    "universal-runtime",
		Status:  "healthy",
		Message: fmt.Sprintf("Runtime available at %s", runtimeURL),
	}, nil
}

func startServerContainerForService(serverURL string) error {
	mode := determineOrchestrationMode()
	if mode == OrchestrationNative {
		return startLocalServerNative(serverURL)
	}
	return startLocalServerViaDocker(serverURL)
}

func startRAGContainerForService(serverURL string) error {
	mode := determineOrchestrationMode()
	if mode == OrchestrationNative {
		return startRAGNative(serverURL)
	}
	orchestrator := NewContainerOrchestrator()
	return orchestrator.startRAGContainer()
}

func startUniversalRuntimeContainerForService(serverURL string) error {
	mode := determineOrchestrationMode()
	if mode == OrchestrationNative {
		return startUniversalRuntimeNative(serverURL)
	}
	// Docker mode not yet implemented for universal runtime
	return fmt.Errorf("universal runtime Docker mode not yet implemented")
}

func waitForServerReadyForService(serverURL string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if _, err := checkServerHealth(serverURL); err == nil {
			return nil
		}
		time.Sleep(1 * time.Second)
	}
	return fmt.Errorf("server did not become ready within %v", timeout)
}

func waitForRAGReadyForService(serverURL string, timeout time.Duration) error {
	orchestrator := NewContainerOrchestrator()
	return orchestrator.waitForRAGReadiness(timeout, serverURL)
}

func waitForUniversalRuntimeReadyForService(serverURL string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if _, err := checkUniversalRuntimeHealthForService(serverURL); err == nil {
			return nil
		}
		time.Sleep(1 * time.Second)
	}
	return fmt.Errorf("universal runtime did not become ready within %v", timeout)
}

// ServiceOrchestrator Implementation

func NewServiceOrchestrator(config *ServiceOrchestrationConfig) *ServiceOrchestrator {
	return &ServiceOrchestrator{
		config:       config,
		serviceGraph: ServiceGraph,
		results:      make(map[string]*ServiceState),
		serviceReady: make(map[string]chan *ServiceState),
		done:         make(chan struct{}),
	}
}

func (so *ServiceOrchestrator) EnsureServices() *OrchestrationResult {
	// Initialize channels
	so.serviceReady["universal-runtime"] = make(chan *ServiceState, 1)
	so.serviceReady["server"] = make(chan *ServiceState, 1)
	so.serviceReady["rag"] = make(chan *ServiceState, 1)

	// Build execution plan based on dependencies
	executionPlan := so.buildExecutionPlan()

	// Execute services in dependency order
	for _, stage := range executionPlan {
		so.executeStage(stage)
	}

	// Build final result
	result := &OrchestrationResult{
		Services:    so.results,
		ServerReady: so.serviceReady["server"],
		RAGReady:    so.serviceReady["rag"],
		Done:        so.done,
	}

	// Set server health from server service state
	if serverState, exists := so.results["server"]; exists && serverState.Status == StatusHealthy {
		if hr, err := checkServerHealth(so.config.ServerURL); err == nil {
			result.ServerHealth = hr
		}
	}

	close(so.done)
	return result
}

func (so *ServiceOrchestrator) buildExecutionPlan() [][]string {
	visited := make(map[string]bool)
	stages := [][]string{}

	for len(visited) < len(so.config.ServiceNeeds) {
		currentStage := []string{}

		for serviceName, requirement := range so.config.ServiceNeeds {
			if requirement == ServiceIgnored || visited[serviceName] {
				continue
			}

			// Check if all dependencies are satisfied
			if so.dependenciesSatisfied(serviceName, visited) {
				currentStage = append(currentStage, serviceName)
			}
		}

		if len(currentStage) == 0 {
			break // No more services can be processed
		}

		stages = append(stages, currentStage)
		for _, service := range currentStage {
			visited[service] = true
		}
	}

	return stages
}

func (so *ServiceOrchestrator) dependenciesSatisfied(serviceName string, visited map[string]bool) bool {
	serviceDef, exists := so.serviceGraph[serviceName]
	if !exists {
		return false
	}

	for _, dep := range serviceDef.Dependencies {
		if !visited[dep] {
			return false
		}
	}
	return true
}

func (so *ServiceOrchestrator) executeStage(serviceNames []string) {
	var wg sync.WaitGroup

	for _, serviceName := range serviceNames {
		requirement := so.config.ServiceNeeds[serviceName]

		wg.Add(1)
		go func(name string, req ServiceRequirement) {
			defer wg.Done()

			state := so.ensureService(name, req)

			so.mu.Lock()
			so.results[name] = state
			so.mu.Unlock()

			// Notify waiters
			if ch, exists := so.serviceReady[name]; exists {
				select {
				case ch <- state:
				default: // Channel full, don't block
				}
			}
		}(serviceName, requirement)
	}

	// Wait for required services in this stage
	so.waitForRequiredServices(serviceNames)
}

func (so *ServiceOrchestrator) waitForRequiredServices(serviceNames []string) {
	for _, serviceName := range serviceNames {
		requirement := so.config.ServiceNeeds[serviceName]
		if requirement == ServiceRequired {
			// Wait for this service to complete
			<-so.serviceReady[serviceName]
		}
	}
}

func (so *ServiceOrchestrator) ensureService(serviceName string, requirement ServiceRequirement) *ServiceState {
	serviceDef, exists := so.serviceGraph[serviceName]
	if !exists {
		return &ServiceState{
			Name:   serviceName,
			Status: StatusFailed,
			Error:  fmt.Errorf("unknown service: %s", serviceName),
		}
	}

	// Step 1: Check if service is already healthy
	if component, err := serviceDef.CheckHealth(so.config.ServerURL); err == nil {
		if strings.EqualFold(component.Status, "healthy") {
			return &ServiceState{
				Name:    serviceName,
				Status:  StatusHealthy,
				Message: fmt.Sprintf("%s is already healthy", serviceName),
				Health:  component,
			}
		}
	}

	// Step 2: Service not healthy, need to start it
	if !so.config.isLocalhost() {
		// Remote server - can't start services, just wait/poll
		return so.waitForServiceRemote(serviceName, serviceDef, requirement)
	}

	// Step 3: Local server - start service
	if !serviceDef.CanStartLocally {
		return &ServiceState{
			Name:   serviceName,
			Status: StatusFailed,
			Error:  fmt.Errorf("service %s cannot be started locally", serviceName),
		}
	}

	// Start the service
	if so.config.PrintStatus {
		OutputProgress("Starting %s service...\n", serviceName)
	}

	if err := serviceDef.StartLocal(so.config.ServerURL); err != nil {
		if so.config.PrintStatus {
			OutputError("Failed to start %s: %v\n", serviceName, err)
		}
		return &ServiceState{
			Name:   serviceName,
			Status: StatusFailed,
			Error:  fmt.Errorf("failed to start %s: %v", serviceName, err),
		}
	}

	// Test progress message to verify the system is working
	if so.config.PrintStatus && serviceName == "rag" {
		OutputProgress("RAG services started, initializing...\n")
	}

	// Wait for service to be ready (if required)
	timeout := so.getServiceTimeout(serviceName, serviceDef)

	if requirement == ServiceRequired {
		if so.config.PrintStatus {
			OutputProgress("Waiting for %s to become ready...\n", serviceName)
		}

		if err := serviceDef.WaitReady(so.config.ServerURL, timeout); err != nil {
			return &ServiceState{
				Name:   serviceName,
				Status: StatusFailed,
				Error:  fmt.Errorf("%s did not become ready: %v", serviceName, err),
			}
		}

		if so.config.PrintStatus {
			OutputSuccess("%s is ready\n", serviceName)
		}

		return &ServiceState{
			Name:    serviceName,
			Status:  StatusHealthy,
			Message: fmt.Sprintf("%s started and ready", serviceName),
		}
	} else {
		// Optional service - start in background, don't wait
		go func() {
			if err := serviceDef.WaitReady(so.config.ServerURL, timeout); err != nil {
				if so.config.PrintStatus {
					OutputWarning("%s service may not be fully ready: %v\n", serviceName, err)
				}
			} else {
				if so.config.PrintStatus {
					OutputSuccess("%s service is ready\n", serviceName)
				}
			}
		}()

		return &ServiceState{
			Name:    serviceName,
			Status:  StatusStarting,
			Message: fmt.Sprintf("%s started in background", serviceName),
		}
	}
}

func (so *ServiceOrchestrator) waitForServiceRemote(serviceName string, serviceDef *ServiceDefinition, requirement ServiceRequirement) *ServiceState {
	timeout := so.getServiceTimeout(serviceName, serviceDef)
	deadline := time.Now().Add(timeout)

	if so.config.PrintStatus {
		OutputWarning("%s service is not healthy on remote server, waiting for it to come online...\n", serviceName)
	}

	for time.Now().Before(deadline) {
		if component, err := serviceDef.CheckHealth(so.config.ServerURL); err == nil {
			if strings.EqualFold(component.Status, "healthy") {
				if so.config.PrintStatus {
					OutputSuccess("%s service is now ready on remote server\n", serviceName)
				}
				return &ServiceState{
					Name:    serviceName,
					Status:  StatusHealthy,
					Message: fmt.Sprintf("%s became healthy on remote server", serviceName),
					Health:  component,
				}
			}
		}
		time.Sleep(2 * time.Second)
	}

	// Timeout reached
	if requirement == ServiceRequired {
		return &ServiceState{
			Name:   serviceName,
			Status: StatusFailed,
			Error:  fmt.Errorf("%s service did not become ready on remote server within %v", serviceName, timeout),
		}
	} else {
		// Optional service - proceed with degraded functionality
		if so.config.PrintStatus {
			OutputWarning("%s service did not become ready within %v, proceeding with degraded functionality\n", serviceName, timeout)
		}
		return &ServiceState{
			Name:    serviceName,
			Status:  StatusDegraded,
			Message: fmt.Sprintf("%s service not ready on remote server", serviceName),
		}
	}
}

func (so *ServiceOrchestrator) getServiceTimeout(serviceName string, serviceDef *ServiceDefinition) time.Duration {
	if timeout, exists := so.config.ServiceTimeouts[serviceName]; exists {
		return timeout
	}
	if so.config.DefaultTimeout > 0 {
		return so.config.DefaultTimeout
	}
	return serviceDef.DefaultTimeout
}

// Command-Specific Configuration Factories

// StartCommandConfig creates config for lf start - Server required, universal runtime and RAG optional (background)
func StartCommandConfig(serverURL string) *ServiceOrchestrationConfig {
	return &ServiceOrchestrationConfig{
		ServerURL:   serverURL,
		PrintStatus: true, // Show progress for lf start so users see what's happening
		ServiceNeeds: map[string]ServiceRequirement{
			"universal-runtime": ServiceOptional, // Start async, don't wait
			"server":            ServiceRequired,
			"rag":               ServiceOptional, // Start async, don't wait
		},
		DefaultTimeout:    45 * time.Second,
		OrchestrationMode: determineOrchestrationMode(),
	}
}

// RAGCommandConfig creates config for RAG commands - Server and RAG required, universal runtime optional
func RAGCommandConfig(serverURL string) *ServiceOrchestrationConfig {
	return &ServiceOrchestrationConfig{
		ServerURL:   serverURL,
		PrintStatus: true,
		ServiceNeeds: map[string]ServiceRequirement{
			"universal-runtime": ServiceOptional, // Start async, don't wait
			"server":            ServiceRequired,
			"rag":               ServiceRequired, // Wait for both server and RAG
		},
		DefaultTimeout: 45 * time.Second,
	}
}

// ChatNoRAGConfig creates config for lf chat --no-rag - Server required, universal runtime optional
func ChatNoRAGConfig(serverURL string) *ServiceOrchestrationConfig {
	return &ServiceOrchestrationConfig{
		ServerURL:   serverURL,
		PrintStatus: true,
		ServiceNeeds: map[string]ServiceRequirement{
			"universal-runtime": ServiceOptional, // Start async, don't wait
			"server":            ServiceRequired,
			// RAG not mentioned = ServiceIgnored
		},
		DefaultTimeout: 45 * time.Second,
	}
}

// ServerOnlyConfig creates config for server-only commands - Server required, universal runtime and RAG optional (background)
func ServerOnlyConfig(serverURL string) *ServiceOrchestrationConfig {
	return &ServiceOrchestrationConfig{
		ServerURL:   serverURL,
		PrintStatus: true,
		ServiceNeeds: map[string]ServiceRequirement{
			"universal-runtime": ServiceOptional, // Start async, don't wait
			"server":            ServiceRequired,
			"rag":               ServiceOptional, // Start async, don't wait
		},
		DefaultTimeout: 45 * time.Second,
	}
}

// New unified service orchestration function
func EnsureServicesWithConfig(config *ServiceOrchestrationConfig) *HealthPayload {
	orchestrator := NewServiceOrchestrator(config)
	result := orchestrator.EnsureServices()

	// Check if any required services failed
	for serviceName, requirement := range config.ServiceNeeds {
		if requirement == ServiceRequired {
			if state, exists := result.Services[serviceName]; exists {
				if state.Status == StatusFailed {
					OutputError("Required service %s failed: %v", serviceName, state.Error)
					os.Exit(1)
				}
			}
		}
	}

	return result.ServerHealth
}

// EnsureServicesWithConfigAndResult returns both health and orchestration result
func EnsureServicesWithConfigAndResult(config *ServiceOrchestrationConfig) (*HealthPayload, *OrchestrationResult) {
	orchestrator := NewServiceOrchestrator(config)
	result := orchestrator.EnsureServices()

	// Check if any required services failed
	for serviceName, requirement := range config.ServiceNeeds {
		if requirement == ServiceRequired {
			if state, exists := result.Services[serviceName]; exists {
				if state.Status == StatusFailed {
					OutputError("Required service %s failed: %v", serviceName, state.Error)
					os.Exit(1)
				}
			}
		}
	}

	return result.ServerHealth, result
}

// FilterHealthForOptionalServices creates a health payload that doesn't show alarming messages for optional services
func FilterHealthForOptionalServices(health *HealthPayload, config *ServiceOrchestrationConfig, mode SessionMode) *HealthPayload {
	if health == nil {
		return nil
	}

	// Create a copy of the health payload
	filtered := &HealthPayload{
		Status:     health.Status,
		Summary:    health.Summary,
		Components: []Component{},
		Seeds:      []Component{},
		Timestamp:  health.Timestamp,
	}

	// Filter components - only include healthy ones or required unhealthy ones
	for _, comp := range health.Components {
		serviceName := getServiceNameFromComponent(&comp)
		requirement, exists := config.ServiceNeeds[serviceName]

		if !exists {
			// Unknown service, include as-is
			filtered.Components = append(filtered.Components, comp)
		} else if requirement == ServiceRequired || strings.EqualFold(comp.Status, "healthy") {
			// Required service (show all statuses) or healthy optional service
			filtered.Components = append(filtered.Components, comp)
		}
		// Skip unhealthy optional services
	}

	// Include seeds for DEV mode
	if mode == SessionModeDev {
		for _, seed := range health.Seeds {
			serviceName := getServiceNameFromComponent(&seed)
			requirement, exists := config.ServiceNeeds[serviceName]

			if !exists {
				// Unknown service, include as-is
				filtered.Seeds = append(filtered.Seeds, seed)
			} else if requirement == ServiceRequired || strings.EqualFold(seed.Status, "healthy") {
				// Required service (show all statuses) or healthy optional service
				filtered.Seeds = append(filtered.Seeds, seed)
			}
			// Skip unhealthy optional services
		}
	}

	// Adjust overall status if we filtered out unhealthy components
	if len(filtered.Components) < len(health.Components) || len(filtered.Seeds) < len(health.Seeds) {
		// Check if remaining components are all healthy
		allHealthy := true
		for _, comp := range filtered.Components {
			if !strings.EqualFold(comp.Status, "healthy") {
				allHealthy = false
				break
			}
		}
		if allHealthy {
			for _, seed := range filtered.Seeds {
				if !strings.EqualFold(seed.Status, "healthy") {
					allHealthy = false
					break
				}
			}
		}

		if allHealthy {
			filtered.Status = "healthy"
			filtered.Summary = "All required services are healthy"
		}
	}

	return filtered
}

// Helper function to determine service name from component
func getServiceNameFromComponent(comp *Component) string {
	name := strings.ToLower(comp.Name)
	if strings.Contains(name, "rag") {
		return "rag"
	}
	if strings.Contains(name, "server") || name == "api" {
		return "server"
	}
	return name // Return as-is for unknown services
}

// checkServerHealth requires /health to be healthy.
func checkServerHealth(serverURL string) (*HealthPayload, error) {
	base := strings.TrimRight(serverURL, "/")
	healthURL := base + "/health"

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, healthURL, nil)
	if err != nil {
		return nil, err
	}
	resp, err := (&http.Client{Timeout: 5 * time.Second}).Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		var payload HealthPayload
		if err := json.Unmarshal(body, &payload); err != nil {
			logDebug(fmt.Sprintf("Invalid health payload: %v", err))
			return nil, fmt.Errorf("invalid health payload: %v", err)
		}
		if strings.EqualFold(payload.Status, "healthy") {
			return &payload, nil
		}
		logDebug(fmt.Sprintf("Server is %s", payload.Status))
		return &payload, &HealthError{Status: payload.Status, HealthResp: payload}
	}
	return nil, fmt.Errorf("unexpected health status %d", resp.StatusCode)
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

	// Get the dynamic image URL using our version-aware resolution
	image, err := getImageURL("server")
	if err != nil {
		return fmt.Errorf("failed to resolve server image URL: %v", err)
	}

	// If a container with this name exists and is running, nothing to do
	if IsContainerRunning(containerName) {
		return nil
	}

	OutputProgress("Starting '%s' as '%s' via Docker...\n", image, containerName)

	// Get Ollama host for container configuration
	ollamaHostVar := os.Getenv("OLLAMA_HOST")
	if ollamaHostVar == "" {
		ollamaHostVar = ollamaHost
	}
	if ollamaHostVar == "" {
		ollamaHostVar = "http://localhost:11434"
	}

	// Prepare container specification
	homeDir, _ := os.UserHomeDir()
	spec := ContainerRunSpec{
		Name:  containerName,
		Image: image,
		StaticPorts: []PortMapping{
			{Host: port, Container: 8000, Protocol: "tcp"},
		},
		Env: make(map[string]string),
		Volumes: []string{
			func() string {
				homeLlamaFarmPath := filepath.Join(homeDir, ".llamafarm")
				dockerPath := convertToDockerPath(homeLlamaFarmPath)
				volumeMount := fmt.Sprintf("%s:%s", dockerPath, "/var/lib/llamafarm")

				// Debug logging for home directory volume mount
				OutputDebug("Home volume mount: %s\n", volumeMount)

				return volumeMount
			}(),
		},
		Labels: map[string]string{
			"llamafarm.component": "server",
			"llamafarm.managed":   "true",
		},
		User: getCurrentUserGroup(),
	}

	// Mount effective working directory into the container at the same path
	if err := setupWorkdirVolumeMount(&spec); err != nil {
		return fmt.Errorf("failed to configure working directory volume: %v", err)
	}

	// Pass through or configure Ollama access inside the container
	if isLocalhost(ollamaHostVar) {
		ollamaPort := resolvePort(ollamaHostVar, 11434)
		spec.AddHosts = []string{"host.docker.internal:host-gateway"}
		spec.Env["OLLAMA_HOST"] = fmt.Sprintf("http://host.docker.internal:%d", ollamaPort)
	} else {
		spec.Env["OLLAMA_HOST"] = ollamaHostVar
	}

	if v, ok := os.LookupEnv("OLLAMA_PORT"); ok && strings.TrimSpace(v) != "" {
		spec.Env["OLLAMA_PORT"] = v
	}

	// Use the Docker SDK-based container starter
	_, err = StartContainerDetachedWithPolicy(spec, &PortResolutionPolicy{
		PreferredHostPort: port,
		Forced:            true,
	})

	return err
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

// prettyPrintHealthProblems prints only the non-healthy components and seeds from a HealthPayload.
// It is intended for concise error reporting.
func prettyPrintHealthProblems(w io.Writer, hr HealthPayload) {
	// Check components
	for _, c := range hr.Components {
		if c.Status != "healthy" {
			icon := iconForStatus(c.Status)
			fmt.Fprintf(w, "  %s %-20s %-10s %s (latency: %dms)\n", icon, c.Name, c.Status, c.Message, c.LatencyMs)
			for k, v := range c.Details {
				fmt.Fprintf(w, "      %s: %v\n", k, v)
			}
		}
	}

	// Check seeds
	for _, s := range hr.Seeds {
		if s.Status != "healthy" {
			icon := iconForStatus(s.Status)
			fmt.Fprintf(w, "  %s %-20s %-10s %s (latency: %dms)\n", icon, s.Name, s.Status, s.Message, s.LatencyMs)
			for k, v := range s.Runtime {
				fmt.Fprintf(w, "      %s: %v\n", k, v)
			}
		}
	}
}

func iconForStatus(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	switch s {
	case "healthy":
		return "✅"
	case "degraded":
		return "⚠️ "
	case "unhealthy":
		return "❌"
	default:
		return "❓"
	}
}

func pingURL(base string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, base, nil)
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

// findRAGComponent finds the RAG component in the health response
func findRAGComponent(hr *HealthPayload) *Component {
	if hr == nil {
		return nil
	}

	// Check components first
	for _, component := range hr.Components {
		name := strings.ToLower(component.Name)
		if name == "rag-service" || strings.HasPrefix(name, "rag-") || strings.HasSuffix(name, "-rag") {
			return &component
		}
	}

	// Check seeds as well
	for _, seed := range hr.Seeds {
		name := strings.ToLower(seed.Name)
		if name == "rag-service" || strings.HasPrefix(name, "rag-") || strings.HasSuffix(name, "-rag") {
			return &seed
		}
	}

	return nil
}

// convertToDockerPath normalizes a host path to use forward slashes which Docker accepts across platforms.
func convertToDockerPath(hostPath string) string {
	converted := filepath.ToSlash(hostPath)

	// Handle Windows drive letters for Docker Desktop (WSL2 backend)
	if runtime.GOOS == "windows" && len(converted) >= 2 && converted[1] == ':' {
		driveLetter := strings.ToLower(string(converted[0]))
		pathWithoutDrive := converted[2:]                // Remove "C:"
		converted = "/" + driveLetter + pathWithoutDrive // Single slash: /c/path

		OutputDebug("Converted Windows path '%s' to Docker format '%s'\n", hostPath, converted)
	}

	return converted
}

// validateDockerVolumePath checks if a path can be safely mounted as a Docker volume
func validateDockerVolumePath(hostPath string) error {
	if strings.TrimSpace(hostPath) == "" {
		return fmt.Errorf("empty path")
	}

	// Check if the path is accessible
	if _, err := os.Stat(hostPath); err != nil {
		return fmt.Errorf("path not accessible: %v", err)
	}

	return nil
}

// setupWorkdirVolumeMount safely sets up the working directory volume mount
// Returns an error if the working directory cannot be determined or validated
func setupWorkdirVolumeMount(spec *ContainerRunSpec) error {
	cwd := getEffectiveCWD()
	if strings.TrimSpace(cwd) == "" {
		return fmt.Errorf("could not determine current directory")
	}

	// Simple validation - check if path is accessible
	if err := validateDockerVolumePath(cwd); err != nil {
		return fmt.Errorf("working directory not accessible (%s): %v", cwd, err)
	}

	// Convert to Docker-compatible path
	dockerPath := convertToDockerPath(cwd)

	volumeMount := fmt.Sprintf("%s:%s", dockerPath, dockerPath)
	spec.Volumes = append(spec.Volumes, volumeMount)

	OutputDebug("Working directory volume mount: %s\n", volumeMount)

	return nil
}
