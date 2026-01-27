package orchestrator

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/llamafarm/cli/cmd/utils"
)

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
	Health  *ComponentHealth // From health payload
}

type ServiceOrchestrationConfig struct {
	ServerURL       string
	PrintStatus     bool
	ServiceNeeds    map[string]ServiceRequirement
	DefaultTimeout  time.Duration
	ServiceTimeouts map[string]time.Duration
	AutoStart       bool
}

type OrchestrationResult struct {
	ServerHealth *HealthPayload
	Services     map[string]*ServiceState

	// Channels for async monitoring
	ServerReady chan *ServiceState
	RAGReady    chan *ServiceState
	Done        chan struct{}
}

// ServiceDefinition defines a service in a declarative way.
// Services only need to specify their configuration; the framework handles starting/stopping.
type ServiceDefinition struct {
	Name            string
	Dependencies    []string
	CanStartLocally bool
	DefaultTimeout  time.Duration

	// Declarative start configuration
	WorkDir         string            // Working directory for the process
	Command         string            // Command to execute (e.g., "uv", "python")
	Args            []string          // Command arguments
	Env             map[string]string // Environment variables
	HealthComponent string            // Component name in /health endpoint (e.g., "server", "rag")

	// Hardware-specific packages (optional)
	// If set, these packages will be installed with hardware-appropriate wheels after uv sync
	HardwarePackages []HardwarePackageSpec

	// Runtime info
	State *ServiceState
}

// Removed: ServiceOrchestrator type - replaced entirely by ServiceManager
// ServiceManager provides a cleaner, declarative approach to service management

type ComponentHealth struct {
	Name      string                 `json:"name"`
	Status    string                 `json:"status"`
	Message   string                 `json:"message"`
	LatencyMs int                    `json:"latency_ms"`
	Details   map[string]interface{} `json:"details,omitempty"`
	Runtime   map[string]interface{} `json:"runtime,omitempty"`
}
type HealthPayload struct {
	Status     string            `json:"status"`
	Summary    string            `json:"summary"`
	Components []ComponentHealth `json:"components"`
	Seeds      []ComponentHealth `json:"seeds"`
	Timestamp  int64             `json:"timestamp"`
}

// HealthError wraps a non-healthy /health response.
type HealthError struct {
	Status     string
	HealthResp HealthPayload
}

func (e *HealthError) Error() string {
	return fmt.Sprintf("server unhealthy: %s", e.Status)
}

const hfHubDisableProgressBars = "1" // Disable HF transfer progress bars, as they can cause headless services to crash

// Service Graph Definition
// Services are defined declaratively - just specify config, the framework handles the rest

var ServiceGraph = map[string]*ServiceDefinition{
	"universal-runtime": {
		Name:            "universal-runtime",
		Dependencies:    []string{"server"},
		CanStartLocally: true,
		DefaultTimeout:  180 * time.Second, // Longer timeout for first-time dependency installation
		WorkDir:         "runtimes/universal",
		Command:         "uv",
		Args: func() []string {
			if runtime.GOOS == "linux" {
				return []string{"run", "--managed-python", "python", "server.py"}
			}
			return []string{"run", "--managed-python", "--no-sync", "python", "server.py"}
		}(),
		Env: map[string]string{
			"LOG_FILE":                     filepath.Join("${LF_DATA_DIR}", "logs", "universal-runtime.log"),
			"LF_RUNTIME_PORT":              "11540",
			"LF_RUNTIME_HOST":              "127.0.0.1",
			"TRANSFORMERS_OUTPUT_DIR":      filepath.Join("${LF_DATA_DIR}", "outputs", "images"),
			"TRANSFORMERS_CACHE_DIR":       filepath.Join("${HOME}", ".cache", "huggingface"),
			"HF_HUB_DISABLE_PROGRESS_BARS": hfHubDisableProgressBars,
			// Device control (empty = inherit from parent environment)
			"TRANSFORMERS_SKIP_MPS":  "", // Set to "1" to skip MPS on macOS
			"TRANSFORMERS_FORCE_CPU": "", // Set to "1" to force CPU (useful in CI)
			// Note: PYTORCH_MPS_HIGH_WATERMARK_RATIO removed - setting it to non-default
			// values causes "invalid low watermark ratio" errors on some PyTorch versions.
			// Let PyTorch use its default memory management.
			"HF_TOKEN": "",
			// In CI environments, use CPU-only PyTorch to avoid downloading 3GB+ of CUDA packages
			"UV_EXTRA_INDEX_URL": "${UV_EXTRA_INDEX_URL}",
		},
		HealthComponent: "universal-runtime",
		HardwarePackages: []HardwarePackageSpec{
			PyTorchSpec,
			// Note: llama.cpp binaries are downloaded separately via InstallLlamaBinary,
			// not via pip. The llamafarm-llama package is installed as a regular dependency.
		},
	},
	"server": {
		Name:            "server",
		Dependencies:    []string{}, // No dependencies
		CanStartLocally: true,
		DefaultTimeout:  90 * time.Second,
		WorkDir:         "server",
		Command:         "uv",
		Args:            []string{"run", "--managed-python", "uvicorn", "main:app", "--host", "0.0.0.0"},
		Env: map[string]string{
			"LOG_FILE":                     filepath.Join("${LF_DATA_DIR}", "logs", "server.log"),
			"OLLAMA_HOST":                  "http://localhost:11434",
			"HF_HUB_DISABLE_PROGRESS_BARS": hfHubDisableProgressBars,
		},
		HealthComponent: "server",
	},
	"rag": {
		Name:            "rag",
		Dependencies:    []string{"server", "universal-runtime"}, // Depends on both
		CanStartLocally: true,
		DefaultTimeout:  180 * time.Second,
		WorkDir:         "rag",
		Command:         "uv",
		Args:            []string{"run", "--managed-python", "python", "main.py"},
		Env: map[string]string{
			"LOG_FILE":                     filepath.Join("${LF_DATA_DIR}", "logs", "rag.log"),
			"HF_HUB_DISABLE_PROGRESS_BARS": hfHubDisableProgressBars,
		},
		HealthComponent: "rag-service",
	},
}

// ServiceManager handles service operations: start, stop, check health, and aggregate status.

type ServiceManager struct {
	serverURL    string
	services     map[string]*ServiceDefinition
	orchestrator *NativeOrchestrator
	mu           sync.Mutex
}

// NewServiceManager returns a new ServiceManager.
func NewServiceManager(serverURL string) (*ServiceManager, error) {
	orchestrator, err := NewOrchestrator(serverURL)

	if err != nil {
		return nil, fmt.Errorf("failed to create orchestrator: %w", err)
	}

	return &ServiceManager{
		serverURL:    serverURL,
		services:     ServiceGraph,
		orchestrator: orchestrator,
	}, nil
}

// EnsureService starts a service and all of its dependencies in the correct order.
// It performs a topological sort of the dependency graph to determine the start order,
// ensures circular dependencies are detected, and verifies each service becomes healthy
// before proceeding to its dependents.
//
// Example: If "rag" depends on ["server", "universal-runtime"], calling
// EnsureService("rag") will start "server" and "universal-runtime" first (in any order
// since they have no dependencies), wait for them to become healthy, then start "rag".
func (sm *ServiceManager) EnsureService(serviceName string) error {
	if _, exists := ServiceGraph[serviceName]; !exists {
		return fmt.Errorf("unknown service: %s", serviceName)
	}

	// Build dependency resolution order using topological sort
	resolvedOrder, err := sm.resolveDependencies(serviceName)
	if err != nil {
		return fmt.Errorf("failed to resolve dependencies for %s: %w", serviceName, err)
	}

	// Ensure each service in dependency order (dependencies before dependents)
	for _, svcName := range resolvedOrder {
		if err := sm.ensureSingleService(svcName); err != nil {
			return fmt.Errorf("failed to ensure dependency %s: %w", svcName, err)
		}
	}

	return nil
}

// EnsureServices starts multiple services and all their dependencies in the correct order.
// This is a convenience method that ensures multiple services, resolving all dependencies
// across all requested services and starting them in the correct topological order.
//
// Example: EnsureServices("server", "universal-runtime") will start both services
func (sm *ServiceManager) EnsureServices(serviceNames ...string) error {
	if len(serviceNames) == 0 {
		return nil
	}

	// Collect all services and their dependencies in proper order
	allServicesMap := make(map[string]bool)
	var orderedServices []string

	for _, serviceName := range serviceNames {
		if _, exists := ServiceGraph[serviceName]; !exists {
			return fmt.Errorf("unknown service: %s", serviceName)
		}

		// Resolve dependencies for this service (returns topologically sorted order)
		resolvedOrder, err := sm.resolveDependencies(serviceName)
		if err != nil {
			return fmt.Errorf("failed to resolve dependencies for %s: %w", serviceName, err)
		}

		// Add services to both map (for deduplication) and ordered list
		for _, svc := range resolvedOrder {
			if !allServicesMap[svc] {
				allServicesMap[svc] = true
				orderedServices = append(orderedServices, svc)
			}
		}
	}

	// Start services in dependency order (server will always be first)
	for _, svcName := range orderedServices {
		if err := sm.ensureSingleService(svcName); err != nil {
			return fmt.Errorf("failed to ensure service %s: %w", svcName, err)
		}
	}

	return nil
}

// EnsureServicesOrExit is a convenience function that creates a ServiceManager and ensures
// the specified services are running. If any error occurs (initialization or service start),
// it prints to stderr and exits with code 1. This is the standard pattern for CLI commands
// that need to ensure services are available.
//
// Example usage:
//
//	orchestrator.EnsureServicesOrExit(serverURL, "server")
//	orchestrator.EnsureServicesOrExit(serverURL, "server", "rag", "universal-runtime")
func EnsureServicesOrExit(serverURL string, serviceNames ...string) {
	sm, err := NewServiceManager(serverURL)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize service manager: %v\n", err)
		os.Exit(1)
	}

	if err := sm.EnsureServices(serviceNames...); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to start services: %v\n", err)
		os.Exit(1)
	}
}

// EnsureServicesWithConfig ensures services are running, respecting AutoStart flag
func (sm *ServiceManager) EnsureServicesWithConfig(config *ServiceOrchestrationConfig, serviceNames ...string) error {
	// If auto-start is disabled, check health but don't start services
	if !config.AutoStart {
		var unhealthyServices []string

		for _, serviceName := range serviceNames {
			serviceDef, exists := ServiceGraph[serviceName]
			if !exists {
				return fmt.Errorf("unknown service: %s", serviceName)
			}

			if !sm.isServiceHealthy(serviceDef) {
				unhealthyServices = append(unhealthyServices, serviceName)
			}
		}

		if len(unhealthyServices) > 0 {
			return fmt.Errorf("services not running and auto-start is disabled: %s (use --auto-start to enable automatic startup)", strings.Join(unhealthyServices, ", "))
		}

		if config.PrintStatus {
			fmt.Println("✓ All required services are already running")
		}
		return nil
	}

	// Auto-start is enabled, use existing logic
	return sm.EnsureServices(serviceNames...)
}

// EnsureServicesOrExitWithConfig wraps EnsureServicesWithConfig with exit behavior
func EnsureServicesOrExitWithConfig(config *ServiceOrchestrationConfig, serviceNames ...string) {
	sm, err := NewServiceManager(config.ServerURL)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize service manager: %v\n", err)
		os.Exit(1)
	}

	if err := sm.EnsureServicesWithConfig(config, serviceNames...); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

// ensureSingleService ensures a single service is running without checking dependencies.
// Uses the declarative service configuration to start the process via the framework.
func (sm *ServiceManager) ensureSingleService(serviceName string) error {
	serviceDef, exists := ServiceGraph[serviceName]
	if !exists {
		return fmt.Errorf("unknown service: %s", serviceName)
	}

	// Check if service is already healthy
	if sm.isServiceHealthy(serviceDef) {
		utils.LogDebug(fmt.Sprintf("Service %s is already healthy", serviceName))
		return nil
	}

	// Service not healthy, start it using the declarative configuration
	utils.LogDebug(fmt.Sprintf("Starting service %s", serviceName))
	if err := sm.startService(serviceDef); err != nil {
		return fmt.Errorf("failed to start service %s: %w", serviceName, err)
	}

	// Wait for service to become ready by polling health endpoint
	if err := sm.waitForServiceReady(serviceDef); err != nil {
		return fmt.Errorf("service %s did not become ready: %w", serviceName, err)
	}

	return nil
}

// startService starts a service using its declarative configuration
func (sm *ServiceManager) startService(serviceDef *ServiceDefinition) error {
	// Build environment variables
	env := sm.orchestrator.getDefaultEnvWithKeys(serviceDef.Env)

	// Build command args - replace "uv" with full path if needed
	command := serviceDef.Command
	if command == "uv" {
		// Use the full path to uv to avoid PATH issues
		command = sm.orchestrator.pythonEnvMgr.uvManager.GetUVPath()
	}
	cmdArgs := append([]string{command}, serviceDef.Args...)

	// Get source directory
	lfDir, _ := utils.GetLFDataDir()
	sourceDir := filepath.Join(lfDir, "src")
	workDir := filepath.Join(sourceDir, serviceDef.WorkDir)

	return sm.orchestrator.processMgr.StartProcess(serviceDef.Name, workDir, env, cmdArgs...)
}

// isServiceHealthy checks if a service is healthy by querying its health component
func (sm *ServiceManager) isServiceHealthy(serviceDef *ServiceDefinition) bool {
	hr, err := sm.GetServerHealth()
	if err != nil {
		utils.LogDebug(fmt.Sprintf("isServiceHealthy(%s): GetServerHealth failed: %v", serviceDef.Name, err))
		return false
	}

	component := findComponent(hr, serviceDef.HealthComponent)
	if component == nil {
		utils.LogDebug(fmt.Sprintf("isServiceHealthy(%s): component '%s' not found in health response", serviceDef.Name, serviceDef.HealthComponent))
		return false
	}

	isHealthy := strings.EqualFold(component.Status, "healthy")
	utils.LogDebug(fmt.Sprintf("isServiceHealthy(%s): component '%s' status='%s', healthy=%v", serviceDef.Name, serviceDef.HealthComponent, component.Status, isHealthy))
	return isHealthy
}

// waitForServiceReady waits for a service to become healthy by polling its health endpoint
func (sm *ServiceManager) waitForServiceReady(serviceDef *ServiceDefinition) error {
	deadline := time.Now().Add(serviceDef.DefaultTimeout)
	pollInterval := 500 * time.Millisecond

	for time.Now().Before(deadline) {
		if sm.isServiceHealthy(serviceDef) {
			return nil
		}
		time.Sleep(pollInterval)
	}

	// Final check after timeout
	if sm.isServiceHealthy(serviceDef) {
		return nil
	}

	return fmt.Errorf("service did not become healthy within %v", serviceDef.DefaultTimeout)
}

// resolveDependencies performs a topological sort to determine the order in which services should be started
func (sm *ServiceManager) resolveDependencies(serviceName string) ([]string, error) {
	visited := make(map[string]bool)
	recursionStack := make(map[string]bool)
	result := []string{}

	var visit func(string) error
	visit = func(name string) error {
		// Check for cycles
		if recursionStack[name] {
			return fmt.Errorf("circular dependency detected involving service: %s", name)
		}

		// Already processed
		if visited[name] {
			return nil
		}

		// Get service definition
		svcDef, exists := ServiceGraph[name]
		if !exists {
			return fmt.Errorf("unknown service in dependency graph: %s", name)
		}

		// Mark as being processed (for cycle detection)
		recursionStack[name] = true

		// Visit all dependencies first
		for _, dep := range svcDef.Dependencies {
			if err := visit(dep); err != nil {
				return err
			}
		}

		// Mark as visited and remove from recursion stack
		recursionStack[name] = false
		visited[name] = true

		// Add to result (dependencies before dependents)
		result = append(result, name)
		return nil
	}

	// Start traversal from requested service
	if err := visit(serviceName); err != nil {
		return nil, err
	}

	return result, nil
}

// Removed: checkServiceHealth - replaced by isServiceHealthy which uses the declarative HealthComponent field

// StopService stops a service and all services that depend on it.
// Services are stopped in reverse dependency order (dependents before dependencies).
//
// Example: If "rag" depends on "server", calling StopService("server") will:
// 1. First stop "rag" (dependent)
// 2. Then stop "server"
func (sm *ServiceManager) StopService(serviceName string) error {
	if _, exists := ServiceGraph[serviceName]; !exists {
		return fmt.Errorf("unknown service: %s", serviceName)
	}

	// Find all services that need to be stopped (service + anything that depends on it)
	servicesToStop := sm.findDependents(serviceName)

	// Stop each service in reverse order (dependents first)
	for i := len(servicesToStop) - 1; i >= 0; i-- {
		svcName := servicesToStop[i]
		if err := sm.stopSingleService(svcName); err != nil {
			// Log error but continue stopping other services
			utils.LogDebug(fmt.Sprintf("Error stopping %s: %v", svcName, err))
		}
	}

	return nil
}

// StopServices stops multiple services and all services that depend on them.
// Services are stopped in reverse dependency order (dependents before dependencies).
//
// Example: StopServices("server", "universal-runtime") will stop all services
// that depend on either of them first, then stop the specified services.
func (sm *ServiceManager) StopServices(serviceNames ...string) error {
	if len(serviceNames) == 0 {
		return nil
	}

	// Collect all services that need to be stopped
	allServicesToStop := make(map[string]bool)
	for _, serviceName := range serviceNames {
		if _, exists := ServiceGraph[serviceName]; !exists {
			return fmt.Errorf("unknown service: %s", serviceName)
		}

		// Find all dependents for this service
		dependents := sm.findDependents(serviceName)
		for _, svc := range dependents {
			allServicesToStop[svc] = true
		}
	}

	// Convert to slice and stop in reverse dependency order
	// Build dependency order first, then reverse it
	orderedServices := []string{}
	for svc := range allServicesToStop {
		// Get the full dependency chain for proper ordering
		resolved, err := sm.resolveDependencies(svc)
		if err != nil {
			continue // Skip if we can't resolve
		}
		for _, s := range resolved {
			if allServicesToStop[s] {
				// Add to ordered list if not already present
				found := false
				for _, existing := range orderedServices {
					if existing == s {
						found = true
						break
					}
				}
				if !found {
					orderedServices = append(orderedServices, s)
				}
			}
		}
	}

	// Stop in reverse order (dependents before dependencies)
	for i := len(orderedServices) - 1; i >= 0; i-- {
		svcName := orderedServices[i]
		if err := sm.stopSingleService(svcName); err != nil {
			utils.LogDebug(fmt.Sprintf("Error stopping %s: %v", svcName, err))
		}
	}

	return nil
}

// stopSingleService stops a single service without checking dependents
func (sm *ServiceManager) stopSingleService(serviceName string) error {
	serviceDef, exists := ServiceGraph[serviceName]
	if !exists {
		return fmt.Errorf("unknown service: %s", serviceName)
	}

	utils.LogDebug(fmt.Sprintf("Stopping service %s", serviceName))
	return sm.orchestrator.processMgr.StopProcess(serviceDef.Name)
}

// findDependents finds all services that directly or indirectly depend on the given service
func (sm *ServiceManager) findDependents(serviceName string) []string {
	dependents := []string{serviceName}
	dependentsMap := make(map[string]bool)
	dependentsMap[serviceName] = true

	// Keep searching for new dependents until we find no more
	changed := true
	for changed {
		changed = false
		for svcName, svcDef := range ServiceGraph {
			if dependentsMap[svcName] {
				continue // Already in the list
			}
			// Check if this service depends on any service in our list
			for _, dep := range svcDef.Dependencies {
				if dependentsMap[dep] {
					dependentsMap[svcName] = true
					dependents = append(dependents, svcName)
					changed = true
					break
				}
			}
		}
	}

	return dependents
}

// StartAll starts all services in the service graph, respecting dependencies.
// Uses EnsureService which handles dependency resolution and health checking.
func (sm *ServiceManager) StartAll() error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	// Start each service (EnsureService handles dependencies automatically)
	for svcName := range sm.services {
		if err := sm.EnsureService(svcName); err != nil {
			return fmt.Errorf("failed to start service %q: %w", svcName, err)
		}
	}
	return nil
}

// StopAll stops all services in the service graph, respecting reverse dependencies.
// Services are stopped in reverse dependency order (dependents before dependencies).
func (sm *ServiceManager) StopAll() error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	// Collect all service names
	serviceNames := make([]string, 0, len(sm.services))
	for svcName := range sm.services {
		serviceNames = append(serviceNames, svcName)
	}

	// Use StopServices to handle proper ordering
	return sm.StopServices(serviceNames...)
}

// GetServerHealth requires /health to be healthy.
func (sm *ServiceManager) GetServerHealth() (*HealthPayload, error) {
	base := strings.TrimRight(sm.serverURL, "/")
	healthURL := base + "/health"

	utils.LogDebug(fmt.Sprintf("GetServerHealth: checking %s", healthURL))

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, healthURL, nil)
	if err != nil {
		utils.LogDebug(fmt.Sprintf("GetServerHealth: failed to create request: %v", err))
		return nil, err
	}
	resp, err := (&http.Client{Timeout: 12 * time.Second}).Do(req)
	if err != nil {
		utils.LogDebug(fmt.Sprintf("GetServerHealth: request failed: %v", err))
		return nil, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		var payload HealthPayload
		if err := json.Unmarshal(body, &payload); err != nil {
			utils.LogDebug(fmt.Sprintf("Invalid health payload: %v", err))
			return nil, fmt.Errorf("invalid health payload: %v", err)
		}
		utils.LogDebug(fmt.Sprintf("GetServerHealth: status=%s, components=%d", payload.Status, len(payload.Components)))
		if strings.EqualFold(payload.Status, "healthy") {
			return &payload, nil
		}
		utils.LogDebug(fmt.Sprintf("Server is %s (returning HealthError)", payload.Status))
		return &payload, &HealthError{Status: payload.Status, HealthResp: payload}
	}
	utils.LogDebug(fmt.Sprintf("GetServerHealth: unexpected status code %d", resp.StatusCode))
	return nil, fmt.Errorf("unexpected health status %d", resp.StatusCode)
}

// Removed: Service-specific health check functions (checkServerHealthForService, checkRAGHealthForService, checkUniversalRuntimeHealthForService)
// These are replaced by the generic isServiceHealthy method which uses the HealthComponent field from ServiceDefinition

// findComponent finds a component in the health response by name
func findComponent(hr *HealthPayload, componentName string) *ComponentHealth {
	if hr == nil {
		return nil
	}

	// Check components first
	for _, component := range hr.Components {
		name := strings.ToLower(component.Name)
		if name == strings.ToLower(componentName) {
			return &component
		}
	}

	return nil
}

// ServiceStatusInfo represents status information for a single service
// This is a simple struct used by the orchestrator layer; the CLI layer
// will convert this to its own ServiceInfo type to avoid circular dependencies
type ServiceStatusInfo struct {
	Name    string
	State   string // "running", "stopped", "not_found"
	PID     int
	LogFile string
	Uptime  time.Duration
	Health  *ComponentHealth
}

// GetServicesStatus returns the current status of all services
// It combines process state information with health data from the server
func (sm *ServiceManager) GetServicesStatus() ([]ServiceStatusInfo, error) {
	// Try to get server health (may fail if server is down)
	healthPayload, serverHealthErr := sm.GetServerHealth()

	var statuses []ServiceStatusInfo

	// Iterate through all services in the service graph
	for serviceName, serviceDef := range ServiceGraph {
		statusInfo := ServiceStatusInfo{
			Name:  serviceName,
			State: "stopped",
		}

		// Try to get process info from the process manager
		procInfo, err := sm.orchestrator.processMgr.GetProcessInfo(serviceName, healthPayload)
		if err == nil {
			// Process is tracked, check if it's running
			if sm.orchestrator.processMgr.isProcessRunning(procInfo) {
				statusInfo.State = "running"
				statusInfo.PID = procInfo.PID
				statusInfo.LogFile = procInfo.LogFile
				statusInfo.Uptime = time.Since(procInfo.StartTime)
			} else {
				statusInfo.State = "stopped"
				statusInfo.LogFile = procInfo.LogFile
			}
		}

		// If server health is available and process is running, get health info
		if serverHealthErr == nil && healthPayload != nil {
			component := findComponent(healthPayload, serviceDef.HealthComponent)
			if component != nil {
				statusInfo.State = "running"
				statusInfo.Health = component

				// Make PID optional: only set if "pid" is present and can be cast to int
				if pidVal, ok := component.Details["pid"]; ok {
					statusInfo.PID = int(pidVal.(float64))
				}

				if uptimeVal, ok := component.Details["uptime"]; ok {
					statusInfo.Uptime = uptimeVal.(time.Duration)
				}

				if logFileVal, ok := component.Details["log_file"]; ok {
					statusInfo.LogFile = logFileVal.(string)
				}
			}
		}

		statuses = append(statuses, statusInfo)
	}

	return statuses, nil
}

// Removed: waitForCondition and old waitForServiceReady - replaced by ServiceManager.waitForServiceReady method
