package orchestrator

import (
	"errors"
	"fmt"
	"os"
	"sync"

	"github.com/llamafarm/cli/cmd/utils"
)

// ErrServiceAlreadyRunning indicates that a service is already running
var ErrServiceAlreadyRunning = errors.New("service is already running")

// NativeOrchestrator manages the native Python/UV infrastructure layer.
//
// Responsibilities:
// - UV installation and management
// - Python environment setup (via UV)
// - Source code download and dependency sync
// - Process management (start/stop/tracking)
// - Environment variable builders for services
//
// NOT responsible for:
// - Service lifecycle management (see ServiceManager)
// - Dependency resolution (see ServiceManager)
// - Health checking (see ServiceManager)
//
// Use ServiceManager for high-level service orchestration.
// NativeOrchestrator provides the infrastructure that ServiceManager builds on.
type NativeOrchestrator struct {
	uvManager    *UVManager
	pythonEnvMgr *PythonEnvManager
	sourceMgr    *SourceManager
	processMgr   *ProcessManager
	initialized  bool
	initMu       sync.Mutex // protects initialized flag
	serverURL    string     // current runtime URL (may be adjusted for port conflicts)
}

// NewNativeOrchestrator creates a new native orchestrator
func NewOrchestrator(serverURL string) (*NativeOrchestrator, error) {
	// Create UV manager
	uvMgr, err := NewUVManager()
	if err != nil {
		return nil, fmt.Errorf("failed to create UV manager: %w", err)
	}

	// Create Python environment manager
	pythonMgr, err := NewPythonEnvManager(uvMgr)
	if err != nil {
		return nil, fmt.Errorf("failed to create Python environment manager: %w", err)
	}

	// Create process manager (needed before source manager for service shutdown during upgrades)
	procMgr, err := NewProcessManager()
	if err != nil {
		return nil, fmt.Errorf("failed to create process manager: %w", err)
	}

	// Create source manager (with process manager for stopping services during upgrades)
	srcMgr, err := NewSourceManager(pythonMgr, procMgr)
	if err != nil {
		return nil, fmt.Errorf("failed to create source manager: %w", err)
	}

	orchestrator := &NativeOrchestrator{
		uvManager:    uvMgr,
		pythonEnvMgr: pythonMgr,
		sourceMgr:    srcMgr,
		processMgr:   procMgr,
		serverURL:    serverURL,
	}

	if err := orchestrator.EnsureNativeEnvironment(); err != nil {
		utils.OutputError("Environment initialization failed: %v\n", err)
		os.Exit(1)
	}

	return orchestrator, nil
}

// EnsureNativeEnvironment ensures the native environment is set up
func (no *NativeOrchestrator) EnsureNativeEnvironment() error {
	no.initMu.Lock()
	defer no.initMu.Unlock()

	if no.initialized {
		return nil
	}

	utils.LogDebug("Setting up native environment...\n")

	// Step 1: Ensure UV is installed
	if _, err := no.uvManager.EnsureUV(); err != nil {
		return fmt.Errorf("failed to ensure UV: %w", err)
	}

	// Step 2: Ensure Python is installed
	if _, err := no.pythonEnvMgr.EnsurePython(); err != nil {
		return fmt.Errorf("failed to ensure Python: %w", err)
	}

	// Step 3: Ensure source code is downloaded and dependencies are synced
	if err := no.sourceMgr.EnsureSource(); err != nil {
		return fmt.Errorf("failed to ensure source code: %w", err)
	}

	no.initialized = true
	utils.LogDebug("Native environment ready\n")
	return nil
}

func (no *NativeOrchestrator) getDefaultEnvWithKeys(envKeysWithDefaults map[string]string) []string {
	env := no.pythonEnvMgr.GetEnvForProcess()

	// Always include core environment keys from the current environment
	// Note: PATH is already set by GetEnvForProcess() with UV bin directory, so we don't override it
	extraEnv := []string{}
	for _, key := range []string{"HOME", "USER", "TMPDIR", "LF_DATA_DIR"} {
		if val := os.Getenv(key); val != "" {
			extraEnv = append(extraEnv, fmt.Sprintf("%s=%s", key, val))
		}
	}

	for key, val := range envKeysWithDefaults {
		if val != "" {
			// Expand environment variable placeholders like ${LF_DATA_DIR}
			expandedVal := os.ExpandEnv(val)
			env = append(env, fmt.Sprintf("%s=%s", key, expandedVal))
		}
	}
	return append(env, extraEnv...)
}

// StopAllProcesses stops all native processes
func (no *NativeOrchestrator) StopAllProcesses() {
	if no.processMgr != nil {
		no.processMgr.StopAllProcesses()
	}
}

// GetProcessManager returns the process manager
func (no *NativeOrchestrator) GetProcessManager() *ProcessManager {
	return no.processMgr
}
