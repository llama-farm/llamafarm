package orchestrator

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
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

	// Create source manager
	srcMgr, err := NewSourceManager(pythonMgr)
	if err != nil {
		return nil, fmt.Errorf("failed to create source manager: %w", err)
	}

	// Create process manager
	procMgr, err := NewProcessManager()
	if err != nil {
		return nil, fmt.Errorf("failed to create process manager: %w", err)
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

// Removed: StartService, StopService, RestartService
// These are now handled by ServiceManager which provides better dependency management
// and health checking. Use ServiceManager.EnsureService() instead.

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

// getServerEnv returns environment variables for the server process
func (no *NativeOrchestrator) getServerEnv() []string {
	env := no.pythonEnvMgr.GetEnvForProcess()

	// Add server-specific environment variables
	homeDir, _ := os.UserHomeDir()
	llamafarmDir := filepath.Join(homeDir, ".llamafarm")

	// Get Ollama host
	ollamaHostVar := os.Getenv("OLLAMA_HOST")
	if ollamaHostVar == "" {
		ollamaHostVar = "http://localhost:11434"
	}

	// Add required environment variables
	env = append(env, fmt.Sprintf("OLLAMA_HOST=%s", ollamaHostVar))
	env = append(env, fmt.Sprintf("LLAMAFARM_HOME=%s", llamafarmDir))

	// Get port from serverURL
	port := resolvePort(no.serverURL, 8000)
	env = append(env, fmt.Sprintf("PORT=%d", port))

	// Set up file logging for the server
	logsDir := filepath.Join(llamafarmDir, "logs")
	serverLogFile := filepath.Join(logsDir, "server.log")
	env = append(env, fmt.Sprintf("LOG_FILE=%s", serverLogFile))

	// Add any other environment variables from current environment
	for _, key := range []string{"PATH", "HOME", "USER", "TMPDIR"} {
		if val := os.Getenv(key); val != "" {
			env = append(env, fmt.Sprintf("%s=%s", key, val))
		}
	}

	return env
}

// getRAGEnv returns environment variables for the RAG process
func (no *NativeOrchestrator) getRAGEnv() []string {
	env := no.pythonEnvMgr.GetEnvForProcess()

	// Add RAG-specific environment variables
	homeDir, _ := os.UserHomeDir()
	llamafarmDir := filepath.Join(homeDir, ".llamafarm")

	// Add required environment variables
	env = append(env, fmt.Sprintf("LLAMAFARM_HOME=%s", llamafarmDir))
	env = append(env, fmt.Sprintf("SERVER_URL=%s", no.serverURL))

	// Add any other environment variables from current environment
	for _, key := range []string{"PATH", "HOME", "USER", "TMPDIR"} {
		if val := os.Getenv(key); val != "" {
			env = append(env, fmt.Sprintf("%s=%s", key, val))
		}
	}

	return env
}

// getUniversalRuntimeEnv returns environment variables for the universal runtime process
func (no *NativeOrchestrator) getUniversalRuntimeEnv() []string {
	env := no.pythonEnvMgr.GetEnvForProcess()

	// Add universal runtime-specific environment variables
	homeDir, _ := os.UserHomeDir()
	llamafarmDir := filepath.Join(homeDir, ".llamafarm")

	// Get environment variables with defaults
	port := os.Getenv("TRANSFORMERS_PORT")
	if port == "" {
		port = "11540"
	}

	host := os.Getenv("TRANSFORMERS_HOST")
	if host == "" {
		host = "127.0.0.1"
	}

	outputDir := os.Getenv("TRANSFORMERS_OUTPUT_DIR")
	if outputDir == "" {
		outputDir = filepath.Join(llamafarmDir, "outputs", "images")
	}

	cacheDir := os.Getenv("TRANSFORMERS_CACHE_DIR")
	if cacheDir == "" {
		cacheDir = filepath.Join(homeDir, ".cache", "huggingface")
	}

	// Add runtime-specific environment variables
	env = append(env, fmt.Sprintf("TRANSFORMERS_PORT=%s", port))
	env = append(env, fmt.Sprintf("TRANSFORMERS_HOST=%s", host))
	env = append(env, fmt.Sprintf("TRANSFORMERS_OUTPUT_DIR=%s", outputDir))
	env = append(env, fmt.Sprintf("HF_HOME=%s", cacheDir))

	// Pass through device override variables if set
	if val := os.Getenv("TRANSFORMERS_SKIP_MPS"); val != "" {
		env = append(env, fmt.Sprintf("TRANSFORMERS_SKIP_MPS=%s", val))
	}
	if val := os.Getenv("TRANSFORMERS_FORCE_CPU"); val != "" {
		env = append(env, fmt.Sprintf("TRANSFORMERS_FORCE_CPU=%s", val))
	}
	// Pass through MPS memory limit configuration
	if val := os.Getenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO"); val != "" {
		env = append(env, fmt.Sprintf("PYTORCH_MPS_HIGH_WATERMARK_RATIO=%s", val))
	}

	// Pass through HuggingFace token if set
	if val := os.Getenv("HF_TOKEN"); val != "" {
		env = append(env, fmt.Sprintf("HF_TOKEN=%s", val))
	}

	// Add any other environment variables from current environment
	for _, key := range []string{"PATH", "HOME", "USER", "TMPDIR"} {
		if val := os.Getenv(key); val != "" {
			env = append(env, fmt.Sprintf("%s=%s", key, val))
		}
	}

	return env
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

// Removed: Old ServiceOrchestrator implementation (300+ lines)
// This has been completely replaced by the cleaner ServiceManager in services.go
// which provides:
// - Declarative service definitions
// - Proper dependency resolution via topological sort
// - Generic framework methods (no per-service functions needed)
// - Better separation of concerns
//
// Use ServiceManager.EnsureService() or ServiceManager.StartAll() instead.
