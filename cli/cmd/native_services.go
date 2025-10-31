package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// NativeOrchestrator manages native process-based service orchestration
type NativeOrchestrator struct {
	uvManager    *UVManager
	pythonEnvMgr *PythonEnvManager
	sourceMgr    *SourceManager
	processMgr   *ProcessManager
	initialized  bool
	initMu       sync.Mutex // protects initialized flag
	serverURL    string
}

// NewNativeOrchestrator creates a new native orchestrator
func NewNativeOrchestrator(serverURL string) (*NativeOrchestrator, error) {
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

	return &NativeOrchestrator{
		uvManager:    uvMgr,
		pythonEnvMgr: pythonMgr,
		sourceMgr:    srcMgr,
		processMgr:   procMgr,
		serverURL:    serverURL,
	}, nil
}

// EnsureNativeEnvironment ensures the native environment is set up
func (no *NativeOrchestrator) EnsureNativeEnvironment() error {
	no.initMu.Lock()
	defer no.initMu.Unlock()

	if no.initialized {
		return nil
	}

	OutputProgress("Setting up native environment...\n")

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
	OutputProgress("Native environment ready\n")
	return nil
}

// StartServerNative starts the server using native processes
func (no *NativeOrchestrator) StartServerNative() error {
	// Ensure environment is ready
	if err := no.EnsureNativeEnvironment(); err != nil {
		return err
	}

	// Check if server is already running
	if no.processMgr.IsProcessHealthy("server") {
		if debug {
			logDebug("Server process already running")
		}
		return nil
	}

	// Verify designer build exists before starting server
	if err := no.sourceMgr.VerifyDesignerBuild(); err != nil {
		if debug {
			logDebug(fmt.Sprintf("Designer build not found: %v", err))
		}
		// Don't fail server startup if designer is missing - server can run without it
		// Designer will show as degraded in health check
	}

	OutputProgress("Starting server via native process...\n")

	// Prepare server environment
	env := no.getServerEnv()

	// Get server directory
	serverDir := no.sourceMgr.GetServerDir()

	// Build command: uv run python main.py
	uvPath := no.uvManager.GetUVPath()
	args := []string{uvPath, "run", "python", "main.py"}

	// Start the process
	if err := no.processMgr.StartProcess("server", serverDir, env, args...); err != nil {
		return fmt.Errorf("failed to start server process: %w", err)
	}

	// Wait a moment for server to start
	time.Sleep(2 * time.Second)

	return nil
}

// StartRAGNative starts the RAG worker using native processes
func (no *NativeOrchestrator) StartRAGNative() error {
	// Ensure environment is ready
	if err := no.EnsureNativeEnvironment(); err != nil {
		return err
	}

	// Check if RAG is already running
	if no.processMgr.IsProcessHealthy("rag") {
		if debug {
			logDebug("RAG process already running")
		}
		return nil
	}

	OutputProgress("Starting RAG worker via native process...\n")

	// Prepare RAG environment
	env := no.getRAGEnv()

	// Get RAG directory
	ragDir := no.sourceMgr.GetRAGDir()

	// Build command: uv run python main.py
	uvPath := no.uvManager.GetUVPath()
	args := []string{uvPath, "run", "python", "main.py"}

	// Start the process
	if err := no.processMgr.StartProcess("rag", ragDir, env, args...); err != nil {
		return fmt.Errorf("failed to start RAG process: %w", err)
	}

	// Wait a moment for RAG to start
	time.Sleep(2 * time.Second)

	// Verify the process is still running (catches immediate failures)
	if !no.processMgr.IsProcessHealthy("rag") {
		return fmt.Errorf("RAG process failed to start or crashed during startup - check logs at %s", filepath.Join(ragDir, "logs"))
	}

	return nil
}

// StartUniversalRuntimeNative starts the universal runtime using native processes
func (no *NativeOrchestrator) StartUniversalRuntimeNative() error {
	// Ensure environment is ready
	if err := no.EnsureNativeEnvironment(); err != nil {
		return err
	}

	// Check if universal runtime is already running
	if no.processMgr.IsProcessHealthy("universal-runtime") {
		if debug {
			logDebug("Universal runtime process already running")
		}
		return nil
	}

	OutputProgress("Starting universal runtime via native process...\n")

	// Prepare universal runtime environment
	env := no.getUniversalRuntimeEnv()

	// Get universal runtime directory
	runtimeDir := no.sourceMgr.GetUniversalRuntimeDir()

	// Build command: uv run python server.py
	uvPath := no.uvManager.GetUVPath()
	args := []string{uvPath, "run", "python", "server.py"}

	// Start the process
	if err := no.processMgr.StartProcess("universal-runtime", runtimeDir, env, args...); err != nil {
		return fmt.Errorf("failed to start universal runtime process: %w", err)
	}

	// Wait a moment for runtime to start
	time.Sleep(2 * time.Second)

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
		ollamaHostVar = ollamaHost
	}
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

	// Set up file logging for the universal runtime
	logsDir := filepath.Join(llamafarmDir, "logs")
	universalLogFile := filepath.Join(logsDir, "universal-runtime.log")
	env = append(env, fmt.Sprintf("LOG_FILE=%s", universalLogFile))

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

// IsServerHealthy checks if the server is healthy via HTTP
func (no *NativeOrchestrator) IsServerHealthy() bool {
	_, err := checkServerHealth(no.serverURL)
	return err == nil
}

// IsRAGHealthy checks if RAG is healthy via HTTP
func (no *NativeOrchestrator) IsRAGHealthy() bool {
	hr, err := checkServerHealth(no.serverURL)
	if err != nil {
		return false
	}

	ragComponent := findRAGComponent(hr)
	if ragComponent == nil {
		return false
	}

	return ragComponent.Status == "healthy"
}

// Global orchestrator instance (for cleanup on exit)
var globalNativeOrchestrator *NativeOrchestrator

// ensureNativeEnvironment is the main entry point for native orchestration
func ensureNativeEnvironment(serverURL string) (*NativeOrchestrator, error) {
	if globalNativeOrchestrator != nil && globalNativeOrchestrator.serverURL == serverURL {
		return globalNativeOrchestrator, nil
	}

	orchestrator, err := NewNativeOrchestrator(serverURL)
	if err != nil {
		return nil, err
	}

	// Ensure environment is set up
	if err := orchestrator.EnsureNativeEnvironment(); err != nil {
		return nil, err
	}

	globalNativeOrchestrator = orchestrator
	return orchestrator, nil
}

// cleanupNativeProcesses cleans up native processes on exit
func cleanupNativeProcesses() {
	if globalNativeOrchestrator != nil {
		if debug {
			logDebug("Cleaning up native processes...")
		}
		globalNativeOrchestrator.StopAllProcesses()
	}
}

// startLocalServerNative starts the server using native processes
func startLocalServerNative(serverURL string) error {
	orchestrator, err := ensureNativeEnvironment(serverURL)
	if err != nil {
		return err
	}

	return orchestrator.StartServerNative()
}

// startRAGNative starts the RAG worker using native processes
func startRAGNative(serverURL string) error {
	orchestrator, err := ensureNativeEnvironment(serverURL)
	if err != nil {
		return err
	}

	return orchestrator.StartRAGNative()
}

// startUniversalRuntimeNative starts the universal runtime using native processes
func startUniversalRuntimeNative(serverURL string) error {
	orchestrator, err := ensureNativeEnvironment(serverURL)
	if err != nil {
		return err
	}

	return orchestrator.StartUniversalRuntimeNative()
}
