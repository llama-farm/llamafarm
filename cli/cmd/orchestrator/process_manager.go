package orchestrator

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"

	"github.com/llamafarm/cli/cmd/utils"
)

// ProcessInfo holds information about a managed process
type ProcessInfo struct {
	Name      string
	Cmd       *exec.Cmd
	LogFile   string
	StartTime time.Time
	Status    string
	PID       int // Store PID explicitly for attached processes
	mu        sync.RWMutex
}

// Service lock and PID file timeout constants
const (
	// ServiceLockTimeout is how long to wait for a service lock before giving up
	ServiceLockTimeout = 30 * time.Second
	// ServiceLockPollInterval is how often to retry acquiring a service lock
	ServiceLockPollInterval = 500 * time.Millisecond
	// PIDFileWaitTimeout is how long to wait for a PID file to be written
	PIDFileWaitTimeout = 10 * time.Second
	// PIDFilePollInterval is how often to check for a PID file
	PIDFilePollInterval = 200 * time.Millisecond
)

// ProcessManager manages native processes for services
type ProcessManager struct {
	logsDir   string
	pidsDir   string
	processes map[string]*ProcessInfo
	mu        sync.RWMutex
}

// NewProcessManager creates a new process manager
func NewProcessManager() (*ProcessManager, error) {
	// Use GetLFDataDir to respect LF_DATA_DIR environment variable
	lfDataDir, err := utils.GetLFDataDir()
	if err != nil {
		return nil, fmt.Errorf("failed to get LF data directory: %w", err)
	}

	logsDir := filepath.Join(lfDataDir, "logs")
	if err := os.MkdirAll(logsDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create logs directory: %w", err)
	}

	pidsDir := filepath.Join(lfDataDir, "pids")
	if err := os.MkdirAll(pidsDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create pids directory: %w", err)
	}

	return &ProcessManager{
		logsDir:   logsDir,
		pidsDir:   pidsDir,
		processes: make(map[string]*ProcessInfo),
	}, nil
}

// StartProcess starts a new process and manages its lifecycle
// It uses file locking to prevent multiple CLI instances from starting duplicate processes.
func (pm *ProcessManager) StartProcess(name string, workDir string, env []string, args ...string) error {
	// Acquire exclusive lock to prevent race conditions with other CLI instances
	// This lock is held during the entire check-start-wait sequence
	lockFile, err := pm.acquireServiceLock(name)
	if err != nil {
		return fmt.Errorf("failed to acquire service lock for %s: %w", name, err)
	}
	defer pm.releaseServiceLock(lockFile)

	// Check if process is already running via PID file (from another CLI instance)
	if pid, found := pm.ReadPIDFile(name); found {
		if isProcessAlive(pid) {
			utils.LogDebug(fmt.Sprintf("Process %s already running (PID %d from pidfile)", name, pid))
			return fmt.Errorf("process %s is already running (PID %d)", name, pid)
		}
		// Stale PID file from crashed process, clean it up
		utils.LogDebug(fmt.Sprintf("Removing stale PID file for %s (PID %d not running)", name, pid))
		pm.removePIDFile(name)
	}

	pm.mu.Lock()

	// Check if process is already running in this CLI instance's memory
	if proc, exists := pm.processes[name]; exists {
		if pm.isProcessRunning(proc) {
			pm.mu.Unlock()
			return fmt.Errorf("process %s is already running (PID %d)", name, proc.PID)
		}
		// Clean up old process info
		delete(pm.processes, name)
	}

	// Create log file
	logFile := filepath.Join(pm.logsDir, fmt.Sprintf("%s.log", name))
	logF, err := os.Create(logFile)
	if err != nil {
		pm.mu.Unlock()
		return fmt.Errorf("failed to create log file: %w", err)
	}

	// Verify working directory exists
	if _, err := os.Stat(workDir); os.IsNotExist(err) {
		logF.Close()
		pm.mu.Unlock()
		return fmt.Errorf("working directory does not exist: %s", workDir)
	}

	// Add LOG_FILE to environment variables
	env = append(env, fmt.Sprintf("LOG_FILE=%s", logFile))

	// Create the command
	cmd := exec.Command(args[0], args[1:]...)
	cmd.Dir = workDir // Critical: processes must run in their project directory (e.g., ~/.llamafarm/src/server)
	cmd.Env = env

	utils.LogDebug(fmt.Sprintf("Starting %s in directory: %s with LOG_FILE=%s\n", name, workDir, logFile))

	// Set up pipes for stdout and stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		logF.Close()
		pm.mu.Unlock()
		return fmt.Errorf("failed to create stdout pipe: %w", err)
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		logF.Close()
		pm.mu.Unlock()
		return fmt.Errorf("failed to create stderr pipe: %w", err)
	}

	// Start the process
	if err := cmd.Start(); err != nil {
		logF.Close()
		pm.mu.Unlock()
		return fmt.Errorf("failed to start process: %w", err)
	}

	// Store process info
	procInfo := &ProcessInfo{
		Name:      name,
		Cmd:       cmd,
		LogFile:   logFile,
		StartTime: time.Now(),
		Status:    "running",
		PID:       cmd.Process.Pid, // Store PID from started process
	}
	pm.processes[name] = procInfo

	// Start goroutines to capture output
	go pm.captureOutput(name, stdout, logF, "stdout")
	go pm.captureOutput(name, stderr, logF, "stderr")

	// Start goroutine to monitor process
	go pm.monitorProcess(name, cmd, logF)

	// After launching, check that the process is still running.
	time.Sleep(2 * time.Second)

	// Check if process is still running using cross-platform check
	if !isProcessAlive(cmd.Process.Pid) {
		// Clean up state if process failed to start
		delete(pm.processes, name)
		pm.mu.Unlock()
		return fmt.Errorf("%s process failed to start or crashed immediately. (run `lf services logs -s %s` to view logs)", name, name)
	}

	utils.LogDebug(fmt.Sprintf("%s process started (PID: %d)\n", name, cmd.Process.Pid))

	// Release the mutex before waiting for PID file - we're done modifying pm.processes
	pm.mu.Unlock()

	// Wait for the Python process to write its PID file before releasing the service lock
	// This prevents race conditions where another CLI instance could start a duplicate
	// process before the first one has registered itself via PID file.
	if err := pm.waitForPIDFile(name, PIDFileWaitTimeout); err != nil {
		// Process started but didn't write PID file in time - log warning but don't fail
		// The process might still be initializing
		utils.LogDebug(fmt.Sprintf("Warning: %v (process may still be initializing)", err))
	}

	return nil
}

// captureOutput captures output from a pipe and writes to log file
// Uses buffered writing with periodic flushing for performance and reliability
func (pm *ProcessManager) captureOutput(name string, reader io.Reader, logFile *os.File, streamName string) {
	// Create buffered writer for efficient disk I/O
	writer := bufio.NewWriter(logFile)
	defer func() {
		// Ensure final flush on exit
		writer.Flush()
		logFile.Sync()
	}()

	scanner := bufio.NewScanner(reader)
	lineCount := 0
	for scanner.Scan() {
		line := scanner.Text()

		// Write to buffered writer
		timestamp := time.Now().Format("2006-01-02 15:04:05")
		logLine := fmt.Sprintf("[%s] [%s] %s\n", timestamp, streamName, line)

		writer.WriteString(logLine)

		// Sync every 10 lines to ensure writes are flushed periodically
		// This prevents log loss if the process is killed abruptly
		lineCount++
		if lineCount%10 == 0 {
			logFile.Sync()
		}

		// Optionally write to debug output
		utils.LogDebug(fmt.Sprintf("[%s] %s\n", name, line))
	}

	// Final sync before goroutine exits to ensure all writes are flushed
	if logFile != nil {
		logFile.Sync()
	}
}

// monitorProcess monitors a process and updates its status
func (pm *ProcessManager) monitorProcess(name string, cmd *exec.Cmd, logFile *os.File) {
	if logFile != nil {
		defer func() {
			// Sync before closing to ensure all buffered writes are flushed to disk
			logFile.Sync()
			logFile.Close()
		}()
	}

	err := cmd.Wait()

	pm.mu.Lock()
	defer pm.mu.Unlock()

	if proc, exists := pm.processes[name]; exists {
		proc.mu.Lock()
		if err != nil {
			proc.Status = fmt.Sprintf("exited with error: %v", err)
			utils.OutputError("%s process exited with error: %v\n", name, err)

			// Output last 20 lines from the log file
			if logs, logErr := pm.GetProcessLogs(name, 20); logErr == nil && len(logs) > 0 {
				utils.OutputError("\nLast 20 lines from %s log:\n", name)
				for _, line := range logs {
					fmt.Fprintf(os.Stderr, "%s\n", line)
				}
				utils.OutputError("\n")
			}
		} else {
			proc.Status = "exited normally"
			utils.LogDebug(fmt.Sprintf("%s process exited normally\n", name))
		}
		proc.mu.Unlock()
	}
}

// StopProcess stops a running process gracefully
func (pm *ProcessManager) StopProcess(name string) error {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	// Look for process in memory or via PID file
	proc, found := pm.findProcess(name)
	if !found {
		return fmt.Errorf("process %s not found (no tracked process or PID file)", name)
	}

	// Try graceful shutdown first
	if err := proc.Cmd.Process.Signal(os.Interrupt); err != nil {
		// If that fails, kill it
		if err := proc.Cmd.Process.Kill(); err != nil {
			return fmt.Errorf("failed to kill process: %w", err)
		}
	}

	// Poll for process exit instead of calling Wait()
	// This avoids two critical bugs:
	// 1. Double Wait() - monitorProcess() already called Wait() for native processes
	// 2. Invalid Wait() - orphaned processes have synthetic Cmd that was never Start()ed
	deadline := time.Now().Add(5 * time.Second)
	pollInterval := 100 * time.Millisecond

	for time.Now().Before(deadline) {
		// Check if process is still running using signal 0 (works for both native and orphaned)
		if !pm.isProcessRunning(proc) {
			utils.OutputProgress("%s process stopped\n", name)
			proc.mu.Lock()
			proc.Status = "stopped"
			proc.mu.Unlock()
			return nil
		}
		time.Sleep(pollInterval)
	}

	// Force kill if graceful shutdown times out
	utils.LogDebug(fmt.Sprintf("Graceful shutdown timeout for %s, forcing kill", name))
	proc.Cmd.Process.Kill()

	// Give it a moment to die after kill
	time.Sleep(500 * time.Millisecond)

	utils.OutputWarning("%s process killed after timeout\n", name)
	proc.mu.Lock()
	proc.Status = "stopped"
	proc.mu.Unlock()

	return nil
}

// KnownServiceNames lists all service names that might have PID files.
// This is used by StopAllProcesses to stop orphaned processes from previous CLI invocations.
var KnownServiceNames = []string{"server", "rag", "universal-runtime"}

// StopAllProcesses stops all managed processes, including orphaned ones from PID files.
// This is important during source code upgrades when services may have been started
// by a different CLI invocation.
func (pm *ProcessManager) StopAllProcesses() {
	// Collect names from in-memory tracking
	pm.mu.RLock()
	names := make(map[string]bool)
	for name := range pm.processes {
		names[name] = true
	}
	pm.mu.RUnlock()

	// Also check for orphaned processes via PID files
	// This handles services started by a different CLI invocation
	for _, name := range KnownServiceNames {
		if !names[name] {
			if pid, found := pm.ReadPIDFile(name); found && isProcessAlive(pid) {
				names[name] = true
				utils.LogDebug(fmt.Sprintf("Found orphaned process %s (PID %d) via PID file", name, pid))
			}
		}
	}

	// Stop all found processes
	for name := range names {
		if err := pm.StopProcess(name); err != nil {
			utils.LogDebug(fmt.Sprintf("Error stopping %s: %v\n", name, err))
		}
	}
}

// isProcessRunning checks if a process is still running
func (pm *ProcessManager) isProcessRunning(proc *ProcessInfo) bool {
	if proc.Cmd == nil || proc.Cmd.Process == nil {
		return false
	}
	return isProcessAlive(proc.Cmd.Process.Pid)
}

// GetProcessStatus returns the status of a process
func (pm *ProcessManager) GetProcessStatus(name string) (string, error) {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	// Look for process in memory or via PID file
	proc, found := pm.findProcess(name)
	if !found {
		return "", fmt.Errorf("process %s not found", name)
	}

	proc.mu.RLock()
	defer proc.mu.RUnlock()

	return proc.Status, nil
}

// GetProcessLogs returns recent logs for a process
func (pm *ProcessManager) GetProcessLogs(name string, lines int) ([]string, error) {
	pm.mu.RLock()
	// Look for process in memory or via PID file
	proc, found := pm.findProcess(name)
	pm.mu.RUnlock()

	if !found {
		return nil, fmt.Errorf("process %s not found", name)
	}

	// Read log file
	file, err := os.Open(proc.LogFile)
	if err != nil {
		return nil, fmt.Errorf("failed to open log file: %w", err)
	}
	defer file.Close()

	// Read all lines
	var allLines []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		allLines = append(allLines, scanner.Text())
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("failed to read log file: %w", err)
	}

	// Return last N lines
	if len(allLines) <= lines {
		return allLines, nil
	}

	return allLines[len(allLines)-lines:], nil
}

// IsProcessHealthy checks if a process is running and healthy
func (pm *ProcessManager) IsProcessHealthy(name string) bool {
	status, err := pm.GetProcessStatus(name)
	if err != nil {
		return false
	}
	return status == "running"
}

// WaitForProcess waits for a process to exit
func (pm *ProcessManager) WaitForProcess(name string) error {
	pm.mu.RLock()
	proc, found := pm.findProcess(name)
	pm.mu.RUnlock()

	if !found {
		return fmt.Errorf("process %s not found", name)
	}

	if proc.Cmd == nil {
		return fmt.Errorf("process %s has no command", name)
	}

	return proc.Cmd.Wait()
}

// GetProcessInfo returns information about a process
func (pm *ProcessManager) GetProcessInfo(name string, healthPayload *HealthPayload) (*ProcessInfo, error) {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	proc, exists := pm.processes[name]
	if !exists {
		return nil, fmt.Errorf("process %s not found", name)
	}

	return proc, nil
}

func (pm *ProcessManager) ReadPIDFile(serviceName string) (int, bool) {
	pidFile := pm.getPIDFilePath(serviceName)
	data, err := os.ReadFile(pidFile)
	if err != nil {
		return 0, false
	}

	var pid int
	_, err = fmt.Sscanf(string(data), "%d", &pid)
	if err != nil {
		return 0, false
	}

	return pid, true
}

// findProcess looks for a process by name, checking both in-memory tracking and PID files.
// It returns the ProcessInfo and a boolean indicating if the process was found.
// For PID file-based processes, it verifies the process is actually running before returning.
func (pm *ProcessManager) findProcess(name string) (*ProcessInfo, bool) {
	// First check in-memory tracking
	if proc, exists := pm.processes[name]; exists {
		return proc, true
	}

	// Check for PID file
	pid, found := pm.ReadPIDFile(name)
	if !found {
		return nil, false
	}

	// Verify the process is actually running using cross-platform check
	if !isProcessAlive(pid) {
		return nil, false
	}

	// Get a process handle for the ProcessInfo struct
	process, err := os.FindProcess(pid)
	if err != nil {
		return nil, false
	}

	// Infer the log file path
	logFile := filepath.Join(pm.logsDir, fmt.Sprintf("%s.log", name))

	// Return a temporary ProcessInfo for this PID
	// Note: This is not added to pm.processes
	return &ProcessInfo{
		Name:    name,
		Cmd:     &exec.Cmd{Process: process},
		PID:     pid,
		Status:  "running",
		LogFile: logFile,
	}, true
}

// getPIDFilePath returns the path to a service's PID file
func (pm *ProcessManager) getPIDFilePath(serviceName string) string {
	return filepath.Join(pm.pidsDir, fmt.Sprintf("%s.pid", serviceName))
}

// getLockFilePath returns the path to a service's lock file
func (pm *ProcessManager) getLockFilePath(serviceName string) string {
	return filepath.Join(pm.pidsDir, fmt.Sprintf("%s.lock", serviceName))
}

// removePIDFile removes a service's PID file
func (pm *ProcessManager) removePIDFile(serviceName string) {
	pidFile := pm.getPIDFilePath(serviceName)
	if err := os.Remove(pidFile); err != nil && !os.IsNotExist(err) {
		utils.LogDebug(fmt.Sprintf("Failed to remove PID file %s: %v", pidFile, err))
	}
}

// waitForPIDFile waits for a PID file to be written and the process to be running
// This is used after starting a process to ensure the Python service has written its PID
func (pm *ProcessManager) waitForPIDFile(serviceName string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)

	for time.Now().Before(deadline) {
		if pid, found := pm.ReadPIDFile(serviceName); found {
			if isProcessAlive(pid) {
				utils.LogDebug(fmt.Sprintf("PID file for %s found (PID %d)", serviceName, pid))
				return nil
			}
		}
		time.Sleep(PIDFilePollInterval)
	}

	return fmt.Errorf("timeout waiting for PID file for %s", serviceName)
}
