package orchestrator

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sync"
	"syscall"
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

// ProcessManager manages native processes for services
type ProcessManager struct {
	homeDir   string
	logsDir   string
	processes map[string]*ProcessInfo
	mu        sync.RWMutex
}

// NewProcessManager creates a new process manager
func NewProcessManager() (*ProcessManager, error) {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return nil, fmt.Errorf("failed to get user home directory: %w", err)
	}

	logsDir := filepath.Join(homeDir, ".llamafarm", "logs")
	if err := os.MkdirAll(logsDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create logs directory: %w", err)
	}

	return &ProcessManager{
		homeDir:   homeDir,
		logsDir:   logsDir,
		processes: make(map[string]*ProcessInfo),
	}, nil
}

// StartProcess starts a new process and manages its lifecycle
func (pm *ProcessManager) StartProcess(name string, workDir string, env []string, args ...string) error {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	// Check if process is already running
	if proc, exists := pm.processes[name]; exists {
		if pm.isProcessRunning(proc) {
			return fmt.Errorf("process %s is already running", name)
		}
		// Clean up old process info
		delete(pm.processes, name)
	}

	// Create log file
	logFile := filepath.Join(pm.logsDir, fmt.Sprintf("%s.log", name))
	logF, err := os.Create(logFile)
	if err != nil {
		return fmt.Errorf("failed to create log file: %w", err)
	}

	// Verify working directory exists
	if _, err := os.Stat(workDir); os.IsNotExist(err) {
		logF.Close()
		return fmt.Errorf("working directory does not exist: %s", workDir)
	}

	// Create the command
	cmd := exec.Command(args[0], args[1:]...)
	cmd.Dir = workDir // Critical: processes must run in their project directory (e.g., ~/.llamafarm/src/server)
	cmd.Env = env

	utils.LogDebug(fmt.Sprintf("Starting %s in directory: %s\n", name, workDir))

	// Set up pipes for stdout and stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		logF.Close()
		return fmt.Errorf("failed to create stdout pipe: %w", err)
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		logF.Close()
		return fmt.Errorf("failed to create stderr pipe: %w", err)
	}

	// Start the process
	if err := cmd.Start(); err != nil {
		logF.Close()
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

	// Check if process is still running using platform-appropriate method
	processStillRunning := true
	if runtime.GOOS == "windows" {
		// On Windows, check if the process has exited
		processStillRunning = cmd.ProcessState == nil || !cmd.ProcessState.Exited()
	} else {
		// On Unix, use signal 0 to check if process exists
		err := cmd.Process.Signal(syscall.Signal(0))
		processStillRunning = err == nil
	}

	if !processStillRunning {
		// Clean up state if process failed to start
		logF.Close()
		// Note: pm.mu is already locked from line 55, no need to lock again
		delete(pm.processes, name)
		return fmt.Errorf("%s process failed to start or crashed immediately (see logs at %s)", name, logFile)
	}

	utils.LogDebug(fmt.Sprintf("%s process started (PID: %d)\n", name, cmd.Process.Pid))
	return nil
}

// captureOutput captures output from a pipe and writes to log file
func (pm *ProcessManager) captureOutput(name string, reader io.Reader, logFile *os.File, streamName string) {
	scanner := bufio.NewScanner(reader)
	lineCount := 0
	for scanner.Scan() {
		line := scanner.Text()

		// Write to log file
		timestamp := time.Now().Format("2006-01-02 15:04:05")
		logLine := fmt.Sprintf("[%s] [%s] %s\n", timestamp, streamName, line)
		logFile.WriteString(logLine)

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

	// Wait for process to exit (with timeout)
	done := make(chan error, 1)
	go func() {
		done <- proc.Cmd.Wait()
	}()

	select {
	case <-done:
		utils.OutputProgress("%s process stopped\n", name)
	case <-time.After(5 * time.Second):
		// Force kill if graceful shutdown times out
		proc.Cmd.Process.Kill()
		utils.OutputWarning("%s process killed after timeout\n", name)
	}

	proc.mu.Lock()
	proc.Status = "stopped"
	proc.mu.Unlock()

	return nil
}

// StopAllProcesses stops all managed processes
func (pm *ProcessManager) StopAllProcesses() {
	pm.mu.RLock()
	names := make([]string, 0, len(pm.processes))
	for name := range pm.processes {
		names = append(names, name)
	}
	pm.mu.RUnlock()

	for _, name := range names {
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

	// On Unix, sending signal 0 checks if a process exists.
	// On Windows, this is not supported, so we check if the process has exited.
	if runtime.GOOS == "windows" {
		return proc.Cmd.ProcessState == nil || !proc.Cmd.ProcessState.Exited()
	}

	err := proc.Cmd.Process.Signal(os.Signal(nil))
	return err == nil
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
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return 0, false
	}

	pidFile := filepath.Join(homeDir, ".llamafarm", "pids", fmt.Sprintf("%s.pid", serviceName))
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

	// Try to find the process by PID
	process, err := os.FindProcess(pid)
	if err != nil {
		return nil, false
	}

	// Verify the process is actually running
	if err := process.Signal(syscall.Signal(0)); err != nil {
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
