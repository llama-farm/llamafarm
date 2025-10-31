package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// checkServiceStatusNative checks the status of a single service using native processes
func checkServiceStatusNative(serviceName string, serverURL string) ServiceInfo {
	// Get the service definition
	_, exists := ServiceGraph[serviceName]
	if !exists {
		return ServiceInfo{
			Name:          serviceName,
			State:         "unknown",
			Orchestration: "native",
		}
	}

	status := ServiceInfo{
		Name:          serviceName,
		State:         "not_found",
		Ports:         make(map[string]string),
		Orchestration: "native",
	}

	// Get the expected log file path (even if process isn't tracked by current session)
	homeDir, err := os.UserHomeDir()
	if err == nil {
		logFile := filepath.Join(homeDir, ".llamafarm", "logs", fmt.Sprintf("%s.log", serviceName))
		if _, err := os.Stat(logFile); err == nil {
			status.LogFile = logFile
		}
	}

	// Check for PID file first - this is the primary method for native service discovery
	pid, pidFileExists := readPIDFile(serviceName)
	if pidFileExists {
		logDebug(fmt.Sprintf("Found PID file for %s with PID %d", serviceName, pid))

		processRunning := isProcessRunning(pid)
		logDebug(fmt.Sprintf("Process %d running check: %v", pid, processRunning))

		if processRunning {
			// Process is running based on PID file
			status.State = "running"
			status.PID = pid

			// Try to get start time from /proc (Linux) or ps (Unix-like)
			if startTime := getProcessStartTime(pid); !startTime.IsZero() {
				duration := time.Since(startTime)
				status.Uptime = formatDuration(duration)
			}

			// Get health status if service is running
			status.Health = getServiceHealth(serviceName, serverURL)
			return status
		} else {
			// Process not running, but be conservative - only clean up if we're very sure
			logDebug(fmt.Sprintf("Process %d not found, marking as stopped (PID file will be cleaned up by service or stop command)", pid))
			status.State = "stopped"
			// Don't automatically clean up PID file here - let the service or explicit stop command do it
			// This prevents race conditions where we check while the process is starting up
			return status
		}
	}

	// Fallback: Check if we have a global native orchestrator with active processes
	// This handles processes started by the current CLI session
	if globalNativeOrchestrator != nil {
		processMgr := globalNativeOrchestrator.GetProcessManager()
		if processMgr != nil {
			// Check process status from the current orchestrator
			processStatus, err := processMgr.GetProcessStatus(serviceName)
			if err == nil {
				// Process is tracked by current orchestrator
				if processStatus == "running" {
					status.State = "running"

					// Get process info
					processMgr.mu.RLock()
					if proc, exists := processMgr.processes[serviceName]; exists {
						if proc.Cmd != nil && proc.Cmd.Process != nil {
							status.PID = proc.Cmd.Process.Pid
						}
						status.LogFile = proc.LogFile

						// Calculate uptime
						if !proc.StartTime.IsZero() {
							duration := time.Since(proc.StartTime)
							status.Uptime = formatDuration(duration)
						}
					}
					processMgr.mu.RUnlock()

					// Get health status if service is running
					status.Health = getServiceHealth(serviceName, serverURL)
				} else {
					status.State = "stopped"
				}
				return status
			}
		}
	}

	// Final fallback: service is not found
	status.State = "not_found"
	return status
}

// readPIDFile reads the PID from a service's PID file
func readPIDFile(serviceName string) (int, bool) {
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

// isProcessRunning checks if a process with the given PID is running
func isProcessRunning(pid int) bool {
	if pid <= 0 {
		return false
	}

	process, err := os.FindProcess(pid)
	if err != nil {
		return false
	} else {
		err := process.Signal(syscall.Signal(0))
		if err == nil {
			return true
		}
	}

	return false
}

// getProcessStartTime gets the start time of a process (best effort)
func getProcessStartTime(pid int) time.Time {
	// This is platform-specific and best effort
	// On Linux, we can read from /proc/[pid]/stat and /proc/uptime
	// On macOS/BSD, we can use ps (not implemented)
	// On Windows, this is more complex (not implemented)

	if runtime.GOOS == "linux" {
		// Try to read from /proc/[pid]/stat
		statFile := fmt.Sprintf("/proc/%d/stat", pid)
		data, err := os.ReadFile(statFile)
		if err != nil {
			return time.Time{}
		}

		fields := strings.Fields(string(data))
		if len(fields) < 22 {
			return time.Time{}
		}

		// Field 22: starttime (in clock ticks since boot)
		startTicks, err := strconv.ParseUint(fields[21], 10, 64)
		if err != nil {
			return time.Time{}
		}

		// Get clock ticks per second
		// Most Linux systems use 100 Hz (USER_HZ), which is a safe default
		// For a more accurate implementation, one could use cgo to call sysconf(_SC_CLK_TCK)
		const clkTck int64 = 100

		// Get system uptime from /proc/uptime
		uptimeData, err := os.ReadFile("/proc/uptime")
		if err != nil {
			return time.Time{}
		}
		uptimeFields := strings.Fields(string(uptimeData))
		if len(uptimeFields) < 1 {
			return time.Time{}
		}
		uptimeSeconds, err := strconv.ParseFloat(uptimeFields[0], 64)
		if err != nil {
			return time.Time{}
		}

		// Calculate boot time
		now := time.Now()
		bootTime := now.Add(-time.Duration(uptimeSeconds * float64(time.Second)))

		// Calculate process start time
		startSeconds := float64(startTicks) / float64(clkTck)
		startTime := bootTime.Add(time.Duration(startSeconds * float64(time.Second)))
		return startTime
	}

	// For other platforms, process uptime is not implemented.
	// Return zero time to indicate we don't have the start time.
	return time.Time{}
}

// stopServiceBySystem stops a service by using PID files and system signals
func stopServiceBySystem(serviceName string) error {
	// Try to read PID file
	pid, pidFileExists := readPIDFile(serviceName)
	if !pidFileExists {
		return fmt.Errorf("service is not running (no PID file found)")
	}

	// Check if process is actually running
	if !isProcessRunning(pid) {
		return fmt.Errorf("service is not running (stale PID file)")
	}

	// Find the process
	process, err := os.FindProcess(pid)
	if err != nil {
		return fmt.Errorf("failed to find process %d: %w", pid, err)
	}

	// Try graceful shutdown first (SIGTERM)
	if err := process.Signal(syscall.SIGTERM); err != nil {
		// If graceful shutdown fails, try SIGKILL
		if err := process.Kill(); err != nil {
			return fmt.Errorf("failed to stop process %d: %w", pid, err)
		}
	}

	// Wait for process to stop with active polling instead of fixed sleep
	stopped := waitForCondition(func() bool {
		return !isProcessRunning(pid)
	}, 5*time.Second, 100*time.Millisecond)

	if !stopped {
		return fmt.Errorf("process %d did not stop after signal", pid)
	}

	return nil
}

// startServicesNative starts multiple services using native processes
func startServicesNative(serviceNames []string, serverURL string) {
	if debug {
		OutputProgress("Starting services with native orchestration...\n")
	}

	// Ensure native environment is set up
	orchestrator, err := ensureNativeEnvironment(serverURL)
	if err != nil {
		OutputError("Failed to set up native environment: %v\n", err)
		os.Exit(1)
	}

	// Start each service
	for _, serviceName := range serviceNames {
		OutputProgress("Starting %s...\n", serviceName)

		// Check if already running
		if orchestrator.processMgr.IsProcessHealthy(serviceName) {
			OutputProgress("%s is already running\n", serviceName)
			continue
		}

		// Start the service
		var startErr error
		switch serviceName {
		case "server":
			startErr = orchestrator.StartServerNative()
		case "rag":
			startErr = orchestrator.StartRAGNative()
		case "universal-runtime":
			startErr = orchestrator.StartUniversalRuntimeNative()
		default:
			OutputError("Unknown service: %s\n", serviceName)
			continue
		}

		if startErr != nil {
			OutputError("Failed to start %s: %v\n", serviceName, startErr)

			// Provide helpful guidance for RAG failures
			if serviceName == "rag" {
				homeDir, _ := os.UserHomeDir()
				logFile := filepath.Join(homeDir, ".llamafarm", "logs", "rag.log")
				fmt.Fprintf(os.Stderr, "\n💡 Tips for troubleshooting RAG startup:\n")
				fmt.Fprintf(os.Stderr, "  • Check the log file: %s\n", logFile)
				fmt.Fprintf(os.Stderr, "  • Ensure the server is running: lf services start server\n")
				fmt.Fprintf(os.Stderr, "  • Verify dependencies are synced: check ~/.llamafarm/src/rag/.venv\n")
				fmt.Fprintf(os.Stderr, "  • Check for Python import errors in the log file\n")
			}
			continue
		}

		OutputSuccess("%s started successfully\n", serviceName)
	}
}

// stopServicesNative stops multiple services using native processes
func stopServicesNative(serviceNames []string, serverURL string) {
	OutputProgress("Stopping services with native orchestration...\n")

	// Check if we have an active orchestrator
	if globalNativeOrchestrator == nil {
		// No active orchestrator - try to find and stop processes by checking PIDs
		OutputProgress("No active orchestrator found. Attempting to stop processes via system signals...\n")
		stopServicesNativeBySystem(serviceNames)
		return
	}

	processMgr := globalNativeOrchestrator.GetProcessManager()
	if processMgr == nil {
		OutputProgress("No process manager available. Attempting to stop processes via system signals...\n")
		stopServicesNativeBySystem(serviceNames)
		return
	}

	// Stop each service using the process manager
	for _, serviceName := range serviceNames {
		OutputProgress("Stopping %s...\n", serviceName)

		// Check if process is tracked
		if !processMgr.IsProcessHealthy(serviceName) {
			OutputProgress("%s is not running\n", serviceName)
			continue
		}

		// Stop the process
		if err := processMgr.StopProcess(serviceName); err != nil {
			OutputError("Failed to stop %s: %v\n", serviceName, err)
			continue
		}

		OutputSuccess("%s stopped successfully\n", serviceName)
	}
}

// stopServicesNativeBySystem stops native services by using PID files
func stopServicesNativeBySystem(serviceNames []string) {
	for _, serviceName := range serviceNames {
		OutputProgress("Stopping %s...\n", serviceName)

		// Stop the service
		if err := stopServiceBySystem(serviceName); err != nil {
			OutputError("Failed to stop %s: %v\n", serviceName, err)
			continue
		}

		OutputSuccess("%s stopped successfully\n", serviceName)
	}
}
