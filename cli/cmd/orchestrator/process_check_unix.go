//go:build !windows

package orchestrator

import (
	"os"
	"syscall"
)

// isProcessAlive checks if a process with the given PID is currently running.
// On Unix systems, this uses signal 0 which doesn't send a signal but checks
// if the process exists and we have permission to signal it.
func isProcessAlive(pid int) bool {
	process, err := os.FindProcess(pid)
	if err != nil {
		return false
	}

	// Signal 0 doesn't actually send a signal, but does error checking:
	// - Returns nil if process exists and we can signal it
	// - Returns an error if process doesn't exist or we lack permission
	err = process.Signal(syscall.Signal(0))
	return err == nil
}
