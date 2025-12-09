//go:build !windows

package orchestrator

import (
	"errors"
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
	// - Returns ESRCH if process doesn't exist
	// - Returns EPERM if process exists but we lack permission (different user)
	err = process.Signal(syscall.Signal(0))

	// EPERM means the process exists but is owned by another user.
	// We still consider it alive to prevent duplicate service starts.
	return err == nil || errors.Is(err, syscall.EPERM)
}
