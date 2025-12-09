//go:build windows

package orchestrator

import (
	"errors"

	"golang.org/x/sys/windows"
)

// isProcessAlive checks if a process with the given PID is currently running.
// On Windows, os.FindProcess always succeeds even for non-existent PIDs,
// so we need to use OpenProcess to verify the process actually exists.
func isProcessAlive(pid int) bool {
	// PROCESS_QUERY_LIMITED_INFORMATION (0x1000) is the minimum access right
	// needed to query if a process exists. This works even for processes
	// running as other users.
	const PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

	handle, err := windows.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, uint32(pid))
	if err != nil {
		// ERROR_ACCESS_DENIED means the process exists but is protected
		// (e.g., antimalware services, system-critical processes).
		// We still consider it alive for consistency with Unix EPERM handling.
		if errors.Is(err, windows.ERROR_ACCESS_DENIED) {
			return true
		}
		// Other errors (ERROR_INVALID_PARAMETER for non-existent PID) mean process doesn't exist
		return false
	}

	// Process exists, close the handle
	windows.CloseHandle(handle)
	return true
}
