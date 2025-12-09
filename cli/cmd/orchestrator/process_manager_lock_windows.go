//go:build windows

package orchestrator

import (
	"fmt"
	"os"
	"time"

	"github.com/llamafarm/cli/cmd/utils"
	"golang.org/x/sys/windows"
)

// acquireServiceLock acquires an exclusive lock on a service's lock file.
// This prevents multiple CLI instances from starting the same service simultaneously.
// On Windows, this uses LockFileEx() for file locking.
//
// The lock is non-blocking with a timeout - if another process holds the lock,
// we'll retry until ServiceLockTimeout before giving up.
func (pm *ProcessManager) acquireServiceLock(serviceName string) (*os.File, error) {
	lockPath := pm.getLockFilePath(serviceName)

	// Open or create the lock file
	lockFile, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		return nil, fmt.Errorf("failed to open lock file %s: %w", lockPath, err)
	}

	// Try to acquire lock with timeout
	deadline := time.Now().Add(ServiceLockTimeout)

	for {
		// Try non-blocking exclusive lock using Windows API
		// LOCKFILE_EXCLUSIVE_LOCK = 0x02
		// LOCKFILE_FAIL_IMMEDIATELY = 0x01
		overlapped := &windows.Overlapped{}
		err := windows.LockFileEx(
			windows.Handle(lockFile.Fd()),
			windows.LOCKFILE_EXCLUSIVE_LOCK|windows.LOCKFILE_FAIL_IMMEDIATELY,
			0,
			1, // Lock 1 byte
			0,
			overlapped,
		)
		if err == nil {
			// Lock acquired successfully
			utils.LogDebug(fmt.Sprintf("Acquired service lock for %s", serviceName))
			return lockFile, nil
		}

		// Only retry if the lock is held by another process.
		// Other errors (invalid handle, I/O errors) should fail immediately.
		if err != windows.ERROR_LOCK_VIOLATION {
			lockFile.Close()
			return nil, fmt.Errorf("failed to acquire lock on %s: %w", serviceName, err)
		}

		// Check if we've exceeded the deadline
		if time.Now().After(deadline) {
			lockFile.Close()
			return nil, fmt.Errorf("timeout waiting for service lock on %s (another process may be starting it)", serviceName)
		}

		// Lock is held by another process, wait and retry
		utils.LogDebug(fmt.Sprintf("Waiting for service lock on %s...", serviceName))
		time.Sleep(ServiceLockPollInterval)
	}
}

// releaseServiceLock releases a service lock and closes the lock file.
func (pm *ProcessManager) releaseServiceLock(lockFile *os.File) {
	if lockFile == nil {
		return
	}

	// Release the lock using Windows API
	overlapped := &windows.Overlapped{}
	if err := windows.UnlockFileEx(
		windows.Handle(lockFile.Fd()),
		0,
		1, // Unlock 1 byte
		0,
		overlapped,
	); err != nil {
		utils.LogDebug(fmt.Sprintf("Failed to release lock: %v", err))
	}

	// Close the file
	lockFile.Close()
}
