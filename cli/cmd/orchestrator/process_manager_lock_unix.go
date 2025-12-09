//go:build !windows

package orchestrator

import (
	"fmt"
	"os"
	"syscall"
	"time"

	"github.com/llamafarm/cli/cmd/utils"
)

// acquireServiceLock acquires an exclusive lock on a service's lock file.
// This prevents multiple CLI instances from starting the same service simultaneously.
// On Unix systems, this uses flock() for advisory file locking.
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
		// Try non-blocking lock
		err := syscall.Flock(int(lockFile.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
		if err == nil {
			// Lock acquired successfully
			utils.LogDebug(fmt.Sprintf("Acquired service lock for %s", serviceName))
			return lockFile, nil
		}

		// Only retry if the lock is held by another process (EWOULDBLOCK/EAGAIN).
		// Other errors (EBADF, etc.) should fail immediately.
		if err != syscall.EWOULDBLOCK {
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

	// Release the lock
	if err := syscall.Flock(int(lockFile.Fd()), syscall.LOCK_UN); err != nil {
		utils.LogDebug(fmt.Sprintf("Failed to release lock: %v", err))
	}

	// Close the file
	lockFile.Close()
}
