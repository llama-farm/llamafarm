//go:build !windows

package hfmodel

import (
	"fmt"
	"os"
	"path/filepath"

	"golang.org/x/sys/unix"
)

// acquireLock takes an exclusive advisory lock on the given lock file path.
// Compatible with huggingface_hub.filelock — both implementations use
// flock(2) under the hood, so a Go writer and a Python writer racing for
// the same blob will serialize cleanly.
//
// Returns a release function that MUST be called (typically via defer) to
// drop the lock and remove the lock file.
func acquireLock(lockPath string) (func(), error) {
	// Make sure the directory exists.
	if err := os.MkdirAll(filepath.Dir(lockPath), 0o755); err != nil {
		return nil, fmt.Errorf("create lock dir: %w", err)
	}

	f, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return nil, fmt.Errorf("open lock file: %w", err)
	}

	// Block until we get an exclusive lock.
	if err := unix.Flock(int(f.Fd()), unix.LOCK_EX); err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("flock: %w", err)
	}

	release := func() {
		// Drop the lock first, then close the fd. Closing the fd would
		// release the lock anyway, but doing it explicitly makes failures
		// easier to attribute.
		_ = unix.Flock(int(f.Fd()), unix.LOCK_UN)
		_ = f.Close()
		// Best-effort lock file removal. Another waiter may have re-grabbed
		// the lock between our unlock and remove — in that case the remove
		// would race, but that's harmless: filelock holders tolerate the
		// file disappearing.
		_ = os.Remove(lockPath)
	}
	return release, nil
}
