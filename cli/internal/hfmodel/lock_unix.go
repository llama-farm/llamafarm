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
// drop the lock.
//
// IMPORTANT: the lock file is intentionally NOT removed on release. Removing
// it would change the inode that future waiters end up flocking, allowing
// two processes to believe they hold the same logical lock simultaneously
// (process A creates inode 1 + flocks; A unlinks; B creates inode 2 + flocks
// successfully because it's a fresh file; meanwhile a third process C may
// still hold an fd open against inode 1). The Python `filelock` package and
// huggingface_hub take the same precaution. The cost of leaving stale
// `.lock` files in the cache is negligible compared to silent blob
// corruption.
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
		// Lock file is intentionally NOT unlinked — see function comment.
	}
	return release, nil
}
