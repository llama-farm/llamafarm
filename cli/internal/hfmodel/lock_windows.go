//go:build windows

package hfmodel

import (
	"fmt"
	"os"
	"path/filepath"

	"golang.org/x/sys/windows"
)

// acquireLock takes an exclusive advisory lock on the given lock file path.
// Uses LockFileEx with LOCKFILE_EXCLUSIVE_LOCK, which is the Windows
// equivalent of flock(LOCK_EX). Compatible with huggingface_hub.filelock,
// which calls the same Win32 API on Windows.
//
// IMPORTANT: the lock file is intentionally NOT removed on release. See the
// extended comment on the unix sibling — same race applies on Windows
// (different handle/identity if the file is recreated between waiters).
func acquireLock(lockPath string) (func(), error) {
	if err := os.MkdirAll(filepath.Dir(lockPath), 0o755); err != nil {
		return nil, fmt.Errorf("create lock dir: %w", err)
	}

	f, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return nil, fmt.Errorf("open lock file: %w", err)
	}

	handle := windows.Handle(f.Fd())
	var ol windows.Overlapped
	// Lock the maximum range. LOCKFILE_EXCLUSIVE_LOCK | wait (no
	// LOCKFILE_FAIL_IMMEDIATELY) blocks until the lock is acquired.
	if err := windows.LockFileEx(handle, windows.LOCKFILE_EXCLUSIVE_LOCK, 0, ^uint32(0), ^uint32(0), &ol); err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("LockFileEx: %w", err)
	}

	release := func() {
		var ol2 windows.Overlapped
		_ = windows.UnlockFileEx(handle, 0, ^uint32(0), ^uint32(0), &ol2)
		_ = f.Close()
		// Lock file is intentionally NOT unlinked — see function comment.
	}
	return release, nil
}
