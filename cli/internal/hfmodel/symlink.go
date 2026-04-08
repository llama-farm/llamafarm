package hfmodel

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
)

// CreateSnapshotSymlink creates a symbolic link from `dst` to `src`, falling
// back to a file move (for new blobs) or copy (for existing blobs) on
// filesystems where symlinks are not supported.
//
// Mirrors huggingface_hub.file_download._create_symlink:
//
//  1. Try a relative symlink first. Relative symlinks survive cache
//     directory moves and have fewer Windows quirks.
//  2. If symlinks are not supported on the destination filesystem, move the
//     blob into the snapshot path (no wasted disk for a fresh download).
//
// Unlike the Python implementation, this Go port does not branch on the
// `new_blob` flag — by the time we call this, the blob has already been
// moved into place under blobs/<etag>, so we always know we own it. The
// "copy instead of move" branch is only relevant when an existing blob is
// shared by multiple snapshots, which we handle by reading the symlink
// support state and choosing copy in that case.
func CreateSnapshotSymlink(src, dst string) error {
	// Best-effort cleanup: remove any pre-existing entry at dst so we can
	// create cleanly. A FileExists race here would be a concurrent
	// downloader's write — but the lock above us serializes those, so this
	// remove is safe.
	if err := os.Remove(dst); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("remove existing dst: %w", err)
	}

	absSrc, err := filepath.Abs(src)
	if err != nil {
		return err
	}
	absDst, err := filepath.Abs(dst)
	if err != nil {
		return err
	}
	dstFolder := filepath.Dir(absDst)

	// Probe symlink support per cache root (cached after first probe).
	if symlinksSupported(dstFolder) {
		relSrc, err := filepath.Rel(dstFolder, absSrc)
		if err != nil {
			// Different volumes — use absolute path. Falls back to copy
			// below if symlink itself fails.
			relSrc = absSrc
		}
		if err := os.Symlink(relSrc, absDst); err == nil {
			return nil
		}
		// Symlink probe said yes but creation failed — fall through to copy.
	}

	// Symlinks not supported. Copy from src to dst. We can't use os.Rename
	// here because the source blob may be referenced by other snapshots
	// (sharing optimization), so a copy is the correct semantic.
	return copyFile(absSrc, absDst)
}

// symlinkSupportCache caches the result of probing each directory for
// symlink support. The map key is the absolute directory path.
var (
	symlinkSupportMu    sync.Mutex
	symlinkSupportCache = map[string]bool{}
)

// symlinksSupported probes whether the given directory supports creating
// symbolic links by attempting to create and remove a tiny test symlink.
// Result is cached per directory for the process lifetime.
func symlinksSupported(dir string) bool {
	symlinkSupportMu.Lock()
	defer symlinkSupportMu.Unlock()
	if v, ok := symlinkSupportCache[dir]; ok {
		return v
	}
	// Make sure the dir exists; if it does not yet, the symlink test would
	// fail for the wrong reason.
	if err := os.MkdirAll(dir, 0o755); err != nil {
		symlinkSupportCache[dir] = false
		return false
	}
	probeName := filepath.Join(dir, ".lf-symlink-probe")
	_ = os.Remove(probeName)
	if err := os.Symlink(".", probeName); err != nil {
		symlinkSupportCache[dir] = false
		return false
	}
	_ = os.Remove(probeName)
	symlinkSupportCache[dir] = true
	return true
}

// copyFile copies a file byte-for-byte from src to dst, preserving content
// (not mode/ownership). Used as the symlink fallback.
func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	tmp := dst + ".tmp"
	out, err := os.OpenFile(tmp, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		_ = out.Close()
		_ = os.Remove(tmp)
		return err
	}
	if err := out.Close(); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, dst)
}
