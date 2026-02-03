package utils

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"
)

// GetLFDataDir returns the directory to store LlamaFarm data.
func GetLFDataDir() (string, error) {
	dataDir := os.Getenv("LF_DATA_DIR")
	if dataDir != "" {
		return dataDir, nil
	}
	if homeDir, err := os.UserHomeDir(); err == nil {
		return filepath.Join(homeDir, ".llamafarm"), nil
	} else {
		return "", fmt.Errorf("getLFDataDir: could not determine home directory: %w", err)
	}
}

// GetProjectsRoot returns the root directory where local project configs are stored.
func GetProjectsRoot() (string, error) {
	lfDir, err := GetLFDataDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(lfDir, "projects"), nil
}

// MoveFile moves a file from src to dst, handling cross-filesystem moves.
// It first attempts os.Rename (fast path for same filesystem), then falls back
// to copy + delete when a cross-device link error is detected.
func MoveFile(src, dst string) error {
	// On Windows, os.Rename fails if destination exists. Remove it first.
	// On Unix, os.Rename atomically replaces the destination, so no removal needed.
	if runtime.GOOS == "windows" {
		if _, err := os.Stat(dst); err == nil {
			if err := os.Remove(dst); err != nil {
				return fmt.Errorf("failed to remove existing file at %s: %w", dst, err)
			}
		}
	}

	err := os.Rename(src, dst)
	if err == nil {
		return nil
	}

	// Check if it's a cross-device link error
	if !isCrossDeviceError(err) {
		return err
	}

	// Fall back to copy + delete
	if err := CopyFile(src, dst); err != nil {
		return err
	}

	// Remove source file after successful copy
	return os.Remove(src)
}

// MoveDir moves a directory from src to dst, handling cross-filesystem moves.
// It first attempts os.Rename (fast path for same filesystem), then falls back
// to recursive copy + delete when a cross-device link error is detected.
func MoveDir(src, dst string) error {
	// On Windows, os.Rename fails if destination exists. Remove it first.
	// On Unix, os.Rename atomically replaces the destination, so no removal needed.
	if runtime.GOOS == "windows" {
		if stat, err := os.Stat(dst); err == nil && stat.IsDir() {
			if err := os.RemoveAll(dst); err != nil {
				return fmt.Errorf("failed to remove existing directory at %s: %w", dst, err)
			}
		}
	}

	err := os.Rename(src, dst)
	if err == nil {
		return nil
	}

	// Check if it's a cross-device link error
	if !isCrossDeviceError(err) {
		return err
	}

	// Fall back to recursive copy + delete
	if err := copyDir(src, dst); err != nil {
		return err
	}

	// Remove source directory after successful copy
	return os.RemoveAll(src)
}

// isCrossDeviceError checks if an error is a cross-device link error (EXDEV).
// This occurs when trying to rename/move a file across different filesystems.
func isCrossDeviceError(err error) bool {
	var linkErr *os.LinkError
	if errors.As(err, &linkErr) {
		return errors.Is(linkErr.Err, syscall.EXDEV)
	}
	return false
}

// CopyFile copies a single file from src to dst, preserving permissions.
func CopyFile(src, dst string) error {
	srcInfo, err := os.Stat(src)
	if err != nil {
		return fmt.Errorf("failed to stat source file: %w", err)
	}

	srcFile, err := os.Open(src)
	if err != nil {
		return fmt.Errorf("failed to open source file: %w", err)
	}
	defer srcFile.Close()

	dstFile, err := os.OpenFile(dst, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, srcInfo.Mode())
	if err != nil {
		return fmt.Errorf("failed to create destination file: %w", err)
	}

	if _, err := io.Copy(dstFile, srcFile); err != nil {
		dstFile.Close()
		os.Remove(dst) // Clean up partial file
		return fmt.Errorf("failed to copy file contents: %w", err)
	}

	// Explicitly close and check error - Close() can fail if data cannot be flushed to disk
	if err := dstFile.Close(); err != nil {
		os.Remove(dst) // Clean up potentially incomplete file
		return fmt.Errorf("failed to finalize destination file: %w", err)
	}

	return nil
}

// copyDir recursively copies a directory from src to dst, preserving permissions.
func copyDir(src, dst string) error {
	srcInfo, err := os.Stat(src)
	if err != nil {
		return fmt.Errorf("failed to stat source directory: %w", err)
	}

	// Create destination directory with same permissions
	if err := os.MkdirAll(dst, srcInfo.Mode()); err != nil {
		return fmt.Errorf("failed to create destination directory: %w", err)
	}

	entries, err := os.ReadDir(src)
	if err != nil {
		return fmt.Errorf("failed to read source directory: %w", err)
	}

	for _, entry := range entries {
		srcPath := filepath.Join(src, entry.Name())
		dstPath := filepath.Join(dst, entry.Name())

		if entry.IsDir() {
			if err := copyDir(srcPath, dstPath); err != nil {
				return err
			}
		} else {
			if err := CopyFile(srcPath, dstPath); err != nil {
				return err
			}
		}
	}

	return nil
}

// RemoveAllWithRetry removes a directory and all its contents with retry logic.
// On Windows, file handles may be held briefly after a process terminates,
// causing "Access is denied" errors. This function retries with delays to
// handle such cases.
func RemoveAllWithRetry(path string) error {
	// Check if path exists
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return nil // Nothing to remove
	}

	// On non-Windows platforms, just use os.RemoveAll directly
	if runtime.GOOS != "windows" {
		return os.RemoveAll(path)
	}

	// Windows-specific: retry with delays to handle locked files
	// This handles the case where Python processes have just terminated
	// but Windows hasn't yet released file handles
	maxRetries := 5
	retryDelay := 1 * time.Second

	var lastErr error
	for i := 0; i < maxRetries; i++ {
		if i > 0 {
			LogDebug(fmt.Sprintf("Retrying directory removal (attempt %d/%d) after delay...", i+1, maxRetries))
			time.Sleep(retryDelay)
			// Increase delay for subsequent retries
			retryDelay = retryDelay * 2
			if retryDelay > 5*time.Second {
				retryDelay = 5 * time.Second
			}
		}

		lastErr = os.RemoveAll(path)
		if lastErr == nil {
			return nil // Success
		}

		// Check if it's an access denied error (common on Windows with locked files)
		if !isAccessDeniedError(lastErr) {
			return lastErr // Different error, don't retry
		}

		LogDebug(fmt.Sprintf("Directory removal failed (access denied), will retry: %v", lastErr))
	}

	return fmt.Errorf("failed to remove directory after %d attempts: %w", maxRetries, lastErr)
}

// isAccessDeniedError checks if an error is an access denied error.
// On Windows, this typically happens when a file is still locked by another process.
func isAccessDeniedError(err error) bool {
	if err == nil {
		return false
	}

	// Check for Windows-specific error codes
	var pathErr *os.PathError
	if errors.As(err, &pathErr) {
		// Windows ERROR_ACCESS_DENIED = 5, ERROR_SHARING_VIOLATION = 32
		if errno, ok := pathErr.Err.(syscall.Errno); ok {
			return errno == 5 || errno == 32
		}
	}

	// Also check error message as fallback
	errStr := err.Error()
	return errors.Is(err, os.ErrPermission) ||
		strings.Contains(errStr, "Access is denied") ||
		strings.Contains(errStr, "being used by another process")
}
