package utils

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"syscall"
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

// MoveFile moves a file from src to dst, handling cross-filesystem moves.
// It first attempts os.Rename (fast path for same filesystem), then falls back
// to copy + delete when a cross-device link error is detected.
func MoveFile(src, dst string) error {
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
