package cmd

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestGetUpgradeStrategy(t *testing.T) {
	strategy := GetUpgradeStrategy()
	if strategy == nil {
		t.Fatal("Expected non-nil upgrade strategy")
	}

	// Test that we get the right strategy type for the current platform
	switch runtime.GOOS {
	case "windows":
		if _, ok := strategy.(*WindowsUpgradeStrategy); !ok {
			t.Error("Expected WindowsUpgradeStrategy on Windows")
		}
	default:
		if _, ok := strategy.(*UnixUpgradeStrategy); !ok {
			t.Error("Expected UnixUpgradeStrategy on Unix-like systems")
		}
	}
}

func TestUnixUpgradeStrategy(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Skipping Unix tests on Windows")
	}

	strategy := &UnixUpgradeStrategy{}

	// Test with a temporary file
	tempDir := t.TempDir()
	testBinary := filepath.Join(tempDir, "test-lf")

	// Create a test binary
	file, err := os.Create(testBinary)
	if err != nil {
		t.Fatalf("Failed to create test binary: %v", err)
	}
	file.Close()

	// Make it executable
	err = os.Chmod(testBinary, 0755)
	if err != nil {
		t.Fatalf("Failed to make test binary executable: %v", err)
	}

	// Test CanUpgrade
	if !strategy.CanUpgrade(testBinary) {
		t.Error("Expected to be able to upgrade test binary")
	}

	// Test GetFallbackDir
	fallbackDir, err := strategy.GetFallbackDir()
	if err != nil {
		t.Errorf("Failed to get fallback directory: %v", err)
	}
	if fallbackDir == "" {
		t.Error("Expected non-empty fallback directory")
	}
}

func TestWindowsUpgradeStrategy(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("Skipping Windows tests on non-Windows systems")
	}

	strategy := &WindowsUpgradeStrategy{}

	// Test with a temporary file
	tempDir := t.TempDir()
	testBinary := filepath.Join(tempDir, "test-lf.exe")

	// Create a test binary
	file, err := os.Create(testBinary)
	if err != nil {
		t.Fatalf("Failed to create test binary: %v", err)
	}
	file.Close()

	// Test CanUpgrade
	if !strategy.CanUpgrade(testBinary) {
		t.Error("Expected to be able to upgrade test binary")
	}

	// Test GetFallbackDir
	fallbackDir, err := strategy.GetFallbackDir()
	if err != nil {
		t.Errorf("Failed to get fallback directory: %v", err)
	}
	if fallbackDir == "" {
		t.Error("Expected non-empty fallback directory")
	}
}

func TestCopyFile(t *testing.T) {
	tempDir := t.TempDir()

	// Create source file
	srcFile := filepath.Join(tempDir, "source.txt")
	content := "test content"
	err := os.WriteFile(srcFile, []byte(content), 0644)
	if err != nil {
		t.Fatalf("Failed to create source file: %v", err)
	}

	// Copy to destination
	dstFile := filepath.Join(tempDir, "destination.txt")
	err = copyFile(srcFile, dstFile)
	if err != nil {
		t.Fatalf("Failed to copy file: %v", err)
	}

	// Verify content
	dstContent, err := os.ReadFile(dstFile)
	if err != nil {
		t.Fatalf("Failed to read destination file: %v", err)
	}

	if string(dstContent) != content {
		t.Errorf("Expected content %s, got %s", content, string(dstContent))
	}

	// Verify permissions
	srcInfo, err := os.Stat(srcFile)
	if err != nil {
		t.Fatalf("Failed to stat source file: %v", err)
	}

	dstInfo, err := os.Stat(dstFile)
	if err != nil {
		t.Fatalf("Failed to stat destination file: %v", err)
	}

	if srcInfo.Mode() != dstInfo.Mode() {
		t.Errorf("Expected mode %v, got %v", srcInfo.Mode(), dstInfo.Mode())
	}
}

func TestCopyFileToNonExistentDir(t *testing.T) {
	tempDir := t.TempDir()

	// Create a source file
	srcFile := filepath.Join(tempDir, "source.txt")
	content := []byte("test content")
	if err := os.WriteFile(srcFile, content, 0644); err != nil {
		t.Fatalf("Failed to create source file: %v", err)
	}

	// Destination directory does not exist
	nonExistentDir := filepath.Join(tempDir, "does_not_exist")
	dstFile := filepath.Join(nonExistentDir, "dest.txt")

	// Attempt to copy
	err := copyFile(srcFile, dstFile)
	if err == nil {
		t.Error("Expected error when copying to non-existent directory, got nil")
	}
}
