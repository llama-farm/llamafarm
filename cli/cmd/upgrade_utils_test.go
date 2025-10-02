package cmd

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestDetectPlatform(t *testing.T) {
	platform := detectPlatform()

	// Should contain OS and architecture
	if !strings.Contains(platform, "-") {
		t.Errorf("Expected platform to contain '-', got: %s", platform)
	}

	parts := strings.Split(platform, "-")
	if len(parts) != 2 {
		t.Errorf("Expected platform to have 2 parts separated by '-', got: %s", platform)
	}

	// Should match current runtime
	expectedOS := runtime.GOOS
	if expectedOS == "darwin" || expectedOS == "linux" || expectedOS == "windows" {
		if parts[0] != expectedOS {
			t.Errorf("Expected OS part to be %s, got: %s", expectedOS, parts[0])
		}
	}
}

func TestGetBinaryNameForPlatform(t *testing.T) {
	tests := []struct {
		platform string
		expected string
	}{
		{"linux-amd64", "lf-linux-amd64"},
		{"darwin-arm64", "lf-darwin-arm64"},
		{"windows-amd64", "lf-windows-amd64.exe"},
		{"windows-arm64", "lf-windows-arm64.exe"},
	}

	for _, test := range tests {
		result := getBinaryNameForPlatform(test.platform)
		if result != test.expected {
			t.Errorf("For platform %s, expected %s, got %s", test.platform, test.expected, result)
		}
	}
}

func TestNormalizeVersion(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"1.0.0", "v1.0.0"},
		{"v1.0.0", "v1.0.0"},
		{"V1.0.0", "v1.0.0"},
		{"v1.2.3-beta", "v1.2.3-beta"},
		{"2.0.0-rc.1", "v2.0.0-rc.1"},
		{"", ""},
	}

	for _, test := range tests {
		result := normalizeVersion(test.input)
		if result != test.expected {
			t.Errorf("For input %s, expected %s, got %s", test.input, test.expected, result)
		}
	}
}

func TestGetCurrentBinaryPath(t *testing.T) {
	path, err := getCurrentBinaryPath()
	if err != nil {
		t.Fatalf("Failed to get current binary path: %v", err)
	}

	if path == "" {
		t.Error("Expected non-empty binary path")
	}

	if !filepath.IsAbs(path) {
		t.Errorf("Expected absolute path, got: %s", path)
	}
}

func TestCanWriteToLocation(t *testing.T) {
	// Test with a temporary file we create
	tempDir := t.TempDir()
	tempFile := filepath.Join(tempDir, "test-binary")

	// Create a test file
	file, err := os.Create(tempFile)
	if err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}
	file.Close()

	// Should be able to write to temp directory
	canWrite := canWriteToLocation(tempFile)
	if !canWrite {
		t.Error("Expected to be able to write to temp directory")
	}
}

func TestGetDefaultUserInstallDir(t *testing.T) {
	dir, err := getDefaultUserInstallDir()
	if err != nil {
		t.Fatalf("Failed to get default user install dir: %v", err)
	}

	if dir == "" {
		t.Error("Expected non-empty install directory")
	}

	if !filepath.IsAbs(dir) {
		t.Errorf("Expected absolute path, got: %s", dir)
	}

	// Directory should exist after calling the function
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		t.Errorf("Expected directory to exist: %s", dir)
	}
}

func TestIsExecutable(t *testing.T) {
	tempDir := t.TempDir()

	if runtime.GOOS == "windows" {
		// Test Windows executable detection
		exeFile := filepath.Join(tempDir, "test.exe")
		err := os.WriteFile(exeFile, []byte("test"), 0644)
		if err != nil {
			t.Fatalf("Failed to create test exe file: %v", err)
		}

		info, err := os.Stat(exeFile)
		if err != nil {
			t.Fatalf("Failed to stat exe file: %v", err)
		}

		if !isExecutable(exeFile, info.Mode()) {
			t.Error("Expected .exe file to be detected as executable on Windows")
		}

		// Test non-executable file
		txtFile := filepath.Join(tempDir, "test.txt")
		err = os.WriteFile(txtFile, []byte("test"), 0644)
		if err != nil {
			t.Fatalf("Failed to create test txt file: %v", err)
		}

		info, err = os.Stat(txtFile)
		if err != nil {
			t.Fatalf("Failed to stat txt file: %v", err)
		}

		if isExecutable(txtFile, info.Mode()) {
			t.Error("Expected .txt file to not be detected as executable on Windows")
		}
	} else {
		// Test Unix executable detection
		execFile := filepath.Join(tempDir, "test-exec")
		err := os.WriteFile(execFile, []byte("#!/bin/sh\necho test"), 0755)
		if err != nil {
			t.Fatalf("Failed to create test executable file: %v", err)
		}

		info, err := os.Stat(execFile)
		if err != nil {
			t.Fatalf("Failed to stat executable file: %v", err)
		}

		if !isExecutable(execFile, info.Mode()) {
			t.Error("Expected file with executable bit to be detected as executable on Unix")
		}

		// Test non-executable file
		nonExecFile := filepath.Join(tempDir, "test-nonexec")
		err = os.WriteFile(nonExecFile, []byte("test"), 0644)
		if err != nil {
			t.Fatalf("Failed to create test non-executable file: %v", err)
		}

		info, err = os.Stat(nonExecFile)
		if err != nil {
			t.Fatalf("Failed to stat non-executable file: %v", err)
		}

		if isExecutable(nonExecFile, info.Mode()) {
			t.Error("Expected file without executable bit to not be detected as executable on Unix")
		}
	}
}

func TestIsExecutable_UncommonWindowsExtensions(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("Windows-specific test")
	}

	extensions := []string{".bat", ".cmd", ".com"}
	for _, ext := range extensions {
		filePath := filepath.Join(t.TempDir(), "test"+ext)
		f, err := os.Create(filePath)
		if err != nil {
			t.Fatalf("Failed to create file %s: %v", filePath, err)
		}
		f.Close()

		info, err := os.Stat(filePath)
		if err != nil {
			t.Fatalf("Failed to stat file %s: %v", filePath, err)
		}

		if !isExecutable(filePath, info.Mode()) {
			t.Errorf("Expected %s to be detected as executable", filePath)
		}
	}
}
