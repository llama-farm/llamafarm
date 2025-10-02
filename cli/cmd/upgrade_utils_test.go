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

func TestIsValidVersion(t *testing.T) {
	tests := []struct {
		version string
		valid   bool
	}{
		{"v1.0.0", true},
		{"1.0.0", true},
		{"v1.2.3-beta", true},
		{"2.0.0-rc.1", true},
		{"", false},
		{"invalid", false},
		{"v", false},
		{"1", false},
	}

	for _, test := range tests {
		result := isValidVersion(test.version)
		if result != test.valid {
			t.Errorf("For version %s, expected %v, got %v", test.version, test.valid, result)
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
		{"V1.0.0", "V1.0.0"},
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
