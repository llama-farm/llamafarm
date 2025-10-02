package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	semver "github.com/Masterminds/semver/v3"
)

// getCurrentBinaryPath returns the absolute path to the currently running binary
func getCurrentBinaryPath() (string, error) {
	execPath, err := os.Executable()
	if err != nil {
		return "", fmt.Errorf("failed to get executable path: %w", err)
	}

	// Resolve any symlinks to get the actual binary path
	realPath, err := filepath.EvalSymlinks(execPath)
	if err != nil {
		// If we can't resolve symlinks, use the original path
		logDebug(fmt.Sprintf("could not resolve symlinks for %s: %v", execPath, err))
		realPath = execPath
	}

	absPath, err := filepath.Abs(realPath)
	if err != nil {
		return "", fmt.Errorf("failed to get absolute path: %w", err)
	}

	return absPath, nil
}

// validateBinaryPath checks if the given path is a valid binary location using cross-platform methods
func validateBinaryPath(path string) error {
	if path == "" {
		return fmt.Errorf("binary path cannot be empty")
	}

	// Check if the file exists
	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("binary not found at %s: %w", path, err)
	}

	// Check if it's a regular file (not a directory)
	if !info.Mode().IsRegular() {
		return fmt.Errorf("path %s is not a regular file", path)
	}

	// Check if it's executable using cross-platform method
	if !isExecutable(path, info.Mode()) {
		return fmt.Errorf("binary at %s is not executable", path)
	}

	return nil
}

// isExecutable checks if a file is executable using cross-platform methods
func isExecutable(path string, mode os.FileMode) bool {
	if runtime.GOOS == "windows" {
		// On Windows, check if it's a .exe file or has executable extension
		ext := strings.ToLower(filepath.Ext(path))
		return ext == ".exe" || ext == ".bat" || ext == ".cmd" || ext == ".com"
	}

	// On Unix-like systems, check the executable bit
	return mode&0111 != 0
}

// canWriteToLocation checks if we have write permissions to the directory containing the binary
func canWriteToLocation(path string) bool {
	if runtime.GOOS == "windows" {
		return canWriteToLocationWindows(path)
	}
	return canWriteToLocationUnix(path)
}

// canWriteToLocationUnix checks write permissions on Unix-like systems
func canWriteToLocationUnix(path string) bool {
	dir := filepath.Dir(path)

	// Try to create a temporary file to test write access
	tempFile := filepath.Join(dir, ".lf_write_test_"+fmt.Sprintf("%d", os.Getpid()))

	file, err := os.Create(tempFile)
	if err != nil {
		return false
	}
	file.Close()
	os.Remove(tempFile)
	return true
}

// canWriteToLocationWindows checks write permissions on Windows
func canWriteToLocationWindows(path string) bool {
	// Try to create a temporary file in the directory to test write access
	dir := filepath.Dir(path)
	tempFile := filepath.Join(dir, ".lf_write_test_"+fmt.Sprintf("%d", os.Getpid()))

	file, err := os.Create(tempFile)
	if err != nil {
		return false
	}
	file.Close()
	os.Remove(tempFile)
	return true
}

// needsElevation determines if elevation (sudo/UAC) is required for the upgrade
func needsElevation(path string) bool {
	if runtime.GOOS == "windows" {
		return needsElevationWindows(path)
	}
	return needsElevationUnix(path)
}

// needsElevationUnix determines if sudo is needed on Unix-like systems
func needsElevationUnix(path string) bool {
	// If we can't write to the location and it's in a system directory, we need sudo
	if canWriteToLocation(path) {
		return false
	}

	// Check if the path is in common system directories using absolute path comparison
	systemDirs := []string{
		"/usr/local/bin",
		"/usr/bin",
		"/bin",
		"/opt",
	}

	absPath, err := filepath.Abs(path)
	if err != nil {
		// If we can't resolve the absolute path, assume elevation is needed
		return true
	}

	// Get the directory containing the binary
	pathDir := filepath.Dir(absPath)

	for _, sysDir := range systemDirs {
		absSysDir, err := filepath.Abs(sysDir)
		if err != nil {
			continue
		}
		// Check if the binary's directory is within the system directory
		if pathDir == absSysDir || strings.HasPrefix(pathDir+string(filepath.Separator), absSysDir+string(filepath.Separator)) {
			return true
		}
	}

	return false
}

// needsElevationWindows determines if UAC elevation is needed on Windows
func needsElevationWindows(path string) bool {
	// If we can write to the location, no elevation needed
	if canWriteToLocation(path) {
		return false
	}

	// Check if the path is in system directories that typically require elevation
	systemDirs := []string{
		"C:\\Program Files",
		"C:\\Program Files (x86)",
		"C:\\Windows",
	}

	upperPath := strings.ToUpper(path)
	for _, sysDir := range systemDirs {
		if strings.HasPrefix(upperPath, strings.ToUpper(sysDir)) {
			return true
		}
	}

	return false
}

// getDefaultUserInstallDir returns a default user-writable installation directory
func getDefaultUserInstallDir() (string, error) {
	if runtime.GOOS == "windows" {
		return getDefaultUserInstallDirWindows()
	}
	return getDefaultUserInstallDirUnix()
}

// getDefaultUserInstallDirUnix returns the default user install directory on Unix-like systems
func getDefaultUserInstallDirUnix() (string, error) {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("failed to get user home directory: %w", err)
	}

	userBinDir := filepath.Join(homeDir, ".local", "bin")

	// Create the directory if it doesn't exist
	if err := os.MkdirAll(userBinDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create user bin directory: %w", err)
	}

	return userBinDir, nil
}

// getDefaultUserInstallDirWindows returns the default user install directory on Windows
func getDefaultUserInstallDirWindows() (string, error) {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("failed to get user home directory: %w", err)
	}

	userBinDir := filepath.Join(homeDir, "AppData", "Local", "Programs", "LlamaFarm")

	// Create the directory if it doesn't exist
	if err := os.MkdirAll(userBinDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create user bin directory: %w", err)
	}

	return userBinDir, nil
}

// detectPlatform returns the platform string used in GitHub releases
func detectPlatform() string {
	goos := runtime.GOOS
	goarch := runtime.GOARCH

	// Map Go OS names to GitHub release names
	switch goos {
	case "darwin":
		// Keep as darwin for macOS
	case "linux":
		// Keep as linux
	case "windows":
		// Keep as windows
	default:
		logDebug(fmt.Sprintf("unknown OS: %s, defaulting to linux", goos))
		goos = "linux"
	}

	// Map Go arch names to GitHub release names
	switch goarch {
	case "amd64":
		// Keep as amd64
	case "arm64":
		// Keep as arm64
	case "386":
		goarch = "386"
	case "arm":
		goarch = "arm"
	default:
		logDebug(fmt.Sprintf("unknown architecture: %s, defaulting to amd64", goarch))
		goarch = "amd64"
	}

	return fmt.Sprintf("%s-%s", goos, goarch)
}

// getBinaryNameForPlatform returns the expected binary name for the given platform
func getBinaryNameForPlatform(platform string) string {
	binaryName := "lf-" + platform
	if strings.Contains(platform, "windows") {
		binaryName += ".exe"
	}
	return binaryName
}

// normalizeVersion ensures version has 'v' prefix and validates it using semver
func normalizeVersion(version string) string {
	if version == "" {
		return ""
	}

	// Strip any existing prefix for validation
	normalized := strings.TrimPrefix(strings.TrimPrefix(version, "v"), "V")

	// Validate using semver library
	_, err := semver.NewVersion(normalized)
	if err != nil {
		// Return original if not valid semver (could be a tag name)
		// but still ensure it has 'v' prefix
		if !strings.HasPrefix(version, "v") && !strings.HasPrefix(version, "V") {
			return "v" + version
		}
		return version
	}

	// Add 'v' prefix to the normalized version
	return "v" + normalized
}
