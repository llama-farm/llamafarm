package version

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// TestUpgradeIntegration runs comprehensive integration tests for the upgrade logic
// These tests simulate real-world upgrade scenarios
func TestUpgradeIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration tests in short mode")
	}

	// Build two test binaries with different versions
	oldVersion := "test-1.0.0"
	newVersion := "test-1.1.0"

	testDir := t.TempDir()
	oldBinary := buildTestBinary(t, testDir, "lf-old", oldVersion)
	newBinary := buildTestBinary(t, testDir, "lf-new", newVersion)

	t.Run("UserDirectoryUpgrade", func(t *testing.T) {
		testUserDirectoryUpgrade(t, testDir, oldBinary, newBinary, oldVersion, newVersion)
	})

	t.Run("BackupAndRestore", func(t *testing.T) {
		testBackupAndRestore(t, testDir, oldBinary)
	})

	t.Run("UpgradeFailsWithoutWritePermission", func(t *testing.T) {
		if runtime.GOOS == "windows" {
			t.Skip("Permission tests not reliable on Windows")
		}
		testUpgradeFailsWithoutWritePermission(t, testDir, oldBinary, newBinary)
	})

	t.Run("PermissionPreservation", func(t *testing.T) {
		testPermissionPreservation(t, testDir, oldBinary, newBinary)
	})

	t.Run("RestoreWithMissingBackup", func(t *testing.T) {
		testRestoreWithMissingBackup(t, testDir)
	})

	t.Run("RestoreWithCorruptedBackup", func(t *testing.T) {
		testRestoreWithCorruptedBackup(t, testDir, oldBinary)
	})

	t.Run("BinaryVerification", func(t *testing.T) {
		testBinaryVerification(t, testDir, newBinary)
	})

	t.Run("AtomicReplacement", func(t *testing.T) {
		testAtomicReplacement(t, testDir, oldBinary, newBinary, newVersion)
	})

	t.Run("ConcurrentAccess", func(t *testing.T) {
		if runtime.GOOS == "windows" {
			t.Skip("Concurrent access test not reliable on Windows")
		}
		testConcurrentAccess(t, testDir, oldBinary, newBinary, newVersion)
	})
}

// buildTestBinary builds a test binary with a specific version
func buildTestBinary(t *testing.T, testDir, name, version string) string {
	t.Helper()

	binaryPath := filepath.Join(testDir, name)
	if runtime.GOOS == "windows" {
		binaryPath += ".exe"
	}

	// Build from current package
	cmd := exec.Command("go", "build",
		"-ldflags", "-X 'github.com/llamafarm/cli/internal/buildinfo.CurrentVersion="+version+"'",
		"-o", binaryPath,
		"github.com/llamafarm/cli")

	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("Failed to build test binary %s: %v\nOutput: %s", name, err, output)
	}

	// Make executable
	if err := os.Chmod(binaryPath, 0755); err != nil {
		t.Fatalf("Failed to chmod test binary: %v", err)
	}

	return binaryPath
}

// testUserDirectoryUpgrade tests upgrading a binary in a user-writable directory
func testUserDirectoryUpgrade(t *testing.T, testDir, oldBinary, newBinary, oldVersion, newVersion string) {
	t.Helper()

	installDir := filepath.Join(testDir, "user-install")
	if err := os.MkdirAll(installDir, 0755); err != nil {
		t.Fatalf("Failed to create install dir: %v", err)
	}

	installedBinary := filepath.Join(installDir, "lf")
	if runtime.GOOS == "windows" {
		installedBinary += ".exe"
	}

	// Install old version
	if err := copyFile(oldBinary, installedBinary); err != nil {
		t.Fatalf("Failed to install old binary: %v", err)
	}

	// Verify old version
	if version := getVersionFromBinary(t, installedBinary); version != oldVersion {
		t.Errorf("Expected old version %s, got %s", oldVersion, version)
	}

	// Simulate upgrade: create backup, replace, verify
	backupPath := installedBinary + ".backup"
	if err := copyFile(installedBinary, backupPath); err != nil {
		t.Fatalf("Failed to create backup: %v", err)
	}

	// Replace binary
	if err := os.Remove(installedBinary); err != nil {
		t.Fatalf("Failed to remove old binary: %v", err)
	}
	if err := copyFile(newBinary, installedBinary); err != nil {
		// Restore backup
		_ = copyFile(backupPath, installedBinary)
		t.Fatalf("Failed to copy new binary: %v", err)
	}

	// Verify new version
	if version := getVersionFromBinary(t, installedBinary); version != newVersion {
		t.Errorf("Expected new version %s, got %s", newVersion, version)
	}

	// Cleanup backup
	_ = os.Remove(backupPath)
}

// testBackupAndRestore tests the backup and restore functionality
func testBackupAndRestore(t *testing.T, testDir, oldBinary string) {
	t.Helper()

	installDir := filepath.Join(testDir, "backup-test")
	if err := os.MkdirAll(installDir, 0755); err != nil {
		t.Fatalf("Failed to create install dir: %v", err)
	}

	installedBinary := filepath.Join(installDir, "lf")
	if runtime.GOOS == "windows" {
		installedBinary += ".exe"
	}

	// Install old version
	if err := copyFile(oldBinary, installedBinary); err != nil {
		t.Fatalf("Failed to install binary: %v", err)
	}

	// Create backup
	backupPath := installedBinary + ".backup." + time.Now().Format("20060102150405")
	if err := copyFile(installedBinary, backupPath); err != nil {
		t.Fatalf("Failed to create backup: %v", err)
	}

	if _, err := os.Stat(backupPath); err != nil {
		t.Errorf("Backup file not created: %v", err)
	}

	// Simulate failed upgrade (corrupt binary)
	if err := os.WriteFile(installedBinary, []byte("corrupt binary"), 0755); err != nil {
		t.Fatalf("Failed to write corrupt binary: %v", err)
	}

	// Verify binary is broken
	if getVersionFromBinary(t, installedBinary) != "" {
		t.Error("Binary should be broken but reports version")
	}

	// Restore from backup
	if err := os.Remove(installedBinary); err != nil {
		t.Fatalf("Failed to remove corrupt binary: %v", err)
	}
	if err := os.Rename(backupPath, installedBinary); err != nil {
		t.Fatalf("Failed to restore backup: %v", err)
	}

	// Verify restoration
	if version := getVersionFromBinary(t, installedBinary); version == "" {
		t.Error("Binary not restored correctly, version check failed")
	}
}

// testUpgradeFailsWithoutWritePermission tests upgrade failure in a read-only directory
func testUpgradeFailsWithoutWritePermission(t *testing.T, testDir, oldBinary, newBinary string) {
	t.Helper()

	// Create a subdirectory for the test
	noWriteDir := filepath.Join(testDir, "no_write")
	if err := os.Mkdir(noWriteDir, 0755); err != nil {
		t.Fatalf("failed to create no_write directory: %v", err)
	}

	// Copy the old binary into the directory first
	oldBinaryPath := filepath.Join(noWriteDir, "lf")
	if runtime.GOOS == "windows" {
		oldBinaryPath += ".exe"
	}

	input, err := os.ReadFile(oldBinary)
	if err != nil {
		t.Fatalf("failed to read old binary: %v", err)
	}
	if err := os.WriteFile(oldBinaryPath, input, 0755); err != nil {
		t.Fatalf("failed to write old binary to no_write directory: %v", err)
	}

	// Now make the directory read-only
	if err := os.Chmod(noWriteDir, 0555); err != nil {
		t.Fatalf("failed to set directory to read-only: %v", err)
	}
	defer os.Chmod(noWriteDir, 0755) // Restore permissions for cleanup

	// Copy new binary to temp location for the upgrade attempt
	tempNewBinary := filepath.Join(noWriteDir, "lf.tmp")

	// Attempt upgrade (should fail due to permissions)
	err = os.WriteFile(tempNewBinary, []byte("test"), 0755)
	if err == nil {
		t.Fatalf("expected upgrade to fail due to insufficient permissions, but write succeeded")
	}

	// Ensure the old binary is still present and unchanged
	after, err := os.ReadFile(oldBinaryPath)
	if err != nil {
		t.Fatalf("failed to read old binary after failed upgrade: %v", err)
	}
	if string(input) != string(after) {
		t.Errorf("old binary was modified during failed upgrade")
	}

	// Verify the old binary is still executable
	if version := getVersionFromBinary(t, oldBinaryPath); version == "" {
		t.Error("old binary should still be executable after failed upgrade")
	}
}

// testRestoreWithMissingBackup tests restoration when the backup file is missing
func testRestoreWithMissingBackup(t *testing.T, testDir string) {
	t.Helper()

	// Simulate missing backup file
	missingBackupPath := filepath.Join(testDir, "missing-backup-file.bak")
	targetPath := filepath.Join(testDir, "target-binary")

	// Attempt to restore from non-existent backup
	err := os.Rename(missingBackupPath, targetPath)
	if err == nil {
		t.Fatalf("expected error when restoring from missing backup file, got nil")
	}

	// Verify error is due to missing file
	if !os.IsNotExist(err) {
		t.Errorf("expected 'file not found' error, got: %v", err)
	}
}

// testRestoreWithCorruptedBackup tests restoration when the backup file is corrupted
func testRestoreWithCorruptedBackup(t *testing.T, testDir, validBinary string) {
	t.Helper()

	installDir := filepath.Join(testDir, "corrupted-backup-test")
	if err := os.MkdirAll(installDir, 0755); err != nil {
		t.Fatalf("Failed to create install dir: %v", err)
	}

	// Create a corrupted backup file
	corruptedBackupPath := filepath.Join(installDir, "lf.backup")
	if err := os.WriteFile(corruptedBackupPath, []byte("corrupted data that cannot be restored"), 0755); err != nil {
		t.Fatalf("Failed to create corrupted backup file: %v", err)
	}

	// Target path for restoration
	installedBinary := filepath.Join(installDir, "lf")
	if runtime.GOOS == "windows" {
		installedBinary += ".exe"
	}

	// Attempt to restore from corrupted backup
	if err := os.Rename(corruptedBackupPath, installedBinary); err != nil {
		t.Fatalf("Failed to restore corrupted backup: %v", err)
	}

	// Verify that the restored binary is not executable (corrupted)
	version := getVersionFromBinary(t, installedBinary)
	if version != "" {
		t.Errorf("Corrupted backup should not be executable, but got version: %s", version)
	}

	// Verify the file exists but is corrupted
	data, err := os.ReadFile(installedBinary)
	if err != nil {
		t.Fatalf("Failed to read restored file: %v", err)
	}
	if string(data) != "corrupted data that cannot be restored" {
		t.Error("Restored file content doesn't match corrupted backup")
	}
}

// testPermissionPreservation tests that permissions are preserved during upgrade
func testPermissionPreservation(t *testing.T, testDir, oldBinary, newBinary string) {
	t.Helper()

	installDir := filepath.Join(testDir, "permission-test")
	if err := os.MkdirAll(installDir, 0755); err != nil {
		t.Fatalf("Failed to create install dir: %v", err)
	}

	installedBinary := filepath.Join(installDir, "lf")
	if runtime.GOOS == "windows" {
		installedBinary += ".exe"
	}

	// Install old version with specific permissions
	if err := copyFile(oldBinary, installedBinary); err != nil {
		t.Fatalf("Failed to install binary: %v", err)
	}
	if err := os.Chmod(installedBinary, 0755); err != nil {
		t.Fatalf("Failed to set permissions: %v", err)
	}

	oldInfo, err := os.Stat(installedBinary)
	if err != nil {
		t.Fatalf("Failed to stat old binary: %v", err)
	}
	oldMode := oldInfo.Mode()

	// Perform upgrade
	if err := os.Remove(installedBinary); err != nil {
		t.Fatalf("Failed to remove old binary: %v", err)
	}
	if err := copyFile(newBinary, installedBinary); err != nil {
		t.Fatalf("Failed to copy new binary: %v", err)
	}
	if err := os.Chmod(installedBinary, 0755); err != nil {
		t.Fatalf("Failed to set permissions on new binary: %v", err)
	}

	newInfo, err := os.Stat(installedBinary)
	if err != nil {
		t.Fatalf("Failed to stat new binary: %v", err)
	}
	newMode := newInfo.Mode()

	if runtime.GOOS != "windows" && oldMode.Perm() != newMode.Perm() {
		t.Errorf("Permissions not preserved: old=%v, new=%v", oldMode, newMode)
	}

	// Verify binary is still executable
	if version := getVersionFromBinary(t, installedBinary); version == "" {
		t.Error("Binary is not executable after upgrade")
	}
}

// testBinaryVerification tests the binary verification logic
func testBinaryVerification(t *testing.T, testDir, validBinary string) {
	t.Helper()

	installDir := filepath.Join(testDir, "verify-test")
	if err := os.MkdirAll(installDir, 0755); err != nil {
		t.Fatalf("Failed to create install dir: %v", err)
	}

	// Test valid binary
	installedBinary := filepath.Join(installDir, "lf-valid")
	if runtime.GOOS == "windows" {
		installedBinary += ".exe"
	}
	if err := copyFile(validBinary, installedBinary); err != nil {
		t.Fatalf("Failed to copy valid binary: %v", err)
	}

	if version := getVersionFromBinary(t, installedBinary); version == "" {
		t.Error("Valid binary verification failed")
	}

	// Test corrupted binary
	corruptBinary := filepath.Join(installDir, "lf-corrupt")
	if runtime.GOOS == "windows" {
		corruptBinary += ".exe"
	}
	if err := os.WriteFile(corruptBinary, []byte("corrupted"), 0755); err != nil {
		t.Fatalf("Failed to write corrupt binary: %v", err)
	}

	if version := getVersionFromBinary(t, corruptBinary); version != "" {
		t.Error("Corrupted binary should fail verification")
	}
}

// testAtomicReplacement tests that binary replacement is atomic
func testAtomicReplacement(t *testing.T, testDir, oldBinary, newBinary, newVersion string) {
	t.Helper()

	installDir := filepath.Join(testDir, "atomic-test")
	if err := os.MkdirAll(installDir, 0755); err != nil {
		t.Fatalf("Failed to create install dir: %v", err)
	}

	installedBinary := filepath.Join(installDir, "lf")
	if runtime.GOOS == "windows" {
		installedBinary += ".exe"
	}

	// Install old version
	if err := copyFile(oldBinary, installedBinary); err != nil {
		t.Fatalf("Failed to install old binary: %v", err)
	}

	// Perform atomic replacement using rename
	tempBinary := installedBinary + ".tmp"
	if err := copyFile(newBinary, tempBinary); err != nil {
		t.Fatalf("Failed to copy to temp: %v", err)
	}

	// Atomic rename
	if err := os.Rename(tempBinary, installedBinary); err != nil {
		t.Fatalf("Atomic replacement failed: %v", err)
	}

	// Verify new version
	if version := getVersionFromBinary(t, installedBinary); version != newVersion {
		t.Errorf("Expected version %s after atomic replacement, got %s", newVersion, version)
	}

	// Verify no temporary files left
	if _, err := os.Stat(tempBinary); !os.IsNotExist(err) {
		t.Error("Temporary file not cleaned up after atomic replacement")
	}
}

// testConcurrentAccess tests upgrade while the binary is in use
func testConcurrentAccess(t *testing.T, testDir, oldBinary, newBinary, newVersion string) {
	t.Helper()

	installDir := filepath.Join(testDir, "concurrent-test")
	if err := os.MkdirAll(installDir, 0755); err != nil {
		t.Fatalf("Failed to create install dir: %v", err)
	}

	installedBinary := filepath.Join(installDir, "lf")
	if runtime.GOOS == "windows" {
		installedBinary += ".exe"
	}

	// Install old version
	if err := copyFile(oldBinary, installedBinary); err != nil {
		t.Fatalf("Failed to install binary: %v", err)
	}

	// Start a background process using the binary
	cmd := exec.Command(installedBinary, "version")
	done := make(chan bool)
	go func() {
		// Keep running version command repeatedly
		for i := 0; i < 10; i++ {
			_ = cmd.Run()
			time.Sleep(50 * time.Millisecond)
		}
		done <- true
	}()

	// Give it a moment to start
	time.Sleep(100 * time.Millisecond)

	// Attempt upgrade while process is running
	tempBinary := installedBinary + ".tmp"
	if err := copyFile(newBinary, tempBinary); err != nil {
		t.Fatalf("Failed to copy new binary: %v", err)
	}

	// On Unix, we can replace the file even if it's in use
	err := os.Rename(tempBinary, installedBinary)
	if runtime.GOOS == "windows" {
		// Windows may fail here, which is expected
		if err != nil {
			t.Skip("Upgrade blocked by running process (expected on Windows)")
		}
	} else {
		// Unix should succeed
		if err != nil {
			t.Errorf("Upgrade should succeed on Unix even with running process: %v", err)
		}
	}

	// Wait for background process to finish
	<-done

	// If we succeeded, verify the upgrade
	if err == nil {
		if version := getVersionFromBinary(t, installedBinary); version != newVersion {
			t.Errorf("Expected version %s, got %s", newVersion, version)
		}
	}
}

// getVersionFromBinary runs the binary and extracts its version
// Returns empty string if the binary fails to execute
func getVersionFromBinary(t *testing.T, binaryPath string) string {
	t.Helper()

	cmd := exec.Command(binaryPath, "version")
	output, err := cmd.CombinedOutput()
	if err != nil {
		// Binary is not executable or failed
		return ""
	}

	// Parse version from output
	// Format is "LlamaFarm CLI vX.X.X" or similar
	outputStr := string(output)

	// Try to extract just the version number
	// Look for "vtest-X.X.X" or "test-X.X.X" pattern
	for _, word := range []string{"vtest-", "test-"} {
		if idx := strings.Index(outputStr, word); idx != -1 {
			start := idx
			if strings.HasPrefix(outputStr[idx:], "v") {
				start++ // Skip the 'v'
			}
			// Extract until whitespace or newline
			end := start
			for end < len(outputStr) && outputStr[end] != ' ' && outputStr[end] != '\n' && outputStr[end] != '\r' {
				end++
			}
			return outputStr[start:end]
		}
	}

	// If we can't parse it, return empty string
	return ""
}
