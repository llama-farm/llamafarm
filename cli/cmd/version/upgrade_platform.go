package version

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"time"

	"github.com/llamafarm/cli/cmd/utils"
)

// UpgradeStrategy defines the interface for platform-specific upgrade operations
type UpgradeStrategy interface {
	CanUpgrade(binaryPath string) bool
	RequiresElevation(binaryPath string) bool
	PerformUpgrade(current, new string) error
	GetFallbackDir() (string, error)
}

// GetUpgradeStrategy returns the appropriate upgrade strategy for the current platform
func GetUpgradeStrategy() UpgradeStrategy {
	switch runtime.GOOS {
	case "windows":
		return &WindowsUpgradeStrategy{}
	default:
		return &UnixUpgradeStrategy{}
	}
}

// UnixUpgradeStrategy handles upgrades on Unix-like systems (Linux, macOS)
type UnixUpgradeStrategy struct{}

func (u *UnixUpgradeStrategy) CanUpgrade(binaryPath string) bool {
	return validateBinaryPath(binaryPath) == nil
}

func (u *UnixUpgradeStrategy) RequiresElevation(binaryPath string) bool {
	return needsElevationUnix(binaryPath)
}

func (u *UnixUpgradeStrategy) GetFallbackDir() (string, error) {
	return getDefaultUserInstallDirUnix()
}

func (u *UnixUpgradeStrategy) PerformUpgrade(current, new string) error {
	if u.RequiresElevation(current) {
		return u.upgradeWithSudo(current, new)
	}
	return u.upgradeDirectly(current, new)
}

func (u *UnixUpgradeStrategy) upgradeDirectly(current, new string) error {
	utils.LogDebug("performing direct upgrade without elevation")

	// Create backup
	backup, err := u.createBackup(current)
	if err != nil {
		return fmt.Errorf("failed to create backup: %w", err)
	}

	// Perform atomic replacement
	err = u.atomicReplace(current, new)
	if err != nil {
		// Attempt to restore backup
		if restoreErr := u.restoreBackup(backup, current); restoreErr != nil {
			utils.LogDebug(fmt.Sprintf("failed to restore backup after upgrade failure: %v", restoreErr))
		}
		return fmt.Errorf("failed to replace binary: %w", err)
	}

	// Verify the upgrade
	if err := u.verifyBinary(current); err != nil {
		// Attempt to restore backup
		if restoreErr := u.restoreBackup(backup, current); restoreErr != nil {
			utils.LogDebug(fmt.Sprintf("failed to restore backup after verification failure: %v", restoreErr))
		}
		return fmt.Errorf("upgrade verification failed: %w", err)
	}

	// Cleanup backup
	u.cleanupBackup(backup)
	return nil
}

func (u *UnixUpgradeStrategy) upgradeWithSudo(current, new string) error {
	utils.LogDebug("performing upgrade with sudo elevation")

	// Step 1: Create backup (with sudo)
	timestamp := strconv.FormatInt(time.Now().Unix(), 10)
	backupPath := current + ".backup." + timestamp

	utils.OutputProgress("Creating backup...")
	if err := u.runSudoCommand("cp", current, backupPath); err != nil {
		return fmt.Errorf("failed to create backup: %w", err)
	}

	// Step 2: Remove old binary first (required on Linux - can't overwrite running executable)
	// This works because Linux allows unlinking a running binary, just not overwriting it
	utils.OutputProgress("Installing new binary...")
	if err := u.runSudoCommand("rm", "-f", current); err != nil {
		_ = u.restoreBackupWithSudo(backupPath, current) // Best effort restore
		return fmt.Errorf("failed to remove old binary: %w", err)
	}

	// Step 3: Copy new binary to location (with sudo)
	if err := u.runSudoCommand("cp", new, current); err != nil {
		_ = u.restoreBackupWithSudo(backupPath, current) // Best effort restore
		return fmt.Errorf("failed to copy new binary: %w", err)
	}

	// Step 4: Set executable permissions (with sudo)
	utils.OutputProgress("Setting executable permissions...")
	if err := u.runSudoCommand("chmod", "+x", current); err != nil {
		_ = u.restoreBackupWithSudo(backupPath, current) // Best effort restore
		return fmt.Errorf("failed to set executable permissions: %w", err)
	}

	// Step 5: Verify the installation
	utils.OutputProgress("Verifying installation...")
	if err := u.verifyBinary(current); err != nil {
		utils.OutputError("Verification failed, restoring backup...")
		if restoreErr := u.restoreBackupWithSudo(backupPath, current); restoreErr != nil {
			return fmt.Errorf("upgrade verification failed and backup restore failed: %w (original: %v)", restoreErr, err)
		}
		return fmt.Errorf("upgrade verification failed, backup restored: %w", err)
	}

	// Step 6: Cleanup backup (with sudo)
	utils.OutputProgress("Cleaning up backup...")
	if err := u.runSudoCommand("rm", "-f", backupPath); err != nil {
		utils.LogDebug(fmt.Sprintf("failed to cleanup backup %s: %v", backupPath, err))
		// Non-fatal - continue
	}

	utils.OutputSuccess("Upgrade completed successfully")
	return nil
}

// runSudoCommand executes a command with sudo, forwarding stdout/stderr
func (u *UnixUpgradeStrategy) runSudoCommand(name string, args ...string) error {
	cmdArgs := append([]string{name}, args...)
	cmd := exec.Command("sudo", cmdArgs...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// restoreBackupWithSudo attempts to restore a backup file using sudo
// Returns an error if restoration fails
func (u *UnixUpgradeStrategy) restoreBackupWithSudo(backupPath, targetPath string) error {
	if err := u.runSudoCommand("mv", backupPath, targetPath); err != nil {
		utils.LogDebug(fmt.Sprintf("failed to restore backup from %s to %s: %v", backupPath, targetPath, err))
		return fmt.Errorf("backup restore failed: %w", err)
	}
	return nil
}

func (u *UnixUpgradeStrategy) createBackup(binaryPath string) (string, error) {
	timestamp := strconv.FormatInt(time.Now().Unix(), 10)
	backupPath := binaryPath + ".backup." + timestamp

	err := copyFile(binaryPath, backupPath)
	if err != nil {
		return "", fmt.Errorf("failed to copy binary to backup location: %w", err)
	}

	return backupPath, nil
}

func (u *UnixUpgradeStrategy) atomicReplace(current, new string) error {
	// Move new binary to final location (handles cross-filesystem moves)
	err := utils.MoveFile(new, current)
	if err != nil {
		return fmt.Errorf("failed to move new binary to final location: %w", err)
	}

	// Make sure it's executable
	err = os.Chmod(current, 0755)
	if err != nil {
		return fmt.Errorf("failed to set executable permissions: %w", err)
	}

	return nil
}

func (u *UnixUpgradeStrategy) restoreBackup(backupPath, current string) error {
	return os.Rename(backupPath, current)
}

func (u *UnixUpgradeStrategy) verifyBinary(binaryPath string) error {
	// Validate the binary path before execution to prevent command injection
	if err := validateBinaryPath(binaryPath); err != nil {
		return fmt.Errorf("binary validation failed: %w", err)
	}
	cmd := exec.Command(binaryPath, "version")
	return cmd.Run()
}

func (u *UnixUpgradeStrategy) cleanupBackup(backupPath string) {
	if err := os.Remove(backupPath); err != nil {
		utils.LogDebug(fmt.Sprintf("failed to cleanup backup %s: %v", backupPath, err))
	}
}

// WindowsUpgradeStrategy handles upgrades on Windows systems
type WindowsUpgradeStrategy struct{}

func (w *WindowsUpgradeStrategy) CanUpgrade(binaryPath string) bool {
	return validateBinaryPath(binaryPath) == nil
}

func (w *WindowsUpgradeStrategy) RequiresElevation(binaryPath string) bool {
	return needsElevationWindows(binaryPath)
}

func (w *WindowsUpgradeStrategy) GetFallbackDir() (string, error) {
	return getDefaultUserInstallDirWindows()
}

func (w *WindowsUpgradeStrategy) PerformUpgrade(current, new string) error {
	if w.RequiresElevation(current) {
		return w.upgradeWithUAC(current, new)
	}
	return w.upgradeDirectly(current, new)
}

func (w *WindowsUpgradeStrategy) upgradeDirectly(current, new string) error {
	utils.LogDebug("performing direct upgrade without elevation")

	// Create backup
	backup, err := w.createBackup(current)
	if err != nil {
		return fmt.Errorf("failed to create backup: %w", err)
	}

	// Perform atomic replacement
	err = w.atomicReplace(current, new)
	if err != nil {
		// Attempt to restore backup
		if restoreErr := w.restoreBackup(backup, current); restoreErr != nil {
			utils.LogDebug(fmt.Sprintf("failed to restore backup after upgrade failure: %v", restoreErr))
		}
		return fmt.Errorf("failed to replace binary: %w", err)
	}

	// Verify the upgrade
	if err := w.verifyBinary(current); err != nil {
		// Attempt to restore backup
		if restoreErr := w.restoreBackup(backup, current); restoreErr != nil {
			utils.LogDebug(fmt.Sprintf("failed to restore backup after verification failure: %v", restoreErr))
		}
		return fmt.Errorf("upgrade verification failed: %w", err)
	}

	// Cleanup backup
	w.cleanupBackup(backup)
	return nil
}

func (w *WindowsUpgradeStrategy) upgradeWithUAC(current, new string) error {
	// For Windows UAC elevation, we would need to restart the process with elevated privileges
	// This is a simplified implementation - in practice, you might want to use a more sophisticated approach
	return fmt.Errorf("UAC elevation not yet implemented - please run as Administrator or install to user directory")
}

func (w *WindowsUpgradeStrategy) createBackup(binaryPath string) (string, error) {
	timestamp := strconv.FormatInt(time.Now().Unix(), 10)
	backupPath := binaryPath + ".backup." + timestamp

	err := copyFile(binaryPath, backupPath)
	if err != nil {
		return "", fmt.Errorf("failed to copy binary to backup location: %w", err)
	}

	return backupPath, nil
}

func (w *WindowsUpgradeStrategy) atomicReplace(current, new string) error {
	// On Windows, we need to handle potential file locking issues
	// Move current to backup location first
	timestamp := strconv.FormatInt(time.Now().Unix(), 10)
	tempBackup := current + ".temp." + timestamp

	err := os.Rename(current, tempBackup)
	if err != nil {
		return fmt.Errorf("failed to move current binary: %w", err)
	}

	// Move new binary to final location (handles cross-filesystem moves)
	err = utils.MoveFile(new, current)
	if err != nil {
		// Restore original if new binary move failed
		utils.MoveFile(tempBackup, current)
		return fmt.Errorf("failed to move new binary to final location: %w", err)
	}

	// Remove temporary backup
	os.Remove(tempBackup)
	return nil
}

func (w *WindowsUpgradeStrategy) restoreBackup(backupPath, current string) error {
	return os.Rename(backupPath, current)
}

func (w *WindowsUpgradeStrategy) verifyBinary(binaryPath string) error {
	// Validate the binary path before execution to prevent command injection
	if err := validateBinaryPath(binaryPath); err != nil {
		return fmt.Errorf("binary validation failed: %w", err)
	}
	cmd := exec.Command(binaryPath, "version")
	return cmd.Run()
}

func (w *WindowsUpgradeStrategy) cleanupBackup(backupPath string) {
	if err := os.Remove(backupPath); err != nil {
		utils.LogDebug(fmt.Sprintf("failed to cleanup backup %s: %v", backupPath, err))
	}
}

// copyFile copies a file from src to dst
func copyFile(src, dst string) error {
	sourceFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer sourceFile.Close()

	destFile, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer destFile.Close()

	_, err = destFile.ReadFrom(sourceFile)
	if err != nil {
		return err
	}

	// Copy permissions
	sourceInfo, err := os.Stat(src)
	if err != nil {
		return err
	}

	return os.Chmod(dst, sourceInfo.Mode())
}
