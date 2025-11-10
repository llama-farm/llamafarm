package version

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"

	"github.com/llamafarm/cli/cmd/utils"
)

// Version will be set by build flags during release builds
var CurrentVersion = "dev"

type UpgradeOpts struct {
	DryRun        bool
	Force         bool
	NoVerify      bool
	InstallDir    string
	TargetVersion string
}

// determineTargetVersion resolves the target version from args or fetches the latest
func determineTargetVersion(targetVersion *string) (string, *UpgradeInfo, error) {
	var info *UpgradeInfo
	var normalizedTargetVersion string

	if targetVersion != nil && *targetVersion != "" {
		normalizedTargetVersion = normalizeVersion(*targetVersion)

		// For specific version, create minimal info
		info = &UpgradeInfo{
			CurrentVersion:          CurrentVersion,
			LatestVersion:           normalizedTargetVersion,
			LatestVersionNormalized: normalizedTargetVersion,
			UpdateAvailable:         true,
		}
	} else {
		// Get latest version
		var err error
		info, err = MaybeCheckForUpgrade(true)
		if err != nil {
			return "", nil, fmt.Errorf("failed to check for updates: %w", err)
		} else if info == nil {
			return "", nil, fmt.Errorf("no release information available")
		} else {
			normalizedTargetVersion = info.LatestVersionNormalized
		}
	}

	return normalizedTargetVersion, info, nil
}

// showUpgradePlan displays the upgrade plan to the user
func showUpgradePlan(info *UpgradeInfo, targetVersion, finalInstallDir string, strategy UpgradeStrategy, canUpgradeInPlace bool, installDir string) {
	utils.OutputInfo("📋 Upgrade Plan:")
	utils.OutputInfo("   Current version: %s", info.CurrentVersion)
	utils.OutputInfo("   Target version:  %s", targetVersion)
	utils.OutputInfo("   Install location: %s", finalInstallDir)
	utils.OutputInfo("   Platform: %s", detectPlatform())

	requiresElevation := strategy.RequiresElevation(finalInstallDir)
	if requiresElevation {
		utils.OutputInfo("   ⚠️  Requires elevation (sudo/Administrator)")
	}

	if !canUpgradeInPlace && installDir == "" {
		// Suggest fallback directory
		fallbackDir, err := strategy.GetFallbackDir()
		if err == nil {
			utils.OutputInfo("   💡 Suggested fallback: %s", fallbackDir)
		}
	}
}

// checkPermissions validates that we have permissions to perform the upgrade
func checkPermissions(canUpgradeInPlace bool, installDir, finalInstallDir string, strategy UpgradeStrategy) error {
	if canUpgradeInPlace || installDir != "" {
		return nil
	}

	requiresElevation := strategy.RequiresElevation(finalInstallDir)
	if !requiresElevation {
		return nil
	}

	utils.OutputInfo("\n❌ Cannot write to %s without elevation", finalInstallDir)
	utils.OutputInfo("\nOptions:")
	utils.OutputInfo("1. Run with elevation: sudo lf version upgrade")

	fallbackDir, err := strategy.GetFallbackDir()
	if err == nil {
		utils.OutputInfo("2. Install to user directory: lf version upgrade --install-dir %s", fallbackDir)
	}

	utils.OutputInfo("3. Manual installation: curl -fsSL https://raw.githubusercontent.com/llama-farm/llamafarm/main/install.sh | bash")
	return fmt.Errorf("insufficient permissions for upgrade")
}

// downloadAndVerifyBinary downloads the binary and optionally verifies its checksum
func downloadAndVerifyBinary(targetVersion, platform string, noVerify bool) (string, error) {
	utils.LogDebug("🔄 Downloading binary...")
	tempBinary, err := downloadBinary(targetVersion, platform)
	if err != nil {
		return "", fmt.Errorf("failed to download binary: %w", err)
	}

	if !noVerify {
		utils.LogDebug("🔄 Verifying checksum...")
		err = verifyChecksum(tempBinary, targetVersion, platform)
		if err != nil {
			cleanupTempFiles([]string{tempBinary})
			return "", fmt.Errorf("checksum verification failed: %w", err)
		}
	} else {
		utils.OutputInfo("⚠️  Skipping checksum verification")
	}

	return tempBinary, nil
}

// determineFinalBinaryPath resolves the final installation path for the binary
func determineFinalBinaryPath(installDir, currentBinary, platform string) (string, error) {
	if installDir != "" {
		// Custom install directory
		binaryName := "lf"
		if strings.Contains(platform, "windows") {
			binaryName += ".exe"
		}
		finalBinaryPath := filepath.Join(installDir, binaryName)

		// Ensure directory exists
		if err := os.MkdirAll(installDir, 0755); err != nil {
			return "", fmt.Errorf("failed to create install directory: %w", err)
		}

		return finalBinaryPath, nil
	}

	// Use current binary location
	return currentBinary, nil
}

// performUpgrade handles the automatic upgrade process
func PerformUpgrade(opts UpgradeOpts) error {
	// Get current binary path
	currentBinary, err := getCurrentBinaryPath()
	if err != nil {
		return fmt.Errorf("failed to determine current binary location: %w", err)
	}

	utils.LogDebug(fmt.Sprintf("🔍 Current binary: %s", currentBinary))

	// Determine target version
	targetVersion, info, err := determineTargetVersion(&opts.TargetVersion)
	if err != nil {
		return err
	}

	// Check if upgrade is necessary
	if !opts.Force && !info.UpdateAvailable && targetVersion == info.CurrentVersionNormalized {
		utils.OutputInfo("✅ Already running version %s", info.CurrentVersion)
		return nil
	}

	// Determine installation directory
	var finalInstallDir string
	if opts.InstallDir != "" {
		finalInstallDir = opts.InstallDir
	} else {
		finalInstallDir = filepath.Dir(currentBinary)
	}

	// Get upgrade strategy
	strategy := GetUpgradeStrategy()

	// Check if we can upgrade to the current location
	canUpgradeInPlace := strategy.CanUpgrade(currentBinary) && canWriteToLocation(currentBinary)

	// Show upgrade plan
	showUpgradePlan(info, targetVersion, finalInstallDir, strategy, canUpgradeInPlace, opts.InstallDir)

	if opts.DryRun {
		utils.OutputInfo("\n🔍 Dry run mode - no changes will be made")
		return nil
	}

	// Check permissions
	if err := checkPermissions(canUpgradeInPlace, opts.InstallDir, finalInstallDir, strategy); err != nil {
		return err
	}

	// Confirm upgrade
	utils.OutputInfo("\n🚀 Starting upgrade to %s...", targetVersion)

	platform := detectPlatform()

	// Download and verify binary
	tempBinary, err := downloadAndVerifyBinary(targetVersion, platform, opts.NoVerify)
	if err != nil {
		return err
	}
	defer cleanupTempFiles([]string{tempBinary})

	// Determine final binary path
	finalBinaryPath, err := determineFinalBinaryPath(opts.InstallDir, currentBinary, platform)
	if err != nil {
		return err
	}

	// Perform upgrade
	utils.LogDebug("🔄 Installing new version...")
	err = strategy.PerformUpgrade(finalBinaryPath, tempBinary)
	if err != nil {
		return fmt.Errorf("upgrade failed: %w", err)
	}

	// Verify installation
	utils.LogDebug("🔄 Verifying installation...")
	if err := validateBinaryPath(finalBinaryPath); err != nil {
		return fmt.Errorf("installation verification failed: %w", err)
	}

	utils.OutputInfo("✅ Upgrade completed successfully!")
	utils.OutputInfo("\nRun 'lf version' to confirm the new version.")

	// Show PATH warning if needed
	if opts.InstallDir != "" && opts.InstallDir != filepath.Dir(currentBinary) {
		utils.OutputInfo("\n💡 Binary installed to: %s", finalBinaryPath)
		utils.OutputInfo("Make sure this directory is in your PATH.")
	}

	// If requested (e.g., from TUI), restart into the updated binary
	if os.Getenv("LF_RESTART_AFTER_UPGRADE") == "1" {
		// Avoid looping if we were invoked as `lf version upgrade`
		argsToUse := os.Args[1:]
		if len(argsToUse) >= 2 && argsToUse[0] == "version" && argsToUse[1] == "upgrade" {
			argsToUse = []string{}
		}
		utils.OutputInfo("\n🔁 Restarting CLI...")
		if runtime.GOOS == "windows" {
			// On Windows, fall back to manual restart
			utils.OutputInfo("Restart is not automated on Windows. Please relaunch the CLI.")
			// Unset the flag to avoid leaking into subsequent processes
			_ = os.Unsetenv("LF_RESTART_AFTER_UPGRADE")
			return nil
		}
		// Re-exec the new binary in-place
		if err := validateBinaryPath(finalBinaryPath); err != nil {
			utils.OutputInfo("\n⚠️  Restart validation failed: %v", err)
		} else {
			// Use a minimal, controlled environment for restart and ensure the flag does not persist
			_ = os.Unsetenv("LF_RESTART_AFTER_UPGRADE")
			if execErr := syscall.Exec(finalBinaryPath, append([]string{finalBinaryPath}, argsToUse...), os.Environ()); execErr != nil {
				utils.OutputInfo("\n⚠️  Restart exec failed: %v", execErr)
			}
		}
		// If Exec returns, show a hint
		utils.OutputInfo("\n⚠️  Restart failed. Please exit and relaunch the CLI.")
	}
	return nil
}

// showManualInstructions displays manual installation instructions as fallback
func showManualInstructions(info *UpgradeInfo) {
	utils.OutputInfo("\n📖 Manual Installation Instructions:")
	utils.OutputInfo("  • macOS / Linux: curl -fsSL https://raw.githubusercontent.com/llama-farm/llamafarm/main/install.sh | bash")
	utils.OutputInfo("  • Windows:       winget install LlamaFarm.CLI")

	if info.ReleaseURL != "" {
		utils.OutputInfo("  • Release notes: %s", info.ReleaseURL)
	}
}
