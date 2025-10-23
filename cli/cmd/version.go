package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"

	"github.com/spf13/cobra"
)

// Version will be set by build flags during release builds
var Version = "dev"

// versionCmd represents the version command
var versionCmd = &cobra.Command{
	Use:   "version",
	Short: "Print the version number of LlamaFarm CLI",
	Long:  "Print the version number of LlamaFarm CLI",
	Run: func(cmd *cobra.Command, args []string) {
		OutputInfo("LlamaFarm CLI %s", formatVersionForDisplay(Version))
	},
}

var upgradeCmd = &cobra.Command{
	Use:   "upgrade [version]",
	Short: "Upgrade LlamaFarm CLI to latest or specified version",
	Long: `Automatically upgrade the LlamaFarm CLI to the latest release or a specified version.

This command can automatically download and install the new version, handling
elevation/sudo as needed. If automatic upgrade fails, manual installation
instructions will be provided.

Examples:
  lf version upgrade              # Upgrade to latest version
  lf version upgrade v1.2.3       # Upgrade to specific version
  lf version upgrade --dry-run    # Show what would be done
  lf version upgrade --force      # Force upgrade even if same version`,
	RunE: func(cmd *cobra.Command, args []string) error {
		return performUpgrade(cmd, args)
	},
}

func init() {
	// Add flags to upgrade command
	upgradeCmd.Flags().Bool("dry-run", false, "Show upgrade plan without executing")
	upgradeCmd.Flags().Bool("force", false, "Force upgrade even if same version")
	upgradeCmd.Flags().Bool("no-verify", false, "Skip checksum verification (not recommended)")
	upgradeCmd.Flags().String("install-dir", "", "Override installation directory")

	versionCmd.AddCommand(upgradeCmd)
	rootCmd.AddCommand(versionCmd)
}

// upgradeFlags contains parsed command-line flags for the upgrade command
type upgradeFlags struct {
	dryRun     bool
	force      bool
	noVerify   bool
	installDir string
}

// parseUpgradeFlags extracts and returns the upgrade command flags
func parseUpgradeFlags(cmd *cobra.Command) upgradeFlags {
	dryRun, _ := cmd.Flags().GetBool("dry-run")
	force, _ := cmd.Flags().GetBool("force")
	noVerify, _ := cmd.Flags().GetBool("no-verify")
	installDir, _ := cmd.Flags().GetString("install-dir")

	return upgradeFlags{
		dryRun:     dryRun,
		force:      force,
		noVerify:   noVerify,
		installDir: installDir,
	}
}

// determineTargetVersion resolves the target version from args or fetches the latest
func determineTargetVersion(args []string) (string, *UpgradeInfo, error) {
	var targetVersion string
	var info *UpgradeInfo

	if len(args) > 0 {
		targetVersion = args[0]
		targetVersion = normalizeVersion(targetVersion)

		// For specific version, create minimal info
		info = &UpgradeInfo{
			CurrentVersion:          Version,
			LatestVersion:           targetVersion,
			LatestVersionNormalized: targetVersion,
			UpdateAvailable:         true,
		}
	} else {
		// Get latest version
		var err error
		info, err = maybeCheckForUpgrade(true)
		if err != nil {
			return "", nil, fmt.Errorf("failed to check for updates: %w", err)
		}
		if info == nil {
			return "", nil, fmt.Errorf("no release information available")
		}
		targetVersion = info.LatestVersionNormalized
	}

	return targetVersion, info, nil
}

// showUpgradePlan displays the upgrade plan to the user
func showUpgradePlan(info *UpgradeInfo, targetVersion, finalInstallDir string, strategy UpgradeStrategy, canUpgradeInPlace bool, installDir string) {
	OutputInfo("📋 Upgrade Plan:")
	OutputInfo("   Current version: %s", info.CurrentVersion)
	OutputInfo("   Target version:  %s", targetVersion)
	OutputInfo("   Install location: %s", finalInstallDir)
	OutputInfo("   Platform: %s", detectPlatform())

	requiresElevation := strategy.RequiresElevation(finalInstallDir)
	if requiresElevation {
		OutputInfo("   ⚠️  Requires elevation (sudo/Administrator)")
	}

	if !canUpgradeInPlace && installDir == "" {
		// Suggest fallback directory
		fallbackDir, err := strategy.GetFallbackDir()
		if err == nil {
			OutputInfo("   💡 Suggested fallback: %s", fallbackDir)
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

	OutputInfo("\n❌ Cannot write to %s without elevation", finalInstallDir)
	OutputInfo("\nOptions:")
	OutputInfo("1. Run with elevation: sudo lf version upgrade")

	fallbackDir, err := strategy.GetFallbackDir()
	if err == nil {
		OutputInfo("2. Install to user directory: lf version upgrade --install-dir %s", fallbackDir)
	}

	OutputInfo("3. Manual installation: curl -fsSL https://raw.githubusercontent.com/llama-farm/llamafarm/main/install.sh | bash")
	return fmt.Errorf("insufficient permissions for upgrade")
}

// downloadAndVerifyBinary downloads the binary and optionally verifies its checksum
func downloadAndVerifyBinary(targetVersion, platform string, noVerify bool) (string, error) {
	OutputInfo("🔄 Downloading binary...")
	tempBinary, err := downloadBinary(targetVersion, platform)
	if err != nil {
		return "", fmt.Errorf("failed to download binary: %w", err)
	}

	if !noVerify {
		OutputInfo("🔄 Verifying checksum...")
		err = verifyChecksum(tempBinary, targetVersion, platform)
		if err != nil {
			cleanupTempFiles([]string{tempBinary})
			return "", fmt.Errorf("checksum verification failed: %w", err)
		}
	} else {
		OutputInfo("⚠️  Skipping checksum verification")
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
func performUpgrade(cmd *cobra.Command, args []string) error {
	flags := parseUpgradeFlags(cmd)

	// Get current binary path
	currentBinary, err := getCurrentBinaryPath()
	if err != nil {
		return fmt.Errorf("failed to determine current binary location: %w", err)
	}

	OutputInfo("🔍 Current binary: %s", currentBinary)

	// Determine target version
	targetVersion, info, err := determineTargetVersion(args)
	if err != nil {
		return err
	}

	// Check if upgrade is necessary
	if !flags.force && !info.UpdateAvailable && targetVersion == info.CurrentVersionNormalized {
		OutputInfo("✅ Already running version %s", info.CurrentVersion)
		return nil
	}

	// Determine installation directory
	var finalInstallDir string
	if flags.installDir != "" {
		finalInstallDir = flags.installDir
	} else {
		finalInstallDir = filepath.Dir(currentBinary)
	}

	// Get upgrade strategy
	strategy := GetUpgradeStrategy()

	// Check if we can upgrade to the current location
	canUpgradeInPlace := strategy.CanUpgrade(currentBinary) && canWriteToLocation(currentBinary)

	// Show upgrade plan
	showUpgradePlan(info, targetVersion, finalInstallDir, strategy, canUpgradeInPlace, flags.installDir)

	if flags.dryRun {
		OutputInfo("\n🔍 Dry run mode - no changes will be made")
		return nil
	}

	// Check permissions
	if err := checkPermissions(canUpgradeInPlace, flags.installDir, finalInstallDir, strategy); err != nil {
		return err
	}

	// Confirm upgrade
	OutputInfo("\n🚀 Starting upgrade to %s...", targetVersion)

	platform := detectPlatform()

	// Download and verify binary
	tempBinary, err := downloadAndVerifyBinary(targetVersion, platform, flags.noVerify)
	if err != nil {
		return err
	}
	defer cleanupTempFiles([]string{tempBinary})

	// Determine final binary path
	finalBinaryPath, err := determineFinalBinaryPath(flags.installDir, currentBinary, platform)
	if err != nil {
		return err
	}

	// Perform upgrade
	OutputInfo("🔄 Installing new version...")
	err = strategy.PerformUpgrade(finalBinaryPath, tempBinary)
	if err != nil {
		return fmt.Errorf("upgrade failed: %w", err)
	}

	// Verify installation
	OutputInfo("🔄 Verifying installation...")
	if err := validateBinaryPath(finalBinaryPath); err != nil {
		return fmt.Errorf("installation verification failed: %w", err)
	}

	OutputInfo("✅ Upgrade completed successfully!")
	OutputInfo("\nRun 'lf version' to confirm the new version.")

	// Show PATH warning if needed
	if flags.installDir != "" && flags.installDir != filepath.Dir(currentBinary) {
		OutputInfo("\n💡 Binary installed to: %s", finalBinaryPath)
		OutputInfo("Make sure this directory is in your PATH.")
	}

	// If requested (e.g., from TUI), restart into the updated binary
	if os.Getenv("LF_RESTART_AFTER_UPGRADE") == "1" {
		// Avoid looping if we were invoked as `lf version upgrade`
		argsToUse := os.Args[1:]
		if len(argsToUse) >= 2 && argsToUse[0] == "version" && argsToUse[1] == "upgrade" {
			argsToUse = []string{}
		}
		OutputInfo("\n🔁 Restarting CLI...")
		if runtime.GOOS == "windows" {
			// On Windows, fall back to manual restart
			OutputInfo("Restart is not automated on Windows. Please relaunch the CLI.")
			// Unset the flag to avoid leaking into subsequent processes
			_ = os.Unsetenv("LF_RESTART_AFTER_UPGRADE")
			return nil
		}
		// Re-exec the new binary in-place
		if err := validateBinaryPath(finalBinaryPath); err != nil {
			OutputInfo("\n⚠️  Restart validation failed: %v", err)
		} else {
			// Use a minimal, controlled environment for restart and ensure the flag does not persist
			_ = os.Unsetenv("LF_RESTART_AFTER_UPGRADE")
			if execErr := syscall.Exec(finalBinaryPath, append([]string{finalBinaryPath}, argsToUse...), os.Environ()); execErr != nil {
				OutputInfo("\n⚠️  Restart exec failed: %v", execErr)
			}
		}
		// If Exec returns, show a hint
		OutputInfo("\n⚠️  Restart failed. Please exit and relaunch the CLI.")
	}
	return nil
}

// showManualInstructions displays manual installation instructions as fallback
func showManualInstructions(info *UpgradeInfo) {
	OutputInfo("\n📖 Manual Installation Instructions:")
	OutputInfo("  • macOS / Linux: curl -fsSL https://raw.githubusercontent.com/llama-farm/llamafarm/main/install.sh | bash")
	OutputInfo("  • Windows:       winget install LlamaFarm.CLI")

	if info.ReleaseURL != "" {
		OutputInfo("  • Release notes: %s", info.ReleaseURL)
	}
}
