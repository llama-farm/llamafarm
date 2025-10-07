package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

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
		fmt.Printf("LlamaFarm CLI %s\n", formatVersionForDisplay(Version))
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
	fmt.Printf("📋 Upgrade Plan:\n")
	fmt.Printf("   Current version: %s\n", info.CurrentVersion)
	fmt.Printf("   Target version:  %s\n", targetVersion)
	fmt.Printf("   Install location: %s\n", finalInstallDir)
	fmt.Printf("   Platform: %s\n", detectPlatform())

	requiresElevation := strategy.RequiresElevation(finalInstallDir)
	if requiresElevation {
		fmt.Printf("   ⚠️  Requires elevation (sudo/Administrator)\n")
	}

	if !canUpgradeInPlace && installDir == "" {
		// Suggest fallback directory
		fallbackDir, err := strategy.GetFallbackDir()
		if err == nil {
			fmt.Printf("   💡 Suggested fallback: %s\n", fallbackDir)
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

	fmt.Printf("\n❌ Cannot write to %s without elevation\n", finalInstallDir)
	fmt.Printf("\nOptions:\n")
	fmt.Printf("1. Run with elevation: sudo lf version upgrade\n")

	fallbackDir, err := strategy.GetFallbackDir()
	if err == nil {
		fmt.Printf("2. Install to user directory: lf version upgrade --install-dir %s\n", fallbackDir)
	}

	fmt.Printf("3. Manual installation: curl -fsSL https://raw.githubusercontent.com/llama-farm/llamafarm/main/install.sh | bash\n")
	return fmt.Errorf("insufficient permissions for upgrade")
}

// downloadAndVerifyBinary downloads the binary and optionally verifies its checksum
func downloadAndVerifyBinary(targetVersion, platform string, noVerify bool) (string, error) {
	fmt.Fprintf(os.Stderr, "🔄 Downloading binary...\n")
	tempBinary, err := downloadBinary(targetVersion, platform)
	if err != nil {
		return "", fmt.Errorf("failed to download binary: %w", err)
	}

	if !noVerify {
		fmt.Fprintf(os.Stderr, "🔄 Verifying checksum...\n")
		err = verifyChecksum(tempBinary, targetVersion, platform)
		if err != nil {
			cleanupTempFiles([]string{tempBinary})
			return "", fmt.Errorf("checksum verification failed: %w", err)
		}
	} else {
		fmt.Fprintf(os.Stderr, "⚠️  Skipping checksum verification\n")
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

	fmt.Fprintf(os.Stderr, "🔍 Current binary: %s\n", currentBinary)

	// Determine target version
	targetVersion, info, err := determineTargetVersion(args)
	if err != nil {
		return err
	}

	// Check if upgrade is necessary
	if !flags.force && !info.UpdateAvailable && targetVersion == info.CurrentVersionNormalized {
		fmt.Printf("✅ Already running version %s\n", info.CurrentVersion)
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
		fmt.Printf("\n🔍 Dry run mode - no changes will be made\n")
		return nil
	}

	// Check permissions
	if err := checkPermissions(canUpgradeInPlace, flags.installDir, finalInstallDir, strategy); err != nil {
		return err
	}

	// Confirm upgrade
	fmt.Printf("\n🚀 Starting upgrade to %s...\n", targetVersion)

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
	fmt.Fprintf(os.Stderr, "🔄 Installing new version...\n")
	err = strategy.PerformUpgrade(finalBinaryPath, tempBinary)
	if err != nil {
		return fmt.Errorf("upgrade failed: %w", err)
	}

	// Verify installation
	fmt.Fprintf(os.Stderr, "🔄 Verifying installation...\n")
	if err := validateBinaryPath(finalBinaryPath); err != nil {
		return fmt.Errorf("installation verification failed: %w", err)
	}

	fmt.Fprintf(os.Stderr, "✅ Upgrade completed successfully!\n")
	fmt.Printf("\nRun 'lf version' to confirm the new version.\n")

	// Show PATH warning if needed
	if flags.installDir != "" && flags.installDir != filepath.Dir(currentBinary) {
		fmt.Printf("\n💡 Binary installed to: %s\n", finalBinaryPath)
		fmt.Printf("Make sure this directory is in your PATH.\n")
	}

	return nil
}

// showManualInstructions displays manual installation instructions as fallback
func showManualInstructions(info *UpgradeInfo) {
	fmt.Printf("\n📖 Manual Installation Instructions:\n")
	fmt.Printf("  • macOS / Linux: curl -fsSL https://raw.githubusercontent.com/llama-farm/llamafarm/main/install.sh | bash\n")
	fmt.Printf("  • Windows:       winget install LlamaFarm.CLI\n")

	if info.ReleaseURL != "" {
		fmt.Printf("  • Release notes: %s\n", info.ReleaseURL)
	}
}
