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
		fmt.Printf("LlamaFarm CLI v%s\n", Version)
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

// performUpgrade handles the automatic upgrade process
func performUpgrade(cmd *cobra.Command, args []string) error {
	// Get flags
	dryRun, _ := cmd.Flags().GetBool("dry-run")
	force, _ := cmd.Flags().GetBool("force")
	noVerify, _ := cmd.Flags().GetBool("no-verify")
	installDir, _ := cmd.Flags().GetString("install-dir")

	// Determine target version
	var targetVersion string
	if len(args) > 0 {
		targetVersion = args[0]
		if !isValidVersion(targetVersion) {
			return fmt.Errorf("invalid version format: %s", targetVersion)
		}
		targetVersion = normalizeVersion(targetVersion)
	}

	// Get current binary path
	currentBinary, err := getCurrentBinaryPath()
	if err != nil {
		return fmt.Errorf("failed to determine current binary location: %w", err)
	}

	fmt.Fprintf(os.Stderr, "🔍 Current binary: %s\n", currentBinary)

	// Get version information
	var info *UpgradeInfo
	if targetVersion == "" {
		// Get latest version
		info, err = maybeCheckForUpgrade(true)
		if err != nil {
			return fmt.Errorf("failed to check for updates: %w", err)
		}
		if info == nil {
			return fmt.Errorf("no release information available")
		}
		targetVersion = info.LatestVersionNormalized
	} else {
		// For specific version, create minimal info
		info = &UpgradeInfo{
			CurrentVersion:          Version,
			LatestVersion:           targetVersion,
			LatestVersionNormalized: targetVersion,
			UpdateAvailable:         true,
		}
	}

	// Check if upgrade is necessary
	if !force && !info.UpdateAvailable && targetVersion == info.CurrentVersionNormalized {
		fmt.Printf("✅ Already running version %s\n", info.CurrentVersion)
		return nil
	}

	// Determine installation directory
	var finalInstallDir string
	if installDir != "" {
		finalInstallDir = installDir
	} else {
		finalInstallDir = filepath.Dir(currentBinary)
	}

	// Get upgrade strategy
	strategy := GetUpgradeStrategy()

	// Check if we can upgrade to the current location
	canUpgradeInPlace := strategy.CanUpgrade(currentBinary) && canWriteToLocation(currentBinary)
	requiresElevation := strategy.RequiresElevation(currentBinary)

	// Show upgrade plan
	fmt.Printf("📋 Upgrade Plan:\n")
	fmt.Printf("   Current version: %s\n", info.CurrentVersion)
	fmt.Printf("   Target version:  %s\n", targetVersion)
	fmt.Printf("   Install location: %s\n", finalInstallDir)
	fmt.Printf("   Platform: %s\n", detectPlatform())

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

	if dryRun {
		fmt.Printf("\n🔍 Dry run mode - no changes will be made\n")
		return nil
	}

	// Handle permission issues
	if !canUpgradeInPlace && installDir == "" {
		if requiresElevation {
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
	}

	// Confirm upgrade
	fmt.Printf("\n🚀 Starting upgrade to %s...\n", targetVersion)

	// Download binary
	fmt.Fprintf(os.Stderr, "🔄 Downloading binary...\n")
	platform := detectPlatform()
	tempBinary, err := downloadBinary(targetVersion, platform)
	if err != nil {
		return fmt.Errorf("failed to download binary: %w", err)
	}
	defer cleanupTempFiles([]string{tempBinary})

	// Verify checksum unless disabled
	if !noVerify {
		fmt.Fprintf(os.Stderr, "🔄 Verifying checksum...\n")
		err = verifyChecksum(tempBinary, targetVersion, platform)
		if err != nil {
			return fmt.Errorf("checksum verification failed: %w", err)
		}
	} else {
		fmt.Fprintf(os.Stderr, "⚠️  Skipping checksum verification\n")
	}

	// Determine final binary path
	var finalBinaryPath string
	if installDir != "" {
		// Custom install directory
		binaryName := "lf"
		if strings.Contains(platform, "windows") {
			binaryName += ".exe"
		}
		finalBinaryPath = filepath.Join(installDir, binaryName)

		// Ensure directory exists
		if err := os.MkdirAll(installDir, 0755); err != nil {
			return fmt.Errorf("failed to create install directory: %w", err)
		}
	} else {
		// Use current binary location
		finalBinaryPath = currentBinary
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
	if installDir != "" && installDir != filepath.Dir(currentBinary) {
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
