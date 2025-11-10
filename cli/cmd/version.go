package cmd

import (
	"github.com/llamafarm/cli/cmd/utils"
	"github.com/llamafarm/cli/cmd/version"
	"github.com/spf13/cobra"
)

// versionCmd represents the version command
var versionCmd = &cobra.Command{
	Use:   "version",
	Short: "Print the version number of LlamaFarm CLI",
	Long:  "Print the version number of LlamaFarm CLI",
	Run: func(cmd *cobra.Command, args []string) {
		utils.OutputInfo("LlamaFarm CLI %s\n", version.FormatVersionForDisplay(version.CurrentVersion))
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
		return version.PerformUpgrade(parseUpgradeFlags(cmd, args))
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

// parseUpgradeFlags extracts and returns the upgrade command flags
func parseUpgradeFlags(cmd *cobra.Command, args []string) version.UpgradeOpts {
	dryRun, _ := cmd.Flags().GetBool("dry-run")
	force, _ := cmd.Flags().GetBool("force")
	noVerify, _ := cmd.Flags().GetBool("no-verify")
	installDir, _ := cmd.Flags().GetString("install-dir")

	var targetVersion string
	if len(args) > 0 {
		targetVersion = args[0]
	}

	return version.UpgradeOpts{
		DryRun:        dryRun,
		Force:         force,
		NoVerify:      noVerify,
		InstallDir:    installDir,
		TargetVersion: targetVersion,
	}
}
