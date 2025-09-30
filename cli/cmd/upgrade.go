package cmd

import (
	"fmt"
	"os"
	"time"

	"github.com/spf13/cobra"
)

var upgradeCmd = &cobra.Command{
	Use:   "upgrade",
	Short: "Check for the latest LlamaFarm CLI release",
	Long:  "Fetches the latest published LlamaFarm CLI release information and prints upgrade instructions.",
	RunE: func(cmd *cobra.Command, args []string) error {
		info, err := maybeCheckForUpgrade(true)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Unable to check for CLI upgrades: %v\n", err)
			return nil
		}
		if info == nil {
			fmt.Println("No release information is currently available.")
			return nil
		}

		fmt.Printf("Current version: %s\n", info.CurrentVersion)
		fmt.Printf("Latest release: %s\n", info.LatestVersion)
		if !info.PublishedAt.IsZero() {
			fmt.Printf("Published: %s\n", info.PublishedAt.Format(time.RFC1123))
		}
		if info.ReleaseURL != "" {
			fmt.Printf("Release notes: %s\n", info.ReleaseURL)
		}
		fmt.Println()
		if !info.CurrentVersionIsSemver {
			fmt.Println("You are running a development build; stable release information is shown for reference.")
		}

		if info.UpdateAvailable {
			fmt.Println("An upgrade is available! Upgrade instructions:")
		} else {
			fmt.Println("You are running the latest published release. Re-install instructions:")
		}

		fmt.Println("  • macOS / Linux: curl -fsSL https://raw.githubusercontent.com/llama-farm/llamafarm/main/install.sh | bash")
		fmt.Println("  • Windows:      winget install LlamaFarm.CLI")

		if info.UpdateAvailable {
			fmt.Println()
			fmt.Println("Tip: re-run this command after upgrading to confirm the new version.")
		}

		return nil
	},
}

func init() {
	rootCmd.AddCommand(upgradeCmd)
}
