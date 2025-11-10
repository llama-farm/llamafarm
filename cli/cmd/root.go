package cmd

import (
	"fmt"
	"os"
	"time"

	"github.com/llamafarm/cli/cmd/utils"
	"github.com/llamafarm/cli/cmd/version"
	"github.com/spf13/cobra"
)

var debug bool
var serverURL string = "http://localhost:8000"
var ollamaHost string = "http://localhost:11434"
var serverStartTimeout time.Duration

var rootCmd = &cobra.Command{
	Use:   "lf",
	Short: "LlamaFarm CLI - Grow AI projects from seed to scale",
	Long: `LlamaFarm CLI is a command line interface for managing and interacting
with your LlamaFarm projects. It provides various commands to help you
manage your data, configurations, models, and operations.

Getting started:
  # Create a new project
  lf init my-project

  # Start working with your project locally
  lf start

  # Send a one-time chat prompt to your project
  lf chat "What is LlamaFarm?"`,

	Run: func(cmd *cobra.Command, args []string) {
		// Default behavior when no subcommand is specified
		fmt.Println("Welcome to LlamaFarm!")
		cmd.Help()
	},
	PersistentPreRunE: func(cmd *cobra.Command, args []string) error {
		// Flags are parsed at this point; honor --Debug
		if debug {
			utils.InitDebugLogger("", true)
		}
		return nil
	},
	PersistentPostRunE: func(cmd *cobra.Command, args []string) error {
		// Avoid duplicate output when the user explicitly runs the upgrade command.
		if cmd != nil && cmd.Name() == "upgrade" {
			return nil
		}

		info, err := version.MaybeCheckForUpgrade(false)
		if err != nil {
			utils.LogDebug(fmt.Sprintf("skipping upgrade notification: %v", err))
			return nil
		}
		if info != nil && info.UpdateAvailable && info.CurrentVersionIsSemver {
			fmt.Fprintf(os.Stderr, "🚀 A new LlamaFarm CLI release (%s) is available. Run 'lf version upgrade' for details.\n", info.LatestVersion)
		}
		return nil
	},
}

// Execute adds all child commands to the root command and sets flags appropriately.
// This is called by main.main(). It only needs to happen once to the rootCmd.
func Execute() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func init() {
	// Global persistent flags
	rootCmd.PersistentFlags().BoolVarP(&debug, "debug", "d", false, "Enable Debug output")
	rootCmd.PersistentFlags().StringVar(&serverURL, "server-url", "http://localhost:8000", "LlamaFarm server URL")
	rootCmd.PersistentFlags().DurationVar(&serverStartTimeout, "server-start-timeout", 45*time.Second, "How long to wait for local server to become ready when auto-starting (e.g. 45s, 1m)")
	rootCmd.PersistentFlags().StringVar(&utils.OverrideCwd, "cwd", "", "Override the current working directory for CLI operations")
}
