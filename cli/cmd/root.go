package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/spf13/cobra"
)

var debug bool
var serverURL string = "http://localhost:8000"
var ollamaHost string = "http://localhost:11434"
var serverStartTimeout time.Duration
var overrideCwd string

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
		// Flags are parsed at this point; honor --debug
		if debug {
			InitDebugLogger("")
		}
		return nil
	},
	PersistentPostRunE: func(cmd *cobra.Command, args []string) error {
		// Avoid duplicate output when the user explicitly runs the upgrade command.
		if cmd != nil && cmd.Name() == "upgrade" {
			return nil
		}

		info, err := maybeCheckForUpgrade(false)
		if err != nil {
			logDebug(fmt.Sprintf("skipping upgrade notification: %v", err))
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
	rootCmd.PersistentFlags().BoolVarP(&debug, "debug", "d", false, "Enable debug output")
	rootCmd.PersistentFlags().StringVar(&serverURL, "server-url", "", "LlamaFarm server URL (default: http://localhost:8000)")
	rootCmd.PersistentFlags().DurationVar(&serverStartTimeout, "server-start-timeout", 45*time.Second, "How long to wait for local server to become ready when auto-starting (e.g. 45s, 1m)")
	rootCmd.PersistentFlags().StringVar(&overrideCwd, "cwd", "", "Override the current working directory for CLI operations")

	if debug {
		InitDebugLogger("")
	}
}

// getLFDataDir returns the directory to store LlamaFarm data.
var getLFDataDir = func() (string, error) {
	dataDir := os.Getenv("LF_DATA_DIR")
	if dataDir != "" {
		return dataDir, nil
	}
	if homeDir, err := os.UserHomeDir(); err == nil {
		return filepath.Join(homeDir, ".llamafarm"), nil
	} else {
		return "", fmt.Errorf("getLFDataDir: could not determine home directory: %w", err)
	}
}

// getEffectiveCWD returns the directory to treat as the working directory.
// If the global --cwd flag is provided, it returns its absolute path; otherwise os.Getwd().
func getEffectiveCWD() string {
	if strings.TrimSpace(overrideCwd) != "" {
		if filepath.IsAbs(overrideCwd) {
			return overrideCwd
		}
		abs, err := filepath.Abs(overrideCwd)
		if err != nil {
			return "."
		}
		return abs
	}

	wd, _ := os.Getwd()
	if wd == "" {
		return "."
	}

	return wd
}
