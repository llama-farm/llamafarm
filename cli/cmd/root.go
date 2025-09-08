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
manage your data, configurations, models,and operations.`,
	Run: func(cmd *cobra.Command, args []string) {
		// Default behavior when no subcommand is specified
		fmt.Println("Welcome to LlamaFarm!")
		cmd.Help()
	},
	PersistentPreRunE: func(cmd *cobra.Command, args []string) error {
		// Flags are parsed at this point; honor --debug and --ollama-host
		if debug {
			InitDebugLogger("")
		}
		// Resolve ollamaHost with precedence: flag > env > default
		if !cmd.Flags().Changed("ollama-host") {
			if v := strings.TrimSpace(os.Getenv("OLLAMA_HOST")); v != "" {
				ollamaHost = v
			}
		}
		ollamaHost = strings.TrimSpace(ollamaHost)
		if ollamaHost == "" {
			ollamaHost = "http://localhost:11434"
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
	rootCmd.PersistentFlags().StringVar(&ollamaHost, "ollama-host", ollamaHost, "Ollama host URL (default: OLLAMA_HOST or http://localhost:11434)")
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
