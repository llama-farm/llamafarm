package cmd

import (
	"fmt"
	"os"
	"strings"

	"llamafarm-cli/cmd/config"

	"github.com/spf13/cobra"
)

// devCmd launches the chat quickly for development at the top level.
var devCmd = &cobra.Command{
	Use:   "dev",
	Short: "Developer mode: launch your project locally",
	Long:  "Start an interactive chat session quickly for development and testing.",
	Run: func(cmd *cobra.Command, args []string) {
		if strings.TrimSpace(serverURL) == "" {
			serverURL = "http://localhost:8000"
		}

		// Resolve ollamaHost for dev: flag > env > default
		if !cmd.Flags().Changed("ollama-host") {
			if v := strings.TrimSpace(os.Getenv("OLLAMA_HOST")); v != "" {
				ollamaHost = v
			}
		}
		if strings.TrimSpace(ollamaHost) == "" {
			ollamaHost = "http://localhost:11434"
		}

		// Load config to get namespace and project for watcher
		cwd := getEffectiveCWD()
		cfg, err := config.LoadConfig(cwd)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error loading config: %v\nRun `lf init` to create a new project if none exists.\n", err)
			os.Exit(1)
		}

		projectInfo, err := cfg.GetProjectInfo()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Warning: Could not extract project info for watcher: %v\n", err)
		} else {
			// Start the config file watcher in background
			if err := StartConfigWatcher(projectInfo.Namespace, projectInfo.Project); err != nil {
				fmt.Fprintf(os.Stderr, "Warning: Failed to start config watcher: %v\n", err)
			}
		}

		serverHealth := ensureServerAvailable(serverURL, false)
		runChatSessionTUI(projectInfo, serverHealth)
	},
}

func init() {
	// Attach to root
	rootCmd.AddCommand(devCmd)

	// Add dev-only flags
	devCmd.Flags().StringVar(&ollamaHost, "ollama-host", ollamaHost, "Ollama host URL")
}
