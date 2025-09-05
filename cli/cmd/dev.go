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

		// Load config to get namespace and project for watcher
		cwd := getEffectiveCWD()
		cfg, err := config.LoadConfig(cwd)
		if err != nil {
			fmt.Fprintf(os.Stderr, "No config file found in target directory. Run `lf init` to create a new project.\n")
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

		if err := ensureServerAvailable(serverURL); err != nil {
			fmt.Fprintf(os.Stderr, "Error ensuring server availability: %v\n", err)
		}
		runChatSessionTUI(projectInfo)
	},
}

func init() {
	// Attach to root
	rootCmd.AddCommand(devCmd)
	// Provide a hint if server URL isn't set
	if serverURL == "" {
		fmt.Fprintln(os.Stderr, "Hint: use --server-url to point to a specific server")
	}
}
