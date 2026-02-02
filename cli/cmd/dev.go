package cmd

import (
	"os"
	"strings"

	"github.com/llamafarm/cli/cmd/config"
	"github.com/llamafarm/cli/cmd/orchestrator"

	"github.com/llamafarm/cli/cmd/utils"
	"github.com/spf13/cobra"
)

// devCmd launches the chat quickly for development at the top level.
var devCmd = &cobra.Command{
	Use:     "dev",
	Short:   "Start working with your project locally",
	Aliases: []string{"start"},
	Long:    "Start your LlamaFarm project locally and open an interactive chat session for development and testing.",
	Run: func(cmd *cobra.Command, args []string) {
		if strings.TrimSpace(serverURL) == "" {
			serverURL = "http://localhost:14345"
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

		start(SessionModeDev)
	},
}

func start(mode SessionMode) {
	// Load config to get namespace and project for watcher
	cwd := utils.GetEffectiveCWD()
	cfg, err := config.LoadConfig(cwd)
	if err != nil {
		utils.OutputError("Error loading config: %v\nRun `lf init` to create a new project if none exists.\n", err)
		os.Exit(1)
	}

	projectInfo, err := cfg.GetProjectInfo()
	if err != nil {
		utils.OutputWarning("Warning: Could not extract project info for watcher: %v\n", err)
	} else {
		// Start the config file watcher in background
		if err := StartConfigWatcher(projectInfo.Namespace, projectInfo.Project); err != nil {
			utils.OutputWarning("Warning: Failed to start config watcher: %v\n", err)
		}
	}

	serverInfo, err := config.GetServerConfig(cwd, serverURL, "", "")
	if err != nil {
		utils.OutputError("Error getting server config: %v\n", err)
		os.Exit(1)
	}
	serverURL = serverInfo.URL

	factory := GetServiceConfigFactory()
	config := factory.ServerOnly(serverURL)
	orchestrator.EnsureServicesOrExitWithConfig(config, "server", "universal-runtime")

	runChatSessionTUI(mode, projectInfo)
}

func init() {
	// Attach to root
	rootCmd.AddCommand(devCmd)

	// Add start-only flags
	devCmd.Flags().StringVar(&ollamaHost, "ollama-host", ollamaHost, "Ollama host URL")
}
