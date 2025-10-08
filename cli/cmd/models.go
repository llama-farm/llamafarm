package cmd

import (
	"fmt"
	"os"
	"strings"

	"llamafarm-cli/cmd/config"

	"github.com/spf13/cobra"
)

// modelsCmd represents the models command namespace
var modelsCmd = &cobra.Command{
	Use:   "models",
	Short: "Manage models and model backends",
	Long: `Manage models, providers, and backends configured in LlamaFarm.

Available commands will include listing models, testing inference, and syncing configs.`,
	Hidden: false,
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Println("LlamaFarm Models Management")
		cmd.Help()
	},
}

var modelsListCmd = &cobra.Command{
	Use:   "list [namespace/project]",
	Short: "List available models for a project",
	Long: `List all configured models for a LlamaFarm project.

Examples:
  # List models for explicit project
  lf models list my-org/my-project

  # List models from current directory config
  lf models list`,
	Run: func(cmd *cobra.Command, args []string) {
		var ns, proj string

		// Parse explicit project if provided
		if len(args) >= 1 && strings.Contains(args[0], "/") {
			parts := strings.SplitN(args[0], "/", 2)
			ns = strings.TrimSpace(parts[0])
			proj = strings.TrimSpace(parts[1])
		}

		cwd := getEffectiveCWD()
		StartConfigWatcherForCommand()

		// Resolve server configuration
		serverCfg, err := config.GetServerConfig(cwd, serverURL, ns, proj)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		serverURL = serverCfg.URL
		ns = serverCfg.Namespace
		proj = serverCfg.Project

		// Ensure server is up
		config := ChatNoRAGConfig(serverURL) // Server only, no need for RAG
		EnsureServicesWithConfig(config)

		// Fetch models using shared function
		models := fetchAvailableModels(serverURL, ns, proj)
		if models == nil {
			fmt.Fprintf(os.Stderr, "Error fetching models from server\n")
			os.Exit(1)
		}

		if len(models) == 0 {
			fmt.Println("No models configured")
			return
		}

		fmt.Printf("Models for %s/%s:\n\n", ns, proj)
		for _, m := range models {
			defaultMarker := ""
			if m.IsDefault {
				defaultMarker = " (default)"
			}
			fmt.Printf("  • %s%s\n", m.Name, defaultMarker)
			if m.Description != "" {
				fmt.Printf("    %s\n", m.Description)
			}
			fmt.Printf("    Provider: %s | Model: %s\n", m.Provider, m.Model)
			fmt.Println()
		}
	},
}

func init() {
	modelsCmd.AddCommand(modelsListCmd)
	rootCmd.AddCommand(modelsCmd)
}
