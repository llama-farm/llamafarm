package cmd

import (
	"fmt"
	"os"
	"strings"

	"github.com/llamafarm/cli/cmd/config"
	"github.com/llamafarm/cli/cmd/orchestrator"

	"github.com/llamafarm/cli/cmd/utils"
	"github.com/llamafarm/cli/internal/hfcache"
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

		cwd := utils.GetEffectiveCWD()
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
		factory := GetServiceConfigFactory()
		config := factory.ServerOnly(serverURL)
		orchestrator.EnsureServicesOrExitWithConfig(config, "server")

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

// modelsCachedCmd lists every HuggingFace model present in the local cache.
// Distinct from `lf models list`, which lists models *configured* in a
// LlamaFarm project. This command reads the cache directly and does not
// require the LlamaFarm server to be running.
var modelsCachedCmd = &cobra.Command{
	Use:   "cached",
	Short: "List HuggingFace models cached on this machine",
	Long: `List every HuggingFace model present in the local Hub cache.

This is the on-disk view: it shows every repo that has been downloaded via
'lf models pull', 'huggingface-cli download', or any other tool that writes
the standard HF cache layout. It does not list models configured in a
LlamaFarm project — for that, use 'lf models list'.

This command reads the cache directly and does not boot the LlamaFarm server.

Examples:
  # Show every cached model
  lf models cached`,
	Run: func(cmd *cobra.Command, args []string) {
		repos, err := hfcache.ScanCache()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error scanning cache: %v\n", err)
			os.Exit(1)
		}
		if len(repos) == 0 {
			fmt.Println("No cached models found.")
			return
		}
		fmt.Printf("Cached HuggingFace models (%d):\n\n", len(repos))
		for _, r := range repos {
			fmt.Printf("  • %s\n", r.RepoID)
			fmt.Printf("    Size: %s | Files: %d\n", utils.FormatBytes(r.SizeOnDisk), r.FileCount)
			fmt.Printf("    Path: %s\n\n", r.RepoPath)
		}
	},
}

func init() {
	modelsCmd.AddCommand(modelsListCmd)
	modelsCmd.AddCommand(modelsCachedCmd)
	rootCmd.AddCommand(modelsCmd)
}
