package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

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
		ensureServerAvailable(serverURL, true)

		// Call API endpoint
		url := fmt.Sprintf("%s/v1/projects/%s/%s/models", strings.TrimSuffix(serverURL, "/"), ns, proj)
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error creating request: %v\n", err)
			os.Exit(1)
		}

		resp, err := getHTTPClient().Do(req)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error fetching models: %v\n", err)
			os.Exit(1)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			fmt.Fprintf(os.Stderr, "Server error %d: %s\n", resp.StatusCode, string(body))
			os.Exit(1)
		}

		var result struct {
			Models []struct {
				ID          string `json:"id"`
				Description string `json:"description"`
				Provider    string `json:"provider"`
				Model       string `json:"model"`
				IsDefault   bool   `json:"is_default"`
			} `json:"models"`
		}

		if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
			fmt.Fprintf(os.Stderr, "Error parsing response: %v\n", err)
			os.Exit(1)
		}

		if len(result.Models) == 0 {
			fmt.Println("No models configured")
			return
		}

		fmt.Printf("Models for %s/%s:\n\n", ns, proj)
		for _, m := range result.Models {
			defaultMarker := ""
			if m.IsDefault {
				defaultMarker = " (default)"
			}
			fmt.Printf("  • %s%s\n", m.ID, defaultMarker)
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
