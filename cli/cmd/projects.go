package cmd

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"llamafarm-cli/cmd/config"

	"github.com/spf13/cobra"
)

// ANSI color helpers (disabled if NO_COLOR is set)
const (
	ansiReset   = "\x1b[0m"
	ansiBold    = "\x1b[1m"
	ansiDim     = "\x1b[2m"
	ansiGreen   = "\x1b[32m"
	ansiYellow  = "\x1b[33m"
	ansiMagenta = "\x1b[35m"
	ansiCyan    = "\x1b[36m"
)

// Chat CLI state variables
var (
	namespace   string
	projectID   string
	sessionID   string
	temperature float64
	maxTokens   int
	streaming   bool
)

// projectsCmd represents the projects command
var projectsCmd = &cobra.Command{
	Use:   "projects",
	Short: "Manage LlamaFarm projects and interact with them",
	Long: `Manage LlamaFarm projects and interact with them through various interfaces.

Available commands:
  chat - Start an interactive chat session with a project`,
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Println("LlamaFarm Projects Management")
		cmd.Help()
	},
}

// projectsListCmd lists projects for a namespace from the server
var projectsListCmd = &cobra.Command{
	Use:     "list",
	Aliases: []string{"ls"},
	Short:   "List projects in a namespace",
	Long:    "List projects available in the specified namespace on the LlamaFarm server.",
	Run: func(cmd *cobra.Command, args []string) {
		// Resolve server URL and namespace (project is not required for list)
		serverCfg, err := config.GetServerConfig(getEffectiveCWD(), serverURL, namespace, projectID)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		serverURL = serverCfg.URL
		ns := strings.TrimSpace(serverCfg.Namespace)

		if ns == "" {
			fmt.Fprintln(os.Stderr, "Error: namespace is required. Provide --namespace or set it in llamafarm.yaml")
			os.Exit(1)
		}

		// Ensure server is up (auto-start locally if needed)
		config := ServerOnlyConfig(serverURL)
		EnsureServicesWithConfig(config)

		// Build request
		url := buildServerURL(serverURL, fmt.Sprintf("/v1/projects/%s", ns))
		req, err := http.NewRequest(http.MethodGet, url, nil)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error creating request: %v\n", err)
			os.Exit(1)
		}

		// Execute
		resp, err := getHTTPClient().Do(req)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error requesting server: %v\n", err)
			os.Exit(1)
		}
		defer resp.Body.Close()
		body, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != http.StatusOK {
			fmt.Fprintf(os.Stderr, "Server returned error %d: %s\n", resp.StatusCode, string(body))
			os.Exit(1)
		}

		var listResp struct {
			Total    int `json:"total"`
			Projects []struct {
				Namespace string `json:"namespace"`
				Name      string `json:"name"`
			} `json:"projects"`
		}
		if err := json.Unmarshal(body, &listResp); err != nil {
			fmt.Fprintf(os.Stderr, "Failed to parse server response: %v\n", err)
			os.Exit(1)
		}

		if listResp.Total == 0 || len(listResp.Projects) == 0 {
			fmt.Printf("No projects found in namespace %s\n", ns)
			return
		}

		for _, p := range listResp.Projects {
			fmt.Printf("%s/%s\n", p.Namespace, p.Name)
		}
	},
}

// projectsDeleteCmd represents the projects delete command
var projectsDeleteCmd = &cobra.Command{
	Use:     "delete [project-id]",
	Aliases: []string{"rm", "remove", "del"},
	Short:   "Delete a project and all its associated resources",
	Long: `Delete a project and all its associated resources from the LlamaFarm server.

This operation is irreversible and will delete all project data.`,
	Args: cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		projectToDelete := args[0]

		// Reuse existing config resolution pattern
		serverCfg, err := config.GetServerConfig(getEffectiveCWD(), serverURL, namespace, projectToDelete)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

		ns := strings.TrimSpace(serverCfg.Namespace)
		if ns == "" {
			fmt.Fprintln(os.Stderr, "Error: namespace is required. Provide --namespace or set it in llamafarm.yaml")
			os.Exit(1)
		}

		// Ensure server is running
		config := ServerOnlyConfig(serverCfg.URL)
		EnsureServicesWithConfig(config)

		// Handle confirmation (follow rag_manage.go pattern)
		force, _ := cmd.Flags().GetBool("force")
		if !force {
			fmt.Printf("⚠️  WARNING: This will permanently delete project '%s/%s' and all associated data\n", ns, projectToDelete)
			fmt.Print("Are you sure? Type 'yes' to confirm: ")

			var response string
			fmt.Scanln(&response)
			if response != "yes" {
				fmt.Println("Operation cancelled")
				return
			}
		}

		// Execute delete (follow datasets.go delete pattern)
		url := buildServerURL(serverCfg.URL, fmt.Sprintf("/v1/projects/%s/%s", ns, projectToDelete))
		req, err := http.NewRequest("DELETE", url, nil)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error creating request: %v\n", err)
			os.Exit(1)
		}

		resp, err := getHTTPClient().Do(req)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error sending request: %v\n", err)
			os.Exit(1)
		}
		defer resp.Body.Close()

		body, readErr := io.ReadAll(resp.Body)
		if resp.StatusCode != http.StatusOK {
			if readErr != nil {
				fmt.Fprintf(os.Stderr, "Failed to delete project '%s/%s' (%d), and body read failed: %v\n",
					ns, projectToDelete, resp.StatusCode, readErr)
				os.Exit(1)
			}
			fmt.Fprintf(os.Stderr, "Failed to delete project '%s/%s' (%d): %s\n",
				ns, projectToDelete, resp.StatusCode, prettyServerError(resp, body))
			os.Exit(1)
		}

		fmt.Printf("✅ Successfully deleted project '%s/%s'\n", ns, projectToDelete)
	},
}

func init() {
	// Server routing flags (align with datasets)
	projectsCmd.PersistentFlags().StringVar(&serverURL, "server-url", "", "LlamaFarm server URL (default: http://localhost:8000)")
	projectsCmd.PersistentFlags().StringVar(&namespace, "namespace", "", "Project namespace (default: from llamafarm.yaml)")
	projectsCmd.PersistentFlags().StringVar(&projectID, "project", "", "Project ID (default: from llamafarm.yaml)")

	// Add delete subcommand with force flag
	projectsDeleteCmd.Flags().BoolP("force", "f", false, "Skip confirmation prompt")

	// Register commands
	projectsCmd.AddCommand(projectsListCmd)
	projectsCmd.AddCommand(projectsDeleteCmd)

	// Add the projects command to root
	rootCmd.AddCommand(projectsCmd)
}
