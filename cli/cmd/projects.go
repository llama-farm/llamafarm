package cmd

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/llamafarm/cli/cmd/config"
	"github.com/llamafarm/cli/cmd/orchestrator"
	"github.com/llamafarm/cli/cmd/utils"
	"github.com/spf13/cobra"
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
		serverCfg, err := config.GetServerConfig(utils.GetEffectiveCWD(), serverURL, namespace, projectID)
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
		orchestrator.EnsureServicesOrExit(serverURL, "server")

		// Build request
		url := buildServerURL(serverURL, fmt.Sprintf("/v1/projects/%s", ns))
		req, err := http.NewRequest(http.MethodGet, url, nil)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error creating request: %v\n", err)
			os.Exit(1)
		}

		// Execute
		resp, err := utils.GetHTTPClient().Do(req)
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

func init() {
	// Add list subcommand to projects
	projectsCmd.AddCommand(projectsListCmd)

	// Add the projects command to root
	rootCmd.AddCommand(projectsCmd)
}
