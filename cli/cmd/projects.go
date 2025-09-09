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
	Use:   "list",
	Short: "List projects in a namespace",
	Long:  "List projects available in the specified namespace on the LlamaFarm server.",
	Run: func(cmd *cobra.Command, args []string) {
		// Resolve config path from persistent flag
		configPath, _ := cmd.Flags().GetString("config")

		// Resolve server URL and namespace (project is not required for list)
		serverCfg, err := config.GetServerConfigLenient(configPath, serverURL, namespace, "")
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
		ensureServerAvailable(serverURL)

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

// chatCmd represents the chat command

func init() {
	// Add persistent flags to projects command
	projectsCmd.PersistentFlags().StringP("config", "c", "", "config file path (default: llamafarm.yaml in current directory)")

	// Add list subcommand to projects
	projectsCmd.AddCommand(projectsListCmd)

	// Add the projects command to root
	rootCmd.AddCommand(projectsCmd)
}
