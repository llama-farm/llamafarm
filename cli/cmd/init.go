package cmd

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"llamafarm-cli/cmd/config"
	"net/http"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"
	"gopkg.in/yaml.v2"
)

// CreateProjectResponse represents the server response when creating a project.
// It contains the created project and its configuration under project.config.
type CreateProjectResponse struct {
	Project struct {
		Config map[string]interface{} `json:"config"`
	} `json:"project"`
}

// initCmd represents the init command
var initCmd = &cobra.Command{
	Use:   "init",
	Short: "Initialize a new LlamaFarm project",
	Long:  `Initialize a new LlamaFarm project in the current directory (or a target path).`,
	Args:  cobra.MaximumNArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Print("Initializing a new LlamaFarm project")
		// Determine target directory
		cwd := getEffectiveCWD()
		projectDir := cwd
		if len(args) > 0 {
			projectDir = args[0]
		}
		if projectDir != "." {
			if err := os.MkdirAll(projectDir, 0755); err != nil {
				fmt.Fprintf(os.Stderr, "Failed to create directory %s: %v\n", projectDir, err)
				os.Exit(1)
			}
		}

		// Derive project name from directory
		absProjectDir, _ := filepath.Abs(projectDir)
		projectName := filepath.Base(absProjectDir)

		fmt.Println(" in", projectDir)

		// Use config.FindConfigFile to check if a llamafarm config file already exists in the target directory
		if configPath, err := config.FindConfigFile(projectDir); err == nil && configPath != "" {
			fmt.Fprintf(os.Stderr, "Error: Project already exists (found %s)\n", configPath)
			os.Exit(1)
		}

		ns := namespace
		if ns == "" {
			ns = "default"
		}

		// Ensure server is available (auto-start locally if needed)
		base := serverURL
		if base == "" {
			base = "http://localhost:8000"
		}
		config := ServerOnlyConfig(base)
		EnsureServicesWithConfig(config)

		// Build URL
		url := buildServerURL(base, fmt.Sprintf("/v1/projects/%s", ns))

		// Prepare payload
		type createProjectRequest struct {
			Name           string  `json:"name"`
			ConfigTemplate *string `json:"config_template,omitempty"`
		}
		var tplPtr *string
		if initConfigTemplate != "" {
			tpl := initConfigTemplate
			tplPtr = &tpl
		}
		bodyBytes, _ := json.Marshal(createProjectRequest{Name: projectName, ConfigTemplate: tplPtr})

		origWD, _ := os.Getwd()
		needChdir := projectDir != "."
		if needChdir {
			if err := os.Chdir(projectDir); err != nil {
				fmt.Fprintf(os.Stderr, "Failed to change directory to %s: %v\n", projectDir, err)
				os.Exit(1)
			}
			defer func() { _ = os.Chdir(origWD) }()
		}

		// Create request
		req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(bodyBytes))
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error creating request: %v\n", err)
			os.Exit(1)
		}
		req.Header.Set("Content-Type", "application/json")

		// Execute
		resp, err := getHTTPClient().Do(req)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error contacting server: %v\n", err)
			os.Exit(1)
		}
		defer resp.Body.Close()
		respBody, _ := io.ReadAll(resp.Body)
		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			fmt.Fprintf(os.Stderr, "Server returned error %d: %s\n", resp.StatusCode, prettyServerError(resp, respBody))
			os.Exit(1)
		}

	// Parse response and write project.config as YAML to absProjectDir/llamafarm.yaml
	var createResp CreateProjectResponse
	if err := json.Unmarshal(respBody, &createResp); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to parse server response: %v\n", err)
		os.Exit(1)
	}

	yamlBytes, err := yaml.Marshal(createResp.Project.Config)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to convert project.config to YAML: %v\n", err)
		os.Exit(1)
	}

	yamlPath := filepath.Join(absProjectDir, "llamafarm.yaml")
	if err := os.WriteFile(yamlPath, yamlBytes, 0644); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to write llamafarm.yaml: %v\n", err)
		os.Exit(1)
	}

		fmt.Printf("Created project %s/%s in %s\n", ns, projectName, absProjectDir)
	},
}

func init() {
	rootCmd.AddCommand(initCmd)
	initCmd.Flags().StringVar(&namespace, "namespace", "", "Project namespace")
	initCmd.Flags().StringVar(&initConfigTemplate, "template", "", "Configuration template to use (optional)")
}

var initConfigTemplate string
