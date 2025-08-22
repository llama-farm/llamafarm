package cmd

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"
)

// initCmd represents the init command
var initCmd = &cobra.Command{
	Use:   "init",
	Short: "Initialize a new LlamaFarm project",
	Long:  `Initialize a new LlamaFarm project in the current directory (or a target path).`,
	Args:  cobra.MaximumNArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Println("Initializing a new LlamaFarm project...")

		// Determine target directory
		projectDir := "."
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
		var projectName string
		if projectDir == "." {
			if wd, err := os.Getwd(); err == nil {
				projectName = filepath.Base(wd)
			} else {
				fmt.Fprintf(os.Stderr, "Failed to determine working directory: %v\n", err)
				os.Exit(1)
			}
		} else {
			projectName = filepath.Base(projectDir)
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
		if err := ensureServerAvailable(base); err != nil {
			fmt.Fprintf(os.Stderr, "Error ensuring server availability: %v\n", err)
			os.Exit(1)
		}

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

		// Change CWD for localhost header if targeting a different path
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

		// Success message
		absPath := projectDir
		if projectDir == "." {
			if wd, err := os.Getwd(); err == nil {
				absPath = wd
			}
		} else {
			if p, err := filepath.Abs(projectDir); err == nil {
				absPath = p
			}
		}
		fmt.Printf("Created project %s/%s in %s\n", ns, projectName, absPath)
	},
}

func init() {
	rootCmd.AddCommand(initCmd)
	initCmd.Flags().StringVar(&namespace, "namespace", "", "Project namespace")
	initCmd.Flags().StringVar(&initConfigTemplate, "template", "", "Configuration template to use (optional)")
}

var initConfigTemplate string
