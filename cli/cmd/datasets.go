package cmd

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"text/tabwriter"

	"llamafarm-cli/cmd/config"

	"github.com/spf13/cobra"
)

var (
	configFile  string
	ragStrategy string
)

// datasetsCmd represents the datasets command
var datasetsCmd = &cobra.Command{
	Use:   "datasets",
	Short: "Manage datasets in your LlamaFarm configuration",
	Long: `Manage datasets on your LlamaFarm server. Datasets are collections
of files that can be ingested into your RAG system for retrieval-augmented generation.

Available commands:
  list    - List all datasets on the server for a project
  add     - Create a dataset on the server (optionally then upload files)
  remove  - Delete a dataset from the server
  ingest  - Upload files to a dataset on the server`,
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Println("LlamaFarm Datasets Management")
		cmd.Help()
	},
}

// ==== API types (mirroring server) ====
type apiDataset struct {
	Name        string   `json:"name"`
	RAGStrategy string   `json:"rag_strategy"`
	Files       []string `json:"files"`
}

type listDatasetsResponse struct {
	Total    int          `json:"total"`
	Datasets []apiDataset `json:"datasets"`
}

type createDatasetRequest struct {
	Name        string `json:"name"`
	RAGStrategy string `json:"rag_strategy"`
}

type createDatasetResponse struct {
	Dataset apiDataset `json:"dataset"`
}

// datasetsListCmd represents the datasets list command
var datasetsListCmd = &cobra.Command{
	Use:   "list",
	Short: "List all datasets on the server for the selected project",
	Long:  `Lists datasets from the LlamaFarm server scoped by namespace/project.`,
	Run: func(cmd *cobra.Command, args []string) {
		// Resolve server and routing
		serverCfg, err := config.GetServerConfig(configFile, serverURL, namespace, projectID)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

		// Ensure server is up (auto-start locally if needed)
		if err := ensureServerAvailable(serverCfg.URL); err != nil {
			fmt.Fprintf(os.Stderr, "Error ensuring server availability: %v\n", err)
			os.Exit(1)
		}

		url := buildServerURL(serverCfg.URL, fmt.Sprintf("/v1/projects/%s/%s/datasets/", serverCfg.Namespace, serverCfg.Project))
		req, err := http.NewRequest("GET", url, nil)
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
				fmt.Fprintf(os.Stderr, "Error (%d), and body read failed: %v\n", resp.StatusCode, readErr)
				os.Exit(1)
			}
			fmt.Fprintf(os.Stderr, "Error (%d): %s\n", resp.StatusCode, prettyServerError(resp, body))
			os.Exit(1)
		}

		var out listDatasetsResponse
		if err := json.Unmarshal(body, &out); err != nil {
			fmt.Fprintf(os.Stderr, "Failed parsing response: %v\n", err)
			os.Exit(1)
		}

		if out.Total == 0 {
			fmt.Println("No datasets found.")
			return
		}

		fmt.Printf("Found %d dataset(s):\n\n", out.Total)
		w := tabwriter.NewWriter(os.Stdout, 0, 0, 3, ' ', 0)
		fmt.Fprintln(w, "NAME\tRAG STRATEGY\tFILE COUNT")
		fmt.Fprintln(w, "----\t------------\t----------")
		for _, ds := range out.Datasets {
			fmt.Fprintf(w, "%s\t%s\t%d\n", ds.Name, emptyDefault(ds.RAGStrategy, "auto"), len(ds.Files))
		}
		w.Flush()
	},
}

// datasetsAddCmd represents the datasets add command
var datasetsAddCmd = &cobra.Command{
	Use:   "add [name] [file1] [file2] ...",
	Short: "Create a new dataset on the server (optionally upload files)",
	Long: `Create a new dataset on the server for the current project.

Examples:
  lf datasets add my-docs
  lf datasets add --rag-strategy auto my-pdfs ./pdfs/*.pdf`,
	Args: cobra.MinimumNArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		serverCfg, err := config.GetServerConfig(configFile, serverURL, namespace, projectID)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

		datasetName := args[0]
		// 1) Create dataset via API
		if ragStrategy == "" {
			ragStrategy = "auto"
		}
		createReq := createDatasetRequest{Name: datasetName, RAGStrategy: ragStrategy}
		payload, _ := json.Marshal(createReq)
		// Ensure server is up
		if err := ensureServerAvailable(serverCfg.URL); err != nil {
			fmt.Fprintf(os.Stderr, "Error ensuring server availability: %v\n", err)
			os.Exit(1)
		}

		url := buildServerURL(serverCfg.URL, fmt.Sprintf("/v1/projects/%s/%s/datasets/", serverCfg.Namespace, serverCfg.Project))
		req, err := http.NewRequest("POST", url, bytes.NewReader(payload))
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error creating request: %v\n", err)
			os.Exit(1)
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := getHTTPClient().Do(req)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error sending request: %v\n", err)
			os.Exit(1)
		}
		body, readErr := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			if readErr != nil {
				fmt.Fprintf(os.Stderr, "Failed to create dataset '%s' (%d), and body read failed: %v\n", datasetName, resp.StatusCode, readErr)
				os.Exit(1)
			}
			fmt.Fprintf(os.Stderr, "Failed to create dataset '%s' (%d): %s\n", datasetName, resp.StatusCode, prettyServerError(resp, body))
			os.Exit(1)
		}
		var created createDatasetResponse
		if err := json.Unmarshal(body, &created); err != nil {
			fmt.Fprintf(os.Stderr, "Failed parsing response: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("✅ Created dataset '%s' (rag: %s)\n", created.Dataset.Name, emptyDefault(created.Dataset.RAGStrategy, "auto"))

		// 2) Optionally upload files if provided
		filePaths := args[1:]
		if len(filePaths) == 0 {
			return
		}
		var filesToUpload []string
		for _, p := range filePaths {
			matches, err := filepath.Glob(p)
			if err != nil || len(matches) == 0 {
				// if direct path or glob error, include as-is; upload will validate
				filesToUpload = append(filesToUpload, p)
				continue
			}
			filesToUpload = append(filesToUpload, matches...)
		}
		uploaded := 0
		for _, fp := range filesToUpload {
			if err := uploadFileToDataset(serverCfg.URL, serverCfg.Namespace, serverCfg.Project, datasetName, fp); err != nil {
				fmt.Fprintf(os.Stderr, "   ⚠️  Failed to upload '%s': %v\n", fp, err)
				continue
			}
			fmt.Printf("   📤 Uploaded: %s\n", fp)
			uploaded++
		}
		fmt.Printf("   Done. Uploaded %d/%d file(s).\n", uploaded, len(filesToUpload))
	},
}

// datasetsRemoveCmd represents the datasets remove command
var datasetsRemoveCmd = &cobra.Command{
	Use:   "remove [name]",
	Short: "Delete a dataset from the server",
	Long:  `Deletes a dataset from the LlamaFarm server for the selected project.`,
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		serverCfg, err := config.GetServerConfig(configFile, serverURL, namespace, projectID)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		datasetName := args[0]
		// Ensure server is up
		if err := ensureServerAvailable(serverCfg.URL); err != nil {
			fmt.Fprintf(os.Stderr, "Error ensuring server availability: %v\n", err)
			os.Exit(1)
		}
		url := buildServerURL(serverCfg.URL, fmt.Sprintf("/v1/projects/%s/%s/datasets/%s", serverCfg.Namespace, serverCfg.Project, datasetName))
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
				fmt.Fprintf(os.Stderr, "Failed to remove dataset '%s' (%d), and body read failed: %v\n", datasetName, resp.StatusCode, readErr)
				os.Exit(1)
			}
			fmt.Fprintf(os.Stderr, "Failed to remove dataset '%s' (%d): %s\n", datasetName, resp.StatusCode, prettyServerError(resp, body))
			os.Exit(1)
		}
		fmt.Printf("✅ Successfully removed dataset '%s'\n", datasetName)
	},
}

// datasetsIngestCmd represents the datasets ingest command
var datasetsIngestCmd = &cobra.Command{
	Use:   "ingest [dataset-name] [file1] [file2] ...",
	Short: "Upload files to a dataset on the server",
	Long: `Uploads one or more files to the specified dataset on the LlamaFarm server.

Examples:
  lf datasets ingest my-docs ./docs/file1.pdf ./docs/file2.txt
  lf datasets ingest my-docs ./pdfs/*.pdf`,
	Args: cobra.MinimumNArgs(2),
	Run: func(cmd *cobra.Command, args []string) {
		serverCfg, err := config.GetServerConfig(configFile, serverURL, namespace, projectID)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

		datasetName := args[0]
		inPaths := args[1:]
		var files []string
		for _, p := range inPaths {
			matches, err := filepath.Glob(p)
			if err != nil || len(matches) == 0 {
				files = append(files, p)
				continue
			}
			files = append(files, matches...)
		}
		if len(files) == 0 {
			fmt.Fprintf(os.Stderr, "No files to upload.\n")
			os.Exit(1)
		}

		// Ensure server is up
		if err := ensureServerAvailable(serverCfg.URL); err != nil {
			fmt.Fprintf(os.Stderr, "Error ensuring server availability: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Starting upload to dataset '%s' (%d file(s))...\n", datasetName, len(files))
		uploaded := 0
		for _, f := range files {
			if err := uploadFileToDataset(serverCfg.URL, serverCfg.Namespace, serverCfg.Project, datasetName, f); err != nil {
				fmt.Fprintf(os.Stderr, "   ⚠️  Failed to upload '%s': %v\n", f, err)
				continue
			}
			fmt.Printf("   📤 Uploaded: %s\n", f)
			uploaded++
		}
		fmt.Printf("Done. Uploaded %d/%d file(s).\n", uploaded, len(files))
	},
}

func init() {
	// Add persistent flags
	datasetsCmd.PersistentFlags().StringVarP(&configFile, "config", "c", "", "config file path (default: llamafarm.yaml in current directory)")
	// Server routing flags (align with projects chat)
	datasetsCmd.PersistentFlags().StringVar(&serverURL, "server-url", "", "LlamaFarm server URL (default: http://localhost:8000)")
	datasetsCmd.PersistentFlags().StringVar(&namespace, "namespace", "", "Project namespace (default: from llamafarm.yaml)")
	datasetsCmd.PersistentFlags().StringVar(&projectID, "project", "", "Project ID (default: from llamafarm.yaml)")

	// Add flags specific to add command
	datasetsAddCmd.Flags().StringVarP(&ragStrategy, "rag-strategy", "r", "auto", "RAG strategy to use for this dataset (default: auto)")

	// Add subcommands to datasets
	datasetsCmd.AddCommand(datasetsListCmd)
	datasetsCmd.AddCommand(datasetsAddCmd)
	datasetsCmd.AddCommand(datasetsRemoveCmd)
	datasetsCmd.AddCommand(datasetsIngestCmd)

	// Add the datasets command to root
	rootCmd.AddCommand(datasetsCmd)
}

// ==== helpers ====
func emptyDefault(s string, d string) string {
	if strings.TrimSpace(s) == "" {
		return d
	}
	return s
}

func uploadFileToDataset(server string, namespace string, project string, dataset string, path string) error {
	// Open file
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	// Prepare multipart form
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)
	part, err := writer.CreateFormFile("file", filepath.Base(path))
	if err != nil {
		return err
	}
	if _, err := io.Copy(part, file); err != nil {
		return err
	}
	if err := writer.Close(); err != nil {
		return err
	}

	// Build request
	url := buildServerURL(server, fmt.Sprintf("/v1/projects/%s/%s/datasets/%s/data", namespace, project, dataset))
	req, err := http.NewRequest("POST", url, &buf)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	resp, err := getHTTPClient().Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	body, readErr := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		if readErr != nil {
			return fmt.Errorf("%s", readErr.Error())
		}
		return fmt.Errorf("%s", prettyServerError(resp, body))
	}
	return nil
}
