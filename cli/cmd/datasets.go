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
	configFile             string
	dataProcessingStrategy string
	database               string
	verbose                bool
	recursive              bool
	pattern                string
	parallel               bool
	batchSize              int
)

// datasetsCmd represents the datasets command
var datasetsCmd = &cobra.Command{
	Use:   "datasets",
	Short: "Manage datasets in your LlamaFarm configuration",
	Long: `Manage datasets on your LlamaFarm server. Datasets are collections
of files that can be ingested into your RAG system for retrieval-augmented generation.

Each dataset must specify:
  - A data processing strategy (from rag.data_processing_strategies in your config)
  - A database (from rag.databases in your config)

Available commands:
  list    - List all datasets on the server for a project
  add     - Create a dataset on the server (optionally then upload files)
  remove  - Delete a dataset from the server
  ingest  - Upload files to a dataset on the server
  process - Process uploaded files into the vector database`,
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Println("LlamaFarm Datasets Management")
		cmd.Help()
	},
}

// ==== API types (mirroring server) ====
type apiDataset struct {
	Name                   string   `json:"name"`
	DataProcessingStrategy string   `json:"data_processing_strategy"`
	Database               string   `json:"database"`
	Files                  []string `json:"files"`
}

type listDatasetsResponse struct {
	Total    int          `json:"total"`
	Datasets []apiDataset `json:"datasets"`
}

type createDatasetRequest struct {
	Name                   string `json:"name"`
	DataProcessingStrategy string `json:"data_processing_strategy"`
	Database               string `json:"database"`
}

type createDatasetResponse struct {
	Dataset apiDataset `json:"dataset"`
}

// datasetsListCmd represents the datasets list command
var datasetsListCmd = &cobra.Command{
	Use:     "list",
	Aliases: []string{"ls"},
	Short:   "List all datasets on the server for the selected project",
	Long:    `Lists datasets from the LlamaFarm server scoped by namespace/project.`,
	Run: func(cmd *cobra.Command, args []string) {
		// Start config watcher for this command
		StartConfigWatcherForCommand()

		// Resolve server and routing
		serverCfg, err := config.GetServerConfig(getEffectiveCWD(), serverURL, namespace, projectID)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

		// Ensure server is up (auto-start locally if needed)
		ensureServerAvailable(serverCfg.URL, true)

		url := buildServerURL(serverCfg.URL, fmt.Sprintf("/v1/projects/%s/%s/datasets/?include_extra_details=false", serverCfg.Namespace, serverCfg.Project))
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
		fmt.Fprintln(w, "NAME\tDATA PROCESSING STRATEGY\tDATABASE\tFILE COUNT")
		fmt.Fprintln(w, "----\t------------------------\t--------\t----------")
		for _, ds := range out.Datasets {
			fmt.Fprintf(w, "%s\t%s\t%s\t%d\n", ds.Name, emptyDefault(ds.DataProcessingStrategy, "auto"), emptyDefault(ds.Database, "auto"), len(ds.Files))
		}
		w.Flush()
	},
}

// datasetsAddCmd represents the datasets add command
var datasetsAddCmd = &cobra.Command{
	Use:     "create [name] [file1] [file2] ...",
	Aliases: []string{"add"},
	Short:   "Create a new dataset on the server (optionally upload files)",
	Long: `Create a new dataset on the server for the current project.

Examples:
  lf datasets add --data-processing-strategy pdf_processing --database main_database my-docs
  lf datasets add -s text_processing -b main_database my-pdfs ./pdfs/*.pdf`,
	Args: cobra.MinimumNArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		// Start config watcher for this command
		StartConfigWatcherForCommand()

		serverCfg, err := config.GetServerConfig(getEffectiveCWD(), serverURL, namespace, projectID)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

		datasetName := args[0]
		// 1) Validate required parameters
		if dataProcessingStrategy == "" {
			fmt.Fprintf(os.Stderr, "Error: --data-processing-strategy is required\n")
			os.Exit(1)
		}
		if database == "" {
			fmt.Fprintf(os.Stderr, "Error: --database is required\n")
			os.Exit(1)
		}

		// 2) Validate strategies and databases exist in project config
		ensureServerAvailable(serverCfg.URL, true)
		if err := validateStrategiesAndDatabases(serverCfg.URL, serverCfg.Namespace, serverCfg.Project, dataProcessingStrategy, database); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

		// 3) Create dataset via API
		createReq := createDatasetRequest{
			Name:                   datasetName,
			DataProcessingStrategy: dataProcessingStrategy,
			Database:               database,
		}
		payload, _ := json.Marshal(createReq)

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
		fmt.Printf("✅ Created dataset '%s' (strategy: %s, database: %s)\n", created.Dataset.Name, created.Dataset.DataProcessingStrategy, created.Dataset.Database)

		// 4) Optionally upload files if provided
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
	Use:     "delete [name]",
	Aliases: []string{"rm", "remove", "del"},
	Short:   "Delete a dataset from the server",
	Long:    `Deletes a dataset from the LlamaFarm server for the selected project.`,
	Args:    cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		// Start config watcher for this command
		StartConfigWatcherForCommand()

		serverCfg, err := config.GetServerConfig(getEffectiveCWD(), serverURL, namespace, projectID)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		datasetName := args[0]
		// Ensure server is up
		ensureServerAvailable(serverCfg.URL, true)
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
	Use:   "ingest [dataset-name] [paths...]",
	Short: "Upload files, directories, or glob patterns to a dataset",
	Long: `Upload files to a dataset using various methods:
  - Single files: lf datasets ingest my-dataset file.pdf
  - Multiple files: lf datasets ingest my-dataset file1.pdf file2.txt
  - Glob patterns: lf datasets ingest my-dataset "*.pdf" "docs/*.md"
  - Directories: lf datasets ingest my-dataset /path/to/docs/
  - Mixed: lf datasets ingest my-dataset file.pdf /docs/ "*.txt"

Flags:
  --recursive/-r: Recursively process directories
  --pattern/-p: Filter pattern for directory contents
  --parallel: Process files in parallel (default: true)
  --batch-size: Number of files per batch (default: 10)

Examples:
  lf datasets ingest my-docs ./docs/file1.pdf ./docs/file2.txt
  lf datasets ingest my-docs ./pdfs/*.pdf
  lf datasets ingest my-docs /path/to/docs/ --recursive
  lf datasets ingest my-docs /path/to/docs/ --pattern "*.pdf" --recursive`,
	Args: cobra.MinimumNArgs(2),
	Run: func(cmd *cobra.Command, args []string) {
		// Start config watcher for this command
		StartConfigWatcherForCommand()

		serverCfg, err := config.GetServerConfig(getEffectiveCWD(), serverURL, namespace, projectID)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

		datasetName := args[0]
		inPaths := args[1:]

		// Ensure server is up
		ensureServerAvailable(serverCfg.URL, true)

		// Use the new smart ingest endpoint
		url := buildServerURL(serverCfg.URL, fmt.Sprintf("/v1/projects/%s/%s/datasets/%s/ingest",
			serverCfg.Namespace, serverCfg.Project, datasetName))

		// Determine the best method based on input
		method := determineIngestMethod(inPaths)
		
		fmt.Printf("Starting smart ingest to dataset '%s'...\n", datasetName)
		fmt.Printf("Method: %s, Paths: %v\n", method, inPaths)

		switch method {
		case "files":
			// Upload actual files
			uploadFilesViaSmartEndpoint(url, inPaths)
		case "paths":
			// Send paths to server for processing
			sendPathsToSmartEndpoint(url, inPaths)
		case "mixed":
			// Handle mixed input
			handleMixedInputViaSmartEndpoint(url, inPaths)
		}
	},
}

// datasetsProcessCmd represents the datasets process command
var datasetsProcessCmd = &cobra.Command{
	Use:   "process [dataset-name]",
	Short: "Process uploaded files into the vector database",
	Long:  `Process all uploaded files in the dataset into the vector database using the configured data processing strategy and embeddings.`,
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		// Start config watcher for this command
		StartConfigWatcherForCommand()

		serverCfg, err := config.GetServerConfig(getEffectiveCWD(), serverURL, namespace, projectID)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

		datasetName := args[0]

		// Ensure server is up
		ensureServerAvailable(serverCfg.URL, true)

		fmt.Printf("Processing dataset '%s'...\n", datasetName)

		// Call the process endpoint
		url := buildServerURL(serverCfg.URL, fmt.Sprintf("/v1/projects/%s/%s/datasets/%s/process",
			serverCfg.Namespace, serverCfg.Project, datasetName))

		req, err := http.NewRequest("POST", url, nil)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error creating request: %v\n", err)
			os.Exit(1)
		}

		resp, err := getHTTPClient().Do(req)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error processing dataset: %v\n", err)
			os.Exit(1)
		}
		defer resp.Body.Close()

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading response: %v\n", err)
			os.Exit(1)
		}

		if resp.StatusCode != http.StatusOK {
			fmt.Fprintf(os.Stderr, "Error: %s\n", prettyServerError(resp, body))
			os.Exit(1)
		}

		// Parse response
		var result struct {
			ProcessedFiles int    `json:"processed_files"`
			SkippedFiles   int    `json:"skipped_files"`
			FailedFiles    int    `json:"failed_files"`
			Strategy       string `json:"strategy,omitempty"`
			Database       string `json:"database,omitempty"`
			Details        []struct {
				Hash       string   `json:"hash"`
				Filename   string   `json:"filename,omitempty"`
				Status     string   `json:"status"`
				Parser     string   `json:"parser,omitempty"`
				Extractors []string `json:"extractors,omitempty"`
				Chunks     *int     `json:"chunks,omitempty"`
				ChunkSize  *int     `json:"chunk_size,omitempty"`
				Embedder   string   `json:"embedder,omitempty"`
				Error      string   `json:"error,omitempty"`
				Reason     string   `json:"reason,omitempty"`
			} `json:"details"`
		}
		if err := json.Unmarshal(body, &result); err != nil {
			fmt.Fprintf(os.Stderr, "Error parsing response: %v\n", err)
			os.Exit(1)
		}

		// Display results - always show configuration
		fmt.Printf("\n📊 Processing Configuration:\n")
		if result.Strategy != "" {
			fmt.Printf("   Strategy: %s\n", result.Strategy)
		}
		if result.Database != "" {
			fmt.Printf("   Database: %s\n", result.Database)
		}

		// Always show detailed processing info
		fmt.Printf("\n📁 File Processing Details:\n")
		fmt.Printf("────────────────────────────────────────────────────────────────────────\n")
		for i, d := range result.Details {
			// Show both filename and hash for clarity
			identifier := ""
			if d.Filename != "" {
				identifier = fmt.Sprintf("%s", d.Filename)
				if len(d.Hash) > 12 {
					identifier += fmt.Sprintf(" [%s...]", d.Hash[:12])
				}
			} else {
				// Just show full hash if no filename
				identifier = d.Hash
			}

			// Color-code status
			statusDisplay := d.Status
			statusBadge := ""
			if d.Status == "processed" {
				statusDisplay = "PROCESSED"
				statusBadge = "✅"
			} else if d.Status == "skipped" {
				statusDisplay = "SKIPPED"
				statusBadge = "⏭️"
			} else if d.Status == "failed" {
				statusDisplay = "FAILED"
				statusBadge = "❌"
			}

			// File header with number, status badge, and identifier
			fmt.Printf("\n   %s [%d] %s\n", statusBadge, i+1, identifier)
			fmt.Printf("       ├─ Status: %s\n", statusDisplay)

			if d.Status == "processed" {
				// Parser information
				if d.Parser != "" {
					fmt.Printf("       ├─ Parser: %s\n", d.Parser)
				}

				// Chunks information - show more detail
				if d.Chunks != nil {
					chunkInfo := fmt.Sprintf("%d chunks created", *d.Chunks)
					if d.ChunkSize != nil {
						chunkInfo += fmt.Sprintf(" (target size: %d chars)", *d.ChunkSize)
					}
					fmt.Printf("       ├─ Chunking: %s\n", chunkInfo)
				}

				// Extractors - show count and types
				if len(d.Extractors) > 0 {
					fmt.Printf("       ├─ Extractors: %d applied\n", len(d.Extractors))
					// Show first 3 extractors inline, rest on new lines
					if len(d.Extractors) <= 3 {
						fmt.Printf("       │   └─ %s\n", strings.Join(d.Extractors, ", "))
					} else {
						for j, ext := range d.Extractors {
							if j < len(d.Extractors)-1 {
								fmt.Printf("       │   ├─ %s\n", ext)
							} else {
								fmt.Printf("       │   └─ %s\n", ext)
							}
						}
					}
				}

				// Embedder information
				if d.Embedder != "" {
					fmt.Printf("       └─ Embedder: %s\n", d.Embedder)
				}
			} else if d.Status == "skipped" {
				if d.Reason == "duplicate" {
					fmt.Printf("       ├─ Reason: All chunks already exist in database\n")
					fmt.Printf("       └─ Action: No new data added (file previously processed)\n")
				} else if d.Reason != "" {
					fmt.Printf("       └─ Reason: %s\n", d.Reason)
				}
				// Still show what would have been used
				if d.Parser != "" {
					fmt.Printf("       Would use parser: %s\n", d.Parser)
				}
				if d.Embedder != "" {
					fmt.Printf("       Would use embedder: %s\n", d.Embedder)
				}
			} else if d.Status == "failed" {
				if d.Error != "" {
					fmt.Printf("       Error: %s\n", d.Error)
				}
			}
		}

		// Summary with more context
		fmt.Printf("\n────────────────────────────────────────────────────────────────────────\n")
		totalFiles := result.ProcessedFiles + result.SkippedFiles + result.FailedFiles
		if totalFiles == 0 {
			fmt.Printf("\n⚠️  No files to process\n")
		} else if result.FailedFiles > 0 {
			fmt.Printf("\n⚠️  Processing Complete with Errors:\n")
		} else if result.SkippedFiles == totalFiles {
			fmt.Printf("\n✓ Processing Complete (All files already in database):\n")
		} else {
			fmt.Printf("\n✅ Processing Complete:\n")
		}

		fmt.Printf("   📊 Total files: %d\n", totalFiles)
		if result.ProcessedFiles > 0 {
			fmt.Printf("   ✅ Successfully processed: %d\n", result.ProcessedFiles)
			// Calculate total chunks from details
			totalChunks := 0
			for _, d := range result.Details {
				if d.Status == "processed" && d.Chunks != nil {
					totalChunks += *d.Chunks
				}
			}
			if totalChunks > 0 {
				fmt.Printf("   📝 Total chunks created: %d\n", totalChunks)
			}
		}
		if result.SkippedFiles > 0 {
			fmt.Printf("   ⏭️  Skipped (duplicates): %d\n", result.SkippedFiles)
		}
		if result.FailedFiles > 0 {
			fmt.Printf("   ❌ Failed: %d\n", result.FailedFiles)
		}
	},
}

func init() {

	// Server routing flags (align with projects chat)
	datasetsCmd.PersistentFlags().StringVar(&serverURL, "server-url", "", "LlamaFarm server URL (default: http://localhost:8000)")
	datasetsCmd.PersistentFlags().StringVar(&namespace, "namespace", "", "Project namespace (default: from llamafarm.yaml)")
	datasetsCmd.PersistentFlags().StringVar(&projectID, "project", "", "Project ID (default: from llamafarm.yaml)")

	// Add flags specific to add command
	datasetsAddCmd.Flags().StringVarP(&dataProcessingStrategy, "data-processing-strategy", "s", "", "Data processing strategy to use for this dataset (required)")
	datasetsAddCmd.Flags().StringVarP(&database, "database", "b", "", "Database to use for this dataset (required)")

	// Mark flags as required
	datasetsAddCmd.MarkFlagRequired("data-processing-strategy")
	datasetsAddCmd.MarkFlagRequired("database")

	// Add flags specific to ingest command
	datasetsIngestCmd.Flags().BoolVarP(&recursive, "recursive", "r", false, "Recursively process directories")
	datasetsIngestCmd.Flags().StringVarP(&pattern, "pattern", "p", "", "Filter pattern for directory contents (e.g., '*.pdf')")
	datasetsIngestCmd.Flags().BoolVar(&parallel, "parallel", true, "Process files in parallel")
	datasetsIngestCmd.Flags().IntVar(&batchSize, "batch-size", 10, "Number of files to upload in each batch")

	// Add subcommands to datasets
	datasetsCmd.AddCommand(datasetsListCmd)
	datasetsCmd.AddCommand(datasetsAddCmd)
	datasetsCmd.AddCommand(datasetsRemoveCmd)
	datasetsCmd.AddCommand(datasetsIngestCmd)
	datasetsCmd.AddCommand(datasetsProcessCmd)

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

// ==== Validation helpers ====

// availableStrategiesResponse represents the server response for available strategies
type availableStrategiesResponse struct {
	DataProcessingStrategies []string `json:"data_processing_strategies"`
	Databases                []string `json:"databases"`
}

// validateStrategiesAndDatabases validates that the specified strategies exist in the project
func validateStrategiesAndDatabases(serverURL, namespace, project, dataProcessingStrategy, database string) error {
	// Get available strategies from server
	url := buildServerURL(serverURL, fmt.Sprintf("/v1/projects/%s/%s/datasets/strategies", namespace, project))
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		// If we can't validate, continue anyway (graceful degradation)
		fmt.Printf("⚠️  Warning: Could not validate strategies: %v\n", err)
		return nil
	}

	resp, err := getHTTPClient().Do(req)
	if err != nil {
		// If we can't validate, continue anyway (graceful degradation)
		fmt.Printf("⚠️  Warning: Could not validate strategies: %v\n", err)
		return nil
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		// If endpoint doesn't exist or returns error, continue anyway
		fmt.Printf("⚠️  Warning: Could not validate strategies (server returned %d)\n", resp.StatusCode)
		return nil
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		fmt.Printf("⚠️  Warning: Could not read validation response: %v\n", err)
		return nil
	}

	var strategies availableStrategiesResponse
	if err := json.Unmarshal(body, &strategies); err != nil {
		fmt.Printf("⚠️  Warning: Could not parse validation response: %v\n", err)
		return nil
	}

	// Validate data processing strategy
	found := false
	for _, s := range strategies.DataProcessingStrategies {
		if s == dataProcessingStrategy {
			found = true
			break
		}
	}
	if !found && len(strategies.DataProcessingStrategies) > 0 {
		return fmt.Errorf("data processing strategy '%s' not found. Available strategies: %s",
			dataProcessingStrategy, strings.Join(strategies.DataProcessingStrategies, ", "))
	}

	// Validate database
	found = false
	for _, db := range strategies.Databases {
		if db == database {
			found = true
			break
		}
	}
	if !found && len(strategies.Databases) > 0 {
		return fmt.Errorf("database '%s' not found. Available databases: %s",
			database, strings.Join(strategies.Databases, ", "))
	}

	return nil
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

// determineIngestMethod analyzes the input paths to determine the best ingestion method
func determineIngestMethod(paths []string) string {
	hasFiles := false
	hasDirs := false
	hasPatterns := false

	for _, path := range paths {
		// Check if it contains glob patterns
		if strings.Contains(path, "*") || strings.Contains(path, "?") || strings.Contains(path, "[") {
			hasPatterns = true
		} else {
			// Check if it's a file or directory
			info, err := os.Stat(path)
			if err == nil {
				if info.IsDir() {
					hasDirs = true
				} else {
					hasFiles = true
				}
			} else {
				// If file doesn't exist, check if parent directory exists
				// (could be a pattern in a valid directory)
				dir := filepath.Dir(path)
				if _, err := os.Stat(dir); err == nil {
					hasPatterns = true
				}
			}
		}
	}

	// Determine method based on what we found
	if hasFiles && !hasDirs && !hasPatterns {
		return "files"
	} else if (hasDirs || hasPatterns) && !hasFiles {
		return "paths"
	} else {
		return "mixed"
	}
}

// uploadFilesViaSmartEndpoint uploads actual files through the smart endpoint
func uploadFilesViaSmartEndpoint(url string, paths []string) {
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	// Add each file to the multipart form
	successCount := 0
	failCount := 0
	for _, path := range paths {
		file, err := os.Open(path)
		if err != nil {
			fmt.Printf("  ❌ Failed to open file %s: %v\n", path, err)
			failCount++
			continue
		}
		defer file.Close()

		part, err := writer.CreateFormFile("files", filepath.Base(path))
		if err != nil {
			fmt.Printf("  ❌ Failed to create form for %s: %v\n", path, err)
			failCount++
			continue
		}

		if _, err := io.Copy(part, file); err != nil {
			fmt.Printf("  ❌ Failed to read file %s: %v\n", path, err)
			failCount++
			continue
		}
		successCount++
	}

	// Add flags
	if recursive {
		writer.WriteField("recursive", "true")
	}
	if pattern != "" {
		writer.WriteField("pattern", pattern)
	}
	if parallel {
		writer.WriteField("parallel", "true")
	}
	writer.WriteField("batch_size", fmt.Sprintf("%d", batchSize))

	if err := writer.Close(); err != nil {
		fmt.Fprintf(os.Stderr, "Error creating request: %v\n", err)
		os.Exit(1)
	}

	// Send request
	req, err := http.NewRequest("POST", url, &buf)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating request: %v\n", err)
		os.Exit(1)
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	resp, err := getHTTPClient().Do(req)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error uploading files: %v\n", err)
		os.Exit(1)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading response: %v\n", err)
		os.Exit(1)
	}

	if resp.StatusCode != http.StatusOK {
		fmt.Fprintf(os.Stderr, "Error: %s\n", prettyServerError(resp, body))
		os.Exit(1)
	}

	// Parse and display response
	displayIngestResponse(body)
}

// sendPathsToSmartEndpoint sends paths for server-side processing
func sendPathsToSmartEndpoint(url string, paths []string) {
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	// Add paths as a JSON array
	pathsJSON, err := json.Marshal(paths)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error marshaling paths: %v\n", err)
		os.Exit(1)
	}
	writer.WriteField("paths", string(pathsJSON))

	// Add flags
	if recursive {
		writer.WriteField("recursive", "true")
	}
	if pattern != "" {
		writer.WriteField("pattern", pattern)
	}
	if parallel {
		writer.WriteField("parallel", "true")
	}
	writer.WriteField("batch_size", fmt.Sprintf("%d", batchSize))

	if err := writer.Close(); err != nil {
		fmt.Fprintf(os.Stderr, "Error creating request: %v\n", err)
		os.Exit(1)
	}

	// Send request
	req, err := http.NewRequest("POST", url, &buf)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating request: %v\n", err)
		os.Exit(1)
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	resp, err := getHTTPClient().Do(req)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error sending paths: %v\n", err)
		os.Exit(1)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading response: %v\n", err)
		os.Exit(1)
	}

	if resp.StatusCode != http.StatusOK {
		fmt.Fprintf(os.Stderr, "Error: %s\n", prettyServerError(resp, body))
		os.Exit(1)
	}

	// Parse and display response
	displayIngestResponse(body)
}

// handleMixedInputViaSmartEndpoint handles mixed files and paths
func handleMixedInputViaSmartEndpoint(url string, paths []string) {
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	var filePaths []string
	var serverPaths []string

	// Separate files and paths
	for _, path := range paths {
		// Check if it's a glob pattern
		if strings.Contains(path, "*") || strings.Contains(path, "?") || strings.Contains(path, "[") {
			serverPaths = append(serverPaths, path)
			continue
		}

		// Check if it exists
		info, err := os.Stat(path)
		if err != nil {
			// If doesn't exist, treat as pattern
			serverPaths = append(serverPaths, path)
			continue
		}

		if info.IsDir() {
			serverPaths = append(serverPaths, path)
		} else {
			filePaths = append(filePaths, path)
		}
	}

	// Add files to multipart
	for _, path := range filePaths {
		file, err := os.Open(path)
		if err != nil {
			fmt.Printf("  ⚠️ Skipping file %s: %v\n", path, err)
			continue
		}
		defer file.Close()

		part, err := writer.CreateFormFile("files", filepath.Base(path))
		if err != nil {
			fmt.Printf("  ⚠️ Failed to create form for %s: %v\n", path, err)
			continue
		}

		if _, err := io.Copy(part, file); err != nil {
			fmt.Printf("  ⚠️ Failed to read file %s: %v\n", path, err)
			continue
		}
	}

	// Add server paths
	if len(serverPaths) > 0 {
		pathsJSON, err := json.Marshal(serverPaths)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error marshaling paths: %v\n", err)
			os.Exit(1)
		}
		writer.WriteField("paths", string(pathsJSON))
	}

	// Add flags
	if recursive {
		writer.WriteField("recursive", "true")
	}
	if pattern != "" {
		writer.WriteField("pattern", pattern)
	}
	if parallel {
		writer.WriteField("parallel", "true")
	}
	writer.WriteField("batch_size", fmt.Sprintf("%d", batchSize))

	if err := writer.Close(); err != nil {
		fmt.Fprintf(os.Stderr, "Error creating request: %v\n", err)
		os.Exit(1)
	}

	// Send request
	req, err := http.NewRequest("POST", url, &buf)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating request: %v\n", err)
		os.Exit(1)
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	resp, err := getHTTPClient().Do(req)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error sending request: %v\n", err)
		os.Exit(1)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading response: %v\n", err)
		os.Exit(1)
	}

	if resp.StatusCode != http.StatusOK {
		fmt.Fprintf(os.Stderr, "Error: %s\n", prettyServerError(resp, body))
		os.Exit(1)
	}

	// Parse and display response
	displayIngestResponse(body)
}

// displayIngestResponse parses and displays the batch ingest response
func displayIngestResponse(body []byte) {
	var result struct {
		Total   int `json:"total"`
		Success int `json:"success"`
		Failed  int `json:"failed"`
		Items   []struct {
			Path    string `json:"path"`
			Status  string `json:"status"`
			Message string `json:"message,omitempty"`
			Hash    string `json:"hash,omitempty"`
		} `json:"items"`
	}

	if err := json.Unmarshal(body, &result); err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing response: %v\n", err)
		os.Exit(1)
	}

	// Display summary
	fmt.Printf("\n📊 Ingestion Summary:\n")
	fmt.Printf("   Total: %d files\n", result.Total)
	fmt.Printf("   ✅ Success: %d\n", result.Success)
	if result.Failed > 0 {
		fmt.Printf("   ❌ Failed: %d\n", result.Failed)
	}

	// Display details if available
	if len(result.Items) > 0 {
		fmt.Printf("\n📁 File Details:\n")
		fmt.Printf("────────────────────────────────────────────────────────────────────────\n")
		for _, item := range result.Items {
			statusBadge := "✅"
			if item.Status == "failed" {
				statusBadge = "❌"
			} else if item.Status == "duplicate" {
				statusBadge = "⏭️"
			}

			fmt.Printf("   %s %s\n", statusBadge, item.Path)
			if item.Message != "" {
				fmt.Printf("      └─ %s\n", item.Message)
			}
			if item.Hash != "" && len(item.Hash) > 12 {
				fmt.Printf("      └─ Hash: %s...\n", item.Hash[:12])
			}
		}
	}

	if result.Failed > 0 {
		os.Exit(1)
	}
}
