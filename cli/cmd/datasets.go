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
	"sync"
	"text/tabwriter"
	"time"

	"github.com/llamafarm/cli/cmd/config"
	"github.com/llamafarm/cli/cmd/orchestrator"

	"github.com/llamafarm/cli/cmd/utils"
	"github.com/spf13/cobra"
	"golang.org/x/term"
)

var (
	configFile             string
	dataProcessingStrategy string
	database               string
	verbose                bool
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
  list    - List all datasets for a project
  create  - Create a dataset (optionally, specify files to upload)
  delete  - Delete a dataset
  upload  - Upload files to a dataset
  process - Process uploaded files using the data processing strategy and database embeddings`,
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
		// Load config first to ensure it's valid before starting watcher
		serverCfg, err := config.GetServerConfig(utils.GetEffectiveCWD(), serverURL, namespace, projectID)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

		// Start config watcher AFTER we've successfully loaded the config
		// This prevents race conditions where the watcher syncs files before we read them
		StartConfigWatcher(serverCfg.Namespace, serverCfg.Project)

		// Ensure required services are running
		orchestrator.EnsureServicesOrExit(serverURL, "server")

		url := buildServerURL(serverCfg.URL, fmt.Sprintf("/v1/projects/%s/%s/datasets/?include_extra_details=false", serverCfg.Namespace, serverCfg.Project))
		req, err := http.NewRequest("GET", url, nil)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error creating request: %v\n", err)
			os.Exit(1)
		}
		resp, err := utils.GetHTTPClient().Do(req)
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
			fmt.Fprintf(os.Stderr, "Error (%d): %s\n", resp.StatusCode, utils.PrettyServerError(resp, body))
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

// datasetsCreateCmd represents the datasets create command
var datasetsCreateCmd = &cobra.Command{
	Use:     "create [name] [file1] [file2] ...",
	Aliases: []string{"add"},
	Short:   "Create a new dataset on the server (optionally upload files)",
	Long: `Create a new dataset on the server for the current project.

Examples:
  lf datasets create --data-processing-strategy pdf_processing --database main_database my-docs
  lf datasets create -s text_processing -b main_database my-pdfs ./pdfs/*.pdf`,
	Args: cobra.MinimumNArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		// Load config first to ensure it's valid before starting watcher
		serverCfg, err := config.GetServerConfig(utils.GetEffectiveCWD(), serverURL, namespace, projectID)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

		// Start config watcher AFTER we've successfully loaded the config
		// This prevents race conditions where the watcher syncs files before we read them
		StartConfigWatcher(serverCfg.Namespace, serverCfg.Project)

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
		orchestrator.EnsureServicesOrExit(serverCfg.URL, "server")
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
		resp, err := utils.GetHTTPClient().Do(req)
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
			fmt.Fprintf(os.Stderr, "Failed to create dataset '%s' (%d): %s\n", datasetName, resp.StatusCode, utils.PrettyServerError(resp, body))
			os.Exit(1)
		}
		var created createDatasetResponse
		if err := json.Unmarshal(body, &created); err != nil {
			fmt.Fprintf(os.Stderr, "Failed parsing response: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("✅ Created dataset '%s' (strategy: %s, database: %s)\n", created.Dataset.Name, created.Dataset.DataProcessingStrategy, created.Dataset.Database)

		// Ensure config is synced after creation so next commands see the new dataset
		if err := EnsureConfigSynced(serverCfg.Namespace, serverCfg.Project); err != nil {
			utils.LogDebug(fmt.Sprintf("Warning: Failed to sync config after dataset creation: %v", err))
		}

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
		skipped := 0
		failed := 0
		for _, fp := range filesToUpload {
			result := uploadFileToDataset(serverCfg.URL, serverCfg.Namespace, serverCfg.Project, datasetName, fp)
			if result.err != nil {
				fmt.Fprintf(os.Stderr, "   ⚠️  Failed to upload '%s': %v\n", fp, result.err)
				failed++
				continue
			}
			if result.skipped {
				fmt.Printf("   ⏭️  Skipped: %s (already in dataset)\n", fp)
				skipped++
			} else {
				fmt.Printf("   📤 Uploaded: %s\n", fp)
				uploaded++
			}
		}
		fmt.Printf("   Done. Uploaded %d", uploaded)
		if skipped > 0 {
			fmt.Printf(", skipped %d duplicate(s)", skipped)
		}
		if failed > 0 {
			fmt.Printf(", failed %d", failed)
		}
		fmt.Printf(" out of %d file(s).\n", len(filesToUpload))
	},
}

// datasetsDeleteCommand represents the datasets remove command
var datasetsDeleteCommand = &cobra.Command{
	Use:     "delete [name]",
	Aliases: []string{"rm", "remove", "del"},
	Short:   "Delete a dataset from the server",
	Long:    `Deletes a dataset from the LlamaFarm server for the selected project.`,
	Args:    cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		// Load config first to ensure it's valid before starting watcher
		serverCfg, err := config.GetServerConfig(utils.GetEffectiveCWD(), serverURL, namespace, projectID)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

		// Start config watcher AFTER we've successfully loaded the config
		StartConfigWatcher(serverCfg.Namespace, serverCfg.Project)

		datasetName := args[0]
		// Ensure server is up
		orchestrator.EnsureServicesOrExit(serverCfg.URL, "server")
		url := buildServerURL(serverCfg.URL, fmt.Sprintf("/v1/projects/%s/%s/datasets/%s", serverCfg.Namespace, serverCfg.Project, datasetName))
		req, err := http.NewRequest("DELETE", url, nil)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error creating request: %v\n", err)
			os.Exit(1)
		}
		resp, err := utils.GetHTTPClient().Do(req)
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
			fmt.Fprintf(os.Stderr, "Failed to remove dataset '%s' (%d): %s\n", datasetName, resp.StatusCode, utils.PrettyServerError(resp, body))
			os.Exit(1)
		}
		fmt.Printf("✅ Successfully removed dataset '%s'\n", datasetName)
	},
}

// datasetsUploadCmd represents the datasets ingest command
var datasetsUploadCmd = &cobra.Command{
	Use:     "upload [dataset-name] [file1] [file2] [dir/] ...",
	Aliases: []string{"ingest"},
	Short:   "Upload files or directories to a dataset on the server",
	Long: `Uploads one or more files or directories to the specified dataset on the LlamaFarm server.

Supports:
  - Single files: ./file.pdf
  - Multiple files: file1.txt file2.md
  - Glob patterns: *.pdf, docs/*.txt
  - Directories: ./docs/ (files in directory only)
  - Recursive patterns: ./docs/**/*.txt (includes subdirectories)
  - Mixed: ./docs/ *.pdf specific.txt

Examples:
  lf datasets upload my-docs ./docs/file1.pdf ./docs/file2.txt
  lf datasets upload my-docs ./pdfs/*.pdf
  lf datasets upload my-docs ./documents/              # All files in directory only
  lf datasets upload my-docs ./documents/**/*          # Include all files in subdirectories
  lf datasets upload my-docs ./docs/**/*.pdf           # All PDFs in docs and subdirectories
  lf datasets upload my-docs ./docs/ *.pdf README.md   # Mixed sources`,
	Args: cobra.MinimumNArgs(2),
	Run: func(cmd *cobra.Command, args []string) {
		// Load config first to ensure it's valid before starting watcher
		serverCfg, err := config.GetServerConfig(utils.GetEffectiveCWD(), serverURL, namespace, projectID)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

		// Start config watcher AFTER we've successfully loaded the config
		// This prevents race conditions where the watcher syncs files before we read them
		StartConfigWatcher(serverCfg.Namespace, serverCfg.Project)

		datasetName := args[0]
		inPaths := args[1:]

		// Expand all paths to get actual files
		fmt.Println("Expanding paths to find files...")
		files, err := expandPathsToFiles(inPaths)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error expanding paths: %v\n", err)
			os.Exit(1)
		}

		if len(files) == 0 {
			fmt.Fprintf(os.Stderr, "No files found to upload.\n")
			os.Exit(1)
		}

		fmt.Printf("Found %d files to upload\n", len(files))

		// Ensure server is up
		orchestrator.EnsureServicesOrExit(serverCfg.URL, "server")

		// Upload in batches with progress display
		const batchSize = 10
		totalBatches := (len(files) + batchSize - 1) / batchSize

		uploaded := 0
		skipped := 0
		failed := 0

		for batchNum := 0; batchNum < totalBatches; batchNum++ {
			start := batchNum * batchSize
			end := start + batchSize
			if end > len(files) {
				end = len(files)
			}

			batchFiles := files[start:end]
			fmt.Printf("\n📦 Uploading batch %d/%d (%d files)\n", batchNum+1, totalBatches, len(batchFiles))

			for _, f := range batchFiles {
				relPath := f
				// Try to show relative path for cleaner output
				if cwd, err := os.Getwd(); err == nil {
					if rel, err := filepath.Rel(cwd, f); err == nil && !strings.HasPrefix(rel, "..") {
						relPath = rel
					}
				}

				result := uploadFileToDataset(serverCfg.URL, serverCfg.Namespace, serverCfg.Project, datasetName, f)
				if result.err != nil {
					fmt.Fprintf(os.Stderr, "   ❌ Failed: %s (%v)\n", relPath, result.err)
					failed++
					continue
				}
				if result.skipped {
					fmt.Printf("   ⏭️  Skipped: %s (already in dataset)\n", relPath)
					skipped++
				} else {
					fmt.Printf("   ✅ Uploaded: %s\n", relPath)
					uploaded++
				}
			}
		}

		// Final summary
		fmt.Printf("\n📊 Final Summary:\n")
		fmt.Printf("   Total files: %d\n", len(files))
		fmt.Printf("   ✅ Successful: %d\n", uploaded)
		if skipped > 0 {
			fmt.Printf("   ⏭️  Skipped (duplicates): %d\n", skipped)
		}
		if failed > 0 {
			fmt.Printf("   ❌ Failed: %d\n", failed)
		}

		// Ensure config is synced after uploads (server updates file list in config)
		if err := EnsureConfigSynced(serverCfg.Namespace, serverCfg.Project); err != nil {
			utils.LogDebug(fmt.Sprintf("Warning: Failed to sync config after uploads: %v", err))
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
		// Load config first to ensure it's valid before starting watcher
		serverCfg, err := config.GetServerConfig(utils.GetEffectiveCWD(), serverURL, namespace, projectID)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

		// Start config watcher AFTER we've successfully loaded the config
		// This prevents race conditions where the watcher syncs files before we read them
		StartConfigWatcher(serverCfg.Namespace, serverCfg.Project)

		// Ensure config is synced BEFORE processing (server may have updated it)
		if err := EnsureConfigSynced(serverCfg.Namespace, serverCfg.Project); err != nil {
			utils.LogDebug(fmt.Sprintf("Warning: Failed to sync config before processing: %v", err))
		}

		datasetName := args[0]

		// Ensure server and RAG are up (process command needs RAG for ingestion)
		orchestrator.EnsureServicesOrExit(serverCfg.URL, "server", "rag", "universal-runtime")

		// Call the dataset actions endpoint
		url := buildServerURL(serverCfg.URL, fmt.Sprintf("/v1/projects/%s/%s/datasets/%s/actions",
			serverCfg.Namespace, serverCfg.Project, datasetName))

		payload := map[string]string{"action_type": "process"}
		payloadBytes, err := json.Marshal(payload)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error encoding request: %v\n", err)
			os.Exit(1)
		}

		req, err := http.NewRequest("POST", url, bytes.NewReader(payloadBytes))
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error creating request: %v\n", err)
			os.Exit(1)
		}
		req.Header.Set("Content-Type", "application/json")

		stopProgress := func() {}
		if term.IsTerminal(int(os.Stdout.Fd())) {
			stopProgress = startProgressSpinner(fmt.Sprintf("Queuing dataset '%s' for processing", datasetName))
		} else {
			fmt.Printf("Queuing dataset '%s' for processing...\n", datasetName)
		}
		resp, err := utils.GetHTTPClientWithTimeout(0).Do(req)
		stopProgress()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error starting dataset processing: %v\n", err)
			os.Exit(1)
		}
		defer resp.Body.Close()

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading response: %v\n", err)
			os.Exit(1)
		}

		if resp.StatusCode != http.StatusOK {
			fmt.Fprintf(os.Stderr, "Error: %s\n", utils.PrettyServerError(resp, body))
			os.Exit(1)
		}

		var actionResp struct {
			Message string `json:"message"`
			TaskURI string `json:"task_uri"`
			TaskID  string `json:"task_id"`
		}
		if err := json.Unmarshal(body, &actionResp); err != nil {
			fmt.Fprintf(os.Stderr, "Error parsing action response: %v\n", err)
			os.Exit(1)
		}

		if actionResp.TaskID == "" {
			fmt.Println("Server did not return a task ID. Processing has been queued but cannot be tracked automatically.")
			fmt.Printf("Response: %s\n", string(body))
			return
		}

		fmt.Printf("\n📨 Task queued: %s\n", actionResp.TaskID)
		fmt.Printf("🔗 Track progress: %s\n", actionResp.TaskURI)
		fmt.Println("⏳ Polling task status (Ctrl+C to stop)...")

		taskStatusURL := buildServerURL(
			serverCfg.URL,
			fmt.Sprintf("/v1/projects/%s/%s/tasks/%s", serverCfg.Namespace, serverCfg.Project, actionResp.TaskID),
		)

		pollInterval := 2 * time.Second
		pollTimeout := 10 * time.Minute // Timeout to prevent hanging forever
		pollStart := time.Now()

		for {
			// Check for timeout
			if time.Since(pollStart) > pollTimeout {
				fmt.Fprintf(os.Stderr, "\n❌ Task polling timed out after %v. Task may still be running in background.\n", pollTimeout)
				fmt.Fprintf(os.Stderr, "   Check status manually: curl %s\n", taskStatusURL)
				os.Exit(1)
			}
			req, err := http.NewRequest("GET", taskStatusURL, nil)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error creating task status request: %v\n", err)
				os.Exit(1)
			}
			statusResp, err := utils.GetHTTPClient().Do(req)
			if err != nil {
				fmt.Fprintf(os.Stderr, "\nError polling task status: %v\n", err)
				os.Exit(1)
			}

			statusBody, err := io.ReadAll(statusResp.Body)
			statusResp.Body.Close()
			if err != nil {
				fmt.Fprintf(os.Stderr, "\nError reading task status: %v\n", err)
				os.Exit(1)
			}

			if statusResp.StatusCode != http.StatusOK {
				fmt.Fprintf(os.Stderr, "\nError tracking task: %s\n", utils.PrettyServerError(statusResp, statusBody))
				os.Exit(1)
			}

			var taskStatus struct {
				TaskID    string          `json:"task_id"`
				State     string          `json:"state"`
				Result    json.RawMessage `json:"result"`
				Error     string          `json:"error"`
				Traceback string          `json:"traceback"`
			}

			if err := json.Unmarshal(statusBody, &taskStatus); err != nil {
				fmt.Fprintf(os.Stderr, "\nError parsing task status: %v\n", err)
				os.Exit(1)
			}

			fmt.Printf("\r   • Task state: %s", taskStatus.State)

			switch taskStatus.State {
			case "PENDING", "STARTED", "PROGRESS":
				time.Sleep(pollInterval)
				continue
			case "SUCCESS":
				fmt.Printf("\r✅ Task %s completed successfully\n", taskStatus.TaskID)

				if len(taskStatus.Result) > 0 && string(taskStatus.Result) != "null" {
					// Parse the async task result
					var rawResult struct {
						ProcessedFiles int               `json:"processed_files"`
						SkippedFiles   int               `json:"skipped_files"`
						FailedFiles    int               `json:"failed_files"`
						Details        []json.RawMessage `json:"details"`
					}
					if err := json.Unmarshal(taskStatus.Result, &rawResult); err != nil {
						fmt.Printf("   Result: %s\n", string(taskStatus.Result))
						return
					}

					// Transform the async result into the expected format (matching origin/main)
					result := struct {
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
					}{
						ProcessedFiles: rawResult.ProcessedFiles,
						SkippedFiles:   rawResult.SkippedFiles,
						FailedFiles:    rawResult.FailedFiles,
					}

					// Parse the detail array format [bool, {...}] and transform to flat structure
					for _, detail := range rawResult.Details {
						var rawArray []json.RawMessage
						if err := json.Unmarshal(detail, &rawArray); err == nil && len(rawArray) >= 2 {
							var success bool
							var info struct {
								Filename   string   `json:"filename"`
								Status     string   `json:"status"`
								Reason     string   `json:"reason"`
								Parser     string   `json:"parser"`
								Extractors []string `json:"extractors"`
								Chunks     *int     `json:"chunks"`
								ChunkSize  *int     `json:"chunk_size"`
								Embedder   string   `json:"embedder"`
								Error      string   `json:"error"`
								Result     struct {
									Filename string `json:"filename"`
								} `json:"result"`
							}
							json.Unmarshal(rawArray[0], &success)
							json.Unmarshal(rawArray[1], &info)

							// Determine status
							status := info.Status
							if status == "" {
								if success {
									status = "processed"
								} else {
									status = "failed"
								}
							}

							// Use filename from result if main filename looks like a hash
							filename := info.Filename
							if info.Result.Filename != "" && len(info.Filename) == 64 {
								filename = info.Result.Filename
							}

							result.Details = append(result.Details, struct {
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
							}{
								Hash:       info.Filename,
								Filename:   filename,
								Status:     status,
								Parser:     info.Parser,
								Extractors: info.Extractors,
								Chunks:     info.Chunks,
								ChunkSize:  info.ChunkSize,
								Embedder:   info.Embedder,
								Error:      info.Error,
								Reason:     info.Reason,
							})
						}
					}

					// Use the exact formatting from origin/main
					fmt.Printf("\n📁 File Processing Details:\n")
					fmt.Printf("────────────────────────────────────────────────────────────────────────\n")
					for i, d := range result.Details {
						// Show both filename and hash for clarity
						identifier := ""
						if d.Filename != "" {
							identifier = d.Filename
							if len(d.Hash) > 12 {
								identifier += fmt.Sprintf(" [%s...]", d.Hash[:12])
							}
						} else {
							identifier = d.Hash
						}

						// Color-code status
						statusDisplay := d.Status
						statusBadge := ""
						switch d.Status {
						case "processed":
							statusDisplay = "PROCESSED"
							statusBadge = "✅"
						case "skipped":
							statusDisplay = "SKIPPED"
							statusBadge = "⏭️"
						case "failed":
							statusDisplay = "FAILED"
							statusBadge = "❌"
						}

						// File header with number, status badge, and identifier
						fmt.Printf("\n   %s [%d] %s\n", statusBadge, i+1, identifier)
						fmt.Printf("       ├─ Status: %s\n", statusDisplay)

						switch d.Status {
						case "processed":
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
						case "skipped":
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
						case "failed":
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

					// Exit with non-zero code if any files failed
					if result.FailedFiles > 0 {
						os.Exit(1)
					}
				}
				return
			case "FAILURE":
				fmt.Printf("\r❌ Task %s failed\n", taskStatus.TaskID)
				if taskStatus.Error != "" {
					fmt.Printf("   Error: %s\n", taskStatus.Error)
				}
				if taskStatus.Traceback != "" {
					fmt.Printf("   Traceback: %s\n", taskStatus.Traceback)
				}
				os.Exit(1)
			default:
				fmt.Printf("\r⚠️  Task %s ended with unexpected state: %s\n", taskStatus.TaskID, taskStatus.State)
				if len(taskStatus.Result) > 0 && string(taskStatus.Result) != "null" {
					fmt.Printf("   Result: %s\n", string(taskStatus.Result))
				}
				os.Exit(1)
			}
		}
	},
}

func init() {

	// Server routing flags (align with projects chat)
	datasetsCmd.PersistentFlags().StringVar(&serverURL, "server-url", "", "LlamaFarm server URL (default: http://localhost:8000)")
	datasetsCmd.PersistentFlags().StringVar(&namespace, "namespace", "", "Project namespace (default: from llamafarm.yaml)")
	datasetsCmd.PersistentFlags().StringVar(&projectID, "project", "", "Project ID (default: from llamafarm.yaml)")

	// Add flags specific to add command
	datasetsCreateCmd.Flags().StringVarP(&dataProcessingStrategy, "data-processing-strategy", "s", "", "Data processing strategy to use for this dataset (required)")
	datasetsCreateCmd.Flags().StringVarP(&database, "database", "b", "", "Database to use for this dataset (required)")

	// Mark flags as required
	datasetsCreateCmd.MarkFlagRequired("data-processing-strategy")
	datasetsCreateCmd.MarkFlagRequired("database")

	// Add subcommands to datasets
	datasetsCmd.AddCommand(datasetsListCmd)
	datasetsCmd.AddCommand(datasetsCreateCmd)
	datasetsCmd.AddCommand(datasetsDeleteCommand)
	datasetsCmd.AddCommand(datasetsUploadCmd)
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

func startProgressSpinner(message string) func() {
	done := make(chan struct{})
	var once sync.Once

	go func() {
		spinnerChars := []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}
		i := 0
		ticker := time.NewTicker(100 * time.Millisecond)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				fmt.Printf("\r%s %s", spinnerChars[i%len(spinnerChars)], message)
				i++
			case <-done:
				fmt.Print("\r")
				return
			}
		}
	}()

	return func() {
		once.Do(func() {
			close(done)
			// Clear the spinner line
			fmt.Print("\r\033[K")
		})
	}
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

	resp, err := utils.GetHTTPClient().Do(req)
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

// isGlobPattern checks if a path contains glob metacharacters
func isGlobPattern(path string) bool {
	// Check for standard glob metacharacters used by filepath.Glob
	return strings.ContainsAny(path, "*?[")
}

// recursiveGlob expands patterns with '**' to match files recursively
func recursiveGlob(pattern string) ([]string, error) {
	var matches []string

	// Check if pattern contains **
	if !strings.Contains(pattern, "**") {
		// No ** pattern, use standard glob
		return filepath.Glob(pattern)
	}

	// Split pattern at first **
	parts := strings.Split(pattern, "**")
	if len(parts) != 2 {
		return nil, fmt.Errorf("invalid pattern: only one ** supported per pattern")
	}

	basePath := parts[0]
	remainingPattern := parts[1]

	// Remove trailing slash from base path if present
	basePath = strings.TrimSuffix(basePath, "/")
	if basePath == "" {
		basePath = "."
	}

	// Remove leading slash from remaining pattern if present
	remainingPattern = strings.TrimPrefix(remainingPattern, "/")

	// Walk the directory tree starting from basePath
	err := filepath.Walk(basePath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			// Skip directories we can't access
			return nil
		}

		// Skip directories unless remaining pattern is empty (meaning we want all files)
		if info.IsDir() && remainingPattern != "" {
			return nil
		}

		// If remaining pattern is empty, match all files
		if remainingPattern == "" || remainingPattern == "*" {
			if !info.IsDir() {
				matches = append(matches, path)
			}
			return nil
		}

		// For patterns like "*.txt", we need to match against the filename
		// For patterns like "subdir/*.txt", we need to match against the relative path
		var matchTarget string
		if strings.Contains(remainingPattern, "/") {
			// Pattern contains path separators, match against relative path
			relPath, err := filepath.Rel(basePath, path)
			if err != nil {
				return nil
			}
			matchTarget = relPath
		} else {
			// Pattern is just a filename pattern, match against the filename only
			matchTarget = filepath.Base(path)
		}

		// Match against the remaining pattern
		matched, err := filepath.Match(remainingPattern, matchTarget)
		if err != nil {
			return nil
		}

		if matched && !info.IsDir() {
			matches = append(matches, path)
		}

		return nil
	})

	return matches, err
}

// expandPathsToFiles expands paths (files, directories, globs) to a list of actual files
// Use ** glob patterns for recursive directory traversal (e.g., "docs/**/*.txt")
func expandPathsToFiles(paths []string) ([]string, error) {
	var allFiles []string
	seen := make(map[string]bool) // Track files to avoid duplicates

	for _, p := range paths {
		// First check if it's a glob pattern
		if isGlobPattern(p) {
			matches, err := recursiveGlob(p)
			if err != nil {
				return nil, fmt.Errorf("error processing glob pattern '%s': %v", p, err)
			}
			for _, match := range matches {
				info, err := os.Stat(match)
				if err != nil {
					continue // Skip files we can't stat
				}
				if !info.IsDir() {
					absPath, _ := filepath.Abs(match)
					if !seen[absPath] {
						seen[absPath] = true
						allFiles = append(allFiles, match)
					}
				}
			}
			continue
		}

		// Check if path exists
		info, err := os.Stat(p)
		if err != nil {
			if os.IsNotExist(err) {
				fmt.Printf("⚠️  Warning: Path does not exist: %s\n", p)
				continue
			}
			return nil, fmt.Errorf("error accessing path '%s': %v", p, err)
		}

		// If it's a file, add it
		if !info.IsDir() {
			absPath, _ := filepath.Abs(p)
			if !seen[absPath] {
				seen[absPath] = true
				allFiles = append(allFiles, p)
			}
			continue
		}

		// It's a directory - read files in directory only (non-recursive)
		// For recursive traversal, users should use ** glob patterns like "dir/**/*"
		entries, err := os.ReadDir(p)
		if err != nil {
			return nil, fmt.Errorf("error reading directory '%s': %v", p, err)
		}
		for _, entry := range entries {
			if !entry.IsDir() {
				fullPath := filepath.Join(p, entry.Name())
				absPath, _ := filepath.Abs(fullPath)
				if !seen[absPath] {
					seen[absPath] = true
					allFiles = append(allFiles, fullPath)
				}
			}
		}
	}

	return allFiles, nil
}

type uploadResult struct {
	skipped bool
	err     error
}

func uploadFileToDataset(server string, namespace string, project string, dataset string, path string) uploadResult {
	// Open file
	file, err := os.Open(path)
	if err != nil {
		return uploadResult{err: err}
	}
	defer file.Close()

	// Prepare multipart form
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)
	part, err := writer.CreateFormFile("file", filepath.Base(path))
	if err != nil {
		return uploadResult{err: err}
	}
	if _, err := io.Copy(part, file); err != nil {
		return uploadResult{err: err}
	}
	if err := writer.Close(); err != nil {
		return uploadResult{err: err}
	}

	// Build request
	url := buildServerURL(server, fmt.Sprintf("/v1/projects/%s/%s/datasets/%s/data", namespace, project, dataset))
	req, err := http.NewRequest("POST", url, &buf)
	if err != nil {
		return uploadResult{err: err}
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	resp, err := utils.GetHTTPClientWithTimeout(0).Do(req)
	if err != nil {
		return uploadResult{err: err}
	}
	defer resp.Body.Close()
	body, readErr := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		if readErr != nil {
			return uploadResult{err: fmt.Errorf("%s", readErr.Error())}
		}
		return uploadResult{err: fmt.Errorf("%s", utils.PrettyServerError(resp, body))}
	}

	// Parse response to check if file was skipped
	var uploadResp struct {
		Skipped bool `json:"skipped"`
	}
	if err := json.Unmarshal(body, &uploadResp); err != nil {
		// If we can't parse the response, assume it was successful (backwards compatibility)
		return uploadResult{skipped: false, err: nil}
	}

	return uploadResult{skipped: uploadResp.Skipped, err: nil}
}
