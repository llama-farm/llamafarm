package cmd

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"llamafarm-cli/cmd/config"

	"github.com/spf13/cobra"
)

var (
	manageDatabase  string
	force           bool
	deleteDocIDs    []string
	deleteFilenames []string
	deleteMetadata  []string
)

// ragClearCmd represents the rag clear command
var ragClearCmd = &cobra.Command{
	Use:   "clear",
	Short: "Clear all documents from RAG database",
	Long: `Clear all documents and vectors from the specified RAG database.
This operation cannot be undone!

Examples:
  # Clear default database (requires confirmation)
  lf rag clear

  # Clear specific database
  lf rag clear --database main_database

  # Clear without confirmation prompt
  lf rag clear --force`,
	Run: func(cmd *cobra.Command, args []string) {
		serverCfg, err := config.GetServerConfig(getEffectiveCWD(), serverURL, namespace, projectID)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

		ensureServerAvailable(serverCfg.URL, true)

		// Confirm operation unless force flag is set
		if !force {
			database := manageDatabase
			if database == "" {
				database = "default"
			}
			fmt.Printf("⚠️  WARNING: This will permanently delete all documents from database '%s'\n", database)
			fmt.Print("Are you sure? Type 'yes' to confirm: ")

			var response string
			fmt.Scanln(&response)
			if response != "yes" {
				fmt.Println("Operation cancelled")
				return
			}
		}

		fmt.Println("🗑️  Clearing database...")
		result, err := clearRAGDatabase(serverCfg, manageDatabase)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error clearing database: %v\n", err)
			os.Exit(1)
		}

		displayClearResult(result)
	},
}

// ragDeleteCmd represents the rag delete command
var ragDeleteCmd = &cobra.Command{
	Use:   "delete",
	Short: "Delete specific documents from RAG database",
	Long: `Delete specific documents from the RAG database by ID, filename, or metadata.

Examples:
  # Delete by document IDs
  lf rag delete --id doc_123 --id doc_456

  # Delete by filenames
  lf rag delete --filename "report.pdf" --filename "data.csv"

  # Delete by metadata
  lf rag delete --metadata "source:manual_upload"

  # Combine criteria (OR operation)
  lf rag delete --filename "old_*.pdf" --metadata "status:deprecated"

  # Force delete without confirmation
  lf rag delete --filename "temp_*.txt" --force`,
	Run: func(cmd *cobra.Command, args []string) {
		// Validate that at least one deletion criteria is provided
		if len(deleteDocIDs) == 0 && len(deleteFilenames) == 0 && len(deleteMetadata) == 0 {
			fmt.Fprintf(os.Stderr, "Error: Must specify at least one deletion criteria (--id, --filename, or --metadata)\n")
			cmd.Help()
			os.Exit(1)
		}

		serverCfg, err := config.GetServerConfig(getEffectiveCWD(), serverURL, namespace, projectID)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

		ensureServerAvailable(serverCfg.URL, true)

		// Build deletion request
		req := buildDeleteRequest()

		// Get count of documents to be deleted
		count, err := getDeleteCount(serverCfg, req)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error checking documents to delete: %v\n", err)
			os.Exit(1)
		}

		if count == 0 {
			fmt.Println("No documents match the deletion criteria")
			return
		}

		// Confirm operation unless force flag is set
		if !force {
			fmt.Printf("⚠️  WARNING: This will delete %d document(s)\n", count)
			fmt.Print("Are you sure? Type 'yes' to confirm: ")

			var response string
			fmt.Scanln(&response)
			if response != "yes" {
				fmt.Println("Operation cancelled")
				return
			}
		}

		fmt.Printf("🗑️  Deleting %d document(s)...\n", count)
		result, err := deleteRAGDocuments(serverCfg, req)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error deleting documents: %v\n", err)
			os.Exit(1)
		}

		displayDeleteResult(result)
	},
}

// ragPruneCmd represents the rag prune command
var ragPruneCmd = &cobra.Command{
	Use:   "prune",
	Short: "Remove duplicate or orphaned vectors",
	Long: `Prune the RAG database by removing duplicate vectors and orphaned chunks.

This command helps maintain database health by:
- Removing duplicate document chunks
- Cleaning up orphaned vectors (vectors without associated documents)
- Removing vectors with corrupted embeddings

Examples:
  # Prune default database
  lf rag prune

  # Prune specific database
  lf rag prune --database main_database

  # Prune without confirmation
  lf rag prune --force`,
	Run: func(cmd *cobra.Command, args []string) {
		serverCfg, err := config.GetServerConfig(getEffectiveCWD(), serverURL, namespace, projectID)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

		ensureServerAvailable(serverCfg.URL, true)

		// Get prune preview
		preview, err := getPrunePreview(serverCfg, manageDatabase)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error getting prune preview: %v\n", err)
			os.Exit(1)
		}

		displayPrunePreview(preview)

		if preview.TotalToRemove == 0 {
			fmt.Println("✅ Database is clean, nothing to prune")
			return
		}

		// Confirm operation unless force flag is set
		if !force {
			fmt.Printf("\n⚠️  WARNING: This will remove %d vectors\n", preview.TotalToRemove)
			fmt.Print("Are you sure? Type 'yes' to confirm: ")

			var response string
			fmt.Scanln(&response)
			if response != "yes" {
				fmt.Println("Operation cancelled")
				return
			}
		}

		fmt.Println("\n🧹 Pruning database...")
		result, err := pruneRAGDatabase(serverCfg, manageDatabase)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error pruning database: %v\n", err)
			os.Exit(1)
		}

		displayPruneResult(result)
	},
}

// ragExportCmd represents the rag export command
var ragExportCmd = &cobra.Command{
	Use:   "export [output-file]",
	Short: "Export RAG database contents",
	Long: `Export the contents of the RAG database to a file for backup or migration.

Examples:
  # Export to JSON file
  lf rag export backup.json

  # Export specific database
  lf rag export --database main_database backup.json

  # Export with metadata only (no content)
  lf rag export --metadata-only backup.json`,
	Args: cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		outputFile := args[0]

		serverCfg, err := config.GetServerConfig(getEffectiveCWD(), serverURL, namespace, projectID)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

		ensureServerAvailable(serverCfg.URL, true)

		fmt.Printf("📦 Exporting database to %s...\n", outputFile)

		exportData, err := exportRAGDatabase(serverCfg, manageDatabase, metadataOnly)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error exporting database: %v\n", err)
			os.Exit(1)
		}

		// Write to file
		if err := writeExportFile(outputFile, exportData); err != nil {
			fmt.Fprintf(os.Stderr, "Error writing export file: %v\n", err)
			os.Exit(1)
		}

		fmt.Printf("✅ Successfully exported %d documents to %s\n", len(exportData.Documents), outputFile)
	},
}

// ragImportCmd represents the rag import command
var ragImportCmd = &cobra.Command{
	Use:   "import [input-file]",
	Short: "Import documents into RAG database",
	Long: `Import documents from a previously exported file into the RAG database.

Examples:
  # Import from JSON file
  lf rag import backup.json

  # Import to specific database
  lf rag import --database main_database backup.json

  # Import with different processing strategy
  lf rag import --strategy universal_processor backup.json`,
	Args: cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		inputFile := args[0]

		serverCfg, err := config.GetServerConfig(getEffectiveCWD(), serverURL, namespace, projectID)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

		ensureServerAvailable(serverCfg.URL, true)

		// Read import file
		importData, err := readImportFile(inputFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading import file: %v\n", err)
			os.Exit(1)
		}

		fmt.Printf("📥 Importing %d documents from %s...\n", len(importData.Documents), inputFile)

		result, err := importRAGDatabase(serverCfg, manageDatabase, ragDataStrategy, importData)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error importing database: %v\n", err)
			os.Exit(1)
		}

		displayImportResult(result)
	},
}

var metadataOnly bool

// Request/Response types

type DeleteRequest struct {
	Database  string                 `json:"database,omitempty"`
	IDs       []string               `json:"ids,omitempty"`
	Filenames []string               `json:"filenames,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

type DeleteResult struct {
	Success          bool     `json:"success"`
	DocumentsDeleted int      `json:"documents_deleted"`
	ChunksDeleted    int      `json:"chunks_deleted"`
	Errors           []string `json:"errors,omitempty"`
}

type ClearResult struct {
	Success          bool   `json:"success"`
	DocumentsCleared int    `json:"documents_cleared"`
	ChunksCleared    int    `json:"chunks_cleared"`
	Message          string `json:"message"`
}

type PrunePreview struct {
	Duplicates    int `json:"duplicates"`
	Orphaned      int `json:"orphaned"`
	Corrupted     int `json:"corrupted"`
	TotalToRemove int `json:"total_to_remove"`
}

type PruneResult struct {
	Success  bool     `json:"success"`
	Removed  int      `json:"removed"`
	Duration float64  `json:"duration_seconds"`
	Errors   []string `json:"errors,omitempty"`
}

type ExportData struct {
	Version    string                 `json:"version"`
	ExportDate string                 `json:"export_date"`
	Database   string                 `json:"database"`
	Documents  []ExportDocument       `json:"documents"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
}

type ExportDocument struct {
	ID        string                 `json:"id"`
	Content   string                 `json:"content,omitempty"`
	Embedding []float64              `json:"embedding,omitempty"`
	Metadata  map[string]interface{} `json:"metadata"`
}

type ImportResult struct {
	Success           bool     `json:"success"`
	DocumentsImported int      `json:"documents_imported"`
	ChunksCreated     int      `json:"chunks_created"`
	Errors            []string `json:"errors,omitempty"`
}

// Helper functions

func buildDeleteRequest() DeleteRequest {
	req := DeleteRequest{
		Database:  manageDatabase,
		IDs:       deleteDocIDs,
		Filenames: deleteFilenames,
	}

	// Parse metadata filters
	if len(deleteMetadata) > 0 {
		req.Metadata = make(map[string]interface{})
		for _, filter := range deleteMetadata {
			parts := strings.SplitN(filter, ":", 2)
			if len(parts) == 2 {
				req.Metadata[parts[0]] = parts[1]
			}
		}
	}

	return req
}

func getDeleteCount(cfg *config.ServerConfig, req DeleteRequest) (int, error) {
	url := buildServerURL(cfg.URL, fmt.Sprintf("/v1/projects/%s/%s/rag/delete/preview", cfg.Namespace, cfg.Project))

	payload, err := json.Marshal(req)
	if err != nil {
		return 0, err
	}

	httpReq, err := http.NewRequest("POST", url, bytes.NewReader(payload))
	if err != nil {
		return 0, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := getHTTPClient().Do(httpReq)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, err
	}

	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("server returned %d: %s", resp.StatusCode, string(body))
	}

	var result struct {
		Count int `json:"count"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return 0, err
	}

	return result.Count, nil
}

func clearRAGDatabase(cfg *config.ServerConfig, database string) (*ClearResult, error) {
	url := buildServerURL(cfg.URL, fmt.Sprintf("/v1/projects/%s/%s/rag/clear", cfg.Namespace, cfg.Project))
	if database != "" {
		url += "?database=" + database
	}

	req, err := http.NewRequest("POST", url, nil)
	if err != nil {
		return nil, err
	}

	resp, err := getHTTPClient().Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("server returned %d: %s", resp.StatusCode, string(body))
	}

	var result ClearResult
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, err
	}

	return &result, nil
}

func deleteRAGDocuments(cfg *config.ServerConfig, req DeleteRequest) (*DeleteResult, error) {
	url := buildServerURL(cfg.URL, fmt.Sprintf("/v1/projects/%s/%s/rag/delete", cfg.Namespace, cfg.Project))

	payload, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequest("POST", url, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := getHTTPClient().Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("server returned %d: %s", resp.StatusCode, string(body))
	}

	var result DeleteResult
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, err
	}

	return &result, nil
}

func getPrunePreview(cfg *config.ServerConfig, database string) (*PrunePreview, error) {
	url := buildServerURL(cfg.URL, fmt.Sprintf("/v1/projects/%s/%s/rag/prune/preview", cfg.Namespace, cfg.Project))
	if database != "" {
		url += "?database=" + database
	}

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}

	resp, err := getHTTPClient().Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("server returned %d: %s", resp.StatusCode, string(body))
	}

	var preview PrunePreview
	if err := json.Unmarshal(body, &preview); err != nil {
		return nil, err
	}

	return &preview, nil
}

func pruneRAGDatabase(cfg *config.ServerConfig, database string) (*PruneResult, error) {
	url := buildServerURL(cfg.URL, fmt.Sprintf("/v1/projects/%s/%s/rag/prune", cfg.Namespace, cfg.Project))
	if database != "" {
		url += "?database=" + database
	}

	req, err := http.NewRequest("POST", url, nil)
	if err != nil {
		return nil, err
	}

	resp, err := getHTTPClient().Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("server returned %d: %s", resp.StatusCode, string(body))
	}

	var result PruneResult
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, err
	}

	return &result, nil
}

func exportRAGDatabase(cfg *config.ServerConfig, database string, metadataOnly bool) (*ExportData, error) {
	url := buildServerURL(cfg.URL, fmt.Sprintf("/v1/projects/%s/%s/rag/export", cfg.Namespace, cfg.Project))

	params := []string{}
	if database != "" {
		params = append(params, "database="+database)
	}
	if metadataOnly {
		params = append(params, "metadata_only=true")
	}
	if len(params) > 0 {
		url += "?" + strings.Join(params, "&")
	}

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}

	resp, err := getHTTPClient().Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("server returned %d: %s", resp.StatusCode, string(body))
	}

	var exportData ExportData
	if err := json.Unmarshal(body, &exportData); err != nil {
		return nil, err
	}

	return &exportData, nil
}

func importRAGDatabase(cfg *config.ServerConfig, database string, strategy string, data *ExportData) (*ImportResult, error) {
	url := buildServerURL(cfg.URL, fmt.Sprintf("/v1/projects/%s/%s/rag/import", cfg.Namespace, cfg.Project))

	params := []string{}
	if database != "" {
		params = append(params, "database="+database)
	}
	if strategy != "" {
		params = append(params, "strategy="+strategy)
	}
	if len(params) > 0 {
		url += "?" + strings.Join(params, "&")
	}

	payload, err := json.Marshal(data)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest("POST", url, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := getHTTPClient().Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("server returned %d: %s", resp.StatusCode, string(body))
	}

	var result ImportResult
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, err
	}

	return &result, nil
}

func writeExportFile(filename string, data *ExportData) error {
	output, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filename, output, 0644)
}

func readImportFile(filename string) (*ExportData, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	var exportData ExportData
	if err := json.Unmarshal(data, &exportData); err != nil {
		return nil, err
	}

	return &exportData, nil
}

// Display functions

func displayClearResult(result *ClearResult) {
	if result.Success {
		fmt.Println("✅ Database cleared successfully!")
		fmt.Printf("   Documents removed: %d\n", result.DocumentsCleared)
		fmt.Printf("   Chunks removed: %d\n", result.ChunksCleared)
		if result.Message != "" {
			fmt.Printf("   %s\n", result.Message)
		}
	} else {
		fmt.Println("❌ Failed to clear database")
		if result.Message != "" {
			fmt.Printf("   Error: %s\n", result.Message)
		}
	}
}

func displayDeleteResult(result *DeleteResult) {
	if result.Success {
		fmt.Println("✅ Documents deleted successfully!")
		fmt.Printf("   Documents removed: %d\n", result.DocumentsDeleted)
		fmt.Printf("   Chunks removed: %d\n", result.ChunksDeleted)
	} else {
		fmt.Println("❌ Failed to delete documents")
		if len(result.Errors) > 0 {
			fmt.Println("   Errors:")
			for _, err := range result.Errors {
				fmt.Printf("     • %s\n", err)
			}
		}
	}
}

func displayPrunePreview(preview *PrunePreview) {
	fmt.Println("\n🔍 Prune Preview:")
	fmt.Printf("   Duplicate vectors: %d\n", preview.Duplicates)
	fmt.Printf("   Orphaned vectors: %d\n", preview.Orphaned)
	fmt.Printf("   Corrupted vectors: %d\n", preview.Corrupted)
	fmt.Printf("   Total to remove: %d\n", preview.TotalToRemove)
}

func displayPruneResult(result *PruneResult) {
	if result.Success {
		fmt.Println("✅ Database pruned successfully!")
		fmt.Printf("   Vectors removed: %d\n", result.Removed)
		fmt.Printf("   Duration: %.2f seconds\n", result.Duration)
	} else {
		fmt.Println("❌ Failed to prune database")
		if len(result.Errors) > 0 {
			fmt.Println("   Errors:")
			for _, err := range result.Errors {
				fmt.Printf("     • %s\n", err)
			}
		}
	}
}

func displayImportResult(result *ImportResult) {
	if result.Success {
		fmt.Println("✅ Import completed successfully!")
		fmt.Printf("   Documents imported: %d\n", result.DocumentsImported)
		fmt.Printf("   Chunks created: %d\n", result.ChunksCreated)
	} else {
		fmt.Println("❌ Import failed")
		if len(result.Errors) > 0 {
			fmt.Println("   Errors:")
			for _, err := range result.Errors {
				fmt.Printf("     • %s\n", err)
			}
		}
	}
}

func init() {
	// Add commands to rag
	ragCmd.AddCommand(ragClearCmd)
	ragCmd.AddCommand(ragDeleteCmd)
	ragCmd.AddCommand(ragPruneCmd)
	ragCmd.AddCommand(ragExportCmd)
	ragCmd.AddCommand(ragImportCmd)

	// Clear command flags
	ragClearCmd.Flags().StringVar(&manageDatabase, "database", "", "Database to clear")
	ragClearCmd.Flags().BoolVarP(&force, "force", "f", false, "Skip confirmation prompt")

	// Delete command flags
	ragDeleteCmd.Flags().StringVar(&manageDatabase, "database", "", "Database to delete from")
	ragDeleteCmd.Flags().StringSliceVar(&deleteDocIDs, "id", []string{}, "Document IDs to delete")
	ragDeleteCmd.Flags().StringSliceVar(&deleteFilenames, "filename", []string{}, "Filenames to delete (supports wildcards)")
	ragDeleteCmd.Flags().StringSliceVar(&deleteMetadata, "metadata", []string{}, "Metadata filters (format: key:value)")
	ragDeleteCmd.Flags().BoolVarP(&force, "force", "f", false, "Skip confirmation prompt")

	// Prune command flags
	ragPruneCmd.Flags().StringVar(&manageDatabase, "database", "", "Database to prune")
	ragPruneCmd.Flags().BoolVarP(&force, "force", "f", false, "Skip confirmation prompt")

	// Export command flags
	ragExportCmd.Flags().StringVar(&manageDatabase, "database", "", "Database to export")
	ragExportCmd.Flags().BoolVar(&metadataOnly, "metadata-only", false, "Export metadata only (no content/embeddings)")

	// Import command flags
	ragImportCmd.Flags().StringVar(&manageDatabase, "database", "", "Database to import into")
	ragImportCmd.Flags().StringVarP(&ragDataStrategy, "strategy", "s", "", "Data processing strategy for import")
}
