package cmd

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"text/tabwriter"
	"time"

	"llamafarm-cli/cmd/config"

	"github.com/spf13/cobra"
)

var (
	statsDatabase   string
	statsVerbose    bool
	statsOutputJSON bool
)

// ragStatsCmd represents the rag stats command
var ragStatsCmd = &cobra.Command{
	Use:   "stats",
	Short: "Show RAG database statistics",
	Long: `Display comprehensive statistics about the RAG database including vector count,
document count, and storage usage.

Examples:
  # Show stats for default database
  lf rag stats

  # Show stats for specific database
  lf rag stats --database main_database

  # Show detailed stats with verbose output
  lf rag stats --verbose

  # Output as JSON
  lf rag stats --json`,
	Run: func(cmd *cobra.Command, args []string) {
		serverCfg, err := config.GetServerConfig(getEffectiveCWD(), serverURL, namespace, projectID)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

		ensureServerAvailable(serverCfg.URL, true)

		stats, err := fetchRAGStats(serverCfg, statsDatabase)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error fetching RAG stats: %v\n", err)
			os.Exit(1)
		}

		if statsOutputJSON {
			displayStatsJSON(stats)
		} else {
			displayStatsTable(stats)
		}
	},
}

// ragHealthCmd represents the rag health command
var ragHealthCmd = &cobra.Command{
	Use:   "health",
	Short: "Check RAG system health",
	Long: `Check the health status of all RAG components including embedder,
vector store, and processing pipeline.

Examples:
  # Check health of default setup
  lf rag health

  # Check health of specific database
  lf rag health --database main_database`,
	Run: func(cmd *cobra.Command, args []string) {
		serverCfg, err := config.GetServerConfig(getEffectiveCWD(), serverURL, namespace, projectID)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

		ensureServerAvailable(serverCfg.URL, true)

		health, err := fetchRAGHealth(serverCfg, statsDatabase)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error checking RAG health: %v\n", err)
			os.Exit(1)
		}

		displayHealthStatus(health)
	},
}

// ragListCmd represents the rag list documents command
var ragListCmd = &cobra.Command{
	Use:   "list",
	Short: "List documents in RAG database",
	Long: `List all documents stored in the RAG database with their metadata.

Examples:
  # List all documents
  lf rag list

  # List documents from specific database
  lf rag list --database main_database

  # List with filters
  lf rag list --filter "file_type:pdf"

  # Limit results
  lf rag list --limit 10`,
	Run: func(cmd *cobra.Command, args []string) {
		serverCfg, err := config.GetServerConfig(getEffectiveCWD(), serverURL, namespace, projectID)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

		ensureServerAvailable(serverCfg.URL, true)

		docs, err := fetchRAGDocuments(serverCfg, statsDatabase, listLimit, metadataFilters)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error listing documents: %v\n", err)
			os.Exit(1)
		}

		displayDocumentList(docs)
	},
}

// ragCompactCmd represents the rag compact command
var ragCompactCmd = &cobra.Command{
	Use:   "compact",
	Short: "Compact and optimize RAG database",
	Long: `Compact the vector database to optimize storage and query performance.
This removes deleted vectors and reorganizes the index.

Examples:
  # Compact default database
  lf rag compact

  # Compact specific database
  lf rag compact --database main_database`,
	Run: func(cmd *cobra.Command, args []string) {
		serverCfg, err := config.GetServerConfig(getEffectiveCWD(), serverURL, namespace, projectID)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

		ensureServerAvailable(serverCfg.URL, true)

		fmt.Println("🔧 Starting database compaction...")
		result, err := compactRAGDatabase(serverCfg, statsDatabase)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error compacting database: %v\n", err)
			os.Exit(1)
		}

		displayCompactionResult(result)
	},
}

// ragReindexCmd represents the rag reindex command
var ragReindexCmd = &cobra.Command{
	Use:   "reindex",
	Short: "Reindex documents in RAG database",
	Long: `Reindex all documents in the RAG database. This can be useful after
configuration changes or to fix index corruption.

Examples:
  # Reindex default database
  lf rag reindex

  # Reindex specific database with different strategy
  lf rag reindex --database main_database --strategy universal_processor`,
	Run: func(cmd *cobra.Command, args []string) {
		serverCfg, err := config.GetServerConfig(getEffectiveCWD(), serverURL, namespace, projectID)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

		ensureServerAvailable(serverCfg.URL, true)

		fmt.Println("🔄 Starting reindexing...")
		result, err := reindexRAGDatabase(serverCfg, statsDatabase, ragDataStrategy)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reindexing database: %v\n", err)
			os.Exit(1)
		}

		displayReindexResult(result)
	},
}

var listLimit int

// RAGStats represents database statistics
type RAGStats struct {
	Database       string                 `json:"database"`
	VectorCount    int                    `json:"vector_count"`
	DocumentCount  int                    `json:"document_count"`
	ChunkCount     int                    `json:"chunk_count"`
	CollectionSize int64                  `json:"collection_size_bytes"`
	IndexSize      int64                  `json:"index_size_bytes"`
	EmbeddingDim   int                    `json:"embedding_dimension"`
	DistanceMetric string                 `json:"distance_metric"`
	LastUpdated    time.Time              `json:"last_updated"`
	Metadata       map[string]interface{} `json:"metadata,omitempty"`
}

// RAGHealth represents health status
type RAGHealth struct {
	Status     string                     `json:"status"`
	Database   string                     `json:"database"`
	Components map[string]ComponentHealth `json:"components"`
	LastCheck  time.Time                  `json:"last_check"`
	Issues     []string                   `json:"issues,omitempty"`
}

// ComponentHealth represents health of a single component
type ComponentHealth struct {
	Name    string  `json:"name"`
	Status  string  `json:"status"`
	Latency float64 `json:"latency_ms"`
	Message string  `json:"message,omitempty"`
}

// RAGDocument represents a document in the database
type RAGDocument struct {
	ID           string                 `json:"id"`
	Filename     string                 `json:"filename"`
	ChunkCount   int                    `json:"chunk_count"`
	Size         int64                  `json:"size_bytes"`
	Parser       string                 `json:"parser_used"`
	DateIngested time.Time              `json:"date_ingested"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
}

// CompactionResult represents the result of database compaction
type CompactionResult struct {
	Success       bool    `json:"success"`
	VectorsBefore int     `json:"vectors_before"`
	VectorsAfter  int     `json:"vectors_after"`
	SpaceSaved    int64   `json:"space_saved_bytes"`
	Duration      float64 `json:"duration_seconds"`
	Message       string  `json:"message"`
}

// ReindexResult represents the result of reindexing
type ReindexResult struct {
	Success            bool     `json:"success"`
	DocumentsProcessed int      `json:"documents_processed"`
	ChunksCreated      int      `json:"chunks_created"`
	Duration           float64  `json:"duration_seconds"`
	Errors             []string `json:"errors,omitempty"`
}

func fetchRAGStats(cfg *config.ServerConfig, database string) (*RAGStats, error) {
	url := buildServerURL(cfg.URL, fmt.Sprintf("/v1/projects/%s/%s/rag/stats", cfg.Namespace, cfg.Project))
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

	var stats RAGStats
	if err := json.Unmarshal(body, &stats); err != nil {
		return nil, err
	}

	return &stats, nil
}

func fetchRAGHealth(cfg *config.ServerConfig, database string) (*RAGHealth, error) {
	url := buildServerURL(cfg.URL, fmt.Sprintf("/v1/projects/%s/%s/rag/health", cfg.Namespace, cfg.Project))
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

	var health RAGHealth
	if err := json.Unmarshal(body, &health); err != nil {
		return nil, err
	}

	return &health, nil
}

func fetchRAGDocuments(cfg *config.ServerConfig, database string, limit int, filters []string) ([]RAGDocument, error) {
	url := buildServerURL(cfg.URL, fmt.Sprintf("/v1/projects/%s/%s/rag/documents", cfg.Namespace, cfg.Project))

	// Add query parameters
	params := []string{}
	if database != "" {
		params = append(params, "database="+database)
	}
	if limit > 0 {
		params = append(params, fmt.Sprintf("limit=%d", limit))
	}
	for _, filter := range filters {
		params = append(params, "filter="+filter)
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

	var docs []RAGDocument
	if err := json.Unmarshal(body, &docs); err != nil {
		return nil, err
	}

	return docs, nil
}

func compactRAGDatabase(cfg *config.ServerConfig, database string) (*CompactionResult, error) {
	url := buildServerURL(cfg.URL, fmt.Sprintf("/v1/projects/%s/%s/rag/compact", cfg.Namespace, cfg.Project))
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

	var result CompactionResult
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, err
	}

	return &result, nil
}

func reindexRAGDatabase(cfg *config.ServerConfig, database string, strategy string) (*ReindexResult, error) {
	url := buildServerURL(cfg.URL, fmt.Sprintf("/v1/projects/%s/%s/rag/reindex", cfg.Namespace, cfg.Project))

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

	var result ReindexResult
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, err
	}

	return &result, nil
}

func displayStatsTable(stats *RAGStats) {
	fmt.Println("\n📊 RAG Database Statistics")
	fmt.Println(strings.Repeat("═", 60))

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintf(w, "Database:\t%s\n", stats.Database)
	fmt.Fprintf(w, "Vectors:\t%d\n", stats.VectorCount)
	fmt.Fprintf(w, "Documents:\t%d\n", stats.DocumentCount)
	fmt.Fprintf(w, "Chunks:\t%d\n", stats.ChunkCount)
	fmt.Fprintf(w, "Avg Chunks/Doc:\t%.1f\n", float64(stats.ChunkCount)/float64(maxInt(stats.DocumentCount, 1)))
	fmt.Fprintf(w, "Collection Size:\t%s\n", formatBytes(stats.CollectionSize))
	fmt.Fprintf(w, "Index Size:\t%s\n", formatBytes(stats.IndexSize))
	fmt.Fprintf(w, "Total Size:\t%s\n", formatBytes(stats.CollectionSize+stats.IndexSize))
	fmt.Fprintf(w, "Embedding Dim:\t%d\n", stats.EmbeddingDim)
	fmt.Fprintf(w, "Distance Metric:\t%s\n", stats.DistanceMetric)
	fmt.Fprintf(w, "Last Updated:\t%s\n", stats.LastUpdated.Format("2006-01-02 15:04:05"))
	w.Flush()

	if statsVerbose && len(stats.Metadata) > 0 {
		fmt.Println("\n📝 Additional Metadata:")
		for k, v := range stats.Metadata {
			fmt.Printf("  %s: %v\n", k, v)
		}
	}

	fmt.Println(strings.Repeat("═", 60))
}

func displayStatsJSON(stats *RAGStats) {
	output, err := json.MarshalIndent(stats, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error formatting JSON: %v\n", err)
		return
	}
	fmt.Println(string(output))
}

func displayHealthStatus(health *RAGHealth) {
	// Determine overall status icon
	statusIcon := "✅"
	switch health.Status {
	case "degraded":
		statusIcon = "⚠️"
	case "unhealthy":
		statusIcon = "❌"
	}

	fmt.Printf("\nRAG database status: %s %s\n", statusIcon, strings.ToLower(health.Status))
	fmt.Println(strings.Repeat("═", 60))

	fmt.Printf("Database: %s\n", health.Database)
	fmt.Printf("Last Check: %s\n\n", health.LastCheck.Format("2006-01-02 15:04:05"))

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "Component\tStatus\tLatency\tMessage")
	fmt.Fprintln(w, "---------\t------\t-------\t-------")

	for name, comp := range health.Components {
		var statusStr string
		switch comp.Status {
		case "healthy":
			statusStr = "✅ " + comp.Status
		case "degraded":
			statusStr = "⚠️  " + comp.Status
		default:
			statusStr = "❌ " + comp.Status
		}

		latencyStr := fmt.Sprintf("%.1fms", comp.Latency)
		messageStr := comp.Message
		if messageStr == "" {
			messageStr = "-"
		}

		fmt.Fprintf(w, "%s\t%s\t%s\t%s\n", name, statusStr, latencyStr, messageStr)
	}
	w.Flush()

	if len(health.Issues) > 0 {
		fmt.Println("\n⚠️  Issues Detected:")
		for _, issue := range health.Issues {
			fmt.Printf("  • %s\n", issue)
		}
	}

	fmt.Println(strings.Repeat("═", 60))
}

func displayDocumentList(docs []RAGDocument) {
	if len(docs) == 0 {
		fmt.Println("No documents found")
		return
	}

	fmt.Printf("\n📚 Documents in RAG Database (%d total)\n", len(docs))
	fmt.Println(strings.Repeat("═", 80))

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "Filename\tChunks\tSize\tParser\tIngested")
	fmt.Fprintln(w, "--------\t------\t----\t------\t--------")

	for _, doc := range docs {
		fmt.Fprintf(w, "%s\t%d\t%s\t%s\t%s\n",
			truncateString(doc.Filename, 30),
			doc.ChunkCount,
			formatBytes(doc.Size),
			doc.Parser,
			doc.DateIngested.Format("2006-01-02"),
		)
	}
	w.Flush()

	fmt.Println(strings.Repeat("═", 80))
}

func displayCompactionResult(result *CompactionResult) {
	if result.Success {
		fmt.Println("✅ Database compaction completed successfully!")
		fmt.Printf("   Vectors before: %d\n", result.VectorsBefore)
		fmt.Printf("   Vectors after: %d\n", result.VectorsAfter)
		fmt.Printf("   Space saved: %s\n", formatBytes(result.SpaceSaved))
		fmt.Printf("   Duration: %.2f seconds\n", result.Duration)
		if result.Message != "" {
			fmt.Printf("   Message: %s\n", result.Message)
		}
	} else {
		fmt.Println("❌ Database compaction failed")
		if result.Message != "" {
			fmt.Printf("   Error: %s\n", result.Message)
		}
	}
}

func displayReindexResult(result *ReindexResult) {
	if result.Success {
		fmt.Println("✅ Reindexing completed successfully!")
		fmt.Printf("   Documents processed: %d\n", result.DocumentsProcessed)
		fmt.Printf("   Chunks created: %d\n", result.ChunksCreated)
		fmt.Printf("   Duration: %.2f seconds\n", result.Duration)
	} else {
		fmt.Println("❌ Reindexing failed")
		if len(result.Errors) > 0 {
			fmt.Println("   Errors:")
			for _, err := range result.Errors {
				fmt.Printf("     • %s\n", err)
			}
		}
	}
}

func formatBytes(bytes int64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(bytes)/float64(div), "KMGTPE"[exp])
}

func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func init() {
	// Add commands to rag
	ragCmd.AddCommand(ragStatsCmd)
	ragCmd.AddCommand(ragHealthCmd)
	ragCmd.AddCommand(ragListCmd)
	ragCmd.AddCommand(ragCompactCmd)
	ragCmd.AddCommand(ragReindexCmd)

	// Stats command flags
	ragStatsCmd.Flags().StringVar(&statsDatabase, "database", "", "Database to get stats for")
	ragStatsCmd.Flags().BoolVarP(&statsVerbose, "verbose", "v", false, "Show detailed statistics")
	ragStatsCmd.Flags().BoolVar(&statsOutputJSON, "json", false, "Output as JSON")

	// Health command flags
	ragHealthCmd.Flags().StringVar(&statsDatabase, "database", "", "Database to check health for")

	// List command flags
	ragListCmd.Flags().StringVar(&statsDatabase, "database", "", "Database to list documents from")
	ragListCmd.Flags().IntVarP(&listLimit, "limit", "l", 50, "Maximum number of documents to list")
	ragListCmd.Flags().StringSliceVarP(&metadataFilters, "filter", "f", []string{}, "Filter documents by metadata")

	// Compact command flags
	ragCompactCmd.Flags().StringVar(&statsDatabase, "database", "", "Database to compact")

	// Reindex command flags
	ragReindexCmd.Flags().StringVar(&statsDatabase, "database", "", "Database to reindex")
	ragReindexCmd.Flags().StringVarP(&ragDataStrategy, "strategy", "s", "", "Data processing strategy for reindexing")
}
