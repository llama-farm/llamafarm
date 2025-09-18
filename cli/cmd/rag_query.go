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
	ragDatabase       string
	ragDataStrategy   string
	retrievalStrategy string
	topK              int
	scoreThreshold    float64
	metadataFilters   []string
	outputFormat      string
	includeMetadata   bool
	includeScore      bool
	distanceMetric    string
	hybridAlpha       float64
	rerankModel       string
	queryExpansion    bool
	ragMaxTokens      int
)

// ragQueryCmd represents the rag query command
var ragQueryCmd = &cobra.Command{
	Use:   "query [query text]",
	Short: "Query documents in the RAG system",
	Long: `Query the RAG system for relevant documents using various retrieval strategies and parameters.

Examples:
  # Basic query using defaults
  lf rag query "What is transformer architecture?"

  # Query with specific database
  lf rag query --database main_database "How do neural networks scale?"

  # Query with custom top-k
  lf rag query --top-k 10 "API authentication methods"

  # Query with different retrieval strategy
  lf rag query --retrieval-strategy filtered_search "customer support issues"

  # Query with metadata filters
  lf rag query --filter "file_type:pdf" --filter "date:2024" "research papers"

  # Query with score threshold
  lf rag query --score-threshold 0.7 "quantum computing"

  # Query with all metadata and scores
  lf rag query --include-metadata --include-score "machine learning"

  # Query with custom distance metric
  lf rag query --distance-metric euclidean "vector databases"

  # Query with hybrid search
  lf rag query --retrieval-strategy hybrid --hybrid-alpha 0.5 "neural networks"`,
	Args: cobra.MinimumNArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		// Combine args into query text
		queryText := strings.Join(args, " ")

		// Get server config
		serverCfg, err := config.GetServerConfig(getEffectiveCWD(), serverURL, namespace, projectID)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

		// Ensure server is available
		ensureServerAvailable(serverCfg.URL, true)

		// Build the request
		queryRequest := buildQueryRequest(queryText)

		// Send to server
		response, err := sendQueryRequest(serverCfg, queryRequest)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error querying RAG: %v\n", err)
			os.Exit(1)
		}

		// Format and display results
		displayQueryResults(response, queryText)
	},
}

// QueryRequest represents the RAG query request
type QueryRequest struct {
	Query             string                 `json:"query"`
	Database          string                 `json:"database,omitempty"`
	DataStrategy      string                 `json:"data_processing_strategy,omitempty"`
	RetrievalStrategy string                 `json:"retrieval_strategy,omitempty"`
	TopK              int                    `json:"top_k"`
	ScoreThreshold    float64                `json:"score_threshold,omitempty"`
	MetadataFilters   map[string]interface{} `json:"metadata_filters,omitempty"`
	DistanceMetric    string                 `json:"distance_metric,omitempty"`
	HybridAlpha       float64                `json:"hybrid_alpha,omitempty"`
	RerankModel       string                 `json:"rerank_model,omitempty"`
	QueryExpansion    bool                   `json:"query_expansion"`
	MaxTokens         int                    `json:"max_tokens,omitempty"`
}

// QueryResult represents a single search result
type QueryResult struct {
	Content    string                 `json:"content"`
	Score      float64                `json:"score"`
	Metadata   map[string]interface{} `json:"metadata"`
	ChunkID    string                 `json:"chunk_id,omitempty"`
	DocumentID string                 `json:"document_id,omitempty"`
}

// QueryResponse represents the full query response
type QueryResponse struct {
	Query             string        `json:"query"`
	Results           []QueryResult `json:"results"`
	TotalResults      int           `json:"total_results"`
	ProcessingTime    float64       `json:"processing_time_ms,omitempty"`
	RetrievalStrategy string        `json:"retrieval_strategy_used"`
	Database          string        `json:"database_used"`
}

func buildQueryRequest(queryText string) QueryRequest {
	req := QueryRequest{
		Query:          queryText,
		TopK:           topK,
		QueryExpansion: queryExpansion,
	}

	// Add optional parameters if provided
	if ragDatabase != "" {
		req.Database = ragDatabase
	}
	if ragDataStrategy != "" {
		req.DataStrategy = ragDataStrategy
	}
	if retrievalStrategy != "" {
		req.RetrievalStrategy = retrievalStrategy
	}
	if scoreThreshold > 0 {
		req.ScoreThreshold = scoreThreshold
	}
	if distanceMetric != "" {
		req.DistanceMetric = distanceMetric
	}
	if hybridAlpha > 0 {
		req.HybridAlpha = hybridAlpha
	}
	if rerankModel != "" {
		req.RerankModel = rerankModel
	}
	if ragMaxTokens > 0 {
		req.MaxTokens = ragMaxTokens
	}

	// Parse metadata filters
	if len(metadataFilters) > 0 {
		req.MetadataFilters = parseMetadataFilters(metadataFilters)
	}

	return req
}

func parseMetadataFilters(filters []string) map[string]interface{} {
	result := make(map[string]interface{})
	for _, filter := range filters {
		parts := strings.SplitN(filter, ":", 2)
		if len(parts) == 2 {
			key := strings.TrimSpace(parts[0])
			value := strings.TrimSpace(parts[1])

			// Try to parse as JSON for complex values
			var parsedValue interface{}
			if err := json.Unmarshal([]byte(value), &parsedValue); err == nil {
				result[key] = parsedValue
			} else {
				// Use as string if not valid JSON
				result[key] = value
			}
		}
	}
	return result
}

func sendQueryRequest(serverCfg *config.ServerConfig, req QueryRequest) (*QueryResponse, error) {
	// Marshal request
	payload, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Build URL
	url := buildServerURL(serverCfg.URL, fmt.Sprintf("/v1/projects/%s/%s/rag/query", serverCfg.Namespace, serverCfg.Project))

	// Create HTTP request
	httpReq, err := http.NewRequest("POST", url, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	// Send request
	resp, err := getHTTPClient().Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	// Read response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	// Check status
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("server returned %d: %s", resp.StatusCode, string(body))
	}

	// Parse response
	var queryResp QueryResponse
	if err := json.Unmarshal(body, &queryResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return &queryResp, nil
}

func displayQueryResults(response *QueryResponse, queryText string) {
	switch outputFormat {
	case "json":
		displayJSONResults(response)
	case "table":
		displayTableResults(response, queryText)
	default:
		displayDefaultResults(response, queryText)
	}
}

func displayDefaultResults(response *QueryResponse, queryText string) {
	fmt.Printf("\n🔍 Query: \"%s\"\n", queryText)
	fmt.Printf("📊 Strategy: %s | Database: %s\n", response.RetrievalStrategy, response.Database)
	fmt.Printf("📝 Found %d results (showing top %d)\n", response.TotalResults, len(response.Results))

	if response.ProcessingTime > 0 {
		fmt.Printf("⏱️  Processing time: %.2fms\n", response.ProcessingTime)
	}

	fmt.Println("\n" + strings.Repeat("─", 80))

	for i, result := range response.Results {
		fmt.Printf("\n%d. ", i+1)

		if includeScore {
			fmt.Printf("[Score: %.4f] ", result.Score)
		}

		// Truncate content if too long
		content := result.Content
		if len(content) > 200 && !includeMetadata {
			content = content[:197] + "..."
		}
		fmt.Printf("%s\n", content)

		if includeMetadata && len(result.Metadata) > 0 {
			fmt.Println("   Metadata:")
			for k, v := range result.Metadata {
				fmt.Printf("     %s: %v\n", k, v)
			}
		}

		if i < len(response.Results)-1 {
			fmt.Println("   " + strings.Repeat("·", 40))
		}
	}

	fmt.Println("\n" + strings.Repeat("─", 80))
}

func displayTableResults(response *QueryResponse, queryText string) {
	fmt.Printf("\nQuery: %s\n", queryText)
	fmt.Printf("Results: %d | Strategy: %s | Database: %s\n\n", response.TotalResults, response.RetrievalStrategy, response.Database)

	// Table header
	if includeScore {
		fmt.Printf("%-5s %-10s %-60s\n", "Rank", "Score", "Content")
		fmt.Printf("%-5s %-10s %-60s\n", "----", "-----", strings.Repeat("-", 60))
	} else {
		fmt.Printf("%-5s %-70s\n", "Rank", "Content")
		fmt.Printf("%-5s %-70s\n", "----", strings.Repeat("-", 70))
	}

	for i, result := range response.Results {
		content := result.Content
		if len(content) > 60 {
			content = content[:57] + "..."
		}

		if includeScore {
			fmt.Printf("%-5d %-10.4f %-60s\n", i+1, result.Score, content)
		} else {
			fmt.Printf("%-5d %-70s\n", i+1, content)
		}
	}
}

func displayJSONResults(response *QueryResponse) {
	// Pretty print JSON
	output, err := json.MarshalIndent(response, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error formatting JSON: %v\n", err)
		return
	}
	fmt.Println(string(output))
}

func init() {
	// Add query command to rag
	ragCmd.AddCommand(ragQueryCmd)

	// Database and strategy flags
	ragQueryCmd.Flags().StringVar(&ragDatabase, "database", "", "Database to query (default: from config)")
	ragQueryCmd.Flags().StringVarP(&ragDataStrategy, "data-strategy", "s", "", "Data processing strategy (default: from config)")
	ragQueryCmd.Flags().StringVarP(&retrievalStrategy, "retrieval-strategy", "r", "", "Retrieval strategy to use (default: from database config)")

	// Query parameters
	ragQueryCmd.Flags().IntVarP(&topK, "top-k", "k", 5, "Number of results to return")
	ragQueryCmd.Flags().Float64Var(&scoreThreshold, "score-threshold", 0.0, "Minimum score threshold for results")
	ragQueryCmd.Flags().StringSliceVarP(&metadataFilters, "filter", "f", []string{}, "Metadata filters (format: key:value)")

	// Advanced parameters
	ragQueryCmd.Flags().StringVar(&distanceMetric, "distance-metric", "", "Distance metric override (cosine, euclidean, manhattan)")
	ragQueryCmd.Flags().Float64Var(&hybridAlpha, "hybrid-alpha", 0.5, "Alpha parameter for hybrid search (0=keyword, 1=semantic)")
	ragQueryCmd.Flags().StringVar(&rerankModel, "rerank-model", "", "Model to use for reranking results")
	ragQueryCmd.Flags().BoolVar(&queryExpansion, "query-expansion", false, "Enable query expansion")
	ragQueryCmd.Flags().IntVar(&ragMaxTokens, "max-tokens", 0, "Maximum tokens per result chunk")

	// Output formatting
	ragQueryCmd.Flags().StringVarP(&outputFormat, "output", "o", "default", "Output format (default, json, table)")
	ragQueryCmd.Flags().BoolVar(&includeMetadata, "include-metadata", false, "Include document metadata in results")
	ragQueryCmd.Flags().BoolVar(&includeScore, "include-score", false, "Include similarity scores in results")
}
