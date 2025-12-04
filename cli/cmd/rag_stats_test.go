package cmd

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/llamafarm/cli/cmd/config"
	"github.com/llamafarm/cli/cmd/utils"
)

func TestFetchRAGStats_Success(t *testing.T) {
	// Mock response matching RAGStats structure
	mockResponse := `{
		"database": "test_db",
		"vector_count": 1000,
		"document_count": 50,
		"chunk_count": 1000,
		"collection_size_bytes": 5242880,
		"index_size_bytes": 1048576,
		"embedding_dimension": 768,
		"distance_metric": "cosine",
		"last_updated": "2025-01-01T12:00:00Z",
		"metadata": {"collection_name": "test_db"}
	}`

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify the request path
		expectedPath := "/v1/projects/test-ns/test-project/rag/stats"
		if r.URL.Path != expectedPath {
			t.Errorf("unexpected path: got %s, want %s", r.URL.Path, expectedPath)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		io.WriteString(w, mockResponse)
	}))
	defer ts.Close()

	cfg := &config.ServerConfig{
		URL:       ts.URL,
		Namespace: "test-ns",
		Project:   "test-project",
	}

	stats, err := fetchRAGStats(cfg, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if stats.Database != "test_db" {
		t.Errorf("expected database 'test_db', got '%s'", stats.Database)
	}
	if stats.VectorCount != 1000 {
		t.Errorf("expected vector_count 1000, got %d", stats.VectorCount)
	}
	if stats.DocumentCount != 50 {
		t.Errorf("expected document_count 50, got %d", stats.DocumentCount)
	}
	if stats.ChunkCount != 1000 {
		t.Errorf("expected chunk_count 1000, got %d", stats.ChunkCount)
	}
	if stats.EmbeddingDim != 768 {
		t.Errorf("expected embedding_dimension 768, got %d", stats.EmbeddingDim)
	}
	if stats.DistanceMetric != "cosine" {
		t.Errorf("expected distance_metric 'cosine', got '%s'", stats.DistanceMetric)
	}
}

func TestFetchRAGStats_WithDatabaseParam(t *testing.T) {
	mockResponse := `{
		"database": "specific_db",
		"vector_count": 500,
		"document_count": 25,
		"chunk_count": 500,
		"collection_size_bytes": 2621440,
		"index_size_bytes": 524288,
		"embedding_dimension": 1536,
		"distance_metric": "l2",
		"last_updated": "2025-01-01T12:00:00Z",
		"metadata": null
	}`

	var receivedQuery string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedQuery = r.URL.RawQuery
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		io.WriteString(w, mockResponse)
	}))
	defer ts.Close()

	cfg := &config.ServerConfig{
		URL:       ts.URL,
		Namespace: "test-ns",
		Project:   "test-project",
	}

	stats, err := fetchRAGStats(cfg, "specific_db")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify query parameter was passed
	if receivedQuery != "database=specific_db" {
		t.Errorf("expected query 'database=specific_db', got '%s'", receivedQuery)
	}

	if stats.Database != "specific_db" {
		t.Errorf("expected database 'specific_db', got '%s'", stats.Database)
	}
	if stats.EmbeddingDim != 1536 {
		t.Errorf("expected embedding_dimension 1536, got %d", stats.EmbeddingDim)
	}
}

func TestFetchRAGStats_NotFoundError(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		io.WriteString(w, `{"detail":"Database 'nonexistent' not found"}`)
	}))
	defer ts.Close()

	cfg := &config.ServerConfig{
		URL:       ts.URL,
		Namespace: "test-ns",
		Project:   "test-project",
	}

	_, err := fetchRAGStats(cfg, "nonexistent")
	if err == nil {
		t.Fatal("expected error, got nil")
	}

	// Verify the error message is human-friendly (uses PrettyServerError)
	if !strings.Contains(err.Error(), "Database 'nonexistent' not found") {
		t.Errorf("expected friendly error message, got: %v", err)
	}
	// Should NOT contain raw JSON
	if strings.Contains(err.Error(), `{"detail"`) {
		t.Errorf("error should not contain raw JSON: %v", err)
	}
}

func TestFetchRAGStats_RAGNotConfigured(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		io.WriteString(w, `{"detail":"RAG not configured for this project"}`)
	}))
	defer ts.Close()

	cfg := &config.ServerConfig{
		URL:       ts.URL,
		Namespace: "test-ns",
		Project:   "test-project",
	}

	_, err := fetchRAGStats(cfg, "")
	if err == nil {
		t.Fatal("expected error, got nil")
	}

	if !strings.Contains(err.Error(), "RAG not configured") {
		t.Errorf("expected 'RAG not configured' in error, got: %v", err)
	}
}

func TestFetchRAGHealth_PrettyError(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		io.WriteString(w, `{"detail":"Database 'missing' not found"}`)
	}))
	defer ts.Close()

	cfg := &config.ServerConfig{
		URL:       ts.URL,
		Namespace: "test-ns",
		Project:   "test-project",
	}

	_, err := fetchRAGHealth(cfg, "missing")
	if err == nil {
		t.Fatal("expected error, got nil")
	}

	// Verify error uses PrettyServerError
	if !strings.Contains(err.Error(), "Database 'missing' not found") {
		t.Errorf("expected friendly error message, got: %v", err)
	}
}

func TestPrettyServerError_ExtractsDetail(t *testing.T) {
	tests := []struct {
		name           string
		body           string
		expectedResult string
	}{
		{
			name:           "simple detail string",
			body:           `{"detail":"Database not found"}`,
			expectedResult: "Database not found",
		},
		{
			name:           "message field",
			body:           `{"message":"Something went wrong"}`,
			expectedResult: "Something went wrong",
		},
		{
			name:           "error field",
			body:           `{"error":"Internal error occurred"}`,
			expectedResult: "Internal error occurred",
		},
		{
			name:           "nested detail message",
			body:           `{"detail":{"message":"Validation failed"}}`,
			expectedResult: "Validation failed",
		},
		{
			name:           "array detail",
			body:           `{"detail":[{"message":"Field required"}]}`,
			expectedResult: "Field required",
		},
		{
			name:           "non-json response",
			body:           `Internal Server Error`,
			expectedResult: "Internal Server Error",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp := &http.Response{
				StatusCode: 400,
				Status:     "400 Bad Request",
				Header:     make(http.Header),
			}

			result := utils.PrettyServerError(resp, []byte(tt.body))
			if result != tt.expectedResult {
				t.Errorf("expected '%s', got '%s'", tt.expectedResult, result)
			}
		})
	}
}

func TestFormatBytes(t *testing.T) {
	tests := []struct {
		bytes    int64
		expected string
	}{
		{0, "0 B"},
		{512, "512 B"},
		{1024, "1.0 KB"},
		{1536, "1.5 KB"},
		{1048576, "1.0 MB"},
		{1073741824, "1.0 GB"},
		{5242880, "5.0 MB"},
	}

	for _, tt := range tests {
		result := formatBytes(tt.bytes)
		if result != tt.expected {
			t.Errorf("formatBytes(%d) = %s, want %s", tt.bytes, result, tt.expected)
		}
	}
}

func TestTruncateString(t *testing.T) {
	tests := []struct {
		input    string
		maxLen   int
		expected string
	}{
		{"short", 10, "short"},
		{"exactly10!", 10, "exactly10!"},
		{"this is a long string", 10, "this is..."},
		{"", 10, ""},
	}

	for _, tt := range tests {
		result := truncateString(tt.input, tt.maxLen)
		if result != tt.expected {
			t.Errorf("truncateString(%q, %d) = %q, want %q", tt.input, tt.maxLen, result, tt.expected)
		}
	}
}
