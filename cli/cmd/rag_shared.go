package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// RetrievalStrategyInfo represents a retrieval strategy configuration
type RetrievalStrategyInfo struct {
	Name      string `json:"name"`
	Type      string `json:"type"`
	IsDefault bool   `json:"is_default"`
}

// DatabaseInfo represents a RAG database configuration
type DatabaseInfo struct {
	Name                string                  `json:"name"`
	Type                string                  `json:"type"`
	IsDefault           bool                    `json:"is_default"`
	RetrievalStrategies []RetrievalStrategyInfo `json:"retrieval_strategies"`
}

// DatabasesResponse represents the response from /rag/databases
type DatabasesResponse struct {
	Databases       []DatabaseInfo `json:"databases"`
	DefaultDatabase *string        `json:"default_database"`
}

// fetchAvailableDatabases fetches the list of RAG databases and their strategies
func fetchAvailableDatabases(serverURL, namespace, projectID string) *DatabasesResponse {
	url := fmt.Sprintf("%s/v1/projects/%s/%s/rag/databases",
		strings.TrimSuffix(serverURL, "/"),
		namespace,
		projectID)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

    req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
        OutputWarning("Failed to build request for databases endpoint: %v", err)
		return nil
	}

    resp, err := getHTTPClient().Do(req)
	if err != nil {
        OutputWarning("Unable to fetch databases from server: %v", err)
		return nil
	}
	defer resp.Body.Close()

    if resp.StatusCode != http.StatusOK {
        OutputWarning("Databases endpoint returned status %d", resp.StatusCode)
		return nil
	}

	var result DatabasesResponse
    if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
        OutputWarning("Failed to decode databases response: %v", err)
		return nil
	}

	return &result
}
