package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/llamafarm/cli/cmd/utils"
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

// fetchAvailableDatabases fetches the list of RAG databases and their strategies.
// Returns the response and an optional warning message for display in the TUI.
func fetchAvailableDatabases(serverURL, namespace, projectID string) (*DatabasesResponse, string) {
	url := fmt.Sprintf("%s/v1/projects/%s/%s/rag/databases",
		strings.TrimSuffix(serverURL, "/"),
		namespace,
		projectID)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Sprintf("Failed to build request for databases endpoint: %v", err)
	}

	resp, err := utils.GetHTTPClient().Do(req)
	if err != nil {
		return nil, fmt.Sprintf("Unable to fetch databases from server: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Sprintf("RAG databases unavailable (status %d)", resp.StatusCode)
	}

	var result DatabasesResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Sprintf("Failed to decode databases response: %v", err)
	}

	return &result, ""
}

// DatasetsResponse represents a simplified datasets list response
type DatasetsResponse struct {
	Total    int             `json:"total"`
	Datasets []DatasetDetail `json:"datasets"`
}

// DatasetDetail contains dataset fields we want to show in the menu
type DatasetDetail struct {
	Name                   string   `json:"name"`
	DataProcessingStrategy string   `json:"data_processing_strategy"`
	Database               string   `json:"database"`
	Files                  []string `json:"files"`
}

// DatasetBrief is a simplified struct the chat/menu can consume easily
type DatasetBrief struct {
	Name      string
	Strategy  string
	Database  string
	FileCount int
}

// fetchAvailableDatasets fetches datasets for a project.
// Returns the datasets and an optional warning message for display in the TUI.
func fetchAvailableDatasets(serverURL, namespace, projectID string) ([]DatasetBrief, string) {
	url := fmt.Sprintf("%s/v1/projects/%s/%s/datasets?include_extra_details=false",
		strings.TrimSuffix(serverURL, "/"),
		namespace,
		projectID)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Sprintf("Failed to build request for datasets endpoint: %v", err)
	}

	resp, err := utils.GetHTTPClient().Do(req)
	if err != nil {
		return nil, fmt.Sprintf("Unable to fetch datasets from server: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Sprintf("Datasets unavailable (status %d)", resp.StatusCode)
	}

	var result DatasetsResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Sprintf("Failed to decode datasets response: %v", err)
	}

	out := make([]DatasetBrief, 0, len(result.Datasets))
	for _, ds := range result.Datasets {
		out = append(out, DatasetBrief{
			Name:      ds.Name,
			Strategy:  ds.DataProcessingStrategy,
			Database:  ds.Database,
			FileCount: len(ds.Files),
		})
	}
	return out, ""
}
