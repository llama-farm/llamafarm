package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// ModelInfo represents a model configuration
type ModelInfo struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Provider    string `json:"provider"`
	Model       string `json:"model"`
	IsDefault   bool   `json:"default"`
}

// fetchAvailableModels fetches the list of available models for a project
func fetchAvailableModels(serverURL, namespace, projectID string) []ModelInfo {
	url := fmt.Sprintf("%s/v1/projects/%s/%s/models",
		strings.TrimSuffix(serverURL, "/"),
		namespace,
		projectID)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		logDebug(fmt.Sprintf("Error creating models request: %v", err))
		return nil
	}

	resp, err := getHTTPClient().Do(req)
	if err != nil {
		logDebug(fmt.Sprintf("Error fetching models: %v", err))
		return nil
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil
	}

	var result struct {
		Models []ModelInfo `json:"models"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		logDebug(fmt.Sprintf("Error decoding models response: %v", err))
		return nil
	}

	return result.Models
}
