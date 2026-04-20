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

// ModelInfo represents a model configuration
type ModelInfo struct {
	Name             string                 `json:"name"`
	Description      string                 `json:"description"`
	Provider         string                 `json:"provider"`
	Model            string                 `json:"model"`
	IsDefault        bool                   `json:"default"`
	RuntimeStatus    string                 `json:"runtime_status"`
	RuntimeLoaded    bool                   `json:"runtime_loaded"`
	RuntimeRunning   bool                   `json:"runtime_running"`
	RuntimeHost      string                 `json:"runtime_host"`
	MemoryUsageBytes int64                  `json:"memory_usage_bytes"`
	MemoryUsageHuman string                 `json:"memory_usage_human"`
	GPUAllocation    string                 `json:"gpu_allocation"`
	UptimeSeconds    int64                  `json:"uptime_seconds"`
	UptimeHuman      string                 `json:"uptime_human"`
	RuntimeMessage   string                 `json:"runtime_message"`
	RuntimeDetails   map[string]interface{} `json:"runtime_details"`
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
		utils.LogDebug(fmt.Sprintf("Error creating models request: %v", err))
		return nil
	}

	resp, err := utils.GetHTTPClient().Do(req)
	if err != nil {
		utils.LogDebug(fmt.Sprintf("Error fetching models: %v", err))
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
		utils.LogDebug(fmt.Sprintf("Error decoding models response: %v", err))
		return nil
	}

	return result.Models
}
