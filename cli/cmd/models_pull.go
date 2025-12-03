package cmd

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/llamafarm/cli/cmd/orchestrator"
	"github.com/llamafarm/cli/cmd/utils"
	"github.com/spf13/cobra"
)

// SSE event structure from the server
type downloadEvent struct {
	Event   string `json:"event"`
	Desc    string `json:"desc"`
	Total   *int64 `json:"total"`
	N       int64  `json:"n"`
	Message string `json:"message"`
}

var modelsPullCmd = &cobra.Command{
	Use:   "pull <model-id>",
	Short: "Download a model from HuggingFace",
	Long: `Download a model from HuggingFace to the local cache.

The model-id can include an optional quantization suffix for GGUF models.

Examples:
  # Download a GGUF model with specific quantization
  lf models pull unsloth/gemma-3-1b-it-gguf:Q4_K_M

  # Download an embedding model
  lf models pull nomic-ai/nomic-embed-text-v1.5

  # Download any HuggingFace model
  lf models pull meta-llama/Llama-2-7b-hf`,
	Args: cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		modelID := args[0]

		// Ensure server is running
		orchestrator.EnsureServicesOrExit(serverURL, "server")

		fmt.Printf("Downloading model: %s\n", modelID)

		err := pullModel(serverURL, modelID)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
	},
}

var modelsStatusCmd = &cobra.Command{
	Use:   "status <model-id>",
	Short: "Check if a model is cached locally",
	Long: `Check if a model exists in the local HuggingFace cache.

Examples:
  # Check if a model is cached
  lf models status unsloth/gemma-3-1b-it-gguf:Q4_K_M

  # Check embedding model
  lf models status nomic-ai/nomic-embed-text-v1.5`,
	Args: cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		modelID := args[0]

		// Ensure server is running
		orchestrator.EnsureServicesOrExit(serverURL, "server")

		cached, err := checkModelStatus(serverURL, modelID)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

		if cached {
			fmt.Printf("✓ Model %s is cached\n", modelID)
			os.Exit(0)
		} else {
			fmt.Printf("✗ Model %s is not cached\n", modelID)
			os.Exit(1)
		}
	},
}

// pullModel downloads a model using the server's SSE endpoint
func pullModel(serverURL, modelID string) error {
	url := fmt.Sprintf("%s/v1/models/download", strings.TrimSuffix(serverURL, "/"))

	requestBody, err := json.Marshal(map[string]string{
		"provider":   "universal",
		"model_name": modelID,
	})
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	// Create request with long timeout for large model downloads
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(requestBody))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	resp, err := utils.GetHTTPClient().Do(req)
	if err != nil {
		return fmt.Errorf("failed to connect to server: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("server returned status %d", resp.StatusCode)
	}

	// Parse SSE stream using buffered reader for better streaming
	reader := bufio.NewReader(resp.Body)
	var lastProgress float64
	var currentDesc string
	downloadComplete := false

	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			return fmt.Errorf("error reading response: %w", err)
		}

		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		var event downloadEvent
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			continue
		}

		switch event.Event {
		case "start":
			currentDesc = event.Desc
			if event.Total != nil && *event.Total > 1024*1024 { // Only show size if > 1MB
				totalMB := float64(*event.Total) / 1024 / 1024
				fmt.Printf("  %s (%.1f MB)...\n", currentDesc, totalMB)
			} else if currentDesc != "" {
				fmt.Printf("  %s...\n", currentDesc)
			}
			os.Stdout.Sync()
		case "progress":
			// Calculate progress percentage
			var progress float64
			if event.Total != nil && *event.Total > 0 {
				progress = float64(event.N) / float64(*event.Total) * 100
			}
			// Only print progress updates every 5% for actual downloads
			if event.Total != nil && *event.Total > 1024*1024 { // Only show for files > 1MB
				if progress-lastProgress >= 5 || progress >= 100 {
					fmt.Printf("\r  Progress: %.0f%%", progress)
					os.Stdout.Sync()
					lastProgress = progress
				}
			}
		case "end":
			// Only show completion if we showed progress
			if lastProgress > 0 {
				fmt.Printf("\r  Progress: 100%%\n")
			}
		case "done":
			fmt.Printf("✓ Download complete\n")
			downloadComplete = true
			return nil
		case "error":
			return fmt.Errorf("download failed: %s", event.Message)
		}
	}

	// If we exit the loop without receiving a "done" event, the download was incomplete
	if !downloadComplete {
		return fmt.Errorf("download incomplete: connection closed before completion")
	}
	return nil
}

// checkModelStatus checks if a model is in the local cache
func checkModelStatus(serverURL, modelID string) (bool, error) {
	url := fmt.Sprintf("%s/v1/models", strings.TrimSuffix(serverURL, "/"))

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return false, fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := utils.GetHTTPClient().Do(req)
	if err != nil {
		return false, fmt.Errorf("failed to connect to server: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return false, fmt.Errorf("server returned status %d", resp.StatusCode)
	}

	var result struct {
		Data []struct {
			ID   string `json:"id"`
			Name string `json:"name"`
		} `json:"data"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return false, fmt.Errorf("failed to parse response: %w", err)
	}

	// Parse model ID to handle quantization suffix
	baseModelID := modelID
	if idx := strings.LastIndex(modelID, ":"); idx != -1 {
		baseModelID = modelID[:idx]
	}

	// Check if model is in the cache
	for _, model := range result.Data {
		if model.ID == baseModelID || model.ID == modelID {
			return true, nil
		}
	}

	return false, nil
}

func init() {
	modelsCmd.AddCommand(modelsPullCmd)
	modelsCmd.AddCommand(modelsStatusCmd)
}
