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

// SSE event structures from the server
type downloadEvent struct {
	Event string `json:"event"`
	// Common fields
	File    string `json:"file,omitempty"`
	Message string `json:"message,omitempty"`
	// Progress fields
	Downloaded  int64    `json:"downloaded,omitempty"`
	Total       int64    `json:"total,omitempty"`
	Percent     float64  `json:"percent,omitempty"`
	BytesPerSec int64    `json:"bytes_per_sec,omitempty"`
	ETASeconds  *float64 `json:"eta_seconds,omitempty"`
	// Init fields
	ModelID      string `json:"model_id,omitempty"`
	Quantization string `json:"quantization,omitempty"`
	SelectedFile string `json:"selected_file,omitempty"`
	TotalSize    int64  `json:"total_size,omitempty"`
	IsGGUF       bool   `json:"is_gguf,omitempty"`
	FileCount    int    `json:"file_count,omitempty"`
	// Done fields
	LocalDir string `json:"local_dir,omitempty"`
	// Cached fields
	Size int64 `json:"size,omitempty"`
	// Keepalive flag
	Keepalive bool `json:"keepalive,omitempty"`
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

	resp, err := utils.GetHTTPClientWithTimeout(0).Do(req)
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
	var currentFile string

	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			return fmt.Errorf("error reading response: %w", err)
		}

		utils.LogDebug(fmt.Sprintf("SSE line: %s", line))

		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		var event downloadEvent
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			continue
		}

		// Skip keepalive events for display purposes
		if event.Keepalive {
			continue
		}

		switch event.Event {
		case "init":
			// Display initial model info
			if event.TotalSize > 0 {
				sizeStr := utils.FormatBytes(event.TotalSize)
				if event.IsGGUF && event.SelectedFile != "" {
					fmt.Printf("  Model: %s\n", event.ModelID)
					fmt.Printf("  File: %s (%s)\n", event.SelectedFile, sizeStr)
				} else {
					fmt.Printf("  Model: %s (%s, %d files)\n", event.ModelID, sizeStr, event.FileCount)
				}
			} else {
				fmt.Printf("  Model: %s (%d files)\n", event.ModelID, event.FileCount)
			}
			os.Stdout.Sync()

		case "start":
			currentFile = event.File
			lastProgress = 0
			if event.Total > 1024*1024 { // Only show size if > 1MB
				sizeStr := utils.FormatBytes(event.Total)
				fmt.Printf("  Downloading %s (%s)...\n", currentFile, sizeStr)
			} else if currentFile != "" {
				fmt.Printf("  Downloading %s...\n", currentFile)
			}
			os.Stdout.Sync()

		case "progress":
			// Use the percent directly from the event
			progress := event.Percent
			if event.Total > 1024*1024 { // Only show for files > 1MB
				// Build progress line with rate and ETA
				rateStr := ""
				etaStr := ""
				if event.BytesPerSec > 0 {
					rateStr = fmt.Sprintf(" @ %s", utils.FormatTransferRate(event.BytesPerSec))
				}
				// Only show ETA if more than 1 second remaining
				if event.ETASeconds != nil && *event.ETASeconds > 1 {
					etaStr = fmt.Sprintf(", ETA: %s", utils.FormatDuration(*event.ETASeconds))
				}
				// Use fixed-width output to ensure clean line overwrites
				line := fmt.Sprintf("  Progress: %.1f%%%s%s", progress, rateStr, etaStr)
				// Pad to 60 chars to overwrite any previous longer content
				fmt.Printf("\r%-60s", line)
				os.Stdout.Sync()
				lastProgress = progress
			}

		case "cached":
			fmt.Printf("  ✓ %s (cached)\n", event.File)
			os.Stdout.Sync()

		case "end":
			// Only show completion if we showed progress
			if lastProgress > 0 {
				fmt.Printf("\r  Progress: 100%%\n")
			}
			lastProgress = 0

		case "done":
			fmt.Printf("✓ Download complete\n")
			return nil

		case "error":
			return fmt.Errorf("download failed: %s", event.Message)
		}
	}

	// If we reach here, the stream ended without a "done" event (e.g., network drop)
	// The downloadComplete flag is never set to true because we return immediately on "done"
	return fmt.Errorf("download incomplete: connection closed before completion")
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
