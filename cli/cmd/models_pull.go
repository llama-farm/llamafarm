package cmd

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/llamafarm/cli/cmd/utils"
	"github.com/llamafarm/cli/internal/hfcache"
	"github.com/llamafarm/cli/internal/hfmodel"
	"github.com/spf13/cobra"
)

var modelsPullCmd = &cobra.Command{
	Use:   "pull <model-id>",
	Short: "Download a model from HuggingFace",
	Long: `Download a model from HuggingFace to the local cache.

The model-id can include an optional quantization suffix for GGUF models.
This command talks directly to the HuggingFace Hub and does not require the
LlamaFarm server to be running.

Examples:
  # Download a GGUF model with specific quantization
  lf models pull unsloth/gemma-3-1b-it-gguf:Q4_K_M

  # Download an embedding model
  lf models pull nomic-ai/nomic-embed-text-v1.5

  # Download any HuggingFace model
  lf models pull meta-llama/Llama-2-7b-hf`,
	Args: cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		modelName := args[0]
		fmt.Printf("Downloading model: %s\n", modelName)

		if err := pullModelNative(cmd.Context(), modelName); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
	},
}

var modelsStatusCmd = &cobra.Command{
	Use:   "status <model-id>",
	Short: "Check if a model is cached locally",
	Long: `Check if a model exists in the local HuggingFace cache.

This command reads the cache directory directly and does not require the
LlamaFarm server to be running.

Examples:
  # Check if a model is cached
  lf models status unsloth/gemma-3-1b-it-gguf:Q4_K_M

  # Check embedding model
  lf models status nomic-ai/nomic-embed-text-v1.5`,
	Args: cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		modelID := args[0]

		// Strip optional ":quant" suffix — the cache stores models keyed by
		// repo id, the quant just selects a file within the repo.
		baseID := modelID
		if idx := strings.LastIndex(modelID, ":"); idx != -1 {
			baseID = modelID[:idx]
		}

		_, err := hfcache.LookupRepo(baseID)
		if err != nil {
			if errors.Is(err, hfcache.ErrNotCached) {
				fmt.Printf("✗ Model %s is not cached\n", modelID)
				os.Exit(1)
			}
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

		fmt.Printf("✓ Model %s is cached\n", modelID)
	},
}

// pullModelNative drives a download via the in-process hfmodel package. The
// rendering of progress events is preserved (rate, ETA, in-place line redraw)
// from the previous SSE-based implementation so the user-facing output is
// identical, but no LlamaFarm server is involved.
func pullModelNative(ctx context.Context, modelName string) error {
	if ctx == nil {
		ctx = context.Background()
	}
	client, err := hfmodel.NewClient()
	if err != nil {
		return fmt.Errorf("init hfmodel client: %w", err)
	}
	plan, err := client.GetModelDownloadPlan(ctx, modelName)
	if err != nil {
		return err
	}

	r := newProgressRenderer()
	if err := client.DownloadModel(ctx, plan, r.callback); err != nil {
		return err
	}
	fmt.Printf("✓ Download complete\n")
	return nil
}

// progressRenderer formats hfmodel.ProgressEvent events as a human-readable
// terminal stream. With multi-file concurrent downloads, per-file in-place
// progress (`\r`-overwrite) doesn't work because multiple files would clobber
// each other's lines, so this renderer:
//
//   - Prints the init/start/cached/end events as plain lines.
//   - Aggregates progress events across all in-flight files into a single
//     bottom-line "[N/M] xxx MB / yyy MB" status that gets redrawn in place.
//
// This keeps the terminal output legible even when 4 files are downloading
// in parallel.
type progressRenderer struct {
	mu sync.Mutex

	totalFiles    int
	completedFile int
	totalBytes    int64
	// inFlight tracks downloaded bytes per still-downloading file.
	inFlight map[string]int64
	// completedBytes accumulates as files finish (end/cached events).
	completedBytes int64

	lastStatusShown bool
}

func newProgressRenderer() *progressRenderer {
	return &progressRenderer{
		inFlight: map[string]int64{},
	}
}

func (r *progressRenderer) callback(ev hfmodel.ProgressEvent) {
	r.mu.Lock()
	defer r.mu.Unlock()

	switch ev.Event {
	case "init":
		r.totalFiles = ev.FileCount
		r.totalBytes = ev.TotalSize
		if ev.TotalSize > 0 {
			sizeStr := utils.FormatBytes(ev.TotalSize)
			if ev.IsGGUF && ev.SelectedFile != "" {
				fmt.Printf("  Model: %s\n", ev.ModelID)
				fmt.Printf("  File: %s (%s)\n", ev.SelectedFile, sizeStr)
			} else {
				fmt.Printf("  Model: %s (%s, %d files)\n", ev.ModelID, sizeStr, ev.FileCount)
			}
		} else {
			fmt.Printf("  Model: %s (%d files)\n", ev.ModelID, ev.FileCount)
		}

	case "start":
		r.inFlight[ev.File] = ev.Downloaded
		r.clearStatusLine()
		if ev.Total > 1024*1024 {
			fmt.Printf("  ↓ %s (%s)\n", ev.File, utils.FormatBytes(ev.Total))
		} else if ev.File != "" {
			fmt.Printf("  ↓ %s\n", ev.File)
		}
		r.drawStatusLine()

	case "progress":
		r.inFlight[ev.File] = ev.Downloaded
		r.drawStatusLine()

	case "cached":
		r.completedFile++
		r.completedBytes += ev.Size
		r.clearStatusLine()
		fmt.Printf("  ✓ %s (cached)\n", ev.File)
		r.drawStatusLine()

	case "end":
		r.completedFile++
		r.completedBytes += ev.Total
		delete(r.inFlight, ev.File)
		r.drawStatusLine()

	case "done":
		r.clearStatusLine()

	case "warning":
		r.clearStatusLine()
		fmt.Fprintf(os.Stderr, "  ! %s\n", ev.Message)
		r.drawStatusLine()

	case "error":
		r.clearStatusLine()
		fmt.Fprintf(os.Stderr, "  ✗ %s: %s\n", ev.File, ev.Message)
	}
}

// drawStatusLine writes a single-line aggregated status to stdout. Caller
// must hold r.mu.
func (r *progressRenderer) drawStatusLine() {
	if r.totalFiles == 0 {
		return
	}
	var sumDownloaded int64
	for _, n := range r.inFlight {
		sumDownloaded += n
	}
	overall := sumDownloaded + r.completedBytes
	if overall > r.totalBytes && r.totalBytes > 0 {
		overall = r.totalBytes
	}
	pct := 0.0
	if r.totalBytes > 0 {
		pct = float64(overall) / float64(r.totalBytes) * 100
	}
	line := fmt.Sprintf("  [%d/%d] %s / %s (%.1f%%)",
		r.completedFile, r.totalFiles,
		utils.FormatBytes(overall), utils.FormatBytes(r.totalBytes), pct)
	fmt.Printf("\r%-72s", line)
	_ = os.Stdout.Sync()
	r.lastStatusShown = true
}

// clearStatusLine erases the current status line. Caller must hold r.mu.
func (r *progressRenderer) clearStatusLine() {
	if !r.lastStatusShown {
		return
	}
	fmt.Printf("\r%-72s\r", "")
	r.lastStatusShown = false
}

func init() {
	modelsCmd.AddCommand(modelsPullCmd)
	modelsCmd.AddCommand(modelsStatusCmd)
}
