package hfmodel

import (
	"context"
	"fmt"
	"sync"

	"golang.org/x/sync/errgroup"
)

// ModelDownloadPlan describes a complete download — what files, how big, and
// for GGUF models which quantization was selected.
type ModelDownloadPlan struct {
	ModelID      string
	Quantization string
	SelectedFile string // for GGUF: the chosen file
	IsGGUF       bool
	TotalSize    int64
	Files        []string // file paths within the repo
	// CommitHash is the resolved commit sha for the requested revision
	// (always "main" today). All files in the plan share the same commit
	// hash because they come from a single HF Hub revision; capturing it
	// here once avoids per-file commit-hash discovery, which is
	// unreliable for LFS files (CDN responses don't echo X-Repo-Commit).
	CommitHash string
}

// GetModelDownloadPlan inspects a model on the HuggingFace Hub and returns a
// plan for what to download. For GGUF repos, the plan contains a single
// quant-selected file. For other repos, all non-directory files in the tree.
//
// modelName may include an optional ":quant" suffix that is parsed and used
// for GGUF selection.
func (c *Client) GetModelDownloadPlan(ctx context.Context, modelName string) (*ModelDownloadPlan, error) {
	modelID, quant := ParseModelWithQuantization(modelName)
	if err := ValidateModelID(modelID); err != nil {
		return nil, err
	}
	if IsOffline() {
		return nil, &OfflineError{ModelID: modelID, Op: "get_model_download_plan"}
	}

	// Resolve the commit sha up-front so all files share the same commit
	// hash. LFS files don't reliably echo X-Repo-Commit through CDN
	// redirects, so we cannot rely on per-file HEAD responses.
	commitHash, err := c.GetModelCommitHash(ctx, modelID)
	if err != nil {
		return nil, err
	}

	tree, err := c.ListRepoTree(ctx, modelID, "main")
	if err != nil {
		return nil, err
	}

	// Bucket files by type.
	var ggufFiles []string
	var allFiles []string
	sizes := make(map[string]int64)
	for _, e := range tree {
		if e.Type != "file" {
			continue
		}
		allFiles = append(allFiles, e.Path)
		sizes[e.Path] = e.Size
		if len(e.Path) >= 5 && e.Path[len(e.Path)-5:] == ".gguf" {
			ggufFiles = append(ggufFiles, e.Path)
		}
	}

	if len(ggufFiles) > 0 {
		selected := SelectGGUFFile(ggufFiles, quant)
		if selected == "" {
			return nil, fmt.Errorf("no suitable GGUF file in %s", modelID)
		}
		return &ModelDownloadPlan{
			ModelID:      modelID,
			Quantization: quant,
			SelectedFile: selected,
			IsGGUF:       true,
			TotalSize:    sizes[selected],
			Files:        []string{selected},
			CommitHash:   commitHash,
		}, nil
	}

	var total int64
	for _, s := range sizes {
		total += s
	}
	return &ModelDownloadPlan{
		ModelID:    modelID,
		IsGGUF:     false,
		TotalSize:  total,
		Files:      allFiles,
		CommitHash: commitHash,
	}, nil
}

// DownloadModel executes a ModelDownloadPlan, downloading every file in the
// plan with bounded concurrency. Emits an `init` event up front, per-file
// `start`/`progress`/`end`/`cached` events, and a final `done` event.
//
// On any per-file failure, the entire pull aborts and the error is returned.
// Already-downloaded blobs are left in place so the user can re-run the
// command and resume.
func (c *Client) DownloadModel(ctx context.Context, plan *ModelDownloadPlan, cb EventCallback) error {
	if cb == nil {
		cb = func(ProgressEvent) {}
	}

	cb(ProgressEvent{
		Event:        "init",
		ModelID:      plan.ModelID,
		Quantization: plan.Quantization,
		SelectedFile: plan.SelectedFile,
		TotalSize:    plan.TotalSize,
		IsGGUF:       plan.IsGGUF,
		FileCount:    len(plan.Files),
	})

	if len(plan.Files) == 0 {
		repoDir, err := repoCacheDir(plan.ModelID)
		if err != nil {
			return err
		}
		cb(ProgressEvent{Event: "done", LocalDir: repoDir})
		return nil
	}

	// Bounded-concurrency download. The progress callback may be invoked
	// from multiple goroutines, so wrap it with a mutex.
	var cbMu sync.Mutex
	safeCB := func(ev ProgressEvent) {
		cbMu.Lock()
		defer cbMu.Unlock()
		cb(ev)
	}

	const concurrency = 4
	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(concurrency)

	for _, file := range plan.Files {
		filename := file
		g.Go(func() error {
			md, err := c.GetFileMetadata(gctx, plan.ModelID, "main", filename)
			if err != nil {
				return fmt.Errorf("metadata for %s: %w", filename, err)
			}
			// Override with the plan's resolved commit hash so all files
			// land in the same snapshot directory. Per-file metadata may
			// be missing X-Repo-Commit for LFS files (the CDN response
			// after redirect doesn't echo it).
			if plan.CommitHash != "" {
				md.CommitHash = plan.CommitHash
			}
			fp := SingleFilePlan{
				RepoID:   plan.ModelID,
				Filename: filename,
				Metadata: md,
			}
			return c.DownloadFile(gctx, fp, safeCB)
		})
	}
	if err := g.Wait(); err != nil {
		return err
	}

	repoDir, err := repoCacheDir(plan.ModelID)
	if err != nil {
		return err
	}
	cb(ProgressEvent{Event: "done", LocalDir: repoDir})
	return nil
}
