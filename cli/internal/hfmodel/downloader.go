package hfmodel

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/llamafarm/cli/internal/hfcache"
)

// progressChunkSize bounds how often progress events fire while writing the
// blob to disk. 1 MiB matches the chunk size the Python implementation uses.
const progressChunkSize = 1024 * 1024

// EventCallback receives a stream of progress events during a download. It
// is called from the goroutine driving the download (no concurrent calls
// per Client.DownloadFile invocation).
type EventCallback func(ProgressEvent)

// ProgressEvent is a tagged-union shape that mirrors the SSE events emitted
// by the existing Python server-side downloader. Field naming is preserved
// verbatim so that a future server-side adapter (out of scope here) can
// re-encode events as SSE without renaming anything.
type ProgressEvent struct {
	// Event discriminator: "init", "start", "progress", "cached", "end",
	// "done", "error", "warning".
	Event string `json:"event"`

	// Init fields.
	ModelID      string `json:"model_id,omitempty"`
	Quantization string `json:"quantization,omitempty"`
	SelectedFile string `json:"selected_file,omitempty"`
	TotalSize    int64  `json:"total_size,omitempty"`
	IsGGUF       bool   `json:"is_gguf,omitempty"`
	FileCount    int    `json:"file_count,omitempty"`

	// Per-file fields.
	File       string `json:"file,omitempty"`
	Downloaded int64  `json:"downloaded,omitempty"`
	Total      int64  `json:"total,omitempty"`
	// Percent is 0-100. omitempty would drop legit 0 values, so always
	// emit it during progress events. The JSON tag is preserved for
	// future SSE adapter compatibility.
	Percent     float64  `json:"percent,omitempty"`
	BytesPerSec int64    `json:"bytes_per_sec,omitempty"`
	ETASeconds  *float64 `json:"eta_seconds,omitempty"`

	// Cached / done fields.
	Size     int64  `json:"size,omitempty"`
	LocalDir string `json:"local_dir,omitempty"`

	// Error / warning.
	Message string `json:"message,omitempty"`
}

// SingleFilePlan describes one file to download as part of a model pull.
type SingleFilePlan struct {
	RepoID   string
	Filename string
	// Metadata is the result of GetFileMetadata for this file. Carries the
	// resolved CDN URL, etag, size, and commit hash.
	Metadata *FileMetadata
}

// repoCacheDir returns the absolute path of the repo's cache directory under
// the configured HuggingFace cache root.
func repoCacheDir(repoID string) (string, error) {
	root, err := hfcache.CacheRoot()
	if err != nil {
		return "", err
	}
	if err := ValidateModelID(repoID); err != nil {
		return "", err
	}
	folder := "models--" + strings.ReplaceAll(repoID, "/", "--")
	return filepath.Join(root, folder), nil
}

// blobPathForETag returns the absolute blob path for an etag.
func blobPathForETag(repoDir, etag string) string {
	// HF stores blobs by raw etag (no quotes, no W/ prefix). For LFS files
	// the etag is the SHA-256 of the content. For non-LFS files it's a git
	// blob oid.
	return filepath.Join(repoDir, "blobs", etag)
}

// snapshotPathForFile returns the snapshot symlink/copy path for a file at a
// given commit.
func snapshotPathForFile(repoDir, commitHash, filename string) string {
	return filepath.Join(repoDir, "snapshots", commitHash, filename)
}

// DownloadFile streams a single file into the HuggingFace cache, writes the
// blob, creates the snapshot symlink (with Windows fallback), and updates
// refs/main with the commit hash. Emits progress events via cb.
//
// If the blob already exists at the expected size, emits a "cached" event
// and returns nil without making a network request.
//
// On any failure, the partial download is left as `<etag>.tmp` so a future
// invocation can resume.
func (c *Client) DownloadFile(ctx context.Context, plan SingleFilePlan, cb EventCallback) error {
	if cb == nil {
		cb = func(ProgressEvent) {}
	}
	if plan.Metadata == nil {
		return fmt.Errorf("plan.Metadata is required")
	}
	if plan.Metadata.ETag == "" {
		return fmt.Errorf("file metadata missing etag for %s", plan.Filename)
	}
	if IsOffline() {
		return &OfflineError{ModelID: plan.RepoID, Op: "download_file"}
	}

	repoDir, err := repoCacheDir(plan.RepoID)
	if err != nil {
		return err
	}
	blobsDir := filepath.Join(repoDir, "blobs")
	refsDir := filepath.Join(repoDir, "refs")
	if err := os.MkdirAll(blobsDir, 0o755); err != nil {
		return fmt.Errorf("create blobs dir: %w", err)
	}
	if err := os.MkdirAll(refsDir, 0o755); err != nil {
		return fmt.Errorf("create refs dir: %w", err)
	}

	blobPath := blobPathForETag(repoDir, plan.Metadata.ETag)

	// Coordinate with concurrent downloaders (Go or Python) via the same
	// .lock file naming convention huggingface_hub.filelock uses.
	release, err := acquireLock(blobPath + ".lock")
	if err != nil {
		return fmt.Errorf("acquire lock: %w", err)
	}
	defer release()

	// Already cached?
	if info, statErr := os.Stat(blobPath); statErr == nil && info.Size() == plan.Metadata.Size {
		cb(ProgressEvent{
			Event: "cached",
			File:  plan.Filename,
			Size:  info.Size(),
		})
	} else {
		// Need to download. Stream into <etag>.tmp, then atomically rename.
		if err := c.streamBlob(ctx, plan, blobPath, cb); err != nil {
			return err
		}
	}

	// Snapshot symlink/copy.
	commit := plan.Metadata.CommitHash
	if commit == "" {
		commit = "main"
	}
	snapPath := snapshotPathForFile(repoDir, commit, plan.Filename)
	if err := os.MkdirAll(filepath.Dir(snapPath), 0o755); err != nil {
		return fmt.Errorf("create snapshot dir: %w", err)
	}
	if err := CreateSnapshotSymlink(blobPath, snapPath); err != nil {
		return fmt.Errorf("create snapshot symlink: %w", err)
	}

	// refs/main → commit hash.
	if commit != "main" {
		if err := writeRefsMain(refsDir, commit); err != nil {
			return fmt.Errorf("write refs/main: %w", err)
		}
	}
	return nil
}

// streamBlob does the actual HTTP fetch + write. Handles resume from a
// pre-existing `.tmp` file via Range/If-Range headers. After a successful
// transfer, the size is verified against the metadata before the rename.
func (c *Client) streamBlob(ctx context.Context, plan SingleFilePlan, blobPath string, cb EventCallback) error {
	tmpPath := blobPath + ".tmp"
	expectedSize := plan.Metadata.Size

	// Build the HTTP request.
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, plan.Metadata.URL, nil)
	if err != nil {
		return err
	}
	c.applyHeaders(req)

	// Resume detection: if a .tmp exists and we have an etag, ask for the
	// remaining bytes via Range with If-Range guarding against etag drift.
	// If-Range MUST be the wire-format quoted etag — sending the unquoted
	// blob-filename form makes most servers ignore the guard and return a
	// full 200 instead of a 206, defeating resume.
	var resumeFrom int64
	if info, statErr := os.Stat(tmpPath); statErr == nil && info.Size() > 0 && info.Size() < expectedSize {
		resumeFrom = info.Size()
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", resumeFrom))
		req.Header.Set("If-Range", quoteETagForHeader(plan.Metadata.ETag))
	}

	// Use a no-timeout client for the body fetch — model files are large
	// and may take hours.
	bodyClient := &http.Client{
		Timeout:   0,
		Transport: c.httpClient.Transport,
	}
	resp, err := bodyClient.Do(req)
	if err != nil {
		return classifyTransportError(err)
	}
	defer resp.Body.Close()

	if err := classifyHTTPStatus(resp, plan.RepoID); err != nil {
		return err
	}

	// Open the temp file. Append on resume, truncate otherwise.
	var f *os.File
	if resp.StatusCode == http.StatusPartialContent && resumeFrom > 0 {
		f, err = os.OpenFile(tmpPath, os.O_WRONLY|os.O_APPEND, 0o644)
	} else {
		// Server returned 200 (etag changed or no Range support) or this
		// is a fresh download. Truncate any stale partial.
		resumeFrom = 0
		f, err = os.OpenFile(tmpPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o644)
	}
	if err != nil {
		return fmt.Errorf("open tmp file: %w", err)
	}

	// Emit `start` once we know we're committed to a transfer.
	cb(ProgressEvent{
		Event:      "start",
		File:       plan.Filename,
		Total:      expectedSize,
		Downloaded: resumeFrom,
	})

	downloaded := resumeFrom
	startTime := time.Now()
	buf := make([]byte, progressChunkSize)
	var lastProgressEmit time.Time
	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			if _, writeErr := f.Write(buf[:n]); writeErr != nil {
				_ = f.Close()
				return fmt.Errorf("write tmp file: %w", writeErr)
			}
			downloaded += int64(n)

			// Emit progress at most every ~100ms to avoid flooding the
			// terminal on fast connections.
			now := time.Now()
			if now.Sub(lastProgressEmit) >= 100*time.Millisecond {
				lastProgressEmit = now
				elapsed := now.Sub(startTime).Seconds()
				var rate int64
				var etaPtr *float64
				if elapsed > 0 {
					rate = int64(float64(downloaded-resumeFrom) / elapsed)
					if rate > 0 && downloaded < expectedSize {
						eta := float64(expectedSize-downloaded) / float64(rate)
						etaPtr = &eta
					}
				}
				var pct float64
				if expectedSize > 0 {
					pct = float64(downloaded) / float64(expectedSize) * 100
				}
				cb(ProgressEvent{
					Event:       "progress",
					File:        plan.Filename,
					Downloaded:  downloaded,
					Total:       expectedSize,
					Percent:     pct,
					BytesPerSec: rate,
					ETASeconds:  etaPtr,
				})
			}
		}
		if readErr != nil {
			if errors.Is(readErr, io.EOF) {
				break
			}
			_ = f.Close()
			return classifyTransportError(readErr)
		}
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("close tmp file: %w", err)
	}

	// Verify size before the rename.
	if expectedSize > 0 && downloaded != expectedSize {
		// Don't delete the tmp — leave it for inspection / next resume.
		return fmt.Errorf("size mismatch for %s: got %d bytes, expected %d", plan.Filename, downloaded, expectedSize)
	}

	if err := os.Rename(tmpPath, blobPath); err != nil {
		return fmt.Errorf("rename blob: %w", err)
	}

	cb(ProgressEvent{
		Event:      "end",
		File:       plan.Filename,
		Downloaded: downloaded,
		Total:      expectedSize,
	})
	return nil
}

// writeRefsMain atomically writes the commit hash into refs/main. Uses
// os.CreateTemp for a unique tmp filename so concurrent goroutines from a
// multi-file download don't race on the same temp path.
func writeRefsMain(refsDir, commitHash string) error {
	if err := os.MkdirAll(refsDir, 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(refsDir, "main.*.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	if _, err := tmp.WriteString(commitHash); err != nil {
		_ = tmp.Close()
		_ = os.Remove(tmpPath)
		return err
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpPath)
		return err
	}
	return os.Rename(tmpPath, filepath.Join(refsDir, "main"))
}
