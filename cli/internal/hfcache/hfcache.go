// Package hfcache provides read-only access to the local HuggingFace Hub cache
// for locating GGUF model files without invoking Python. It mirrors the minimum
// subset of common/llamafarm_common/model_utils.py that the `lf models path`
// command needs.
//
// The package deliberately emits snapshot-paths (stable, human-readable paths
// like `.../snapshots/<commit>/file.gguf`) rather than resolved blob-paths, so
// that downstream tooling (ansible, packer) sees meaningful filenames.
package hfcache

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/llamafarm/cli/internal/modelformat"
)

// ErrNotCached is returned when a requested model is not present in the local
// HuggingFace cache.
var ErrNotCached = errors.New("model not cached")

// ErrInvalidRepoID is returned when a repo_id fails sanitization checks.
var ErrInvalidRepoID = errors.New("invalid repo id")

// SnapshotFile describes a single file inside a HuggingFace cache snapshot.
type SnapshotFile struct {
	// RepoID is the HuggingFace repository identifier ("org/name").
	RepoID string
	// SnapshotPath is the absolute snapshot path:
	//   <cache>/models--<org>--<name>/snapshots/<commit>/<filename>
	// This path preserves the original filename and is stable across HF cache
	// garbage-collection of blob files.
	SnapshotPath string
	// Filename is the terminal filename (no directory parts).
	Filename string
	// Size is the file size in bytes, resolved through any symlink chain.
	Size int64
}

// CacheRoot returns the root of the local HuggingFace Hub cache. Honors the
// standard HF_HUB_CACHE env var, then HF_HOME, then falls back to
// `~/.cache/huggingface/hub`.
func CacheRoot() (string, error) {
	if v := os.Getenv("HF_HUB_CACHE"); v != "" {
		return v, nil
	}
	if v := os.Getenv("HF_HOME"); v != "" {
		return filepath.Join(v, "hub"), nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("home dir: %w", err)
	}
	return filepath.Join(home, ".cache", "huggingface", "hub"), nil
}

// repoIDPattern matches a sanitized HuggingFace repo id: "org/name" or "name".
// Allows alphanumerics, hyphens, underscores, and periods.
var repoIDPattern = regexp.MustCompile(`^[a-zA-Z0-9_.\-]+(/[a-zA-Z0-9_.\-]+)?$`)

// ValidateRepoID returns nil if the repo_id looks safe to use in a filesystem
// path, mirroring the sanitization in model_utils.py._validate_model_id.
func ValidateRepoID(repoID string) error {
	if strings.Contains(repoID, "..") ||
		strings.HasPrefix(repoID, "/") ||
		strings.HasPrefix(repoID, `\`) {
		return fmt.Errorf("%w: path traversal not allowed: %s", ErrInvalidRepoID, repoID)
	}
	if !repoIDPattern.MatchString(repoID) {
		return fmt.Errorf("%w: format %s", ErrInvalidRepoID, repoID)
	}
	return nil
}

// repoCacheDir returns the cache subdirectory for a repo_id, or an error if
// the id is malformed.
func repoCacheDir(root, repoID string) (string, error) {
	if err := ValidateRepoID(repoID); err != nil {
		return "", err
	}
	// HF stores models in `models--<org>--<name>` directories.
	sanitized := strings.ReplaceAll(repoID, "/", "--")
	return filepath.Join(root, "models--"+sanitized), nil
}

// decodeRepoFolderName converts a `models--<...>` cache directory name back
// into a repo id. Decoding is intentionally tolerant: the encoding
// (`repo_id.replace("/", "--")`) is not strictly invertible when an org or
// repo name legitimately contains `--`, so we use a robust heuristic:
//
//   - Strip the `models--` prefix.
//   - If no `--` remains, the repo has no org component (rare but valid):
//     return the suffix unchanged.
//   - Otherwise split on the FIRST `--`. The first segment becomes the org,
//     everything after is the repo name (preserving any literal `--`).
//
// Returns the decoded repo id and a `decoded` flag indicating whether the
// result is a clean round-trip. When `decoded` is false the caller may
// choose to fall back to using the raw folder name as the repo id rather
// than dropping the entry.
func decodeRepoFolderName(folder string) (repoID string, decoded bool) {
	suffix := strings.TrimPrefix(folder, "models--")
	if suffix == folder {
		// Not a model folder.
		return "", false
	}
	if !strings.Contains(suffix, "--") {
		// No org component — repo id is the suffix as-is.
		return suffix, ValidateRepoID(suffix) == nil
	}
	// Split on the FIRST `--` so a repo whose name contains literal `--`
	// (encoded as `----`) round-trips correctly. This is a heuristic — a
	// repo whose ORG contains `--` cannot be perfectly decoded, but in
	// practice HF org names use single hyphens, so split-first is right
	// for the overwhelming majority of cases.
	idx := strings.Index(suffix, "--")
	candidate := suffix[:idx] + "/" + suffix[idx+2:]
	return candidate, ValidateRepoID(candidate) == nil
}

// ListCachedFiles returns every non-empty file found under the repo's
// snapshots directory, one entry per filename. If multiple snapshots contain
// the same filename, the entry from the newest snapshot (lexicographically
// last commit hash) is returned.
func ListCachedFiles(repoID string) ([]SnapshotFile, error) {
	root, err := CacheRoot()
	if err != nil {
		return nil, err
	}
	repoDir, err := repoCacheDir(root, repoID)
	if err != nil {
		return nil, err
	}
	snapshotsDir := filepath.Join(repoDir, "snapshots")
	if _, err := os.Stat(snapshotsDir); err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("%w: %s", ErrNotCached, repoID)
		}
		return nil, err
	}

	entries, err := os.ReadDir(snapshotsDir)
	if err != nil {
		return nil, err
	}

	// Process snapshots in sorted order so the last one wins.
	snapshotDirs := make([]string, 0, len(entries))
	for _, e := range entries {
		if e.IsDir() {
			snapshotDirs = append(snapshotDirs, e.Name())
		}
	}
	sort.Strings(snapshotDirs)

	out := make(map[string]SnapshotFile)
	for _, snap := range snapshotDirs {
		snapDir := filepath.Join(snapshotsDir, snap)
		files, err := os.ReadDir(snapDir)
		if err != nil {
			continue
		}
		for _, f := range files {
			if f.IsDir() {
				continue
			}
			name := f.Name()
			fullPath := filepath.Join(snapDir, name)
			// Resolve through any symlink chain to verify non-empty.
			info, err := os.Stat(fullPath)
			if err != nil {
				continue
			}
			if info.IsDir() || info.Size() == 0 {
				continue
			}
			out[name] = SnapshotFile{
				RepoID:       repoID,
				SnapshotPath: fullPath,
				Filename:     name,
				Size:         info.Size(),
			}
		}
	}

	result := make([]SnapshotFile, 0, len(out))
	for _, v := range out {
		result = append(result, v)
	}
	sort.Slice(result, func(i, j int) bool {
		return result[i].Filename < result[j].Filename
	})
	return result, nil
}

// QuantizationPreferenceOrder mirrors GGUF_QUANTIZATION_PREFERENCE_ORDER from
// common/llamafarm_common/model_utils.py.
var QuantizationPreferenceOrder = []string{
	"Q4_K_M", "Q4_K", "Q5_K_M", "Q5_K",
	"Q8_0", "Q6_K",
	"Q4_K_S", "Q5_K_S", "Q3_K_M", "Q2_K",
	"F16",
}

// isSplitGGUF matches files like "model-00001-of-00003.gguf".
var splitGGUFRegex = regexp.MustCompile(`(?i)-\d{5}-of-\d{5}[.-]`)

// LocateGGUF returns the main GGUF weights file for a repo, applying the
// standard quantization preference logic. The preferredQuant argument, if
// non-empty, is tried first. Returns ErrNotCached if nothing matches.
func LocateGGUF(repoID, preferredQuant string) (SnapshotFile, error) {
	files, err := ListCachedFiles(repoID)
	if err != nil {
		return SnapshotFile{}, err
	}
	// Filter to GGUF files that are NOT mmproj.
	ggufs := make([]SnapshotFile, 0, len(files))
	for _, f := range files {
		if !strings.HasSuffix(strings.ToLower(f.Filename), ".gguf") {
			continue
		}
		if modelformat.IsMmprojName(f.Filename) {
			continue
		}
		ggufs = append(ggufs, f)
	}
	if len(ggufs) == 0 {
		return SnapshotFile{}, fmt.Errorf("%w: no GGUF weights in %s", ErrNotCached, repoID)
	}

	// Prefer non-split files when possible.
	nonSplit := make([]SnapshotFile, 0, len(ggufs))
	for _, f := range ggufs {
		if !splitGGUFRegex.MatchString(f.Filename) {
			nonSplit = append(nonSplit, f)
		}
	}
	working := nonSplit
	if len(working) == 0 {
		working = ggufs
	}

	// If there's only one candidate, return it.
	if len(working) == 1 {
		return working[0], nil
	}

	// If the caller named a quant, honor it first.
	if preferredQuant != "" {
		wantUpper := strings.ToUpper(preferredQuant)
		for _, f := range working {
			if strings.ToUpper(modelformat.ParseQuantization(f.Filename)) == wantUpper {
				return f, nil
			}
		}
	}

	// Otherwise walk the preference order.
	for _, pref := range QuantizationPreferenceOrder {
		for _, f := range working {
			if strings.ToUpper(modelformat.ParseQuantization(f.Filename)) == pref {
				return f, nil
			}
		}
	}

	// Fall back to the first file.
	return working[0], nil
}

// LocateMmproj returns the multimodal projector file for a repo if one is
// cached. Returns (zero, nil) if no mmproj is present (not cached ≠ error),
// and (zero, ErrNotCached) only if the repo itself is missing.
func LocateMmproj(repoID string) (SnapshotFile, error) {
	files, err := ListCachedFiles(repoID)
	if err != nil {
		return SnapshotFile{}, err
	}
	candidates := make([]SnapshotFile, 0)
	for _, f := range files {
		if modelformat.IsMmprojName(f.Filename) {
			candidates = append(candidates, f)
		}
	}
	if len(candidates) == 0 {
		return SnapshotFile{}, nil
	}
	// Prefer f16, then bf16, then f32.
	for _, precision := range []string{"f16", "bf16", "f32"} {
		for _, f := range candidates {
			if strings.EqualFold(modelformat.ParseMmprojPrecision(f.Filename), precision) {
				return f, nil
			}
		}
	}
	return candidates[0], nil
}

// RepoInfo describes a cached HuggingFace repo at a high level.
type RepoInfo struct {
	// RepoID is the canonical "org/name" identifier.
	RepoID string
	// RepoPath is the absolute path of the repo directory inside the cache
	// (`<cache>/models--<org>--<name>`).
	RepoPath string
	// SizeOnDisk is the sum of unique blob sizes referenced by all snapshots
	// of this repo. Mirrors `huggingface_hub.scan_cache_dir`'s definition:
	// shared blobs are counted once.
	SizeOnDisk int64
	// FileCount is the number of distinct files (by basename) found across
	// all snapshots.
	FileCount int
}

// LookupRepo returns information about a cached repo, or ErrNotCached when the
// repo is not present. Used by `lf models status` and similar read-only checks
// that need to know "is this model on disk?" without booting the server.
func LookupRepo(repoID string) (*RepoInfo, error) {
	root, err := CacheRoot()
	if err != nil {
		return nil, err
	}
	repoDir, err := repoCacheDir(root, repoID)
	if err != nil {
		return nil, err
	}
	if _, err := os.Stat(repoDir); err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("%w: %s", ErrNotCached, repoID)
		}
		return nil, err
	}

	size, files, err := repoSizeAndFiles(repoDir)
	if err != nil {
		return nil, err
	}
	if files == 0 {
		// Repo dir exists but has no real files (e.g. partial init / broken
		// symlinks). Treat as not-cached so callers don't get a misleading
		// "yes, present" answer for an unusable cache entry.
		return nil, fmt.Errorf("%w: %s", ErrNotCached, repoID)
	}

	return &RepoInfo{
		RepoID:     repoID,
		RepoPath:   repoDir,
		SizeOnDisk: size,
		FileCount:  files,
	}, nil
}

// ScanCache walks the entire HuggingFace Hub cache root and returns one
// RepoInfo per cached model repo. Mirrors `huggingface_hub.scan_cache_dir`'s
// behavior for the `repo_type == "model"` case. Returns an empty slice (not an
// error) when the cache root does not exist yet.
func ScanCache() ([]RepoInfo, error) {
	root, err := CacheRoot()
	if err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}

	var out []RepoInfo
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		name := e.Name()
		// Only model repos. HF also stores `datasets--*` and `spaces--*`.
		if !strings.HasPrefix(name, "models--") {
			continue
		}
		repoID, decoded := decodeRepoFolderName(name)
		if !decoded {
			// Decoding ambiguous (e.g. an org with literal `--` in its
			// name). Don't drop the entry — surface it with the raw
			// suffix so `lf models cached` stays complete. Validate the
			// raw suffix loosely: if it contains path traversal we
			// really do skip it (defensive).
			raw := strings.TrimPrefix(name, "models--")
			if strings.Contains(raw, "..") || raw == "" {
				continue
			}
			repoID = raw
		}
		repoDir := filepath.Join(root, name)
		size, files, err := repoSizeAndFiles(repoDir)
		if err != nil || files == 0 {
			continue
		}
		out = append(out, RepoInfo{
			RepoID:     repoID,
			RepoPath:   repoDir,
			SizeOnDisk: size,
			FileCount:  files,
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].RepoID < out[j].RepoID })
	return out, nil
}

// repoSizeAndFiles walks a single repo directory and returns its size on disk
// (sum of unique blob sizes) and the count of distinct snapshot files.
//
// Size is computed from the `blobs/` directory rather than from snapshot
// symlinks, because multiple snapshot files can point at the same blob and we
// want to count each blob exactly once — matching `scan_cache_dir`.
//
// Skips download-machinery artifacts:
//   - `<etag>.tmp` — partial download in progress (the CLI's resumable
//     downloader keeps these around between runs).
//   - `<etag>.lock` — file-lock sentinel; intentionally never removed
//     (see hfmodel/lock_unix.go for the reason).
//
// Counting these would inflate `SizeOnDisk` and confuse `lf models cached`.
func repoSizeAndFiles(repoDir string) (size int64, fileCount int, err error) {
	blobsDir := filepath.Join(repoDir, "blobs")
	if entries, err := os.ReadDir(blobsDir); err == nil {
		for _, e := range entries {
			if e.IsDir() {
				continue
			}
			name := e.Name()
			if strings.HasSuffix(name, ".tmp") || strings.HasSuffix(name, ".lock") {
				continue
			}
			info, statErr := e.Info()
			if statErr != nil {
				continue
			}
			size += info.Size()
		}
	}
	// File count: distinct file paths (relative to the snapshot root, so
	// nested files like `voices/a.bin` and `voices/b.bin` count separately)
	// across all snapshots. Recurses into subdirectories to match
	// huggingface_hub.scan_cache_dir, which walks the full snapshot tree.
	snapshotsDir := filepath.Join(repoDir, "snapshots")
	seen := make(map[string]struct{})
	if snapEntries, err := os.ReadDir(snapshotsDir); err == nil {
		for _, snap := range snapEntries {
			if !snap.IsDir() {
				continue
			}
			snapDir := filepath.Join(snapshotsDir, snap.Name())
			_ = filepath.WalkDir(snapDir, func(path string, d os.DirEntry, walkErr error) error {
				if walkErr != nil || d.IsDir() {
					return nil //nolint:nilerr // best-effort walk
				}
				// Use the path relative to the snapshot dir as the key, so
				// the same file in different snapshots counts once but
				// distinct files in subdirs count separately.
				rel, relErr := filepath.Rel(snapDir, path)
				if relErr != nil {
					return nil
				}
				if info, statErr := os.Stat(path); statErr == nil && !info.IsDir() && info.Size() > 0 {
					seen[rel] = struct{}{}
				}
				return nil
			})
		}
	}
	return size, len(seen), nil
}

// sha256Sidecar describes the on-disk record of a previously-computed sha256.
type sha256Sidecar struct {
	Path    string `json:"path"`     // realpath of the source file
	Size    int64  `json:"size"`     // file size at hashing time
	ModTime int64  `json:"mod_time"` // source file mtime (unix seconds)
	SHA256  string `json:"sha256"`   // hex digest
}

// sidecarDir returns the directory where sha256 sidecars are stored. This is
// CLI-owned rather than placed inside the HF cache to avoid polluting HF's
// namespace. Layout: $XDG_CACHE_HOME/llamafarm/sha256/<content-id>.json.
func sidecarDir() (string, error) {
	if v := os.Getenv("LLAMAFARM_SHA256_CACHE_DIR"); v != "" {
		return v, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	// Use the llamafarm CLI cache root, not the llama.cpp cache.
	return filepath.Join(home, ".llamafarm", "cache", "sha256"), nil
}

// contentID returns a filesystem-safe identifier for a sidecar file based on
// the source file's absolute realpath.
func contentID(realpath string) string {
	h := sha256.Sum256([]byte(realpath))
	return hex.EncodeToString(h[:])
}

// SHA256 returns the sha256 hex digest of the given snapshot file, caching the
// result in a sidecar file keyed on (realpath, size, mtime). Subsequent calls
// return the cached value without re-hashing unless the source file has
// changed.
func SHA256(f SnapshotFile) (string, error) {
	realpath, err := filepath.EvalSymlinks(f.SnapshotPath)
	if err != nil {
		return "", fmt.Errorf("resolve %s: %w", f.SnapshotPath, err)
	}
	info, err := os.Stat(realpath)
	if err != nil {
		return "", fmt.Errorf("stat %s: %w", realpath, err)
	}
	size := info.Size()
	mtime := info.ModTime().Unix()

	sidecarRoot, err := sidecarDir()
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(sidecarRoot, 0o755); err != nil {
		return "", err
	}
	sidecarPath := filepath.Join(sidecarRoot, contentID(realpath)+".json")

	// Try to read an existing sidecar.
	if data, err := os.ReadFile(sidecarPath); err == nil {
		var rec sha256Sidecar
		if err := json.Unmarshal(data, &rec); err == nil {
			if rec.Path == realpath && rec.Size == size && rec.ModTime == mtime && rec.SHA256 != "" {
				return rec.SHA256, nil
			}
		}
	}

	// Compute + persist.
	digest, err := hashFile(realpath)
	if err != nil {
		return "", err
	}
	rec := sha256Sidecar{
		Path:    realpath,
		Size:    size,
		ModTime: mtime,
		SHA256:  digest,
	}
	data, err := json.Marshal(rec)
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(sidecarPath, data, 0o644); err != nil {
		return "", fmt.Errorf("write sidecar: %w", err)
	}
	return digest, nil
}

func hashFile(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
