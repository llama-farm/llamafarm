package hfcache

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// setupFakeCache builds a minimal HF cache layout in a tmp dir and points env
// vars to it. Returns the root path for direct inspection.
func setupFakeCache(t *testing.T) string {
	t.Helper()
	tmp := t.TempDir()
	cacheRoot := filepath.Join(tmp, "hub")
	sidecarRoot := filepath.Join(tmp, "sidecars")
	t.Setenv("HF_HUB_CACHE", cacheRoot)
	t.Setenv("HF_HOME", "") // ensure HF_HUB_CACHE wins
	t.Setenv("LLAMAFARM_SHA256_CACHE_DIR", sidecarRoot)
	return cacheRoot
}

// writeGGUF creates a fake GGUF file with proper magic bytes and given content
// under the specified snapshot directory.
func writeGGUF(t *testing.T, snapshotDir, filename, body string) string {
	t.Helper()
	if err := os.MkdirAll(snapshotDir, 0o755); err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(snapshotDir, filename)
	content := append([]byte("GGUF"), []byte(body)...)
	if err := os.WriteFile(p, content, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// layoutRepo creates `models--<org>--<name>/snapshots/<commit>/` under cacheRoot.
func layoutRepo(t *testing.T, cacheRoot, repoID, commit string) string {
	t.Helper()
	return filepath.Join(cacheRoot, "models--"+strings.ReplaceAll(repoID, "/", "--"), "snapshots", commit)
}

func TestCacheRoot_EnvOverrides(t *testing.T) {
	t.Setenv("HF_HUB_CACHE", "/custom/hub")
	t.Setenv("HF_HOME", "/ignored")
	got, err := CacheRoot()
	if err != nil {
		t.Fatal(err)
	}
	if got != "/custom/hub" {
		t.Errorf("got %q", got)
	}
}

func TestCacheRoot_FallbackToHFHome(t *testing.T) {
	t.Setenv("HF_HUB_CACHE", "")
	t.Setenv("HF_HOME", "/hfhome")
	got, err := CacheRoot()
	if err != nil {
		t.Fatal(err)
	}
	if got != filepath.Join("/hfhome", "hub") {
		t.Errorf("got %q", got)
	}
}

func TestValidateRepoID(t *testing.T) {
	good := []string{"unsloth/Qwen3-1.7B-GGUF", "bartowski/smollm-135m-GGUF", "model", "org_with_underscore/name.with.dot"}
	for _, g := range good {
		if err := ValidateRepoID(g); err != nil {
			t.Errorf("expected %q to validate: %v", g, err)
		}
	}
	bad := []string{"../../etc/passwd", "/absolute", `\windows`, "has..dotdot", "two/slashes/here", "bad char!"}
	for _, b := range bad {
		if err := ValidateRepoID(b); err == nil {
			t.Errorf("expected %q to fail", b)
		}
	}
}

func TestListCachedFiles_MissingRepoReturnsNotCached(t *testing.T) {
	setupFakeCache(t)
	_, err := ListCachedFiles("nobody/nope")
	if !errors.Is(err, ErrNotCached) {
		t.Errorf("got %v, want ErrNotCached", err)
	}
}

func TestListCachedFiles_ReturnsSnapshotFiles(t *testing.T) {
	cacheRoot := setupFakeCache(t)
	snapDir := layoutRepo(t, cacheRoot, "unsloth/Qwen3-1.7B-GGUF", "commit123")
	writeGGUF(t, snapDir, "qwen3-1.7b.Q4_K_M.gguf", "weights")
	writeGGUF(t, snapDir, "qwen3-1.7b.Q8_0.gguf", "weights")

	files, err := ListCachedFiles("unsloth/Qwen3-1.7B-GGUF")
	if err != nil {
		t.Fatal(err)
	}
	if len(files) != 2 {
		t.Errorf("got %d files, want 2", len(files))
	}
	for _, f := range files {
		if !strings.Contains(f.SnapshotPath, "snapshots/commit123") {
			t.Errorf("snapshot path should contain snapshots/commit123: %s", f.SnapshotPath)
		}
		if f.Size == 0 {
			t.Error("expected non-zero size")
		}
	}
}

func TestListCachedFiles_NewestSnapshotWins(t *testing.T) {
	cacheRoot := setupFakeCache(t)
	oldSnap := layoutRepo(t, cacheRoot, "unsloth/Qwen3", "aaa")
	newSnap := layoutRepo(t, cacheRoot, "unsloth/Qwen3", "zzz")
	writeGGUF(t, oldSnap, "qwen3-1.7b.Q4_K_M.gguf", "old")
	writeGGUF(t, newSnap, "qwen3-1.7b.Q4_K_M.gguf", "new-content-bigger")

	files, err := ListCachedFiles("unsloth/Qwen3")
	if err != nil {
		t.Fatal(err)
	}
	if len(files) != 1 {
		t.Fatalf("got %d files, want 1 (dedup by name)", len(files))
	}
	if !strings.Contains(files[0].SnapshotPath, "snapshots/zzz") {
		t.Errorf("expected newest snapshot to win, got %s", files[0].SnapshotPath)
	}
}

func TestLocateGGUF_PreferredQuantWins(t *testing.T) {
	cacheRoot := setupFakeCache(t)
	snapDir := layoutRepo(t, cacheRoot, "unsloth/Qwen3", "c1")
	writeGGUF(t, snapDir, "qwen3-1.7b.Q4_K_M.gguf", "a")
	writeGGUF(t, snapDir, "qwen3-1.7b.Q8_0.gguf", "b")

	sf, err := LocateGGUF("unsloth/Qwen3", "Q8_0")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(sf.Filename, "Q8_0") {
		t.Errorf("expected Q8_0 winner, got %s", sf.Filename)
	}
}

func TestLocateGGUF_DefaultPreferenceOrder(t *testing.T) {
	cacheRoot := setupFakeCache(t)
	snapDir := layoutRepo(t, cacheRoot, "unsloth/Qwen3", "c1")
	writeGGUF(t, snapDir, "qwen3-1.7b.Q8_0.gguf", "a")
	writeGGUF(t, snapDir, "qwen3-1.7b.Q5_K_M.gguf", "b")
	writeGGUF(t, snapDir, "qwen3-1.7b.Q4_K_M.gguf", "c")

	sf, err := LocateGGUF("unsloth/Qwen3", "")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(sf.Filename, "Q4_K_M") {
		t.Errorf("expected Q4_K_M as default preference winner, got %s", sf.Filename)
	}
}

func TestLocateGGUF_SkipsMmproj(t *testing.T) {
	cacheRoot := setupFakeCache(t)
	snapDir := layoutRepo(t, cacheRoot, "ggml-org/Qwen2.5-Omni-GGUF", "c1")
	writeGGUF(t, snapDir, "qwen.Q4_K_M.gguf", "a")
	writeGGUF(t, snapDir, "mmproj-qwen-f16.gguf", "b")

	sf, err := LocateGGUF("ggml-org/Qwen2.5-Omni-GGUF", "")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(sf.Filename, "Q4_K_M") {
		t.Errorf("expected weights file, got %s", sf.Filename)
	}
}

func TestLocateGGUF_NotCached(t *testing.T) {
	setupFakeCache(t)
	_, err := LocateGGUF("nobody/nope", "")
	if !errors.Is(err, ErrNotCached) {
		t.Errorf("got %v, want ErrNotCached", err)
	}
}

func TestLocateMmproj_NoneReturnsZero(t *testing.T) {
	cacheRoot := setupFakeCache(t)
	snapDir := layoutRepo(t, cacheRoot, "unsloth/Qwen3", "c1")
	writeGGUF(t, snapDir, "qwen3-1.7b.Q4_K_M.gguf", "a")

	sf, err := LocateMmproj("unsloth/Qwen3")
	if err != nil {
		t.Fatal(err)
	}
	if sf.Filename != "" {
		t.Errorf("expected zero value, got %s", sf.Filename)
	}
}

func TestLocateMmproj_PrefersF16(t *testing.T) {
	cacheRoot := setupFakeCache(t)
	snapDir := layoutRepo(t, cacheRoot, "ggml-org/Qwen2.5-Omni", "c1")
	writeGGUF(t, snapDir, "mmproj-qwen-f32.gguf", "a")
	writeGGUF(t, snapDir, "mmproj-qwen-f16.gguf", "b")

	sf, err := LocateMmproj("ggml-org/Qwen2.5-Omni")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(sf.Filename, "f16") {
		t.Errorf("expected f16 preference, got %s", sf.Filename)
	}
}

func TestLocateMmproj_MissingRepoReturnsNotCached(t *testing.T) {
	setupFakeCache(t)
	_, err := LocateMmproj("nobody/nope")
	if !errors.Is(err, ErrNotCached) {
		t.Errorf("got %v, want ErrNotCached", err)
	}
}

func TestSHA256_ComputeAndCache(t *testing.T) {
	cacheRoot := setupFakeCache(t)
	snapDir := layoutRepo(t, cacheRoot, "unsloth/Qwen3", "c1")
	writeGGUF(t, snapDir, "qwen3.Q4_K_M.gguf", "deterministic-content")

	sf, err := LocateGGUF("unsloth/Qwen3", "")
	if err != nil {
		t.Fatal(err)
	}

	h1, err := SHA256(sf)
	if err != nil {
		t.Fatal(err)
	}
	if len(h1) != 64 {
		t.Errorf("expected 64-char hex digest, got %d", len(h1))
	}

	// Second call should return the cached value without re-hashing; we can
	// verify by checking the sidecar exists.
	sidecars, err := os.ReadDir(os.Getenv("LLAMAFARM_SHA256_CACHE_DIR"))
	if err != nil {
		t.Fatal(err)
	}
	if len(sidecars) != 1 {
		t.Errorf("expected exactly 1 sidecar, got %d", len(sidecars))
	}

	h2, err := SHA256(sf)
	if err != nil {
		t.Fatal(err)
	}
	if h1 != h2 {
		t.Errorf("cached sha256 mismatch: %q vs %q", h1, h2)
	}
}

func TestSHA256_InvalidatesOnChange(t *testing.T) {
	cacheRoot := setupFakeCache(t)
	snapDir := layoutRepo(t, cacheRoot, "unsloth/Qwen3", "c1")
	path := writeGGUF(t, snapDir, "qwen3.Q4_K_M.gguf", "original")

	sf, err := LocateGGUF("unsloth/Qwen3", "")
	if err != nil {
		t.Fatal(err)
	}
	h1, err := SHA256(sf)
	if err != nil {
		t.Fatal(err)
	}

	// Mutate the file and bump its mtime.
	if err := os.WriteFile(path, []byte("GGUFmodified-different-length"), 0o644); err != nil {
		t.Fatal(err)
	}

	// Re-stat to pick up new size.
	sf2, err := LocateGGUF("unsloth/Qwen3", "")
	if err != nil {
		t.Fatal(err)
	}
	h2, err := SHA256(sf2)
	if err != nil {
		t.Fatal(err)
	}
	if h1 == h2 {
		t.Error("expected sha256 to change after file mutation")
	}
}
