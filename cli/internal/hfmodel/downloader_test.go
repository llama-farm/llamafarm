package hfmodel

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
)

// withFakeCache redirects HF_HUB_CACHE to a tmp dir and returns the root.
func withFakeCache(t *testing.T) string {
	t.Helper()
	tmp := t.TempDir()
	t.Setenv("HF_HUB_CACHE", tmp)
	t.Setenv("HF_HOME", "")
	return tmp
}

// fakeBlobServer serves a fixed body and supports Range requests. Strictly
// requires `If-Range` to use the RFC 7232 quoted form (`"<etag>"`) — any
// other format is rejected as "etag does not match" and the server falls
// back to a full 200 transfer, exactly as a real HTTP server would.
func fakeBlobServer(t *testing.T, body []byte, etag string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("ETag", `"`+etag+`"`)
		w.Header().Set("X-Repo-Commit", "commit-abc")
		// HEAD: just return headers.
		if r.Method == http.MethodHead {
			w.Header().Set("Content-Length", strconv.Itoa(len(body)))
			w.WriteHeader(http.StatusOK)
			return
		}
		// GET (with optional Range)
		rng := r.Header.Get("Range")
		ifRange := r.Header.Get("If-Range")
		// Require the strict RFC quoted form. This is what catches the
		// "we sent the unquoted blob-name form" bug — without strict
		// matching here the server would silently accept the wrong
		// format and resume would only fail in production against real
		// HF Hub servers.
		expectedIfRange := `"` + etag + `"`
		if rng != "" && ifRange == expectedIfRange {
			// Parse "bytes=N-"
			var from int64
			_, err := fmt.Sscanf(rng, "bytes=%d-", &from)
			if err != nil || from < 0 || from >= int64(len(body)) {
				w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
				return
			}
			w.Header().Set("Content-Length", strconv.Itoa(len(body)-int(from)))
			w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", from, len(body)-1, len(body)))
			w.WriteHeader(http.StatusPartialContent)
			_, _ = w.Write(body[from:])
			return
		}
		// Full GET
		w.Header().Set("Content-Length", strconv.Itoa(len(body)))
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(body)
	}))
}

func TestDownloadFile_FreshDownload(t *testing.T) {
	cacheRoot := withFakeCache(t)
	body := []byte("GGUFhello-world-this-is-a-fake-blob-for-testing-purposes")
	srv := fakeBlobServer(t, body, "etag-1")
	defer srv.Close()
	c := newTestClient(t, srv)

	plan := SingleFilePlan{
		RepoID:   "org/repo",
		Filename: "model.gguf",
		Metadata: &FileMetadata{
			Filename:   "model.gguf",
			URL:        srv.URL + "/file",
			Size:       int64(len(body)),
			ETag:       "etag-1",
			CommitHash: "commit-abc",
		},
	}
	var events []ProgressEvent
	cb := func(ev ProgressEvent) { events = append(events, ev) }
	if err := c.DownloadFile(context.Background(), plan, cb); err != nil {
		t.Fatal(err)
	}

	// Verify blob written.
	blobPath := filepath.Join(cacheRoot, "models--org--repo", "blobs", "etag-1")
	got, err := os.ReadFile(blobPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(body) {
		t.Errorf("blob content mismatch")
	}

	// Verify snapshot symlink.
	snapPath := filepath.Join(cacheRoot, "models--org--repo", "snapshots", "commit-abc", "model.gguf")
	info, err := os.Lstat(snapPath)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode()&os.ModeSymlink == 0 {
		t.Error("expected snapshot to be a symlink on this OS")
	}

	// Verify refs/main.
	refData, err := os.ReadFile(filepath.Join(cacheRoot, "models--org--repo", "refs", "main"))
	if err != nil {
		t.Fatal(err)
	}
	if string(refData) != "commit-abc" {
		t.Errorf("refs/main = %q, want commit-abc", string(refData))
	}

	// Verify event sequence.
	hasEvent := func(name string) bool {
		for _, ev := range events {
			if ev.Event == name {
				return true
			}
		}
		return false
	}
	if !hasEvent("start") || !hasEvent("end") {
		t.Errorf("missing start/end events: %v", events)
	}
}

func TestDownloadFile_AlreadyCached(t *testing.T) {
	cacheRoot := withFakeCache(t)
	body := []byte("GGUFexisting")
	// Pre-populate the blob.
	blobDir := filepath.Join(cacheRoot, "models--org--repo", "blobs")
	if err := os.MkdirAll(blobDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(blobDir, "etag-cached"), body, 0o644); err != nil {
		t.Fatal(err)
	}

	called := int32(0)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&called, 1)
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()
	c := newTestClient(t, srv)

	plan := SingleFilePlan{
		RepoID:   "org/repo",
		Filename: "model.gguf",
		Metadata: &FileMetadata{
			Filename:   "model.gguf",
			URL:        srv.URL + "/file",
			Size:       int64(len(body)),
			ETag:       "etag-cached",
			CommitHash: "commit-cached",
		},
	}
	var events []ProgressEvent
	cb := func(ev ProgressEvent) { events = append(events, ev) }
	if err := c.DownloadFile(context.Background(), plan, cb); err != nil {
		t.Fatal(err)
	}
	if atomic.LoadInt32(&called) != 0 {
		t.Errorf("server should not have been called for cached blob")
	}
	found := false
	for _, ev := range events {
		if ev.Event == "cached" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected cached event, got %v", events)
	}
}

func TestDownloadFile_ResumeFromPartial(t *testing.T) {
	cacheRoot := withFakeCache(t)
	body := []byte(strings.Repeat("X", 4096) + strings.Repeat("Y", 4096))
	srv := fakeBlobServer(t, body, "etag-resume")
	defer srv.Close()
	c := newTestClient(t, srv)

	// Pre-write a partial: first 4096 bytes.
	blobDir := filepath.Join(cacheRoot, "models--org--repo", "blobs")
	if err := os.MkdirAll(blobDir, 0o755); err != nil {
		t.Fatal(err)
	}
	tmpPath := filepath.Join(blobDir, "etag-resume.tmp")
	if err := os.WriteFile(tmpPath, body[:4096], 0o644); err != nil {
		t.Fatal(err)
	}

	plan := SingleFilePlan{
		RepoID:   "org/repo",
		Filename: "model.gguf",
		Metadata: &FileMetadata{
			Filename:   "model.gguf",
			URL:        srv.URL + "/file",
			Size:       int64(len(body)),
			ETag:       "etag-resume",
			CommitHash: "commit-resume",
		},
	}
	if err := c.DownloadFile(context.Background(), plan, func(ProgressEvent) {}); err != nil {
		t.Fatal(err)
	}
	// Verify final blob content matches.
	got, err := os.ReadFile(filepath.Join(blobDir, "etag-resume"))
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(body) {
		t.Errorf("resumed blob content mismatch: got %d bytes", len(got))
	}
}

func TestDownloadFile_OfflineRefuses(t *testing.T) {
	withFakeCache(t)
	srv := fakeBlobServer(t, []byte("nope"), "e")
	defer srv.Close()
	c := newTestClient(t, srv)
	t.Setenv("LLAMAFARM_OFFLINE", "1")

	plan := SingleFilePlan{
		RepoID:   "org/repo",
		Filename: "f",
		Metadata: &FileMetadata{Filename: "f", URL: srv.URL, Size: 4, ETag: "e", CommitHash: "c"},
	}
	err := c.DownloadFile(context.Background(), plan, nil)
	var oe *OfflineError
	if !errors.As(err, &oe) {
		t.Errorf("got %v, want *OfflineError", err)
	}
}

func TestDownloadFile_SizeMismatch(t *testing.T) {
	withFakeCache(t)
	body := []byte("only-12-bytes")
	srv := fakeBlobServer(t, body, "etag-mismatch")
	defer srv.Close()
	c := newTestClient(t, srv)

	plan := SingleFilePlan{
		RepoID:   "org/repo",
		Filename: "f",
		Metadata: &FileMetadata{
			Filename:   "f",
			URL:        srv.URL + "/file",
			Size:       9999, // wrong on purpose
			ETag:       "etag-mismatch",
			CommitHash: "commit-mismatch",
		},
	}
	err := c.DownloadFile(context.Background(), plan, func(ProgressEvent) {})
	if err == nil || !strings.Contains(err.Error(), "size mismatch") {
		t.Errorf("expected size mismatch error, got %v", err)
	}
}

func TestSymlinksSupported_OnTmpDir(t *testing.T) {
	dir := t.TempDir()
	if !symlinksSupported(dir) {
		t.Skip("symlinks not supported on this filesystem; skipping")
	}
	// Second call should hit cache and still return true.
	if !symlinksSupported(dir) {
		t.Error("cached lookup returned false")
	}
}

func TestCreateSnapshotSymlink_FallbackCopy(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "blob")
	if err := os.WriteFile(src, []byte("hello"), 0o644); err != nil {
		t.Fatal(err)
	}
	dst := filepath.Join(dir, "snap")

	// Force fallback by poisoning the cache for this dir.
	symlinkSupportMu.Lock()
	symlinkSupportCache[filepath.Dir(dst)] = false
	symlinkSupportMu.Unlock()

	if err := CreateSnapshotSymlink(src, dst); err != nil {
		t.Fatal(err)
	}
	// Should be a copy now, not a symlink.
	info, err := os.Lstat(dst)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode()&os.ModeSymlink != 0 {
		t.Error("expected copy fallback, got symlink")
	}
	got, _ := os.ReadFile(dst)
	if string(got) != "hello" {
		t.Errorf("copy content wrong: %q", got)
	}

	// Reset cache so other tests aren't affected.
	symlinkSupportMu.Lock()
	delete(symlinkSupportCache, filepath.Dir(dst))
	symlinkSupportMu.Unlock()
}

func TestQuoteETagForHeader(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{"", ""},
		{"abc123", `"abc123"`},
		{"sha256-of-blob", `"sha256-of-blob"`},
		// Defensive: already-quoted should not be double-quoted.
		{`"abc123"`, `"abc123"`},
		{`W/"abc123"`, `W/"abc123"`},
	}
	for _, tc := range cases {
		t.Run(tc.in, func(t *testing.T) {
			if got := quoteETagForHeader(tc.in); got != tc.want {
				t.Errorf("quoteETagForHeader(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

func TestAcquireLock_LockFileSurvivesRelease(t *testing.T) {
	// Regression test for the inode-identity bug: if release() unlinked
	// the lock file, two waiters could end up flocking different inodes
	// and both succeed concurrently. The lock file MUST stay on disk
	// after release.
	dir := t.TempDir()
	lockPath := filepath.Join(dir, "blob.lock")

	rel, err := acquireLock(lockPath)
	if err != nil {
		t.Fatal(err)
	}
	rel()

	if _, err := os.Stat(lockPath); err != nil {
		t.Errorf("lock file was removed on release: %v", err)
	}
}

func TestAcquireLock_SerializesGoroutines(t *testing.T) {
	dir := t.TempDir()
	lockPath := filepath.Join(dir, "test.lock")

	// Acquire once.
	rel1, err := acquireLock(lockPath)
	if err != nil {
		t.Fatal(err)
	}
	// Try to acquire again from a goroutine — should block until rel1 fires.
	acquired := make(chan struct{})
	go func() {
		rel2, err := acquireLock(lockPath)
		if err != nil {
			t.Errorf("second acquire failed: %v", err)
			return
		}
		close(acquired)
		rel2()
	}()

	// Make sure the goroutine hasn't acquired yet.
	select {
	case <-acquired:
		t.Error("second acquire succeeded while first holder was active")
	default:
	}

	// Release the first holder; the goroutine should now proceed.
	rel1()
	<-acquired
}
