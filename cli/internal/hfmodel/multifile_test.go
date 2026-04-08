package hfmodel

import (
	"context"
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

// fakeRepoServer simulates a HuggingFace Hub for a multi-file repo. It serves
// the tree listing, HEAD/GET for each file, and tracks how many times each
// endpoint was hit so tests can assert behavior.
func fakeRepoServer(t *testing.T, files map[string][]byte) (*httptest.Server, *int32) {
	t.Helper()
	var hits int32
	mux := http.NewServeMux()

	// /api/models/<id> (model info) and /api/models/<id>/tree/<rev> (tree listing).
	mux.HandleFunc("/api/models/", func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&hits, 1)
		w.Header().Set("Content-Type", "application/json")
		if !strings.Contains(r.URL.Path, "/tree/") {
			// Model info endpoint — return the resolved commit sha.
			_, _ = w.Write([]byte(`{"sha":"commit-multi"}`))
			return
		}
		// Tree listing.
		var entries []TreeEntry
		for name, body := range files {
			entries = append(entries, TreeEntry{
				Type: "file",
				Path: name,
				Size: int64(len(body)),
				OID:  "oid-" + name,
			})
		}
		// Manual JSON encode to keep ordering deterministic enough for tests.
		var sb strings.Builder
		sb.WriteString("[")
		for i, e := range entries {
			if i > 0 {
				sb.WriteString(",")
			}
			fmt.Fprintf(&sb, `{"type":%q,"path":%q,"size":%d,"oid":%q}`, e.Type, e.Path, e.Size, e.OID)
		}
		sb.WriteString("]")
		_, _ = w.Write([]byte(sb.String()))
	})

	// File resolve (HEAD + GET).
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		// Strip "/org/repo/resolve/main/" prefix to get filename.
		parts := strings.SplitN(strings.TrimPrefix(r.URL.Path, "/"), "/", 5)
		if len(parts) < 5 {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		filename := parts[4]
		body, ok := files[filename]
		if !ok {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.Header().Set("ETag", `"etag-`+filename+`"`)
		w.Header().Set("X-Repo-Commit", "commit-multi")
		w.Header().Set("Content-Length", strconv.Itoa(len(body)))
		if r.Method == http.MethodHead {
			w.WriteHeader(http.StatusOK)
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(body)
	})

	return httptest.NewServer(mux), &hits
}

func TestGetModelDownloadPlan_GGUFRepo(t *testing.T) {
	withFakeCache(t)
	files := map[string][]byte{
		"model.Q4_K_M.gguf": []byte(strings.Repeat("a", 100)),
		"model.Q8_0.gguf":   []byte(strings.Repeat("b", 200)),
		"README.md":         []byte("hi"),
	}
	srv, _ := fakeRepoServer(t, files)
	defer srv.Close()
	c := newTestClient(t, srv)

	plan, err := c.GetModelDownloadPlan(context.Background(), "org/repo:Q8_0")
	if err != nil {
		t.Fatal(err)
	}
	if !plan.IsGGUF {
		t.Error("expected IsGGUF=true")
	}
	if plan.SelectedFile != "model.Q8_0.gguf" {
		t.Errorf("selected: got %q", plan.SelectedFile)
	}
	if plan.TotalSize != 200 {
		t.Errorf("size: got %d, want 200", plan.TotalSize)
	}
	if len(plan.Files) != 1 {
		t.Errorf("expected single file in plan for GGUF, got %d", len(plan.Files))
	}
}

func TestGetModelDownloadPlan_TransformerRepo(t *testing.T) {
	withFakeCache(t)
	files := map[string][]byte{
		"config.json":  []byte(`{"hello":"world"}`),
		"tokenizer.json": []byte("tok"),
		"model.safetensors": []byte("weights"),
	}
	srv, _ := fakeRepoServer(t, files)
	defer srv.Close()
	c := newTestClient(t, srv)

	plan, err := c.GetModelDownloadPlan(context.Background(), "org/repo")
	if err != nil {
		t.Fatal(err)
	}
	if plan.IsGGUF {
		t.Error("expected IsGGUF=false")
	}
	if len(plan.Files) != 3 {
		t.Errorf("expected 3 files, got %d: %v", len(plan.Files), plan.Files)
	}
	if plan.TotalSize == 0 {
		t.Error("expected non-zero TotalSize")
	}
}

func TestDownloadModel_TransformerEndToEnd(t *testing.T) {
	cacheRoot := withFakeCache(t)
	files := map[string][]byte{
		"config.json":       []byte(`{"hello":"world"}`),
		"tokenizer.json":    []byte("tokenizer-bytes"),
		"model.safetensors": []byte(strings.Repeat("X", 4096)),
	}
	srv, _ := fakeRepoServer(t, files)
	defer srv.Close()
	c := newTestClient(t, srv)

	plan, err := c.GetModelDownloadPlan(context.Background(), "org/repo")
	if err != nil {
		t.Fatal(err)
	}

	var events []ProgressEvent
	cb := func(ev ProgressEvent) { events = append(events, ev) }
	if err := c.DownloadModel(context.Background(), plan, cb); err != nil {
		t.Fatal(err)
	}

	// init + done at minimum.
	if events[0].Event != "init" {
		t.Errorf("first event = %q, want init", events[0].Event)
	}
	if events[len(events)-1].Event != "done" {
		t.Errorf("last event = %q, want done", events[len(events)-1].Event)
	}
	if events[0].FileCount != 3 {
		t.Errorf("init.FileCount = %d, want 3", events[0].FileCount)
	}

	// Each file's blob should exist on disk.
	for filename := range files {
		blob := filepath.Join(cacheRoot, "models--org--repo", "blobs", "etag-"+filename)
		if _, err := os.Stat(blob); err != nil {
			t.Errorf("blob missing for %s: %v", filename, err)
		}
		// And a snapshot symlink should resolve to the blob.
		snap := filepath.Join(cacheRoot, "models--org--repo", "snapshots", "commit-multi", filename)
		if _, err := os.Stat(snap); err != nil {
			t.Errorf("snapshot missing for %s: %v", filename, err)
		}
	}

	// refs/main should contain the commit hash.
	refs, err := os.ReadFile(filepath.Join(cacheRoot, "models--org--repo", "refs", "main"))
	if err != nil {
		t.Fatal(err)
	}
	if string(refs) != "commit-multi" {
		t.Errorf("refs/main = %q, want commit-multi", string(refs))
	}
}

func TestDownloadModel_OfflineRefuses(t *testing.T) {
	withFakeCache(t)
	srv, _ := fakeRepoServer(t, map[string][]byte{"f": []byte("x")})
	defer srv.Close()
	c := newTestClient(t, srv)
	t.Setenv("LLAMAFARM_OFFLINE", "1")

	_, err := c.GetModelDownloadPlan(context.Background(), "org/repo")
	if err == nil || !strings.Contains(err.Error(), "offline") {
		t.Errorf("expected offline error, got %v", err)
	}
}
