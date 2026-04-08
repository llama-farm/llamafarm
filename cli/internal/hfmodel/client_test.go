package hfmodel

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// newTestClient builds a Client wired to a fake HF endpoint with no token
// discovery (uses an isolated home dir to suppress real env tokens).
func newTestClient(t *testing.T, srv *httptest.Server) *Client {
	t.Helper()
	withFakeHome(t)
	c, err := NewClient(WithEndpoint(srv.URL), WithToken(""))
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestClient_ListRepoTree_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasPrefix(r.URL.Path, "/api/models/org/repo/tree/") {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`[
			{"type":"file","path":"config.json","size":1024,"oid":"abc"},
			{"type":"file","path":"model.Q4_K_M.gguf","size":4000000,"oid":"def"},
			{"type":"directory","path":"subdir","oid":"ghi"}
		]`))
	}))
	defer srv.Close()
	c := newTestClient(t, srv)

	tree, err := c.ListRepoTree(context.Background(), "org/repo", "main")
	if err != nil {
		t.Fatal(err)
	}
	if len(tree) != 3 {
		t.Fatalf("got %d entries, want 3", len(tree))
	}
	if tree[1].Path != "model.Q4_K_M.gguf" || tree[1].Size != 4000000 {
		t.Errorf("file entry wrong: %+v", tree[1])
	}
}

func TestClient_ListRepoTree_Pagination(t *testing.T) {
	pageCount := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		pageCount++
		if pageCount == 1 {
			w.Header().Set("Link", `<`+absURL(r, "/api/models/org/repo/tree/main?cursor=2")+`>; rel="next"`)
			_, _ = w.Write([]byte(`[{"type":"file","path":"a","size":1,"oid":"x"}]`))
			return
		}
		_, _ = w.Write([]byte(`[{"type":"file","path":"b","size":2,"oid":"y"}]`))
	}))
	defer srv.Close()
	c := newTestClient(t, srv)

	tree, err := c.ListRepoTree(context.Background(), "org/repo", "main")
	if err != nil {
		t.Fatal(err)
	}
	if len(tree) != 2 {
		t.Errorf("got %d entries, want 2 (paginated)", len(tree))
	}
	if pageCount != 2 {
		t.Errorf("expected 2 server hits, got %d", pageCount)
	}
}

func absURL(r *http.Request, path string) string {
	return "http://" + r.Host + path
}

func TestClient_ListRepoTree_401(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer srv.Close()
	c := newTestClient(t, srv)

	_, err := c.ListRepoTree(context.Background(), "org/repo", "main")
	if !errors.Is(err, ErrUnauthorized) {
		t.Errorf("got %v, want ErrUnauthorized", err)
	}
	if !strings.Contains(err.Error(), "huggingface-cli login") &&
		!strings.Contains(err.Error(), "HF_TOKEN") {
		t.Errorf("error message missing remediation: %v", err)
	}
}

func TestClient_ListRepoTree_403_Gated(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Error-Code", "GatedRepo")
		w.WriteHeader(http.StatusForbidden)
		_, _ = w.Write([]byte(`{"error":"This repo is gated"}`))
	}))
	defer srv.Close()
	c := newTestClient(t, srv)

	_, err := c.ListRepoTree(context.Background(), "meta-llama/Llama-2-7b", "main")
	var gated *GatedError
	if !errors.As(err, &gated) {
		t.Fatalf("got %v, want *GatedError", err)
	}
	if !strings.Contains(err.Error(), "huggingface.co/meta-llama/Llama-2-7b") {
		t.Errorf("gated error missing accept-terms URL: %v", err)
	}
}

func TestClient_ListRepoTree_403_NotGated(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusForbidden)
	}))
	defer srv.Close()
	c := newTestClient(t, srv)

	_, err := c.ListRepoTree(context.Background(), "org/repo", "main")
	if !errors.Is(err, ErrForbidden) {
		t.Errorf("got %v, want ErrForbidden", err)
	}
}

func TestClient_ListRepoTree_404(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()
	c := newTestClient(t, srv)

	_, err := c.ListRepoTree(context.Background(), "nobody/nope", "main")
	var nf *NotFoundError
	if !errors.As(err, &nf) {
		t.Fatalf("got %v, want *NotFoundError", err)
	}
	if !strings.Contains(err.Error(), "nobody/nope") {
		t.Errorf("not-found error missing model id: %v", err)
	}
}

func TestClient_GetFileMetadata_BasicFile(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("ETag", `"deadbeef"`)
		w.Header().Set("X-Repo-Commit", "abc123")
		w.Header().Set("Content-Length", "1024")
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()
	c := newTestClient(t, srv)

	md, err := c.GetFileMetadata(context.Background(), "org/repo", "main", "config.json")
	if err != nil {
		t.Fatal(err)
	}
	if md.ETag != "deadbeef" {
		t.Errorf("etag: got %q want deadbeef", md.ETag)
	}
	if md.CommitHash != "abc123" {
		t.Errorf("commit: got %q want abc123", md.CommitHash)
	}
	if md.Size != 1024 {
		t.Errorf("size: got %d want 1024", md.Size)
	}
}

func TestClient_GetFileMetadata_LFSFile(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("ETag", `"redirect-etag"`)
		w.Header().Set("X-Linked-Etag", `"sha256-of-actual-blob"`)
		w.Header().Set("X-Linked-Size", "5000000000")
		w.Header().Set("X-Repo-Commit", "commit42")
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()
	c := newTestClient(t, srv)

	md, err := c.GetFileMetadata(context.Background(), "org/repo", "main", "model.gguf")
	if err != nil {
		t.Fatal(err)
	}
	// LFS files: linked etag wins.
	if md.ETag != "sha256-of-actual-blob" {
		t.Errorf("etag: got %q want linked", md.ETag)
	}
	if md.Size != 5_000_000_000 {
		t.Errorf("size: got %d want 5e9", md.Size)
	}
}

func TestClient_OfflineMode_RefusesNetwork(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("server should not be called in offline mode")
	}))
	defer srv.Close()
	c := newTestClient(t, srv)
	t.Setenv("LLAMAFARM_OFFLINE", "1")

	_, err := c.ListRepoTree(context.Background(), "org/repo", "main")
	var oe *OfflineError
	if !errors.As(err, &oe) {
		t.Errorf("got %v, want *OfflineError", err)
	}
	_, err = c.GetFileMetadata(context.Background(), "org/repo", "main", "f")
	if !errors.As(err, &oe) {
		t.Errorf("metadata: got %v, want *OfflineError", err)
	}
}

func TestNewClient_WithTokenEmptyForcesAnonymous(t *testing.T) {
	// Set a real token in the env so we can verify WithToken("") suppresses
	// it. Without the explicit-anonymous flag, NewClient would pick this up
	// via DiscoverToken().
	withFakeHome(t)
	t.Setenv("HF_TOKEN", "should-not-be-picked-up")

	c, err := NewClient(WithToken(""))
	if err != nil {
		t.Fatal(err)
	}
	if c.token != "" {
		t.Errorf("WithToken(\"\") did not suppress discovery: got %q", c.token)
	}
}

func TestNewClient_DefaultDiscoversFromEnv(t *testing.T) {
	withFakeHome(t)
	t.Setenv("HF_TOKEN", "discovered-token")

	c, err := NewClient()
	if err != nil {
		t.Fatal(err)
	}
	if c.token != "discovered-token" {
		t.Errorf("default did not discover env token: got %q", c.token)
	}
}

func TestNewClient_WithTokenExplicitOverridesEnv(t *testing.T) {
	withFakeHome(t)
	t.Setenv("HF_TOKEN", "env-token")

	c, err := NewClient(WithToken("override"))
	if err != nil {
		t.Fatal(err)
	}
	if c.token != "override" {
		t.Errorf("explicit token did not win: got %q", c.token)
	}
}

func TestIsOffline_TruthyValues(t *testing.T) {
	cases := []struct {
		val  string
		want bool
	}{
		{"1", true}, {"true", true}, {"TRUE", true}, {"yes", true}, {"YES", true},
		{"on", true}, {"On", true},
		{"0", false}, {"false", false}, {"", false}, {"nope", false},
	}
	for _, tc := range cases {
		t.Run(tc.val, func(t *testing.T) {
			t.Setenv("LLAMAFARM_OFFLINE", tc.val)
			if got := IsOffline(); got != tc.want {
				t.Errorf("IsOffline()=%v, want %v", got, tc.want)
			}
		})
	}
}
