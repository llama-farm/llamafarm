package llamabinary

import (
	"archive/tar"
	"compress/gzip"
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// buildTarGz writes a minimal tar.gz containing the given files to dstPath.
func buildTarGz(t *testing.T, dstPath string, files map[string][]byte) {
	t.Helper()
	f, err := os.Create(dstPath)
	if err != nil {
		t.Fatalf("create tar: %v", err)
	}
	defer f.Close()
	gz := gzip.NewWriter(f)
	tw := tar.NewWriter(gz)
	for name, body := range files {
		hdr := &tar.Header{
			Name: name,
			Mode: 0o755,
			Size: int64(len(body)),
		}
		if err := tw.WriteHeader(hdr); err != nil {
			t.Fatalf("tar header %s: %v", name, err)
		}
		if _, err := tw.Write(body); err != nil {
			t.Fatalf("tar write %s: %v", name, err)
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatalf("tar close: %v", err)
	}
	if err := gz.Close(); err != nil {
		t.Fatalf("gz close: %v", err)
	}
}

// serveFile stands up an httptest.Server that serves the given file bytes at /.
func serveFile(t *testing.T, bodyPath string) *httptest.Server {
	t.Helper()
	body, err := os.ReadFile(bodyPath)
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/octet-stream")
		w.Write(body)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestDownload_FetchesExtractsAndCaches(t *testing.T) {
	tmp := t.TempDir()
	cacheRoot := filepath.Join(tmp, "cache")
	t.Setenv("LLAMAFARM_CACHE_DIR", cacheRoot)

	// Build a tar.gz containing libllama.so + a dep lib.
	archivePath := filepath.Join(tmp, "llama.tar.gz")
	buildTarGz(t, archivePath, map[string][]byte{
		"llama-test/libllama.so": []byte(strings.Repeat("L", 4096)),
		"llama-test/libggml.so":  []byte(strings.Repeat("G", 4096)),
	})

	srv := serveFile(t, archivePath)

	// Use a synthetic version so no other test interferes. Swap the spec
	// resolver via the override map.
	target := Target{OS: "linux", Arch: "arm64", Accelerator: "cpu"}
	testVersion := "bTEST001"

	// Install an override for SpecFor by wrapping it. We can't reassign the
	// package function directly, so instead we use the exported helper to
	// register a per-test override.
	SetTestSpecForOverride(func(tt Target, v string) (Spec, error) {
		if tt == target && v == testVersion {
			return Spec{
				URL:     srv.URL + "/llama.tar.gz",
				LibPath: "libllama.so",
				LibName: "libllama.so",
			}, nil
		}
		return Spec{}, ErrSpecNotAvailable
	})
	defer SetTestSpecForOverride(nil)

	res, err := Download(context.Background(), target, testVersion)
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	if res.Cached {
		t.Error("expected fresh download, got Cached=true")
	}
	if _, err := os.Stat(res.LibPath); err != nil {
		t.Errorf("main lib missing: %v", err)
	}
	// Dependency should have been extracted alongside.
	depPath := filepath.Join(res.DestDir, "libggml.so")
	if _, err := os.Stat(depPath); err != nil {
		t.Errorf("dep lib missing: %v", err)
	}

	// A second call should short-circuit as cached.
	res2, err := Download(context.Background(), target, testVersion)
	if err != nil {
		t.Fatalf("second Download: %v", err)
	}
	if !res2.Cached {
		t.Error("expected Cached=true on second call")
	}
}

func TestDownload_NoPrebuilt(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("LLAMAFARM_CACHE_DIR", filepath.Join(tmp, "cache"))

	// A combination that SpecFor does not support.
	target := Target{OS: "darwin", Arch: "arm64", Accelerator: "cuda"}
	_, err := Download(context.Background(), target, "b7694")
	if err == nil {
		t.Error("expected error for unsupported combo")
	}
}
