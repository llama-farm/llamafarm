package orchestrator

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestNewBinaryManager(t *testing.T) {
	t.Run("uses provided version", func(t *testing.T) {
		mgr, err := NewBinaryManager("v1.2.3")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if mgr.version != "v1.2.3" {
			t.Errorf("got version %q, want %q", mgr.version, "v1.2.3")
		}
	})

	t.Run("falls back to latest for empty version", func(t *testing.T) {
		mgr, err := NewBinaryManager("")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// Should fall back to latest if CLI version is also empty/dev
		if mgr.version != "latest" && mgr.version == "" {
			t.Errorf("expected non-empty version, got %q", mgr.version)
		}
	})

	t.Run("falls back to latest for dev version", func(t *testing.T) {
		mgr, err := NewBinaryManager("dev")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		// Should fall back to latest if CLI version is also dev
		if mgr.version != "latest" && mgr.version == "dev" {
			t.Errorf("expected non-dev version, got %q", mgr.version)
		}
	})
}

func TestBuildDownloadURL(t *testing.T) {
	tests := []struct {
		name     string
		version  string
		filename string
		want     string
	}{
		{
			name:     "latest version",
			version:  "latest",
			filename: "llamafarm-server-linux-x86_64",
			want:     "https://github.com/llama-farm/llamafarm/releases/latest/download/llamafarm-server-linux-x86_64",
		},
		{
			name:     "version with v prefix",
			version:  "v1.2.3",
			filename: "llamafarm-rag-macos-arm64",
			want:     "https://github.com/llama-farm/llamafarm/releases/download/v1.2.3/llamafarm-rag-macos-arm64",
		},
		{
			name:     "version without v prefix",
			version:  "0.0.26",
			filename: "llamafarm-runtime-windows-x86_64.exe",
			want:     "https://github.com/llama-farm/llamafarm/releases/download/v0.0.26/llamafarm-runtime-windows-x86_64.exe",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mgr := &BinaryManager{version: tt.version}
			got := mgr.buildDownloadURL(tt.filename)
			if got != tt.want {
				t.Errorf("buildDownloadURL() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestVersionFileOperations(t *testing.T) {
	origBinDir, origBinDirSet := os.LookupEnv("LF_BIN_DIR")
	defer func() {
		if origBinDirSet {
			os.Setenv("LF_BIN_DIR", origBinDir)
		} else {
			os.Unsetenv("LF_BIN_DIR")
		}
	}()

	tmpDir := t.TempDir()
	os.Setenv("LF_BIN_DIR", tmpDir)

	t.Run("write and read version file", func(t *testing.T) {
		mgr := &BinaryManager{version: "v1.2.3"}

		// Write version file
		if err := mgr.writeInstalledBinariesVersion(); err != nil {
			t.Fatalf("failed to write version: %v", err)
		}

		// Verify file exists
		versionFile := filepath.Join(tmpDir, ".pyapp-version")
		if _, err := os.Stat(versionFile); err != nil {
			t.Fatalf("version file not created: %v", err)
		}

		// Read version file
		version, err := mgr.getInstalledBinariesVersion()
		if err != nil {
			t.Fatalf("failed to read version: %v", err)
		}

		if version != "v1.2.3" {
			t.Errorf("got version %q, want %q", version, "v1.2.3")
		}
	})

	t.Run("read returns error when file missing", func(t *testing.T) {
		// Use a different temp directory with no version file
		tmpDir2 := t.TempDir()
		os.Setenv("LF_BIN_DIR", tmpDir2)

		mgr := &BinaryManager{version: "v1.2.3"}
		_, err := mgr.getInstalledBinariesVersion()
		if err == nil {
			t.Error("expected error when version file is missing")
		}
	})

	t.Run("handles version with whitespace", func(t *testing.T) {
		tmpDir3 := t.TempDir()
		os.Setenv("LF_BIN_DIR", tmpDir3)

		versionFile := filepath.Join(tmpDir3, ".pyapp-version")
		if err := os.WriteFile(versionFile, []byte("  v1.2.3  \n"), 0644); err != nil {
			t.Fatal(err)
		}

		mgr := &BinaryManager{}
		version, err := mgr.getInstalledBinariesVersion()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		if version != "v1.2.3" {
			t.Errorf("got version %q, want %q (whitespace should be trimmed)", version, "v1.2.3")
		}
	})
}

func TestAreBinariesInstalled(t *testing.T) {
	origBinDir, origBinDirSet := os.LookupEnv("LF_BIN_DIR")
	defer func() {
		if origBinDirSet {
			os.Setenv("LF_BIN_DIR", origBinDir)
		} else {
			os.Unsetenv("LF_BIN_DIR")
		}
	}()

	t.Run("returns false when binaries missing", func(t *testing.T) {
		tmpDir := t.TempDir()
		os.Setenv("LF_BIN_DIR", tmpDir)

		mgr := &BinaryManager{version: "v1.2.3"}
		if mgr.areBinariesInstalled() {
			t.Error("expected false when binaries are missing")
		}
	})

	t.Run("returns false when version file missing", func(t *testing.T) {
		tmpDir := t.TempDir()
		os.Setenv("LF_BIN_DIR", tmpDir)

		// Create fake binaries but no version file
		createFakeBinaries(t, tmpDir)

		mgr := &BinaryManager{version: "v1.2.3"}
		if mgr.areBinariesInstalled() {
			t.Error("expected false when version file is missing")
		}
	})

	t.Run("returns false when versions mismatch", func(t *testing.T) {
		tmpDir := t.TempDir()
		os.Setenv("LF_BIN_DIR", tmpDir)

		// Create binaries and version file
		createFakeBinaries(t, tmpDir)
		versionFile := filepath.Join(tmpDir, ".pyapp-version")
		if err := os.WriteFile(versionFile, []byte("v1.0.0\n"), 0644); err != nil {
			t.Fatal(err)
		}

		mgr := &BinaryManager{version: "v1.2.3"}
		if mgr.areBinariesInstalled() {
			t.Error("expected false when versions mismatch")
		}
	})

	t.Run("returns true when all binaries exist and version matches", func(t *testing.T) {
		tmpDir := t.TempDir()
		os.Setenv("LF_BIN_DIR", tmpDir)

		// Create binaries and matching version file
		createFakeBinaries(t, tmpDir)
		versionFile := filepath.Join(tmpDir, ".pyapp-version")
		if err := os.WriteFile(versionFile, []byte("v1.2.3\n"), 0644); err != nil {
			t.Fatal(err)
		}

		mgr := &BinaryManager{version: "v1.2.3"}
		if !mgr.areBinariesInstalled() {
			t.Error("expected true when all binaries exist and version matches")
		}
	})

	t.Run("returns false when only some binaries exist", func(t *testing.T) {
		tmpDir := t.TempDir()
		os.Setenv("LF_BIN_DIR", tmpDir)

		// Create only server and rag binaries, not runtime
		suffix := getPlatformSuffix()
		ext := ""
		if runtime.GOOS == "windows" {
			ext = ".exe"
		}

		serverPath := filepath.Join(tmpDir, "llamafarm-server-"+suffix+ext)
		ragPath := filepath.Join(tmpDir, "llamafarm-rag-"+suffix+ext)

		if err := os.WriteFile(serverPath, []byte("fake"), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(ragPath, []byte("fake"), 0755); err != nil {
			t.Fatal(err)
		}

		// Create version file
		versionFile := filepath.Join(tmpDir, ".pyapp-version")
		if err := os.WriteFile(versionFile, []byte("v1.2.3\n"), 0644); err != nil {
			t.Fatal(err)
		}

		mgr := &BinaryManager{version: "v1.2.3"}
		if mgr.areBinariesInstalled() {
			t.Error("expected false when runtime binary is missing")
		}
	})
}

func TestDownloadBinary(t *testing.T) {
	origSource, origSourceSet := os.LookupEnv("LF_BINARY_SOURCE")
	defer func() {
		if origSourceSet {
			os.Setenv("LF_BINARY_SOURCE", origSource)
		} else {
			os.Unsetenv("LF_BINARY_SOURCE")
		}
	}()

	t.Run("routes to artifact download when LF_BINARY_SOURCE=artifact", func(t *testing.T) {
		os.Setenv("LF_BINARY_SOURCE", "artifact")
		os.Setenv("LF_ARTIFACT_RUN_ID", "")
		os.Setenv("GITHUB_TOKEN", "")

		mgr := &BinaryManager{version: "v1.2.3"}
		err := mgr.DownloadBinary("server")

		// Should fail because env vars are not set, but confirms routing
		if err == nil {
			t.Error("expected error when artifact env vars not set")
		}
		if err != nil && err.Error() != "LF_ARTIFACT_RUN_ID must be set when LF_BINARY_SOURCE=artifact" {
			t.Errorf("unexpected error: %v", err)
		}
	})

	t.Run("routes to release download by default", func(t *testing.T) {
		os.Unsetenv("LF_BINARY_SOURCE")

		mgr := &BinaryManager{version: "v1.2.3"}
		// Will fail with network error, but confirms routing to release download
		err := mgr.DownloadBinary("server")
		if err == nil {
			t.Error("expected error (network error expected in test)")
		}
		// We don't check specific error since it's a network error
	})

	t.Run("returns error for unknown component", func(t *testing.T) {
		os.Unsetenv("LF_BINARY_SOURCE")

		mgr := &BinaryManager{version: "v1.2.3"}
		err := mgr.DownloadBinary("unknown-component")
		if err == nil {
			t.Error("expected error for unknown component")
		}
		if err != nil && err.Error() != "unknown component: unknown-component" {
			t.Errorf("unexpected error message: %v", err)
		}
	})
}

// createFakeBinaries creates fake binary files for all three components
func createFakeBinaries(t *testing.T, dir string) {
	t.Helper()

	suffix := getPlatformSuffix()
	ext := ""
	if runtime.GOOS == "windows" {
		ext = ".exe"
	}

	components := []string{"server", "rag", "runtime"}
	for _, component := range components {
		name := "llamafarm-" + component + "-" + suffix + ext
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte("fake binary"), 0755); err != nil {
			t.Fatalf("failed to create fake %s binary: %v", component, err)
		}
	}
}
