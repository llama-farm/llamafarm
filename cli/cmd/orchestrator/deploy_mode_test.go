package orchestrator

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestIsBinaryMode(t *testing.T) {
	orig, origSet := os.LookupEnv("LF_DEPLOY_MODE")
	defer func() {
		if origSet {
			os.Setenv("LF_DEPLOY_MODE", orig)
		} else {
			os.Unsetenv("LF_DEPLOY_MODE")
		}
	}()

	os.Setenv("LF_DEPLOY_MODE", "binary")
	if !IsBinaryMode() {
		t.Error("expected IsBinaryMode() = true when LF_DEPLOY_MODE=binary")
	}

	os.Setenv("LF_DEPLOY_MODE", "")
	if IsBinaryMode() {
		t.Error("expected IsBinaryMode() = false when LF_DEPLOY_MODE is empty")
	}

	os.Unsetenv("LF_DEPLOY_MODE")
	if IsBinaryMode() {
		t.Error("expected IsBinaryMode() = false when LF_DEPLOY_MODE is unset")
	}
}

func TestGetBinDir(t *testing.T) {
	origBinDir, origBinDirSet := os.LookupEnv("LF_BIN_DIR")
	origDataDir, origDataDirSet := os.LookupEnv("LF_DATA_DIR")
	defer func() {
		if origBinDirSet {
			os.Setenv("LF_BIN_DIR", origBinDir)
		} else {
			os.Unsetenv("LF_BIN_DIR")
		}
		if origDataDirSet {
			os.Setenv("LF_DATA_DIR", origDataDir)
		} else {
			os.Unsetenv("LF_DATA_DIR")
		}
	}()

	t.Run("uses LF_BIN_DIR when set", func(t *testing.T) {
		os.Setenv("LF_BIN_DIR", "/custom/bin")
		dir, err := GetBinDir()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if dir != "/custom/bin" {
			t.Errorf("got %q, want %q", dir, "/custom/bin")
		}
	})

	t.Run("falls back to LF_DATA_DIR/bin", func(t *testing.T) {
		os.Setenv("LF_BIN_DIR", "")
		os.Setenv("LF_DATA_DIR", "/tmp/lf-test-data")
		dir, err := GetBinDir()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		want := filepath.Join("/tmp/lf-test-data", "bin")
		if dir != want {
			t.Errorf("got %q, want %q", dir, want)
		}
	})
}

func TestGetPlatformSuffix(t *testing.T) {
	suffix := getPlatformSuffix()
	if suffix == "" {
		t.Fatal("getPlatformSuffix() returned empty string")
	}

	// Verify the suffix matches current platform
	switch runtime.GOOS {
	case "darwin":
		if suffix != "macos-"+mapArch(runtime.GOARCH) {
			t.Errorf("unexpected suffix on darwin: %q", suffix)
		}
	case "linux":
		if suffix != "linux-"+mapArch(runtime.GOARCH) {
			t.Errorf("unexpected suffix on linux: %q", suffix)
		}
	case "windows":
		if suffix != "windows-"+mapArch(runtime.GOARCH) {
			t.Errorf("unexpected suffix on windows: %q", suffix)
		}
	}
}

// mapArch converts Go arch to PyApp convention for test assertions.
func mapArch(goarch string) string {
	switch goarch {
	case "amd64":
		return "x86_64"
	default:
		return goarch
	}
}

func TestResolveBinaryPath(t *testing.T) {
	origBinDir, origBinDirSet := os.LookupEnv("LF_BIN_DIR")
	defer func() {
		if origBinDirSet {
			os.Setenv("LF_BIN_DIR", origBinDir)
		} else {
			os.Unsetenv("LF_BIN_DIR")
		}
	}()

	// Create a temp directory with fake binaries
	tmpDir := t.TempDir()
	os.Setenv("LF_BIN_DIR", tmpDir)

	t.Run("unknown service returns error", func(t *testing.T) {
		_, err := ResolveBinaryPath("nonexistent-service")
		if err == nil {
			t.Fatal("expected error for unknown service")
		}
	})

	t.Run("missing binary returns error", func(t *testing.T) {
		_, err := ResolveBinaryPath("server")
		if err == nil {
			t.Fatal("expected error when binary is missing")
		}
	})

	t.Run("finds platform-suffixed binary", func(t *testing.T) {
		suffix := getPlatformSuffix()
		ext := ""
		if runtime.GOOS == "windows" {
			ext = ".exe"
		}
		name := "llamafarm-server-" + suffix + ext
		fpath := filepath.Join(tmpDir, name)
		if err := os.WriteFile(fpath, []byte("fake"), 0755); err != nil {
			t.Fatal(err)
		}

		got, err := ResolveBinaryPath("server")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != fpath {
			t.Errorf("got %q, want %q", got, fpath)
		}
		os.Remove(fpath)
	})

	t.Run("falls back to bare name", func(t *testing.T) {
		ext := ""
		if runtime.GOOS == "windows" {
			ext = ".exe"
		}
		fpath := filepath.Join(tmpDir, "llamafarm-rag"+ext)
		if err := os.WriteFile(fpath, []byte("fake"), 0755); err != nil {
			t.Fatal(err)
		}

		got, err := ResolveBinaryPath("rag")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != fpath {
			t.Errorf("got %q, want %q", got, fpath)
		}
		os.Remove(fpath)
	})

	t.Run("prefers platform-suffixed over bare name", func(t *testing.T) {
		suffix := getPlatformSuffix()
		ext := ""
		if runtime.GOOS == "windows" {
			ext = ".exe"
		}

		barePath := filepath.Join(tmpDir, "llamafarm-runtime"+ext)
		suffixedPath := filepath.Join(tmpDir, "llamafarm-runtime-"+suffix+ext)
		if err := os.WriteFile(barePath, []byte("bare"), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(suffixedPath, []byte("suffixed"), 0755); err != nil {
			t.Fatal(err)
		}

		got, err := ResolveBinaryPath("universal-runtime")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != suffixedPath {
			t.Errorf("got %q, want platform-suffixed path %q", got, suffixedPath)
		}
		os.Remove(barePath)
		os.Remove(suffixedPath)
	})
}
