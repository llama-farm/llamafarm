package hfmodel

import (
	"os"
	"path/filepath"
	"testing"
)

// withFakeHome unsets all HF env vars and points $HOME at a tmp dir so token
// discovery is fully isolated from the developer's real environment.
func withFakeHome(t *testing.T) string {
	t.Helper()
	tmp := t.TempDir()
	t.Setenv("HOME", tmp)
	t.Setenv("HF_TOKEN", "")
	t.Setenv("HUGGING_FACE_HUB_TOKEN", "")
	t.Setenv("HF_HOME", "")
	return tmp
}

func writeFile(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestDiscoverToken_HFTokenEnvWins(t *testing.T) {
	home := withFakeHome(t)
	writeFile(t, filepath.Join(home, ".cache", "huggingface", "token"), "from-file")
	t.Setenv("HF_TOKEN", "from-env")

	tok, err := DiscoverToken()
	if err != nil {
		t.Fatal(err)
	}
	if tok != "from-env" {
		t.Errorf("got %q, want from-env", tok)
	}
}

func TestDiscoverToken_HuggingFaceHubTokenEnv(t *testing.T) {
	withFakeHome(t)
	t.Setenv("HUGGING_FACE_HUB_TOKEN", "hub-env")

	tok, err := DiscoverToken()
	if err != nil {
		t.Fatal(err)
	}
	if tok != "hub-env" {
		t.Errorf("got %q, want hub-env", tok)
	}
}

func TestDiscoverToken_HFHomeTokenFile(t *testing.T) {
	withFakeHome(t)
	hfHome := t.TempDir()
	t.Setenv("HF_HOME", hfHome)
	writeFile(t, filepath.Join(hfHome, "token"), "hf-home-token\n")

	tok, err := DiscoverToken()
	if err != nil {
		t.Fatal(err)
	}
	if tok != "hf-home-token" {
		t.Errorf("got %q, want hf-home-token (trimmed)", tok)
	}
}

func TestDiscoverToken_DefaultCachePath(t *testing.T) {
	home := withFakeHome(t)
	writeFile(t, filepath.Join(home, ".cache", "huggingface", "token"), "  default-cache-token  ")

	tok, err := DiscoverToken()
	if err != nil {
		t.Fatal(err)
	}
	if tok != "default-cache-token" {
		t.Errorf("got %q, want default-cache-token (trimmed)", tok)
	}
}

func TestDiscoverToken_LegacyPath(t *testing.T) {
	home := withFakeHome(t)
	// Don't create the new path; only the legacy one.
	writeFile(t, filepath.Join(home, ".huggingface", "token"), "legacy-token")

	tok, err := DiscoverToken()
	if err != nil {
		t.Fatal(err)
	}
	if tok != "legacy-token" {
		t.Errorf("got %q, want legacy-token", tok)
	}
}

func TestDiscoverToken_NoTokenAvailable(t *testing.T) {
	withFakeHome(t)
	tok, err := DiscoverToken()
	if err != nil {
		t.Fatal(err)
	}
	if tok != "" {
		t.Errorf("got %q, want empty", tok)
	}
}

func TestDiscoverToken_EmptyFileTreatedAsMissing(t *testing.T) {
	home := withFakeHome(t)
	writeFile(t, filepath.Join(home, ".cache", "huggingface", "token"), "   \n   ")
	writeFile(t, filepath.Join(home, ".huggingface", "token"), "fallback")

	tok, err := DiscoverToken()
	if err != nil {
		t.Fatal(err)
	}
	if tok != "fallback" {
		t.Errorf("got %q, want fallback (empty file should be skipped)", tok)
	}
}
