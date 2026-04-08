package hfmodel

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
)

// DiscoverToken returns a HuggingFace authentication token using the same
// precedence order as huggingface_hub.get_token. The first non-empty source
// wins. Returns ("", nil) when no token is available — that is a valid state
// (anonymous requests work for public models).
//
// Precedence:
//  1. $HF_TOKEN
//  2. $HUGGING_FACE_HUB_TOKEN
//  3. $HF_HOME/token (when $HF_HOME is set)
//  4. ~/.cache/huggingface/token
//  5. ~/.huggingface/token (legacy path)
func DiscoverToken() (string, error) {
	if v := strings.TrimSpace(os.Getenv("HF_TOKEN")); v != "" {
		return v, nil
	}
	if v := strings.TrimSpace(os.Getenv("HUGGING_FACE_HUB_TOKEN")); v != "" {
		return v, nil
	}
	if hfHome := os.Getenv("HF_HOME"); hfHome != "" {
		if tok, ok, err := readTokenFile(filepath.Join(hfHome, "token")); err != nil {
			return "", err
		} else if ok {
			return tok, nil
		}
	}
	home, err := os.UserHomeDir()
	if err != nil {
		// Without a home dir we cannot find the legacy token files. Treat
		// as anonymous rather than failing — public models still work.
		return "", nil
	}
	if tok, ok, err := readTokenFile(filepath.Join(home, ".cache", "huggingface", "token")); err != nil {
		return "", err
	} else if ok {
		return tok, nil
	}
	if tok, ok, err := readTokenFile(filepath.Join(home, ".huggingface", "token")); err != nil {
		return "", err
	} else if ok {
		return tok, nil
	}
	return "", nil
}

// readTokenFile reads a token file, trims whitespace, and reports whether a
// non-empty token was found. Returns (token, true, nil) on success,
// ("", false, nil) when the file is missing or empty, and an error only on
// unexpected I/O failures (permission denied, etc.).
func readTokenFile(path string) (string, bool, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", false, nil
		}
		return "", false, err
	}
	tok := strings.TrimSpace(string(data))
	if tok == "" {
		return "", false, nil
	}
	return tok, true, nil
}
