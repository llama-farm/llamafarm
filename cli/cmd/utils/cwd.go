package utils

import (
	"os"
	"path/filepath"
	"strings"
)

var OverrideCwd string

// ExpandTilde expands the tilde (~) prefix to the user's home directory.
// It handles:
//   - "~" alone → home directory
//   - "~/path" → home directory + path
//
// Note: "~user/path" syntax is NOT supported as it requires looking up other
// users' home directories, which has security implications and platform-specific
// behavior. Such paths are returned unchanged.
//
// If os.UserHomeDir() fails, the original path is returned unchanged.
func ExpandTilde(path string) string {
	if path == "" {
		return path
	}

	// Handle "~" alone
	if path == "~" {
		home, err := os.UserHomeDir()
		if err != nil {
			return path
		}
		return home
	}

	// Handle "~/..." pattern
	if strings.HasPrefix(path, "~/") {
		home, err := os.UserHomeDir()
		if err != nil {
			return path
		}
		return filepath.Join(home, path[2:])
	}

	// All other cases: return unchanged
	// This includes ~user/path, absolute paths, relative paths, etc.
	return path
}

// GetEffectiveCWD returns the directory to treat as the working directory.
// If the global --cwd flag is provided, it expands tilde and returns its
// absolute path; otherwise os.Getwd().
func GetEffectiveCWD() string {
	if strings.TrimSpace(OverrideCwd) != "" {
		// Expand tilde before checking if path is absolute
		expandedPath := ExpandTilde(OverrideCwd)

		if filepath.IsAbs(expandedPath) {
			return expandedPath
		}
		abs, err := filepath.Abs(expandedPath)
		if err != nil {
			return "."
		}
		return abs
	}

	wd, _ := os.Getwd()
	if wd == "" {
		return "."
	}

	return wd
}
