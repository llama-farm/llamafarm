package utils

import (
	"os"
	"path/filepath"
	"strings"
)

var OverrideCwd string

// getEffectiveCWD returns the directory to treat as the working directory.
// If the global --cwd flag is provided, it returns its absolute path; otherwise os.Getwd().
func GetEffectiveCWD() string {
	if strings.TrimSpace(OverrideCwd) != "" {
		if filepath.IsAbs(OverrideCwd) {
			return OverrideCwd
		}
		abs, err := filepath.Abs(OverrideCwd)
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
