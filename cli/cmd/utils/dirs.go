package utils

import (
	"fmt"
	"os"
	"path/filepath"
)

// GetLFDataDir returns the directory to store LlamaFarm data.
func GetLFDataDir() (string, error) {
	dataDir := os.Getenv("LF_DATA_DIR")
	if dataDir != "" {
		return dataDir, nil
	}
	if homeDir, err := os.UserHomeDir(); err == nil {
		return filepath.Join(homeDir, ".llamafarm"), nil
	} else {
		return "", fmt.Errorf("getLFDataDir: could not determine home directory: %w", err)
	}
}
