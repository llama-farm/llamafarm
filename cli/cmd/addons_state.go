package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"github.com/gofrs/flock"
	"github.com/llamafarm/cli/cmd/utils"
)

const addonsStateVersion = "1"

type InstalledAddon struct {
	Name        string    `json:"name"`
	Version     string    `json:"version"`
	Component   string    `json:"component"`
	InstalledAt time.Time `json:"installed_at"`
	Platform    string    `json:"platform"` // "macos-arm64", "linux-x86_64"
}

type AddonsState struct {
	Version         string                      `json:"version"`
	InstalledAddons map[string]*InstalledAddon  `json:"installed_addons"`
}

func LoadAddonsState() (*AddonsState, error) {
	statePath, err := getAddonsStatePath()
	if err != nil {
		return nil, err
	}

	if _, err := os.Stat(statePath); os.IsNotExist(err) {
		return &AddonsState{
			Version:         addonsStateVersion,
			InstalledAddons: make(map[string]*InstalledAddon),
		}, nil
	}

	// Create a file lock (using .lock file to avoid locking the actual state file)
	lockPath := statePath + ".lock"
	fileLock := flock.New(lockPath)

	// Acquire shared lock for reading (with timeout)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	locked, err := fileLock.TryRLockContext(ctx, 100*time.Millisecond)
	if err != nil {
		return nil, fmt.Errorf("failed to acquire lock: %w", err)
	}
	if !locked {
		return nil, fmt.Errorf("timeout waiting for state file lock")
	}
	defer fileLock.Unlock()

	// Read the state file
	data, err := os.ReadFile(statePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read state file: %w", err)
	}

	var state AddonsState
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, fmt.Errorf("failed to parse state file: %w", err)
	}

	if state.InstalledAddons == nil {
		state.InstalledAddons = make(map[string]*InstalledAddon)
	}

	return &state, nil
}

func SaveAddonsState(state *AddonsState) error {
	statePath, err := getAddonsStatePath()
	if err != nil {
		return err
	}

	if err := os.MkdirAll(filepath.Dir(statePath), 0755); err != nil {
		return fmt.Errorf("failed to create state directory: %w", err)
	}

	// Create a file lock
	lockPath := statePath + ".lock"
	fileLock := flock.New(lockPath)

	// Acquire exclusive lock (with timeout)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	locked, err := fileLock.TryLockContext(ctx, 100*time.Millisecond)
	if err != nil {
		return fmt.Errorf("failed to acquire lock: %w", err)
	}
	if !locked {
		return fmt.Errorf("timeout waiting for state file lock")
	}
	defer fileLock.Unlock()

	// Write to temporary file first, then rename atomically
	tempPath := statePath + ".tmp"
	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal state: %w", err)
	}

	if err := os.WriteFile(tempPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write temp file: %w", err)
	}
	defer os.Remove(tempPath) // Clean up temp file if rename fails

	// Atomic rename
	if err := os.Rename(tempPath, statePath); err != nil {
		return fmt.Errorf("failed to rename temp file: %w", err)
	}

	return nil
}

func getAddonsStatePath() (string, error) {
	lfDir, err := utils.GetLFDataDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(lfDir, "addons.json"), nil
}

func (s *AddonsState) IsAddonInstalled(name string) bool {
	_, exists := s.InstalledAddons[name]
	return exists
}

func (s *AddonsState) MarkInstalled(name, version, component, platform string) {
	s.InstalledAddons[name] = &InstalledAddon{
		Name:        name,
		Version:     version,
		Component:   component,
		InstalledAt: time.Now(),
		Platform:    platform,
	}
}

func (s *AddonsState) MarkUninstalled(name string) {
	delete(s.InstalledAddons, name)
}

func getAddonsDir() (string, error) {
	lfDir, err := utils.GetLFDataDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(lfDir, "addons"), nil
}

func getPlatformString() string {
	osName := runtime.GOOS
	arch := runtime.GOARCH

	// Map Go OS names to PyApp conventions
	if osName == "darwin" {
		osName = "macos"
	}

	// Map Go arch names to PyApp conventions
	switch arch {
	case "amd64":
		arch = "x86_64"
	}

	return fmt.Sprintf("%s-%s", osName, arch)
}
