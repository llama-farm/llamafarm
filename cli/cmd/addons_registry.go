package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/llamafarm/cli/cmd/orchestrator"
	"github.com/llamafarm/cli/cmd/utils"
	"gopkg.in/yaml.v3"
)

// AddonDefinition defines an addon with its metadata and dependencies
type AddonDefinition struct {
	Name          string                                     `yaml:"name"`
	DisplayName   string                                     `yaml:"display_name"`
	Description   string                                     `yaml:"description"`
	Component     string                                     `yaml:"component"`
	Version       string                                     `yaml:"version"`
	Dependencies  []string                                   `yaml:"dependencies"`
	Packages      []string                                   `yaml:"packages"`
	KeepPackages  []string                                   `yaml:"keep_packages"`
	HardwareNotes    map[orchestrator.HardwareCapability]string `yaml:"-"`
	HardwareNotesRaw map[string]string                         `yaml:"hardware_notes"`
}

// AddonRegistryStore holds loaded addon definitions. Create a new instance per
// CLI invocation via NewAddonRegistryStore() -- there is no sync.Once caching,
// so a transient failure doesn't stick for the process lifetime.
type AddonRegistryStore struct {
	addons map[string]*AddonDefinition
}

// NewAddonRegistryStore loads the addon registry from YAML files on disk.
func NewAddonRegistryStore() (*AddonRegistryStore, error) {
	r := &AddonRegistryStore{
		addons: make(map[string]*AddonDefinition),
	}
	if err := r.load(); err != nil {
		return nil, err
	}
	return r, nil
}

// Get returns the addon definition for the given name, or nil if not found.
func (r *AddonRegistryStore) Get(name string) (*AddonDefinition, bool) {
	addon, ok := r.addons[name]
	return addon, ok
}

// SortedNames returns all addon names in sorted order.
func (r *AddonRegistryStore) SortedNames() []string {
	names := make([]string, 0, len(r.addons))
	for name := range r.addons {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func (r *AddonRegistryStore) load() error {
	// Find registry directory - check multiple locations
	var registryDir string

	// 1. Check in source directory (for development)
	lfDir, _ := utils.GetLFDataDir()
	if lfDir != "" {
		srcPath := filepath.Join(lfDir, "src", "addons", "registry")
		if info, e := os.Stat(srcPath); e == nil && info.IsDir() {
			registryDir = srcPath
		}
	}

	// 2. Check relative to executable (for binary distribution)
	if registryDir == "" {
		exePath, _ := os.Executable()
		if exePath != "" {
			relPath := filepath.Join(filepath.Dir(exePath), "..", "addons", "registry")
			if info, e := os.Stat(relPath); e == nil && info.IsDir() {
				registryDir = relPath
			}
		}
	}

	// 3. Fallback to current directory + ../addons/registry
	if registryDir == "" {
		fallbackPath := filepath.Join("..", "addons", "registry")
		if info, e := os.Stat(fallbackPath); e == nil && info.IsDir() {
			registryDir = fallbackPath
		}
	}

	// If no valid registry directory found, return error with search paths
	if registryDir == "" {
		searchPaths := []string{}
		if lfDir != "" {
			searchPaths = append(searchPaths, filepath.Join(lfDir, "src", "addons", "registry"))
		}
		if exePath, _ := os.Executable(); exePath != "" {
			searchPaths = append(searchPaths, filepath.Join(filepath.Dir(exePath), "..", "addons", "registry"))
		}
		searchPaths = append(searchPaths, filepath.Join("..", "addons", "registry"))

		return fmt.Errorf("addon registry directory not found. Searched:\n  - %s", strings.Join(searchPaths, "\n  - "))
	}

	// Load all .yaml files from the directory
	entries, e := os.ReadDir(registryDir)
	if e != nil {
		return fmt.Errorf("failed to read addon registry directory at %s: %w", registryDir, e)
	}

	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".yaml" {
			continue
		}

		addonPath := filepath.Join(registryDir, entry.Name())
		data, e := os.ReadFile(addonPath)
		if e != nil {
			utils.LogDebug(fmt.Sprintf("Warning: failed to read %s: %v", entry.Name(), e))
			continue
		}

		var addon AddonDefinition
		if e := yaml.Unmarshal(data, &addon); e != nil {
			utils.LogDebug(fmt.Sprintf("Warning: failed to parse %s: %v", entry.Name(), e))
			continue
		}

		// Validate required fields
		if addon.Name == "" {
			utils.LogDebug(fmt.Sprintf("Warning: addon in %s missing name field", entry.Name()))
			continue
		}

		// Map string hardware notes to HardwareCapability enum
		addon.HardwareNotes = make(map[orchestrator.HardwareCapability]string)
		for key, value := range addon.HardwareNotesRaw {
			switch key {
			case "cuda":
				addon.HardwareNotes[orchestrator.HardwareCUDA] = value
			case "metal":
				addon.HardwareNotes[orchestrator.HardwareMetal] = value
			case "rocm":
				addon.HardwareNotes[orchestrator.HardwareROCm] = value
			case "cpu":
				addon.HardwareNotes[orchestrator.HardwareCPU] = value
			}
		}

		r.addons[addon.Name] = &addon
	}

	if len(r.addons) == 0 {
		return fmt.Errorf("no valid addons found in %s", registryDir)
	}

	return nil
}

