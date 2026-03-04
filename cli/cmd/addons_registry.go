package cmd

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"

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

	// 4. Fall back to embedded registry (always available in the binary)
	if registryDir == "" {
		utils.LogDebug("Using embedded addon registry")
		return r.loadFromEmbedded()
	}

	// Load all .yaml files from the on-disk directory
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

		if e := r.parseAddon(entry.Name(), data); e != nil {
			utils.LogDebug(fmt.Sprintf("Warning: %v", e))
			continue
		}
	}

	if len(r.addons) == 0 {
		return fmt.Errorf("no valid addons found in %s", registryDir)
	}

	return nil
}

// loadFromEmbedded reads addon YAML files from the embedded registry filesystem.
func (r *AddonRegistryStore) loadFromEmbedded() error {
	entries, e := fs.ReadDir(embeddedRegistry, "registry")
	if e != nil {
		return fmt.Errorf("failed to read embedded addon registry: %w", e)
	}

	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".yaml" {
			continue
		}

		data, e := fs.ReadFile(embeddedRegistry, "registry/"+entry.Name())
		if e != nil {
			utils.LogDebug(fmt.Sprintf("Warning: failed to read embedded %s: %v", entry.Name(), e))
			continue
		}

		if e := r.parseAddon(entry.Name(), data); e != nil {
			utils.LogDebug(fmt.Sprintf("Warning: %v", e))
			continue
		}
	}

	if len(r.addons) == 0 {
		return fmt.Errorf("no valid addons found in embedded registry")
	}

	return nil
}

// parseAddon unmarshals YAML data into an AddonDefinition and adds it to the store.
func (r *AddonRegistryStore) parseAddon(filename string, data []byte) error {
	var addon AddonDefinition
	if e := yaml.Unmarshal(data, &addon); e != nil {
		return fmt.Errorf("failed to parse %s: %v", filename, e)
	}

	if addon.Name == "" {
		return fmt.Errorf("addon in %s missing name field", filename)
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
	return nil
}

