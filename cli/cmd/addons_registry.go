package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"

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
	HardwareNotes    map[orchestrator.HardwareCapability]string `yaml:"-"`
	HardwareNotesRaw map[string]string                         `yaml:"hardware_notes"`
}

// AddonRegistry maps addon names to their definitions
var AddonRegistry map[string]*AddonDefinition

var registryOnce sync.Once

// LoadAddonRegistry loads the addon registry from individual YAML files in addons/registry/
func LoadAddonRegistry() error {
	var err error
	registryOnce.Do(func() {
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

			err = fmt.Errorf("addon registry directory not found. Searched:\n  - %s", filepath.Join(searchPaths...))
			return
		}

		// Load all .yaml files from the directory
		AddonRegistry = make(map[string]*AddonDefinition)

		entries, e := os.ReadDir(registryDir)
		if e != nil {
			err = fmt.Errorf("failed to read addon registry directory at %s: %w", registryDir, e)
			return
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

			AddonRegistry[addon.Name] = &addon
		}

		if len(AddonRegistry) == 0 {
			err = fmt.Errorf("no valid addons found in %s", registryDir)
		}
	})

	return err
}
