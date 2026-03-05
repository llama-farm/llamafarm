package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/llamafarm/cli/cmd/orchestrator"
	"github.com/llamafarm/cli/cmd/utils"
)

// getVenvPackageNames returns the set of normalized package names installed in a
// component's venv. It scans .dist-info directories in site-packages, which is
// the standard Python mechanism for tracking installed packages.
//
// Returns an empty map (not an error) if the venv doesn't exist yet, so that
// callers can safely proceed without removing any addon packages.
func getVenvPackageNames(component string) map[string]bool {
	packages := make(map[string]bool)

	sitePackagesDir, err := getComponentSitePackagesDir(component)
	if err != nil {
		utils.LogDebug(fmt.Sprintf("Cannot locate venv for %s: %v", component, err))
		return packages
	}

	entries, err := os.ReadDir(sitePackagesDir)
	if err != nil {
		utils.LogDebug(fmt.Sprintf("Cannot read site-packages for %s: %v", component, err))
		return packages
	}

	for _, entry := range entries {
		if !entry.IsDir() || !strings.HasSuffix(entry.Name(), ".dist-info") {
			continue
		}

		// dist-info format: "{name}-{version}.dist-info"
		// e.g. "numpy-2.4.2.dist-info" -> "numpy"
		baseName := strings.TrimSuffix(entry.Name(), ".dist-info")
		parts := strings.SplitN(baseName, "-", 2)
		if len(parts) == 0 {
			continue
		}

		pkgName := normalizePackageName(parts[0])
		packages[pkgName] = true
	}

	return packages
}

// getInstalledAddonPackageNames returns the set of top-level Python package
// directory names provided by all currently installed addons, excluding the
// named addon. This prevents addon installation from removing packages that
// another addon has already installed.
func getInstalledAddonPackageNames(excludeAddon string) map[string]bool {
	packages := make(map[string]bool)

	state, err := LoadAddonsState()
	if err != nil {
		utils.LogDebug(fmt.Sprintf("Cannot load addon state: %v", err))
		return packages
	}

	addonsDir, err := getAddonsDir()
	if err != nil {
		return packages
	}

	for name := range state.InstalledAddons {
		if name == excludeAddon {
			continue
		}

		addonPath := filepath.Join(addonsDir, name)
		entries, err := os.ReadDir(addonPath)
		if err != nil {
			continue
		}

		for _, entry := range entries {
			if !entry.IsDir() {
				continue
			}
			if strings.HasSuffix(entry.Name(), ".dist-info") || strings.HasSuffix(entry.Name(), ".data") {
				continue
			}
			packages[entry.Name()] = true
		}
	}

	return packages
}

// getComponentSitePackagesDir returns the path to a component's venv
// site-packages directory. Returns an error if the venv or site-packages
// directory doesn't exist.
func getComponentSitePackagesDir(component string) (string, error) {
	lfDir, err := utils.GetLFDataDir()
	if err != nil {
		return "", err
	}

	// Map component name to working directory using the service graph
	serviceDef, exists := orchestrator.ServiceGraph[component]
	if !exists {
		return "", fmt.Errorf("unknown component: %s", component)
	}

	venvBase := filepath.Join(lfDir, "src", serviceDef.WorkDir, ".venv")

	if runtime.GOOS == "windows" {
		dir := filepath.Join(venvBase, "Lib", "site-packages")
		if info, err := os.Stat(dir); err == nil && info.IsDir() {
			return dir, nil
		}
		return "", fmt.Errorf("site-packages not found at %s", dir)
	}

	// Unix: .venv/lib/python*/site-packages/
	libDir := filepath.Join(venvBase, "lib")
	entries, err := os.ReadDir(libDir)
	if err != nil {
		return "", err
	}

	for _, entry := range entries {
		if entry.IsDir() && strings.HasPrefix(entry.Name(), "python") {
			dir := filepath.Join(libDir, entry.Name(), "site-packages")
			if info, err := os.Stat(dir); err == nil && info.IsDir() {
				return dir, nil
			}
		}
	}

	return "", fmt.Errorf("site-packages not found under %s", libDir)
}

// normalizePackageName converts a Python package/distribution name to its
// normalized module form (lowercase, hyphens to underscores).
func normalizePackageName(name string) string {
	return strings.ReplaceAll(strings.ToLower(name), "-", "_")
}
