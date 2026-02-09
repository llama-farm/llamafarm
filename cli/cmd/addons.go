package cmd

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"text/tabwriter"
	"time"

	"github.com/gofrs/flock"
	"github.com/llamafarm/cli/cmd/orchestrator"
	"github.com/llamafarm/cli/cmd/utils"
	"github.com/spf13/cobra"
)

// addonNamePattern validates addon names (alphanumeric, hyphens, underscores only)
var addonNamePattern = regexp.MustCompile(`^[a-z0-9_-]+$`)

// validateAddonName validates an addon name to prevent injection attacks
func validateAddonName(name string) error {
	if name == "" {
		return fmt.Errorf("addon name cannot be empty")
	}
	if len(name) > 50 {
		return fmt.Errorf("addon name too long (max 50 characters)")
	}
	if !addonNamePattern.MatchString(name) {
		return fmt.Errorf("invalid addon name: %s (must contain only lowercase letters, numbers, hyphens, and underscores)", name)
	}
	return nil
}

var addonsCmd = &cobra.Command{
	Use:   "addons",
	Short: "Manage LlamaFarm addons",
	Long: `Manage optional LlamaFarm addons.

Addons extend functionality with additional capabilities like speech-to-text,
text-to-speech, etc. They are installed as separate packages.

Available commands:
  list      - List available and installed addons
  install   - Install an addon
  uninstall - Uninstall an addon`,
}

var addonsListCmd = &cobra.Command{
	Use:   "list",
	Short: "List available and installed addons",
	Run:   runAddonsList,
}

var addonsInstallCmd = &cobra.Command{
	Use:   "install <addon-name>",
	Short: "Install an addon",
	Long: `Install an addon by downloading platform-specific wheel packages.

The addon will be available after restarting the affected service.

Examples:
  lf addons install stt    # Install speech-to-text
  lf addons install tts    # Install text-to-speech`,
	Args: cobra.ExactArgs(1),
	Run:  runAddonsInstall,
}

var addonsUninstallCmd = &cobra.Command{
	Use:   "uninstall <addon-name>",
	Short: "Uninstall an addon",
	Args:  cobra.ExactArgs(1),
	Run:   runAddonsUninstall,
}

func init() {
	rootCmd.AddCommand(addonsCmd)
	addonsCmd.AddCommand(addonsListCmd)
	addonsCmd.AddCommand(addonsInstallCmd)
	addonsCmd.AddCommand(addonsUninstallCmd)
}

func runAddonsList(cmd *cobra.Command, args []string) {
	if err := LoadAddonRegistry(); err != nil {
		utils.OutputError("Failed to load addon registry: %v\n", err)
		os.Exit(1)
	}

	state, err := LoadAddonsState()
	if err != nil {
		utils.OutputError("Failed to load state: %v\n", err)
		os.Exit(1)
	}

	utils.OutputInfo("Available Addons:\n\n")

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "NAME\tDESCRIPTION\tCOMPONENT\tSTATUS")

	// Sort addons
	names := make([]string, 0, len(AddonRegistry))
	for name := range AddonRegistry {
		names = append(names, name)
	}
	sort.Strings(names)

	for _, name := range names {
		addon := AddonRegistry[name]
		status := "Not installed"

		if installed, ok := state.InstalledAddons[name]; ok {
			status = fmt.Sprintf("Installed (v%s)", installed.Version)
		}

		fmt.Fprintf(w, "%s\t%s\t%s\t%s\n",
			addon.Name, addon.Description, addon.Component, status)
	}
	w.Flush()
}

// resolveDependencies returns the list of addons to install in order (dependencies first)
func resolveDependencies(addonName string, state *AddonsState, visited map[string]bool, stack map[string]bool) ([]string, error) {
	// Check for circular dependency
	if stack[addonName] {
		return nil, fmt.Errorf("circular dependency detected: %s", addonName)
	}

	// Already processed
	if visited[addonName] {
		return []string{}, nil
	}

	// Validate addon exists
	addon, ok := AddonRegistry[addonName]
	if !ok {
		return nil, fmt.Errorf("unknown addon: %s", addonName)
	}

	// Mark as being processed
	stack[addonName] = true
	visited[addonName] = true

	var installOrder []string

	// Process dependencies first
	for _, dep := range addon.Dependencies {
		deps, err := resolveDependencies(dep, state, visited, stack)
		if err != nil {
			return nil, err
		}
		installOrder = append(installOrder, deps...)
	}

	// Add this addon if not already installed
	if !state.IsAddonInstalled(addonName) {
		installOrder = append(installOrder, addonName)
	}

	// Unmark from processing stack
	delete(stack, addonName)

	return installOrder, nil
}

func runAddonsInstall(cmd *cobra.Command, args []string) {
	// Acquire global install lock to prevent concurrent installations
	lfDir, err := utils.GetLFDataDir()
	if err != nil {
		utils.OutputError("Failed to get data directory: %v\n", err)
		os.Exit(1)
	}

	lockPath := filepath.Join(lfDir, "addons-install.lock")
	installLock := flock.New(lockPath)

	// Try to acquire lock with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	locked, err := installLock.TryLockContext(ctx, 100*time.Millisecond)
	if err != nil {
		utils.OutputError("Failed to acquire installation lock: %v\n", err)
		os.Exit(1)
	}
	if !locked {
		utils.OutputError("Another addon installation is in progress. Please wait and try again.\n")
		os.Exit(1)
	}
	defer installLock.Unlock()

	if err := LoadAddonRegistry(); err != nil {
		utils.OutputError("Failed to load addon registry: %v\n", err)
		os.Exit(1)
	}

	addonName := args[0]

	// Validate addon name format
	if err := validateAddonName(addonName); err != nil {
		utils.OutputError("%v\n", err)
		os.Exit(1)
	}

	// Validate addon exists
	addon, ok := AddonRegistry[addonName]
	if !ok {
		utils.OutputError("Unknown addon: %s\n", addonName)
		utils.OutputInfo("Run 'lf addons list' to see available addons.\n")
		os.Exit(1)
	}

	// Load state
	state, err := LoadAddonsState()
	if err != nil {
		utils.OutputError("Failed to load state: %v\n", err)
		os.Exit(1)
	}

	// Check if already installed
	if state.IsAddonInstalled(addonName) {
		utils.OutputInfo("Addon '%s' is already installed.\n", addon.DisplayName)
		return
	}

	// Resolve dependencies
	installOrder, err := resolveDependencies(addonName, state, make(map[string]bool), make(map[string]bool))
	if err != nil {
		utils.OutputError("Dependency resolution failed: %v\n", err)
		os.Exit(1)
	}

	// Show what will be installed
	if len(installOrder) > 1 {
		utils.OutputInfo("Installing %s and its dependencies:\n", addon.DisplayName)
		for _, name := range installOrder {
			utils.OutputInfo("  - %s\n", AddonRegistry[name].DisplayName)
		}
		fmt.Println()
	}

	// Detect hardware and show notes
	hardware := orchestrator.DetectHardware()
	utils.OutputInfo("Detected hardware: %s\n", hardware)

	// Get server URL for service management
	serverURLToUse := serverURL
	if serverURLToUse == "" {
		serverURLToUse = "http://localhost:14345"
	}

	sm, err := orchestrator.NewServiceManager(serverURLToUse)
	if err != nil {
		utils.OutputError("Failed to initialize service manager: %v\n", err)
		os.Exit(1)
	}

	// Collect all services that need to be stopped
	servicesToRestart := make(map[string]bool)
	for _, installName := range installOrder {
		installAddon := AddonRegistry[installName]
		servicesToRestart[installAddon.Component] = true
	}

	// Stop all affected services once before installing
	if len(servicesToRestart) > 0 {
		utils.OutputInfo("\nStopping affected services...\n")
		for service := range servicesToRestart {
			utils.OutputInfo("  Stopping %s...\n", service)
			if err := sm.StopServices(service); err != nil {
				utils.OutputError("Failed to stop service %s: %v\n", service, err)
				os.Exit(1)
			}
		}
	}

	// Install each addon
	downloader := NewAddonDownloader("") // Use current version
	platform := getPlatformString()

	for _, installName := range installOrder {
		installAddon := AddonRegistry[installName]

		utils.OutputInfo("\nInstalling %s...\n", installAddon.DisplayName)

		// Show hardware notes for this addon
		if note, ok := installAddon.HardwareNotes[hardware]; ok {
			utils.OutputInfo("Note: %s\n", note)
		}

		// Download and install (skip if no packages - meta addon)
		if len(installAddon.Packages) > 0 {
			if err := downloader.DownloadAndInstallAddon(installAddon); err != nil {
				utils.OutputError("Installation failed: %v\n", err)
				os.Exit(1)
			}
		} else {
			utils.OutputInfo("Meta-addon (no packages to install)\n")
		}

		// Update state
		state.MarkInstalled(installAddon.Name, installAddon.Version, installAddon.Component, platform)
		if err := SaveAddonsState(state); err != nil {
			utils.OutputError("Warning: Failed to save state: %v\n", err)
		}

		utils.OutputSuccess("Addon '%s' installed successfully!\n", installAddon.DisplayName)
	}

	// Restart all affected services once after installation
	fmt.Println()
	utils.OutputSuccess("All addons installed successfully!\n")
	utils.OutputInfo("\nRestarting services (this may take up to 30 seconds)...\n")
	for service := range servicesToRestart {
		utils.OutputInfo("  Starting %s...\n", service)
		if err := sm.EnsureService(service); err != nil {
			utils.OutputError("Failed to start service %s: %v\n", service, err)
			utils.OutputInfo("You can manually start it with: lf services start %s\n", service)
		} else {
			utils.OutputSuccess("  %s started and health check passed\n", service)
		}
	}
}

func runAddonsUninstall(cmd *cobra.Command, args []string) {
	// Acquire global install lock to prevent concurrent install/uninstall races
	lfDir, err := utils.GetLFDataDir()
	if err != nil {
		utils.OutputError("Failed to get data directory: %v\n", err)
		os.Exit(1)
	}

	lockPath := filepath.Join(lfDir, "addons-install.lock")
	uninstallLock := flock.New(lockPath)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	locked, err := uninstallLock.TryLockContext(ctx, 100*time.Millisecond)
	if err != nil {
		utils.OutputError("Failed to acquire installation lock: %v\n", err)
		os.Exit(1)
	}
	if !locked {
		utils.OutputError("Another addon operation is in progress. Please wait and try again.\n")
		os.Exit(1)
	}
	defer uninstallLock.Unlock()

	if err := LoadAddonRegistry(); err != nil {
		utils.OutputError("Failed to load addon registry: %v\n", err)
		os.Exit(1)
	}

	addonName := args[0]

	// Validate addon name format
	if err := validateAddonName(addonName); err != nil {
		utils.OutputError("%v\n", err)
		os.Exit(1)
	}

	addon, ok := AddonRegistry[addonName]
	if !ok {
		utils.OutputError("Unknown addon: %s\n", addonName)
		os.Exit(1)
	}

	state, err := LoadAddonsState()
	if err != nil {
		utils.OutputError("Failed to load state: %v\n", err)
		os.Exit(1)
	}

	if !state.IsAddonInstalled(addonName) {
		utils.OutputInfo("Addon '%s' is not installed.\n", addon.DisplayName)
		return
	}

	// Get addon directory
	addonsDir, err := getAddonsDir()
	if err != nil {
		utils.OutputError("Failed to get addons directory: %v\n", err)
		os.Exit(1)
	}

	addonPath := filepath.Join(addonsDir, addonName)

	// Stop the affected service first
	serverURLToUse := serverURL
	if serverURLToUse == "" {
		serverURLToUse = "http://localhost:14345"
	}

	sm, err := orchestrator.NewServiceManager(serverURLToUse)
	if err != nil {
		utils.OutputError("Failed to initialize service manager: %v\n", err)
		os.Exit(1)
	}

	utils.OutputInfo("Stopping %s service...\n", addon.Component)
	if err := sm.StopServices(addon.Component); err != nil {
		utils.OutputWarning("Warning: Failed to stop service: %v\n", err)
		// Continue anyway - user can manually stop it
	}

	// Remove addon files
	utils.OutputInfo("Removing addon files...\n")
	if err := os.RemoveAll(addonPath); err != nil {
		utils.OutputError("Failed to remove addon files: %v\n", err)
		utils.OutputInfo("You may need to manually remove: %s\n", addonPath)
		os.Exit(1)
	}

	// Remove from state
	state.MarkUninstalled(addonName)
	if err := SaveAddonsState(state); err != nil {
		utils.OutputError("Failed to save state: %v\n", err)
		os.Exit(1)
	}

	utils.OutputSuccess("Addon '%s' uninstalled successfully.\n", addon.DisplayName)
	utils.OutputInfo("Restart the service: lf services start %s\n", addon.Component)
}
