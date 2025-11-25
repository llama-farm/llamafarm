package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/llamafarm/cli/cmd/config"

	"github.com/fsnotify/fsnotify"
	"github.com/llamafarm/cli/cmd/utils"
)

type ConfigPaths struct {
	CwdDir         string
	CwdConfigPath  string
	HomeConfigDir  string
	HomeConfigPath string
}

func SyncConfigForProject(namespace, project string) error {
	configPaths, err := resolveConfigPaths(namespace, project)
	if err != nil {
		return fmt.Errorf("failed to resolve config paths: %w", err)
	}

	cwdConfigPath := configPaths.CwdConfigPath
	homeConfigPath := configPaths.HomeConfigPath

	return syncConfigFiles(cwdConfigPath, homeConfigPath)
}

// StartConfigWatcher starts a background file watcher that synchronizes
// llamafarm config files (yaml/toml/json) between the current directory and the home directory
func StartConfigWatcher(namespace, project string) error {
	configPaths, err := resolveConfigPaths(namespace, project)
	if err != nil {
		return fmt.Errorf("failed to resolve config paths: %w", err)
	}

	cwd := configPaths.CwdDir
	cwdConfigPath := configPaths.CwdConfigPath
	homeConfigDir := configPaths.HomeConfigDir
	homeConfigPath := configPaths.HomeConfigPath

	// Create watcher
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return fmt.Errorf("failed to create file watcher: %w", err)
	}

	if err := watcher.Add(homeConfigDir); err != nil {
		return fmt.Errorf("failed to watch home config directory: %w", err)
	}

	// Watch directories for config file creation/changes
	// Watch current directory
	if err := watcher.Add(cwd); err != nil {
		return fmt.Errorf("failed to watch current directory: %w", err)
	}

	utils.LogDebug(fmt.Sprintf("cwdConfigPath: %s, homeConfigPath: %s", cwdConfigPath, homeConfigPath))
	cwdConfigInfo, ciErr := os.Stat(cwdConfigPath)
	homeConfigInfo, hiErr := os.Stat(homeConfigPath)
	utils.LogDebug(fmt.Sprintf("cwdConfigInfo: %v, homeConfigInfo: %v", cwdConfigInfo, homeConfigInfo))
	utils.LogDebug(fmt.Sprintf("ciErr: %v, hiErr: %v", ciErr, hiErr))

	// Sync config files with priority: prefer configs that have valid name/namespace fields
	// This ensures that valid configs don't get overwritten by invalid ones
	if ciErr == nil && hiErr == nil {
		// Both exist - check which one has valid name/namespace fields
		cwdCfg, cwdErr := config.LoadConfigFile(cwdConfigPath)
		homeCfg, homeErr := config.LoadConfigFile(homeConfigPath)

		cwdHasValid := cwdErr == nil && func() bool {
			_, err := cwdCfg.GetProjectInfo()
			return err == nil
		}()

		homeHasValid := homeErr == nil && func() bool {
			_, err := homeCfg.GetProjectInfo()
			return err == nil
		}()

		if cwdHasValid && !homeHasValid {
			// CWD config is valid, home is not - sync CWD -> home
			if err := syncConfigFiles(cwdConfigPath, homeConfigPath); err != nil {
				fmt.Fprintf(os.Stderr, "Failed to sync config files: %v\n", err)
			}
		} else if homeHasValid && !cwdHasValid {
			// Home config is valid, CWD is not - sync home -> CWD
			if err := syncConfigFiles(homeConfigPath, cwdConfigPath); err != nil {
				fmt.Fprintf(os.Stderr, "Failed to sync config files: %v\n", err)
			}
		} else if cwdHasValid && homeHasValid {
			// Both valid - sync the newer one to the older one
			// This allows changes from the designer (home) to propagate to CWD
			if homeConfigInfo.ModTime().After(cwdConfigInfo.ModTime()) {
				// Home config is newer (e.g., edited via designer) - sync home -> CWD
				if err := syncConfigFiles(homeConfigPath, cwdConfigPath); err != nil {
					fmt.Fprintf(os.Stderr, "Failed to sync config files: %v\n", err)
				}
			} else {
				// CWD config is newer or same age - sync CWD -> home
				if err := syncConfigFiles(cwdConfigPath, homeConfigPath); err != nil {
					fmt.Fprintf(os.Stderr, "Failed to sync config files: %v\n", err)
				}
			}
		}
		// If neither has valid name/namespace, don't sync (will fail later anyway)
	} else if ciErr == nil && hiErr != nil {
		// Only CWD config exists - sync CWD -> home
		if err := syncConfigFiles(cwdConfigPath, homeConfigPath); err != nil {
			fmt.Fprintf(os.Stderr, "Failed to sync config files: %v\n", err)
		}
	} else if hiErr == nil && ciErr != nil {
		// Only home config exists - sync home -> CWD
		if err := syncConfigFiles(homeConfigPath, cwdConfigPath); err != nil {
			fmt.Fprintf(os.Stderr, "Failed to sync config files: %v\n", err)
		}
	}

	// Also watch existing config files if they exist
	if cwdConfigPath != "" {
		if err := watcher.Add(cwdConfigPath); err != nil {
			fmt.Fprintf(os.Stderr, "Warning: Failed to watch cwd config file %s: %v\n", cwdConfigPath, err)
		}
	}

	if homeConfigPath != "" {
		if err := watcher.Add(homeConfigPath); err != nil {
			fmt.Fprintf(os.Stderr, "Warning: Failed to watch home config file %s: %v\n", homeConfigPath, err)
		}
	}

	// Start watcher goroutine
	go func() {
		defer watcher.Close()

		// Track last modification times to avoid infinite loops and debounce rapid changes
		lastModTimes := make(map[string]time.Time)
		// Debounce duration to wait for file changes to settle
		// Keep this short (100ms) to ensure rapid sync while preventing bounce
		const debounceDuration = 100 * time.Millisecond

		for {
			select {
			case event, ok := <-watcher.Events:
				if !ok {
					return
				}

				// Only process write events
				if event.Has(fsnotify.Write) {
					sourcePath := event.Name

					// Check if this is a directory event (file creation)
					if info, err := os.Stat(sourcePath); err == nil && info.IsDir() {
						// Check if any llamafarm config file was created in this directory
						if configFile, err := config.FindConfigFile(sourcePath); err == nil {
							// Add the new file to watch
							if err := watcher.Add(configFile); err != nil {
								fmt.Fprintf(os.Stderr, "Failed to watch new config file %s: %v\n", configFile, err)
							}
							sourcePath = configFile
						} else {
							continue
						}
					}

					// Skip if we recently synced this file (debounce to prevent rapid successive syncs)
					if lastMod, exists := lastModTimes[sourcePath]; exists {
						if time.Since(lastMod) < debounceDuration {
							continue
						}
					}

					// Check if this is a config file we're interested in
					if !config.IsConfigFile(sourcePath) {
						continue
					}

					// Determine target path
					var targetPath string
					if filepath.Dir(sourcePath) == cwd {
						// Source is in current directory, target should be in home directory
						targetBaseName := filepath.Base(sourcePath)
						targetPath = filepath.Join(homeConfigDir, targetBaseName)
					} else if strings.HasPrefix(sourcePath, homeConfigDir) {
						// Source is in home directory, target should be in current directory
						targetBaseName := filepath.Base(sourcePath)
						targetPath = filepath.Join(cwd, targetBaseName)
					} else {
						continue
					}

					// Also check if we recently synced to the target path (prevents ping-pong)
					if lastMod, exists := lastModTimes[targetPath]; exists {
						if time.Since(lastMod) < debounceDuration {
							continue
						}
					}

					// Wait briefly to let the file write complete
					time.Sleep(20 * time.Millisecond)

					// Sync the files
					if err := syncConfigFiles(sourcePath, targetPath); err != nil {
						fmt.Fprintf(os.Stderr, "Failed to sync config files: %v\n", err)
						continue
					}

					// Update last modification times for both source and target
					now := time.Now()
					lastModTimes[sourcePath] = now
					lastModTimes[targetPath] = now

					utils.LogDebug(fmt.Sprintf("Synced %s -> %s\n", sourcePath, targetPath))
				}

			case err, ok := <-watcher.Errors:
				if !ok {
					return
				}
				fmt.Fprintf(os.Stderr, "Watcher error: %v\n", err)
			}
		}
	}()

	if debug {
		fmt.Fprintf(os.Stderr, "Watching project: %s\n", cwd)
	}
	utils.LogDebug(fmt.Sprintf("Watching target directory: %s\n", cwdConfigPath))
	utils.LogDebug(fmt.Sprintf("Watching home directory: %s\n", homeConfigDir))

	return nil
}

func resolveConfigPaths(namespace, project string) (*ConfigPaths, error) {
	if namespace == "" || project == "" {
		return nil, fmt.Errorf("namespace and project are required for config watcher")
	}

	// Get the effective current working directory
	cwd := utils.GetEffectiveCWD()

	// Find config files in both locations
	cwdConfigPath, err := config.FindConfigFile(cwd)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Warning: No llamafarm config file found in current directory: %v\n", err)
	}

	homeDir, err := os.UserHomeDir()
	if err != nil {
		return nil, fmt.Errorf("failed to get user home directory: %w", err)
	}
	homeConfigDir := filepath.Join(homeDir, ".llamafarm", "projects", namespace, project)

	// Watch home directory (create if needed)
	if err := os.MkdirAll(homeConfigDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create home config directory: %w", err)
	}

	homeConfigPath, _ := config.FindConfigFile(homeConfigDir)
	if homeConfigPath == "" {
		homeConfigPath = filepath.Join(homeConfigDir, "llamafarm.yaml")
	}

	// Compute the config file name
	var configFileName string
	if cwdConfigPath != "" {
		configFileName = filepath.Base(cwdConfigPath)
	} else if homeConfigPath != "" {
		configFileName = filepath.Base(homeConfigPath)
	} else {
		configFileName = "llamafarm.yaml"
	}

	if homeConfigPath == "" {
		homeConfigPath = filepath.Join(homeConfigDir, configFileName)
	}

	if cwdConfigPath == "" {
		cwdConfigPath = filepath.Join(cwd, configFileName)
	}

	return &ConfigPaths{
		CwdDir:         cwd,
		CwdConfigPath:  cwdConfigPath,
		HomeConfigDir:  homeConfigDir,
		HomeConfigPath: homeConfigPath,
	}, nil
}

// syncConfigFiles copies the source config file to the target location using atomic writes
func syncConfigFiles(sourcePath, targetPath string) error {
	// Ensure target directory exists
	targetDir := filepath.Dir(targetPath)
	if err := os.MkdirAll(targetDir, 0755); err != nil {
		return fmt.Errorf("failed to create target directory %s: %w", targetDir, err)
	}

	// Read source file contents
	sourceData, err := os.ReadFile(sourcePath)
	if err != nil {
		return fmt.Errorf("failed to read source file %s: %w", sourcePath, err)
	}

	// Get source file info for permissions
	sourceInfo, err := os.Stat(sourcePath)
	if err != nil {
		return fmt.Errorf("failed to stat source file %s: %w", sourcePath, err)
	}

	// Write to target atomically using temp file + rename
	tmpFile, err := os.CreateTemp(targetDir, ".llamafarm-sync-*.tmp")
	if err != nil {
		return fmt.Errorf("failed to create temp file in %s: %w", targetDir, err)
	}
	tmpPath := tmpFile.Name()

	// Clean up temp file on error
	var renamed bool
	defer func() {
		if tmpFile != nil {
			tmpFile.Close()
		}
		// Always remove temp file if rename didn't succeed
		if !renamed {
			os.Remove(tmpPath)
		}
	}()

	// Write source data to temp file
	if _, err := tmpFile.Write(sourceData); err != nil {
		return fmt.Errorf("failed to write to temp file: %w", err)
	}

	// Sync to ensure data is written to disk
	if err := tmpFile.Sync(); err != nil {
		return fmt.Errorf("failed to sync temp file: %w", err)
	}

	// Close the temp file
	if err := tmpFile.Close(); err != nil {
		return fmt.Errorf("failed to close temp file: %w", err)
	}

	// Set permissions to match source file
	if err := os.Chmod(tmpPath, sourceInfo.Mode()); err != nil {
		return fmt.Errorf("failed to set permissions on temp file: %w", err)
	}

	// Atomically rename temp file to target file
	if err := os.Rename(tmpPath, targetPath); err != nil {
		return fmt.Errorf("failed to rename temp file to target: %w", err)
	}
	renamed = true

	return nil
}

// StartConfigWatcherForCommand is a helper function that starts the config watcher
// for any command that needs it. It loads the project configuration and starts
// the watcher if both namespace and project can be determined.
func StartConfigWatcherForCommand() {
	// Load config to get namespace and project for watcher
	cwd := utils.GetEffectiveCWD()
	cfg, err := config.LoadConfig(cwd)
	if err != nil {
		// If no config file found, don't start watcher (not an error)
		return
	}

	projectInfo, err := cfg.GetProjectInfo()
	if err != nil {
		// If project info can't be extracted, don't start watcher (not an error)
		return
	}

	// Start the config file watcher in background
	if err := StartConfigWatcher(projectInfo.Namespace, projectInfo.Project); err != nil {
		fmt.Fprintf(os.Stderr, "Warning: Failed to start config watcher: %v\n", err)
	}
}

// EnsureConfigSynced forces an immediate sync of config files and waits for it to complete.
// This should be called after API operations that modify the config to ensure the local
// file is up-to-date before the command returns.
func EnsureConfigSynced(namespace, project string) error {
	configPaths, err := resolveConfigPaths(namespace, project)
	if err != nil {
		return fmt.Errorf("failed to resolve config paths: %w", err)
	}

	cwdConfigPath := configPaths.CwdConfigPath
	homeConfigPath := configPaths.HomeConfigPath

	// Check which file exists and is newer
	cwdInfo, cwdErr := os.Stat(cwdConfigPath)
	homeInfo, homeErr := os.Stat(homeConfigPath)

	if cwdErr != nil && homeErr != nil {
		// Neither file exists - nothing to sync
		return nil
	}

	// Determine sync direction based on modification times and validity
	var sourcePath, targetPath string

	if cwdErr == nil && homeErr == nil {
		// Both exist - sync the newer one to the older one, but validate first
		cwdCfg, cwdLoadErr := config.LoadConfigFile(cwdConfigPath)
		homeCfg, homeLoadErr := config.LoadConfigFile(homeConfigPath)

		cwdValid := cwdLoadErr == nil && func() bool {
			_, err := cwdCfg.GetProjectInfo()
			return err == nil
		}()

		homeValid := homeLoadErr == nil && func() bool {
			_, err := homeCfg.GetProjectInfo()
			return err == nil
		}()

		// Prefer valid configs, then prefer newer configs
		if homeValid && !cwdValid {
			sourcePath = homeConfigPath
			targetPath = cwdConfigPath
		} else if cwdValid && !homeValid {
			sourcePath = cwdConfigPath
			targetPath = homeConfigPath
		} else if homeInfo.ModTime().After(cwdInfo.ModTime()) {
			// Both valid or both invalid - use the newer one
			sourcePath = homeConfigPath
			targetPath = cwdConfigPath
		} else {
			sourcePath = cwdConfigPath
			targetPath = homeConfigPath
		}
	} else if homeErr == nil {
		// Only home exists
		sourcePath = homeConfigPath
		targetPath = cwdConfigPath
	} else {
		// Only cwd exists
		sourcePath = cwdConfigPath
		targetPath = homeConfigPath
	}

	// Perform sync
	utils.LogDebug(fmt.Sprintf("Forcing config sync: %s -> %s", sourcePath, targetPath))
	if err := syncConfigFiles(sourcePath, targetPath); err != nil {
		return fmt.Errorf("failed to sync config files: %w", err)
	}

	return nil
}
