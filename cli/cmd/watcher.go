package cmd

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"llamafarm-cli/cmd/config"

	"github.com/fsnotify/fsnotify"
)

// StartConfigWatcher starts a background file watcher that synchronizes
// llamafarm config files (yaml/toml/json) between the current directory and the home directory
func StartConfigWatcher(namespace, project string) error {
	if namespace == "" || project == "" {
		return fmt.Errorf("namespace and project are required for config watcher")
	}

	// Get the effective current working directory
	cwd := getEffectiveCWD()

	// Find config files in both locations
	cwdConfigPath, err := config.FindConfigFile(cwd)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Warning: No llamafarm config file found in current directory: %v\n", err)
	}

	homeDir, err := os.UserHomeDir()
	if err != nil {
		return fmt.Errorf("failed to get user home directory: %w", err)
	}
	homeConfigDir := filepath.Join(homeDir, ".llamafarm", "projects", namespace, project)

	// Watch home directory (create if needed)
	if err := os.MkdirAll(homeConfigDir, 0755); err != nil {
		return fmt.Errorf("failed to create home config directory: %w", err)
	}

	// Create watcher
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return fmt.Errorf("failed to create file watcher: %w", err)
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

	if err := watcher.Add(homeConfigDir); err != nil {
		return fmt.Errorf("failed to watch home config directory: %w", err)
	}

	// Watch directories for config file creation/changes
	// Watch current directory
	if err := watcher.Add(cwd); err != nil {
		return fmt.Errorf("failed to watch current directory: %w", err)
	}

	logDebug(fmt.Sprintf("cwdConfigPath: %s, homeConfigPath: %s", cwdConfigPath, homeConfigPath))
	cwdConfigInfo, ciErr := os.Stat(cwdConfigPath)
	homeConfigInfo, hiErr := os.Stat(homeConfigPath)
	logDebug(fmt.Sprintf("cwdConfigInfo: %v, homeConfigInfo: %v", cwdConfigInfo, homeConfigInfo))
	logDebug(fmt.Sprintf("ciErr: %v, hiErr: %v", ciErr, hiErr))
	// logDebug(fmt.Sprintf("cwdConfigInfo.ModTime(): %v, homeConfigInfo.ModTime(): %v", cwdConfigInfo.ModTime(), homeConfigInfo.ModTime()))
	if ciErr == nil && (hiErr != nil || cwdConfigInfo.ModTime().After(homeConfigInfo.ModTime())) {
		if err := syncConfigFiles(cwdConfigPath, homeConfigPath); err != nil {
			fmt.Fprintf(os.Stderr, "Failed to sync config files: %v\n", err)
		}
	} else if hiErr == nil && (ciErr != nil || homeConfigInfo.ModTime().After(cwdConfigInfo.ModTime())) {
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

		// Track last modification times to avoid infinite loops
		lastModTimes := make(map[string]time.Time)

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

					// Skip if we just modified this file (to prevent infinite loops)
					if lastMod, exists := lastModTimes[sourcePath]; exists {
						if time.Since(lastMod) < time.Second {
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

					// Sync the files
					if err := syncConfigFiles(sourcePath, targetPath); err != nil {
						fmt.Fprintf(os.Stderr, "Failed to sync config files: %v\n", err)
						continue
					}

					// Update last modification time
					lastModTimes[targetPath] = time.Now()

					logDebug(fmt.Sprintf("Synced %s -> %s\n", sourcePath, targetPath))
				}

			case err, ok := <-watcher.Errors:
				if !ok {
					return
				}
				fmt.Fprintf(os.Stderr, "Watcher error: %v\n", err)
			}
		}
	}()

	fmt.Fprintf(os.Stderr, "Watching project: %s\n", cwd)
	logDebug(fmt.Sprintf("Watching target directory: %s\n", cwdConfigPath))
	logDebug(fmt.Sprintf("Watching home directory: %s\n", homeConfigDir))

	return nil
}

// syncConfigFiles copies the source config file to the target location
func syncConfigFiles(sourcePath, targetPath string) error {
	// Ensure target directory exists
	targetDir := filepath.Dir(targetPath)
	if err := os.MkdirAll(targetDir, 0755); err != nil {
		return fmt.Errorf("failed to create target directory %s: %w", targetDir, err)
	}

	// Open source file
	sourceFile, err := os.Open(sourcePath)
	if err != nil {
		return fmt.Errorf("failed to open source file %s: %w", sourcePath, err)
	}
	defer sourceFile.Close()

	// Create target file
	targetFile, err := os.Create(targetPath)
	if err != nil {
		return fmt.Errorf("failed to create target file %s: %w", targetPath, err)
	}
	defer targetFile.Close()

	// Copy contents
	_, err = io.Copy(targetFile, sourceFile)
	if err != nil {
		return fmt.Errorf("failed to copy file contents: %w", err)
	}

	// Ensure data is written
	if err := targetFile.Sync(); err != nil {
		return fmt.Errorf("failed to sync target file: %w", err)
	}

	return nil
}
