package cmd

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
)

// Environment Variables:
//   LF_VERSION_REF - Override the git ref (branch, tag, or commit SHA) to download
//                    source code from. Useful for CI/CD to test specific branches.
//                    Examples: "main", "feat-universal-runtime", "v1.2.3", "abc123..."
//                    Default: "main" branch

const (
	// GitHub repository information
	githubOwner = "llama-farm"
	githubRepo  = "llamafarm"
)

// SourceManager handles downloading and managing LlamaFarm source code
type SourceManager struct {
	homeDir       string
	srcDir        string
	versionFile   string
	pythonEnvMgr  *PythonEnvManager
	currentSource string // tracks what source version is currently installed
}

// NewSourceManager creates a new source code manager
func NewSourceManager(pythonEnvMgr *PythonEnvManager) (*SourceManager, error) {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return nil, fmt.Errorf("failed to get user home directory: %w", err)
	}

	llamafarmDir := filepath.Join(homeDir, ".llamafarm")
	srcDir := filepath.Join(llamafarmDir, "src")
	versionFile := filepath.Join(llamafarmDir, ".source_version")

	return &SourceManager{
		homeDir:      homeDir,
		srcDir:       srcDir,
		versionFile:  versionFile,
		pythonEnvMgr: pythonEnvMgr,
	}, nil
}

// EnsureSource ensures source code is downloaded and dependencies are synced
func (m *SourceManager) EnsureSource() error {
	// Determine what version we need
	targetVersion, err := m.GetCurrentCLIVersion()
	if err != nil {
		return fmt.Errorf("failed to determine CLI version: %w", err)
	}

	// Check if we already have this version
	currentVersion, _ := m.readVersionFile()
	if currentVersion == targetVersion && m.isSourceInstalled() {
		if debug {
			logDebug(fmt.Sprintf("Source code already at version: %s", targetVersion))
		}

		// Still need to ensure dependencies are synced and datamodel is generated
		// (in case source is there but uv sync wasn't run or schema changed)
		if !m.areDependenciesSynced() {
			OutputProgress("Source code found, syncing dependencies...\n")
			if err := m.SyncDependencies(); err != nil {
				return err
			}
		}

		// Always ensure datamodel is generated before starting services
		if err := m.GenerateDatamodel(); err != nil {
			return fmt.Errorf("failed to generate datamodel: %w", err)
		}

		return nil
	}

	// Need to download new source
	OutputProgress("Downloading LlamaFarm source code (%s)...\n", targetVersion)

	if err := m.DownloadSource(targetVersion); err != nil {
		return fmt.Errorf("failed to download source: %w", err)
	}

	// Update version file
	if err := m.writeVersionFile(targetVersion); err != nil {
		OutputWarning("Warning: could not write version file: %v\n", err)
	}

	// Sync dependencies
	if err := m.SyncDependencies(); err != nil {
		return fmt.Errorf("failed to sync dependencies: %w", err)
	}

	// Generate datamodel (must be done after dependencies are synced)
	if err := m.GenerateDatamodel(); err != nil {
		return fmt.Errorf("failed to generate datamodel: %w", err)
	}

	return nil
}

// GetCurrentCLIVersion determines the version of the currently running CLI
// It first checks for the LF_VERSION_REF environment variable (useful for CI/CD),
// then falls back to "main" branch by default.
func (m *SourceManager) GetCurrentCLIVersion() (string, error) {
	// Check for LF_VERSION_REF environment variable first (CI/CD override)
	if versionRef := strings.TrimSpace(os.Getenv("LF_VERSION_REF")); versionRef != "" {
		if debug {
			logDebug(fmt.Sprintf("Using version from LF_VERSION_REF: %s", versionRef))
		}
		return versionRef, nil
	}

	// Check if we're in development mode
	// In dev mode, we use "main" branch
	// In release mode, we use the CLI version tag

	// For now, always use "main" - in production this would check the actual CLI version
	// TODO: Implement proper version detection from build-time variables
	return "main", nil
}

// DownloadSource downloads the source code for a specific version
// Supports branches, tags, and commit SHAs via GitHub's archive API
func (m *SourceManager) DownloadSource(version string) error {
	// Build download URL
	// For branches: https://github.com/llama-farm/llamafarm/archive/refs/heads/{branch}.tar.gz
	// For tags: https://github.com/llama-farm/llamafarm/archive/refs/tags/{tag}.tar.gz
	// For commits/any ref: https://github.com/llama-farm/llamafarm/archive/{ref}.tar.gz

	var downloadURL string

	// Try to intelligently determine the ref type
	if version == "main" || version == "dev" {
		// Known branch names
		downloadURL = fmt.Sprintf("https://github.com/%s/%s/archive/refs/heads/main.tar.gz",
			githubOwner, githubRepo)
	} else if strings.HasPrefix(version, "v") && len(version) > 1 {
		// Looks like a version tag (starts with 'v')
		downloadURL = fmt.Sprintf("https://github.com/%s/%s/archive/refs/tags/%s.tar.gz",
			githubOwner, githubRepo, version)
	} else if len(version) == 40 {
		// Looks like a full commit SHA (40 hex characters)
		downloadURL = fmt.Sprintf("https://github.com/%s/%s/archive/%s.tar.gz",
			githubOwner, githubRepo, version)
	} else {
		// Assume it's a branch name or tag - try as branch first
		// GitHub will return 404 if not found, and the generic ref format works for most cases
		downloadURL = fmt.Sprintf("https://github.com/%s/%s/archive/refs/heads/%s.tar.gz",
			githubOwner, githubRepo, version)
	}

	if debug {
		logDebug(fmt.Sprintf("Downloading source from: %s", downloadURL))
	}

	// Download the archive
	resp, err := http.Get(downloadURL)
	if err != nil {
		return fmt.Errorf("failed to download: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to download: HTTP %d", resp.StatusCode)
	}

	// Create temporary directory
	tmpDir, err := os.MkdirTemp("", "llamafarm-source-*")
	if err != nil {
		return fmt.Errorf("failed to create temp directory: %w", err)
	}
	defer os.RemoveAll(tmpDir)

	// Download to temp file
	archivePath := filepath.Join(tmpDir, "source.tar.gz")
	tmpFile, err := os.Create(archivePath)
	if err != nil {
		return fmt.Errorf("failed to create temp file: %w", err)
	}

	_, err = io.Copy(tmpFile, resp.Body)
	tmpFile.Close()
	if err != nil {
		return fmt.Errorf("failed to write archive: %w", err)
	}

	// Extract to temporary location
	extractDir := filepath.Join(tmpDir, "extracted")
	if err := os.MkdirAll(extractDir, 0755); err != nil {
		return fmt.Errorf("failed to create extraction directory: %w", err)
	}

	if err := m.extractTarGz(archivePath, extractDir); err != nil {
		return fmt.Errorf("failed to extract archive: %w", err)
	}

	// GitHub archives extract to a directory named "repo-branch" or "repo-version"
	// Find the extracted directory
	entries, err := os.ReadDir(extractDir)
	if err != nil {
		return fmt.Errorf("failed to read extracted directory: %w", err)
	}

	if len(entries) == 0 {
		return fmt.Errorf("extracted archive is empty")
	}

	extractedSrcDir := filepath.Join(extractDir, entries[0].Name())

	// Remove old source directory if it exists
	if err := os.RemoveAll(m.srcDir); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to remove old source directory: %w", err)
	}

	// Move extracted source to final location
	if err := os.Rename(extractedSrcDir, m.srcDir); err != nil {
		return fmt.Errorf("failed to move source to final location: %w", err)
	}

	OutputProgress("Source code extracted to %s\n", m.srcDir)
	m.currentSource = version
	return nil
}

// SyncDependencies runs `uv sync` on server and rag directories
func (m *SourceManager) SyncDependencies() error {
	serverDir := filepath.Join(m.srcDir, "server")
	ragDir := filepath.Join(m.srcDir, "rag")

	// Verify directories exist
	if _, err := os.Stat(serverDir); os.IsNotExist(err) {
		return fmt.Errorf("server directory not found: %s", serverDir)
	}
	if _, err := os.Stat(ragDir); os.IsNotExist(err) {
		return fmt.Errorf("rag directory not found: %s", ragDir)
	}

	// Run uv sync on both directories in parallel for speed
	var wg sync.WaitGroup
	var serverErr, ragErr error

	wg.Add(2)

	// Sync server dependencies
	go func() {
		defer wg.Done()
		OutputProgress("Syncing server dependencies...\n")
		serverErr = m.syncDirectory(serverDir, "server")
	}()

	// Sync rag dependencies
	go func() {
		defer wg.Done()
		OutputProgress("Syncing RAG dependencies...\n")
		ragErr = m.syncDirectory(ragDir, "rag")
	}()

	wg.Wait()

	// Check for errors
	if serverErr != nil {
		return fmt.Errorf("failed to sync server dependencies: %w", serverErr)
	}
	if ragErr != nil {
		return fmt.Errorf("failed to sync rag dependencies: %w", ragErr)
	}

	OutputProgress("Dependencies synced successfully\n")
	return nil
}

// syncDirectory runs `uv sync` in a specific directory
func (m *SourceManager) syncDirectory(dir string, name string) error {
	uvPath := m.pythonEnvMgr.uvManager.GetUVPath()

	// Verify the directory exists and contains pyproject.toml
	pyprojectPath := filepath.Join(dir, "pyproject.toml")
	if _, err := os.Stat(pyprojectPath); os.IsNotExist(err) {
		return fmt.Errorf("pyproject.toml not found in %s", dir)
	}

	// Run UV sync command in the specific project directory
	// This ensures .venv is created in the correct location
	cmd := exec.Command(uvPath, "sync")
	cmd.Dir = dir // Critical: run from project directory so .venv is created there
	cmd.Env = m.pythonEnvMgr.GetEnvForProcess()

	if debug {
		logDebug(fmt.Sprintf("Running 'uv sync' in directory: %s", dir))
	}

	// Capture output for debugging
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("uv sync failed in %s: %w\nOutput: %s", name, err, string(output))
	}

	if debug {
		logDebug(fmt.Sprintf("%s uv sync output: %s", name, string(output)))
	}

	return nil
}

// isSourceInstalled checks if source code is installed
func (m *SourceManager) isSourceInstalled() bool {
	serverDir := filepath.Join(m.srcDir, "server")
	ragDir := filepath.Join(m.srcDir, "rag")

	_, serverErr := os.Stat(serverDir)
	_, ragErr := os.Stat(ragDir)

	return serverErr == nil && ragErr == nil
}

// areDependenciesSynced checks if dependencies are already synced
func (m *SourceManager) areDependenciesSynced() bool {
	// Check for .venv directories as an indicator that uv sync has been run
	serverVenv := filepath.Join(m.srcDir, "server", ".venv")
	ragVenv := filepath.Join(m.srcDir, "rag", ".venv")

	_, serverErr := os.Stat(serverVenv)
	_, ragErr := os.Stat(ragVenv)

	return serverErr == nil && ragErr == nil
}

// readVersionFile reads the current version from the version file
func (m *SourceManager) readVersionFile() (string, error) {
	data, err := os.ReadFile(m.versionFile)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(data)), nil
}

// writeVersionFile writes the current version to the version file
func (m *SourceManager) writeVersionFile(version string) error {
	return os.WriteFile(m.versionFile, []byte(version), 0644)
}

// extractTarGz extracts a tar.gz archive to the specified directory
func (m *SourceManager) extractTarGz(archivePath, destDir string) error {
	file, err := os.Open(archivePath)
	if err != nil {
		return err
	}
	defer file.Close()

	gzr, err := gzip.NewReader(file)
	if err != nil {
		return err
	}
	defer gzr.Close()

	tr := tar.NewReader(gzr)

	for {
		header, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		target := filepath.Join(destDir, header.Name)

		// Security: ensure the target path is within destDir
		if !strings.HasPrefix(target, filepath.Clean(destDir)+string(os.PathSeparator)) {
			return fmt.Errorf("illegal file path in archive: %s", header.Name)
		}

		switch header.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0755); err != nil {
				return err
			}
		case tar.TypeReg:
			// Ensure parent directory exists
			if err := os.MkdirAll(filepath.Dir(target), 0755); err != nil {
				return err
			}

			f, err := os.OpenFile(target, os.O_CREATE|os.O_RDWR, os.FileMode(header.Mode))
			if err != nil {
				return err
			}
			if _, err := io.Copy(f, tr); err != nil {
				f.Close()
				return err
			}
			f.Close()
		}
	}

	return nil
}

// extractZip extracts a zip archive to the specified directory
func (m *SourceManager) extractZip(archivePath, destDir string) error {
	r, err := zip.OpenReader(archivePath)
	if err != nil {
		return err
	}
	defer r.Close()

	for _, f := range r.File {
		fpath := filepath.Join(destDir, f.Name)

		// Security: ensure the target path is within destDir
		if !strings.HasPrefix(fpath, filepath.Clean(destDir)+string(os.PathSeparator)) {
			return fmt.Errorf("illegal file path in archive: %s", f.Name)
		}

		if f.FileInfo().IsDir() {
			os.MkdirAll(fpath, os.ModePerm)
			continue
		}

		if err := os.MkdirAll(filepath.Dir(fpath), os.ModePerm); err != nil {
			return err
		}

		outFile, err := os.OpenFile(fpath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, f.Mode())
		if err != nil {
			return err
		}

		rc, err := f.Open()
		if err != nil {
			outFile.Close()
			return err
		}

		_, err = io.Copy(outFile, rc)
		outFile.Close()
		rc.Close()

		if err != nil {
			return err
		}
	}

	return nil
}

// GetSourceDir returns the path to the source directory
func (m *SourceManager) GetSourceDir() string {
	return m.srcDir
}

// GetServerDir returns the path to the server source directory
func (m *SourceManager) GetServerDir() string {
	return filepath.Join(m.srcDir, "server")
}

// GetRAGDir returns the path to the RAG source directory
func (m *SourceManager) GetRAGDir() string {
	return filepath.Join(m.srcDir, "rag")
}

// GetDesignerDir returns the path to the designer source directory
func (m *SourceManager) GetDesignerDir() string {
	return filepath.Join(m.srcDir, "designer")
}

// GetConfigDir returns the path to the config source directory
func (m *SourceManager) GetConfigDir() string {
	return filepath.Join(m.srcDir, "config")
}

// GetUniversalRuntimeDir returns the path to the universal runtime source directory
func (m *SourceManager) GetUniversalRuntimeDir() string {
	return filepath.Join(m.srcDir, "runtimes", "universal")
}

// GenerateDatamodel generates the config datamodel types
// This must be run after source download and dependency sync, but before starting services
func (m *SourceManager) GenerateDatamodel() error {
	configDir := m.GetConfigDir()

	// Check if config directory exists
	if _, err := os.Stat(configDir); os.IsNotExist(err) {
		return fmt.Errorf("config directory not found: %s", configDir)
	}

	// Check if generate_types.py exists
	generateScript := filepath.Join(configDir, "generate_types.py")
	if _, err := os.Stat(generateScript); os.IsNotExist(err) {
		// Fallback to shell script if Python version doesn't exist yet
		generateScript = filepath.Join(configDir, "generate-types.sh")
		if _, err := os.Stat(generateScript); os.IsNotExist(err) {
			OutputWarning("Warning: generate script not found, skipping datamodel generation\n")
			return nil
		}
	}

	OutputProgress("Generating config datamodel...\n")

	uvPath := m.pythonEnvMgr.uvManager.GetUVPath()

	// Run the generation script
	cmd := exec.Command(uvPath, "run", "python", "generate_types.py")
	cmd.Dir = configDir
	cmd.Env = m.pythonEnvMgr.GetEnvForProcess()

	if debug {
		logDebug(fmt.Sprintf("Running datamodel generation in: %s", configDir))
	}

	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("datamodel generation failed: %w\nOutput: %s", err, string(output))
	}

	if debug {
		logDebug(fmt.Sprintf("Datamodel generation output: %s", string(output)))
	}

	OutputProgress("Config datamodel generated successfully\n")
	return nil
}
