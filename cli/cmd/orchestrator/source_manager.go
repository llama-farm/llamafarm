package orchestrator

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/llamafarm/cli/cmd/utils"
	"github.com/llamafarm/cli/internal/buildinfo"
)

// Environment Variables:
//   LF_VERSION_REF - Override the git ref (branch, tag, or commit SHA) to download
//                    source code from. Useful for CI/CD to test specific branches.
//                    Examples: "main", "feat-universal-runtime", "v1.2.3", "abc123..."
//                    Default: "main" branch
//   LF_GITHUB_TOKEN - GitHub token for downloading artifacts from GitHub Actions.
//                    Also checks GITHUB_TOKEN if LF_GITHUB_TOKEN is not set.
//                    Required for downloading branch artifacts (optional for releases).

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
	processMgr    *ProcessManager // for stopping services during upgrades
	currentSource string          // tracks what source version is currently installed
	mu            sync.Mutex      // protects against parallel downloads
}

// NewSourceManager creates a new source code manager
func NewSourceManager(pythonEnvMgr *PythonEnvManager, processMgr *ProcessManager) (*SourceManager, error) {
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
		processMgr:   processMgr,
	}, nil
}

// EnsureSource ensures source code is downloaded and dependencies are synced
// This method is protected by a mutex to prevent parallel downloads
func (m *SourceManager) EnsureSource() error {
	// Lock to prevent parallel downloads from multiple service starts
	m.mu.Lock()
	defer m.mu.Unlock()

	// Determine what version we need
	targetVersion, err := m.GetCurrentCLIVersion()
	if err != nil {
		return fmt.Errorf("failed to determine CLI version: %w", err)
	}

	// Dev builds should skip source management entirely - developer runs services manually
	// Check the actual build version, not targetVersion (which returns "main" for dev builds)
	if buildinfo.CurrentVersion == "dev" || buildinfo.CurrentVersion == "" {
		return nil
	}

	// Check if we already have this version
	currentVersion, _ := m.readVersionFile()
	if currentVersion == targetVersion && m.isSourceInstalled() {
		utils.LogDebug(fmt.Sprintf("Source code already at version: %s", targetVersion))

		// Still need to ensure dependencies are synced and datamodel is generated
		// (in case source is there but uv sync wasn't run or schema changed)
		if !m.areDependenciesSynced() {
			utils.LogDebug("Source code found, syncing dependencies...")
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

	// Source is out of sync - stop all running services before upgrading
	utils.LogDebug(fmt.Sprintf("Source version mismatch (current: %s, target: %s) - stopping all services before upgrade", currentVersion, targetVersion))
	if m.processMgr != nil {
		m.processMgr.StopAllProcesses()
		utils.LogDebug("All services stopped successfully")
	}

	// Need to download new source
	utils.LogDebug(fmt.Sprintf("Downloading LlamaFarm source code (%s)...", targetVersion))

	if err := m.DownloadSource(targetVersion); err != nil {
		return fmt.Errorf("failed to download source: %w", err)
	}

	// Update version file
	if err := m.writeVersionFile(targetVersion); err != nil {
		utils.LogDebug(fmt.Sprintf("Warning: could not write version file: %v", err))
	}

	// Sync dependencies (config and common sequentially, then server and rag in parallel)
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
// then uses the CLI's Version variable (set at build time), falling back to "main" for dev builds.
func (m *SourceManager) GetCurrentCLIVersion() (string, error) {
	// Check for LF_VERSION_REF environment variable first (CI/CD override)
	if versionRef := strings.TrimSpace(os.Getenv("LF_VERSION_REF")); versionRef != "" {
		utils.LogDebug(fmt.Sprintf("Using version from LF_VERSION_REF: %s", versionRef))
		return versionRef, nil
	}

	// Use the CLI's actual version (set by build flags during release)
	cliVersion := strings.TrimSpace(buildinfo.CurrentVersion)

	// Development builds (Version = "dev") should use "main" branch
	if cliVersion == "" || cliVersion == "dev" {
		utils.LogDebug("CLI is dev build, using main branch for source")
		return "main", nil
	}

	// For release builds, use the version tag directly
	// The Version variable should already include the "v" prefix (e.g., "v1.2.3")
	utils.LogDebug(fmt.Sprintf("Using CLI version for source: %s", cliVersion))
	return cliVersion, nil
}

// getBranchArtifactURL attempts to find a GitHub Actions artifact for a branch
// Returns the download URL if found, or empty string if not available
// Requires LF_GITHUB_TOKEN environment variable to be set
func (m *SourceManager) getBranchArtifactURL(branchName string) (string, error) {
	// Check for GitHub token
	token := strings.TrimSpace(os.Getenv("GITHUB_TOKEN"))
	if token == "" {
		return "", fmt.Errorf("no GitHub token available (set GITHUB_TOKEN)")
	}

	// Construct artifact name based on branch (matching CI workflow naming)
	// The CI workflow creates artifacts named: source-archive-{branch}-{sha}
	// We'll search for artifacts matching the branch pattern
	artifactNamePattern := fmt.Sprintf("source-archive-%s-", branchName)

	// Query GitHub Actions API for workflow runs
	// We'll look for the latest successful run of the "pack" job
	workflowRunsURL := fmt.Sprintf("https://api.github.com/repos/%s/%s/actions/workflows/cli.yml/runs?branch=%s&status=success&per_page=1",
		githubOwner, githubRepo, branchName)

	req, err := http.NewRequest("GET", workflowRunsURL, nil)
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Accept", "application/vnd.github+json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))
	req.Header.Set("X-GitHub-Api-Version", "2022-11-28")

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to query workflow runs: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("workflow runs API returned status %d", resp.StatusCode)
	}

	var workflowRuns struct {
		WorkflowRuns []struct {
			ID         int64  `json:"id"`
			HeadSHA    string `json:"head_sha"`
			HeadBranch string `json:"head_branch"`
		} `json:"workflow_runs"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&workflowRuns); err != nil {
		return "", fmt.Errorf("failed to decode workflow runs response: %w", err)
	}

	if len(workflowRuns.WorkflowRuns) == 0 {
		return "", fmt.Errorf("no successful workflow runs found for branch %s", branchName)
	}

	runID := workflowRuns.WorkflowRuns[0].ID

	// Get artifacts for this run
	artifactsURL := fmt.Sprintf("https://api.github.com/repos/%s/%s/actions/runs/%d/artifacts",
		githubOwner, githubRepo, runID)

	req, err = http.NewRequest("GET", artifactsURL, nil)
	if err != nil {
		return "", fmt.Errorf("failed to create artifacts request: %w", err)
	}

	req.Header.Set("Accept", "application/vnd.github+json")
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))
	req.Header.Set("X-GitHub-Api-Version", "2022-11-28")

	resp, err = client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to query artifacts: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("artifacts API returned status %d", resp.StatusCode)
	}

	var artifacts struct {
		Artifacts []struct {
			Name               string `json:"name"`
			ArchiveDownloadURL string `json:"archive_download_url"`
		} `json:"artifacts"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&artifacts); err != nil {
		return "", fmt.Errorf("failed to decode artifacts response: %w", err)
	}

	// Find artifact matching our pattern
	for _, artifact := range artifacts.Artifacts {
		if strings.HasPrefix(artifact.Name, artifactNamePattern) {
			// The archive_download_url is a zip containing both the tar.gz and sha256 files
			// We need to download and extract it, then find the tar.gz file
			// For now, return the URL - we'll handle extraction in DownloadSource
			return artifact.ArchiveDownloadURL, nil
		}
	}

	return "", fmt.Errorf("no matching artifact found for branch %s", branchName)
}

// DownloadSource downloads the source code for a specific version
// For release tags (v*): Downloads from packaged source archive on GitHub releases
// For branches: Tries to download from GitHub Actions artifacts if token is available, otherwise falls back to GitHub archive API
// For commits: Uses GitHub's archive API (fallback, may not include built designer)
func (m *SourceManager) DownloadSource(version string) error {
	var downloadURL string
	var usePackagedArchive bool

	// For release tags (v*), use the packaged source archive which includes built designer
	if strings.HasPrefix(version, "v") && len(version) > 1 {
		// Use the packaged source archive from GitHub releases
		downloadURL = fmt.Sprintf("https://github.com/%s/%s/releases/download/%s/llamafarm-dist-%s.tar.gz",
			githubOwner, githubRepo, version, version)
		usePackagedArchive = true
	} else {
		// For branches, try to download from GitHub Actions artifacts first
		// This requires a GitHub token and will fall back to archive API if not available
		if !strings.HasPrefix(version, "v") && len(version) != 40 {
			// This looks like a branch name (not a commit SHA)
			if artifactURL, err := m.getBranchArtifactURL(version); err == nil && artifactURL != "" {
				downloadURL = artifactURL
				usePackagedArchive = true
				utils.LogDebug(fmt.Sprintf("Found artifact for branch %s: %s", version, artifactURL))
			} else {
				if err != nil {
					utils.LogDebug(fmt.Sprintf("Could not get artifact for branch %s: %v, falling back to archive", version, err))
				}
				// Fall back to GitHub archive API
				if version == "main" || version == "dev" {
					downloadURL = fmt.Sprintf("https://github.com/%s/%s/archive/refs/heads/main.tar.gz",
						githubOwner, githubRepo)
				} else {
					downloadURL = fmt.Sprintf("https://github.com/%s/%s/archive/refs/heads/%s.tar.gz",
						githubOwner, githubRepo, version)
				}
				usePackagedArchive = false
			}
		} else if len(version) == 40 {
			// Looks like a full commit SHA (40 hex characters)
			downloadURL = fmt.Sprintf("https://github.com/%s/%s/archive/%s.tar.gz",
				githubOwner, githubRepo, version)
			usePackagedArchive = false
		} else {
			// Assume it's a branch name or tag - try as branch first
			downloadURL = fmt.Sprintf("https://github.com/%s/%s/archive/refs/heads/%s.tar.gz",
				githubOwner, githubRepo, version)
			usePackagedArchive = false
		}
	}

	utils.LogDebug(fmt.Sprintf("Downloading source from: %s (packaged: %v)", downloadURL, usePackagedArchive))

	// Download the archive
	// For GitHub Actions artifacts, we need to use authentication
	var resp *http.Response
	var err error
	if usePackagedArchive && strings.Contains(downloadURL, "api.github.com") {
		// This is a GitHub Actions artifact, requires authentication
		req, err := http.NewRequest("GET", downloadURL, nil)
		if err != nil {
			return fmt.Errorf("failed to create request: %w", err)
		}

		token := strings.TrimSpace(os.Getenv("GITHUB_TOKEN"))
		if token == "" {
			return fmt.Errorf("GitHub token required for artifact download (set GITHUB_TOKEN)")
		}

		req.Header.Set("Accept", "application/vnd.github+json")
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))
		req.Header.Set("X-GitHub-Api-Version", "2022-11-28")

		client := &http.Client{Timeout: 5 * time.Minute}
		resp, err = client.Do(req)
		if err != nil {
			return fmt.Errorf("failed to download artifact: %w", err)
		}
		defer resp.Body.Close()
	} else {
		// Regular download
		resp, err = http.Get(downloadURL)
		if err != nil {
			return fmt.Errorf("failed to download: %w", err)
		}
		defer resp.Body.Close()
	}

	// For packaged archives, if we get 404, fall back to GitHub archive
	if resp.StatusCode == http.StatusNotFound && usePackagedArchive && !strings.Contains(downloadURL, "api.github.com") {
		utils.LogDebug(fmt.Sprintf("Packaged archive not found for %s, falling back to GitHub archive", version))
		// Fall back to GitHub archive API
		downloadURL = fmt.Sprintf("https://github.com/%s/%s/archive/refs/tags/%s.tar.gz",
			githubOwner, githubRepo, version)
		resp.Body.Close()
		resp, err = http.Get(downloadURL)
		if err != nil {
			return fmt.Errorf("failed to download fallback archive: %w", err)
		}
		defer resp.Body.Close()
		usePackagedArchive = false
	}

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to download: HTTP %d", resp.StatusCode)
	}

	// Create temporary directory
	tmpDir, err := os.MkdirTemp("", "llamafarm-source-*")
	if err != nil {
		return fmt.Errorf("failed to create temp directory: %w", err)
	}
	defer os.RemoveAll(tmpDir)

	// Handle GitHub Actions artifact (ZIP) vs regular tar.gz
	var archivePath string
	if usePackagedArchive && strings.Contains(downloadURL, "api.github.com") {
		// This is a GitHub Actions artifact ZIP, extract it first
		zipPath := filepath.Join(tmpDir, "artifact.zip")
		zipFile, err := os.Create(zipPath)
		if err != nil {
			return fmt.Errorf("failed to create zip file: %w", err)
		}

		_, err = io.Copy(zipFile, resp.Body)
		zipFile.Close()
		if err != nil {
			return fmt.Errorf("failed to write zip file: %w", err)
		}

		// Extract ZIP to find the tar.gz file
		zipExtractDir := filepath.Join(tmpDir, "zip-extract")
		if err := os.MkdirAll(zipExtractDir, 0755); err != nil {
			return fmt.Errorf("failed to create zip extraction directory: %w", err)
		}

		if err := m.extractZip(zipPath, zipExtractDir); err != nil {
			return fmt.Errorf("failed to extract artifact zip: %w", err)
		}

		// Find the tar.gz file in the extracted ZIP
		entries, err := os.ReadDir(zipExtractDir)
		if err != nil {
			return fmt.Errorf("failed to read zip extraction directory: %w", err)
		}

		found := false
		for _, entry := range entries {
			if strings.HasSuffix(entry.Name(), ".tar.gz") && !strings.HasSuffix(entry.Name(), ".sha256") {
				archivePath = filepath.Join(zipExtractDir, entry.Name())
				found = true
				break
			}
		}

		if !found {
			return fmt.Errorf("tar.gz file not found in artifact zip")
		}
	} else {
		// Regular tar.gz download
		archivePath = filepath.Join(tmpDir, "source.tar.gz")
		tmpFile, err := os.Create(archivePath)
		if err != nil {
			return fmt.Errorf("failed to create temp file: %w", err)
		}

		_, err = io.Copy(tmpFile, resp.Body)
		tmpFile.Close()
		if err != nil {
			return fmt.Errorf("failed to write archive: %w", err)
		}
	}

	// Extract to temporary location
	extractDir := filepath.Join(tmpDir, "extracted")
	if err := os.MkdirAll(extractDir, 0755); err != nil {
		return fmt.Errorf("failed to create extraction directory: %w", err)
	}

	if err := m.extractTarGz(archivePath, extractDir); err != nil {
		return fmt.Errorf("failed to extract archive: %w", err)
	}

	// Find the extracted directory
	// Packaged archives: "llamafarm-{version}"
	// GitHub archives: "llamafarm-{branch}" or "llamafarm-{version}"
	entries, err := os.ReadDir(extractDir)
	if err != nil {
		return fmt.Errorf("failed to read extracted directory: %w", err)
	}

	if len(entries) == 0 {
		return fmt.Errorf("extracted archive is empty")
	}

	extractedSrcDir := filepath.Join(extractDir, entries[0].Name())

	// Remove old source directory if it exists
	// Use RemoveAllWithRetry to handle Windows file locking issues
	// where Python processes may still hold file handles briefly after termination
	if err := utils.RemoveAllWithRetry(m.srcDir); err != nil {
		return fmt.Errorf("failed to remove old source directory: %w", err)
	}

	// Move extracted source to final location (handles cross-filesystem moves)
	if err := utils.MoveDir(extractedSrcDir, m.srcDir); err != nil {
		return fmt.Errorf("failed to move source to final location: %w", err)
	}

	utils.LogDebug(fmt.Sprintf("Source code extracted to %s", m.srcDir))
	m.currentSource = version
	return nil
}

// SyncDependencies runs `uv sync` on config, common, server, rag, and universal-runtime directories
func (m *SourceManager) SyncDependencies() error {
	configDir := filepath.Join(m.srcDir, "config")
	commonDir := filepath.Join(m.srcDir, "common")
	serverDir := filepath.Join(m.srcDir, "server")
	ragDir := filepath.Join(m.srcDir, "rag")
	universalRuntimeDir := filepath.Join(m.srcDir, "runtimes", "universal")

	// Verify directories exist
	if _, err := os.Stat(configDir); os.IsNotExist(err) {
		return fmt.Errorf("config directory not found: %s", configDir)
	}
	if _, err := os.Stat(commonDir); os.IsNotExist(err) {
		return fmt.Errorf("common directory not found: %s", commonDir)
	}
	if _, err := os.Stat(serverDir); os.IsNotExist(err) {
		return fmt.Errorf("server directory not found: %s", serverDir)
	}
	if _, err := os.Stat(ragDir); os.IsNotExist(err) {
		return fmt.Errorf("rag directory not found: %s", ragDir)
	}
	if _, err := os.Stat(universalRuntimeDir); os.IsNotExist(err) {
		return fmt.Errorf("universal-runtime directory not found: %s", universalRuntimeDir)
	}

	// First, sync config and common (which are dependencies of server and rag)
	utils.LogDebug("Syncing config dependencies...")
	if err := m.syncDirectory(configDir, "config", false); err != nil {
		return fmt.Errorf("failed to sync config dependencies: %w", err)
	}
	if err := m.installHardwareWheels(configDir, "config"); err != nil {
		return err
	}

	utils.LogDebug("Syncing common dependencies...")
	if err := m.syncDirectory(commonDir, "common", false); err != nil {
		return fmt.Errorf("failed to sync common dependencies: %w", err)
	}
	if err := m.installHardwareWheels(commonDir, "common"); err != nil {
		return err
	}

	// Now sync server, rag, and universal-runtime in parallel
	var wg sync.WaitGroup
	var serverErr, ragErr, universalRuntimeErr error

	wg.Add(3)

	// Sync server dependencies
	go func() {
		defer wg.Done()
		utils.LogDebug("Syncing server dependencies...")
		serverErr = m.syncDirectory(serverDir, "server", false)
		if serverErr == nil {
			serverErr = m.installHardwareWheels(serverDir, "server")
		}
	}()

	// Sync rag dependencies
	go func() {
		defer wg.Done()
		utils.LogDebug("Syncing RAG dependencies...")
		ragErr = m.syncDirectory(ragDir, "rag", false)
		if ragErr == nil {
			ragErr = m.installHardwareWheels(ragDir, "rag")
		}
	}()

	// Sync universal-runtime dependencies (do NOT use PyTorch index for uv sync)
	// Hardware-specific wheels will be installed separately after sync
	go func() {
		defer wg.Done()
		utils.LogDebug("Syncing universal-runtime base dependencies...")
		universalRuntimeErr = m.syncDirectory(universalRuntimeDir, "universal-runtime", false)
		if universalRuntimeErr == nil {
			universalRuntimeErr = m.installHardwareWheels(universalRuntimeDir, "universal-runtime")
		}
	}()

	wg.Wait()

	// Check for errors
	if serverErr != nil {
		return fmt.Errorf("failed to sync server dependencies: %w", serverErr)
	}
	if ragErr != nil {
		return fmt.Errorf("failed to sync rag dependencies: %w", ragErr)
	}
	if universalRuntimeErr != nil {
		return fmt.Errorf("failed to sync universal-runtime dependencies: %w", universalRuntimeErr)
	}

	utils.LogDebug("Dependencies synced successfully")
	return nil
}

// installHardwareWheels installs hardware-specific binary wheels for a component
// This must be called after syncDirectory to ensure base dependencies are installed first
// If the component has no hardware-specific packages, this is a no-op
func (m *SourceManager) installHardwareWheels(dir string, componentName string) error {
	// Get the hardware-dependent packages for this component
	packages := GetComponentPackages(componentName)

	// Install llama.cpp binaries for universal-runtime
	// This downloads pre-built binaries to a cache directory that the Python package can find
	if componentName == "universal-runtime" {
		utils.LogDebug("Ensuring llama.cpp binaries are installed...")
		if _, err := EnsureLlamaBinary(); err != nil {
			return fmt.Errorf("failed to install llama.cpp binaries: %w", err)
		}
		utils.LogDebug("llama.cpp binaries ready")
	}

	// No-op if no hardware-specific packages are defined
	if len(packages) == 0 {
		utils.LogDebug(fmt.Sprintf("No hardware-specific packages for %s, skipping wheel installation", componentName))
		// Still need to create marker file for universal-runtime even if no packages
		if componentName == "universal-runtime" {
			markerFile := filepath.Join(dir, ".hardware-wheels-installed")
			if err := os.WriteFile(markerFile, []byte("installed"), 0644); err != nil {
				utils.LogDebug(fmt.Sprintf("Warning: could not create hardware wheels marker: %v", err))
			}
		}
		return nil
	}

	utils.LogDebug(fmt.Sprintf("Installing hardware-specific binary wheels for %s...", componentName))

	// Install the packages with hardware-specific wheels
	if err := InstallHardwarePackages(m.pythonEnvMgr, dir, packages); err != nil {
		return fmt.Errorf("failed to install hardware-specific wheels for %s: %w", componentName, err)
	}

	utils.LogDebug(fmt.Sprintf("Hardware-specific wheels for %s installed successfully", componentName))

	// Create marker file to indicate hardware wheels have been installed
	// This allows areDependenciesSynced to skip re-syncing when not needed
	if componentName == "universal-runtime" {
		markerFile := filepath.Join(dir, ".hardware-wheels-installed")
		if err := os.WriteFile(markerFile, []byte("installed"), 0644); err != nil {
			utils.LogDebug(fmt.Sprintf("Warning: could not create hardware wheels marker: %v", err))
		}
	}

	return nil
}

// syncDirectory runs `uv sync` in a specific directory
// keepPyTorchIndex controls whether UV_EXTRA_INDEX_URL should be preserved
func (m *SourceManager) syncDirectory(dir string, name string, keepPyTorchIndex bool) error {
	uvPath := m.pythonEnvMgr.uvManager.GetUVPath()

	// Verify the directory exists and contains pyproject.toml
	pyprojectPath := filepath.Join(dir, "pyproject.toml")
	if _, err := os.Stat(pyprojectPath); os.IsNotExist(err) {
		return fmt.Errorf("pyproject.toml not found in %s", dir)
	}

	// Run UV sync command in the specific project directory
	// This ensures .venv is created in the correct location
	// Use --no-install-workspace to prevent UV from installing workspace members, which can
	// cause hardware-specific packages like llamafarm-llama to be built from source during
	// config/common sync instead of being installed via the CLI's hardware detection
	// Use --no-group dev to skip dev dependencies (only install main dependencies)
	cmd := exec.Command(uvPath, "sync", "--managed-python", "--no-install-workspace", "--frozen", "--no-group", "dev")
	cmd.Dir = dir // Critical: run from project directory so .venv is created there

	// Get base environment
	env := m.pythonEnvMgr.GetEnvForProcess()

	// Filter out UV_EXTRA_INDEX_URL unless this component needs PyTorch
	// The PyTorch index should only be used by universal-runtime, not by server/rag/config/common
	// Using the PyTorch index for server/rag causes issues because it has incomplete/incompatible
	// versions of common packages (e.g., markupsafe 3.0.2 only for cp313, requests 2.28.1)
	if !keepPyTorchIndex {
		filteredEnv := make([]string, 0, len(env))
		for _, e := range env {
			if !strings.HasPrefix(e, "UV_EXTRA_INDEX_URL=") {
				filteredEnv = append(filteredEnv, e)
			}
		}
		env = filteredEnv
	}

	cmd.Env = env

	utils.LogDebug(fmt.Sprintf("Running 'uv sync' in directory: %s", dir))

	// Capture output for debugging
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("uv sync failed in %s: %w\nOutput: %s", name, err, string(output))
	}

	utils.LogDebug(fmt.Sprintf("%s uv sync output: %s", name, string(output)))

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
	configVenv := filepath.Join(m.srcDir, "config", ".venv")
	commonVenv := filepath.Join(m.srcDir, "common", ".venv")
	serverVenv := filepath.Join(m.srcDir, "server", ".venv")
	ragVenv := filepath.Join(m.srcDir, "rag", ".venv")
	universalRuntimeVenv := filepath.Join(m.srcDir, "runtimes", "universal", ".venv")

	_, configErr := os.Stat(configVenv)
	_, commonErr := os.Stat(commonVenv)
	_, serverErr := os.Stat(serverVenv)
	_, ragErr := os.Stat(ragVenv)
	_, universalRuntimeErr := os.Stat(universalRuntimeVenv)

	venvsSynced := configErr == nil && commonErr == nil && serverErr == nil && ragErr == nil && universalRuntimeErr == nil

	// Also check for hardware wheels marker file (for universal-runtime)
	// This ensures hardware-specific packages like torch and llamafarm-llama are installed
	hardwareWheelsMarker := filepath.Join(m.srcDir, "runtimes", "universal", ".hardware-wheels-installed")
	_, hardwareWheelsErr := os.Stat(hardwareWheelsMarker)

	return venvsSynced && hardwareWheelsErr == nil
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

// VerifyDesignerBuild checks if designer/dist/index.html exists
func (m *SourceManager) VerifyDesignerBuild() error {
	designerDir := m.GetDesignerDir()
	indexPath := filepath.Join(designerDir, "dist", "index.html")

	if _, err := os.Stat(indexPath); err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("designer build not found at %s - designer static files must be built before starting server", indexPath)
		}
		return fmt.Errorf("failed to check designer build at %s: %w", indexPath, err)
	}

	return nil
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
// It checks if datamodel.py exists and is up-to-date to avoid unnecessary regeneration
func (m *SourceManager) GenerateDatamodel() error {
	configDir := m.GetConfigDir()

	// Check if config directory exists
	if _, err := os.Stat(configDir); os.IsNotExist(err) {
		return fmt.Errorf("config directory not found: %s", configDir)
	}

	generateScript := filepath.Join(configDir, "generate_types.py")
	if _, err := os.Stat(generateScript); os.IsNotExist(err) {
		return fmt.Errorf("generate_types.py not found, skipping datamodel generation: %w", err)
	}

	// Check if datamodel.py already exists and is up-to-date
	datamodelPath := filepath.Join(configDir, "datamodel.py")
	schemaPath := filepath.Join(configDir, "schema.yaml")
	ragSchemaPath := filepath.Join(m.srcDir, "rag", "schema.yaml")

	datamodelExists := false
	var datamodelModTime time.Time
	if info, err := os.Stat(datamodelPath); err == nil {
		datamodelExists = true
		datamodelModTime = info.ModTime()
	}

	// Check if schema files are newer than datamodel.py
	needsRegeneration := !datamodelExists

	if datamodelExists {
		// Check schema.yaml modification time
		if schemaInfo, err := os.Stat(schemaPath); err == nil {
			if schemaInfo.ModTime().After(datamodelModTime) {
				needsRegeneration = true
			}
		}

		// Check rag/schema.yaml modification time
		if !needsRegeneration {
			if ragSchemaInfo, err := os.Stat(ragSchemaPath); err == nil {
				if ragSchemaInfo.ModTime().After(datamodelModTime) {
					needsRegeneration = true
				}
			}
		}

		// Check compile_schema.py modification time (it's part of the generation process)
		if !needsRegeneration {
			compileScriptPath := filepath.Join(configDir, "compile_schema.py")
			if compileInfo, err := os.Stat(compileScriptPath); err == nil {
				if compileInfo.ModTime().After(datamodelModTime) {
					needsRegeneration = true
				}
			}
		}
	}

	// Skip generation if datamodel is up-to-date
	if !needsRegeneration {
		utils.LogDebug("Datamodel is up-to-date, skipping generation")
		return nil
	}

	uvPath := m.pythonEnvMgr.uvManager.GetUVPath()

	// Run the generation script
	// Use --no-sync to avoid attempting to sync dependencies during run
	cmd := exec.Command(uvPath, "run", "--no-sync", "--managed-python", "python", "generate_types.py")
	cmd.Dir = configDir
	cmd.Env = m.pythonEnvMgr.GetEnvForProcess()

	utils.LogDebug(fmt.Sprintf("Running datamodel generation in: %s", configDir))

	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("datamodel generation failed: %w\nOutput: %s", err, string(output))
	}

	utils.LogDebug(fmt.Sprintf("Datamodel generation output: %s", string(output)))

	utils.LogDebug("Config datamodel generated successfully\n")
	return nil
}
