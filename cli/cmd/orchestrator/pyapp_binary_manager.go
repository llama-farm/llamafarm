package orchestrator

import (
	"archive/zip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/llamafarm/cli/cmd/utils"
	"github.com/llamafarm/cli/internal/buildinfo"
)

// BinaryManager handles downloading and managing PyApp component binaries
type BinaryManager struct {
	version string
}

// NewBinaryManager creates a new binary manager
func NewBinaryManager(version string) (*BinaryManager, error) {
	// Resolve version to use
	resolvedVersion := version
	if resolvedVersion == "" || resolvedVersion == "dev" {
		// For dev builds, try to use the CLI version
		cliVersion := buildinfo.CurrentVersion
		if cliVersion != "" && cliVersion != "dev" {
			resolvedVersion = cliVersion
		} else {
			// Fall back to latest release
			resolvedVersion = "latest"
		}
	}

	return &BinaryManager{
		version: resolvedVersion,
	}, nil
}

// EnsureBinaries ensures all required PyApp binaries are downloaded
func (m *BinaryManager) EnsureBinaries() error {
	utils.LogDebug(fmt.Sprintf("Ensuring PyApp binaries for version %s\n", m.version))

	// Check if binaries already exist and match version
	if m.areBinariesInstalled() {
		utils.LogDebug("PyApp binaries already installed\n")
		return nil
	}

	utils.LogDebug("Downloading PyApp binaries...\n")

	// Download each component binary
	components := []string{"server", "rag", "universal-runtime"}
	for _, component := range components {
		if err := m.DownloadBinary(component); err != nil {
			return fmt.Errorf("failed to download %s binary: %w", component, err)
		}
	}

	// Write version file to track installed version
	if err := m.writeInstalledBinariesVersion(); err != nil {
		utils.LogDebug(fmt.Sprintf("Warning: failed to write version file: %v\n", err))
		// Don't fail the whole operation if we can't write the version file
	}

	utils.LogDebug("All PyApp binaries downloaded successfully\n")
	return nil
}

// DownloadBinary downloads a specific component binary from GitHub releases or artifacts
func (m *BinaryManager) DownloadBinary(component string) error {
	// Check if we should download from artifacts (E2E test mode)
	if os.Getenv("LF_BINARY_SOURCE") == "artifact" {
		return m.downloadBinaryFromArtifact(component)
	}

	// Default: download from GitHub releases
	return m.downloadBinaryFromRelease(component)
}

// downloadBinaryFromRelease downloads a binary from GitHub releases (production mode)
func (m *BinaryManager) downloadBinaryFromRelease(component string) error {
	// Map component names to binary names
	binaryName, ok := serviceBinaryNames[component]
	if !ok {
		return fmt.Errorf("unknown component: %s", component)
	}

	// Get platform suffix
	platformSuffix := getPlatformSuffix()

	// Build binary filename
	ext := ""
	if runtime.GOOS == "windows" {
		ext = ".exe"
	}
	filename := fmt.Sprintf("%s-%s%s", binaryName, platformSuffix, ext)

	// Build download URL
	downloadURL := m.buildDownloadURL(filename)
	utils.LogDebug(fmt.Sprintf("Downloading %s from %s\n", filename, downloadURL))

	return m.downloadAndInstallBinary(downloadURL, filename, nil)
}

// downloadBinaryFromArtifact downloads a binary from GitHub Actions artifacts (E2E test mode)
func (m *BinaryManager) downloadBinaryFromArtifact(component string) error {
	runID := os.Getenv("LF_ARTIFACT_RUN_ID")
	if runID == "" {
		return fmt.Errorf("LF_ARTIFACT_RUN_ID must be set when LF_BINARY_SOURCE=artifact")
	}

	token := os.Getenv("GITHUB_TOKEN")
	if token == "" {
		return fmt.Errorf("GITHUB_TOKEN must be set when LF_BINARY_SOURCE=artifact")
	}

	// Map component names to binary names
	binaryName, ok := serviceBinaryNames[component]
	if !ok {
		return fmt.Errorf("unknown component: %s", component)
	}

	// Get platform suffix
	platformSuffix := getPlatformSuffix()

	// Build artifact name pattern (matches what pyapp.yml uploads)
	// Extract the component part from the binary name (e.g., "llamafarm-runtime" -> "runtime")
	// Artifact format: llamafarm-{binary-component}-pyapp-{platform}
	binaryComponent := strings.TrimPrefix(binaryName, "llamafarm-")
	artifactName := fmt.Sprintf("llamafarm-%s-pyapp-%s", binaryComponent, platformSuffix)

	utils.LogDebug(fmt.Sprintf("Downloading %s binary from artifact %s (run %s)\n", component, artifactName, runID))

	// Get artifact download URL
	artifactURL, err := m.getArtifactDownloadURL(runID, artifactName, token)
	if err != nil {
		return fmt.Errorf("failed to get artifact URL: %w", err)
	}

	// Build expected binary filename
	ext := ""
	if runtime.GOOS == "windows" {
		ext = ".exe"
	}
	filename := fmt.Sprintf("%s-%s%s", binaryName, platformSuffix, ext)

	// Download artifact (it's a zip containing the binary)
	headers := map[string]string{"Authorization": "Bearer " + token}
	return m.downloadAndInstallBinaryFromZip(artifactURL, filename, headers)
}

// getArtifactDownloadURL queries GitHub API to get the download URL for an artifact
func (m *BinaryManager) getArtifactDownloadURL(runID, artifactName, token string) (string, error) {
	// Query GitHub API for artifacts from this run
	apiURL := fmt.Sprintf("https://api.github.com/repos/%s/%s/actions/runs/%s/artifacts",
		githubOwner, githubRepo, runID)

	req, err := http.NewRequest("GET", apiURL, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Accept", "application/vnd.github+json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("GitHub API returned status %d", resp.StatusCode)
	}

	// Parse artifacts list
	var result struct {
		Artifacts []struct {
			Name               string `json:"name"`
			ArchiveDownloadURL string `json:"archive_download_url"`
		} `json:"artifacts"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("failed to decode artifacts response: %w", err)
	}

	// Find matching artifact
	for _, artifact := range result.Artifacts {
		if artifact.Name == artifactName {
			return artifact.ArchiveDownloadURL, nil
		}
	}

	return "", fmt.Errorf("artifact %s not found in run %s", artifactName, runID)
}

// downloadAndInstallBinary downloads a binary file and installs it to the bin directory
func (m *BinaryManager) downloadAndInstallBinary(downloadURL, filename string, headers map[string]string) error {
	// Get bin directory
	binDir, err := GetBinDir()
	if err != nil {
		return err
	}

	// Create bin directory if it doesn't exist
	if err := os.MkdirAll(binDir, 0755); err != nil {
		return fmt.Errorf("failed to create bin directory: %w", err)
	}

	// Download to temp file in binDir to ensure os.Rename works (same filesystem)
	tmpFile, err := os.CreateTemp(binDir, ".pyapp-*.tmp")
	if err != nil {
		return fmt.Errorf("failed to create temp file: %w", err)
	}
	defer os.Remove(tmpFile.Name())
	defer tmpFile.Close()

	// Create request with optional headers
	req, err := http.NewRequest("GET", downloadURL, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	for key, value := range headers {
		req.Header.Set(key, value)
	}

	// Download the binary
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to download: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download failed with status %d", resp.StatusCode)
	}

	// Write to temp file and compute checksum
	hash := sha256.New()
	writer := io.MultiWriter(tmpFile, hash)
	written, err := io.Copy(writer, resp.Body)
	if err != nil {
		return fmt.Errorf("failed to download: %w", err)
	}
	utils.LogDebug(fmt.Sprintf("Downloaded %d bytes\n", written))

	// Close temp file before moving it
	tmpFile.Close()

	// Move to final location
	destPath := filepath.Join(binDir, filename)

	// On Windows, os.Rename fails if destination exists. Remove it first.
	// On Unix, os.Rename atomically replaces the destination, so no removal needed.
	if runtime.GOOS == "windows" {
		if _, err := os.Stat(destPath); err == nil {
			if err := os.Remove(destPath); err != nil {
				return fmt.Errorf("failed to remove existing binary at %s: %w", destPath, err)
			}
		}
	}

	if err := os.Rename(tmpFile.Name(), destPath); err != nil {
		return fmt.Errorf("failed to move binary to %s: %w", destPath, err)
	}

	// Make executable on Unix
	if runtime.GOOS != "windows" {
		if err := os.Chmod(destPath, 0755); err != nil {
			return fmt.Errorf("failed to make binary executable: %w", err)
		}
	}

	checksumStr := hex.EncodeToString(hash.Sum(nil))
	utils.LogDebug(fmt.Sprintf("Installed %s (SHA256: %s)\n", filename, checksumStr[:16]))

	return nil
}

// downloadAndInstallBinaryFromZip downloads a zip artifact, extracts the binary, and installs it
func (m *BinaryManager) downloadAndInstallBinaryFromZip(downloadURL, filename string, headers map[string]string) error {
	// Download artifact zip to temp file
	tmpZip, err := os.CreateTemp("", "artifact-*.zip")
	if err != nil {
		return fmt.Errorf("failed to create temp file: %w", err)
	}
	defer os.Remove(tmpZip.Name())
	defer tmpZip.Close()

	// Create request with headers
	req, err := http.NewRequest("GET", downloadURL, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	for key, value := range headers {
		req.Header.Set(key, value)
	}

	// Download the zip
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to download artifact: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("artifact download failed with status %d", resp.StatusCode)
	}

	written, err := io.Copy(tmpZip, resp.Body)
	if err != nil {
		return fmt.Errorf("failed to download artifact: %w", err)
	}
	tmpZip.Close()
	utils.LogDebug(fmt.Sprintf("Downloaded artifact zip (%d bytes)\n", written))

	// Open and extract the binary from the zip
	zipReader, err := zip.OpenReader(tmpZip.Name())
	if err != nil {
		return fmt.Errorf("failed to open artifact zip: %w", err)
	}
	defer zipReader.Close()

	// Find the binary in the zip
	var binaryFile *zip.File
	for _, f := range zipReader.File {
		if f.Name == filename {
			binaryFile = f
			break
		}
	}

	if binaryFile == nil {
		return fmt.Errorf("binary %s not found in artifact zip", filename)
	}

	// Extract the binary
	binDir, err := GetBinDir()
	if err != nil {
		return err
	}

	if err := os.MkdirAll(binDir, 0755); err != nil {
		return fmt.Errorf("failed to create bin directory: %w", err)
	}

	// Extract to temp file in binDir for atomic installation
	tmpFile, err := os.CreateTemp(binDir, ".pyapp-*.tmp")
	if err != nil {
		return fmt.Errorf("failed to create temp file: %w", err)
	}
	tmpPath := tmpFile.Name()
	defer os.Remove(tmpPath)
	defer tmpFile.Close()

	// Open the file in the zip
	rc, err := binaryFile.Open()
	if err != nil {
		return fmt.Errorf("failed to open binary in zip: %w", err)
	}
	defer rc.Close()

	// Extract with checksum
	hash := sha256.New()
	writer := io.MultiWriter(tmpFile, hash)
	extracted, err := io.Copy(writer, rc)
	if err != nil {
		return fmt.Errorf("failed to extract binary: %w", err)
	}
	tmpFile.Close()

	// Make executable on Unix
	if runtime.GOOS != "windows" {
		if err := os.Chmod(tmpPath, 0755); err != nil {
			return fmt.Errorf("failed to make binary executable: %w", err)
		}
	}

	// Atomic move to final location
	destPath := filepath.Join(binDir, filename)

	// On Windows, os.Rename fails if destination exists. Remove it first.
	// On Unix, os.Rename atomically replaces the destination, so no removal needed.
	if runtime.GOOS == "windows" {
		if _, err := os.Stat(destPath); err == nil {
			if err := os.Remove(destPath); err != nil {
				return fmt.Errorf("failed to remove existing binary at %s: %w", destPath, err)
			}
		}
	}

	if err := os.Rename(tmpPath, destPath); err != nil {
		return fmt.Errorf("failed to move binary to %s: %w", destPath, err)
	}

	checksumStr := hex.EncodeToString(hash.Sum(nil))
	utils.LogDebug(fmt.Sprintf("Extracted and installed %s (%d bytes, SHA256: %s)\n", filename, extracted, checksumStr[:16]))

	return nil
}

// buildDownloadURL constructs the GitHub release download URL
func (m *BinaryManager) buildDownloadURL(filename string) string {
	version := m.version
	if version == "latest" {
		// Use GitHub's latest release redirect
		return fmt.Sprintf("https://github.com/%s/%s/releases/latest/download/%s",
			githubOwner, githubRepo, filename)
	}

	// Ensure version has 'v' prefix for release tags
	if !strings.HasPrefix(version, "v") {
		version = "v" + version
	}

	return fmt.Sprintf("https://github.com/%s/%s/releases/download/%s/%s",
		githubOwner, githubRepo, version, filename)
}

// areBinariesInstalled checks if all required binaries are already present and match the CLI version
func (m *BinaryManager) areBinariesInstalled() bool {
	// Check if all binaries exist
	components := []string{"server", "rag", "universal-runtime"}
	for _, component := range components {
		if _, err := ResolveBinaryPath(component); err != nil {
			return false
		}
	}

	// Check if installed version matches CLI version
	installedVersion, err := m.getInstalledBinariesVersion()
	if err != nil {
		// No version file or can't read it - treat as not installed
		utils.LogDebug(fmt.Sprintf("Binary version check failed: %v\n", err))
		return false
	}

	if installedVersion != m.version {
		utils.LogDebug(fmt.Sprintf("Binary version mismatch: installed=%s, required=%s\n", installedVersion, m.version))
		return false
	}

	return true
}

// getInstalledBinariesVersion reads the version file from the bin directory
func (m *BinaryManager) getInstalledBinariesVersion() (string, error) {
	binDir, err := GetBinDir()
	if err != nil {
		return "", err
	}

	versionFile := filepath.Join(binDir, ".pyapp-version")
	data, err := os.ReadFile(versionFile)
	if err != nil {
		return "", err
	}

	return strings.TrimSpace(string(data)), nil
}

// writeInstalledBinariesVersion writes the version file after successful binary downloads
func (m *BinaryManager) writeInstalledBinariesVersion() error {
	binDir, err := GetBinDir()
	if err != nil {
		return err
	}

	versionFile := filepath.Join(binDir, ".pyapp-version")
	return os.WriteFile(versionFile, []byte(m.version+"\n"), 0644)
}
