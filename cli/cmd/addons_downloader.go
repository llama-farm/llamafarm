package cmd

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/llamafarm/cli/cmd/utils"
	"github.com/llamafarm/cli/internal/buildinfo"
)

type AddonDownloader struct {
	version string // LlamaFarm version for downloading wheels
	client  *http.Client
}

func NewAddonDownloader(version string) *AddonDownloader {
	// Resolve version to use (same logic as BinaryManager)
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
	return &AddonDownloader{
		version: resolvedVersion,
		client:  &http.Client{Timeout: 10 * time.Minute},
	}
}

// DownloadAndInstallAddon downloads the addon wheel bundle and extracts it
func (d *AddonDownloader) DownloadAndInstallAddon(addon *AddonDefinition) error {
	utils.OutputInfo("Downloading %s addon...\n", addon.DisplayName)

	// Build download URL
	platform := getPlatformString()
	filename := fmt.Sprintf("%s-wheels-%s.tar.gz", addon.Name, platform)
	url := d.buildDownloadURL(filename)
	checksumURL := d.buildDownloadURL(filename + ".sha256")

	utils.LogDebug(fmt.Sprintf("Download URL: %s", url))

	// Download to temp file
	tempFile, err := os.CreateTemp("", fmt.Sprintf("addon-%s-*.tar.gz", addon.Name))
	if err != nil {
		return fmt.Errorf("failed to create temp file: %w", err)
	}
	defer os.Remove(tempFile.Name())
	defer tempFile.Close()

	if err := d.downloadFile(url, tempFile); err != nil {
		return fmt.Errorf("failed to download: %w", err)
	}

	// Download and verify checksum
	utils.OutputInfo("Verifying checksum...\n")
	if err := d.verifyChecksum(tempFile.Name(), checksumURL); err != nil {
		return fmt.Errorf("checksum verification failed: %w", err)
	}

	// Extract to addons directory
	addonsDir, err := getAddonsDir()
	if err != nil {
		return err
	}

	addonPath := filepath.Join(addonsDir, addon.Name)
	if err := os.MkdirAll(addonPath, 0755); err != nil {
		return fmt.Errorf("failed to create addon directory: %w", err)
	}

	utils.OutputInfo("Extracting addon...\n")
	if err := d.extractTarGz(tempFile.Name(), addonPath); err != nil {
		return fmt.Errorf("failed to extract: %w", err)
	}

	// Extract wheel files so Python can import the packages
	utils.OutputInfo("Installing packages...\n")
	if err := d.extractWheelFiles(addonPath); err != nil {
		return fmt.Errorf("failed to install packages: %w", err)
	}

	// Remove common packages that would conflict with venv dependencies
	// Only keep addon-specific packages
	utils.OutputInfo("Cleaning up dependencies...\n")
	if err := d.removeCommonPackages(addonPath, addon); err != nil {
		// Log warning but don't fail - addon might still work
		utils.LogDebug(fmt.Sprintf("Warning: failed to clean up dependencies: %v", err))
	}

	utils.OutputSuccess("Addon %s downloaded successfully\n", addon.DisplayName)
	return nil
}

func (d *AddonDownloader) buildDownloadURL(filename string) string {
	// Download from GitHub releases
	// https://github.com/llama-farm/llamafarm/releases/download/v0.0.26/stt-wheels-macos-arm64.tar.gz

	// Allow overriding owner/repo via environment variables (useful for forks/private deployments)
	owner := os.Getenv("LF_ADDON_REPO_OWNER")
	if owner == "" {
		owner = "llama-farm"
	}

	repo := os.Getenv("LF_ADDON_REPO_NAME")
	if repo == "" {
		repo = "llamafarm"
	}

	// Allow overriding via environment variable for testing
	// LF_ADDON_RELEASE_TAG=v0.0.27-snapshot lf addons install stt
	if envTag := os.Getenv("LF_ADDON_RELEASE_TAG"); envTag != "" {
		utils.LogDebug(fmt.Sprintf("Using addon release tag from LF_ADDON_RELEASE_TAG: %s", envTag))
		return fmt.Sprintf("https://github.com/%s/%s/releases/download/%s/%s", owner, repo, envTag, filename)
	}

	if d.version == "latest" {
		return fmt.Sprintf("https://github.com/%s/%s/releases/latest/download/%s", owner, repo, filename)
	}
	return fmt.Sprintf("https://github.com/%s/%s/releases/download/v%s/%s", owner, repo, d.version, filename)
}

func (d *AddonDownloader) downloadFile(url string, dest *os.File) error {
	resp, err := d.client.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download failed: %s", resp.Status)
	}

	_, err = io.Copy(dest, resp.Body)
	return err
}

// verifyChecksum downloads the checksum file and verifies the downloaded file
func (d *AddonDownloader) verifyChecksum(filePath, checksumURL string) error {
	// Download checksum file
	resp, err := d.client.Get(checksumURL)
	if err != nil {
		return fmt.Errorf("failed to download checksum: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("checksum download failed: %s", resp.Status)
	}

	// Read expected checksum
	checksumData, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read checksum: %w", err)
	}

	// Parse checksum (format: "hash  filename" or just "hash")
	checksumStr := strings.TrimSpace(string(checksumData))
	fields := strings.Fields(checksumStr)
	if len(fields) == 0 {
		return fmt.Errorf("invalid checksum file format")
	}
	expectedChecksum := fields[0]

	// Calculate actual checksum
	file, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("failed to open file for checksum: %w", err)
	}
	defer file.Close()

	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return fmt.Errorf("failed to calculate checksum: %w", err)
	}
	actualChecksum := hex.EncodeToString(hash.Sum(nil))

	// Compare checksums
	if actualChecksum != expectedChecksum {
		return fmt.Errorf("checksum mismatch: expected %s, got %s", expectedChecksum, actualChecksum)
	}

	utils.LogDebug(fmt.Sprintf("Checksum verified: %s", actualChecksum))
	return nil
}

// extractWheelFiles extracts all .whl files in a directory
// Wheel files are ZIP archives - we extract them so Python can import the packages
func (d *AddonDownloader) extractWheelFiles(addonDir string) error {
	// Find all .whl files
	files, err := os.ReadDir(addonDir)
	if err != nil {
		return err
	}

	for _, file := range files {
		if !file.IsDir() && filepath.Ext(file.Name()) == ".whl" {
			wheelPath := filepath.Join(addonDir, file.Name())
			utils.LogDebug(fmt.Sprintf("Extracting wheel: %s", file.Name()))

			// Extract wheel (it's a ZIP archive)
			if err := d.extractWheel(wheelPath, addonDir); err != nil {
				return fmt.Errorf("failed to extract %s: %w", file.Name(), err)
			}

			// Remove the wheel file after extraction
			if err := os.Remove(wheelPath); err != nil {
				utils.LogDebug(fmt.Sprintf("Warning: failed to remove wheel file %s: %v", file.Name(), err))
			}
		}
	}

	return nil
}

// extractWheel extracts a wheel file (ZIP archive) to the destination directory
func (d *AddonDownloader) extractWheel(wheelPath, destDir string) error {
	file, err := os.Open(wheelPath)
	if err != nil {
		return err
	}
	defer file.Close()

	// Get file size for ZIP reader
	stat, err := file.Stat()
	if err != nil {
		return err
	}

	// Create ZIP reader
	zipReader, err := zip.NewReader(file, stat.Size())
	if err != nil {
		return err
	}

	// Resolve destination directory to absolute path for security checks
	absDestDir, err := filepath.Abs(destDir)
	if err != nil {
		return fmt.Errorf("failed to resolve destination directory: %w", err)
	}

	// Extract each file from the ZIP
	for _, zipFile := range zipReader.File {
		// Security: Prevent path traversal attacks
		cleanName := filepath.Clean(zipFile.Name)
		if strings.Contains(cleanName, "..") {
			return fmt.Errorf("illegal path in wheel: %s", zipFile.Name)
		}

		target := filepath.Join(destDir, cleanName)

		// Verify the resolved path is within destDir
		absTarget, err := filepath.Abs(target)
		if err != nil {
			return fmt.Errorf("failed to resolve target path: %w", err)
		}
		if !strings.HasPrefix(absTarget, absDestDir+string(os.PathSeparator)) && absTarget != absDestDir {
			return fmt.Errorf("illegal path in wheel (would extract outside destination): %s", zipFile.Name)
		}

		if zipFile.FileInfo().IsDir() {
			// Create directory
			if err := os.MkdirAll(target, 0755); err != nil {
				return err
			}
		} else {
			// Create parent directories
			if err := os.MkdirAll(filepath.Dir(target), 0755); err != nil {
				return err
			}

			// Extract file
			outFile, err := os.OpenFile(target, os.O_CREATE|os.O_RDWR|os.O_TRUNC, zipFile.Mode())
			if err != nil {
				return err
			}

			rc, err := zipFile.Open()
			if err != nil {
				outFile.Close()
				return err
			}

			_, err = io.Copy(outFile, rc)
			rc.Close()
			outFile.Close()
			if err != nil {
				return err
			}
		}
	}

	return nil
}

// removeCommonPackages removes packages that are likely already in the venv and would cause conflicts
// Keeps only the core addon packages specified in the addon definition
func (d *AddonDownloader) removeCommonPackages(addonDir string, addon *AddonDefinition) error {
	// Extract primary package names from the addon's package list
	// e.g., "faster-whisper>=1.0.0" -> "faster_whisper"
	keepPackages := make(map[string]bool)
	for _, pkg := range addon.Packages {
		// Extract package name before version specifiers
		pkgName := strings.Split(pkg, ">=")[0]
		pkgName = strings.Split(pkgName, "==")[0]
		pkgName = strings.Split(pkgName, "<")[0]
		pkgName = strings.Split(pkgName, ">")[0]
		pkgName = strings.TrimSpace(pkgName)
		// Convert to module name format (hyphens to underscores)
		pkgName = strings.ReplaceAll(pkgName, "-", "_")
		keepPackages[pkgName] = true
		utils.LogDebug(fmt.Sprintf("Keeping addon package: %s", pkgName))
	}

	// Also keep packages that are direct dependencies of the main packages
	// For faster-whisper, we need ctranslate2 which might not be in the venv
	keepPackages["ctranslate2"] = true

	// List of common packages to remove (already in venv)
	removePatterns := []string{
		"huggingface_hub", "numpy", "torch", "transformers",
		"tokenizers", "packaging", "filelock", "typing_extensions",
		"certifi", "idna", "charset_normalizer", "urllib3",
		"requests", "tqdm", "pyyaml", "click", "setuptools",
		"sympy", "mpmath", "onnxruntime", "protobuf",
		"h11", "httpcore", "httpx", "anyio", "sniffio",
		"coloredlogs", "humanfriendly", "flatbuffers",
		"fsspec", "shellingham", "typer", "hf_xet",
		"google", "yaml", "_yaml", "_distutils_hack",
		"pkg_resources",
	}

	files, err := os.ReadDir(addonDir)
	if err != nil {
		return err
	}

	for _, file := range files {
		if !file.IsDir() {
			continue
		}

		// Skip .dist-info directories
		if strings.HasSuffix(file.Name(), ".dist-info") || strings.HasSuffix(file.Name(), ".data") {
			continue
		}

		// Check if this package should be kept
		shouldKeep := keepPackages[file.Name()]
		if shouldKeep {
			continue
		}

		// Check if it matches a remove pattern
		shouldRemove := false
		for _, pattern := range removePatterns {
			if file.Name() == pattern || strings.HasPrefix(file.Name(), pattern+"_") {
				shouldRemove = true
				break
			}
		}

		if shouldRemove {
			dirPath := filepath.Join(addonDir, file.Name())
			utils.LogDebug(fmt.Sprintf("Removing common package: %s", file.Name()))
			if err := os.RemoveAll(dirPath); err != nil {
				utils.LogDebug(fmt.Sprintf("Warning: failed to remove %s: %v", file.Name(), err))
			}
		}
	}

	// Also remove .dist-info, .data directories, and standalone files for removed packages
	files2, _ := os.ReadDir(addonDir)
	for _, file := range files2 {
		name := file.Name()

		// Remove .dist-info and .data directories for common packages
		if strings.HasSuffix(name, ".dist-info") || strings.HasSuffix(name, ".data") {
			baseName := strings.TrimSuffix(name, ".dist-info")
			baseName = strings.TrimSuffix(baseName, ".data")
			// Extract package name from dist-info (e.g., "numpy-2.4.2.dist-info" -> "numpy")
			parts := strings.Split(baseName, "-")
			if len(parts) > 0 {
				pkgName := parts[0]
				// Check if this is a removed package
				for _, pattern := range removePatterns {
					if pkgName == pattern || strings.HasPrefix(pkgName, pattern+"_") {
						dirPath := filepath.Join(addonDir, name)
						utils.LogDebug(fmt.Sprintf("Removing metadata: %s", name))
						os.RemoveAll(dirPath)
						break
					}
				}
			}
		}

		// Remove standalone files (.py, .pth)
		if !file.IsDir() && (strings.HasSuffix(name, ".py") || strings.HasSuffix(name, ".pth")) {
			// Check if it's a removed package file
			baseName := strings.TrimSuffix(name, ".py")
			baseName = strings.TrimSuffix(baseName, ".pth")
			for _, pattern := range removePatterns {
				if baseName == pattern || strings.HasPrefix(baseName, pattern+"_") {
					filePath := filepath.Join(addonDir, name)
					utils.LogDebug(fmt.Sprintf("Removing file: %s", name))
					os.Remove(filePath)
					break
				}
			}
		}
	}

	return nil
}

func (d *AddonDownloader) extractTarGz(tarGzPath, destDir string) error {
	file, err := os.Open(tarGzPath)
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

	// Resolve destination directory to absolute path for security checks
	absDestDir, err := filepath.Abs(destDir)
	if err != nil {
		return fmt.Errorf("failed to resolve destination directory: %w", err)
	}

	for {
		header, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		// Security: Prevent path traversal attacks
		// Clean the name to remove any ".." components
		cleanName := filepath.Clean(header.Name)
		if strings.Contains(cleanName, "..") {
			return fmt.Errorf("illegal path in archive: %s", header.Name)
		}

		target := filepath.Join(destDir, cleanName)

		// Verify the resolved path is within destDir
		absTarget, err := filepath.Abs(target)
		if err != nil {
			return fmt.Errorf("failed to resolve target path: %w", err)
		}
		if !strings.HasPrefix(absTarget, absDestDir+string(os.PathSeparator)) && absTarget != absDestDir {
			return fmt.Errorf("illegal path in archive (would extract outside destination): %s", header.Name)
		}

		switch header.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0755); err != nil {
				return err
			}
		case tar.TypeReg:
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
		default:
			// Ignore symlinks and other special file types for security
			utils.LogDebug(fmt.Sprintf("Skipping unsupported file type in archive: %s (type: %c)", header.Name, header.Typeflag))
		}
	}

	return nil
}
