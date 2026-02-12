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
	"regexp"
	"strings"
	"time"

	"github.com/llamafarm/cli/cmd/utils"
	"github.com/llamafarm/cli/internal/buildinfo"
)

// githubSlugPattern validates GitHub owner, repo, and tag segments.
// Allows alphanumeric, hyphens, underscores, and dots (standard GitHub naming).
var githubSlugPattern = regexp.MustCompile(`^[a-zA-Z0-9._-]+$`)

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
	// LIFO order: Close runs first, then Remove (so the file is closed before deletion).
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
	} else if !githubSlugPattern.MatchString(owner) {
		utils.OutputError("Invalid LF_ADDON_REPO_OWNER: %q (must be alphanumeric, hyphens, underscores, dots)\n", owner)
		owner = "llama-farm"
	}

	repo := os.Getenv("LF_ADDON_REPO_NAME")
	if repo == "" {
		repo = "llamafarm"
	} else if !githubSlugPattern.MatchString(repo) {
		utils.OutputError("Invalid LF_ADDON_REPO_NAME: %q (must be alphanumeric, hyphens, underscores, dots)\n", repo)
		repo = "llamafarm"
	}

	// Allow overriding via environment variable for testing
	// LF_ADDON_RELEASE_TAG=v0.0.27-snapshot lf addons install stt
	if envTag := os.Getenv("LF_ADDON_RELEASE_TAG"); envTag != "" {
		if !githubSlugPattern.MatchString(envTag) {
			utils.OutputError("Invalid LF_ADDON_RELEASE_TAG: %q (must be alphanumeric, hyphens, underscores, dots)\n", envTag)
		} else {
			utils.LogDebug(fmt.Sprintf("Using addon release tag from LF_ADDON_RELEASE_TAG: %s", envTag))
			return fmt.Sprintf("https://github.com/%s/%s/releases/download/%s/%s", owner, repo, envTag, filename)
		}
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
				// Handle close error to prevent silent data loss
				if closeErr := outFile.Close(); closeErr != nil {
					return fmt.Errorf("failed to open zip entry: %w; close also failed: %v", err, closeErr)
				}
				return err
			}

			_, err = io.Copy(outFile, rc)
			rc.Close()
			if err != nil {
				// Handle close error to prevent silent data loss
				if closeErr := outFile.Close(); closeErr != nil {
					return fmt.Errorf("copy failed: %w; close also failed: %v", err, closeErr)
				}
				return err
			}
			if err := outFile.Close(); err != nil {
				return err
			}
		}
	}

	return nil
}

// removeCommonPackages removes packages from the extracted addon directory that
// are already present in the component's venv. This prevents PYTHONPATH conflicts
// at runtime while keeping addon-specific transitive dependencies.
//
// The decision of what to remove is based on dynamic venv introspection rather
// than a hardcoded list, so it stays in sync automatically as venv dependencies
// change across releases.
func (d *AddonDownloader) removeCommonPackages(addonDir string, addon *AddonDefinition) error {
	// 1. Build the set of packages that must always be kept.
	keepPackages := extractAddonPackageNames(addon)

	// 2. Also keep any packages explicitly listed in keep_packages YAML field.
	//    This covers transitive dependencies the addon author knows are required
	//    and might not be in the venv (e.g., ctranslate2 for faster-whisper).
	for _, pkg := range addon.KeepPackages {
		normalized := normalizePackageName(pkg)
		keepPackages[normalized] = true
		utils.LogDebug(fmt.Sprintf("Keeping package (via keep_packages): %s", normalized))
	}

	// 3. Discover what's already installed in the component's venv.
	//    If the venv doesn't exist yet, this returns an empty map and we
	//    conservatively keep everything.
	venvPackages := getVenvPackageNames(addon.Component)

	// 4. Discover packages provided by other installed addons so we don't
	//    remove something a sibling addon depends on.
	otherAddonPackages := getInstalledAddonPackageNames(addon.Name)

	if len(venvPackages) == 0 {
		utils.LogDebug("Venv package list empty; skipping cleanup (conservative)")
		return nil
	}

	// 5. First pass: remove package directories that are already in the venv.
	files, err := os.ReadDir(addonDir)
	if err != nil {
		return err
	}

	removed := make(map[string]bool) // track removals for metadata cleanup

	for _, file := range files {
		if !file.IsDir() {
			continue
		}
		name := file.Name()
		normalized := normalizePackageName(name)

		if strings.HasSuffix(name, ".dist-info") || strings.HasSuffix(name, ".data") {
			continue
		}

		if keepPackages[normalized] {
			continue
		}
		if otherAddonPackages[name] || otherAddonPackages[normalized] {
			utils.LogDebug(fmt.Sprintf("Keeping %s (provided by another addon)", name))
			continue
		}
		if venvPackages[normalized] {
			dirPath := filepath.Join(addonDir, name)
			utils.LogDebug(fmt.Sprintf("Removing %s (already in venv)", name))
			if err := os.RemoveAll(dirPath); err != nil {
				utils.LogDebug(fmt.Sprintf("Warning: failed to remove %s: %v", name, err))
			}
			removed[normalized] = true
		}
	}

	// 6. Second pass: clean up .dist-info, .data dirs and standalone files
	//    for packages that were removed above.
	files2, _ := os.ReadDir(addonDir)
	for _, file := range files2 {
		name := file.Name()

		if strings.HasSuffix(name, ".dist-info") || strings.HasSuffix(name, ".data") {
			suffix := ".dist-info"
			if strings.HasSuffix(name, ".data") {
				suffix = ".data"
			}
			baseName := strings.TrimSuffix(name, suffix)
			parts := strings.SplitN(baseName, "-", 2)
			if len(parts) > 0 {
				pkgName := normalizePackageName(parts[0])
				if keepPackages[pkgName] || otherAddonPackages[pkgName] {
					continue
				}
				if venvPackages[pkgName] {
					dirPath := filepath.Join(addonDir, name)
					utils.LogDebug(fmt.Sprintf("Removing metadata: %s", name))
					os.RemoveAll(dirPath)
				}
			}
			continue
		}

		// Standalone .py / .pth files
		if !file.IsDir() && (strings.HasSuffix(name, ".py") || strings.HasSuffix(name, ".pth")) {
			baseName := strings.TrimSuffix(strings.TrimSuffix(name, ".py"), ".pth")
			normalized := normalizePackageName(baseName)
			if keepPackages[normalized] || otherAddonPackages[normalized] {
				continue
			}
			if venvPackages[normalized] {
				filePath := filepath.Join(addonDir, name)
				utils.LogDebug(fmt.Sprintf("Removing file: %s", name))
				os.Remove(filePath)
			}
		}
	}

	return nil
}

// extractAddonPackageNames builds the set of normalized Python module names
// from an addon's Packages list (direct dependencies only).
func extractAddonPackageNames(addon *AddonDefinition) map[string]bool {
	keepPackages := make(map[string]bool)

	for _, pkg := range addon.Packages {
		var pkgName string

		if strings.HasPrefix(pkg, "http://") || strings.HasPrefix(pkg, "https://") {
			// URL-based package spec: extract name from wheel filename
			lastSlash := strings.LastIndex(pkg, "/")
			if lastSlash != -1 && strings.HasSuffix(pkg, ".whl") {
				filename := strings.TrimSuffix(pkg[lastSlash+1:], ".whl")
				parts := strings.Split(filename, "-")
				if len(parts) > 0 {
					pkgName = parts[0]
				}
			}
			if pkgName == "" {
				utils.LogDebug(fmt.Sprintf("Warning: could not extract package name from URL: %s", pkg))
				continue
			}
		} else {
			// Strip version specifiers
			pkgName = strings.Split(pkg, ">=")[0]
			pkgName = strings.Split(pkgName, "==")[0]
			pkgName = strings.Split(pkgName, "<")[0]
			pkgName = strings.Split(pkgName, ">")[0]
			pkgName = strings.TrimSpace(pkgName)
		}

		pkgName = normalizePackageName(pkgName)
		keepPackages[pkgName] = true
		utils.LogDebug(fmt.Sprintf("Keeping addon package: %s", pkgName))
	}

	return keepPackages
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
			f, err := os.OpenFile(target, os.O_CREATE|os.O_RDWR|os.O_TRUNC, os.FileMode(header.Mode))
			if err != nil {
				return err
			}
			if _, err := io.Copy(f, tr); err != nil {
				// Handle close error to prevent silent data loss
				if closeErr := f.Close(); closeErr != nil {
					return fmt.Errorf("copy failed: %w; close also failed: %v", err, closeErr)
				}
				return err
			}
			if err := f.Close(); err != nil {
				return err
			}
		default:
			// Ignore symlinks and other special file types for security
			utils.LogDebug(fmt.Sprintf("Skipping unsupported file type in archive: %s (type: %c)", header.Name, header.Typeflag))
		}
	}

	return nil
}
