package cmd

import (
	"archive/tar"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/llamafarm/cli/cmd/utils"
)

type AddonDownloader struct {
	version string // LlamaFarm version for downloading wheels
}

func NewAddonDownloader(version string) *AddonDownloader {
	if version == "" || version == "dev" {
		version = "latest" // or use CLI version
	}
	return &AddonDownloader{version: version}
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
	resp, err := http.Get(url)
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
	resp, err := http.Get(checksumURL)
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
