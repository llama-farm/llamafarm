package cmd

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"crypto/sha256"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

const (
	// UV version to use - pinning to a stable version
	uvVersion = "0.9.3"
)

// UVManager handles installation and management of the UV package manager
type UVManager struct {
	homeDir string
	binDir  string
	uvPath  string
}

// NewUVManager creates a new UV manager instance
func NewUVManager() (*UVManager, error) {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return nil, fmt.Errorf("failed to get user home directory: %w", err)
	}

	binDir := filepath.Join(homeDir, ".llamafarm", "bin")
	uvPath := filepath.Join(binDir, "uv")
	if runtime.GOOS == "windows" {
		uvPath = filepath.Join(binDir, "uv.exe")
	}

	return &UVManager{
		homeDir: homeDir,
		binDir:  binDir,
		uvPath:  uvPath,
	}, nil
}

// EnsureUV ensures UV is installed and returns the path to the binary
func (m *UVManager) EnsureUV() (string, error) {
	// Check if UV is already installed
	if m.isUVInstalled() {
		if debug {
			logDebug("UV already installed at: " + m.uvPath)
		}
		return m.uvPath, nil
	}

	// Create bin directory if it doesn't exist
	if err := os.MkdirAll(m.binDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create bin directory: %w", err)
	}

	// Download and install UV
	if err := m.downloadUV(); err != nil {
		return "", fmt.Errorf("failed to download UV: %w", err)
	}

	// Make executable on Unix systems
	if runtime.GOOS != "windows" {
		if err := os.Chmod(m.uvPath, 0755); err != nil {
			return "", fmt.Errorf("failed to make UV executable: %w", err)
		}
	}

	OutputProgress("UV installed successfully to %s\n", m.binDir)
	return m.uvPath, nil
}

// isUVInstalled checks if UV is already installed
func (m *UVManager) isUVInstalled() bool {
	if _, err := os.Stat(m.uvPath); err == nil {
		// UV file exists, verify it's executable
		if runtime.GOOS != "windows" {
			info, err := os.Stat(m.uvPath)
			if err == nil && info.Mode()&0111 != 0 {
				return true
			}
		} else {
			return true
		}
	}
	return false
}

// downloadUV downloads the appropriate UV binary for the current platform
func (m *UVManager) downloadUV() error {
	downloadURL, err := m.getDownloadURL()
	if err != nil {
		return err
	}

	OutputProgress("Downloading UV %s for %s/%s...\n", uvVersion, runtime.GOOS, runtime.GOARCH)

	// Download the binary
	resp, err := http.Get(downloadURL)
	if err != nil {
		return fmt.Errorf("failed to download UV: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to download UV: HTTP %d", resp.StatusCode)
	}

	// Create temporary directory for extraction
	tmpDir, err := os.MkdirTemp("", "uv-download-*")
	if err != nil {
		return fmt.Errorf("failed to create temp directory: %w", err)
	}
	defer os.RemoveAll(tmpDir)

	// Download to temp file
	archivePath := filepath.Join(tmpDir, "uv-archive")
	tmpFile, err := os.Create(archivePath)
	if err != nil {
		return fmt.Errorf("failed to create temp file: %w", err)
	}

	_, err = io.Copy(tmpFile, resp.Body)
	tmpFile.Close()
	if err != nil {
		return fmt.Errorf("failed to write archive: %w", err)
	}

	// Extract archive based on platform
	if runtime.GOOS == "windows" {
		if err := m.extractZip(archivePath, tmpDir); err != nil {
			return fmt.Errorf("failed to extract zip: %w", err)
		}
	} else {
		if err := m.extractTarGz(archivePath, tmpDir); err != nil {
			return fmt.Errorf("failed to extract tar.gz: %w", err)
		}
	}

	// Find the UV binary in the extracted files
	uvBinaryName := "uv"
	if runtime.GOOS == "windows" {
		uvBinaryName = "uv.exe"
	}

	// Look for UV binary in extracted directory
	extractedUV := filepath.Join(tmpDir, uvBinaryName)
	if _, err := os.Stat(extractedUV); err != nil {
		// Try looking in subdirectory (some archives have nested structure)
		matches, _ := filepath.Glob(filepath.Join(tmpDir, "*", uvBinaryName))
		if len(matches) > 0 {
			extractedUV = matches[0]
		} else {
			return fmt.Errorf("UV binary not found in archive")
		}
	}

	// Move to final location
	if err := os.Rename(extractedUV, m.uvPath); err != nil {
		// On Windows, rename can fail if file exists, so try remove first
		os.Remove(m.uvPath)
		if err := os.Rename(extractedUV, m.uvPath); err != nil {
			return fmt.Errorf("failed to install UV binary: %w", err)
		}
	}

	return nil
}

// getDownloadURL returns the appropriate download URL for the current platform
func (m *UVManager) getDownloadURL() (string, error) {
	var platform string
	var arch string

	// Determine platform
	switch runtime.GOOS {
	case "darwin":
		platform = "apple-darwin"
	case "linux":
		platform = "unknown-linux-gnu"
	case "windows":
		platform = "pc-windows-msvc"
	default:
		return "", fmt.Errorf("unsupported operating system: %s", runtime.GOOS)
	}

	// Determine architecture
	switch runtime.GOARCH {
	case "amd64":
		arch = "x86_64"
	case "arm64":
		arch = "aarch64"
	default:
		return "", fmt.Errorf("unsupported architecture: %s", runtime.GOARCH)
	}

	// Construct download URL
	// Format: https://github.com/astral-sh/uv/releases/download/0.5.11/uv-x86_64-apple-darwin.tar.gz
	baseURL := "https://github.com/astral-sh/uv/releases/download"

	var filename string
	if runtime.GOOS == "windows" {
		filename = fmt.Sprintf("uv-%s-%s.zip", arch, platform)
	} else {
		filename = fmt.Sprintf("uv-%s-%s.tar.gz", arch, platform)
	}

	url := fmt.Sprintf("%s/%s/%s", baseURL, uvVersion, filename)
	return url, nil
}

// GetUVPath returns the path to the UV binary
func (m *UVManager) GetUVPath() string {
	return m.uvPath
}

// GetUVVersion returns the installed version of UV
func (m *UVManager) GetUVVersion() (string, error) {
	if !m.isUVInstalled() {
		return "", fmt.Errorf("UV is not installed")
	}

	cmd := exec.Command(m.uvPath, "--version")
	output, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("failed to get UV version: %w", err)
	}

	// Output format: "uv 0.5.11"
	version := strings.TrimSpace(string(output))
	return version, nil
}

// ValidateChecks performs basic validation that UV is working
func (m *UVManager) Validate() error {
	if !m.isUVInstalled() {
		return fmt.Errorf("UV is not installed")
	}

	// Try to run UV with --version to verify it works
	cmd := exec.Command(m.uvPath, "--version")
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("UV binary is not functional: %w", err)
	}

	return nil
}

// sanitizePath validates and sanitizes an archive entry path to prevent directory traversal
func sanitizePath(destDir, entryPath string) (string, error) {
	// Clean the entry path to remove any ".." or other suspicious elements
	cleanPath := filepath.Clean(entryPath)

	// Reject absolute paths
	if filepath.IsAbs(cleanPath) {
		return "", fmt.Errorf("archive contains absolute path: %s", entryPath)
	}

	// Join with destination and clean again
	target := filepath.Join(destDir, cleanPath)

	// Ensure the target is within destDir by checking if destDir is a prefix
	// We use filepath.Rel to check if the target is within destDir
	rel, err := filepath.Rel(destDir, target)
	if err != nil {
		return "", fmt.Errorf("failed to resolve path: %w", err)
	}

	// If the relative path starts with "..", it's trying to escape
	if strings.HasPrefix(rel, ".."+string(filepath.Separator)) || rel == ".." {
		return "", fmt.Errorf("archive contains path traversal: %s", entryPath)
	}

	return target, nil
}

// extractTarGz extracts a tar.gz archive to the specified directory
func (m *UVManager) extractTarGz(archivePath, destDir string) error {
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

		// Sanitize path to prevent directory traversal
		target, err := sanitizePath(destDir, header.Name)
		if err != nil {
			return err
		}

		switch header.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0755); err != nil {
				return err
			}
		case tar.TypeReg:
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
func (m *UVManager) extractZip(archivePath, destDir string) error {
	r, err := zip.OpenReader(archivePath)
	if err != nil {
		return err
	}
	defer r.Close()

	for _, f := range r.File {
		// Sanitize path to prevent directory traversal
		fpath, err := sanitizePath(destDir, f.Name)
		if err != nil {
			return err
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

// computeSHA256 computes the SHA256 checksum of a file
func computeSHA256(filepath string) (string, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return "", err
	}
	defer file.Close()

	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return "", err
	}

	return fmt.Sprintf("%x", hash.Sum(nil)), nil
}
