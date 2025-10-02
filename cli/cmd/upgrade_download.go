package cmd

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

const (
	downloadTimeout = 5 * time.Minute
	maxDownloadSize = 100 * 1024 * 1024 // 100MB max download size
)

// downloadBinary downloads the binary for the specified version and platform
func downloadBinary(version, platform string) (string, error) {
	if version == "" {
		return "", fmt.Errorf("version cannot be empty")
	}

	if platform == "" {
		return "", fmt.Errorf("platform cannot be empty")
	}

	// Normalize version to ensure it has 'v' prefix
	version = normalizeVersion(version)

	// Construct the download URL
	binaryName := getBinaryNameForPlatform(platform)
	downloadURL := fmt.Sprintf("https://github.com/llama-farm/llamafarm/releases/download/%s/%s", version, binaryName)

	logDebug(fmt.Sprintf("downloading binary from: %s", downloadURL))

	// Create temporary file for download
	tempFile, err := createTempFile(binaryName)
	if err != nil {
		return "", fmt.Errorf("failed to create temporary file: %w", err)
	}

	// Download the binary
	err = downloadFile(downloadURL, tempFile)
	if err != nil {
		os.Remove(tempFile)
		return "", fmt.Errorf("failed to download binary: %w", err)
	}

	logDebug(fmt.Sprintf("binary downloaded to: %s", tempFile))
	return tempFile, nil
}

// verifyChecksum downloads and verifies the SHA256 checksum of the binary
func verifyChecksum(binaryPath, version, platform string) error {
	if binaryPath == "" {
		return fmt.Errorf("binary path cannot be empty")
	}

	// Normalize version
	version = normalizeVersion(version)

	// Construct checksum URL
	binaryName := getBinaryNameForPlatform(platform)
	checksumURL := fmt.Sprintf("https://github.com/llama-farm/llamafarm/releases/download/%s/%s.sha256", version, binaryName)

	logDebug(fmt.Sprintf("downloading checksum from: %s", checksumURL))

	// Download checksum file
	checksumFile, err := createTempFile(binaryName + ".sha256")
	if err != nil {
		return fmt.Errorf("failed to create temporary checksum file: %w", err)
	}
	defer os.Remove(checksumFile)

	err = downloadFile(checksumURL, checksumFile)
	if err != nil {
		return fmt.Errorf("failed to download checksum file: %w", err)
	}

	// Read expected checksum
	expectedChecksum, err := readChecksumFile(checksumFile)
	if err != nil {
		return fmt.Errorf("failed to read checksum file: %w", err)
	}

	// Calculate actual checksum
	actualChecksum, err := calculateSHA256(binaryPath)
	if err != nil {
		return fmt.Errorf("failed to calculate binary checksum: %w", err)
	}

	// Compare checksums
	if !strings.EqualFold(expectedChecksum, actualChecksum) {
		return fmt.Errorf("checksum verification failed: expected %s, got %s", expectedChecksum, actualChecksum)
	}

	logDebug("checksum verification passed")
	return nil
}

// downloadFile downloads a file from the given URL to the specified local path
func downloadFile(url, localPath string) error {
	// Create HTTP client with timeout
	client := &http.Client{
		Timeout: downloadTimeout,
	}

	// Create request
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	// Set user agent
	userAgent := fmt.Sprintf("LlamaFarmCLI/%s", strings.TrimSpace(Version))
	if userAgent == "LlamaFarmCLI/" {
		userAgent = "LlamaFarmCLI"
	}
	req.Header.Set("User-Agent", userAgent)

	// Add GitHub token if available (for higher rate limits)
	if token := strings.TrimSpace(os.Getenv("LF_GITHUB_TOKEN")); token != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))
	}

	// Make request
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	// Check response status
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download failed with status %d: %s", resp.StatusCode, resp.Status)
	}

	// Check content length
	if resp.ContentLength > maxDownloadSize {
		return fmt.Errorf("download size %d exceeds maximum allowed size %d", resp.ContentLength, maxDownloadSize)
	}

	// Create output file
	outFile, err := os.Create(localPath)
	if err != nil {
		return fmt.Errorf("failed to create output file: %w", err)
	}

	// Ensure file is cleaned up on error
	defer func() {
		outFile.Close()
		// If there was an error, remove the incomplete file
		if err != nil {
			os.Remove(localPath)
		}
	}()

	// Copy with size limit
	limitedReader := io.LimitReader(resp.Body, maxDownloadSize)
	_, err = io.Copy(outFile, limitedReader)
	if err != nil {
		return fmt.Errorf("failed to write file: %w", err)
	}

	return nil
}

// createTempFile creates a temporary file with the given name pattern
func createTempFile(namePattern string) (string, error) {
	tempDir := os.TempDir()

	// Create a unique temporary file
	tempFile, err := os.CreateTemp(tempDir, "lf-upgrade-"+namePattern+"-*")
	if err != nil {
		return "", fmt.Errorf("failed to create temporary file: %w", err)
	}

	tempPath := tempFile.Name()
	tempFile.Close()

	return tempPath, nil
}

// readChecksumFile reads the checksum from a SHA256 checksum file
func readChecksumFile(checksumPath string) (string, error) {
	content, err := os.ReadFile(checksumPath)
	if err != nil {
		return "", fmt.Errorf("failed to read checksum file: %w", err)
	}

	// Checksum files typically have format: "checksum  filename"
	// We want just the checksum part
	checksumLine := strings.TrimSpace(string(content))
	if checksumLine == "" {
		return "", fmt.Errorf("checksum file is empty")
	}

	// Split on whitespace and take the first part (the checksum)
	parts := strings.Fields(checksumLine)
	if len(parts) == 0 {
		return "", fmt.Errorf("invalid checksum file format")
	}

	checksum := parts[0]
	if len(checksum) != 64 { // SHA256 is 64 hex characters
		return "", fmt.Errorf("invalid checksum length: expected 64 characters, got %d", len(checksum))
	}

	return checksum, nil
}

// calculateSHA256 calculates the SHA256 checksum of a file
func calculateSHA256(filePath string) (string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return "", fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	hasher := sha256.New()
	_, err = io.Copy(hasher, file)
	if err != nil {
		return "", fmt.Errorf("failed to calculate checksum: %w", err)
	}

	checksum := hex.EncodeToString(hasher.Sum(nil))
	return checksum, nil
}

// cleanupTempFiles removes the specified temporary files
func cleanupTempFiles(paths []string) {
	for _, path := range paths {
		if path != "" {
			if err := os.Remove(path); err != nil {
				logDebug(fmt.Sprintf("failed to cleanup temp file %s: %v", path, err))
			} else {
				logDebug(fmt.Sprintf("cleaned up temp file: %s", path))
			}
		}
	}
}
