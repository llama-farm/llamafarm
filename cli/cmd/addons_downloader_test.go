package cmd

import (
	"archive/tar"
	"compress/gzip"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestExtractTarGz_PathTraversal(t *testing.T) {
	// Create a temporary directory for testing
	tempDir := t.TempDir()
	tarPath := filepath.Join(tempDir, "test.tar.gz")
	extractDir := filepath.Join(tempDir, "extract")

	// Create a malicious tar.gz with path traversal attempts
	if err := createMaliciousTarGz(tarPath); err != nil {
		t.Fatalf("Failed to create test tar.gz: %v", err)
	}

	// Create downloader
	d := &AddonDownloader{version: "test"}

	// Attempt to extract (should fail)
	err := d.extractTarGz(tarPath, extractDir)
	if err == nil {
		t.Fatal("extractTarGz() should reject path traversal attempts")
	}

	if !strings.Contains(err.Error(), "illegal path") {
		t.Errorf("extractTarGz() error should mention illegal path, got: %v", err)
	}

	// Verify no files were extracted outside extractDir
	// Check if parent directory has any unexpected files
	parentDir := filepath.Dir(extractDir)
	entries, err := os.ReadDir(parentDir)
	if err != nil {
		t.Fatalf("Failed to read parent dir: %v", err)
	}

	for _, entry := range entries {
		if entry.Name() == "escape.txt" {
			t.Error("Path traversal succeeded - file escaped extraction directory")
		}
	}
}

func TestExtractTarGz_ValidFiles(t *testing.T) {
	// Create a temporary directory for testing
	tempDir := t.TempDir()
	tarPath := filepath.Join(tempDir, "valid.tar.gz")
	extractDir := filepath.Join(tempDir, "extract")

	// Create a valid tar.gz
	if err := createValidTarGz(tarPath); err != nil {
		t.Fatalf("Failed to create test tar.gz: %v", err)
	}

	// Create downloader
	d := &AddonDownloader{version: "test"}

	// Extract
	err := d.extractTarGz(tarPath, extractDir)
	if err != nil {
		t.Errorf("extractTarGz() failed on valid archive: %v", err)
	}

	// Verify file was extracted
	extractedFile := filepath.Join(extractDir, "test.txt")
	if _, err := os.Stat(extractedFile); os.IsNotExist(err) {
		t.Error("extractTarGz() did not extract valid file")
	}

	// Read content
	content, err := os.ReadFile(extractedFile)
	if err != nil {
		t.Fatalf("Failed to read extracted file: %v", err)
	}

	if string(content) != "test content" {
		t.Errorf("Extracted file content = %s, want 'test content'", string(content))
	}
}

func TestExtractTarGz_SymlinksIgnored(t *testing.T) {
	// Create a temporary directory for testing
	tempDir := t.TempDir()
	tarPath := filepath.Join(tempDir, "symlink.tar.gz")
	extractDir := filepath.Join(tempDir, "extract")

	// Create a tar.gz with symlinks
	if err := createTarGzWithSymlink(tarPath); err != nil {
		t.Fatalf("Failed to create test tar.gz: %v", err)
	}

	// Create downloader
	d := &AddonDownloader{version: "test"}

	// Extract (should succeed but skip symlinks)
	err := d.extractTarGz(tarPath, extractDir)
	if err != nil {
		t.Errorf("extractTarGz() failed: %v", err)
	}

	// Verify symlink was not created
	symlinkPath := filepath.Join(extractDir, "link.txt")
	info, err := os.Lstat(symlinkPath)
	if err == nil && info.Mode()&os.ModeSymlink != 0 {
		t.Error("extractTarGz() created symlink (should skip)")
	}
}

// Helper functions to create test archives

func createMaliciousTarGz(path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	gzw := gzip.NewWriter(file)
	defer gzw.Close()

	tw := tar.NewWriter(gzw)
	defer tw.Close()

	// Add a file with path traversal
	header := &tar.Header{
		Name: "../../../escape.txt",
		Mode: 0644,
		Size: 14,
	}
	if err := tw.WriteHeader(header); err != nil {
		return err
	}
	if _, err := tw.Write([]byte("malicious data")); err != nil {
		return err
	}

	return nil
}

func createValidTarGz(path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	gzw := gzip.NewWriter(file)
	defer gzw.Close()

	tw := tar.NewWriter(gzw)
	defer tw.Close()

	// Add a valid file
	content := []byte("test content")
	header := &tar.Header{
		Name: "test.txt",
		Mode: 0644,
		Size: int64(len(content)),
	}
	if err := tw.WriteHeader(header); err != nil {
		return err
	}
	if _, err := tw.Write(content); err != nil {
		return err
	}

	return nil
}

func createTarGzWithSymlink(path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	gzw := gzip.NewWriter(file)
	defer gzw.Close()

	tw := tar.NewWriter(gzw)
	defer tw.Close()

	// Add a regular file
	content := []byte("target")
	header := &tar.Header{
		Name: "target.txt",
		Mode: 0644,
		Size: int64(len(content)),
	}
	if err := tw.WriteHeader(header); err != nil {
		return err
	}
	if _, err := tw.Write(content); err != nil {
		return err
	}

	// Add a symlink
	header = &tar.Header{
		Name:     "link.txt",
		Typeflag: tar.TypeSymlink,
		Linkname: "target.txt",
	}
	if err := tw.WriteHeader(header); err != nil {
		return err
	}

	return nil
}
