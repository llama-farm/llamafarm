package cmd

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

func TestCalculateSHA256(t *testing.T) {
	tempDir := t.TempDir()
	testFile := tempDir + "/test.txt"
	content := "test content for checksum"

	// Create test file
	err := os.WriteFile(testFile, []byte(content), 0644)
	if err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	// Calculate checksum
	checksum, err := calculateSHA256(testFile)
	if err != nil {
		t.Fatalf("Failed to calculate checksum: %v", err)
	}

	// Verify checksum length
	if len(checksum) != 64 {
		t.Errorf("Expected checksum length 64, got %d", len(checksum))
	}

	// Calculate expected checksum
	hasher := sha256.New()
	hasher.Write([]byte(content))
	expected := hex.EncodeToString(hasher.Sum(nil))

	if checksum != expected {
		t.Errorf("Expected checksum %s, got %s", expected, checksum)
	}
}

func TestReadChecksumFile(t *testing.T) {
	tempDir := t.TempDir()
	checksumFile := tempDir + "/test.sha256"

	// Test valid checksum file
	expectedChecksum := "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
	checksumContent := fmt.Sprintf("%s  test-binary", expectedChecksum)

	err := os.WriteFile(checksumFile, []byte(checksumContent), 0644)
	if err != nil {
		t.Fatalf("Failed to create checksum file: %v", err)
	}

	checksum, err := readChecksumFile(checksumFile)
	if err != nil {
		t.Fatalf("Failed to read checksum file: %v", err)
	}

	if checksum != expectedChecksum {
		t.Errorf("Expected checksum %s, got %s", expectedChecksum, checksum)
	}
}

func TestReadChecksumFileInvalid(t *testing.T) {
	tempDir := t.TempDir()

	tests := []struct {
		name    string
		content string
	}{
		{"empty file", ""},
		{"short checksum", "abc123"},
		{"no content", "   "},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			checksumFile := tempDir + "/" + test.name + ".sha256"
			err := os.WriteFile(checksumFile, []byte(test.content), 0644)
			if err != nil {
				t.Fatalf("Failed to create test file: %v", err)
			}

			_, err = readChecksumFile(checksumFile)
			if err == nil {
				t.Error("Expected error for invalid checksum file")
			}
		})
	}
}

func TestReadChecksumFileWhitespaceAndMultiline(t *testing.T) {
	tempDir := t.TempDir()
	expectedChecksum := "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"

	// Case 1: Extra whitespace before and after
	checksumFile1 := tempDir + "/checksum1.txt"
	content1 := "   " + expectedChecksum + "   \n"
	if err := os.WriteFile(checksumFile1, []byte(content1), 0644); err != nil {
		t.Fatalf("Failed to write checksum file: %v", err)
	}
	checksum, err := readChecksumFile(checksumFile1)
	if err != nil {
		t.Fatalf("Failed to read checksum file: %v", err)
	}
	if checksum != expectedChecksum {
		t.Errorf("Expected checksum %s, got %s", expectedChecksum, checksum)
	}

	// Case 2: Multiple lines, checksum on first line
	checksumFile2 := tempDir + "/checksum2.txt"
	content2 := expectedChecksum + "\nextra line\n"
	if err := os.WriteFile(checksumFile2, []byte(content2), 0644); err != nil {
		t.Fatalf("Failed to write checksum file: %v", err)
	}
	checksum, err = readChecksumFile(checksumFile2)
	if err != nil {
		t.Fatalf("Failed to read checksum file: %v", err)
	}
	if checksum != expectedChecksum {
		t.Errorf("Expected checksum %s, got %s", expectedChecksum, checksum)
	}

	// Case 3: Multiple lines, checksum on second line
	checksumFile3 := tempDir + "/checksum3.txt"
	content3 := "not a checksum\n" + expectedChecksum + "\n"
	if err := os.WriteFile(checksumFile3, []byte(content3), 0644); err != nil {
		t.Fatalf("Failed to write checksum file: %v", err)
	}
	checksum, err = readChecksumFile(checksumFile3)
	if err == nil && checksum == expectedChecksum {
		t.Errorf("Expected error for checksum not on first line, got valid checksum")
	}

	// Case 4: Checksum surrounded by tabs and spaces
	checksumFile4 := tempDir + "/checksum4.txt"
	content4 := "\t " + expectedChecksum + " \t\n"
	if err := os.WriteFile(checksumFile4, []byte(content4), 0644); err != nil {
		t.Fatalf("Failed to write checksum file: %v", err)
	}
	checksum, err = readChecksumFile(checksumFile4)
	if err != nil {
		t.Fatalf("Failed to read checksum file: %v", err)
	}
	if checksum != expectedChecksum {
		t.Errorf("Expected checksum %s, got %s", expectedChecksum, checksum)
	}
}

func TestDownloadFile(t *testing.T) {
	// Create test server
	testContent := "test binary content"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Check User-Agent header
		userAgent := r.Header.Get("User-Agent")
		if !strings.Contains(userAgent, "LlamaFarmCLI") {
			t.Errorf("Expected User-Agent to contain 'LlamaFarmCLI', got: %s", userAgent)
		}

		w.WriteHeader(http.StatusOK)
		w.Write([]byte(testContent))
	}))
	defer server.Close()

	// Download to temporary file
	tempDir := t.TempDir()
	tempFile := tempDir + "/downloaded"

	err := downloadFile(server.URL, tempFile)
	if err != nil {
		t.Fatalf("Failed to download file: %v", err)
	}

	// Verify content
	content, err := os.ReadFile(tempFile)
	if err != nil {
		t.Fatalf("Failed to read downloaded file: %v", err)
	}

	if string(content) != testContent {
		t.Errorf("Expected content %s, got %s", testContent, string(content))
	}
}

func TestDownloadFileError(t *testing.T) {
	// Test with server that returns error
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		w.Write([]byte("Not found"))
	}))
	defer server.Close()

	tempDir := t.TempDir()
	tempFile := tempDir + "/downloaded"

	err := downloadFile(server.URL, tempFile)
	if err == nil {
		t.Error("Expected error for 404 response")
	}
}

func TestDownloadFileExceedsMaxDownloadSize(t *testing.T) {
	// Create a response body larger than maxDownloadSize
	oversizedContent := make([]byte, maxDownloadSize+1024)
	for i := range oversizedContent {
		oversizedContent[i] = 'A'
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", fmt.Sprintf("%d", len(oversizedContent)))
		w.WriteHeader(http.StatusOK)
		w.Write(oversizedContent)
	}))
	defer server.Close()

	tempFile := t.TempDir() + "/oversized_download"
	err := downloadFile(server.URL, tempFile)
	if err == nil {
		t.Fatalf("Expected error for oversized download, got nil")
	}
	if !strings.Contains(err.Error(), "exceeds maximum allowed size") {
		t.Errorf("Expected error about exceeding max size, got: %v", err)
	}
}

func TestCreateTempFile(t *testing.T) {
	tempPath, err := createTempFile("test-binary")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	defer os.Remove(tempPath)

	// Verify file was created
	if _, err := os.Stat(tempPath); os.IsNotExist(err) {
		t.Error("Temp file was not created")
	}

	// Verify filename pattern
	if !strings.Contains(tempPath, "lf-upgrade-test-binary") {
		t.Errorf("Expected filename to contain pattern, got: %s", tempPath)
	}
}

func TestCleanupTempFiles(t *testing.T) {
	tempDir := t.TempDir()

	// Create test files
	file1 := tempDir + "/temp1"
	file2 := tempDir + "/temp2"

	os.WriteFile(file1, []byte("test"), 0644)
	os.WriteFile(file2, []byte("test"), 0644)

	// Verify files exist
	if _, err := os.Stat(file1); os.IsNotExist(err) {
		t.Fatal("Test file 1 should exist")
	}
	if _, err := os.Stat(file2); os.IsNotExist(err) {
		t.Fatal("Test file 2 should exist")
	}

	// Cleanup
	cleanupTempFiles([]string{file1, file2})

	// Verify files are gone
	if _, err := os.Stat(file1); !os.IsNotExist(err) {
		t.Error("Test file 1 should be deleted")
	}
	if _, err := os.Stat(file2); !os.IsNotExist(err) {
		t.Error("Test file 2 should be deleted")
	}
}
