package cmd

import (
	"encoding/base64"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Test file creation helpers
func createTestImageFile(t *testing.T, dir string) string {
	t.Helper()
	// PNG header (magic bytes)
	pngData := []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A}
	path := filepath.Join(dir, "test.png")
	err := os.WriteFile(path, pngData, 0644)
	if err != nil {
		t.Fatalf("failed to create test image: %v", err)
	}
	return path
}

func createTestTextFile(t *testing.T, dir string, content string) string {
	t.Helper()
	path := filepath.Join(dir, "test.txt")
	err := os.WriteFile(path, []byte(content), 0644)
	if err != nil {
		t.Fatalf("failed to create test text file: %v", err)
	}
	return path
}

func createTestJPEGFile(t *testing.T, dir string) string {
	t.Helper()
	// JPEG header (magic bytes)
	jpegData := []byte{0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46}
	path := filepath.Join(dir, "test.jpg")
	err := os.WriteFile(path, jpegData, 0644)
	if err != nil {
		t.Fatalf("failed to create test JPEG: %v", err)
	}
	return path
}

// TestGetFileType tests MIME type detection via magic bytes
func TestGetFileType(t *testing.T) {
	tmpDir := t.TempDir()

	tests := []struct {
		name     string
		setup    func() string
		expected FileType
	}{
		{
			name: "PNG image",
			setup: func() string {
				return createTestImageFile(t, tmpDir)
			},
			expected: FileTypeImage,
		},
		{
			name: "JPEG image",
			setup: func() string {
				return createTestJPEGFile(t, tmpDir)
			},
			expected: FileTypeImage,
		},
		{
			name: "Text file",
			setup: func() string {
				return createTestTextFile(t, tmpDir, "Hello, World!")
			},
			expected: FileTypeText,
		},
		{
			name: "Non-existent file",
			setup: func() string {
				return filepath.Join(tmpDir, "nonexistent.txt")
			},
			expected: FileTypeUnknown,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := tt.setup()
			got := getFileType(path)
			if got != tt.expected {
				t.Errorf("getFileType() = %v, want %v", got, tt.expected)
			}
		})
	}
}

// TestGetMimeType tests MIME type string detection
func TestGetMimeType(t *testing.T) {
	tmpDir := t.TempDir()

	tests := []struct {
		name     string
		setup    func() string
		expected string
	}{
		{
			name: "PNG image",
			setup: func() string {
				return createTestImageFile(t, tmpDir)
			},
			expected: "image/png",
		},
		{
			name: "JPEG image",
			setup: func() string {
				return createTestJPEGFile(t, tmpDir)
			},
			expected: "image/jpeg",
		},
		{
			name: "Text file",
			setup: func() string {
				return createTestTextFile(t, tmpDir, "Plain text content")
			},
			expected: "text/plain; charset=utf-8",
		},
		{
			name: "Non-existent file",
			setup: func() string {
				return filepath.Join(tmpDir, "nonexistent.txt")
			},
			expected: "application/octet-stream",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := tt.setup()
			got := getMimeType(path)
			if got != tt.expected {
				t.Errorf("getMimeType() = %v, want %v", got, tt.expected)
			}
		})
	}
}

// TestIsMediaFile tests media file detection
func TestIsMediaFile(t *testing.T) {
	tmpDir := t.TempDir()

	tests := []struct {
		name     string
		setup    func() string
		expected bool
	}{
		{
			name: "PNG image is media",
			setup: func() string {
				return createTestImageFile(t, tmpDir)
			},
			expected: true,
		},
		{
			name: "Text file is not media",
			setup: func() string {
				return createTestTextFile(t, tmpDir, "Not media")
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := tt.setup()
			got := isMediaFile(path)
			if got != tt.expected {
				t.Errorf("isMediaFile() = %v, want %v", got, tt.expected)
			}
		})
	}
}

// TestIsTextFile tests text file detection
func TestIsTextFile(t *testing.T) {
	tmpDir := t.TempDir()

	tests := []struct {
		name     string
		setup    func() string
		expected bool
	}{
		{
			name: "Text file is text",
			setup: func() string {
				return createTestTextFile(t, tmpDir, "Text content")
			},
			expected: true,
		},
		{
			name: "PNG image is not text",
			setup: func() string {
				return createTestImageFile(t, tmpDir)
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := tt.setup()
			got := isTextFile(path)
			if got != tt.expected {
				t.Errorf("isTextFile() = %v, want %v", got, tt.expected)
			}
		})
	}
}

// TestEncodeMediaToBase64 tests base64 encoding of media files
func TestEncodeMediaToBase64(t *testing.T) {
	tmpDir := t.TempDir()
	imagePath := createTestImageFile(t, tmpDir)

	dataURL, err := encodeMediaToBase64(imagePath)
	if err != nil {
		t.Fatalf("encodeMediaToBase64() error = %v", err)
	}

	if !strings.HasPrefix(dataURL, "data:image/png;base64,") {
		t.Errorf("dataURL prefix = %v, want data:image/png;base64,", dataURL[:23])
	}

	// Extract and verify base64 data
	base64Data := strings.TrimPrefix(dataURL, "data:image/png;base64,")
	decoded, err := base64.StdEncoding.DecodeString(base64Data)
	if err != nil {
		t.Errorf("failed to decode base64: %v", err)
	}

	// Verify PNG magic bytes
	if len(decoded) < 8 || decoded[0] != 0x89 || decoded[1] != 0x50 {
		t.Error("decoded data does not match PNG magic bytes")
	}
}

// TestReadTextFile tests reading text file content
func TestReadTextFile(t *testing.T) {
	tmpDir := t.TempDir()
	expectedContent := "Hello, this is test content!"
	textPath := createTestTextFile(t, tmpDir, expectedContent)

	content, err := readTextFile(textPath)
	if err != nil {
		t.Fatalf("readTextFile() error = %v", err)
	}

	if content != expectedContent {
		t.Errorf("readTextFile() = %v, want %v", content, expectedContent)
	}
}

// TestDetectFileInInput tests file path detection in user input
func TestDetectFileInInput(t *testing.T) {
	tmpDir := t.TempDir()
	imagePath := createTestImageFile(t, tmpDir)
	textPath := createTestTextFile(t, tmpDir, "content")

	tests := []struct {
		name           string
		input          string
		expectedHasFile bool
		expectedPath   string
		expectedType   FileType
	}{
		{
			name:           "Just a file path",
			input:          imagePath,
			expectedHasFile: true,
			expectedPath:   imagePath,
			expectedType:   FileTypeImage,
		},
		{
			name:           "File in sentence",
			input:          "analyze " + imagePath,
			expectedHasFile: true,
			expectedPath:   imagePath,
			expectedType:   FileTypeImage,
		},
		{
			name:           "No file",
			input:          "just some text",
			expectedHasFile: false,
			expectedPath:   "",
			expectedType:   FileTypeUnknown,
		},
		{
			name:           "Text file with punctuation",
			input:          "what's in " + textPath + "?",
			expectedHasFile: true,
			expectedPath:   textPath,
			expectedType:   FileTypeText,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hasFile, path, fileType, _ := detectFileInInput(tt.input)
			if hasFile != tt.expectedHasFile {
				t.Errorf("detectFileInInput() hasFile = %v, want %v", hasFile, tt.expectedHasFile)
			}
			if path != tt.expectedPath {
				t.Errorf("detectFileInInput() path = %v, want %v", path, tt.expectedPath)
			}
			if fileType != tt.expectedType {
				t.Errorf("detectFileInInput() fileType = %v, want %v", fileType, tt.expectedType)
			}
		})
	}
}

// TestBuildMultimodalMessage tests multimodal message construction
func TestBuildMultimodalMessage(t *testing.T) {
	tmpDir := t.TempDir()
	imagePath := createTestImageFile(t, tmpDir)
	textPath := createTestTextFile(t, tmpDir, "File content")

	tests := []struct {
		name      string
		role      string
		text      string
		filePaths []string
		wantErr   bool
		validate  func(*testing.T, Message)
	}{
		{
			name:      "Simple text message",
			role:      "user",
			text:      "Hello",
			filePaths: []string{},
			wantErr:   false,
			validate: func(t *testing.T, msg Message) {
				if msg.Role != "user" {
					t.Errorf("Role = %v, want user", msg.Role)
				}
				if str, ok := msg.Content.(string); !ok || str != "Hello" {
					t.Errorf("Content = %v, want Hello", msg.Content)
				}
			},
		},
		{
			name:      "Image message",
			role:      "user",
			text:      "Describe this",
			filePaths: []string{imagePath},
			wantErr:   false,
			validate: func(t *testing.T, msg Message) {
				parts, ok := msg.Content.([]MessageContentPart)
				if !ok {
					t.Fatalf("Content is not []MessageContentPart")
				}
				if len(parts) != 2 {
					t.Errorf("len(parts) = %v, want 2", len(parts))
				}
				if parts[0].Type != "text" || parts[0].Text != "Describe this" {
					t.Error("First part should be text")
				}
				if parts[1].Type != "image_url" {
					t.Error("Second part should be image_url")
				}
			},
		},
		{
			name:      "Text file appended",
			role:      "user",
			text:      "Analyze:",
			filePaths: []string{textPath},
			wantErr:   false,
			validate: func(t *testing.T, msg Message) {
				str, ok := msg.Content.(string)
				if !ok {
					t.Fatalf("Content should be string for text-only message")
				}
				if !strings.Contains(str, "Analyze:") || !strings.Contains(str, "File content") {
					t.Errorf("Content = %v, should contain both prompt and file content", str)
				}
			},
		},
		{
			name:      "Non-existent file",
			role:      "user",
			text:      "Test",
			filePaths: []string{filepath.Join(tmpDir, "nonexistent.txt")},
			wantErr:   true,
			validate:  nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			msg, err := buildMultimodalMessage(tt.role, tt.text, tt.filePaths)
			if (err != nil) != tt.wantErr {
				t.Errorf("buildMultimodalMessage() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && tt.validate != nil {
				tt.validate(t, msg)
			}
		})
	}
}

// TestExtractBase64Media tests extracting base64 media from response text
func TestExtractBase64Media(t *testing.T) {
	tests := []struct {
		name             string
		text             string
		expectedHasMedia bool
		expectedMimeType string
	}{
		{
			name:             "Image data URL",
			text:             "Here's an image: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA",
			expectedHasMedia: true,
			expectedMimeType: "image/png",
		},
		{
			name:             "Video data URL",
			text:             "Video: data:video/mp4;base64,AAAAIGZ0eXBpc29t",
			expectedHasMedia: true,
			expectedMimeType: "video/mp4",
		},
		{
			name:             "No media",
			text:             "Just plain text with no media",
			expectedHasMedia: false,
			expectedMimeType: "",
		},
		{
			name:             "Audio data URL",
			text:             "Sound: data:audio/mpeg;base64,SUQzBAAAAAAAI1RTU0UAAAA=",
			expectedHasMedia: true,
			expectedMimeType: "audio/mpeg",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hasMedia, data, mimeType, _ := extractBase64Media(tt.text)
			if hasMedia != tt.expectedHasMedia {
				t.Errorf("extractBase64Media() hasMedia = %v, want %v", hasMedia, tt.expectedHasMedia)
			}
			if hasMedia {
				if mimeType != tt.expectedMimeType {
					t.Errorf("extractBase64Media() mimeType = %v, want %v", mimeType, tt.expectedMimeType)
				}
				if len(data) == 0 {
					t.Error("extractBase64Media() data is empty")
				}
			}
		})
	}
}

// TestSaveMediaFromBase64 tests saving decoded media to file
func TestSaveMediaFromBase64(t *testing.T) {
	// Create valid PNG base64 data
	pngData := []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A}
	base64Data := base64.StdEncoding.EncodeToString(pngData)

	// Change to temp directory for test
	originalWd, _ := os.Getwd()
	tmpDir := t.TempDir()
	os.Chdir(tmpDir)
	defer os.Chdir(originalWd)

	filename, err := saveMediaFromBase64(base64Data, "image/png")
	if err != nil {
		t.Fatalf("saveMediaFromBase64() error = %v", err)
	}

	// Check file was created
	if !strings.HasSuffix(filename, ".png") {
		t.Errorf("filename = %v, want *.png", filename)
	}

	// Verify file content
	savedData, err := os.ReadFile(filename)
	if err != nil {
		t.Fatalf("failed to read saved file: %v", err)
	}

	if len(savedData) != len(pngData) {
		t.Errorf("saved data length = %v, want %v", len(savedData), len(pngData))
	}
}

// TestFileExists tests the fileExists helper
func TestFileExists(t *testing.T) {
	tmpDir := t.TempDir()
	existingFile := createTestTextFile(t, tmpDir, "exists")

	tests := []struct {
		name     string
		path     string
		expected bool
	}{
		{
			name:     "Existing file",
			path:     existingFile,
			expected: true,
		},
		{
			name:     "Non-existent file",
			path:     filepath.Join(tmpDir, "nonexistent.txt"),
			expected: false,
		},
		{
			name:     "Directory",
			path:     tmpDir,
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := fileExists(tt.path)
			if got != tt.expected {
				t.Errorf("fileExists() = %v, want %v", got, tt.expected)
			}
		})
	}
}
