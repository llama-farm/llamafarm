package cmd

import (
	"encoding/base64"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// File type categories and their MIME types
var imageExtensions = map[string]string{
	".jpg":  "image/jpeg",
	".jpeg": "image/jpeg",
	".png":  "image/png",
	".gif":  "image/gif",
	".webp": "image/webp",
	".bmp":  "image/bmp",
	".tiff": "image/tiff",
	".tif":  "image/tiff",
}

var videoExtensions = map[string]string{
	".mp4":  "video/mp4",
	".mov":  "video/quicktime",
	".avi":  "video/x-msvideo",
	".mkv":  "video/x-matroska",
	".webm": "video/webm",
	".flv":  "video/x-flv",
	".wmv":  "video/x-ms-wmv",
}

var audioExtensions = map[string]string{
	".mp3":  "audio/mpeg",
	".wav":  "audio/wav",
	".ogg":  "audio/ogg",
	".m4a":  "audio/mp4",
	".flac": "audio/flac",
	".aac":  "audio/aac",
}

var textExtensions = map[string]string{
	".txt":  "text/plain",
	".md":   "text/markdown",
	".json": "application/json",
	".xml":  "application/xml",
	".yaml": "text/yaml",
	".yml":  "text/yaml",
	".toml": "text/toml",
	".csv":  "text/csv",
	".log":  "text/plain",
}

// FileType represents the category of a file
type FileType int

const (
	FileTypeUnknown FileType = iota
	FileTypeImage
	FileTypeVideo
	FileTypeAudio
	FileTypeText
)

// getFileType determines the type of file based on extension
func getFileType(path string) FileType {
	ext := strings.ToLower(filepath.Ext(path))

	if _, ok := imageExtensions[ext]; ok {
		return FileTypeImage
	}
	if _, ok := videoExtensions[ext]; ok {
		return FileTypeVideo
	}
	if _, ok := audioExtensions[ext]; ok {
		return FileTypeAudio
	}
	if _, ok := textExtensions[ext]; ok {
		return FileTypeText
	}

	return FileTypeUnknown
}

// isImageFile checks if a file path has an image extension
func isImageFile(path string) bool {
	return getFileType(path) == FileTypeImage
}

// isMediaFile checks if a file is image, video, or audio (requires base64 encoding)
func isMediaFile(path string) bool {
	ft := getFileType(path)
	return ft == FileTypeImage || ft == FileTypeVideo || ft == FileTypeAudio
}

// isTextFile checks if a file is a text file (can be sent as plain text)
func isTextFile(path string) bool {
	return getFileType(path) == FileTypeText
}

// getMimeType returns the MIME type for a file
func getMimeType(path string) string {
	ext := strings.ToLower(filepath.Ext(path))

	if mimeType, ok := imageExtensions[ext]; ok {
		return mimeType
	}
	if mimeType, ok := videoExtensions[ext]; ok {
		return mimeType
	}
	if mimeType, ok := audioExtensions[ext]; ok {
		return mimeType
	}
	if mimeType, ok := textExtensions[ext]; ok {
		return mimeType
	}

	return "application/octet-stream"
}

// encodeMediaToBase64 reads a media file (image/video/audio) and returns a base64-encoded data URL
func encodeMediaToBase64(path string) (string, error) {
	// Read the file
	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("failed to read media file: %w", err)
	}

	// Get MIME type
	mimeType := getMimeType(path)

	// Encode to base64
	encoded := base64.StdEncoding.EncodeToString(data)

	// Create data URL
	dataURL := fmt.Sprintf("data:%s;base64,%s", mimeType, encoded)

	return dataURL, nil
}

// readTextFile reads a text file and returns its content as a string
func readTextFile(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("failed to read text file: %w", err)
	}
	return string(data), nil
}

// detectFileInInput checks if the input string is or contains a file path
// Returns (hasFile, filePath, fileType, remainingText)
func detectFileInInput(input string) (bool, string, FileType, string) {
	input = strings.TrimSpace(input)

	// Check if the entire input is a file path
	if fileExists(input) {
		fileType := getFileType(input)
		if fileType != FileTypeUnknown {
			return true, input, fileType, ""
		}
	}

	// Check for common patterns like "describe photo.jpg" or "what's in document.txt?"
	words := strings.Fields(input)
	for _, word := range words {
		// Remove common punctuation
		cleanWord := strings.TrimRight(word, "?!.,;:")
		if fileExists(cleanWord) {
			fileType := getFileType(cleanWord)
			if fileType != FileTypeUnknown {
				// Found a file reference in the text
				// Keep the text but mark the file
				return true, cleanWord, fileType, input
			}
		}
	}

	return false, "", FileTypeUnknown, input
}

// fileExists checks if a file exists
func fileExists(path string) bool {
	info, err := os.Stat(path)
	if err != nil {
		return false
	}
	return !info.IsDir()
}

// buildMultimodalMessage creates a multimodal message with text and/or media files
// filePaths can contain images, videos, audio, or text files
func buildMultimodalMessage(role string, text string, filePaths []string) (Message, error) {
	var parts []MessageContentPart
	textContent := text

	// Process files first
	for _, filePath := range filePaths {
		fileType := getFileType(filePath)

		switch fileType {
		case FileTypeText:
			// Read text files and append to text content
			content, err := readTextFile(filePath)
			if err != nil {
				return Message{}, fmt.Errorf("failed to read text file %s: %w", filePath, err)
			}
			// Add as a labeled text block
			filename := filepath.Base(filePath)
			if textContent != "" {
				textContent += "\n\n"
			}
			textContent += fmt.Sprintf("Content from %s:\n%s", filename, content)

		case FileTypeImage, FileTypeVideo, FileTypeAudio:
			// Encode media files as base64
			dataURL, err := encodeMediaToBase64(filePath)
			if err != nil {
				return Message{}, fmt.Errorf("failed to encode media file %s: %w", filePath, err)
			}

			parts = append(parts, MessageContentPart{
				Type: "image_url", // OpenAI uses "image_url" for all media types
				ImageURL: &ImageURLContent{
					URL:    dataURL,
					Detail: "auto",
				},
			})

		default:
			return Message{}, fmt.Errorf("unsupported file type: %s", filePath)
		}
	}

	// Add text part if present
	if textContent != "" {
		parts = append([]MessageContentPart{{
			Type: "text",
			Text: textContent,
		}}, parts...)
	}

	// If only one text part, use simple string content for compatibility
	if len(parts) == 1 && parts[0].Type == "text" {
		return Message{
			Role:    role,
			Content: textContent,
		}, nil
	}

	// If no parts, return simple text message
	if len(parts) == 0 {
		return Message{
			Role:    role,
			Content: textContent,
		}, nil
	}

	// Otherwise use multimodal format
	return Message{
		Role:    role,
		Content: parts,
	}, nil
}

// extractBase64Media extracts base64-encoded media data (image/video/audio) from a response
// Returns (hasMedia, mediaData, mimeType, remainingText)
func extractBase64Media(text string) (bool, string, string, string) {
	// Look for data URL patterns: data:image/..., data:video/..., data:audio/...
	prefixes := []string{"data:image/", "data:video/", "data:audio/"}
	base64Marker := ";base64,"

	for _, dataURLPrefix := range prefixes {
		idx := strings.Index(text, dataURLPrefix)
		if idx == -1 {
			continue
		}

		// Find the MIME type
		mimeStart := idx + len(dataURLPrefix)
		mimeEnd := strings.Index(text[mimeStart:], base64Marker)
		if mimeEnd == -1 {
			continue
		}

		// Extract the full MIME type
		mediaType := dataURLPrefix[5 : len(dataURLPrefix)-1] // Extract "image", "video", or "audio"
		mimeType := mediaType + "/" + text[mimeStart:mimeStart+mimeEnd]

		// Find the base64 data
		dataStart := mimeStart + mimeEnd + len(base64Marker)

		// Find the end of base64 data (whitespace, quote, or end of string)
		dataEnd := dataStart
		for dataEnd < len(text) && !isWhitespaceOrDelimiter(text[dataEnd]) {
			dataEnd++
		}

		base64Data := text[dataStart:dataEnd]

		// Remove the data URL from the text
		remainingText := text[:idx] + text[dataEnd:]

		return true, base64Data, mimeType, strings.TrimSpace(remainingText)
	}

	return false, "", "", text
}

// isWhitespaceOrDelimiter checks if a character is whitespace or a delimiter
func isWhitespaceOrDelimiter(c byte) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '"' || c == '\'' || c == ')' || c == ']' || c == '}'
}

// saveMediaFromBase64 decodes base64 media data and saves it to a file
func saveMediaFromBase64(base64Data, mimeType string) (string, error) {
	// Decode base64
	mediaData, err := base64.StdEncoding.DecodeString(base64Data)
	if err != nil {
		return "", fmt.Errorf("failed to decode base64 media: %w", err)
	}

	// Determine file extension from MIME type
	ext := ".bin" // default
	switch mimeType {
	// Images
	case "image/jpeg":
		ext = ".jpg"
	case "image/png":
		ext = ".png"
	case "image/gif":
		ext = ".gif"
	case "image/webp":
		ext = ".webp"
	case "image/bmp":
		ext = ".bmp"
	case "image/tiff":
		ext = ".tiff"
	// Videos
	case "video/mp4":
		ext = ".mp4"
	case "video/quicktime":
		ext = ".mov"
	case "video/x-msvideo":
		ext = ".avi"
	case "video/x-matroska":
		ext = ".mkv"
	case "video/webm":
		ext = ".webm"
	// Audio
	case "audio/mpeg":
		ext = ".mp3"
	case "audio/wav":
		ext = ".wav"
	case "audio/ogg":
		ext = ".ogg"
	case "audio/mp4":
		ext = ".m4a"
	}

	// Generate filename with timestamp
	filename := fmt.Sprintf("output_%s%s", strings.ReplaceAll(strings.ReplaceAll(
		fmt.Sprintf("%d", os.Getpid()), " ", "_"), ":", ""), ext)

	// Write to current directory
	err = os.WriteFile(filename, mediaData, 0644)
	if err != nil {
		return "", fmt.Errorf("failed to write media file: %w", err)
	}

	return filename, nil
}
