package utils

import (
	"bytes"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"
)

func TestLogBodyContent(t *testing.T) {
	tests := []struct {
		name           string
		bodyContent    string
		expectedLog    string
		shouldHaveBody bool
	}{
		{
			name:           "nil body",
			bodyContent:    "",
			expectedLog:    "<nil>",
			shouldHaveBody: false,
		},
		{
			name:           "empty body",
			bodyContent:    "",
			expectedLog:    "<empty>",
			shouldHaveBody: true,
		},
		{
			name:           "JSON body",
			bodyContent:    `{"name":"test_dataset","database":"main_database"}`,
			expectedLog:    `{"name":"test_dataset","database":"main_database"}`,
			shouldHaveBody: true,
		},
		{
			name:           "large body truncation",
			bodyContent:    strings.Repeat("a", 2000),
			expectedLog:    "... (truncated)",
			shouldHaveBody: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var body io.ReadCloser
			if tt.name == "nil body" {
				body = nil
			} else if tt.name == "empty body" {
				body = io.NopCloser(bytes.NewReader([]byte{}))
			} else {
				body = io.NopCloser(bytes.NewReader([]byte(tt.bodyContent)))
			}

			// Reset logger state for clean test
			ResetDebugLoggerForTesting()
			defer ResetDebugLoggerForTesting()

			result := LogBodyContent(body, "test body")

			// Verify body can still be read if not nil
			if tt.shouldHaveBody && result != nil {
				restoredBytes, err := io.ReadAll(result)
				if err != nil {
					t.Errorf("Failed to read restored body: %v", err)
				}
				if tt.name != "empty body" && len(restoredBytes) == 0 && len(tt.bodyContent) > 0 {
					t.Errorf("Restored body is empty but original was not")
				}
			}

			if !tt.shouldHaveBody && result != nil {
				t.Errorf("Expected nil result but got non-nil")
			}
		})
	}
}

func TestLogHeaders(t *testing.T) {
	tests := []struct {
		name              string
		headers           http.Header
		expectRedacted    []string // Expected header names that should be redacted
		expectNotRedacted []string // Expected header names that should NOT be redacted
		sensitiveValues   []string // Values that should NOT appear in logs
	}{
		{
			name: "redact authorization header",
			headers: http.Header{
				"Authorization": {"Bearer secret-token"},
				"Content-Type":  {"application/json"},
			},
			expectRedacted:    []string{"Authorization"},
			expectNotRedacted: []string{"Content-Type"},
			sensitiveValues:   []string{"secret-token"},
		},
		{
			name: "redact API key headers",
			headers: http.Header{
				"X-Api-Key":    {"secret-key"},
				"Api-Key":      {"another-secret"},
				"Content-Type": {"application/json"},
			},
			expectRedacted:    []string{"X-Api-Key", "Api-Key"},
			expectNotRedacted: []string{"Content-Type"},
			sensitiveValues:   []string{"secret-key", "another-secret"},
		},
		{
			name: "redact cookie and session headers",
			headers: http.Header{
				"Cookie":       {"session=abc123"},
				"Set-Cookie":   {"session=abc123; HttpOnly"},
				"X-Session-Id": {"session-id-123"},
				"Session-Id":   {"another-session"},
				"User-Agent":   {"LlamaFarm-CLI/1.0"},
			},
			expectRedacted:    []string{"Cookie", "Set-Cookie", "X-Session-Id", "Session-Id"},
			expectNotRedacted: []string{"User-Agent"},
			sensitiveValues:   []string{"session=abc123", "session-id-123", "another-session"},
		},
		{
			name: "redact token headers",
			headers: http.Header{
				"X-Auth-Token":    {"auth-token-123"},
				"X-Access-Token":  {"access-token-456"},
				"X-Refresh-Token": {"refresh-token-789"},
				"X-Csrf-Token":    {"csrf-token-abc"},
				"Accept":          {"application/json"},
			},
			expectRedacted:    []string{"X-Auth-Token", "X-Access-Token", "X-Refresh-Token", "X-Csrf-Token"},
			expectNotRedacted: []string{"Accept"},
			sensitiveValues:   []string{"auth-token-123", "access-token-456", "refresh-token-789", "csrf-token-abc"},
		},
		{
			name: "redact proxy and authentication headers",
			headers: http.Header{
				"Proxy-Authorization": {"Basic secret"},
				"Www-Authenticate":    {"Basic realm=test"},
				"Authentication":      {"Bearer token"},
				"Host":                {"api.llamafarm.ai"},
			},
			expectRedacted:    []string{"Proxy-Authorization", "Www-Authenticate", "Authentication"},
			expectNotRedacted: []string{"Host"},
			sensitiveValues:   []string{"Basic secret", "Bearer token"},
		},
		{
			name:              "empty headers",
			headers:           http.Header{},
			expectRedacted:    []string{},
			expectNotRedacted: []string{},
			sensitiveValues:   []string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Reset logger state for clean test
			ResetDebugLoggerForTesting()

			// Initialize debug logger to capture output
			testLogPath := "test_headers_" + strings.ReplaceAll(tt.name, " ", "_") + ".log"
			err := InitDebugLogger(testLogPath, false)
			if err != nil {
				t.Fatalf("Failed to initialize debug logger: %v", err)
			}
			// Clean up in correct order: close logger before removing file
			defer os.Remove(testLogPath)
			defer ResetDebugLoggerForTesting()

			// Log the headers
			LogHeaders("test", tt.headers)

			// Sync to ensure all data is written
			if debugFile != nil {
				err = debugFile.Sync()
				if err != nil {
					t.Fatalf("Failed to sync debug log: %v", err)
				}
			}

			// Read the debug log to verify redaction
			logContent, err := os.ReadFile(testLogPath)
			if err != nil {
				t.Fatalf("Failed to read debug log: %v", err)
			}
			logStr := string(logContent)

			// Verify sensitive headers are redacted
			for _, header := range tt.expectRedacted {
				// Get the canonical form that Go uses for this header
				canonicalHeader := http.CanonicalHeaderKey(header)
				if !strings.Contains(logStr, canonicalHeader+": [REDACTED]") {
					t.Errorf("Expected header %q (canonical: %q) to be redacted, but it wasn't found as redacted in log.\nLog content:\n%s",
						header, canonicalHeader, logStr)
				}
			}

			// Verify sensitive values don't appear in the log
			for _, value := range tt.sensitiveValues {
				if strings.Contains(logStr, value) {
					t.Errorf("Sensitive value %q should not appear in log.\nLog content:\n%s", value, logStr)
				}
			}

			// Verify non-sensitive headers are not redacted
			for _, header := range tt.expectNotRedacted {
				canonicalHeader := http.CanonicalHeaderKey(header)
				if strings.Contains(logStr, canonicalHeader+": [REDACTED]") {
					t.Errorf("Header %q (canonical: %q) should not be redacted", header, canonicalHeader)
				}
				// The value should be in the log
				values := tt.headers.Values(header)
				if len(values) > 0 && !strings.Contains(logStr, values[0]) {
					t.Errorf("Non-sensitive header %q value %q should appear in log.\nLog content:\n%s",
						header, values[0], logStr)
				}
			}
		})
	}
}
