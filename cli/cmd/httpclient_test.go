package cmd

import (
	"bytes"
	"io"
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

			// Capture debug output
			originalDebug := debug
			debug = true
			defer func() { debug = originalDebug }()

			// Reset logger state for clean test
			ResetDebugLoggerForTesting()
			defer ResetDebugLoggerForTesting()

			result := logBodyContent(body, "test body")

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
