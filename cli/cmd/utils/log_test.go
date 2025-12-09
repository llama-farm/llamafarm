package utils

import (
	"os"
	"strings"
	"testing"
)

func TestSanitizeLogMessage(t *testing.T) {
	tests := []struct {
		name             string
		input            string
		shouldContain    []string
		shouldNotContain []string
	}{
		{
			name:             "bearer token",
			input:            "Authorization: Bearer sk-abc123def456xyz789",
			shouldContain:    []string{"Authorization:", "Bearer", "[REDACTED]"},
			shouldNotContain: []string{"sk-abc123def456xyz789"},
		},
		{
			name:             "API key with equals",
			input:            "api_key=1234567890abcdef1234567890",
			shouldContain:    []string{"api_key=", "[REDACTED]"},
			shouldNotContain: []string{"1234567890abcdef1234567890"},
		},
		{
			name:             "OpenAI style key",
			input:            "Using key sk-proj-abcdefghijklmnopqrstuvwxyz123456789",
			shouldContain:    []string{"Using key", "[REDACTED-KEY]"},
			shouldNotContain: []string{"sk-proj-abcdefghijklmnopqrstuvwxyz123456789"},
		},
		{
			name:             "password in query param",
			input:            "Connecting with password=MySecret123!@#",
			shouldContain:    []string{"password=", "[REDACTED]"},
			shouldNotContain: []string{"MySecret123!@#"},
		},
		{
			name:             "JWT token",
			input:            "Token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
			shouldContain:    []string{"Token:", "[REDACTED-JWT]"},
			shouldNotContain: []string{"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"},
		},
		{
			name:             "AWS access key",
			input:            "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE",
			shouldContain:    []string{"AWS_ACCESS_KEY_ID=", "[REDACTED-AWS-KEY]"},
			shouldNotContain: []string{"AKIAIOSFODNN7EXAMPLE"},
		},
		{
			name:             "session ID",
			input:            "session_id=abc123def456ghi789jkl012mno345pqr678",
			shouldContain:    []string{"session_id=", "[REDACTED]"},
			shouldNotContain: []string{"abc123def456ghi789jkl012mno345pqr678"},
		},
		{
			name:             "access token",
			input:            "access_token: ghp_1234567890abcdefghijklmnopqrstuvwxyz",
			shouldContain:    []string{"access_token:", "[REDACTED]"},
			shouldNotContain: []string{"ghp_1234567890abcdefghijklmnopqrstuvwxyz"},
		},
		{
			name:             "refresh token",
			input:            "refresh-token=rt_abcdefghijklmnopqrstuvwxyz1234567890",
			shouldContain:    []string{"refresh-token=", "[REDACTED]"},
			shouldNotContain: []string{"rt_abcdefghijklmnopqrstuvwxyz1234567890"},
		},
		{
			name:             "private key",
			input:            "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7VJTUt9Us8cKj\n-----END PRIVATE KEY-----",
			shouldContain:    []string{"[REDACTED-PRIVATE-KEY]"},
			shouldNotContain: []string{"MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7VJTUt9Us8cKj"},
		},
		{
			name:             "cookie value",
			input:            "cookie: sessionid=abc123; path=/; secure",
			shouldContain:    []string{"cookie:", "[REDACTED]"},
			shouldNotContain: []string{"sessionid=abc123"},
		},
		{
			name:             "multiple sensitive patterns",
			input:            "api_key=secret123456789012 and password=MyPass123 with token=tok_abcdef123456789012345",
			shouldContain:    []string{"api_key=", "password=", "token=", "[REDACTED]"},
			shouldNotContain: []string{"secret123456789012", "MyPass123", "tok_abcdef123456789012345"},
		},
		{
			name:             "non-sensitive data unchanged",
			input:            "Processing file: data.json with 1234 records at 2024-01-15",
			shouldContain:    []string{"Processing file:", "data.json", "1234 records", "2024-01-15"},
			shouldNotContain: []string{"[REDACTED]"},
		},
		{
			name:             "authorization header with basic auth",
			input:            "Authorization: Basic dXNlcjpwYXNzd29yZA==",
			shouldContain:    []string{"Authorization:", "Basic", "[REDACTED]"},
			shouldNotContain: []string{"dXNlcjpwYXNzd29yZA=="},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := sanitizeLogMessage(tt.input)

			// Check for strings that should be present
			for _, expected := range tt.shouldContain {
				if !strings.Contains(result, expected) {
					t.Errorf("Expected sanitized message to contain %q\nInput:  %s\nOutput: %s",
						expected, tt.input, result)
				}
			}

			// Check for strings that should NOT be present
			for _, forbidden := range tt.shouldNotContain {
				if strings.Contains(result, forbidden) {
					t.Errorf("Expected sanitized message NOT to contain %q\nInput:  %s\nOutput: %s",
						forbidden, tt.input, result)
				}
			}
		})
	}
}

func TestLogDebugSanitization(t *testing.T) {
	// Reset logger state for clean test
	ResetDebugLoggerForTesting()
	defer ResetDebugLoggerForTesting()

	// Initialize debug logger to capture output
	testLogPath := "test_debug_sanitization.log"
	err := InitDebugLogger(testLogPath, false)
	if err != nil {
		t.Fatalf("Failed to initialize debug logger: %v", err)
	}
	defer os.Remove(testLogPath)

	// Log a message with sensitive data
	LogDebug("Connecting with api_key=secret123456789012 and password=MyPassword123")

	// Sync to ensure all data is written
	if debugFile != nil {
		err = debugFile.Sync()
		if err != nil {
			t.Fatalf("Failed to sync debug log: %v", err)
		}
	}

	// Read the debug log to verify sanitization
	logContent, err := os.ReadFile(testLogPath)
	if err != nil {
		t.Fatalf("Failed to read debug log: %v", err)
	}
	logStr := string(logContent)

	// Verify sensitive data was redacted
	if strings.Contains(logStr, "secret123456789012") {
		t.Errorf("API key should have been redacted from log:\n%s", logStr)
	}
	if strings.Contains(logStr, "MyPassword123") {
		t.Errorf("Password should have been redacted from log:\n%s", logStr)
	}

	// Verify redaction markers are present
	if !strings.Contains(logStr, "[REDACTED]") {
		t.Errorf("Expected [REDACTED] markers in log:\n%s", logStr)
	}
}
