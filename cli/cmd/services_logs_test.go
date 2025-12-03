package cmd

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestIsValidService(t *testing.T) {
	validServices := []string{"server", "rag", "universal-runtime"}

	tests := []struct {
		name     string
		service  string
		expected bool
	}{
		{"valid server", "server", true},
		{"valid rag", "rag", true},
		{"valid universal-runtime", "universal-runtime", true},
		{"invalid service", "invalid", false},
		{"empty service", "", false},
		{"random string", "foobar", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isValidService(tt.service, validServices)
			if result != tt.expected {
				t.Errorf("isValidService(%q) = %v, want %v", tt.service, result, tt.expected)
			}
		})
	}
}

func TestGetServiceLogFile(t *testing.T) {
	tests := []struct {
		name        string
		serviceName string
		wantSuffix  string
	}{
		{"server log", "server", ".llamafarm/logs/server.log"},
		{"rag log", "rag", ".llamafarm/logs/rag.log"},
		{"runtime log", "universal-runtime", ".llamafarm/logs/universal-runtime.log"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logFile, err := getServiceLogFile(tt.serviceName)
			if err != nil {
				t.Fatalf("getServiceLogFile(%q) returned error: %v", tt.serviceName, err)
			}

			// Check that the path ends with the expected suffix
			if !filepath.IsAbs(logFile) {
				t.Errorf("getServiceLogFile(%q) = %q, want absolute path", tt.serviceName, logFile)
			}

			if filepath.Base(logFile) != tt.serviceName+".log" {
				t.Errorf("getServiceLogFile(%q) = %q, want basename %q", tt.serviceName, logFile, tt.serviceName+".log")
			}
		})
	}
}

func TestDisplayTailLines(t *testing.T) {
	// Create a temporary log file with known content
	tmpDir := t.TempDir()
	logFile := filepath.Join(tmpDir, "test.log")

	// Write 10 lines to the file
	content := ""
	for i := 1; i <= 10; i++ {
		content += "Line " + string(rune('0'+i)) + "\n"
	}
	if err := os.WriteFile(logFile, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to create test log file: %v", err)
	}

	tests := []struct {
		name      string
		tailLines int
		wantLines int
	}{
		{"tail 3 lines", 3, 3},
		{"tail 5 lines", 5, 5},
		{"tail 15 lines (more than available)", 15, 10},
		{"tail 0 lines", 0, 10}, // Special case: 0 means all
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			file, err := os.Open(logFile)
			if err != nil {
				t.Fatalf("Failed to open test log file: %v", err)
			}
			defer file.Close()

			// We can't easily test the output without capturing stdout,
			// but we can at least verify the function doesn't error
			if tt.tailLines == 0 {
				// For tail 0, just verify the file can be read
				return
			}

			err = displayTailLines(file, tt.tailLines, "")
			if err != nil {
				t.Errorf("displayTailLines(%d) returned error: %v", tt.tailLines, err)
			}
		})
	}
}

func TestGetServicePrefix(t *testing.T) {
	tests := []struct {
		name        string
		service     string
		wantContain string
	}{
		{"server prefix", "server", "server"},
		{"rag prefix", "rag", "rag"},
		{"runtime prefix", "universal-runtime", "universal-runtime"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			prefix := getServicePrefix(tt.service)
			if !strings.Contains(prefix, tt.wantContain) {
				t.Errorf("getServicePrefix(%q) = %q, want to contain %q", tt.service, prefix, tt.wantContain)
			}

			// Verify no ANSI color codes in output
			if strings.Contains(prefix, "\033[") {
				t.Errorf("getServicePrefix(%q) contains ANSI color codes: %q", tt.service, prefix)
			}

			// Verify it has bracket format
			if !strings.HasPrefix(prefix, "[") || !strings.Contains(prefix, "]") {
				t.Errorf("getServicePrefix(%q) = %q, want bracket format [service]", tt.service, prefix)
			}
		})
	}
}

func TestExtractTimestamp(t *testing.T) {
	tests := []struct {
		name      string
		line      string
		wantValid bool
	}{
		{
			name:      "valid timestamp",
			line:      "[2025-11-17 11:49:02] [stderr] test message",
			wantValid: true,
		},
		{
			name:      "invalid format",
			line:      "no timestamp here",
			wantValid: false,
		},
		{
			name:      "empty line",
			line:      "",
			wantValid: false,
		},
		{
			name:      "partial timestamp",
			line:      "[2025-11-17",
			wantValid: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ts := extractTimestamp(tt.line)
			isValid := !ts.IsZero()

			if isValid != tt.wantValid {
				t.Errorf("extractTimestamp(%q) valid = %v, want %v", tt.line, isValid, tt.wantValid)
			}
		})
	}
}
