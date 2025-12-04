package utils

import (
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"sync"

	tea "github.com/charmbracelet/bubbletea"
)

var (
	debugOnce   sync.Once
	debugFile   *os.File
	debugLogger *log.Logger
	enableDebug bool = false

	// Compiled regex patterns for sanitization (compiled once at startup)
	// NOTE: Order matters! More specific patterns should come before generic ones.
	sensitivePatterns = []struct {
		pattern     *regexp.Regexp
		replacement string
	}{
		// JWT tokens (must come before generic token patterns) - three base64 segments separated by dots
		{regexp.MustCompile(`\beyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+`), "[REDACTED-JWT]"},
		// OpenAI-style keys (sk-, pk-, sess-) - specific format with project/org prefixes
		{regexp.MustCompile(`\b(sk|pk|sess)-[a-zA-Z0-9\-_]{20,}`), "[REDACTED-KEY]"},
		// AWS access keys
		{regexp.MustCompile(`\bAKIA[A-Z0-9]{16}\b`), "[REDACTED-AWS-KEY]"},
		// Private keys (PEM format indicators)
		{regexp.MustCompile(`(?i)-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----[\s\S]*?-----END\s+(RSA\s+)?PRIVATE\s+KEY-----`), "[REDACTED-PRIVATE-KEY]"},
		// Authorization header values
		{regexp.MustCompile(`(?i)(authorization[=:\s]+['"]?)(Basic|Bearer|Digest)\s+[a-zA-Z0-9\-_\.=]+`), "${1}${2} [REDACTED]"},
		// Bearer tokens (standalone)
		{regexp.MustCompile(`(?i)(bearer\s+)[a-zA-Z0-9\-_\.]+`), "${1}[REDACTED]"},
		// API keys (common formats)
		{regexp.MustCompile(`(?i)(api[_-]?key[=:\s]+['"]?)[a-zA-Z0-9\-_]{16,}`), "${1}[REDACTED]"},
		{regexp.MustCompile(`(?i)(apikey[=:\s]+['"]?)[a-zA-Z0-9\-_]{16,}`), "${1}[REDACTED]"},
		// Passwords in URLs or params
		{regexp.MustCompile(`(?i)(password[=:\s]+['"]?)[^\s&'"]+`), "${1}[REDACTED]"},
		{regexp.MustCompile(`(?i)(passwd[=:\s]+['"]?)[^\s&'"]+`), "${1}[REDACTED]"},
		{regexp.MustCompile(`(?i)(pwd[=:\s]+['"]?)[^\s&'"]+`), "${1}[REDACTED]"},
		// Access/Refresh tokens (specific, before generic token)
		{regexp.MustCompile(`(?i)(access[_-]?token[=:\s]+['"]?)[a-zA-Z0-9\-_\.]{16,}`), "${1}[REDACTED]"},
		{regexp.MustCompile(`(?i)(refresh[_-]?token[=:\s]+['"]?)[a-zA-Z0-9\-_\.]{16,}`), "${1}[REDACTED]"},
		// Tokens (generic - keep after more specific token patterns)
		{regexp.MustCompile(`(?i)(token[=:\s]+['"]?)[a-zA-Z0-9\-_\.]{16,}`), "${1}[REDACTED]"},
		// Session IDs
		{regexp.MustCompile(`(?i)(session[_-]?id[=:\s]+['"]?)[a-zA-Z0-9\-_]{16,}`), "${1}[REDACTED]"},
		{regexp.MustCompile(`(?i)(sid[=:\s]+['"]?)[a-zA-Z0-9\-_]{16,}`), "${1}[REDACTED]"},
		// Cookies (basic pattern)
		{regexp.MustCompile(`(?i)(cookie[=:\s]+['"]?)[^;\n]+`), "${1}[REDACTED]"},
	}
)

// InitDebugLogger initializes a shared file-backed logger and Bubble Tea logging.
// If path is empty, it defaults to "debug.log". Safe to call multiple times.
func InitDebugLogger(path string, debug bool) error {
	enableDebug = debug
	var initErr error
	debugOnce.Do(func() {
		if path == "" {
			cwd := GetEffectiveCWD()
			path = filepath.Join(cwd, "debug.log")
		}

		absPath, _ := filepath.Abs(path)

		if debug {
			fmt.Printf(
				"[DEBUG] Logging to: %s\n",
				func() string {
					if absPath != "" {
						return absPath
					}
					return path
				}(),
			)
		}

		// Use Bubble Tea's LogToFile which handles file creation and setup properly
		f, err := tea.LogToFile(path, "debug")
		if err != nil {
			initErr = err
			return
		}

		// Store the file handle for proper cleanup
		debugFile = f

		debugLogger = log.New(io.MultiWriter(f), "", log.LstdFlags)
	})
	return initErr
}

// CloseDebugLogger closes the underlying debug log file if it was opened.
func CloseDebugLogger() {
	if debugFile != nil {
		_ = debugFile.Sync() // Ensure all data is written to disk
		_ = debugFile.Close()
	}
}

// ResetDebugLoggerForTesting resets the debug logger state for testing purposes.
// This allows tests to reinitialize the logger with different file paths.
// WARNING: This should ONLY be called from tests!
func ResetDebugLoggerForTesting() {
	CloseDebugLogger()
	debugOnce = sync.Once{}
	debugFile = nil
	debugLogger = nil
}

// sanitizeLogMessage applies regex patterns to redact sensitive information from log messages.
// This provides defense-in-depth protection against accidental logging of credentials.
func sanitizeLogMessage(msg string) string {
	sanitized := msg
	for _, sp := range sensitivePatterns {
		sanitized = sp.pattern.ReplaceAllString(sanitized, sp.replacement)
	}
	return sanitized
}

// LogDebug writes a debug message to the debug log file and optionally to stderr.
// SECURITY: This function automatically sanitizes common sensitive patterns (API keys,
// tokens, passwords, etc.), but callers should still avoid logging sensitive data when
// possible. See LogHeaders() for an example of explicit redaction before logging.
func LogDebug(msg string) {
	if debugLogger == nil {
		if err := InitDebugLogger("debug.log", enableDebug); err != nil {
			// Use OutputError if available, otherwise fallback to stderr
			OutputError("failed to initialize debug logger: %v\n", err)
		}
	}

	if debugLogger != nil {
		// Sanitize the message to remove sensitive patterns
		sanitized := sanitizeLogMessage(msg)

		// Always write to file (debugLogger writes to file only, not stderr)
		debugLogger.Println(sanitized)

		// Only write to stderr when debug mode is enabled
		if enableDebug {
			// Route through the output system for TUI compatibility
			// This writes to stderr as per the project's requirement
			sendMessage(DebugMessage, "%s", sanitized)
		}
	}
}
