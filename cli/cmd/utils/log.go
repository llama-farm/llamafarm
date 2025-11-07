package utils

import (
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"sync"

	tea "github.com/charmbracelet/bubbletea"
)

var (
	debugOnce   sync.Once
	debugFile   *os.File
	debugLogger *log.Logger
	enableDebug bool = false
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

func LogDebug(msg string) {
	if debugLogger == nil {
		if err := InitDebugLogger("debug.log", enableDebug); err != nil {
			// Use OutputError if available, otherwise fallback to stderr
			OutputError("failed to initialize debug logger: %v\n", err)
		}
	}

	if debugLogger != nil {
		// Always write to file (debugLogger writes to file only, not stderr)
		debugLogger.Println(msg)

		// Only write to stderr when debug mode is enabled
		if enableDebug {
			// Route through the output system for TUI compatibility
			// This writes to stderr as per the project's requirement
			sendMessage(DebugMessage, "%s", msg)
		}
	}
}
