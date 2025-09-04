package cmd

import (
	"fmt"
	"io"
	"log"
	"os"
	"sync"

	tea "github.com/charmbracelet/bubbletea"
)

var (
	debugOnce   sync.Once
	debugFile   *os.File
	debugLogger *log.Logger
)

// InitDebugLogger initializes a shared file-backed logger and Bubble Tea logging.
// If path is empty, it defaults to "debug.log". Safe to call multiple times.
func InitDebugLogger(path string) error {
	var initErr error
	debugOnce.Do(func() {
		if path == "" {
			path = "debug.log"
		}

		// Use Bubble Tea's LogToFile which handles file creation and setup properly
		f, err := tea.LogToFile(path, "debug")
		if err != nil {
			initErr = err
			return
		}

		// Store the file handle for proper cleanup
		debugFile = f
		// Bubble Tea's LogToFile already sets up a logger, but we create our own
		// that writes to both the file and stderr as per project requirements
		debugLogger = log.New(io.MultiWriter(f, os.Stderr), "", log.LstdFlags)
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

func logDebug(msg string) {
	if len(os.Getenv("DEBUG")) == 0 {
		return
	}
	if debugLogger == nil {
		if err := InitDebugLogger("debug.log"); err != nil {
			fmt.Fprintln(os.Stderr, "failed to initialize debug logger:", err)
		}
	}
	if debugLogger != nil {
		// Write to both stderr and the file to meet the project's requirement
		// that debug messages go to stderr while also persisting to disk.
		debugLogger.Println(msg)
	}
}
