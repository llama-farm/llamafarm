package cmd

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
)

// InitDebugLogger initializes a shared file-backed logger and Bubble Tea logging.
// If path is empty, it defaults to "debug.log". Safe to call multiple times.
func InitDebugLogger(path string) error {
	var initErr error
	debugOnce.Do(func() {
		if path == "" {
			cwd := getEffectiveCWD()
			path = filepath.Join(cwd, "debug.log")
		}

		absPath, _ := filepath.Abs(path)

		fmt.Printf(
			"[DEBUG] Logging to: %s\n",
			func() string {
				if absPath != "" {
					return absPath
				}
				return path
			}(),
		)

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

func logDebug(msg string) {
	if !debug {
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
