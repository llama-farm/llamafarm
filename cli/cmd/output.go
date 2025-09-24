package cmd

import (
	"fmt"
	"io"
	"os"
	"sync"

	tea "github.com/charmbracelet/bubbletea"
)

// MessageType represents the type of output message
type MessageType int

const (
	InfoMessage MessageType = iota
	WarningMessage
	ErrorMessage
	SuccessMessage
	ProgressMessage
	DebugMessage
)

// OutputMessage represents a message to be displayed
type OutputMessage struct {
	Type    MessageType
	Content string
	Writer  io.Writer // fallback writer when not in TUI mode
}

// TUIMessageMsg is a Bubble Tea message for routing output to the TUI
type TUIMessageMsg struct {
	Message OutputMessage
}

// OutputManager manages all CLI output routing
type OutputManager struct {
	mu           sync.RWMutex
	tuiProgram   *tea.Program
	inTUIMode    bool
	messageQueue []OutputMessage
}

var outputManager = &OutputManager{}

// SetTUIMode configures the output manager for TUI mode
func SetTUIMode(program *tea.Program) {
	outputManager.mu.Lock()
	defer outputManager.mu.Unlock()
	outputManager.tuiProgram = program
	outputManager.inTUIMode = true

	// Send any queued messages to the TUI
	for _, msg := range outputManager.messageQueue {
		if program != nil {
			program.Send(TUIMessageMsg{Message: msg})
		}
	}
	outputManager.messageQueue = nil
}

// ClearTUIMode disables TUI mode
func ClearTUIMode() {
	outputManager.mu.Lock()
	defer outputManager.mu.Unlock()
	outputManager.tuiProgram = nil
	outputManager.inTUIMode = false
	outputManager.messageQueue = nil
}

// sendMessage routes a message to the appropriate output destination
func sendMessage(msgType MessageType, format string, args ...interface{}) {
	content := fmt.Sprintf(format, args...)

	msg := OutputMessage{
		Type:    msgType,
		Content: content,
		Writer:  getDefaultWriter(msgType),
	}

	outputManager.mu.RLock()
	inTUI := outputManager.inTUIMode
	program := outputManager.tuiProgram
	outputManager.mu.RUnlock()

	if inTUI && program != nil {
		// Send to TUI
		program.Send(TUIMessageMsg{Message: msg})
	} else if inTUI {
		// TUI mode but no program yet, queue the message
		outputManager.mu.Lock()
		outputManager.messageQueue = append(outputManager.messageQueue, msg)
		outputManager.mu.Unlock()
	} else {
		// Direct output mode
		fmt.Fprint(msg.Writer, FormatMessage(msg))
	}
}

// getDefaultWriter returns the appropriate writer for each message type
func getDefaultWriter(msgType MessageType) io.Writer {
	switch msgType {
	case ErrorMessage, WarningMessage, DebugMessage:
		return os.Stderr
	default:
		return os.Stdout
	}
}

// Public API functions for different message types

// OutputInfo sends an informational message
func OutputInfo(format string, args ...interface{}) {
	sendMessage(InfoMessage, format, args...)
}

// OutputWarning sends a warning message
func OutputWarning(format string, args ...interface{}) {
	sendMessage(WarningMessage, format, args...)
}

// OutputError sends an error message
func OutputError(format string, args ...interface{}) {
	sendMessage(ErrorMessage, format, args...)
}

// OutputSuccess sends a success message
func OutputSuccess(format string, args ...interface{}) {
	sendMessage(SuccessMessage, format, args...)
}

// OutputProgress sends a progress message
func OutputProgress(format string, args ...interface{}) {
	sendMessage(ProgressMessage, format, args...)
}

// OutputDebug sends a debug message (respects the global debug flag)
func OutputDebug(format string, args ...interface{}) {
	if !debug {
		return
	}
	sendMessage(DebugMessage, format, args...)
}

// FormatMessageForTUI formats a message for display in the TUI
func FormatMessage(msg OutputMessage) string {
	var prefix string
	switch msg.Type {
	case InfoMessage:
		prefix = "ℹ️"
	case WarningMessage:
		prefix = "⚠️"
	case ErrorMessage:
		prefix = "❌"
	case SuccessMessage:
		prefix = "✅"
	case ProgressMessage:
		prefix = "🔄"
	case DebugMessage:
		prefix = "🐛"
	}

	return fmt.Sprintf("%s  %s", prefix, msg.Content)
}
