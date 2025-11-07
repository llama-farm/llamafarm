package cmd

import (
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"time"

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
	NoEmoji bool      // if true, don't add emoji prefix
}

// TUIMessageMsg is a Bubble Tea message for routing output to the TUI
type TUIMessageMsg struct {
	Message OutputMessage
}

// OutputManager manages all CLI output routing
type OutputManager struct {
	mu                  sync.RWMutex
	tuiProgram          *tea.Program
	inTUIMode           bool
	messageQueue        []OutputMessage
	lastProgressMessage string
	progressMessageSent bool
	disableEmojis       bool // global flag to disable emojis
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

// SetEmojiEnabled controls whether emojis are added to output messages globally
func SetEmojiEnabled(enabled bool) {
	outputManager.mu.Lock()
	defer outputManager.mu.Unlock()
	outputManager.disableEmojis = !enabled
}

// EmojiEnabled returns whether emojis are currently enabled
func EmojiEnabled() bool {
	outputManager.mu.RLock()
	defer outputManager.mu.RUnlock()
	return !outputManager.disableEmojis
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
	sendMessageWithOptions(msgType, false, format, args...)
}

// sendMessageWithOptions routes a message with optional emoji control
func sendMessageWithOptions(msgType MessageType, noEmoji bool, format string, args ...interface{}) {
	content := fmt.Sprintf(format, args...)

	outputManager.mu.RLock()
	disableEmojis := outputManager.disableEmojis
	outputManager.mu.RUnlock()

	msg := OutputMessage{
		Type:    msgType,
		Content: content,
		Writer:  getDefaultWriter(msgType),
		NoEmoji: noEmoji || disableEmojis,
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

// OutputInfoPlain sends an informational message without emoji
func OutputInfoPlain(format string, args ...interface{}) {
	sendMessageWithOptions(InfoMessage, true, format, args...)
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
	outputManager.mu.RLock()
	inTUI := outputManager.inTUIMode
	outputManager.mu.RUnlock()

	if inTUI {
		// In TUI mode, consolidate progress messages to avoid multiple lines
		sendConsolidatedProgressMessage(format, args...)
	} else {
		// In terminal mode, send progress messages normally
		sendMessage(ProgressMessage, format, args...)
	}
}

// sendConsolidatedProgressMessage handles progress messages in TUI mode by updating a single line
func sendConsolidatedProgressMessage(format string, args ...interface{}) {
	content := fmt.Sprintf(format, args...)

	// Remove carriage return characters that are meant for terminal overwriting
	// In TUI mode, we don't want these as they interfere with message display
	content = strings.TrimPrefix(content, "\r")
	content = strings.ReplaceAll(content, "\r", "")

	outputManager.mu.Lock()
	defer outputManager.mu.Unlock()

	// Update the last progress message
	outputManager.lastProgressMessage = content

	// If this is the first progress message or we haven't sent one recently, send it
	if !outputManager.progressMessageSent {
		msg := OutputMessage{
			Type:    ProgressMessage,
			Content: content,
			Writer:  os.Stdout,
		}

		if outputManager.tuiProgram != nil {
			outputManager.tuiProgram.Send(TUIMessageMsg{Message: msg})
		}
		outputManager.progressMessageSent = true

		// Reset the flag after a short delay to allow periodic updates
		go func() {
			time.Sleep(100 * time.Millisecond) // Reduced from 500ms to 100ms for more frequent updates
			outputManager.mu.Lock()
			defer outputManager.mu.Unlock()

			outputManager.progressMessageSent = false

			// If there's a pending progress message that wasn't sent due to throttling, send it now
			if outputManager.lastProgressMessage != "" && outputManager.tuiProgram != nil {
				msg := OutputMessage{
					Type:    ProgressMessage,
					Content: outputManager.lastProgressMessage,
					Writer:  os.Stdout,
				}
				outputManager.tuiProgram.Send(TUIMessageMsg{Message: msg})
			}
		}()
	}
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
	if msg.NoEmoji {
		return msg.Content
	}

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
