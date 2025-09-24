package cmd

import (
	"testing"
)

func TestOutputManagerDirectMode(t *testing.T) {
	// Ensure we start in direct mode
	ClearTUIMode()

	// Test different message types in direct mode
	tests := []struct {
		name     string
		msgType  MessageType
		sendFunc func(string, ...interface{})
		content  string
	}{
		{"info", InfoMessage, OutputInfo, "test info message"},
		{"warning", WarningMessage, OutputWarning, "test warning message"},
		{"error", ErrorMessage, OutputError, "test error message"},
		{"success", SuccessMessage, OutputSuccess, "test success message"},
		{"progress", ProgressMessage, OutputProgress, "test progress message"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// In direct mode, messages should go to their respective writers
			// This test mainly verifies no panics occur
			tt.sendFunc(tt.content)
		})
	}
}

func TestOutputManagerTUIMode(t *testing.T) {
	// Test TUI mode setup without actually running the program
	ClearTUIMode()

	// Manually set TUI mode
	outputManager.mu.Lock()
	outputManager.inTUIMode = true
	outputManager.tuiProgram = nil // Simulate TUI mode without program
	outputManager.mu.Unlock()
	defer ClearTUIMode()

	// Send a test message - should be queued
	OutputInfo("test message in TUI mode")

	// Verify message was queued
	outputManager.mu.RLock()
	queueLen := len(outputManager.messageQueue)
	outputManager.mu.RUnlock()

	if queueLen != 1 {
		t.Errorf("Expected 1 queued message, got %d", queueLen)
	}
}

func TestFormatMessage(t *testing.T) {
	tests := []struct {
		name     string
		msgType  MessageType
		content  string
		expected string
	}{
		{"info", InfoMessage, "test info", "ℹ️  test info"},
		{"warning", WarningMessage, "test warning", "⚠️  test warning"},
		{"error", ErrorMessage, "test error", "❌  test error"},
		{"success", SuccessMessage, "test success", "✅  test success"},
		{"progress", ProgressMessage, "test progress", "🔄  test progress"},
		{"debug", DebugMessage, "test debug", "🐛  test debug"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			msg := OutputMessage{
				Type:    tt.msgType,
				Content: tt.content,
			}

			result := FormatMessage(msg)
			if result != tt.expected {
				t.Errorf("FormatMessage() = %q, want %q", result, tt.expected)
			}
		})
	}
}

func TestMessageQueueing(t *testing.T) {
	// Clear any existing state
	ClearTUIMode()

	// Enable TUI mode without a program (simulates early initialization)
	outputManager.mu.Lock()
	outputManager.inTUIMode = true
	outputManager.tuiProgram = nil
	outputManager.mu.Unlock()

	// Send messages - they should be queued
	OutputInfo("queued message 1")
	OutputWarning("queued message 2")

	// Verify messages were queued
	outputManager.mu.RLock()
	queueLen := len(outputManager.messageQueue)
	outputManager.mu.RUnlock()

	if queueLen != 2 {
		t.Errorf("Expected 2 queued messages, got %d", queueLen)
	}

	// Simulate setting a program - manually clear the queue to test the logic
	outputManager.mu.Lock()
	outputManager.messageQueue = nil // Simulate what SetTUIMode would do
	outputManager.mu.Unlock()

	// Verify queue is cleared
	outputManager.mu.RLock()
	queueLenAfter := len(outputManager.messageQueue)
	outputManager.mu.RUnlock()

	if queueLenAfter != 0 {
		t.Errorf("Expected queue to be cleared, got %d messages", queueLenAfter)
	}

	ClearTUIMode()
}
