package cmd

import (
	"strings"
	"testing"
)

func TestRenderAssistantContent(t *testing.T) {
	tests := []struct {
		name              string
		input             string
		hasThink          bool
		expectedInContent []string
		notInContent      []string
	}{
		{
			name:              "No think tags",
			input:             "This is a normal response",
			hasThink:          false,
			expectedInContent: []string{"This is a normal response"},
			notInContent:      []string{"<think>", "</think>"},
		},
		{
			name:              "Single think tag",
			input:             "Let me think. <think>Internal reasoning here</think> Here's my answer.",
			hasThink:          true,
			expectedInContent: []string{"Let me think.", "Internal reasoning here", "Here's my answer."},
			notInContent:      []string{"<think>", "</think>"},
		},
		{
			name:              "Multiple think tags",
			input:             "<think>First thought</think> Some text <think>Second thought</think> More text",
			hasThink:          true,
			expectedInContent: []string{"First thought", "Some text", "Second thought", "More text"},
			notInContent:      []string{"<think>", "</think>"},
		},
		{
			name:              "Unclosed think tag",
			input:             "Here's a thought: <think>This continues to the end",
			hasThink:          true,
			expectedInContent: []string{"Here's a thought:", "This continues to the end"},
			notInContent:      []string{"<think>"},
		},
		{
			name:              "Think tag only",
			input:             "<think>Only thinking, no response</think>",
			hasThink:          true,
			expectedInContent: []string{"Only thinking, no response"},
			notInContent:      []string{"<think>", "</think>"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := renderAssistantContent(tt.input, 100)

			// Verify result is not empty
			if result == "" {
				t.Errorf("renderAssistantContent() returned empty string for input: %s", tt.input)
			}

			// Verify expected content is present
			for _, expected := range tt.expectedInContent {
				if !strings.Contains(result, expected) {
					t.Errorf("renderAssistantContent() should contain '%s', got: %s", expected, result)
				}
			}

			// Verify tags are not in the result (they should be parsed and removed)
			for _, notExpected := range tt.notInContent {
				if strings.Contains(result, notExpected) {
					t.Errorf("renderAssistantContent() should not contain '%s', got: %s", notExpected, result)
				}
			}

			// Note: ANSI codes may or may not appear depending on terminal detection
			// The styling is applied via lipgloss, which handles terminal compatibility
		})
	}
}

func TestRenderAssistantContentPreservesNonThinkContent(t *testing.T) {
	input := "Before <think>thinking</think> After"
	result := renderAssistantContent(input, 100)

	// Should contain all the text (tags are part of the parsing, but content is preserved)
	if !strings.Contains(result, "Before") {
		t.Errorf("renderAssistantContent() should preserve content before think tags")
	}
	if !strings.Contains(result, "After") {
		t.Errorf("renderAssistantContent() should preserve content after think tags")
	}
	if !strings.Contains(result, "thinking") {
		t.Errorf("renderAssistantContent() should preserve content inside think tags")
	}
}

func TestRenderToolCallDeterministicOrdering(t *testing.T) {
	// Create a tool call with multiple arguments
	toolCall := ToolCallItem{
		ID:   "test_call_123",
		Type: "function",
		Function: ToolCallFunction{
			Name:      "test_function",
			Arguments: `{"zebra": "last", "apple": "first", "middle": "second", "banana": "third"}`,
		},
	}

	// Render the tool call multiple times
	results := make([]string, 10)
	for i := 0; i < 10; i++ {
		results[i] = renderToolCall(toolCall, 100)
	}

	// All results should be identical (deterministic)
	firstResult := results[0]
	for i, result := range results {
		if result != firstResult {
			t.Errorf("renderToolCall() produced different output on iteration %d.\nExpected: %s\nGot: %s", i, firstResult, result)
		}
	}

	// Verify alphabetical ordering of keys (apple, banana, middle, zebra)
	if !strings.Contains(firstResult, "apple") {
		t.Error("renderToolCall() should contain 'apple' argument")
	}
	
	// Check that 'apple' appears before 'zebra' in the output
	appleIdx := strings.Index(firstResult, "apple")
	zebraIdx := strings.Index(firstResult, "zebra")
	if appleIdx >= zebraIdx {
		t.Errorf("renderToolCall() should show arguments in alphabetical order (apple before zebra)")
	}
	
	// Check that 'banana' appears before 'middle'
	bananaIdx := strings.Index(firstResult, "banana")
	middleIdx := strings.Index(firstResult, "middle")
	if bananaIdx >= middleIdx {
		t.Errorf("renderToolCall() should show arguments in alphabetical order (banana before middle)")
	}
}

