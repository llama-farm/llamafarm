package cmd

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// ToolHandler defines the interface for CLI tool implementations
type ToolHandler interface {
	// Name returns the full tool name (e.g., "cli.dataset_upload")
	Name() string
	// Execute runs the tool with the given arguments and returns a result
	Execute(args map[string]interface{}, ctx *ChatSessionContext) (string, error)
}

// ToolRegistry maintains registered tool handlers
type ToolRegistry struct {
	handlers map[string]ToolHandler
}

// NewToolRegistry creates a new tool registry with default handlers
func NewToolRegistry() *ToolRegistry {
	registry := &ToolRegistry{
		handlers: make(map[string]ToolHandler),
	}

	// Register default handlers
	registry.Register(&DatasetUploadHandler{})

	return registry
}

// Register adds a tool handler to the registry
func (r *ToolRegistry) Register(handler ToolHandler) {
	r.handlers[handler.Name()] = handler
}

// Get retrieves a handler by name, returns nil if not found
func (r *ToolRegistry) Get(name string) ToolHandler {
	return r.handlers[name]
}

// CanHandle returns true if the registry has a handler for the given tool name
func (r *ToolRegistry) CanHandle(name string) bool {
	return r.handlers[name] != nil
}

// Global tool registry instance
var globalToolRegistry = NewToolRegistry()

// DatasetUploadHandler implements the cli.dataset_upload tool
type DatasetUploadHandler struct{}

func (h *DatasetUploadHandler) Name() string {
	return "cli.dataset_upload"
}

func (h *DatasetUploadHandler) Execute(args map[string]interface{}, ctx *ChatSessionContext) (string, error) {
	// Extract required parameters
	filepath, ok := args["filepath"].(string)
	if !ok || filepath == "" {
		return "", fmt.Errorf("missing or invalid 'filepath' parameter")
	}

	namespace, ok := args["namespace"].(string)
	if !ok || namespace == "" {
		return "", fmt.Errorf("missing or invalid 'namespace' parameter")
	}

	project, ok := args["project"].(string)
	if !ok || project == "" {
		return "", fmt.Errorf("missing or invalid 'project' parameter")
	}

	dataset, ok := args["dataset"].(string)
	if !ok || dataset == "" {
		return "", fmt.Errorf("missing or invalid 'dataset' parameter")
	}

	// Validate file exists
	if _, err := os.Stat(filepath); os.IsNotExist(err) {
		return "", fmt.Errorf("file does not exist: %s", filepath)
	}

	// Upload the file using existing function
	result := uploadFileToDataset(ctx.ServerURL, namespace, project, dataset, filepath)
	if result.err != nil {
		return "", fmt.Errorf("failed to upload file: %w", result.err)
	}

	// Return success message
	return fmt.Sprintf("Successfully uploaded file '%s' to dataset '%s'", filepath, dataset), nil
}

// ToolCall represents a parsed tool call from the API
type ToolCall struct {
	ID        string
	Name      string
	Arguments string
}

// ParseToolCallMessage parses a [TOOL_CALL] formatted message
// Format: [TOOL_CALL]name|id|arguments
func ParseToolCallMessage(msg string) (*ToolCall, error) {
	if !strings.HasPrefix(msg, "[TOOL_CALL]") {
		return nil, fmt.Errorf("not a tool call message")
	}

	parts := strings.SplitN(strings.TrimPrefix(msg, "[TOOL_CALL]"), "|", 3)
	if len(parts) < 3 {
		return nil, fmt.Errorf("invalid tool call format")
	}

	return &ToolCall{
		Name:      parts[0],
		ID:        parts[1],
		Arguments: parts[2],
	}, nil
}

// ExecuteToolCall executes a tool call if it's a CLI tool and returns the result message
func ExecuteToolCall(tc *ToolCall, ctx *ChatSessionContext) (*Message, error) {
	// Check if this is a CLI tool
	if !strings.HasPrefix(tc.Name, "cli.") {
		return nil, nil // Not a CLI tool, skip
	}

	// Check if we have a handler for this tool
	handler := globalToolRegistry.Get(tc.Name)
	if handler == nil {
		return nil, fmt.Errorf("no handler registered for tool: %s", tc.Name)
	}

	// Parse arguments
	var args map[string]interface{}
	if err := json.Unmarshal([]byte(tc.Arguments), &args); err != nil {
		return nil, fmt.Errorf("failed to parse tool arguments: %w", err)
	}

	// Execute the tool
	result, err := handler.Execute(args, ctx)
	if err != nil {
		return nil, fmt.Errorf("tool execution failed: %w", err)
	}

	// Return tool result message
	return &Message{
		Role:       "tool",
		Content:    result,
		ToolCallID: tc.ID,
	}, nil
}
