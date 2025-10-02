package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"gopkg.in/yaml.v2"
)

func TestBuildChatAPIURL(t *testing.T) {
	ctx := &ChatSessionContext{ServerURL: "http://localhost:8000"}
	var got string
	// Inference path when no ns/project
	_, err := buildChatAPIURL(ctx)
	if err == nil {
		t.Fatalf("expected error, got %v", err)
	}

	// Project-scoped path when ns/project provided
	ctx.Namespace = "org"
	ctx.ProjectID = "proj"
	got, err = buildChatAPIURL(ctx)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	want := "http://localhost:8000/v1/projects/org/proj/chat/completions"
	if got != want {
		t.Fatalf("expected %q, got %q", want, got)
	}
}

func TestNewDefaultContextFromGlobals(t *testing.T) {
	serverURL = "http://localhost:8000"
	namespace = "ns"
	projectID = "proj"
	sessionID = "sess"
	temperature = 0.5
	maxTokens = 123
	streaming = true

	ctx := newDefaultContextFromGlobals()
	if ctx.ServerURL != serverURL || ctx.Namespace != namespace || ctx.ProjectID != projectID || ctx.SessionID != sessionID || ctx.Temperature != temperature || ctx.MaxTokens != maxTokens || ctx.Streaming != streaming {
		t.Fatalf("context not initialized from globals correctly")
	}
	if ctx.HTTPClient == nil {
		t.Fatalf("expected HTTPClient set")
	}
	if ctx.SessionMode != SessionModeProject {
		t.Fatalf("expected session mode SessionModeProject, got %v", ctx.SessionMode)
	}
	if ctx.SessionNamespace != namespace || ctx.SessionProject != projectID {
		t.Fatalf("expected session namespace/project to mirror globals")
	}
}

func TestWriteSessionContext(t *testing.T) {
	// Create a temporary directory for testing
	tempDir, err := os.MkdirTemp("", "llamafarm_test")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Mock getLFDataDir to return our temp directory
	origGetLFDataDir := getLFDataDir
	getLFDataDir = func() (string, error) {
		return filepath.Join(tempDir, ".llamafarm"), nil
	}
	defer func() { getLFDataDir = origGetLFDataDir }()

	// Change to temp directory
	originalCwd, _ := os.Getwd()
	defer os.Chdir(originalCwd)
	os.Chdir(tempDir)

	// Test writing session context
	testSessionID := "test-session-123"
	ctx := &ChatSessionContext{SessionMode: SessionModeProject, Namespace: "test", ProjectID: "test"}
	err = writeSessionContext(ctx, testSessionID)
	if err != nil {
		t.Fatalf("failed to write session context: %v", err)
	}

	// Verify projects directory and context file were created at the expected path
	lfDir, _ := getLFDataDir()
	base := filepath.Join(lfDir, "projects", "test", "test", "cli", "context")
	if _, err := os.Stat(base); os.IsNotExist(err) {
		t.Fatalf("projects/test/test/cli/context directory was not created")
	}
	contextFile := filepath.Join(base, "context.yaml")
	if _, err := os.Stat(contextFile); os.IsNotExist(err) {
		t.Fatalf("context.yaml file was not created at %s", contextFile)
	}

	// Read and verify the content
	content, err := os.ReadFile(contextFile)
	if err != nil {
		t.Fatalf("failed to read context file: %v", err)
	}

	// Parse the YAML content
	var contextData map[string]interface{}
	if err := yaml.Unmarshal(content, &contextData); err != nil {
		t.Fatalf("failed to parse YAML: %v", err)
	}

	// Verify the session ID
	if contextData["session_id"] != testSessionID {
		t.Fatalf("expected session_id %q, got %q", testSessionID, contextData["session_id"])
	}

	// Verify timestamp exists and is a valid RFC3339 timestamp
	timestampStr, ok := contextData["timestamp"].(string)
	if !ok {
		t.Fatalf("timestamp not found or not a string")
	}

	if _, err := time.Parse(time.RFC3339, timestampStr); err != nil {
		t.Fatalf("timestamp not in RFC3339 format: %v", err)
	}
}

func TestReadSessionContext(t *testing.T) {
	// Create a temporary directory for testing
	tempDir, err := os.MkdirTemp("", "llamafarm_test")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Mock getLFDataDir to return our temp directory
	origGetLFDataDir := getLFDataDir
	getLFDataDir = func() (string, error) {
		return filepath.Join(tempDir, ".llamafarm"), nil
	}
	defer func() { getLFDataDir = origGetLFDataDir }()

	// Test reading non-existent context file
	ctx := &ChatSessionContext{SessionMode: SessionModeProject, Namespace: "test", ProjectID: "test"}
	context, err := readSessionContext(ctx)
	if err != nil {
		t.Fatalf("expected no error for non-existent file, got: %v", err)
	}
	if context != nil {
		t.Fatalf("expected nil context for non-existent file, got: %v", context)
	}

	// Create a valid context file at the expected session path
	sessionPath, err := ctx.sessionFilePath()
	if err != nil {
		t.Fatalf("failed to compute session file path: %v", err)
	}
	if err := os.MkdirAll(filepath.Dir(sessionPath), 0755); err != nil {
		t.Fatalf("failed to create context dir: %v", err)
	}

	contextFile := sessionPath
	testSessionID := "test-session-456"
	testTimestamp := "2024-01-15T10:30:45Z"
	yamlContent := fmt.Sprintf("session_id: %s\ntimestamp: %s\n", testSessionID, testTimestamp)

	if err := os.WriteFile(contextFile, []byte(yamlContent), 0644); err != nil {
		t.Fatalf("failed to write test context file: %v", err)
	}

	// Test reading valid context file
	context, err = readSessionContext(ctx)
	if err != nil {
		t.Fatalf("expected no error for valid file, got: %v", err)
	}
	if context == nil {
		t.Fatalf("expected context to be non-nil")
	}
	if context.SessionID != testSessionID {
		t.Fatalf("expected session_id %q, got %q", testSessionID, context.SessionID)
	}
	if context.Timestamp != testTimestamp {
		t.Fatalf("expected timestamp %q, got %q", testTimestamp, context.Timestamp)
	}

	// Test reading invalid YAML
	invalidYamlContent := "invalid: yaml: content: [\n"
	if err := os.WriteFile(contextFile, []byte(invalidYamlContent), 0644); err != nil {
		t.Fatalf("failed to write invalid test context file: %v", err)
	}

	context, err = readSessionContext(ctx)
	if err == nil {
		t.Fatalf("expected error for invalid YAML, got nil")
	}
	if context != nil {
		t.Fatalf("expected nil context for invalid YAML, got: %v", context)
	}

	// Test reading file with empty session ID
	emptySessionYaml := "session_id: \"\"\ntimestamp: 2024-01-15T10:30:45Z\n"
	if err := os.WriteFile(contextFile, []byte(emptySessionYaml), 0644); err != nil {
		t.Fatalf("failed to write empty session test file: %v", err)
	}

	context, err = readSessionContext(ctx)
	if err != nil {
		t.Fatalf("expected no error for empty session ID, got: %v", err)
	}
	if context != nil {
		t.Fatalf("expected nil context for empty session ID, got: %v", context)
	}
}
