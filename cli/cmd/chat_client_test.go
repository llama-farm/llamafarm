package cmd

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/llamafarm/cli/cmd/utils"
	"gopkg.in/yaml.v2"
)

func TestChatManager_BuildCurlCommand(t *testing.T) {
	// Test that ChatManager can build proper curl command
	cfg := &ChatConfig{
		ServerURL:   "http://localhost:8000",
		Namespace:   "org",
		ProjectID:   "proj",
		SessionMode: SessionModeStateless,
		RAGEnabled:  false,
	}

	mgr, err := NewChatManager(cfg)
	if err != nil {
		t.Fatalf("failed to create manager: %v", err)
	}

	messages := []Message{{Role: "user", Content: "test"}}
	curlCmd, err := mgr.BuildCurlCommand(messages)
	if err != nil {
		t.Fatalf("failed to build curl: %v", err)
	}

	// Verify the URL is in the curl command
	if !strings.Contains(curlCmd, "http://localhost:8000/v1/projects/org/proj/chat/completions") {
		t.Fatalf("curl command doesn't contain expected URL: %s", curlCmd)
	}

	// Verify it's a POST request
	if !strings.Contains(curlCmd, "curl -X POST") {
		t.Fatalf("curl command doesn't contain POST method: %s", curlCmd)
	}
}

func TestChatManager_Creation(t *testing.T) {
	// Test that ChatManager can be created with proper config
	cfg := &ChatConfig{
		ServerURL:        "http://localhost:8000",
		Namespace:        "ns",
		ProjectID:        "proj",
		SessionMode:      SessionModeProject,
		SessionNamespace: "ns",
		SessionProject:   "proj",
		Temperature:      0.5,
		MaxTokens:        123,
		RAGEnabled:       true,
	}

	mgr, err := NewChatManager(cfg)
	if err != nil {
		t.Fatalf("failed to create manager: %v", err)
	}

	gotCfg := mgr.GetConfig()
	if gotCfg.ServerURL != cfg.ServerURL {
		t.Fatalf("expected ServerURL %s, got %s", cfg.ServerURL, gotCfg.ServerURL)
	}
	if gotCfg.Namespace != cfg.Namespace {
		t.Fatalf("expected Namespace %s, got %s", cfg.Namespace, gotCfg.Namespace)
	}
	if gotCfg.ProjectID != cfg.ProjectID {
		t.Fatalf("expected ProjectID %s, got %s", cfg.ProjectID, gotCfg.ProjectID)
	}
	if !gotCfg.RAGEnabled {
		t.Fatalf("expected RAGEnabled to be true, got false")
	}
}

func TestChatManager_SessionPersistence(t *testing.T) {
	// Create a temporary directory for testing
	tempDir, err := os.MkdirTemp("", "llamafarm_test")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Set LF_DATA_DIR to our temp directory
	origLFDataDir := os.Getenv("LF_DATA_DIR")
	os.Setenv("LF_DATA_DIR", filepath.Join(tempDir, ".llamafarm"))
	defer func() {
		if origLFDataDir != "" {
			os.Setenv("LF_DATA_DIR", origLFDataDir)
		} else {
			os.Unsetenv("LF_DATA_DIR")
		}
	}()

	// Create ChatManager and set a session ID
	cfg := &ChatConfig{
		ServerURL:        "http://localhost:8000",
		Namespace:        "test",
		ProjectID:        "test",
		SessionMode:      SessionModeProject,
		SessionNamespace: "test",
		SessionProject:   "test",
	}

	mgr, err := NewChatManager(cfg)
	if err != nil {
		t.Fatalf("failed to create manager: %v", err)
	}

	// Set a session ID
	testSessionID := "test-session-123"
	err = mgr.SetSessionID(testSessionID)
	if err != nil {
		t.Fatalf("failed to set session ID: %v", err)
	}

	// Verify the session ID was stored
	gotSessionID := mgr.GetSessionID()
	if gotSessionID != testSessionID {
		t.Fatalf("expected session ID %q, got %q", testSessionID, gotSessionID)
	}

	// Verify the context file was written
	lfDir, _ := utils.GetLFDataDir()
	base := filepath.Join(lfDir, "projects", "test", "test", "cli", "context")
	contextFile := filepath.Join(base, "context.yaml")
	if _, err := os.Stat(contextFile); os.IsNotExist(err) {
		t.Fatalf("context.yaml file was not created at %s", contextFile)
	}

	// Read and verify the content
	content, err := os.ReadFile(contextFile)
	if err != nil {
		t.Fatalf("failed to read context file: %v", err)
	}

	var contextData map[string]interface{}
	if err := yaml.Unmarshal(content, &contextData); err != nil {
		t.Fatalf("failed to parse YAML: %v", err)
	}

	// Verify the session ID
	if contextData["session_id"] != testSessionID {
		t.Fatalf("expected session_id %q, got %q", testSessionID, contextData["session_id"])
	}

	// Verify timestamp exists
	timestampStr, ok := contextData["timestamp"].(string)
	if !ok {
		t.Fatalf("timestamp not found or not a string")
	}

	if _, err := time.Parse(time.RFC3339, timestampStr); err != nil {
		t.Fatalf("timestamp not in RFC3339 format: %v", err)
	}

	// Create a new manager and verify it loads the existing session
	mgr2, err := NewChatManager(cfg)
	if err != nil {
		t.Fatalf("failed to create second manager: %v", err)
	}

	got2SessionID := mgr2.GetSessionID()
	if got2SessionID != testSessionID {
		t.Fatalf("expected second manager to load session ID %q, got %q", testSessionID, got2SessionID)
	}
}

func TestChatManager_ClearSession(t *testing.T) {
	// Create a temporary directory for testing
	tempDir, err := os.MkdirTemp("", "llamafarm_test")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Set LF_DATA_DIR to our temp directory
	origLFDataDir := os.Getenv("LF_DATA_DIR")
	os.Setenv("LF_DATA_DIR", filepath.Join(tempDir, ".llamafarm"))
	defer func() {
		if origLFDataDir != "" {
			os.Setenv("LF_DATA_DIR", origLFDataDir)
		} else {
			os.Unsetenv("LF_DATA_DIR")
		}
	}()

	// Create ChatManager with a session
	cfg := &ChatConfig{
		ServerURL:        "http://localhost:8000",
		Namespace:        "test",
		ProjectID:        "test",
		SessionMode:      SessionModeProject,
		SessionNamespace: "test",
		SessionProject:   "test",
	}

	mgr, err := NewChatManager(cfg)
	if err != nil {
		t.Fatalf("failed to create manager: %v", err)
	}

	// Set initial session ID
	initialSessionID := "test-session-initial"
	err = mgr.SetSessionID(initialSessionID)
	if err != nil {
		t.Fatalf("failed to set session ID: %v", err)
	}

	if mgr.GetSessionID() != initialSessionID {
		t.Fatalf("expected initial session ID %q, got %q", initialSessionID, mgr.GetSessionID())
	}

	// Clear the session
	err = mgr.ClearSession()
	if err != nil {
		t.Fatalf("failed to clear session: %v", err)
	}

	// Verify a new session ID was created
	newSessionID := mgr.GetSessionID()
	if newSessionID == "" {
		t.Fatalf("expected new session ID after clear, got empty string")
	}
	if newSessionID == initialSessionID {
		t.Fatalf("expected different session ID after clear, got same ID: %q", newSessionID)
	}

	// Verify the new session ID was persisted
	mgr2, err := NewChatManager(cfg)
	if err != nil {
		t.Fatalf("failed to create second manager: %v", err)
	}

	if mgr2.GetSessionID() != newSessionID {
		t.Fatalf("expected second manager to load new session ID %q, got %q", newSessionID, mgr2.GetSessionID())
	}
}
