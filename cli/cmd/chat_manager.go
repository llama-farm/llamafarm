package cmd

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/llamafarm/cli/cmd/utils"
	"gopkg.in/yaml.v2"
)

// ChatManager centralizes all chat operations and state management.
// It provides a unified interface for both command-line and TUI chat interactions.
type ChatManager struct {
	// Configuration (lock-free reads via atomic.Value)
	config     atomic.Value // stores *ChatConfig
	httpClient utils.HTTPClient

	// Session management
	sessionMu   sync.RWMutex
	sessionID   string
	sessionFile string

	// Streaming state
	activeMu     sync.Mutex
	activeCancel context.CancelFunc
}

// ChatConfig holds all configuration for a chat session
type ChatConfig struct {
	// Target project
	ServerURL string
	Namespace string
	ProjectID string

	// Session configuration
	SessionMode      SessionMode
	SessionNamespace string // Where to store session (may differ from target)
	SessionProject   string

	// Model parameters
	Model       string
	Temperature float64
	MaxTokens   int

	// RAG configuration
	RAGEnabled           bool
	RAGDatabase          string
	RAGRetrievalStrategy string
	RAGTopK              int
	RAGScoreThreshold    float64
}

// StreamChunk represents a piece of the streaming response
type StreamChunk struct {
	Type     ChunkType
	Content  string
	ToolCall *ToolCall // Only set for tool call chunks
	Error    error     // Only set for error chunks
}

// ChunkType identifies the type of streaming chunk
type ChunkType int

const (
	ChunkTypeContent ChunkType = iota
	ChunkTypeToolCall
	ChunkTypeError
	ChunkTypeDone
)

// NewChatManager creates a manager with the given configuration
func NewChatManager(cfg *ChatConfig) (*ChatManager, error) {
	if cfg.ServerURL == "" || cfg.Namespace == "" || cfg.ProjectID == "" {
		return nil, fmt.Errorf("server URL, namespace, and project are required")
	}

	mgr := &ChatManager{
		httpClient: utils.GetHTTPClient(),
	}
	mgr.config.Store(cfg)

	// Initialize session file path
	sessionFile, err := mgr.computeSessionFilePath()
	if err != nil {
		return nil, fmt.Errorf("failed to compute session path: %w", err)
	}
	mgr.sessionFile = sessionFile

	// Load existing session if available
	if cfg.SessionMode != SessionModeStateless {
		mgr.loadSession()
	}

	return mgr, nil
}

// SendMessage sends a single message and returns the complete response
func (m *ChatManager) SendMessage(msg string) (string, error) {
	messages := []Message{{Role: "user", Content: msg}}
	return m.SendMessages(messages)
}

// SendMessages sends multiple messages and returns the complete response
func (m *ChatManager) SendMessages(messages []Message) (string, error) {
	var result strings.Builder
	err := m.StreamMessages(messages, func(chunk StreamChunk) error {
		if chunk.Type == ChunkTypeContent {
			result.WriteString(chunk.Content)
		}
		return nil
	})

	return result.String(), err
}

// StreamMessages sends messages and calls handler for each chunk.
// The handler is called sequentially for each chunk and should return quickly.
func (m *ChatManager) StreamMessages(messages []Message, handler func(StreamChunk) error) error {
	// Ensure we have a session ID if not stateless
	cfg := m.GetConfig()
	if cfg.SessionMode != SessionModeStateless && m.GetSessionID() == "" {
		m.sessionMu.Lock()
		m.sessionID = uuid.New().String()
		m.sessionMu.Unlock()
		if err := m.saveSession(); err != nil {
			utils.LogDebug(fmt.Sprintf("Failed to save new session: %v", err))
		}
	}

	// Build HTTP request
	req, err := m.buildHTTPRequest(messages)
	if err != nil {
		return fmt.Errorf("failed to build request: %w", err)
	}

	// Execute streaming request
	ctx, cancel := context.WithCancel(context.Background())
	m.activeMu.Lock()
	m.activeCancel = cancel
	m.activeMu.Unlock()

	defer func() {
		m.activeMu.Lock()
		m.activeCancel = nil
		m.activeMu.Unlock()
	}()

	return m.executeStreamingRequest(ctx, req, handler)
}

// Cancel aborts the active streaming request
func (m *ChatManager) Cancel() {
	m.activeMu.Lock()
	if m.activeCancel != nil {
		m.activeCancel()
	}
	m.activeMu.Unlock()
}

// GetSessionID returns the current session ID (thread-safe)
func (m *ChatManager) GetSessionID() string {
	m.sessionMu.RLock()
	defer m.sessionMu.RUnlock()
	return m.sessionID
}

// SetSessionID updates the session ID (thread-safe)
func (m *ChatManager) SetSessionID(id string) error {
	m.sessionMu.Lock()
	m.sessionID = id
	m.sessionMu.Unlock()
	return m.saveSession()
}

// ClearSession deletes the current session and creates a new one
func (m *ChatManager) ClearSession() error {
	m.sessionMu.Lock()
	oldSessionID := m.sessionID
	m.sessionMu.Unlock()

	// Delete server-side session
	if oldSessionID != "" {
		cfg := m.GetConfig()
		deleteURL := fmt.Sprintf("%s/v1/projects/%s/%s/chat/sessions/%s",
			strings.TrimSuffix(cfg.ServerURL, "/"),
			cfg.Namespace,
			cfg.ProjectID,
			oldSessionID)

		req, err := http.NewRequest("DELETE", deleteURL, nil)
		if err == nil {
			resp, err := m.httpClient.Do(req)
			if err == nil {
				resp.Body.Close()
				utils.LogDebug(fmt.Sprintf("Deleted server session %s", oldSessionID))
			}
		}
	}

	// Create new session
	m.sessionMu.Lock()
	m.sessionID = uuid.New().String()
	m.sessionMu.Unlock()

	return m.saveSession()
}

// FetchHistory retrieves chat history from the server
func (m *ChatManager) FetchHistory() ([]Message, error) {
	sessionID := m.GetSessionID()
	if sessionID == "" {
		return nil, nil
	}

	cfg := m.GetConfig()
	url := fmt.Sprintf("%s/v1/projects/%s/%s/chat/sessions/%s/history",
		strings.TrimSuffix(cfg.ServerURL, "/"),
		cfg.Namespace,
		cfg.ProjectID,
		sessionID)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, err
	}

	resp, err := m.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("server returned %d", resp.StatusCode)
	}

	var result struct {
		Messages []SessionHistoryMessage `json:"messages"`
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if err := json.Unmarshal(body, &result); err != nil {
		return nil, err
	}

	// Convert to Message format, preserving tool call structure
	var messages []Message
	for _, msg := range result.Messages {
		message := Message{
			Role:       msg.Role,
			Content:    msg.Content,
			ToolCallID: msg.ToolCallID, // Preserve tool_call_id for tool result messages
		}

		// Convert tool calls to the Message format
		if len(msg.ToolCalls) > 0 {
			message.ToolCalls = make([]ToolCallItem, len(msg.ToolCalls))
			for i, tc := range msg.ToolCalls {
				message.ToolCalls[i] = ToolCallItem{
					ID:   tc.ID,
					Type: tc.Type,
					Function: ToolCallFunction{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				}
			}
		}

		messages = append(messages, message)
	}

	return messages, nil
}

// GetConfig returns a copy of the current configuration (thread-safe, lock-free)
func (m *ChatManager) GetConfig() ChatConfig {
	return *m.config.Load().(*ChatConfig)
}

// UpdateConfig updates specific configuration fields (thread-safe)
// Creates a new config copy with updates applied
func (m *ChatManager) UpdateConfig(updateFn func(*ChatConfig)) {
	// Loop until we successfully update (handles rare race with concurrent updates)
	for {
		old := m.config.Load().(*ChatConfig)
		// Create a copy to modify
		updated := *old
		updateFn(&updated)
		// Atomically swap if nothing changed in the meantime
		if m.config.CompareAndSwap(old, &updated) {
			return
		}
	}
}

// BuildCurlCommand generates a curl command for debugging
func (m *ChatManager) BuildCurlCommand(messages []Message) (string, error) {
	// Capture current config (lock-free read)
	cfg := m.GetConfig()

	url, err := m.buildChatAPIURL()
	if err != nil {
		return "", err
	}

	streamTrue := true
	var filteredMessages []Message
	for _, msg := range messages {
		if msg.Role != "client" && msg.Role != "error" {
			filteredMessages = append(filteredMessages, msg)
		}
	}

	request := ChatRequest{Messages: filteredMessages, Stream: &streamTrue}

	// Include model if specified
	if cfg.Model != "" {
		request.Model = &cfg.Model
	}

	if cfg.RAGEnabled {
		request.RAGEnabled = &cfg.RAGEnabled
		if cfg.RAGDatabase != "" {
			request.RAGDatabase = &cfg.RAGDatabase
		}
		if cfg.RAGRetrievalStrategy != "" {
			request.RAGRetrievalStrategy = &cfg.RAGRetrievalStrategy
		}
		if cfg.RAGTopK > 0 {
			request.RAGTopK = &cfg.RAGTopK
		}
		if cfg.RAGScoreThreshold > 0 {
			request.RAGScoreThreshold = &cfg.RAGScoreThreshold
		}
	}

	jsonData, err := json.Marshal(request)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	headers := http.Header{}
	headers.Set("Content-Type", "application/json")
	headers.Set("Accept", "text/event-stream")

	return buildChatCurlCommand(url, jsonData, headers), nil
}

// Private methods

func (m *ChatManager) loadSession() {
	if m.sessionFile == "" {
		return
	}

	if _, err := os.Stat(m.sessionFile); os.IsNotExist(err) {
		return
	}

	data, err := os.ReadFile(m.sessionFile)
	if err != nil {
		utils.LogDebug(fmt.Sprintf("Failed to read session file: %v", err))
		return
	}

	var context SessionContext
	if err := yaml.Unmarshal(data, &context); err != nil {
		utils.LogDebug(fmt.Sprintf("Failed to parse session YAML: %v", err))
		return
	}

	if context.SessionID != "" {
		m.sessionMu.Lock()
		m.sessionID = context.SessionID
		m.sessionMu.Unlock()
		utils.LogDebug(fmt.Sprintf("Loaded session ID: %s", context.SessionID))
	}
}

func (m *ChatManager) saveSession() error {
	if m.sessionFile == "" {
		return nil
	}

	m.sessionMu.RLock()
	sessionID := m.sessionID
	m.sessionMu.RUnlock()

	if sessionID == "" {
		return nil
	}

	dir := filepath.Dir(m.sessionFile)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create context directory: %w", err)
	}

	contextData := map[string]interface{}{
		"session_id": sessionID,
		"timestamp":  time.Now().Format(time.RFC3339),
	}

	yamlData, err := yaml.Marshal(contextData)
	if err != nil {
		return fmt.Errorf("failed to marshal context data to YAML: %w", err)
	}

	if err := os.WriteFile(m.sessionFile, yamlData, 0644); err != nil {
		return fmt.Errorf("failed to write context file: %w", err)
	}

	utils.LogDebug(fmt.Sprintf("Saved session ID %s to %s", sessionID, m.sessionFile))
	return nil
}

func (m *ChatManager) computeSessionFilePath() (string, error) {
	cfg := m.GetConfig()
	if cfg.SessionMode == SessionModeStateless {
		return "", nil
	}

	lfDataDir, err := utils.GetLFDataDir()
	if err != nil {
		return "", err
	}

	ns := cfg.SessionNamespace
	if strings.TrimSpace(ns) == "" {
		ns = cfg.Namespace
	}
	proj := cfg.SessionProject
	if strings.TrimSpace(proj) == "" {
		proj = cfg.ProjectID
	}
	if strings.TrimSpace(ns) == "" || strings.TrimSpace(proj) == "" {
		return "", fmt.Errorf("session requires namespace and project")
	}

	var base string
	switch cfg.SessionMode {
	case SessionModeDev:
		base = filepath.Join(lfDataDir, "projects", ns, proj, "cli", "context", "dev")
	default:
		base = filepath.Join(lfDataDir, "projects", ns, proj, "cli", "context")
	}

	return filepath.Join(base, "context.yaml"), nil
}

func (m *ChatManager) buildChatAPIURL() (string, error) {
	cfg := m.GetConfig()
	base := strings.TrimSuffix(cfg.ServerURL, "/")
	if cfg.Namespace == "" || cfg.ProjectID == "" {
		return "", fmt.Errorf("namespace and project id are required to build chat API URL")
	}
	return fmt.Sprintf("%s/v1/projects/%s/%s/chat/completions", base, cfg.Namespace, cfg.ProjectID), nil
}

func (m *ChatManager) buildHTTPRequest(messages []Message) (*http.Request, error) {
	// Capture current config (lock-free read)
	cfg := m.GetConfig()

	url, err := m.buildChatAPIURL()
	if err != nil {
		return nil, err
	}

	streamTrue := true
	// Filter out client messages - they're only for display
	var filteredMessages []Message
	for _, msg := range messages {
		if msg.Role != "client" && msg.Role != "error" {
			filteredMessages = append(filteredMessages, msg)
		}
	}

	request := ChatRequest{Messages: filteredMessages, Stream: &streamTrue}

	// Include model if specified
	if cfg.Model != "" {
		request.Model = &cfg.Model
	}

	// Always include rag_enabled to let the server know the explicit intent
	request.RAGEnabled = &cfg.RAGEnabled
	// Include additional RAG params only when enabled
	if cfg.RAGEnabled {
		if cfg.RAGDatabase != "" {
			request.RAGDatabase = &cfg.RAGDatabase
		}
		if cfg.RAGRetrievalStrategy != "" {
			request.RAGRetrievalStrategy = &cfg.RAGRetrievalStrategy
		}
		if cfg.RAGTopK > 0 {
			request.RAGTopK = &cfg.RAGTopK
		}
		if cfg.RAGScoreThreshold > 0 {
			request.RAGScoreThreshold = &cfg.RAGScoreThreshold
		}
	}

	jsonData, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	utils.LogDebug(fmt.Sprintf("Request JSON: %s", string(jsonData)))

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("Cache-Control", "no-cache")
	req.Header.Set("Connection", "keep-alive")

	sessionID := m.GetSessionID()
	if sessionID != "" {
		req.Header.Set("X-Session-ID", sessionID)
	} else if cfg.SessionMode == SessionModeStateless {
		req.Header.Set("X-No-Session", "true")
	}

	if cfg.SessionMode == SessionModeDev {
		req.Header.Set("X-Active-Project", cfg.SessionNamespace+"/"+cfg.SessionProject)
	}

	return req, nil
}

func (m *ChatManager) executeStreamingRequest(ctx context.Context, req *http.Request, handler func(StreamChunk) error) error {
	req = req.WithContext(ctx)

	utils.LogDebug(fmt.Sprintf("HTTP %s %s", req.Method, req.URL.String()))
	utils.LogHeaders("request", req.Header)

	hc := &http.Client{Timeout: 0, Transport: &http.Transport{DisableCompression: true, IdleConnTimeout: 0}}
	resp, err := hc.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, readErr := io.ReadAll(resp.Body)
		if readErr != nil {
			return fmt.Errorf("server returned error %d and body read failed: %v", resp.StatusCode, readErr)
		}
		return fmt.Errorf("server returned error %d: %s", resp.StatusCode, utils.PrettyServerError(resp, body))
	}

	utils.LogDebug(fmt.Sprintf("Response status: %d %s", resp.StatusCode, http.StatusText(resp.StatusCode)))
	utils.LogHeaders("response", resp.Header)

	// Update session ID from response header
	if sessionIDHeader := resp.Header.Get("X-Session-ID"); sessionIDHeader != "" {
		cfg := m.GetConfig()
		if cfg.SessionMode == SessionModeStateless {
			// Don't persist session in stateless mode
		} else {
			if err := m.SetSessionID(sessionIDHeader); err != nil {
				utils.LogDebug(fmt.Sprintf("Failed to save session: %v", err))
			}
		}
	}

	return m.parseStreamingResponse(resp.Body, handler)
}

func (m *ChatManager) parseStreamingResponse(body io.Reader, handler func(StreamChunk) error) error {
	reader := bufio.NewReader(body)

	// Track accumulated tool calls by index
	toolCallAccumulator := make(map[int]struct {
		ID        string
		Type      string
		Name      string
		Arguments strings.Builder
	})

	for {
		line, err := reader.ReadString('\n')
		utils.LogDebug(fmt.Sprintf("Stream line: %s", strings.TrimSpace(line)))

		if err != nil {
			if err == io.EOF {
				break
			}
			return fmt.Errorf("stream read error: %w", err)
		}

		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			continue
		}
		if !strings.HasPrefix(line, "data:") {
			continue
		}

		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "[DONE]" {
			break
		}

		var chunk struct {
			Choices []struct {
				Delta struct {
					Role      string `json:"role,omitempty"`
					Content   string `json:"content,omitempty"`
					ToolCalls []struct {
						Index    int    `json:"index"`
						ID       string `json:"id,omitempty"`
						Type     string `json:"type,omitempty"`
						Function struct {
							Name      string `json:"name,omitempty"`
							Arguments string `json:"arguments,omitempty"`
						} `json:"function"`
					} `json:"tool_calls,omitempty"`
				} `json:"delta"`
				FinishReason string `json:"finish_reason,omitempty"`
			} `json:"choices"`
		}

		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			continue
		}

		if len(chunk.Choices) == 0 {
			continue
		}

		choice := chunk.Choices[0]
		delta := choice.Delta

		// Handle tool calls
		if len(delta.ToolCalls) > 0 {
			for _, tc := range delta.ToolCalls {
				// Non-CLI tools are emitted immediately for display
				if tc.Function.Name != "" && !strings.HasPrefix(tc.Function.Name, "cli.") {
					toolCall := &ToolCall{
						ID:        tc.ID,
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					}
					if err := handler(StreamChunk{Type: ChunkTypeToolCall, ToolCall: toolCall}); err != nil {
						return err
					}
					continue
				}

				// CLI tools are accumulated
				accumulated := toolCallAccumulator[tc.Index]
				if tc.ID != "" {
					accumulated.ID = tc.ID
				}
				if tc.Type != "" {
					accumulated.Type = tc.Type
				}
				if tc.Function.Name != "" {
					accumulated.Name = tc.Function.Name
					utils.LogDebug(fmt.Sprintf("Tool call started: %s (index: %d)", tc.Function.Name, tc.Index))
				}
				if tc.Function.Arguments != "" {
					accumulated.Arguments.WriteString(tc.Function.Arguments)
				}
				toolCallAccumulator[tc.Index] = accumulated
			}
		}

		// Check if streaming finished with tool_calls
		if choice.FinishReason == "tool_calls" {
			utils.LogDebug(fmt.Sprintf("Tool calls complete, emitting %d tool call(s)", len(toolCallAccumulator)))
			// Emit all accumulated tool calls
			for idx, tc := range toolCallAccumulator {
				utils.LogDebug(fmt.Sprintf("Emitting complete tool call %d: %s", idx, tc.Name))
				toolCall := &ToolCall{
					ID:        tc.ID,
					Name:      tc.Name,
					Arguments: tc.Arguments.String(),
				}
				if err := handler(StreamChunk{Type: ChunkTypeToolCall, ToolCall: toolCall}); err != nil {
					return err
				}
			}
			// Clear accumulator after emitting
			toolCallAccumulator = make(map[int]struct {
				ID        string
				Type      string
				Name      string
				Arguments strings.Builder
			})
		}

		// Send content if present
		if delta.Content != "" {
			utils.LogDebug(fmt.Sprintf("Content chunk: %s", delta.Content))
			if err := handler(StreamChunk{Type: ChunkTypeContent, Content: delta.Content}); err != nil {
				return err
			}
		}
	}

	// Signal completion
	return handler(StreamChunk{Type: ChunkTypeDone})
}
