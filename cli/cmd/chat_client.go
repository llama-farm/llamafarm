package cmd

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/llamafarm/cli/cmd/utils"
	"gopkg.in/yaml.v2"
)

// ToolCallFunction represents the function details of a tool call
type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ToolCallItem represents a single tool call in a message
type ToolCallItem struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function ToolCallFunction `json:"function"`
}

// Message represents a single chat message
type Message struct {
	Role       string         `json:"role"`
	Content    string         `json:"content"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
	ToolCalls  []ToolCallItem `json:"tool_calls,omitempty"`
}

// ChatRequest represents the request payload for the chat API
type ChatRequest struct {
	Model            *string            `json:"model,omitempty"`
	Messages         []Message          `json:"messages"`
	Metadata         map[string]string  `json:"metadata,omitempty"`
	Modalities       []string           `json:"modalities,omitempty"`
	ResponseFormat   map[string]string  `json:"response_format,omitempty"`
	Stream           *bool              `json:"stream,omitempty"`
	Temperature      *float64           `json:"temperature,omitempty"`
	TopP             *float64           `json:"top_p,omitempty"`
	TopK             *int               `json:"top_k,omitempty"`
	MaxTokens        *int               `json:"max_tokens,omitempty"`
	Stop             []string           `json:"stop,omitempty"`
	FrequencyPenalty *float64           `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float64           `json:"presence_penalty,omitempty"`
	LogitBias        map[string]float64 `json:"logit_bias,omitempty"`
	// RAG fields
	RAGEnabled           *bool    `json:"rag_enabled,omitempty"`
	RAGDatabase          *string  `json:"database,omitempty"`
	RAGRetrievalStrategy *string  `json:"rag_retrieval_strategy,omitempty"`
	RAGTopK              *int     `json:"rag_top_k,omitempty"`
	RAGScoreThreshold    *float64 `json:"rag_score_threshold,omitempty"`
}

// ChatChoice represents a choice in the chat response
type ChatChoice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

// ChatResponse represents the response from the chat API
type ChatResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []ChatChoice `json:"choices"`
}

// SessionMode controls how the CLI manages chat session state.
type SessionMode int

const (
	SessionModeProject SessionMode = iota
	SessionModeStateless
	SessionModeDev
)

// ChatSessionContext encapsulates CLI session and connection state.
type ChatSessionContext struct {
	ServerURL   string
	Namespace   string
	ProjectID   string
	SessionID   string
	SessionMode SessionMode
	// SessionNamespace and SessionProject determine where client-side session
	// context is persisted; for dev sessions they map to the user's project.
	SessionNamespace string
	SessionProject   string
	Temperature      float64
	MaxTokens        int
	Streaming        bool
	HTTPClient       utils.HTTPClient
	Model            string
	// RAG fields
	RAGEnabled           bool
	RAGDatabase          string
	RAGRetrievalStrategy string
	RAGTopK              int
	RAGScoreThreshold    float64
}

// sessionFilePath returns the path to the session context file
// This is kept for the legacy readSessionContext/writeSessionContext functions
func (ctx *ChatSessionContext) sessionFilePath() (string, error) {
	if ctx == nil {
		return "", fmt.Errorf("session context is nil")
	}

	lfDataDir, err := utils.GetLFDataDir()
	if err != nil {
		return "", err
	}

	ns := ctx.SessionNamespace
	if strings.TrimSpace(ns) == "" {
		ns = ctx.Namespace
	}
	proj := ctx.SessionProject
	if strings.TrimSpace(proj) == "" {
		proj = ctx.ProjectID
	}
	if strings.TrimSpace(ns) == "" || strings.TrimSpace(proj) == "" {
		return "", fmt.Errorf("session requires namespace and project")
	}

	switch ctx.SessionMode {
	case SessionModeStateless:
		return "", nil
	case SessionModeDev:
		base := filepath.Join(lfDataDir, "projects", ns, proj, "cli", "context", "dev")
		return filepath.Join(base, "context.yaml"), nil
	default:
		base := filepath.Join(lfDataDir, "projects", ns, proj, "cli", "context")
		return filepath.Join(base, "context.yaml"), nil
	}
}

// shellEscapeSingleQuotes safely wraps a string for POSIX shells using single
// quotes. It follows the standard pattern of ending the string, escaping the
// single quote, and resuming the quoted string.
func shellEscapeSingleQuotes(s string) string {
	return "'" + strings.ReplaceAll(s, "'", "'\"'\"'") + "'"
}

func buildChatCurlCommand(url string, body []byte, headers http.Header) string {
	var b strings.Builder
	b.WriteString("curl -X POST ")
	for key, values := range headers {
		if strings.EqualFold(key, "authorization") {
			continue
		}
		for _, v := range values {
			header := fmt.Sprintf("%s: %s", key, v)
			b.WriteString("-H ")
			b.WriteString(shellEscapeSingleQuotes(header))
			b.WriteByte(' ')
		}
	}
	if auth := headers.Get("Authorization"); auth != "" {
		b.WriteString("-H ")
		b.WriteString(shellEscapeSingleQuotes("Authorization: Bearer <redacted>"))
		b.WriteByte(' ')
	}
	if len(body) > 0 {
		b.WriteString("-d ")
		b.WriteString(shellEscapeSingleQuotes(string(body)))
		b.WriteByte(' ')
	}
	b.WriteString(shellEscapeSingleQuotes(url))
	return b.String()
}

// SessionContext represents the structure of the session context file
type SessionContext struct {
	SessionID string `yaml:"session_id"`
	Timestamp string `yaml:"timestamp"`
}

// readSessionContext reads the session context from the YAML file if it exists
func readSessionContext(ctx *ChatSessionContext) (*SessionContext, error) {
	if ctx == nil {
		return nil, fmt.Errorf("context is required")
	}

	path, err := ctx.sessionFilePath()
	if err != nil {
		return nil, err
	}
	if path == "" {
		return nil, nil
	}
	contextFile := path

	if _, err := os.Stat(contextFile); os.IsNotExist(err) {
		return nil, nil
	}

	data, err := os.ReadFile(contextFile)
	if err != nil {
		return nil, fmt.Errorf("failed to read context file: %w", err)
	}

	var context SessionContext
	if err := yaml.Unmarshal(data, &context); err != nil {
		return nil, fmt.Errorf("failed to parse context YAML: %w", err)
	}

	if context.SessionID == "" {
		return nil, nil
	}

	utils.LogDebug(fmt.Sprintf("readSessionContext: returning context from path %s: %+v", contextFile, context))

	return &context, nil
}

// writeSessionContext writes the current session ID to a YAML file in the .llamafarm directory
func writeSessionContext(ctx *ChatSessionContext, sessionID string) error {
	if sessionID == "" {
		return nil
	}

	var contextFile string
	if ctx != nil {
		path, err := ctx.sessionFilePath()
		if err != nil {
			return err
		}
		if path == "" {
			return nil
		}
		contextFile = path
	}

	dir := filepath.Dir(contextFile)
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

	if err := os.WriteFile(contextFile, yamlData, 0644); err != nil {
		return fmt.Errorf("failed to write context file: %w", err)
	}

	return nil
}

type SessionHistory struct {
	Messages []SessionHistoryMessage `json:"messages"`
}

type SessionHistoryMessage struct {
	Role       string `json:"role"`
	Content    string `json:"content"`
	ToolCallID string `json:"tool_call_id,omitempty"`
	ToolCalls  []struct {
		ID       string `json:"id"`
		Type     string `json:"type"`
		Function struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"function"`
	} `json:"tool_calls,omitempty"`
}

type SessionHistoryResponse struct {
	Messages []SessionHistoryMessage `json:"messages"`
}

// fetchSessionHistory retrieves the chat history for a session from the server.
// Returns a slice of user messages suitable for CLI history (up/down arrow).
func fetchSessionHistory(serverURL, namespace, projectID, sessionID string) SessionHistory {
	if sessionID == "" {
		return SessionHistory{}
	}
	url := fmt.Sprintf("%s/v1/projects/%s/%s/chat/sessions/%s/history", strings.TrimSuffix(serverURL, "/"), namespace, projectID, sessionID)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		utils.LogDebug(fmt.Sprintf("fetchSessionHistory: failed to create request: %v", err))
		return SessionHistory{}
	}
	resp, err := utils.GetHTTPClient().Do(req)
	if err != nil {
		utils.LogDebug(fmt.Sprintf("fetchSessionHistory: failed to send request: %v", err))
		return SessionHistory{}
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		utils.LogDebug(fmt.Sprintf("fetchSessionHistory: failed to get history: %d", resp.StatusCode))
		return SessionHistory{}
	}

	var result SessionHistoryResponse
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		utils.LogDebug(fmt.Sprintf("fetchSessionHistory: failed to read body: %v", err))
		return SessionHistory{}
	}
	if err := json.NewDecoder(bytes.NewReader(body)).Decode(&result); err != nil {
		utils.LogDebug(fmt.Sprintf("fetchSessionHistory: failed to decode history: %v, %s", err, string(body)))
		return SessionHistory{}
	}

	// Return the messages directly - they already have all fields including tool_calls
	return SessionHistory{
		Messages: result.Messages,
	}
}
