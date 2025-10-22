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
	"time"

	"gopkg.in/yaml.v2"
)

// Message represents a single chat message
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
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
	HTTPClient       HTTPClient
	Model            string
	// RAG fields
	RAGEnabled           bool
	RAGDatabase          string
	RAGRetrievalStrategy string
	RAGTopK              int
	RAGScoreThreshold    float64
}

func newDefaultContextFromGlobals() *ChatSessionContext {
	return &ChatSessionContext{
		ServerURL:        serverURL,
		Namespace:        namespace,
		ProjectID:        projectID,
		SessionID:        sessionID,
		SessionMode:      SessionModeProject,
		SessionNamespace: namespace,
		SessionProject:   projectID,
		Temperature:      temperature,
		MaxTokens:        maxTokens,
		Streaming:        streaming,
		HTTPClient:       getHTTPClient(),
		RAGEnabled:       true, // RAG is enabled by default
	}
}

func (ctx *ChatSessionContext) sessionFilePath() (string, error) {
	if ctx == nil {
		return "", fmt.Errorf("session context is nil")
	}

	lfDataDir, err := getLFDataDir()
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
		return "", fmt.Errorf("dev session requires namespace and project")
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

func buildChatCurl(messages []Message, ctx *ChatSessionContext) (string, error) {
	if ctx == nil {
		ctx = newDefaultContextFromGlobals()
	}

	url, err := buildChatAPIURL(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to build chat API URL: %w", err)
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
	if ctx.Model != "" {
		request.Model = &ctx.Model
	}

	if ctx.RAGEnabled {
		request.RAGEnabled = &ctx.RAGEnabled
		if ctx.RAGDatabase != "" {
			request.RAGDatabase = &ctx.RAGDatabase
		}
		if ctx.RAGRetrievalStrategy != "" {
			request.RAGRetrievalStrategy = &ctx.RAGRetrievalStrategy
		}
		if ctx.RAGTopK > 0 {
			request.RAGTopK = &ctx.RAGTopK
		}
		if ctx.RAGScoreThreshold > 0 {
			request.RAGScoreThreshold = &ctx.RAGScoreThreshold
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

// buildChatAPIURL chooses the appropriate endpoint based on whether
// namespace and project are set. If both are provided, it uses the
// project-scoped chat completions endpoint; otherwise it falls back
// to the inference chat endpoint.
func buildChatAPIURL(ctx *ChatSessionContext) (string, error) {
	base := strings.TrimSuffix(ctx.ServerURL, "/")
	if ctx.Namespace == "" || ctx.ProjectID == "" {
		return "", fmt.Errorf("namespace and project id are required to build chat API URL")
	}
	return fmt.Sprintf("%s/v1/projects/%s/%s/chat/completions", base, ctx.Namespace, ctx.ProjectID), nil
}

// sendChatRequest connects to the server with stream=true and returns the full assistant message.
func sendChatRequest(messages []Message, ctx *ChatSessionContext) (string, error) {
	chunks, errs, cancel := startChatStream(messages, ctx)
	defer cancel()
	var builder strings.Builder
	for {
		select {
		case s, ok := <-chunks:
			if !ok {
				return builder.String(), nil
			}
			builder.WriteString(s)
		case err := <-errs:
			if err != nil {
				return "", err
			}
		}
	}
}

// startChatStream opens a streaming chat request and returns a channel of
// content chunks and an error channel. The caller should read until the
// chunks channel is closed. The returned cancel function aborts the stream.
func startChatStream(messages []Message, ctx *ChatSessionContext) (<-chan string, <-chan error, func()) {
	outCh := make(chan string, 16)
	errCh := make(chan error, 1)
	var cancelFn context.CancelFunc = func() {}

	go func() {
		defer close(outCh)
		if ctx == nil {
			ctx = newDefaultContextFromGlobals()
		}

		if ctx.SessionMode == SessionModeStateless {
			ctx.SessionID = ""
			sessionID = ""
		} else {
			if existingContext, err := readSessionContext(ctx); err != nil {
				logDebug(fmt.Sprintf("Failed to read session context: %v", err))
			} else if existingContext != nil && existingContext.SessionID != "" {
				ctx.SessionID = existingContext.SessionID
				sessionID = existingContext.SessionID
				logDebug(fmt.Sprintf("Using existing session ID: %s", existingContext.SessionID))
			}
		}

		url, err := buildChatAPIURL(ctx)
		if err != nil {
			errCh <- fmt.Errorf("failed to build chat API URL: %w", err)
			return
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
		if ctx.Model != "" {
			request.Model = &ctx.Model
		}

		// Always include rag_enabled to let the server know the explicit intent
		request.RAGEnabled = &ctx.RAGEnabled
		// Include additional RAG params only when enabled
		if ctx.RAGEnabled {
			if ctx.RAGDatabase != "" {
				request.RAGDatabase = &ctx.RAGDatabase
			}
			if ctx.RAGRetrievalStrategy != "" {
				request.RAGRetrievalStrategy = &ctx.RAGRetrievalStrategy
			}
			if ctx.RAGTopK > 0 {
				request.RAGTopK = &ctx.RAGTopK
			}
			if ctx.RAGScoreThreshold > 0 {
				request.RAGScoreThreshold = &ctx.RAGScoreThreshold
			}
		}

		jsonData, err := json.Marshal(request)
		logDebug(fmt.Sprintf("JSON DATA: %s", string(jsonData)))
		if err != nil {
			errCh <- fmt.Errorf("failed to marshal request: %w", err)
			return
		}

		reqCtx, cancel := context.WithCancel(context.Background())
		cancelFn = cancel

		req, err := http.NewRequestWithContext(reqCtx, "POST", url, bytes.NewBuffer(jsonData))
		if err != nil {
			errCh <- fmt.Errorf("failed to create request: %w", err)
			return
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "text/event-stream")
		req.Header.Set("Cache-Control", "no-cache")
		req.Header.Set("Connection", "keep-alive")
		if ctx.SessionID != "" {
			req.Header.Set("X-Session-ID", ctx.SessionID)
		} else if ctx.SessionMode == SessionModeStateless {
			req.Header.Set("X-No-Session", "true")
		}
		logDebug(fmt.Sprintf("HTTP %s %s", req.Method, req.URL.String()))
		logHeaders("request", req.Header)
		logDebug(fmt.Sprintf("  -> body: %s", req.Body))

		hc := &http.Client{Timeout: 0, Transport: &http.Transport{DisableCompression: true, IdleConnTimeout: 0}}
		resp, err := hc.Do(req)
		if err != nil {
			errCh <- fmt.Errorf("failed to send request: %w", err)
			return
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			body, readErr := io.ReadAll(resp.Body)
			if readErr != nil {
				errCh <- fmt.Errorf("server returned error %d and body read failed: %v", resp.StatusCode, readErr)
				return
			}
			errCh <- fmt.Errorf("server returned error %d: %s", resp.StatusCode, prettyServerError(resp, body))
			return
		}

		logDebug(fmt.Sprintf("  -> %d %s", resp.StatusCode, http.StatusText(resp.StatusCode)))
		logHeaders("response", resp.Header)
		if sessionIDHeader := resp.Header.Get("X-Session-ID"); sessionIDHeader != "" {
			if ctx.SessionMode == SessionModeStateless {
				ctx.SessionID = ""
				sessionID = ""
			} else {
				ctx.SessionID = sessionIDHeader
				sessionID = sessionIDHeader
				if err := writeSessionContext(ctx, sessionIDHeader); err != nil {
					logDebug(fmt.Sprintf("Failed to write session context: %v", err))
				}
			}
		}

		reader := bufio.NewReader(resp.Body)
		for {
			line, err := reader.ReadString('\n')
			logDebug(fmt.Sprintf("STREAM LINE: %v", line))
			if err != nil {
				if err == io.EOF {
					break
				}
				errCh <- fmt.Errorf("stream read error: %w", err)
				return
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
						Role    string `json:"role,omitempty"`
						Content string `json:"content,omitempty"`
					} `json:"delta"`
				} `json:"choices"`
			}
			if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
				continue
			}
			if len(chunk.Choices) == 0 {
				continue
			}
			delta := chunk.Choices[0].Delta
			if delta.Content != "" {
				logDebug(fmt.Sprintf("Sending chunk: %s", delta.Content))
				outCh <- delta.Content
			}
		}
	}()

	return outCh, errCh, func() { cancelFn() }
}

// SessionContext represents the structure of the session context file
type SessionContext struct {
	SessionID string `yaml:"session_id"`
	Timestamp string `yaml:"timestamp"`
}

// readSessionContext reads the session context from the YAML file if it exists
func readSessionContext(ctx *ChatSessionContext) (*SessionContext, error) {
	var contextFile string
	if ctx == nil {
		// Fallback inference when context not provided: try project-scoped location first
		if inferred := newDefaultContextFromGlobals(); inferred != nil {
			if path, err := inferred.sessionFilePath(); err == nil && strings.TrimSpace(path) != "" {
				contextFile = path
			}
		}
		// Legacy fallback: CWD/.llamafarm/context.yaml
		if strings.TrimSpace(contextFile) == "" {
			cwd := getEffectiveCWD()
			contextFile = filepath.Join(cwd, ".llamafarm", "context.yaml")
		}
	} else {
		path, err := ctx.sessionFilePath()
		if err != nil {
			return nil, err
		}
		if path == "" {
			return nil, nil
		}
		contextFile = path
	}

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

	logDebug(fmt.Sprintf("readSessionContext: returning context from path %s: %+v", contextFile, context))

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
	Role    string `json:"role"`
	Content string `json:"content"`
}

type SessionHistoryResponse struct {
	Messages []struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"messages"`
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
		logDebug(fmt.Sprintf("fetchSessionHistory: failed to create request: %v", err))
		return SessionHistory{}
	}
	resp, err := getHTTPClient().Do(req)
	if err != nil {
		logDebug(fmt.Sprintf("fetchSessionHistory: failed to send request: %v", err))
		return SessionHistory{}
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		logDebug(fmt.Sprintf("fetchSessionHistory: failed to get history: %d", resp.StatusCode))
		return SessionHistory{}
	}

	var result SessionHistoryResponse
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		logDebug(fmt.Sprintf("fetchSessionHistory: failed to read body: %v", err))
		return SessionHistory{}
	}
	if err := json.NewDecoder(bytes.NewReader(body)).Decode(&result); err != nil {
		logDebug(fmt.Sprintf("fetchSessionHistory: failed to decode history: %v, %s", err, string(body)))
		return SessionHistory{}
	}
	var messages []SessionHistoryMessage
	for _, msg := range result.Messages {
		messages = append(messages, SessionHistoryMessage{Role: msg.Role, Content: msg.Content})
	}

	return SessionHistory{Messages: messages}
}
