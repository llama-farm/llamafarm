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

// ChatMessage represents a single chat message
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatRequest represents the request payload for the chat API
type ChatRequest struct {
	Model            *string            `json:"model,omitempty"`
	Messages         []ChatMessage      `json:"messages"`
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
}

// ChatChoice represents a choice in the chat response
type ChatChoice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

// ChatResponse represents the response from the chat API
type ChatResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []ChatChoice `json:"choices"`
}

// ChatSessionContext encapsulates CLI session and connection state.
type ChatSessionContext struct {
	ServerURL   string
	Namespace   string
	ProjectID   string
	SessionID   string
	Temperature float64
	MaxTokens   int
	Streaming   bool
	HTTPClient  HTTPClient
}

func newDefaultContextFromGlobals() *ChatSessionContext {
	return &ChatSessionContext{
		ServerURL:   serverURL,
		Namespace:   namespace,
		ProjectID:   projectID,
		SessionID:   sessionID,
		Temperature: temperature,
		MaxTokens:   maxTokens,
		Streaming:   streaming,
		HTTPClient:  getHTTPClient(),
	}
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
func sendChatRequest(messages []ChatMessage, ctx *ChatSessionContext) (string, error) {
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
func startChatStream(messages []ChatMessage, ctx *ChatSessionContext) (<-chan string, <-chan error, func()) {
	outCh := make(chan string, 16)
	errCh := make(chan error, 1)
	var cancelFn context.CancelFunc = func() {}

	go func() {
		defer close(outCh)
		if ctx == nil {
			ctx = newDefaultContextFromGlobals()
		}

		// Read existing session context if available
		if existingContext, err := readSessionContext(); err != nil {
			logDebug(fmt.Sprintf("Failed to read session context: %v", err))
		} else if existingContext != nil && existingContext.SessionID != "" {
			// Use existing session ID if found
			ctx.SessionID = existingContext.SessionID
			sessionID = existingContext.SessionID
			logDebug(fmt.Sprintf("Using existing session ID: %s", existingContext.SessionID))
		}

		url, err := buildChatAPIURL(ctx)
		if err != nil {
			errCh <- fmt.Errorf("failed to build chat API URL: %w", err)
			return
		}
		streamTrue := true
		request := ChatRequest{Messages: messages, Stream: &streamTrue}

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
			ctx.SessionID = sessionIDHeader
			sessionID = sessionIDHeader
			// Write session context to YAML file
			if err := writeSessionContext(sessionIDHeader); err != nil {
				logDebug(fmt.Sprintf("Failed to write session context: %v", err))
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
func readSessionContext() (*SessionContext, error) {
	// Get effective working directory from config
	cwd := getEffectiveCWD()
	if cwd == "" {
		return nil, fmt.Errorf("failed to determine effective working directory")
	}

	// Check if context file exists
	contextFile := filepath.Join(cwd, ".llamafarm", "context.yaml")
	if _, err := os.Stat(contextFile); os.IsNotExist(err) {
		// File doesn't exist, return nil (no existing session)
		return nil, nil
	}

	// Read the context file
	data, err := os.ReadFile(contextFile)
	if err != nil {
		return nil, fmt.Errorf("failed to read context file: %w", err)
	}

	// Parse the YAML
	var context SessionContext
	if err := yaml.Unmarshal(data, &context); err != nil {
		return nil, fmt.Errorf("failed to parse context YAML: %w", err)
	}

	// Validate the session ID
	if context.SessionID == "" {
		return nil, nil // Invalid context, treat as no session
	}

	return &context, nil
}

// writeSessionContext writes the current session ID to a YAML file in the .llamafarm directory
func writeSessionContext(sessionID string) error {
	if sessionID == "" {
		return nil
	}

	// Get effective working directory from config
	cwd := getEffectiveCWD()
	if cwd == "" {
		return fmt.Errorf("failed to determine effective working directory")
	}

	// Create .llamafarm directory if it doesn't exist
	llamafarmDir := filepath.Join(cwd, ".llamafarm")
	if err := os.MkdirAll(llamafarmDir, 0755); err != nil {
		return fmt.Errorf("failed to create .llamafarm directory: %w", err)
	}

	// Create context data structure
	contextData := map[string]interface{}{
		"session_id": sessionID,
		"timestamp":  time.Now().Format(time.RFC3339),
	}

	// Convert to YAML
	yamlData, err := yaml.Marshal(contextData)
	if err != nil {
		return fmt.Errorf("failed to marshal context data to YAML: %w", err)
	}

	// Write to context.yaml file
	contextFile := filepath.Join(llamafarmDir, "context.yaml")
	if err := os.WriteFile(contextFile, yamlData, 0644); err != nil {
		return fmt.Errorf("failed to write context file: %w", err)
	}

	return nil
}

// deleteChatSession attempts to close the current server-side session.
func deleteChatSession() error {
	// If we don't have a session ID, try to read it from context file
	if sessionID == "" {
		if existingContext, err := readSessionContext(); err != nil {
			logDebug(fmt.Sprintf("Failed to read session context for deletion: %v", err))
		} else if existingContext != nil && existingContext.SessionID != "" {
			sessionID = existingContext.SessionID
		} else {
			// No session to delete
			return nil
		}
	}

	if sessionID == "" {
		return nil
	}
	url := fmt.Sprintf("%s/v1/projects/%s/%s/chat/session/%s", strings.TrimSuffix(serverURL, "/"), namespace, projectID, sessionID)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "DELETE", url, nil)
	if err != nil {
		return nil
	}
	resp, err := getHTTPClient().Do(req)
	if err != nil {
		return nil
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	return nil
}
