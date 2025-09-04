package cmd

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
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
func buildChatAPIURL(ctx *ChatSessionContext) string {
	base := strings.TrimSuffix(ctx.ServerURL, "/")
	if ctx.Namespace != "" && ctx.ProjectID != "" {
		return fmt.Sprintf("%s/v1/projects/%s/%s/chat/completions", base, ctx.Namespace, ctx.ProjectID)
	}
	return fmt.Sprintf("%s/v1/inference/chat", base)
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

		url := buildChatAPIURL(ctx)
		streamTrue := true
		request := ChatRequest{Messages: messages, Stream: &streamTrue}
		if !strings.Contains(url, "/v1/projects/") {
			meta := map[string]string{}
			if ctx.Namespace != "" {
				meta["namespace"] = ctx.Namespace
			}
			if ctx.ProjectID != "" {
				meta["project_id"] = ctx.ProjectID
			}
			request.Metadata = meta
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

// deleteChatSession attempts to close the current server-side session.
func deleteChatSession() error {
	if sessionID == "" {
		return nil
	}
	url := fmt.Sprintf("%s/v1/inference/chat/session/%s", strings.TrimSuffix(serverURL, "/"), sessionID)
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
