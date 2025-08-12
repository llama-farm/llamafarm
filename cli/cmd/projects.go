package cmd

import (
    "context"
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
    "os/signal"
	"strings"
    "syscall"
	"time"

	"github.com/spf13/cobra"
	"llamafarm-cli/cmd/config"
)

var (
	serverURL   string
	namespace   string
	projectID   string
	sessionID   string
	temperature float64
	maxTokens   int
    streaming   bool
    verbose     bool
)

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

// projectsCmd represents the projects command
var projectsCmd = &cobra.Command{
	Use:   "projects",
	Short: "Manage LlamaFarm projects and interact with them",
	Long: `Manage LlamaFarm projects and interact with them through various interfaces.

Available commands:
  chat - Start an interactive chat session with a project`,
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Println("LlamaFarm Projects Management")
		cmd.Help()
	},
}

// ChatMessage represents a single chat message
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatRequest represents the request payload for the chat API
type ChatRequest struct {
	Model             *string                `json:"model,omitempty"`
	Messages          []ChatMessage          `json:"messages"`
	Metadata          map[string]string      `json:"metadata,omitempty"`
	Modalities        []string               `json:"modalities,omitempty"`
	ResponseFormat    map[string]string      `json:"response_format,omitempty"`
	Stream            *bool                  `json:"stream,omitempty"`
	Temperature       *float64               `json:"temperature,omitempty"`
	TopP              *float64               `json:"top_p,omitempty"`
	TopK              *int                   `json:"top_k,omitempty"`
	MaxTokens         *int                   `json:"max_tokens,omitempty"`
	Stop              []string               `json:"stop,omitempty"`
	FrequencyPenalty  *float64               `json:"frequency_penalty,omitempty"`
	PresencePenalty   *float64               `json:"presence_penalty,omitempty"`
	LogitBias         map[string]float64     `json:"logit_bias,omitempty"`
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

// HTTP client types and helpers are centralized in httpclient.go

// chatCmd represents the chat command
var chatCmd = &cobra.Command{
	Use:   "chat",
	Short: "Start an interactive chat session with a project",
	Long: `Start an interactive chat session with a LlamaFarm project.
This will connect to the server and allow you to have a conversation
using the project's configured models and RAG data.

Examples:
  lf projects chat                                           # Use defaults from llamafarm.yaml
  lf projects chat --server-url http://localhost:8000       # Override server URL
  lf projects chat --namespace my-org --project my-project  # Override project settings
  lf projects chat --config /path/to/config.yaml            # Use specific config file`,
	Run: func(cmd *cobra.Command, args []string) {
		// Get the config file path from the flag
		configPath, _ := cmd.Flags().GetString("config")

        // Get server configuration (lenient: namespace/project optional)
        serverConfig, err := config.GetServerConfigLenient(configPath, serverURL, namespace, projectID)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

		// Update global variables with resolved values
		serverURL = serverConfig.URL
		namespace = serverConfig.Namespace
		projectID = serverConfig.Project

		startChatSession()
	},
}

func startChatSession() {
    // Handle Ctrl+C gracefully
    sigCh := make(chan os.Signal, 1)
    signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
    go func() {
        <-sigCh
        fmt.Print("\n^C\n")
        // Try to end the server session gracefully (best-effort)
        if sessionID != "" {
            _ = deleteChatSession()
        }
        fmt.Println("👋 You have left the pasture. Safe travels, little llama!")
        os.Exit(0)
    }()
	fmt.Printf("🌾 Starting LlamaFarm chat session...\n")
	fmt.Printf("📡 Server: %s\n", serverURL)
        if namespace != "" || projectID != "" {
            fmt.Printf("📁 Project: %s/%s\n", namespace, projectID)
        } else {
            fmt.Printf("📁 Project: (not specified)\n")
        }
	if sessionID != "" {
		fmt.Printf("🆔 Session: %s\n", sessionID)
	}
	fmt.Printf("\nType 'exit' or 'quit' to end the session, 'clear' to start a new session.\n")
	fmt.Printf("Type your message and press Enter to send.\n\n")

	var conversationHistory []ChatMessage
	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Print("You: ")
		if !scanner.Scan() {
			break
		}

		userInput := strings.TrimSpace(scanner.Text())
		if userInput == "" {
			continue
		}

		// Handle special commands
		switch strings.ToLower(userInput) {
		case "exit", "quit":
			fmt.Println("👋 Goodbye!")
			return
		case "clear":
			conversationHistory = []ChatMessage{}
			sessionID = ""
			fmt.Println("🧹 Session cleared. Starting fresh conversation.")
			continue
		case "help":
			printChatHelp()
			continue
		}

		// Add user message to conversation history
		userMessage := ChatMessage{
			Role:    "user",
			Content: userInput,
		}
		conversationHistory = append(conversationHistory, userMessage)

        // Send request to server
        fmt.Print("Assistant: ")
        if streaming {
            assistantMessage, err := sendChatRequestStream(conversationHistory)
            if err != nil {
                fmt.Fprintf(os.Stderr, "Error: %v\n", err)
                continue
            }
            fmt.Printf("\n\n")
            conversationHistory = append(conversationHistory, ChatMessage{Role: "assistant", Content: assistantMessage})
        } else {
            response, err := sendChatRequest(conversationHistory)
            if err != nil {
                fmt.Fprintf(os.Stderr, "Error: %v\n", err)
                continue
            }
            if len(response.Choices) > 0 {
                assistantMessage := response.Choices[0].Message.Content
                fmt.Printf("%s\n\n", assistantMessage)
                conversationHistory = append(conversationHistory, ChatMessage{Role: "assistant", Content: assistantMessage})
            } else {
                fmt.Println("No response received.")
            }
        }
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "Error reading input: %v\n", err)
	}
}

func sendChatRequest(messages []ChatMessage) (*ChatResponse, error) {
    // Use a context wrapper to avoid hidden global state and aid testability
    ctx := newDefaultContextFromGlobals()
    resp, err := sendChatRequestWithContext(messages, ctx)
    // Sync session back to global for backward compatibility
    sessionID = ctx.SessionID
    return resp, err
}

func sendChatRequestWithContext(messages []ChatMessage, ctx *ChatSessionContext) (*ChatResponse, error) {
    // Construct the API URL
    url := fmt.Sprintf("%s/v1/inference/chat",
        strings.TrimSuffix(ctx.ServerURL, "/"))

    // Create request payload (OpenAI-compatible) and pass routing via metadata
    meta := map[string]string{}
    if ctx.Namespace != "" {
        meta["namespace"] = ctx.Namespace
    }
    if ctx.ProjectID != "" {
        meta["project_id"] = ctx.ProjectID
    }
    request := ChatRequest{Messages: messages, Metadata: meta}

    // Add optional parameters if provided
    if ctx.Temperature >= 0 {
        request.Temperature = &ctx.Temperature
    }
    if ctx.MaxTokens > 0 {
        request.MaxTokens = &ctx.MaxTokens
    }

    // Marshal request to JSON
    jsonData, err := json.Marshal(request)
    if err != nil {
        return nil, fmt.Errorf("failed to marshal request: %w", err)
    }

    // Create HTTP request
    req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
    if err != nil {
        return nil, fmt.Errorf("failed to create request: %w", err)
    }

    // Set headers
    req.Header.Set("Content-Type", "application/json")
    if ctx.SessionID != "" {
        req.Header.Set("X-Session-ID", ctx.SessionID)
    }

    // Send request
    client := ctx.HTTPClient
    if client == nil {
        client = getHTTPClient()
    }
    resp, err := client.Do(req)
    if err != nil {
        return nil, fmt.Errorf("failed to send request: %w", err)
    }
    defer resp.Body.Close()

    // Read response body
    body, err := io.ReadAll(resp.Body)
    if err != nil {
        return nil, fmt.Errorf("failed to read response: %w", err)
    }

    // Check for error status codes
    if resp.StatusCode != http.StatusOK {
        return nil, fmt.Errorf("server returned error %d: %s", resp.StatusCode, string(body))
    }

    // Parse response
    var chatResponse ChatResponse
    if err := json.Unmarshal(body, &chatResponse); err != nil {
        return nil, fmt.Errorf("failed to parse response: %w", err)
    }

    // Extract session ID from response if available
    if sessionIDHeader := resp.Header.Get("X-Session-ID"); sessionIDHeader != "" {
        ctx.SessionID = sessionIDHeader
    }

    return &chatResponse, nil
}

// sendChatRequestStream connects to the server with stream=true and
// consumes Server-Sent Events, printing deltas as they arrive.
// It returns the full assistant message that was streamed.
func sendChatRequestStream(messages []ChatMessage) (string, error) {
    // Use a context wrapper to avoid hidden global state and aid testability
    ctx := newDefaultContextFromGlobals()
    out, err := sendChatRequestStreamWithContext(messages, ctx)
    // Sync session back to global for backward compatibility
    sessionID = ctx.SessionID
    return out, err
}

func sendChatRequestStreamWithContext(messages []ChatMessage, ctx *ChatSessionContext) (string, error) {
    url := fmt.Sprintf("%s/v1/inference/chat", strings.TrimSuffix(ctx.ServerURL, "/"))

    streamTrue := true
    meta := map[string]string{}
    if ctx.Namespace != "" {
        meta["namespace"] = ctx.Namespace
    }
    if ctx.ProjectID != "" {
        meta["project_id"] = ctx.ProjectID
    }
    request := ChatRequest{Messages: messages, Metadata: meta, Stream: &streamTrue}

    jsonData, err := json.Marshal(request)
    if err != nil {
        return "", fmt.Errorf("failed to marshal request: %w", err)
    }

    // Allow cancellation if the process is interrupted
    reqCtx, cancel := context.WithCancel(context.Background())
    defer cancel()
    req, err := http.NewRequestWithContext(reqCtx, "POST", url, bytes.NewBuffer(jsonData))
    if err != nil {
        return "", fmt.Errorf("failed to create request: %w", err)
    }
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("Accept", "text/event-stream")
    req.Header.Set("Cache-Control", "no-cache")
    req.Header.Set("Connection", "keep-alive")
    if ctx.SessionID != "" {
        req.Header.Set("X-Session-ID", ctx.SessionID)
    }
    _ = addLocalhostCWDHeader(req)
    if verbose {
        fmt.Fprintf(os.Stderr, "HTTP %s %s\n", req.Method, req.URL.String())
        logHeaders("request", req.Header)
    }

    // We want to stream the response; do not set a short timeout.
    hc := &http.Client{Timeout: 0, Transport: &http.Transport{DisableCompression: true}}
    resp, err := hc.Do(req)
    if err != nil {
        return "", fmt.Errorf("failed to send request: %w", err)
    }
    defer resp.Body.Close()

    if resp.StatusCode != http.StatusOK {
        body, _ := io.ReadAll(resp.Body)
        return "", fmt.Errorf("server returned error %d: %s", resp.StatusCode, string(body))
    }
    if verbose {
        fmt.Fprintf(os.Stderr, "  -> %d %s\n", resp.StatusCode, http.StatusText(resp.StatusCode))
        logHeaders("response", resp.Header)
    }

    if sessionIDHeader := resp.Header.Get("X-Session-ID"); sessionIDHeader != "" {
        ctx.SessionID = sessionIDHeader
    }

    // SSE parsing: lines in form "data: {json}\n" and a blank line between events.
    reader := bufio.NewReader(resp.Body)
    writer := bufio.NewWriter(os.Stdout)
    defer writer.Flush()
    var builder strings.Builder
    for {
        line, err := reader.ReadString('\n')
        if err != nil {
            if err == io.EOF {
                break
            }
            return "", fmt.Errorf("stream read error: %w", err)
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
            // Ignore malformed lines
            continue
        }
        if len(chunk.Choices) == 0 {
            continue
        }
        delta := chunk.Choices[0].Delta
        if delta.Content != "" {
            // Write and flush immediately so the user sees streaming output
            _, _ = writer.WriteString(delta.Content)
            _ = writer.Flush()
            builder.WriteString(delta.Content)
        }
    }

    return builder.String(), nil
}

// deleteChatSession attempts to close the current server-side session.
// It is best-effort and returns nil on failure to avoid blocking shutdown.
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
    _ = addLocalhostCWDHeader(req)
    resp, err := getHTTPClient().Do(req)
    if err != nil {
        return nil
    }
    io.Copy(io.Discard, resp.Body)
    resp.Body.Close()
    return nil
}

func printChatHelp() {
	fmt.Printf(`
Available commands:
  help  - Show this help message
  clear - Clear conversation history and start a new session
  exit  - Exit the chat session
  quit  - Exit the chat session

Chat parameters:
  Temperature: %.1f
  Max tokens: %d
  Server: %s
  Project: %s/%s

`, temperature, maxTokens, serverURL, namespace, projectID)
}

func init() {
	// Add persistent flags to projects command
	projectsCmd.PersistentFlags().StringP("config", "c", "", "config file path (default: llamafarm.yaml in current directory)")

	// Add flags to chat command
	chatCmd.Flags().StringVar(&serverURL, "server-url", "", "LlamaFarm server URL (default: http://localhost:8000)")
	chatCmd.Flags().StringVar(&namespace, "namespace", "", "Project namespace (default: from llamafarm.yaml)")
	chatCmd.Flags().StringVar(&projectID, "project", "", "Project ID (default: from llamafarm.yaml)")
	chatCmd.Flags().StringVar(&sessionID, "session-id", "", "Existing session ID to continue conversation")
	chatCmd.Flags().Float64Var(&temperature, "temperature", 0.7, "Sampling temperature (0.0 to 2.0)")
	chatCmd.Flags().IntVar(&maxTokens, "max-tokens", 1000, "Maximum number of tokens to generate")
    chatCmd.Flags().BoolVar(&streaming, "stream", true, "Stream assistant responses")
    chatCmd.Flags().BoolVarP(&verbose, "verbose", "v", false, "Verbose HTTP logging")

	// No flags are required now - they can come from config file

	// Add subcommands to projects
	projectsCmd.AddCommand(chatCmd)

	// Add the projects command to root
	rootCmd.AddCommand(projectsCmd)
}

//