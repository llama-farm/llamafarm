package cmd

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"llamafarm-cli/cmd/config"

	"github.com/spf13/cobra"
)

var (
	namespace   string
	projectID   string
	sessionID   string
	temperature float64
	maxTokens   int
	streaming   bool
)

// Chat client types and helpers are defined in chat_client.go

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

// projectsListCmd lists projects for a namespace from the server
var projectsListCmd = &cobra.Command{
	Use:   "list",
	Short: "List projects in a namespace",
	Long:  "List projects available in the specified namespace on the LlamaFarm server.",
	Run: func(cmd *cobra.Command, args []string) {
		// Resolve config path from persistent flag
		configPath, _ := cmd.Flags().GetString("config")

		// Resolve server URL and namespace (project is not required for list)
		serverCfg, err := config.GetServerConfigLenient(configPath, serverURL, namespace, "")
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		serverURL = serverCfg.URL
		ns := strings.TrimSpace(serverCfg.Namespace)

		if ns == "" {
			fmt.Fprintln(os.Stderr, "Error: namespace is required. Provide --namespace or set it in llamafarm.yaml")
			os.Exit(1)
		}

		// Ensure server is up (auto-start locally if needed)
		if err := ensureServerAvailable(serverURL); err != nil {
			fmt.Fprintf(os.Stderr, "Error ensuring server availability: %v\n", err)
			os.Exit(1)
		}

		// Build request
		url := buildServerURL(serverURL, fmt.Sprintf("/v1/projects/%s", ns))
		req, err := http.NewRequest(http.MethodGet, url, nil)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error creating request: %v\n", err)
			os.Exit(1)
		}

		// Execute
		resp, err := getHTTPClient().Do(req)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error requesting server: %v\n", err)
			os.Exit(1)
		}
		defer resp.Body.Close()
		body, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != http.StatusOK {
			fmt.Fprintf(os.Stderr, "Server returned error %d: %s\n", resp.StatusCode, string(body))
			os.Exit(1)
		}

		var listResp struct {
			Total    int `json:"total"`
			Projects []struct {
				Namespace string `json:"namespace"`
				Name      string `json:"name"`
			} `json:"projects"`
		}
		if err := json.Unmarshal(body, &listResp); err != nil {
			fmt.Fprintf(os.Stderr, "Failed to parse server response: %v\n", err)
			os.Exit(1)
		}

		if listResp.Total == 0 || len(listResp.Projects) == 0 {
			fmt.Printf("No projects found in namespace %s\n", ns)
			return
		}

		for _, p := range listResp.Projects {
			fmt.Printf("%s/%s\n", p.Namespace, p.Name)
		}
	},
}

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

		// Ensure server is up (auto-start locally if needed)
		if err := ensureServerAvailable(serverURL); err != nil {
			fmt.Fprintf(os.Stderr, "Error ensuring server availability: %v\n", err)
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
	chatCmd.Flags().StringVar(&namespace, "namespace", "", "Project namespace (default: from llamafarm.yaml)")
	chatCmd.Flags().StringVar(&projectID, "project", "", "Project ID (default: from llamafarm.yaml)")
	chatCmd.Flags().StringVar(&sessionID, "session-id", "", "Existing session ID to continue conversation")
	chatCmd.Flags().Float64Var(&temperature, "temperature", 0.7, "Sampling temperature (0.0 to 2.0)")
	chatCmd.Flags().IntVar(&maxTokens, "max-tokens", 1000, "Maximum number of tokens to generate")
	chatCmd.Flags().BoolVar(&streaming, "stream", true, "Stream assistant responses")

	// No flags are required now - they can come from config file

	// Add subcommands to projects
	projectsCmd.AddCommand(chatCmd)

	// Add list subcommand to projects
	projectsCmd.AddCommand(projectsListCmd)

	// Add the projects command to root
	rootCmd.AddCommand(projectsCmd)
}
