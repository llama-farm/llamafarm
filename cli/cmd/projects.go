package cmd

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"llamafarm-cli/cmd/config"

	"github.com/spf13/cobra"
)

// ANSI color helpers (disabled if NO_COLOR is set)
const (
	ansiReset   = "\x1b[0m"
	ansiBold    = "\x1b[1m"
	ansiDim     = "\x1b[2m"
	ansiGreen   = "\x1b[32m"
	ansiYellow  = "\x1b[33m"
	ansiMagenta = "\x1b[35m"
	ansiCyan    = "\x1b[36m"
)

func colorize(code, s string) string {
	if os.Getenv("NO_COLOR") != "" {
		return s
	}
	return code + s + ansiReset
}

func bold(s string) string { return colorize(ansiBold, s) }
func dim(s string) string  { return colorize(ansiDim, s) }

// UI status line management
var (
	statusMu     sync.Mutex
	statusActive bool
	statusLine   string
)

func renderStatus(prefix, text string) {
	statusMu.Lock()
	statusActive = true
	statusLine = prefix + " " + text
	statusMu.Unlock()
	_, _ = fmt.Fprint(os.Stdout, "\r"+statusLine+"\033[K")
}

func endStatus() {
	statusMu.Lock()
	wasActive := statusActive
	statusActive = false
	statusMu.Unlock()
	if wasActive {
		_, _ = fmt.Fprint(os.Stdout, "\r\033[K")
	}
}

func safePrint(s string) {
	statusMu.Lock()
	active := statusActive
	line := statusLine
	statusMu.Unlock()
	if active {
		_, _ = fmt.Fprint(os.Stdout, "\r\033[K")
	}
	fmt.Print(s)
	if active {
		_, _ = fmt.Fprint(os.Stdout, "\r"+line+"\033[K")
	}
}

func safePrintf(format string, a ...any) {
	statusMu.Lock()
	active := statusActive
	line := statusLine
	statusMu.Unlock()
	if active {
		_, _ = fmt.Fprint(os.Stdout, "\r\033[K")
	}
	fmt.Printf(format, a...)
	if active {
		_, _ = fmt.Fprint(os.Stdout, "\r"+line+"\033[K")
	}
}

func safeErrorf(format string, a ...any) {
	statusMu.Lock()
	active := statusActive
	line := statusLine
	statusMu.Unlock()
	if active {
		_, _ = fmt.Fprint(os.Stdout, "\r\033[K")
	}
	fmt.Fprintf(os.Stderr, format, a...)
	if active {
		_, _ = fmt.Fprint(os.Stdout, "\r"+line+"\033[K")
	}
}

// startThinkingAnimation prints a quick type-out of "Thinking..." and then
// pulses the dots until stop() is called.
func startThinkingAnimation(prefix string, w io.Writer) (stop func()) {
	stopCh := make(chan struct{}, 1)
	go func() {
		// Quick type effect using carriage-return redraws
		thinking := "Thinking..."
		for i := 1; i <= len(thinking); i++ {
			select {
			case <-stopCh:
				return
			default:
				renderStatus(prefix, thinking[:i])
				time.Sleep(35 * time.Millisecond)
			}
		}
		// Pulse dots with brightness transitions using CR redraw
		styles := []string{"", ansiDim, "", ansiBold}
		idx := 0
		for {
			select {
			case <-stopCh:
				return
			default:
				code := styles[idx%len(styles)]
				dots := "..."
				if code != "" && os.Getenv("NO_COLOR") == "" {
					dots = code + dots + ansiReset
				}
				renderStatus(prefix, "Thinking"+dots)
				idx++
				time.Sleep(260 * time.Millisecond)
			}
		}
	}()
	return func() {
		select {
		case stopCh <- struct{}{}:
		default:
		}
		endStatus()
	}
}

// Chat client types and helpers are defined in chat_client.go

// quitChat attempts to close the current session and exits the program.
func quitChat() {
	if sessionID != "" {
		_ = deleteChatSession()
	}
	safePrint("👋 You have left the pasture. Safe travels, little llama!\n")
	os.Exit(0)
}

// Chat CLI state variables
var (
	namespace   string
	projectID   string
	sessionID   string
	temperature float64
	maxTokens   int
	streaming   bool
)

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
	// Resolve server URL and ensure it's reachable (auto-start if localhost)
	if strings.TrimSpace(serverURL) == "" {
		serverURL = "http://localhost:8000"
	}
	if err := ensureServerAvailable(serverURL); err != nil {
		fmt.Fprintf(os.Stderr, "Error ensuring server availability: %v\n", err)
	}
	// safePrintf("🌾 Starting LlamaFarm chat session...\n")
	runChatSessionTUI()
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
