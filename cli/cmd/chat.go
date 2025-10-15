package cmd

import (
	"fmt"
	"os"
	"strings"

	"llamafarm-cli/cmd/config"

	"github.com/spf13/cobra"
)

var (
	runInputFile         string
	runRAGDatabase       string
	runRetrievalStrategy string
	runRAGTopK           int
	runNoRAG             bool
	runRAGScoreThreshold float64
	runModel             string
	dryRun               bool

)

// chatCmd represents the `lf chat` command
var chatCmd = &cobra.Command{
	Use:   "chat [namespace/project] \"input\"",
	Short: "Chat with a LlamaFarm project (RAG enabled by default)",
	Long: `Chat with a LlamaFarm project. RAG is enabled by default unless --no-rag is used.

Examples:
  # Explicit project and inline input
  lf chat my-org/my-project "What models are configured?"

  # Explicit project and input file
  lf chat my-org/my-project -f ./prompt.txt

  # Project inferred from llamafarm.yaml, inline input
  lf chat "What models are configured?"

  # Project inferred from llamafarm.yaml, input file
  lf chat -f ./prompt.txt

  # Chat with RAG (default behavior)
  lf chat "What is transformer architecture?"

  # Chat with specific database
  lf chat --database main_database "Explain attention mechanism"

  # Chat with custom retrieval strategy and top-k
  lf chat --retrieval-strategy filtered_search --rag-top-k 10 "How do neural networks work?"

  # Run WITHOUT RAG (LLM only)
  lf chat --no-rag "What is machine learning?"

  # Select specific model
  lf chat --model fast "What is machine learning?"`,

	Args: func(cmd *cobra.Command, args []string) error {
		// Valid forms:
		// 1) chat <ns>/<proj> <input>
		// 2) chat <ns>/<proj> --file <path>
		// 3) chat <input>              (ns/proj inferred from config)
		// 4) chat --file <path>        (ns/proj inferred from config)
		// 5) chat

		if len(args) == 0 {
			return nil
		}

		// Check if first arg looks like namespace/project (not a file path)
		// File paths start with /, ./, ../ or contain file extensions
		firstArg := args[0]
		isFilePath := strings.HasPrefix(firstArg, "/") ||
					  strings.HasPrefix(firstArg, "./") ||
					  strings.HasPrefix(firstArg, "../") ||
					  strings.Contains(firstArg, ".")  // Has file extension

		if strings.Contains(firstArg, "/") && !isFilePath {
			// Looks like explicit project provided (namespace/project format)
			if strings.Count(firstArg, "/") != 1 {
				return fmt.Errorf("project must be in format 'namespace/project', got: %s", firstArg)
			}
			// If no file, require inline input as second arg
			if runInputFile == "" && len(args) < 2 {
				return fmt.Errorf("provide an input string or --file")
			}
			// If file is set, do not allow a third arg
			if runInputFile != "" && len(args) >= 2 {
				return fmt.Errorf("specify either --file or an inline input, not both")
			}
			return nil
		}

		// No explicit project; first arg is the inline input (or file path).
		// If a file is also provided via --file flag, it's ambiguous/invalid.
		if runInputFile != "" {
			return fmt.Errorf("specify either --file or an inline input, not both")
		}
		return nil
	},
	Run: func(cmd *cobra.Command, args []string) {
		// Resolve project and input according to args pattern
		var ns, proj string

		// Resolve input
		var input string
		if runInputFile != "" {
			data, err := os.ReadFile(runInputFile)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error reading file '%s': %v\n", runInputFile, err)
				os.Exit(1)
			}
			input = string(data)
		} else if len(args) >= 1 {
			// Check if first arg is namespace/project (not a file path)
			firstArg := args[0]
			isFilePath := strings.HasPrefix(firstArg, "/") ||
						  strings.HasPrefix(firstArg, "./") ||
						  strings.HasPrefix(firstArg, "../") ||
						  strings.Contains(firstArg, ".")

			if strings.Contains(firstArg, "/") && !isFilePath {
				// Explicit project, inline input follows
				if len(args) >= 2 {
					input = args[1]
				}
			} else {
				// No explicit project, first arg is inline input (or file path)
				input = args[0]
			}
		}

		// Start an interactive chat session if no input is provided
		if input == "" {
			start(SessionModeProject)
			return
		}

		// Parse explicit project if provided
		if len(args) >= 1 {
			firstArg := args[0]
			isFilePath := strings.HasPrefix(firstArg, "/") ||
						  strings.HasPrefix(firstArg, "./") ||
						  strings.HasPrefix(firstArg, "../") ||
						  strings.Contains(firstArg, ".")

			if strings.Contains(firstArg, "/") && !isFilePath {
				parts := strings.SplitN(firstArg, "/", 2)
				ns = strings.TrimSpace(parts[0])
				proj = strings.TrimSpace(parts[1])
			}
		}

		cwd := getEffectiveCWD()

		StartConfigWatcherForCommand()

		// Resolve server configuration (strict): if ns/proj are absent, require from llamafarm.yaml
		serverCfg, err := config.GetServerConfig(cwd, serverURL, ns, proj)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		serverURL = serverCfg.URL
		ns = serverCfg.Namespace
		proj = serverCfg.Project

		// Construct context for request (without contacting server yet)
		ctx := &ChatSessionContext{
			ServerURL:        serverURL,
			Namespace:        ns,
			ProjectID:        proj,
			SessionMode:      SessionModeStateless,
			SessionNamespace: ns,
			SessionProject:   proj,
			Temperature:      temperature,
			MaxTokens:        maxTokens,
			HTTPClient:       getHTTPClient(),
			Model:            runModel,
			// RAG settings - RAG is enabled by default unless --no-rag is used
			RAGEnabled:           !runNoRAG,
			RAGDatabase:          runRAGDatabase,
			RAGRetrievalStrategy: runRetrievalStrategy,
			RAGTopK:              runRAGTopK,
			RAGScoreThreshold:    runRAGScoreThreshold,
		}

		// Check if input contains or is a file
		var messages []Message
		hasFile, filePath, fileType, remainingText := detectFileInInput(input)

		if hasFile {
			// Build message with file
			msg, err := buildMultimodalMessage("user", remainingText, []string{filePath})
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error processing file: %v\n", err)
				os.Exit(1)
			}
			messages = []Message{msg}

			// Show appropriate icon based on file type
			switch fileType {
			case FileTypeImage:
				fmt.Fprintf(os.Stderr, "📸 Sending image: %s\n", filePath)
			case FileTypeVideo:
				fmt.Fprintf(os.Stderr, "🎥 Sending video: %s\n", filePath)
			case FileTypeAudio:
				fmt.Fprintf(os.Stderr, "🎵 Sending audio: %s\n", filePath)
			case FileTypeText:
				fmt.Fprintf(os.Stderr, "📄 Sending text file: %s\n", filePath)
			}
		} else {
			// Simple text message
			messages = []Message{{Role: "user", Content: input}}
		}

		if dryRun {
			if err := printRunCurlCommand(messages, ctx); err != nil {
				fmt.Fprintf(os.Stderr, "Error generating curl command: %v\n", err)
				os.Exit(1)
			}
			return
		}

		// Ensure server is up (auto-start locally if needed)
		var config *ServiceOrchestrationConfig
		if runNoRAG {
			config = ChatNoRAGConfig(serverURL) // Server only, completely ignore RAG
		} else {
			config = RAGCommandConfig(serverURL) // Wait for both server and RAG
		}
		// Ensure health checks reflect project context before contacting server
		EnsureServicesWithConfig(config)
		resp, err := sendChatRequest(messages, ctx)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		if resp == "" {
			fmt.Printf("No response received\n")
		} else {
			// Check if response contains media (image/video/audio)
			hasMedia, mediaData, mimeType, textOnly := extractBase64Media(resp)
			if hasMedia {
				// Save the media file
				filename, err := saveMediaFromBase64(mediaData, mimeType)
				if err != nil {
					fmt.Fprintf(os.Stderr, "Warning: Failed to save media: %v\n", err)
					fmt.Printf("%s\n", resp)
				} else {
					fmt.Printf("%s\n", textOnly)

					// Show appropriate icon based on MIME type
					var icon string
					if strings.HasPrefix(mimeType, "image/") {
						icon = "💾 Image"
					} else if strings.HasPrefix(mimeType, "video/") {
						icon = "💾 Video"
					} else if strings.HasPrefix(mimeType, "audio/") {
						icon = "💾 Audio"
					} else {
						icon = "💾 Media"
					}
					fmt.Printf("\n%s saved to: %s\n", icon, filename)
				}
			} else {
				fmt.Printf("%s\n", resp)
			}
		}
	},
}

func init() {
	chatCmd.Flags().StringVarP(&runInputFile, "file", "f", "", "path to file containing input text")

	chatCmd.Flags().StringVar(&runModel, "model", "", "Model to use for the request (default: from config)")
	chatCmd.Flags().BoolVar(&runNoRAG, "no-rag", false, "Disable RAG (use LLM only without document retrieval)")
	chatCmd.Flags().StringVar(&runRAGDatabase, "database", "", "Database to use for RAG (default: from config)")
	chatCmd.Flags().StringVar(&runRetrievalStrategy, "retrieval-strategy", "", "Retrieval strategy to use (default: from database config)")
	chatCmd.Flags().IntVar(&runRAGTopK, "rag-top-k", 5, "Number of RAG results to retrieve")
	chatCmd.Flags().Float64Var(&runRAGScoreThreshold, "rag-score-threshold", 0.0, "Minimum score threshold for RAG results")
	chatCmd.Flags().BoolVar(&dryRun, "dry-run", false, "Print the equivalent curl command instead of executing the request")

	rootCmd.AddCommand(chatCmd)
}

func printRunCurlCommand(messages []Message, ctx *ChatSessionContext) error {
	curlCmd, err := buildChatCurl(messages, ctx)
	if err != nil {
		return err
	}
	fmt.Println(curlCmd)
	return nil
}
