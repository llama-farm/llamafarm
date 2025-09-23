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
)

// runCmd represents the `lf run` command
var runCmd = &cobra.Command{
	Use:   "run [namespace/project] [input]",
	Short: "Run a one-off prompt against a project",
	Long: `Run a one-off prompt against a LlamaFarm project.

Examples:
  # Explicit project and inline input
  lf run my-org/my-project "What models are configured?"

  # Explicit project and input file
  lf run my-org/my-project -f ./prompt.txt

  # Project inferred from llamafarm.yaml, inline input
  lf run "What models are configured?"

  # Project inferred from llamafarm.yaml, input file
  lf run -f ./prompt.txt

  # Run with RAG (default behavior)
  lf run "What is transformer architecture?"

  # Run with specific database
  lf run --database main_database "Explain attention mechanism"

  # Run with custom retrieval strategy and top-k
  lf run --retrieval-strategy filtered_search --rag-top-k 10 "How do neural networks work?"

  # Run WITHOUT RAG (LLM only)
  lf run --no-rag "What is machine learning?"`,
	Args: func(cmd *cobra.Command, args []string) error {
		// Valid forms:
		// 1) run <ns>/<proj> <input>
		// 2) run <ns>/<proj> --file <path>
		// 3) run <input>              (ns/proj inferred from config)
		// 4) run --file <path>        (ns/proj inferred from config)

		if len(args) == 0 {
			// Must at least have input via file
			if runInputFile == "" {
				return fmt.Errorf("provide an input string or --file")
			}
			return nil
		}

		if strings.Contains(args[0], "/") {
			// Explicit project provided
			if strings.Count(args[0], "/") != 1 {
				return fmt.Errorf("project must be in format 'namespace/project', got: %s", args[0])
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

		// No explicit project; first arg is the inline input.
		// If a file is also provided, it's ambiguous/invalid.
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
			if strings.Contains(args[0], "/") {
				// Explicit project, inline input follows
				if len(args) >= 2 {
					input = args[1]
				}
			} else {
				// No explicit project, first arg is inline input
				input = args[0]
			}
		}

		// Parse explicit project if provided
		if len(args) >= 1 && strings.Contains(args[0], "/") {
			parts := strings.SplitN(args[0], "/", 2)
			ns = strings.TrimSpace(parts[0])
			proj = strings.TrimSpace(parts[1])
		}

		cwd := getEffectiveCWD()

		if ns == "" || proj == "" {
			cfg, err := config.LoadConfig(cwd)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error loading config: %v\n", err)
				os.Exit(1)
			}

			projectInfo, err := cfg.GetProjectInfo()
			if err != nil {
				fmt.Fprintf(os.Stderr, "Warning: Could not extract project info for watcher: %v\n", err)
			} else {
				// Start the config file watcher in background
				if err := StartConfigWatcher(projectInfo.Namespace, projectInfo.Project); err != nil {
					fmt.Fprintf(os.Stderr, "Warning: Failed to start config watcher: %v\n", err)
				}
			}
		}

		// Resolve server configuration (strict): if ns/proj are absent, require from llamafarm.yaml
		serverCfg, err := config.GetServerConfig(cwd, serverURL, ns, proj)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		serverURL = serverCfg.URL
		ns = serverCfg.Namespace
		proj = serverCfg.Project

		// Ensure server is up (auto-start locally if needed)
		ensureServerAvailable(serverURL, true)

		// Construct context and call the project-scoped chat completions via shared helpers
		ctx := &ChatSessionContext{
			ServerURL:   serverURL,
			Namespace:   ns,
			ProjectID:   proj,
			Temperature: temperature,
			MaxTokens:   maxTokens,
			HTTPClient:  getHTTPClient(),
			// RAG settings - RAG is enabled by default unless --no-rag is used
			RAGEnabled:           !runNoRAG,
			RAGDatabase:          runRAGDatabase,
			RAGRetrievalStrategy: runRetrievalStrategy,
			RAGTopK:              runRAGTopK,
			RAGScoreThreshold:    runRAGScoreThreshold,
		}

		messages := []ChatMessage{{Role: "user", Content: input}}
		resp, err := sendChatRequest(messages, ctx)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		if resp == "" {
			fmt.Printf("No response received\n")
		} else {
			fmt.Printf("%s\n", resp)
		}
	},
}

func init() {
	runCmd.Flags().StringVarP(&runInputFile, "file", "f", "", "path to file containing input text")

	runCmd.Flags().BoolVar(&runNoRAG, "no-rag", false, "Disable RAG (use LLM only without document retrieval)")
	runCmd.Flags().StringVar(&runRAGDatabase, "database", "", "Database to use for RAG (default: from config)")
	runCmd.Flags().StringVar(&runRetrievalStrategy, "retrieval-strategy", "", "Retrieval strategy to use (default: from database config)")
	runCmd.Flags().IntVar(&runRAGTopK, "rag-top-k", 5, "Number of RAG results to retrieve")
	runCmd.Flags().Float64Var(&runRAGScoreThreshold, "rag-score-threshold", 0.0, "Minimum score threshold for RAG results")

	rootCmd.AddCommand(runCmd)
}
