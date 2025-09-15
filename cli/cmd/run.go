package cmd

import (
	"fmt"
	"os"
	"strings"

	"llamafarm-cli/cmd/config"

	"github.com/spf13/cobra"
)

var (
	runInputFile string
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
  lf run -f ./prompt.txt`,
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

		// Resolve server configuration (strict): if ns/proj are absent, require from llamafarm.yaml
		serverCfg, err := config.GetServerConfig("", serverURL, ns, proj)
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
	rootCmd.AddCommand(runCmd)
}
