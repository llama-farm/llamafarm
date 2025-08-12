package cmd

import (
    "fmt"
    "os"
    "strings"

    "github.com/spf13/cobra"
)

var (
    runInputFile string
)

// runCmd represents the `lf run` command
var runCmd = &cobra.Command{
    Use:   "run <namespace>/<project> [input]",
    Short: "Run a one-off prompt against a project",
    Long: `Run a one-off prompt against a LlamaFarm project.

Examples:
  lf run my-org/my-project "What models are configured?"
  lf run my-org/my-project -f ./prompt.txt`,
    Args: func(cmd *cobra.Command, args []string) error {
        if len(args) < 1 {
            return fmt.Errorf("requires project in format 'namespace/project'")
        }
        if strings.Count(args[0], "/") != 1 {
            return fmt.Errorf("project must be in format 'namespace/project', got: %s", args[0])
        }
        // Either a file or an inline input must be provided
        if runInputFile == "" && len(args) < 2 {
            return fmt.Errorf("provide an input string or --file")
        }
        // Do not allow both file and inline input
        if runInputFile != "" && len(args) >= 2 {
            return fmt.Errorf("specify either --file or an inline input, not both")
        }
        return nil
    },
    Run: func(cmd *cobra.Command, args []string) {
        // Parse namespace/project
        parts := strings.SplitN(args[0], "/", 2)
        ns := strings.TrimSpace(parts[0])
        proj := strings.TrimSpace(parts[1])

        // Resolve input
        var input string
        if runInputFile != "" {
            data, err := os.ReadFile(runInputFile)
            if err != nil {
                fmt.Fprintf(os.Stderr, "Error reading file '%s': %v\n", runInputFile, err)
                os.Exit(1)
            }
            input = string(data)
        } else if len(args) >= 2 {
            input = args[1]
        }

        // Stub behavior: just echo what would run
        fmt.Printf("🌾 LlamaFarm run (stub)\n")
        fmt.Printf("📁 Project: %s/%s\n", ns, proj)
        if runInputFile != "" {
            fmt.Printf("📄 Input file: %s\n", runInputFile)
        } else {
            fmt.Printf("💬 Inline input provided\n")
        }
        // Show a short preview of input for confirmation
        preview := input
        if len(preview) > 200 {
            preview = preview[:200] + "..."
        }
        fmt.Printf("\n— Input Preview —\n%s\n\n", preview)
        fmt.Println("(This is a stub. Actual execution will be implemented in a future update.)")
    },
}

func init() {
    runCmd.Flags().StringVarP(&runInputFile, "file", "f", "", "path to file containing input text")
    rootCmd.AddCommand(runCmd)
}
