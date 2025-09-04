package cmd

import (
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"
)

// devCmd launches the chat quickly for development at the top level.
var devCmd = &cobra.Command{
	Use:   "dev",
	Short: "Developer mode: launch your project locally",
	Long:  "Start an interactive chat session quickly for development and testing.",
	Run: func(cmd *cobra.Command, args []string) {
		if strings.TrimSpace(serverURL) == "" {
			serverURL = "http://localhost:8000"
		}
		if err := ensureInferenceRuntimeAvailable(); err != nil {
			fmt.Fprintf(os.Stderr, "Error ensuring inference runtime availability: %v\n", err)
			os.Exit(1)
		}
		if err := ensureServerAvailable(serverURL); err != nil {
			fmt.Fprintf(os.Stderr, "Error ensuring server availability: %v\n", err)
		}
		runChatSessionTUI()
	},
}

func init() {
	// Attach to root
	rootCmd.AddCommand(devCmd)
}
