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
		if err := ensureServerAvailable(serverURL); err != nil {
			fmt.Fprintf(os.Stderr, "Error ensuring server availability: %v\n", err)
		}
		runChatSessionTUI()
	},
}

func init() {
	// Attach to root
	rootCmd.AddCommand(devCmd)
	// Provide a hint if server URL isn't set
	if serverURL == "" {
		fmt.Fprintln(os.Stderr, "Hint: use --server-url to point to a specific server")
	}
}
