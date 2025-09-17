package cmd

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/spf13/cobra"
)

// designerCmd represents the designer command
var designerCmd = &cobra.Command{
	Use:   "designer",
	Short: "Manage LlamaFarm designer environment",
	Long:  `Commands for managing the LlamaFarm designer environment, including starting and stopping the llamafarm designer and runtime.`,
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Println("LlamaFarm Designer")
		cmd.Help()
	},
}

// designerStartCmd represents the designer start command
var designerStartCmd = &cobra.Command{
	Use:   "start",
	Short: "Start the LlamaFarm designer container",
	Long:  `Start the LlamaFarm designer container to access the web-based designer interface.`,
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Println("Starting LlamaFarm designer container...")

		// Check if Docker is available
		if err := ensureDockerAvailable(); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}

		// Determine preferred port (default 7724) with env override
		preferred := 7724
		if v := strings.TrimSpace(os.Getenv("LF_DESIGNER_PORT")); v != "" {
			if p, err := strconv.Atoi(v); err == nil && p > 0 && p <= 65535 {
				preferred = p
			}
		}

		url, err := StartDesignerInBackground(cmd.Context(), DesignerLaunchOptions{PreferredPort: preferred, Forced: false})
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error starting container: %v\n", err)
			os.Exit(1)
		}

		fmt.Println("🌾 LlamaFarm designer started successfully!")
		fmt.Printf("🌐 Open your browser and navigate to: %s\n", url)
		fmt.Println("\nTo stop the designer, run: lf designer stop")
	},
}

func init() {
	// Add the start subcommand to designer
	designerCmd.AddCommand(designerStartCmd)

	// Add the designer command to root
	rootCmd.AddCommand(designerCmd)
}
