package cmd

import (
	"fmt"
	"os"
	"os/exec"

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

		// Pull the latest llamafarm image if needed
		fmt.Println("Pulling latest LlamaFarm image...")
		if err := pullImage("ghcr.io/llama-farm/llamafarm/designer:latest"); err != nil {
			fmt.Printf("Warning: Failed to pull latest image: %v\n", err)
			fmt.Println("Continuing with existing local image...")
		}

		// Start the container
		fmt.Println("Starting container...")
		dockerArgs := []string{
			"run",
			"-d", // Run in detached mode
			"--name", "llamafarm-designer",
			"-p", "8080:8080", // Map port 8080
			"-v", fmt.Sprintf("%s:/workspace", getEffectiveCWD()), // Mount current directory
			"ghcr.io/llama-farm/llamafarm/designer:latest",
		}

		startCmd := exec.Command("docker", dockerArgs...)
		output, err := startCmd.CombinedOutput()

		if err != nil {
			fmt.Fprintf(os.Stderr, "Error starting container: %v\n", err)
			fmt.Fprintf(os.Stderr, "Output: %s\n", output)
			os.Exit(1)
		}

		fmt.Println("🌾 LlamaFarm designer started successfully!")
		fmt.Println("🌐 Open your browser and navigate to: http://localhost:8080")
		fmt.Println("📁 Your current directory is mounted at /workspace in the container")
		fmt.Println("\nTo stop the designer, run: lf designer stop")
	},
}

func init() {
	// Add the start subcommand to designer
	designerCmd.AddCommand(designerStartCmd)

	// Add the designer command to root
	rootCmd.AddCommand(designerCmd)
}
