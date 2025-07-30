package cmd

import (
	"fmt"
	"llamafarm-cli/cmd/config"
	"os"

	"github.com/spf13/cobra"
)

// initCmd represents the init command
var initCmd = &cobra.Command{
	Use:   "init",
	Short: "Initialize a new LlamaFarm project",
	Long:  `Initialize a new LlamaFarm project with a default configuration.`,
	Args:  cobra.MaximumNArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Println("Initializing a new LlamaFarm project...")

		// Create a new project directory
		projectDir := "."
		if len(args) > 0 {
			projectDir = args[0]
		}

		if projectDir != "." {
			os.MkdirAll(projectDir, 0755)
		}

		configPath := projectDir + "/llamafarm.yaml"

		if _, err := os.Stat(configPath); err == nil {
			fmt.Printf("llamafarm.yaml already exists at %s\n", configPath)
		} else {
			configContent, err := config.Generate()
			if err != nil {
				fmt.Printf("Failed to generate llamafarm.yaml: %v\n", err)
				return
			}

			err = os.WriteFile(configPath, []byte(configContent), 0644)
			if err != nil {
				fmt.Printf("Failed to create llamafarm.yaml: %v\n", err)
			} else {
				absPath, err := os.Getwd()
				if err == nil && projectDir != "." {
					absPath = absPath + "/" + projectDir
				}
				fmt.Printf("Created llamafarm.yaml in %s\n", absPath)
			}
		}
	},
}

func init() {
	rootCmd.AddCommand(initCmd)
}
