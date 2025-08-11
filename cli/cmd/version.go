package cmd

import (
	"fmt"

	"github.com/spf13/cobra"
)

// Version will be set by build flags during release builds
var Version = "dev"

// versionCmd represents the version command
var versionCmd = &cobra.Command{
	Use:   "version",
	Short: "Print the version number of LlamaFarm CLI",
	Long:  "Print the version number of LlamaFarm CLI",
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Printf("LlamaFarm CLI v%s\n", Version)
	},
}

func init() {
	rootCmd.AddCommand(versionCmd)
}