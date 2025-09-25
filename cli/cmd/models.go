package cmd

import (
	"fmt"

	"github.com/spf13/cobra"
)

// modelsCmd represents the models command namespace
var modelsCmd = &cobra.Command{
	Use:   "models",
	Short: "Manage models and model backends",
	Long: `Manage models, providers, and backends configured in LlamaFarm.

Available commands will include listing models, testing inference, and syncing configs.`,
	Hidden: true,
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Println("LlamaFarm Models Management")
		cmd.Help()
	},
}

func init() {
	rootCmd.AddCommand(modelsCmd)
}
