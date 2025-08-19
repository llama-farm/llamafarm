package cmd

import (
	"fmt"

	"github.com/spf13/cobra"
)

// ragCmd represents the rag command namespace
var ragCmd = &cobra.Command{
	Use:   "rag",
	Short: "Manage RAG data and operations",
	Long: `Manage Retrieval-Augmented Generation (RAG) data and operations for LlamaFarm projects.

Available commands will include data ingestion, status checks, and maintenance tasks.`,
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Println("LlamaFarm RAG Management")
		cmd.Help()
	},
}

func init() {
	rootCmd.AddCommand(ragCmd)
}
