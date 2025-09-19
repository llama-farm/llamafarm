package cmd

import (
	"github.com/spf13/cobra"
)

// ragCmd represents the rag command namespace
var ragCmd = &cobra.Command{
	Use:   "rag",
	Short: "Manage RAG data and operations",
	Long: `Manage Retrieval-Augmented Generation (RAG) data and operations for LlamaFarm projects.

The RAG system allows you to:
• Query documents using semantic search
• View database statistics and health
• Manage database content (clear, delete, prune)
• Export and import data for backup/migration`,
}

func init() {
	rootCmd.AddCommand(ragCmd)
}
