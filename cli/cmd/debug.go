// cli/cmd/debug.go
package cmd

import (
	"github.com/spf13/cobra"
)

var debugCmd = &cobra.Command{
	Use:   "debug",
	Short: "Debugging utilities for local LlamaFarm stacks",
	Long:  "Tools to inspect, diagnose, and troubleshoot local LlamaFarm dev services.",
}

func init() {
	rootCmd.AddCommand(debugCmd)
}

