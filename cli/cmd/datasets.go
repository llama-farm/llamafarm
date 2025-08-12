package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"text/tabwriter"

	"github.com/spf13/cobra"
	"llamafarm-cli/cmd/config"
)

var (
	configFile string
	datasetParser string
)

// datasetsCmd represents the datasets command
var datasetsCmd = &cobra.Command{
	Use:   "datasets",
	Short: "Manage datasets in your LlamaFarm configuration",
	Long: `Manage datasets in your LlamaFarm configuration. Datasets are collections
of files that can be ingested into your RAG system for retrieval-augmented generation.

Available commands:
  list    - List all datasets in your configuration
  add     - Add a new dataset to your configuration
  remove  - Remove a dataset from your configuration
  ingest  - Start ingestion of specific dataset(s) or all datasets`,
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Println("LlamaFarm Datasets Management")
		cmd.Help()
	},
}

// datasetsListCmd represents the datasets list command
var datasetsListCmd = &cobra.Command{
	Use:   "list",
	Short: "List all datasets in your configuration",
	Long:  `List all datasets configured in your llamafarm.yaml file with their names, parsers, and file counts.`,
	Run: func(cmd *cobra.Command, args []string) {
		cfg, err := config.LoadConfig(configFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error loading config: %v\n", err)
			os.Exit(1)
		}

		if len(cfg.Datasets) == 0 {
			fmt.Println("No datasets configured.")
			return
		}

		fmt.Printf("Found %d dataset(s):\n\n", len(cfg.Datasets))

		// Create a tab writer for aligned output
		w := tabwriter.NewWriter(os.Stdout, 0, 0, 3, ' ', 0)
		fmt.Fprintln(w, "NAME\tPARSER\tFILES\tFILE PATHS")
		fmt.Fprintln(w, "----\t------\t-----\t----------")

		for _, dataset := range cfg.Datasets {
			parser := dataset.Parser
			if parser == "" {
				parser = "auto"
			}

			fileCount := len(dataset.Files)
			filePaths := strings.Join(dataset.Files, ", ")
			if len(filePaths) > 60 {
				filePaths = filePaths[:57] + "..."
			}

			fmt.Fprintf(w, "%s\t%s\t%d\t%s\n", dataset.Name, parser, fileCount, filePaths)
		}

		w.Flush()
	},
}

// datasetsAddCmd represents the datasets add command
var datasetsAddCmd = &cobra.Command{
	Use:   "add [name] [file1] [file2] ...",
	Short: "Add a new dataset to your configuration",
	Long: `Add a new dataset to your llamafarm.yaml configuration file.

Examples:
  lf datasets add my-docs ./docs/file1.pdf ./docs/file2.txt
  lf datasets add --parser pdf-parser my-pdfs ./pdfs/*.pdf`,
	Args: func(cmd *cobra.Command, args []string) error {
		if len(args) < 2 {
			return fmt.Errorf("requires at least 2 arguments: dataset name and at least one file path")
		}
		return nil
	},
	Run: func(cmd *cobra.Command, args []string) {
		cfg, err := config.LoadConfig(configFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error loading config: %v\n", err)
			os.Exit(1)
		}

		datasetName := args[0]
		filePaths := args[1:]

		// Validate file paths exist
		var validFiles []string
		for _, filePath := range filePaths {
			// Handle glob patterns
			matches, err := filepath.Glob(filePath)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Invalid file pattern '%s': %v\n", filePath, err)
				continue
			}

			if len(matches) == 0 {
				// Check if it's a direct file path
				if _, err := os.Stat(filePath); err != nil {
					fmt.Fprintf(os.Stderr, "Warning: File '%s' does not exist\n", filePath)
				}
				validFiles = append(validFiles, filePath)
			} else {
				validFiles = append(validFiles, matches...)
			}
		}

		if len(validFiles) == 0 {
			fmt.Fprintf(os.Stderr, "Error: No valid files provided\n")
			os.Exit(1)
		}

		// Create new dataset
		dataset := config.Dataset{
			Name:  datasetName,
			Files: validFiles,
		}

		if datasetParser != "" {
			dataset.Parser = datasetParser
		}

		// Add to configuration
		if err := cfg.AddDataset(dataset); err != nil {
			fmt.Fprintf(os.Stderr, "Error adding dataset: %v\n", err)
			os.Exit(1)
		}

		// Save configuration
		if err := config.SaveConfig(cfg, configFile); err != nil {
			fmt.Fprintf(os.Stderr, "Error saving config: %v\n", err)
			os.Exit(1)
		}

		fmt.Printf("✅ Successfully added dataset '%s' with %d file(s)\n", datasetName, len(validFiles))
		if datasetParser != "" {
			fmt.Printf("   Parser: %s\n", datasetParser)
		}
		fmt.Printf("   Files: %s\n", strings.Join(validFiles, ", "))
	},
}

// datasetsRemoveCmd represents the datasets remove command
var datasetsRemoveCmd = &cobra.Command{
	Use:   "remove [name]",
	Short: "Remove a dataset from your configuration",
	Long:  `Remove a dataset from your llamafarm.yaml configuration file.`,
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		cfg, err := config.LoadConfig(configFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error loading config: %v\n", err)
			os.Exit(1)
		}

		datasetName := args[0]

		// Check if dataset exists
		if dataset, _ := cfg.FindDatasetByName(datasetName); dataset == nil {
			fmt.Fprintf(os.Stderr, "Error: Dataset '%s' not found\n", datasetName)
			os.Exit(1)
		}

		// Remove from configuration
		if err := cfg.RemoveDataset(datasetName); err != nil {
			fmt.Fprintf(os.Stderr, "Error removing dataset: %v\n", err)
			os.Exit(1)
		}

		// Save configuration
		if err := config.SaveConfig(cfg, configFile); err != nil {
			fmt.Fprintf(os.Stderr, "Error saving config: %v\n", err)
			os.Exit(1)
		}

		fmt.Printf("✅ Successfully removed dataset '%s'\n", datasetName)
	},
}

// datasetsIngestCmd represents the datasets ingest command
var datasetsIngestCmd = &cobra.Command{
	Use:   "ingest [dataset-name]",
	Short: "Start ingestion of dataset(s)",
	Long: `Start ingestion of specific dataset(s) or all datasets if no name is provided.
This will process the files in the dataset(s) and add them to your vector store for retrieval.

Examples:
  lf datasets ingest                # Ingest all datasets
  lf datasets ingest my-docs        # Ingest specific dataset`,
	Args: cobra.MaximumNArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		cfg, err := config.LoadConfig(configFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error loading config: %v\n", err)
			os.Exit(1)
		}

		if len(cfg.Datasets) == 0 {
			fmt.Println("No datasets configured for ingestion.")
			return
		}

		var datasetsToIngest []config.Dataset

		if len(args) == 0 {
			// Ingest all datasets
			datasetsToIngest = cfg.Datasets
			fmt.Printf("Starting ingestion of all %d dataset(s)...\n", len(datasetsToIngest))
		} else {
			// Ingest specific dataset
			datasetName := args[0]
			dataset, _ := cfg.FindDatasetByName(datasetName)
			if dataset == nil {
				fmt.Fprintf(os.Stderr, "Error: Dataset '%s' not found\n", datasetName)
				os.Exit(1)
			}
			datasetsToIngest = []config.Dataset{*dataset}
			fmt.Printf("Starting ingestion of dataset '%s'...\n", datasetName)
		}

		// TODO: Implement actual ingestion logic
		// For now, this is a placeholder that shows what would be ingested
		for _, dataset := range datasetsToIngest {
			fmt.Printf("\n📊 Dataset: %s\n", dataset.Name)
			parser := dataset.Parser
			if parser == "" {
				parser = "auto"
			}
			fmt.Printf("   Parser: %s\n", parser)
			fmt.Printf("   Files to process: %d\n", len(dataset.Files))

			for i, file := range dataset.Files {
				fmt.Printf("   [%d] %s\n", i+1, file)
			}
		}

		fmt.Printf("\n🚀 Ingestion process would start for %d dataset(s)\n", len(datasetsToIngest))
		fmt.Printf("💡 Note: Actual ingestion implementation coming soon!\n")
		fmt.Printf("   This will process files using the configured RAG pipeline:\n")
		fmt.Printf("   - Parse files using specified parsers\n")
		fmt.Printf("   - Generate embeddings using configured embedders\n")
		fmt.Printf("   - Store vectors in configured vector stores\n")
	},
}

func init() {
	// Add persistent flags
	datasetsCmd.PersistentFlags().StringVarP(&configFile, "config", "c", "", "config file path (default: llamafarm.yaml in current directory)")

	// Add flags specific to add command
	datasetsAddCmd.Flags().StringVarP(&datasetParser, "parser", "p", "", "parser to use for this dataset (default: auto)")

	// Add subcommands to datasets
	datasetsCmd.AddCommand(datasetsListCmd)
	datasetsCmd.AddCommand(datasetsAddCmd)
	datasetsCmd.AddCommand(datasetsRemoveCmd)
	datasetsCmd.AddCommand(datasetsIngestCmd)

	// Add the datasets command to root
	rootCmd.AddCommand(datasetsCmd)
}