package cmd

import (
	"encoding/json"
	"fmt"
	"os"
	"text/tabwriter"

	"llamafarm-cli/cmd/config"

	"github.com/spf13/cobra"
)

// modelsCmd represents the models command
var modelsCmd = &cobra.Command{
	Use:   "models",
	Short: "Manage model configurations",
	Long: `Manage model configurations including listing, showing details, and setting defaults.

Examples:
  # List all configured models
  lf models list

  # Show details for a specific model
  lf models show creative

  # Set the default model
  lf models set-default precise

  # Import all available Ollama models
  lf models import-ollama

  # Import with custom prefix
  lf models import-ollama --prefix ollama-`,
}

// modelsListCmd lists all configured models
var modelsListCmd = &cobra.Command{
	Use:   "list",
	Short: "List all configured models",
	Long: `List all configured models with their providers and default status.

Example:
  lf models list`,
	Run: func(cmd *cobra.Command, args []string) {
		cwd := getEffectiveCWD()
		cfg, err := config.LoadConfig(cwd)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error loading config: %v\n", err)
			os.Exit(1)
		}

		// Get runtime models from config
		runtimeModels := cfg.GetRuntimeModels()
		defaultModel := cfg.GetDefaultModel()

		if len(runtimeModels) == 0 {
			// Check for legacy runtime
			if runtime := cfg.GetRuntime(); runtime != nil {
				fmt.Println("Using legacy runtime configuration (migrate to runtime_models for multi-model support)")
				fmt.Printf("Provider: %s\n", runtime.Provider)
				fmt.Printf("Model: %s\n", runtime.Model)
				return
			}
			fmt.Println("No models configured")
			return
		}

		// Print table header
		w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
		fmt.Fprintf(w, "NAME\tPROVIDER\tMODEL\tDEFAULT\n")

		// Print each model
		for _, model := range runtimeModels {
			name := model.Name
			provider := model.Provider
			modelID := model.Model
			isDefault := ""
			if name == defaultModel {
				isDefault = "✓"
			}
			fmt.Fprintf(w, "%s\t%s\t%s\t%s\n", name, provider, modelID, isDefault)
		}

		w.Flush()
	},
}

// modelsShowCmd shows details for a specific model
var modelsShowCmd = &cobra.Command{
	Use:   "show [model-name]",
	Short: "Show details for a specific model",
	Long: `Show detailed configuration for a specific model.

Example:
  lf models show creative`,
	Args: cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		modelName := args[0]
		cwd := getEffectiveCWD()
		cfg, err := config.LoadConfig(cwd)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error loading config: %v\n", err)
			os.Exit(1)
		}

		// Find the model
		model := cfg.GetRuntimeModel(modelName)
		if model == nil {
			fmt.Fprintf(os.Stderr, "Model '%s' not found\n", modelName)
			os.Exit(1)
		}

		// Display model details
		fmt.Printf("Name: %s\n", model.Name)
		fmt.Printf("Provider: %s\n", model.Provider)
		fmt.Printf("Model: %s\n", model.Model)
		
		if model.BaseURL != "" {
			fmt.Printf("Base URL: %s\n", model.BaseURL)
		}
		
		if model.InstructorMode != "" {
			fmt.Printf("Instructor Mode: %s\n", model.InstructorMode)
		}

		// Display parameters if present
		if model.Parameters != nil {
			fmt.Println("\nParameters:")
			// Marshal to JSON for pretty printing
			paramsJSON, err := json.MarshalIndent(model.Parameters, "  ", "  ")
			if err == nil {
				fmt.Printf("  %s\n", string(paramsJSON))
			}
		}

		// Check if this is the default
		if model.Name == cfg.GetDefaultModel() {
			fmt.Println("\n✓ This is the default model")
		}
	},
}

// modelsSetDefaultCmd sets the default model
var modelsSetDefaultCmd = &cobra.Command{
	Use:   "set-default [model-name]",
	Short: "Set the default model",
	Long: `Set the default model to use when no --model flag is specified.

Example:
  lf models set-default precise`,
	Args: cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		modelName := args[0]
		cwd := getEffectiveCWD()
		
		// Load config
		cfg, err := config.LoadConfig(cwd)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error loading config: %v\n", err)
			os.Exit(1)
		}

		// Verify model exists
		model := cfg.GetRuntimeModel(modelName)
		if model == nil {
			fmt.Fprintf(os.Stderr, "Model '%s' not found\n", modelName)
			
			// List available models
			models := cfg.GetRuntimeModels()
			if len(models) > 0 {
				fmt.Fprintf(os.Stderr, "\nAvailable models:\n")
				for _, m := range models {
					fmt.Fprintf(os.Stderr, "  - %s\n", m.Name)
				}
			}
			os.Exit(1)
		}

		// Update default_model in config
		err = cfg.SetDefaultModel(modelName)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error setting default model: %v\n", err)
			os.Exit(1)
		}

		// Save config
		configPath := getConfigPath(cwd)
		err = cfg.SaveToFile(configPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error saving config: %v\n", err)
			os.Exit(1)
		}

		fmt.Printf("✓ Set default model to '%s'\n", modelName)
	},
}

// Variables for import-ollama command
var (
	importPrefix      string
	importSetDefault  string
	importFilters     []string
)

// modelsImportOllamaCmd imports models from Ollama
var modelsImportOllamaCmd = &cobra.Command{
	Use:   "import-ollama",
	Short: "Import models from Ollama",
	Long: `Discover and import all available models from Ollama.

Examples:
  # Import all available models
  lf models import-ollama

  # Import with custom prefix
  lf models import-ollama --prefix ollama-

  # Import and set default
  lf models import-ollama --set-default llama3.2:3b

  # Import only specific models
  lf models import-ollama --filter "llama*" --filter "mistral*"`,
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Println("Discovering Ollama models...")
		
		// This would require calling the RuntimeManager's import_ollama_models method
		// For now, we'll show a placeholder implementation
		fmt.Println("Note: Full Ollama import requires Python RuntimeManager integration")
		fmt.Println("This feature will be available after server integration")
		
		// Placeholder for discovered models
		fmt.Println("\nExample output:")
		fmt.Println("Found 8 models:")
		fmt.Println("  ✓ llama3.1:8b (already configured as 'primary')")
		fmt.Println("  + Adding llama3.2:3b as 'llama3-2-3b'")
		fmt.Println("  + Adding mistral:7b as 'mistral-7b'")
		fmt.Println("  + Adding phi3:mini as 'phi3-mini'")
		fmt.Println("\nTo implement: Call Python RuntimeManager via server API")
	},
}

func init() {
	// Add subcommands to models command
	modelsCmd.AddCommand(modelsListCmd)
	modelsCmd.AddCommand(modelsShowCmd)
	modelsCmd.AddCommand(modelsSetDefaultCmd)
	modelsCmd.AddCommand(modelsImportOllamaCmd)

	// Add flags for import-ollama
	modelsImportOllamaCmd.Flags().StringVar(&importPrefix, "prefix", "", "Prefix to add to imported model names")
	modelsImportOllamaCmd.Flags().StringVar(&importSetDefault, "set-default", "", "Set this model as default after import")
	modelsImportOllamaCmd.Flags().StringSliceVar(&importFilters, "filter", []string{}, "Filter patterns for models to import")

	// Add models command to root
	rootCmd.AddCommand(modelsCmd)
}

// Helper function to get config path
func getConfigPath(cwd string) string {
	return fmt.Sprintf("%s/llamafarm.yaml", cwd)
}