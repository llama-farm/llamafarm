package cmd

import (
	"os"
	"os/signal"
	"strings"
	"syscall"

	"llamafarm-cli/cmd/config"

	"github.com/spf13/cobra"
)

// devCmd launches the chat quickly for development at the top level.
var devCmd = &cobra.Command{
	Use:     "dev",
	Short:   "Start working with your project locally",
	Aliases: []string{"start"},
	Long:    "Start your LlamaFarm project locally and open an interactive chat session for development and testing.",
	Run: func(cmd *cobra.Command, args []string) {
		if strings.TrimSpace(serverURL) == "" {
			serverURL = "http://localhost:8000"
		}

		// Resolve ollamaHost for dev: flag > env > default
		if !cmd.Flags().Changed("ollama-host") {
			if v := strings.TrimSpace(os.Getenv("OLLAMA_HOST")); v != "" {
				ollamaHost = v
			}
		}
		if strings.TrimSpace(ollamaHost) == "" {
			ollamaHost = "http://localhost:11434"
		}

		start(SessionModeDev)
	},
}

func start(mode SessionMode) {
	// Set up signal handler for cleanup
	setupCleanupHandler()

	// Load config to get namespace and project for watcher
	cwd := getEffectiveCWD()
	cfg, err := config.LoadConfig(cwd)
	if err != nil {
		OutputError("Error loading config: %v\nRun `lf init` to create a new project if none exists.\n", err)
		os.Exit(1)
	}

	projectInfo, err := cfg.GetProjectInfo()
	if err != nil {
		OutputWarning("Warning: Could not extract project info for watcher: %v\n", err)
	} else {
		// Start the config file watcher in background
		if err := StartConfigWatcher(projectInfo.Namespace, projectInfo.Project); err != nil {
			OutputWarning("Warning: Failed to start config watcher: %v\n", err)
		}
	}

	serverInfo, err := config.GetServerConfig(cwd, serverURL, "", "")
	if err != nil {
		OutputError("Error getting server config: %v\n", err)
		os.Exit(1)
	}
	serverURL = serverInfo.URL

	// Use new service orchestrator for development
	config := StartCommandConfig(serverURL)
	serverHealth, _ := EnsureServicesWithConfigAndResult(config)

	// Filter health status to avoid alarming messages for optional services (like RAG)
	// that are starting in the background
	if serverHealth != nil {
		serverHealth = FilterHealthForOptionalServices(serverHealth, config, mode)
	}

	runChatSessionTUI(mode, projectInfo, serverHealth)

	// Cleanup on exit
	cleanupNativeProcesses()
}

// setupCleanupHandler sets up signal handlers to cleanup processes on exit
func setupCleanupHandler() {
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)

	go func() {
		<-c
		// Cleanup processes
		cleanupNativeProcesses()
		os.Exit(0)
	}()
}

func init() {
	// Attach to root
	rootCmd.AddCommand(devCmd)

	// Add start-only flags
	devCmd.Flags().StringVar(&ollamaHost, "ollama-host", ollamaHost, "Ollama host URL")
}
