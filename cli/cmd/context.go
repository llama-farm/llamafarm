package cmd

import (
	"time"

	"github.com/llamafarm/cli/cmd/orchestrator"
	"github.com/llamafarm/cli/cmd/utils"
)

type CLIContext struct {
	// Global flags
	Debug              bool
	ServerURL          string
	OllamaHost         string
	ServerStartTimeout time.Duration
	OverrideCwd        string
	AutoStart          bool
}

// NewCLIContext creates a new CLIContext with default values.
func NewCLIContext() *CLIContext {
	return &CLIContext{
		Debug:              false,
		ServerURL:          "http://localhost:8000",
		OllamaHost:         "http://localhost:11434",
		ServerStartTimeout: 45 * time.Second,
		OverrideCwd:        "",
		AutoStart:          true,
	}
}

// GetCLIContext returns the current CLI context from global flags.
// This is a transitional helper during refactoring.
func GetCLIContext() *CLIContext {
	return &CLIContext{
		Debug:              debug,
		ServerURL:          serverURL,
		OllamaHost:         ollamaHost,
		ServerStartTimeout: serverStartTimeout,
		OverrideCwd:        utils.OverrideCwd,
		AutoStart:          autoStart,
	}
}

// ServiceConfigFactory creates service orchestration configs with CLI context.
type ServiceConfigFactory struct {
	ctx *CLIContext
}

// NewServiceConfigFactory creates a factory that automatically injects CLI context.
func NewServiceConfigFactory(ctx *CLIContext) *ServiceConfigFactory {
	return &ServiceConfigFactory{ctx: ctx}
}

// StartCommand creates config for lf start - Server required, RAG optional (background)
func (f *ServiceConfigFactory) StartCommand(serverURL string) *orchestrator.ServiceOrchestrationConfig {
	return &orchestrator.ServiceOrchestrationConfig{
		ServerURL:   serverURL,
		PrintStatus: true,
		ServiceNeeds: map[string]orchestrator.ServiceRequirement{
			"server": orchestrator.ServiceRequired,
			"rag":    orchestrator.ServiceOptional,
		},
		DefaultTimeout: f.ctx.ServerStartTimeout,
		AutoStart:       f.ctx.AutoStart,
	}
}

// RAGCommand creates config for RAG commands - Both server and RAG required
func (f *ServiceConfigFactory) RAGCommand(serverURL string) *orchestrator.ServiceOrchestrationConfig {
	return &orchestrator.ServiceOrchestrationConfig{
		ServerURL:   serverURL,
		PrintStatus: true,
		ServiceNeeds: map[string]orchestrator.ServiceRequirement{
			"server": orchestrator.ServiceRequired,
			"rag":    orchestrator.ServiceRequired,
		},
		DefaultTimeout: f.ctx.ServerStartTimeout,
		AutoStart:       f.ctx.AutoStart,
	}
}

// ChatNoRAG creates config for lf chat --no-rag - Only server
func (f *ServiceConfigFactory) ChatNoRAG(serverURL string) *orchestrator.ServiceOrchestrationConfig {
	return &orchestrator.ServiceOrchestrationConfig{
		ServerURL:   serverURL,
		PrintStatus: true,
		ServiceNeeds: map[string]orchestrator.ServiceRequirement{
			"server": orchestrator.ServiceRequired,
		},
		DefaultTimeout: f.ctx.ServerStartTimeout,
		AutoStart:       f.ctx.AutoStart,
	}
}

// ServerOnly creates config for server-only commands - Server required, RAG optional (background)
func (f *ServiceConfigFactory) ServerOnly(serverURL string) *orchestrator.ServiceOrchestrationConfig {
	return &orchestrator.ServiceOrchestrationConfig{
		ServerURL:   serverURL,
		PrintStatus: true,
		ServiceNeeds: map[string]orchestrator.ServiceRequirement{
			"server": orchestrator.ServiceRequired,
			"rag":    orchestrator.ServiceOptional,
		},
		DefaultTimeout: f.ctx.ServerStartTimeout,
		AutoStart:       f.ctx.AutoStart,
	}
}

// Global factory instance for transitional use
var globalServiceConfigFactory *ServiceConfigFactory

// GetServiceConfigFactory returns a factory using the current CLI context.
// This is a transitional helper that allows gradual migration.
func GetServiceConfigFactory() *ServiceConfigFactory {
	if globalServiceConfigFactory == nil {
		globalServiceConfigFactory = NewServiceConfigFactory(GetCLIContext())
	}
	return globalServiceConfigFactory
}

// ResetServiceConfigFactory resets the global factory (useful for tests)
func ResetServiceConfigFactory() {
	globalServiceConfigFactory = nil
}
