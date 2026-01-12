---
name: cli-skills
description: CLI best practices for LlamaFarm. Covers Cobra, Bubbletea, Lipgloss patterns for Go CLI development.
allowed-tools: Read, Grep, Glob
user-invocable: false
---

# CLI Skills for LlamaFarm

Framework-specific patterns for the LlamaFarm CLI. These guidelines extend the [shared Go skills](../go-skills/SKILL.md) with Cobra, Bubbletea, and Lipgloss best practices.

## Tech Stack

- Go 1.24+
- Cobra (CLI framework)
- Bubbletea (TUI framework)
- Lipgloss (terminal styling)
- Bubbles (TUI components)

## Directory Structure

```
cli/
  cmd/             # Cobra command implementations
    config/        # Configuration types and loading
    orchestrator/  # Service management and process control
    utils/         # Shared utilities (HTTP, output, logging)
    version/       # Version and upgrade handling
  internal/        # Internal packages (not exported)
    tui/           # Reusable TUI components
    buildinfo/     # Build-time information
```

## Quick Reference

### Cobra Commands
- Use `RunE` over `Run` for error handling
- Register flags in `init()` functions
- Use persistent flags for shared options
- Validate arguments with `Args` field

### Bubbletea TUI
- Implement `Init()`, `Update()`, `View()` interface
- Use message types for state changes
- Return `tea.Cmd` for async operations
- Keep state immutable in `Update()`

### Lipgloss Styling
- Define styles as package-level constants
- Use `lipgloss.NewStyle()` for styling
- Handle terminal width dynamically
- Support color themes via style variables

## Shared Go Skills

This skill extends the base Go skills. See:

| Link | Description |
|------|-------------|
| [go-skills/SKILL.md](../go-skills/SKILL.md) | Overview and quick reference |
| [go-skills/patterns.md](../go-skills/patterns.md) | Idiomatic Go patterns |
| [go-skills/concurrency.md](../go-skills/concurrency.md) | Goroutines, channels, sync |
| [go-skills/error-handling.md](../go-skills/error-handling.md) | Error wrapping, sentinels |
| [go-skills/testing.md](../go-skills/testing.md) | Table-driven tests, mocks |
| [go-skills/security.md](../go-skills/security.md) | Input validation, secure coding |

## CLI-Specific Checklists

| File | Description |
|------|-------------|
| [cobra.md](cobra.md) | Cobra command patterns, flags, validation |
| [bubbletea.md](bubbletea.md) | Bubbletea Model/Update/View patterns |
| [performance.md](performance.md) | CLI-specific optimizations |

## Key Patterns in This Codebase

### Command Registration Pattern
```go
var myCmd = &cobra.Command{
    Use:   "mycommand [args]",
    Short: "Brief description",
    Long:  `Extended description with examples.`,
    Args:  cobra.MaximumNArgs(1),
    RunE: func(cmd *cobra.Command, args []string) error {
        // Implementation with error returns
        return nil
    },
}

func init() {
    rootCmd.AddCommand(myCmd)
    myCmd.Flags().StringVar(&flagVar, "flag", "default", "Flag description")
}
```

### TUI Model Pattern
```go
type myModel struct {
    viewport viewport.Model
    textarea textarea.Model
    width    int
    height   int
    err      error
}

func (m myModel) Init() tea.Cmd {
    return tea.Batch(m.textarea.Focus(), doAsyncWork())
}

func (m myModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
    var cmds []tea.Cmd
    switch msg := msg.(type) {
    case tea.WindowSizeMsg:
        m.width = msg.Width
        m.height = msg.Height
    case tea.KeyMsg:
        switch msg.String() {
        case "ctrl+c":
            return m, tea.Quit
        }
    }
    return m, tea.Batch(cmds...)
}

func (m myModel) View() string {
    return lipgloss.JoinVertical(lipgloss.Left,
        m.viewport.View(),
        m.textarea.View(),
    )
}
```

### Output API Pattern
```go
// Use the output API for consistent messaging
utils.OutputInfo("Starting service %s...", serviceName)
utils.OutputSuccess("Service started successfully")
utils.OutputError("Failed to start service: %v", err)
utils.OutputProgress("Downloading model...")
utils.OutputWarning("Service already running")
```

### Service Orchestration Pattern
```go
// Ensure services are running before operations
factory := GetServiceConfigFactory()
config := factory.ServerOnly(serverURL)
orchestrator.EnsureServicesOrExitWithConfig(config, "server")

// Or with multiple services
config := factory.RAGCommand(serverURL)
orchestrator.EnsureServicesOrExitWithConfig(config, "server", "rag", "universal-runtime")
```

## Guidelines

1. **User Experience First**: CLI should feel responsive and provide clear feedback
2. **Graceful Degradation**: Handle missing services, network errors, and timeouts gracefully
3. **Consistent Output**: Use the output API for all user-facing messages
4. **Cross-Platform**: Test on macOS, Linux, and Windows
5. **Terminal Compatibility**: Test with different terminal emulators and sizes
