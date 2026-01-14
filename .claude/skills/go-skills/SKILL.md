---
name: go-skills
description: Shared Go best practices for LlamaFarm CLI. Covers idiomatic patterns, error handling, and testing.
allowed-tools: Read, Grep, Glob
user-invocable: false
---

# Go Skills for LlamaFarm CLI

Shared Go best practices for LlamaFarm CLI development. These guidelines ensure idiomatic, maintainable, and secure Go code.

## Tech Stack

- Go 1.24+
- Cobra (CLI framework)
- Bubbletea (TUI framework)
- Lipgloss (terminal styling)

## Directory Structure

```
cli/
  cmd/           # Command implementations
    config/      # Configuration types and loading
    orchestrator/ # Service management
    utils/       # Shared utilities
    version/     # Version and upgrade handling
  internal/      # Internal packages
    tui/         # TUI components
    buildinfo/   # Build information
```

## Quick Reference

### Error Handling
- Always wrap errors with context: `fmt.Errorf("operation failed: %w", err)`
- Use sentinel errors for expected conditions: `var ErrNotFound = errors.New("not found")`
- Check errors immediately after function calls

### Concurrency
- Use `sync.Mutex` for shared state protection
- Use `sync.RWMutex` when reads dominate writes
- Use channels for goroutine communication
- Always use `defer` for mutex unlocks

### Testing
- Use table-driven tests for comprehensive coverage
- Use interfaces for mockability
- Test file names: `*_test.go` in same package

### Security
- Never log credentials or tokens
- Redact sensitive headers in debug logs
- Validate all external input
- Use `context.Context` for cancellation

## Checklist Files

| File | Description |
|------|-------------|
| [patterns.md](patterns.md) | Idiomatic Go patterns |
| [concurrency.md](concurrency.md) | Goroutines, channels, sync |
| [error-handling.md](error-handling.md) | Error wrapping, sentinels |
| [testing.md](testing.md) | Table-driven tests, mocks |
| [security.md](security.md) | Input validation, secure coding |

## Go Proverbs to Remember

1. "Don't communicate by sharing memory; share memory by communicating"
2. "Errors are values"
3. "A little copying is better than a little dependency"
4. "Clear is better than clever"
5. "Design the architecture, name the components, document the details"

## Common Patterns in This Codebase

### HTTP Client Interface
```go
type HTTPClient interface {
    Do(req *http.Request) (*http.Response, error)
}
```

### Process Management with Mutex
```go
type ProcessManager struct {
    mu        sync.RWMutex
    processes map[string]*ProcessInfo
}
```

### Cobra Command Pattern
```go
var myCmd = &cobra.Command{
    Use:   "mycommand",
    Short: "Brief description",
    RunE: func(cmd *cobra.Command, args []string) error {
        // Implementation
        return nil
    },
}
```

### Bubbletea Model Pattern
```go
type myModel struct {
    // State fields
}

func (m myModel) Init() tea.Cmd { return nil }
func (m myModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) { /* ... */ }
func (m myModel) View() string { return "" }
```
