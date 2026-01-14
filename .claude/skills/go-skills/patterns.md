# Idiomatic Go Patterns

Best practices for writing idiomatic Go code in the LlamaFarm CLI.

## Checklist

### 1. Use Named Return Values Sparingly

**Description**: Named return values should only be used when they add clarity to documentation or enable deferred error handling.

**Search Pattern**:
```bash
grep -rn "func.*\(.*\).*\(.*,.*\)" cli/ --include="*.go" | grep -v "_test.go"
```

**Pass Criteria**: Named returns are used for documentation or defer patterns, not just convenience.

**Fail Criteria**: Named returns used unnecessarily, leading to confusing code with naked returns.

**Severity**: Low

**Recommendation**: Remove named returns unless they serve documentation or defer purposes. Use explicit returns.

---

### 2. Accept Interfaces, Return Structs

**Description**: Functions should accept interfaces for flexibility and return concrete types for clarity.

**Search Pattern**:
```bash
grep -rn "func.*interface{}" cli/ --include="*.go"
```

**Pass Criteria**: Functions accept narrow interfaces (e.g., `io.Reader`) and return concrete structs.

**Fail Criteria**: Functions return interfaces or accept overly broad interfaces like `interface{}`.

**Severity**: Medium

**Recommendation**: Define small, focused interfaces. Return concrete types. Example:
```go
// Good
type HTTPClient interface {
    Do(req *http.Request) (*http.Response, error)
}

func NewClient() *DefaultHTTPClient { ... }

// Avoid
func NewClient() HTTPClient { ... }  // Returns interface
```

---

### 3. Use Constructor Functions

**Description**: Complex structs should have constructor functions that validate and initialize.

**Search Pattern**:
```bash
grep -rn "func New" cli/ --include="*.go"
```

**Pass Criteria**: Constructors validate inputs, set defaults, and return errors when appropriate.

**Fail Criteria**: Struct initialization scattered throughout code without validation.

**Severity**: Medium

**Recommendation**: Use `NewXxx` pattern:
```go
func NewProcessManager() (*ProcessManager, error) {
    lfDataDir, err := utils.GetLFDataDir()
    if err != nil {
        return nil, fmt.Errorf("failed to get LF data directory: %w", err)
    }
    // ... initialization
    return &ProcessManager{...}, nil
}
```

---

### 4. Use Functional Options for Configuration

**Description**: For structs with many optional configuration parameters, use functional options.

**Search Pattern**:
```bash
grep -rn "type.*Option.*func" cli/ --include="*.go"
```

**Pass Criteria**: Complex configuration uses functional options pattern for clarity and extensibility.

**Fail Criteria**: Long constructor parameter lists or excessive struct field exposure.

**Severity**: Low

**Recommendation**:
```go
type Option func(*Config)

func WithTimeout(d time.Duration) Option {
    return func(c *Config) { c.Timeout = d }
}

func NewClient(opts ...Option) *Client {
    cfg := defaultConfig()
    for _, opt := range opts {
        opt(&cfg)
    }
    return &Client{cfg: cfg}
}
```

---

### 5. Embed for Composition

**Description**: Use struct embedding for composition instead of inheritance-like patterns.

**Search Pattern**:
```bash
grep -rn "type.*struct {$" -A5 cli/ --include="*.go" | grep -E "^\s+\*?[A-Z]"
```

**Pass Criteria**: Embedding used to share behavior (e.g., embedding `sync.Mutex`).

**Fail Criteria**: Deep inheritance hierarchies or excessive embedding that obscures behavior.

**Severity**: Low

**Recommendation**:
```go
type ProcessInfo struct {
    Name    string
    Cmd     *exec.Cmd
    mu      sync.RWMutex  // Embedded for locking
}
```

---

### 6. Use `defer` for Cleanup

**Description**: Use `defer` for resource cleanup to ensure cleanup happens even on error paths.

**Search Pattern**:
```bash
grep -rn "defer" cli/ --include="*.go"
```

**Pass Criteria**: All file handles, locks, and connections use `defer` for cleanup.

**Fail Criteria**: Manual cleanup at multiple return points, risk of resource leaks.

**Severity**: High

**Recommendation**:
```go
func readFile(path string) ([]byte, error) {
    f, err := os.Open(path)
    if err != nil {
        return nil, err
    }
    defer f.Close()  // Always executes

    return io.ReadAll(f)
}
```

---

### 7. Keep Functions Small and Focused

**Description**: Functions should do one thing well. Extract helper functions for complex logic.

**Search Pattern**:
```bash
# Find functions longer than 50 lines
awk '/^func /{start=NR; name=$0} /^}$/ && start{if(NR-start>50) print FILENAME":"start": "name}' cli/**/*.go
```

**Pass Criteria**: Most functions are under 50 lines. Complex logic is extracted into helpers.

**Fail Criteria**: Monolithic functions with deeply nested logic.

**Severity**: Medium

**Recommendation**: Extract logical blocks into well-named helper functions. Use early returns.

---

### 8. Use Constants for Magic Values

**Description**: Define constants for repeated values and configuration defaults.

**Search Pattern**:
```bash
grep -rn "const (" cli/ --include="*.go"
```

**Pass Criteria**: Timeouts, buffer sizes, and configuration values are defined as constants.

**Fail Criteria**: Magic numbers scattered throughout code.

**Severity**: Medium

**Recommendation**:
```go
const (
    ServiceLockTimeout     = 30 * time.Second
    ServiceLockPollInterval = 500 * time.Millisecond
    PIDFileWaitTimeout     = 10 * time.Second
)
```

---

### 9. Use Type Aliases for Clarity

**Description**: Define type aliases to add semantic meaning to primitive types.

**Search Pattern**:
```bash
grep -rn "^type.*int$\|^type.*string$" cli/ --include="*.go"
```

**Pass Criteria**: Domain-specific types like `SessionMode` or `ServiceStatus` are defined.

**Fail Criteria**: Raw primitives used everywhere without semantic context.

**Severity**: Low

**Recommendation**:
```go
type SessionMode int

const (
    SessionModeProject SessionMode = iota
    SessionModeStateless
    SessionModeDev
)
```

---

### 10. Avoid Package-Level State

**Description**: Minimize package-level variables. Prefer dependency injection.

**Search Pattern**:
```bash
grep -rn "^var " cli/ --include="*.go" | grep -v "_test.go"
```

**Pass Criteria**: Package-level state is limited to singletons (like `rootCmd`) or configuration.

**Fail Criteria**: Mutable package-level state that makes testing difficult.

**Severity**: Medium

**Recommendation**: Pass dependencies explicitly through constructors or function parameters. Use package-level vars only for immutable constants or required singletons.
