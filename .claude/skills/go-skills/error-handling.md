# Error Handling

Best practices for error creation, wrapping, and handling in the LlamaFarm CLI.

## Checklist

### 1. Always Check Errors Immediately

**Description**: Check error return values immediately after function calls.

**Search Pattern**:
```bash
grep -rn ", err :=\|, err =\|, _ :=" cli/ --include="*.go" | head -50
```

**Pass Criteria**: Every error is checked. No ignored error returns (except intentionally).

**Fail Criteria**: Error returns ignored with `_`. Errors checked multiple lines later.

**Severity**: Critical

**Recommendation**:
```go
// Good
f, err := os.Open(path)
if err != nil {
    return fmt.Errorf("failed to open file: %w", err)
}
defer f.Close()

// Bad - error ignored
f, _ := os.Open(path)  // NEVER do this
```

---

### 2. Wrap Errors with Context

**Description**: Wrap errors using `fmt.Errorf` with `%w` verb to add context.

**Search Pattern**:
```bash
grep -rn "fmt.Errorf.*%w" cli/ --include="*.go"
```

**Pass Criteria**: Errors are wrapped with context describing what operation failed.

**Fail Criteria**: Raw errors returned without context. Using `%v` instead of `%w`.

**Severity**: High

**Recommendation**:
```go
func (pm *ProcessManager) StartProcess(name string) error {
    if err := cmd.Start(); err != nil {
        return fmt.Errorf("failed to start process %s: %w", name, err)
    }
    return nil
}
```

---

### 3. Use Sentinel Errors for Expected Conditions

**Description**: Define package-level sentinel errors for expected error conditions.

**Search Pattern**:
```bash
grep -rn "var Err.*= errors.New\|var Err.*= fmt.Errorf" cli/ --include="*.go"
```

**Pass Criteria**: Common error conditions have named sentinel errors. Callers can use `errors.Is()`.

**Fail Criteria**: Stringly-typed error checking with string comparison.

**Severity**: Medium

**Recommendation**:
```go
// Define sentinel errors
var ErrServiceAlreadyRunning = errors.New("service is already running")
var ErrProcessNotFound = errors.New("process not found")

// Use in code
if isRunning {
    return ErrServiceAlreadyRunning
}

// Check in caller
if errors.Is(err, ErrServiceAlreadyRunning) {
    // Handle expected case
}
```

---

### 4. Use errors.Is and errors.As for Checking

**Description**: Use `errors.Is()` and `errors.As()` instead of type assertions.

**Search Pattern**:
```bash
grep -rn "errors\.Is\|errors\.As" cli/ --include="*.go"
```

**Pass Criteria**: Error checking uses `errors.Is()` and `errors.As()` for wrapped errors.

**Fail Criteria**: Direct type assertions or string matching on error messages.

**Severity**: Medium

**Recommendation**:
```go
// Check for specific error
if errors.Is(err, os.ErrNotExist) {
    // File doesn't exist
}

// Extract typed error
var healthErr *HealthError
if errors.As(err, &healthErr) {
    fmt.Printf("Server unhealthy: %s\n", healthErr.Status)
}
```

---

### 5. Create Custom Error Types When Needed

**Description**: Define custom error types for errors that carry additional context.

**Search Pattern**:
```bash
grep -rn "func.*Error().*string" cli/ --include="*.go"
```

**Pass Criteria**: Custom error types implement `error` interface. Carry relevant context.

**Fail Criteria**: Overuse of custom types. Context that could be in wrap message.

**Severity**: Low

**Recommendation**:
```go
type HealthError struct {
    Status     string
    HealthResp HealthPayload
}

func (e *HealthError) Error() string {
    return fmt.Sprintf("server unhealthy: %s", e.Status)
}

// Usage
return &HealthError{Status: "degraded", HealthResp: payload}
```

---

### 6. Don't Log and Return

**Description**: Either log an error OR return it, not both.

**Search Pattern**:
```bash
grep -rn "log\|Log" -A2 cli/ --include="*.go" | grep "return.*err"
```

**Pass Criteria**: Errors are logged at the top level only. Lower levels just return.

**Fail Criteria**: Same error logged multiple times as it propagates up.

**Severity**: Medium

**Recommendation**:
```go
// In library code - just return
func loadConfig() (*Config, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return nil, fmt.Errorf("read config: %w", err)
    }
    return parseConfig(data)
}

// At top level - log and handle
func Execute() {
    if err := rootCmd.Execute(); err != nil {
        fmt.Fprintf(os.Stderr, "Error: %v\n", err)
        os.Exit(1)
    }
}
```

---

### 7. Handle Errors Close to the Source

**Description**: Handle errors as close to where they occur as practical.

**Search Pattern**:
```bash
grep -rn "if err != nil" cli/ --include="*.go" | wc -l
```

**Pass Criteria**: Error handling happens immediately after the call. No distant error checks.

**Fail Criteria**: Errors stored and checked later. Complex error handling logic.

**Severity**: Medium

**Recommendation**:
```go
// Good - handle immediately
resp, err := client.Do(req)
if err != nil {
    return nil, fmt.Errorf("request failed: %w", err)
}
defer resp.Body.Close()

// Bad - storing for later
var savedErr error
resp, savedErr = client.Do(req)
// ... other code ...
if savedErr != nil {  // Easy to forget
    return nil, savedErr
}
```

---

### 8. Use Early Returns

**Description**: Use early returns to handle errors and reduce nesting.

**Search Pattern**:
```bash
grep -rn "if err != nil {" -A3 cli/ --include="*.go" | grep "return"
```

**Pass Criteria**: Functions use early returns. Happy path is not deeply nested.

**Fail Criteria**: Deep nesting with else blocks. Complex control flow.

**Severity**: Medium

**Recommendation**:
```go
// Good - early returns
func process(name string) error {
    if name == "" {
        return errors.New("name required")
    }

    data, err := load(name)
    if err != nil {
        return fmt.Errorf("load %s: %w", name, err)
    }

    return save(data)  // Happy path at end
}

// Bad - deep nesting
func process(name string) error {
    if name != "" {
        data, err := load(name)
        if err == nil {
            // Happy path deeply nested
        }
    }
}
```

---

### 9. Provide Actionable Error Messages

**Description**: Error messages should help users understand what to do.

**Search Pattern**:
```bash
grep -rn "fmt.Errorf\|errors.New" cli/ --include="*.go"
```

**Pass Criteria**: Error messages describe the problem and suggest resolution.

**Fail Criteria**: Cryptic error messages. Technical jargon without context.

**Severity**: Medium

**Recommendation**:
```go
// Good - actionable
return fmt.Errorf("service %s failed to start. Run 'lf services logs -s %s' to view logs", name, name)

// Bad - not actionable
return errors.New("start failed")
```

---

### 10. Use Panic Only for Programming Errors

**Description**: Reserve `panic` for unrecoverable programming errors, not runtime errors.

**Search Pattern**:
```bash
grep -rn "panic(" cli/ --include="*.go"
```

**Pass Criteria**: Panic used only for invariant violations, nil pointer protection, or init failures.

**Fail Criteria**: Panic used for expected runtime errors like file not found.

**Severity**: High

**Recommendation**:
```go
// Acceptable - programming error
func MustParse(s string) *Config {
    cfg, err := Parse(s)
    if err != nil {
        panic(fmt.Sprintf("invalid config: %v", err))
    }
    return cfg
}

// Bad - runtime error
func ReadFile(path string) []byte {
    data, err := os.ReadFile(path)
    if err != nil {
        panic(err)  // NEVER do this
    }
    return data
}
```
