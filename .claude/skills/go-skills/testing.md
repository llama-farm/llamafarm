# Testing Patterns

Best practices for writing tests in the LlamaFarm CLI.

## Checklist

### 1. Use Table-Driven Tests

**Description**: Structure tests as tables of inputs and expected outputs.

**Search Pattern**:
```bash
grep -rn "tests := \[\]struct\|tt := range tests" cli/ --include="*_test.go"
```

**Pass Criteria**: Complex test scenarios use table-driven pattern. Easy to add cases.

**Fail Criteria**: Repetitive test code. Hard to add new test cases.

**Severity**: Medium

**Recommendation**:
```go
func TestResolveDependencies(t *testing.T) {
    tests := []struct {
        name        string
        serviceName string
        wantOrder   []string
        wantErr     bool
        errContains string
    }{
        {
            name:        "resolve server with no dependencies",
            serviceName: "server",
            wantOrder:   []string{"server"},
            wantErr:     false,
        },
        {
            name:        "unknown service returns error",
            serviceName: "unknown-service",
            wantErr:     true,
            errContains: "unknown service",
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            got, err := sm.resolveDependencies(tt.serviceName)
            // assertions...
        })
    }
}
```

---

### 2. Use t.Run for Subtests

**Description**: Use `t.Run()` to create named subtests for better output and isolation.

**Search Pattern**:
```bash
grep -rn "t\.Run(" cli/ --include="*_test.go"
```

**Pass Criteria**: All table-driven tests use `t.Run()`. Subtests have descriptive names.

**Fail Criteria**: Tests without subtests. Unclear which case failed.

**Severity**: Medium

**Recommendation**:
```go
for _, tt := range tests {
    t.Run(tt.name, func(t *testing.T) {
        // Each subtest runs in isolation
        // Can use t.Parallel() for concurrent execution
    })
}
```

---

### 3. Use Interfaces for Mocking

**Description**: Define interfaces for external dependencies to enable mocking.

**Search Pattern**:
```bash
grep -rn "type.*interface {" cli/ --include="*.go"
```

**Pass Criteria**: External dependencies (HTTP, filesystem) accessed through interfaces.

**Fail Criteria**: Direct use of concrete types. Cannot test without real dependencies.

**Severity**: High

**Recommendation**:
```go
// Define interface
type HTTPClient interface {
    Do(req *http.Request) (*http.Response, error)
}

// Use in production
var httpClient HTTPClient = &DefaultHTTPClient{Timeout: 60 * time.Second}

// Swap for testing
func SetHTTPClientForTest(client HTTPClient) {
    httpClient = client
}

// Mock in tests
type mockHTTPClient struct {
    response *http.Response
    err      error
}

func (m *mockHTTPClient) Do(req *http.Request) (*http.Response, error) {
    return m.response, m.err
}
```

---

### 4. Test Both Success and Error Cases

**Description**: Every test should verify both happy path and error conditions.

**Search Pattern**:
```bash
grep -rn "wantErr" cli/ --include="*_test.go"
```

**Pass Criteria**: Tests include error cases. Both `wantErr: true` and `wantErr: false` present.

**Fail Criteria**: Only happy path tested. Error handling untested.

**Severity**: High

**Recommendation**:
```go
tests := []struct {
    name    string
    input   string
    want    *Result
    wantErr bool
}{
    {"valid input", "good", &Result{...}, false},
    {"empty input", "", nil, true},
    {"invalid format", "bad", nil, true},
}
```

---

### 5. Use t.TempDir for Temporary Files

**Description**: Use `t.TempDir()` for tests that need temporary directories.

**Search Pattern**:
```bash
grep -rn "t\.TempDir\|os\.MkdirTemp" cli/ --include="*_test.go"
```

**Pass Criteria**: Tests use `t.TempDir()` for automatic cleanup. No temp file leaks.

**Fail Criteria**: Manual temp directory creation. Cleanup in defer forgotten.

**Severity**: Medium

**Recommendation**:
```go
func TestSessionPersistence(t *testing.T) {
    // t.TempDir() automatically cleans up
    tempDir := t.TempDir()

    // Or for Go 1.14 compatibility
    tempDir, err := os.MkdirTemp("", "test")
    if err != nil {
        t.Fatalf("failed to create temp dir: %v", err)
    }
    defer os.RemoveAll(tempDir)
}
```

---

### 6. Use t.Cleanup for Resource Cleanup

**Description**: Use `t.Cleanup()` for cleanup that must run after test completion.

**Search Pattern**:
```bash
grep -rn "t\.Cleanup\|defer.*t\." cli/ --include="*_test.go"
```

**Pass Criteria**: Resources registered with `t.Cleanup()`. Cleanup runs even on failure.

**Fail Criteria**: Cleanup in defer that doesn't run on fatal. Resource leaks.

**Severity**: Medium

**Recommendation**:
```go
func TestWithEnvVar(t *testing.T) {
    orig := os.Getenv("MY_VAR")
    os.Setenv("MY_VAR", "test-value")

    t.Cleanup(func() {
        if orig != "" {
            os.Setenv("MY_VAR", orig)
        } else {
            os.Unsetenv("MY_VAR")
        }
    })

    // Test code...
}
```

---

### 7. Use t.Helper for Test Helpers

**Description**: Mark test helper functions with `t.Helper()` for better error reporting.

**Search Pattern**:
```bash
grep -rn "t\.Helper()" cli/ --include="*_test.go"
```

**Pass Criteria**: Helper functions call `t.Helper()`. Errors point to actual test line.

**Fail Criteria**: Errors point to helper function instead of failing test.

**Severity**: Low

**Recommendation**:
```go
func assertNoError(t *testing.T, err error) {
    t.Helper()  // Error will point to caller, not this line
    if err != nil {
        t.Fatalf("unexpected error: %v", err)
    }
}

func assertEqual(t *testing.T, got, want interface{}) {
    t.Helper()
    if got != want {
        t.Errorf("got %v, want %v", got, want)
    }
}
```

---

### 8. Test File Naming Convention

**Description**: Test files should be named `*_test.go` in the same package.

**Search Pattern**:
```bash
find cli/ -name "*_test.go" -type f
```

**Pass Criteria**: Test files follow `xxx_test.go` pattern. Located with source files.

**Fail Criteria**: Tests in separate directory. Non-standard naming.

**Severity**: Low

**Recommendation**:
```
cli/cmd/
  chat_client.go
  chat_client_test.go    # Same package, same directory
  orchestrator/
    services.go
    services_test.go     # Same package, same directory
```

---

### 9. Use Parallel Tests When Safe

**Description**: Use `t.Parallel()` for tests that don't share state.

**Search Pattern**:
```bash
grep -rn "t\.Parallel()" cli/ --include="*_test.go"
```

**Pass Criteria**: Independent tests run in parallel. Test suite completes faster.

**Fail Criteria**: Tests with shared state run in parallel causing flakes.

**Severity**: Low

**Recommendation**:
```go
func TestSomething(t *testing.T) {
    tests := []struct{...}

    for _, tt := range tests {
        tt := tt  // Capture range variable
        t.Run(tt.name, func(t *testing.T) {
            t.Parallel()  // Safe if no shared state
            // Test code...
        })
    }
}
```

---

### 10. Verify Error Messages

**Description**: When testing error cases, verify the error message content.

**Search Pattern**:
```bash
grep -rn "errContains\|strings.Contains.*err" cli/ --include="*_test.go"
```

**Pass Criteria**: Error tests verify error message contains expected text.

**Fail Criteria**: Only checking `err != nil`. Wrong error type could pass.

**Severity**: Medium

**Recommendation**:
```go
func TestErrorMessages(t *testing.T) {
    tests := []struct {
        name        string
        input       string
        errContains string
    }{
        {"missing file", "nonexistent", "no such file"},
        {"invalid format", "bad.txt", "invalid format"},
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            _, err := process(tt.input)
            if err == nil {
                t.Fatal("expected error")
            }
            if !strings.Contains(err.Error(), tt.errContains) {
                t.Errorf("error %q should contain %q", err, tt.errContains)
            }
        })
    }
}
```
