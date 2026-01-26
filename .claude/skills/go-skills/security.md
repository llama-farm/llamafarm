# Security Patterns

Best practices for secure coding in the LlamaFarm CLI.

## Checklist

### 1. Never Log Credentials or Tokens

**Description**: Sensitive data (API keys, tokens, passwords) must never appear in logs.

**Search Pattern**:
```bash
grep -rn "LogDebug\|log\.Print\|fmt\.Print" cli/ --include="*.go" | grep -i "token\|password\|secret\|key\|auth"
```

**Pass Criteria**: Sensitive values are redacted before logging. Sanitization applied.

**Fail Criteria**: Credentials visible in debug logs. Tokens printed to stderr.

**Severity**: Critical

**Recommendation**:
```go
// Use sanitization for all logs
func LogDebug(msg string) {
    sanitized := sanitizeLogMessage(msg)
    debugLogger.Println(sanitized)
}

// Redact headers explicitly
func LogHeaders(kind string, hdr http.Header) {
    sensitiveHeaders := map[string]struct{}{
        "authorization": {},
        "cookie":        {},
        "x-api-key":     {},
    }
    for k, vals := range hdr {
        if _, sensitive := sensitiveHeaders[strings.ToLower(k)]; sensitive {
            LogDebug(fmt.Sprintf("%s header: %s: [REDACTED]", kind, k))
        } else {
            LogDebug(fmt.Sprintf("%s header: %s: %s", kind, k, vals))
        }
    }
}
```

---

### 2. Use Regex Sanitization for Logs

**Description**: Apply regex patterns to automatically redact sensitive data in logs.

**Search Pattern**:
```bash
grep -rn "regexp\|Regexp" cli/ --include="*.go"
```

**Pass Criteria**: Log sanitization catches JWTs, API keys, passwords, and other secrets.

**Fail Criteria**: Manual redaction only. Patterns miss common credential formats.

**Severity**: High

**Recommendation**:
```go
var sensitivePatterns = []struct {
    pattern     *regexp.Regexp
    replacement string
}{
    // JWT tokens
    {regexp.MustCompile(`\beyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+`), "[REDACTED-JWT]"},
    // API keys
    {regexp.MustCompile(`\b(sk|pk|sess)-[a-zA-Z0-9\-_]{20,}`), "[REDACTED-KEY]"},
    // Bearer tokens
    {regexp.MustCompile(`(?i)(bearer\s+)[a-zA-Z0-9\-_\.]+`), "${1}[REDACTED]"},
    // Passwords
    {regexp.MustCompile(`(?i)(password[=:\s]+['"]?)[^\s&'"]+`), "${1}[REDACTED]"},
}

func sanitizeLogMessage(msg string) string {
    sanitized := msg
    for _, sp := range sensitivePatterns {
        sanitized = sp.pattern.ReplaceAllString(sanitized, sp.replacement)
    }
    return sanitized
}
```

---

### 3. Validate All External Input

**Description**: Validate input from users, files, and network before use.

**Search Pattern**:
```bash
grep -rn "os\.Args\|flag\.\|cobra\.\|args\[" cli/ --include="*.go"
```

**Pass Criteria**: All user input validated. Length limits enforced. Invalid input rejected.

**Fail Criteria**: Unchecked input passed to system calls. No validation on file paths.

**Severity**: Critical

**Recommendation**:
```go
func processFile(path string) error {
    // Validate path
    if path == "" {
        return errors.New("path required")
    }

    // Clean path to prevent traversal
    cleanPath := filepath.Clean(path)

    // Check within allowed directory
    if !strings.HasPrefix(cleanPath, allowedDir) {
        return errors.New("path outside allowed directory")
    }

    // Check file exists and is regular file
    info, err := os.Stat(cleanPath)
    if err != nil {
        return fmt.Errorf("cannot access file: %w", err)
    }
    if !info.Mode().IsRegular() {
        return errors.New("not a regular file")
    }

    return nil
}
```

---

### 4. Use Context for Request Timeouts

**Description**: All HTTP requests must have timeouts to prevent resource exhaustion.

**Search Pattern**:
```bash
grep -rn "http\.NewRequest\|http\.Get\|http\.Post" cli/ --include="*.go"
```

**Pass Criteria**: All HTTP requests use context with timeout. No indefinite waits.

**Fail Criteria**: Requests without timeout. Default http.Client used.

**Severity**: High

**Recommendation**:
```go
func fetchData(url string) ([]byte, error) {
    ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
    defer cancel()

    req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
    if err != nil {
        return nil, err
    }

    // Also set client timeout as backup
    client := &http.Client{Timeout: 12 * time.Second}
    resp, err := client.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    return io.ReadAll(resp.Body)
}
```

---

### 5. Limit Response Body Size

**Description**: Limit the size of data read from external sources.

**Search Pattern**:
```bash
grep -rn "io\.ReadAll\|ioutil\.ReadAll" cli/ --include="*.go"
```

**Pass Criteria**: Response body reads have size limits. Large bodies handled as streams.

**Fail Criteria**: Unbounded reads that could exhaust memory.

**Severity**: Medium

**Recommendation**:
```go
const maxBodySize = 10 * 1024 * 1024  // 10MB

func readBody(resp *http.Response) ([]byte, error) {
    // Limit reader prevents memory exhaustion
    limitedReader := io.LimitReader(resp.Body, maxBodySize)
    data, err := io.ReadAll(limitedReader)
    if err != nil {
        return nil, err
    }

    if int64(len(data)) == maxBodySize {
        return nil, errors.New("response body too large")
    }

    return data, nil
}
```

---

### 6. Secure File Permissions

**Description**: Create files with restrictive permissions. Never world-writable.

**Search Pattern**:
```bash
grep -rn "os\.Create\|os\.WriteFile\|os\.OpenFile\|0777\|0666" cli/ --include="*.go"
```

**Pass Criteria**: Files created with 0644 or more restrictive. Directories with 0755.

**Fail Criteria**: World-writable files (0666, 0777). Sensitive data in world-readable files.

**Severity**: High

**Recommendation**:
```go
// Config files - owner read/write only
if err := os.WriteFile(path, data, 0600); err != nil {
    return err
}

// Log files - owner read/write, group/other read
if err := os.WriteFile(logPath, data, 0644); err != nil {
    return err
}

// Directories - owner full, group/other read/execute
if err := os.MkdirAll(dir, 0755); err != nil {
    return err
}

// PID files - restrictive
if err := os.WriteFile(pidPath, []byte(pid), 0644); err != nil {
    return err
}
```

---

### 7. Shell Escape Arguments

**Description**: When building shell commands, properly escape arguments.

**Search Pattern**:
```bash
grep -rn "exec\.Command\|fmt\.Sprintf.*shell\|strings\.Builder" cli/ --include="*.go"
```

**Pass Criteria**: Shell arguments are escaped. No command injection possible.

**Fail Criteria**: User input directly in shell commands. Unescaped special characters.

**Severity**: Critical

**Recommendation**:
```go
// Use exec.Command with separate arguments (safe)
cmd := exec.Command("git", "commit", "-m", message)

// When building shell strings for display, escape properly
func shellEscapeSingleQuotes(s string) string {
    return "'" + strings.ReplaceAll(s, "'", "'\"'\"'") + "'"
}

// For curl commands shown to users
func buildCurlCommand(url string, body []byte) string {
    var b strings.Builder
    b.WriteString("curl -X POST ")
    b.WriteString("-d ")
    b.WriteString(shellEscapeSingleQuotes(string(body)))
    b.WriteString(" ")
    b.WriteString(shellEscapeSingleQuotes(url))
    return b.String()
}
```

---

### 8. Avoid Path Traversal

**Description**: Validate file paths to prevent directory traversal attacks.

**Search Pattern**:
```bash
grep -rn "filepath\.Join\|filepath\.Clean\|os\.Open\|os\.ReadFile" cli/ --include="*.go"
```

**Pass Criteria**: All paths cleaned with `filepath.Clean`. Paths validated against base directory.

**Fail Criteria**: User-controlled paths used directly. `../` sequences possible.

**Severity**: Critical

**Recommendation**:
```go
func safeReadFile(baseDir, userPath string) ([]byte, error) {
    // Clean the user-provided path
    cleanPath := filepath.Clean(userPath)

    // Remove any leading path separators
    cleanPath = strings.TrimPrefix(cleanPath, string(filepath.Separator))

    // Join with base directory
    fullPath := filepath.Join(baseDir, cleanPath)

    // Verify result is still within base directory
    if !strings.HasPrefix(fullPath, filepath.Clean(baseDir)+string(filepath.Separator)) {
        return nil, errors.New("path escapes base directory")
    }

    return os.ReadFile(fullPath)
}
```

---

### 9. Handle Signals Gracefully

**Description**: Handle OS signals for graceful shutdown and cleanup.

**Search Pattern**:
```bash
grep -rn "os\.Signal\|signal\.Notify\|syscall\." cli/ --include="*.go"
```

**Pass Criteria**: SIGINT and SIGTERM handled. Resources cleaned up on shutdown.

**Fail Criteria**: Abrupt termination. Resources left in inconsistent state.

**Severity**: Medium

**Recommendation**:
```go
func setupSignalHandler(cleanup func()) {
    sigCh := make(chan os.Signal, 1)
    signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

    go func() {
        <-sigCh
        fmt.Println("\nShutting down gracefully...")
        cleanup()
        os.Exit(0)
    }()
}

// Usage
func main() {
    setupSignalHandler(func() {
        orchestrator.StopAllProcesses()
        utils.CloseDebugLogger()
    })

    cmd.Execute()
}
```

---

### 10. Redact Sensitive Data in Error Messages

**Description**: Error messages should not expose sensitive data.

**Search Pattern**:
```bash
grep -rn "fmt\.Errorf\|errors\.New" cli/ --include="*.go" | grep -i "token\|password\|key"
```

**Pass Criteria**: Error messages describe the problem without exposing secrets.

**Fail Criteria**: Passwords or tokens visible in error output.

**Severity**: High

**Recommendation**:
```go
// Bad - exposes token
return fmt.Errorf("authentication failed with token: %s", token)

// Good - describes problem without exposing secret
return fmt.Errorf("authentication failed: invalid or expired token")

// Good - includes request ID for debugging without secrets
return fmt.Errorf("API request failed (request_id=%s): %w", requestID, err)
```
