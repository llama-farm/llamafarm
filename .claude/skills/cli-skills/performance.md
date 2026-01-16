# CLI Performance Optimizations

Best practices for building responsive and efficient CLI applications in LlamaFarm.

## Checklist

### 1. Lazy Load Heavy Dependencies

**Description**: Defer loading of expensive resources until they are actually needed.

**Search Pattern**:
```bash
grep -rn "func init()" cli/ --include="*.go" -A10
```

**Pass Criteria**: `init()` functions only register commands and flags, not load data.

**Fail Criteria**: Heavy operations (HTTP calls, file I/O) in `init()` slow down all commands.

**Severity**: High

**Recommendation**:
```go
// Good - lazy loading
var configCache *Config

func getConfig() (*Config, error) {
    if configCache != nil {
        return configCache, nil
    }
    cfg, err := loadConfig()
    if err != nil {
        return nil, err
    }
    configCache = cfg
    return configCache, nil
}

// Avoid - eager loading in init()
func init() {
    config, _ = loadConfig()  // Slows down all commands
}
```

---

### 2. Use Connection Pooling for HTTP

**Description**: Reuse HTTP clients and connections across requests.

**Search Pattern**:
```bash
grep -rn "http.Client" cli/ --include="*.go"
grep -rn "GetHTTPClient" cli/ --include="*.go"
```

**Pass Criteria**: Single HTTP client instance reused across requests.

**Fail Criteria**: Creating new `http.Client` for each request.

**Severity**: Medium

**Recommendation**:
```go
// utils/httpclient.go
var httpClient *http.Client
var httpOnce sync.Once

func GetHTTPClient() *http.Client {
    httpOnce.Do(func() {
        httpClient = &http.Client{
            Timeout: 30 * time.Second,
            Transport: &http.Transport{
                MaxIdleConns:        100,
                MaxIdleConnsPerHost: 10,
                IdleConnTimeout:     90 * time.Second,
            },
        }
    })
    return httpClient
}

// Usage
resp, err := utils.GetHTTPClient().Do(req)
```

---

### 3. Stream Large Responses

**Description**: Stream large API responses instead of buffering entire response in memory.

**Search Pattern**:
```bash
grep -rn "io.ReadAll" cli/ --include="*.go"
```

**Pass Criteria**: Large responses (chat completions, file downloads) use streaming.

**Fail Criteria**: Reading entire response into memory before processing.

**Severity**: High

**Recommendation**:
```go
// Good - streaming
func streamResponse(resp *http.Response, callback func(chunk string)) error {
    reader := bufio.NewReader(resp.Body)
    for {
        line, err := reader.ReadString('\n')
        if err == io.EOF {
            break
        }
        if err != nil {
            return err
        }
        callback(line)
    }
    return nil
}

// Avoid - buffering
func readResponse(resp *http.Response) (string, error) {
    body, err := io.ReadAll(resp.Body)  // May use excessive memory
    return string(body), err
}
```

---

### 4. Throttle Progress Updates

**Description**: Limit frequency of progress message updates to prevent UI flickering.

**Search Pattern**:
```bash
grep -rn "OutputProgress" cli/ --include="*.go"
```

**Pass Criteria**: Progress updates throttled to reasonable interval (100-500ms).

**Fail Criteria**: Progress updates on every byte, causing performance issues.

**Severity**: Medium

**Recommendation**:
```go
// utils/output.go - throttled progress
func sendConsolidatedProgressMessage(format string, args ...interface{}) {
    content := fmt.Sprintf(format, args...)

    outputManager.mu.Lock()
    defer outputManager.mu.Unlock()

    outputManager.lastProgressMessage = content

    if !outputManager.progressMessageSent {
        // Send immediately
        sendToTUI(content)
        outputManager.progressMessageSent = true

        // Reset flag after delay
        go func() {
            time.Sleep(100 * time.Millisecond)
            outputManager.mu.Lock()
            outputManager.progressMessageSent = false
            outputManager.mu.Unlock()
        }()
    }
}
```

---

### 5. Cache Expensive Computations

**Description**: Cache results of expensive operations that don't change frequently.

**Search Pattern**:
```bash
grep -rn "Cache\|cache" cli/ --include="*.go"
```

**Pass Criteria**: Model lists, configurations cached with appropriate TTL.

**Fail Criteria**: Repeated API calls for same data in short time period.

**Severity**: Medium

**Recommendation**:
```go
type cachedResult struct {
    data      interface{}
    timestamp time.Time
}

var cache = struct {
    sync.RWMutex
    items map[string]cachedResult
}{items: make(map[string]cachedResult)}

func getCached(key string, ttl time.Duration, fetch func() (interface{}, error)) (interface{}, error) {
    cache.RLock()
    if item, ok := cache.items[key]; ok && time.Since(item.timestamp) < ttl {
        cache.RUnlock()
        return item.data, nil
    }
    cache.RUnlock()

    data, err := fetch()
    if err != nil {
        return nil, err
    }

    cache.Lock()
    cache.items[key] = cachedResult{data: data, timestamp: time.Now()}
    cache.Unlock()

    return data, nil
}
```

---

### 6. Minimize Render Calls

**Description**: Only re-render TUI when state actually changes.

**Search Pattern**:
```bash
grep -rn "SetContent\|GotoBottom" cli/ --include="*.go"
```

**Pass Criteria**: Viewport content updated only when messages change.

**Fail Criteria**: Re-rendering on every Update call regardless of state change.

**Severity**: Medium

**Recommendation**:
```go
// Use content hashing to detect changes
func computeTranscriptKey(m chatModel) string {
    if len(m.messages) == 0 {
        return "empty"
    }
    msg := m.messages[len(m.messages)-1]
    h := fnv.New64a()
    io.WriteString(h, msg.Role)
    io.WriteString(h, msg.Content)
    return fmt.Sprintf("%x", h.Sum64())
}

func computeTranscript(m chatModel) string {
    key := computeTranscriptKey(m)
    if lastTranscriptKey == key {
        return m.transcript  // Return cached
    }
    // Recompute only if changed
    lastTranscriptKey = key
    return renderMessages(m.messages)
}
```

---

### 7. Use Goroutines for Parallel Operations

**Description**: Run independent operations concurrently to reduce total wait time.

**Search Pattern**:
```bash
grep -rn "go func" cli/ --include="*.go"
```

**Pass Criteria**: Independent API calls run in parallel with proper synchronization.

**Fail Criteria**: Sequential operations that could be parallelized.

**Severity**: Medium

**Recommendation**:
```go
func fetchProjectData(ns, proj string) (*ProjectData, error) {
    var wg sync.WaitGroup
    var models []ModelInfo
    var databases *DatabasesResponse
    var modelsErr, dbErr error

    wg.Add(2)

    go func() {
        defer wg.Done()
        models, modelsErr = fetchModels(ns, proj)
    }()

    go func() {
        defer wg.Done()
        databases, dbErr = fetchDatabases(ns, proj)
    }()

    wg.Wait()

    if modelsErr != nil {
        return nil, modelsErr
    }
    if dbErr != nil {
        return nil, dbErr
    }

    return &ProjectData{Models: models, Databases: databases}, nil
}
```

---

### 8. Efficient String Building

**Description**: Use `strings.Builder` for concatenating multiple strings.

**Search Pattern**:
```bash
grep -rn "strings.Builder" cli/ --include="*.go"
```

**Pass Criteria**: String building uses `strings.Builder` or `bytes.Buffer`.

**Fail Criteria**: String concatenation with `+` in loops.

**Severity**: Low

**Recommendation**:
```go
// Good
func renderMessages(messages []Message) string {
    var b strings.Builder
    for _, msg := range messages {
        b.WriteString(formatMessage(msg))
        b.WriteString("\n")
    }
    return b.String()
}

// Avoid
func renderMessages(messages []Message) string {
    result := ""
    for _, msg := range messages {
        result += formatMessage(msg) + "\n"  // Creates new string each iteration
    }
    return result
}
```

---

### 9. Protect Against Negative Dimensions

**Description**: Guard against negative viewport dimensions that cause panics.

**Search Pattern**:
```bash
grep -rn "viewport.Height\|viewport.Width" cli/ --include="*.go"
```

**Pass Criteria**: Dimension calculations check for and prevent negative values.

**Fail Criteria**: Negative dimensions passed to viewport causing slice bounds panic.

**Severity**: High

**Recommendation**:
```go
func (m myModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
    switch msg := msg.(type) {
    case tea.WindowSizeMsg:
        headerHeight := lipgloss.Height(m.renderHeader())
        footerHeight := lipgloss.Height(m.renderFooter())

        // CRITICAL: Prevent negative height
        newHeight := msg.Height - headerHeight - footerHeight
        if newHeight < 1 {
            newHeight = 1
        }

        m.viewport.Height = newHeight
        m.viewport.Width = msg.Width

        // Also protect textarea width
        newWidth := msg.Width - 2
        if newWidth < 10 {
            newWidth = 10
        }
        m.textarea.SetWidth(newWidth)
    }
    return m, nil
}
```

---

### 10. Use Context Timeouts

**Description**: Set appropriate timeouts for all network operations.

**Search Pattern**:
```bash
grep -rn "context.WithTimeout" cli/ --include="*.go"
```

**Pass Criteria**: All HTTP requests and long operations have timeouts.

**Fail Criteria**: Operations can hang indefinitely.

**Severity**: High

**Recommendation**:
```go
func fetchWithTimeout(url string) ([]byte, error) {
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()

    req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
    if err != nil {
        return nil, err
    }

    resp, err := utils.GetHTTPClient().Do(req)
    if err != nil {
        if errors.Is(err, context.DeadlineExceeded) {
            return nil, fmt.Errorf("request timed out after 30s")
        }
        return nil, err
    }
    defer resp.Body.Close()

    return io.ReadAll(resp.Body)
}
```

---

### 11. Avoid Blocking the Main Loop

**Description**: Never perform blocking operations in Update() - use commands instead.

**Search Pattern**:
```bash
grep -rn "func (m.*) Update" cli/ --include="*.go" -A30 | grep -E "http\.|os\.|io\."
```

**Pass Criteria**: All I/O operations wrapped in tea.Cmd functions.

**Fail Criteria**: Direct HTTP calls or file operations in Update().

**Severity**: High

**Recommendation**:
```go
// Good - async command
func (m myModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
    switch msg := msg.(type) {
    case tea.KeyMsg:
        if msg.String() == "enter" {
            return m, fetchDataCmd(m.input)  // Returns immediately
        }
    case dataMsg:
        m.data = msg.data  // Handle result
    }
    return m, nil
}

func fetchDataCmd(input string) tea.Cmd {
    return func() tea.Msg {
        data, err := fetchFromAPI(input)  // Runs in goroutine
        if err != nil {
            return errorMsg{err: err}
        }
        return dataMsg{data: data}
    }
}

// Avoid - blocking in Update
func (m myModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
    switch msg := msg.(type) {
    case tea.KeyMsg:
        if msg.String() == "enter" {
            data, _ := fetchFromAPI(m.input)  // BLOCKS UI
            m.data = data
        }
    }
    return m, nil
}
```

---

### 12. Profile Before Optimizing

**Description**: Use Go's profiling tools to identify actual bottlenecks.

**Search Pattern**:
```bash
grep -rn "pprof\|runtime/pprof" cli/ --include="*.go"
```

**Pass Criteria**: Performance-critical code paths are profiled and optimized based on data.

**Fail Criteria**: Premature optimization without profiling.

**Severity**: Low

**Recommendation**:
```go
// Add profiling support for development
import _ "net/http/pprof"

func main() {
    if os.Getenv("ENABLE_PPROF") == "1" {
        go func() {
            log.Println(http.ListenAndServe("localhost:6060", nil))
        }()
    }
    // ...
}

// Profile with:
// ENABLE_PPROF=1 lf chat
// go tool pprof http://localhost:6060/debug/pprof/profile?seconds=30
```
