# Concurrency Patterns

Best practices for goroutines, channels, and synchronization in the LlamaFarm CLI.

## Checklist

### 1. Protect Shared State with Mutex

**Description**: All shared mutable state must be protected by a mutex.

**Search Pattern**:
```bash
grep -rn "sync\.Mutex\|sync\.RWMutex" cli/ --include="*.go"
```

**Pass Criteria**: Every struct with shared mutable state includes a mutex. Access is synchronized.

**Fail Criteria**: Shared state accessed without synchronization. Race conditions possible.

**Severity**: Critical

**Recommendation**:
```go
type ProcessManager struct {
    mu        sync.RWMutex          // Protects processes map
    processes map[string]*ProcessInfo
}

func (pm *ProcessManager) GetProcess(name string) (*ProcessInfo, bool) {
    pm.mu.RLock()
    defer pm.mu.RUnlock()
    proc, ok := pm.processes[name]
    return proc, ok
}
```

---

### 2. Use RWMutex When Reads Dominate

**Description**: Use `sync.RWMutex` when read operations significantly outnumber writes.

**Search Pattern**:
```bash
grep -rn "RLock\|RUnlock" cli/ --include="*.go"
```

**Pass Criteria**: Read-heavy operations use `RLock`/`RUnlock`. Write operations use `Lock`/`Unlock`.

**Fail Criteria**: Using `sync.Mutex` for read-heavy workloads, causing unnecessary contention.

**Severity**: Medium

**Recommendation**:
```go
func (pm *ProcessManager) GetProcessStatus(name string) (string, error) {
    pm.mu.RLock()  // Multiple readers can proceed
    defer pm.mu.RUnlock()

    proc, found := pm.findProcess(name)
    if !found {
        return "", fmt.Errorf("process %s not found", name)
    }
    return proc.Status, nil
}
```

---

### 3. Always Defer Mutex Unlock

**Description**: Use `defer` for mutex unlock to ensure release on all code paths.

**Search Pattern**:
```bash
grep -rn "\.Lock()" -A2 cli/ --include="*.go" | grep -v "defer"
```

**Pass Criteria**: Every `Lock()` is immediately followed by `defer Unlock()`.

**Fail Criteria**: Manual unlock at multiple return points, risk of deadlock on panic.

**Severity**: Critical

**Recommendation**:
```go
func (pm *ProcessManager) StopAllProcesses() {
    pm.mu.RLock()
    names := make([]string, 0, len(pm.processes))
    for name := range pm.processes {
        names = append(names, name)
    }
    pm.mu.RUnlock()  // Release before calling StopProcess

    for _, name := range names {
        pm.StopProcess(name)
    }
}
```

---

### 4. Use Channels for Goroutine Communication

**Description**: Prefer channels over shared memory for goroutine coordination.

**Search Pattern**:
```bash
grep -rn "make(chan" cli/ --include="*.go"
```

**Pass Criteria**: Goroutines communicate via typed channels. Channel ownership is clear.

**Fail Criteria**: Goroutines share state through global variables without proper sync.

**Severity**: High

**Recommendation**:
```go
func (m *chatModel) startStream() tea.Cmd {
    ch := make(chan tea.Msg, 32)  // Buffered for async
    m.streamCh = ch

    go func() {
        defer close(ch)  // Always close when done
        // Send messages through channel
        ch <- responseMsg{content: data}
    }()

    return listen(ch)
}
```

---

### 5. Buffer Channels Appropriately

**Description**: Choose buffer size based on producer/consumer patterns.

**Search Pattern**:
```bash
grep -rn "make(chan.*," cli/ --include="*.go"
```

**Pass Criteria**: Buffered channels used when producer shouldn't block. Unbuffered for synchronization.

**Fail Criteria**: Unbuffered channels causing deadlock. Oversized buffers wasting memory.

**Severity**: Medium

**Recommendation**:
```go
// Buffered: producer shouldn't block on slow consumer
ch := make(chan tea.Msg, 32)

// Unbuffered: synchronization point needed
done := make(chan struct{})
```

---

### 6. Close Channels from Producer Side

**Description**: Only the channel producer should close the channel.

**Search Pattern**:
```bash
grep -rn "close(" cli/ --include="*.go"
```

**Pass Criteria**: Channels are closed by the goroutine that sends to them. Receivers never close.

**Fail Criteria**: Receivers closing channels, causing panic on send.

**Severity**: Critical

**Recommendation**:
```go
go func() {
    defer close(ch)  // Producer closes
    for _, item := range items {
        ch <- item
    }
}()

// Receiver just reads
for msg := range ch {
    process(msg)
}
```

---

### 7. Use Context for Cancellation

**Description**: Use `context.Context` for timeout and cancellation propagation.

**Search Pattern**:
```bash
grep -rn "context\." cli/ --include="*.go"
```

**Pass Criteria**: Long-running operations accept context. Cancellation is respected.

**Fail Criteria**: Operations cannot be cancelled. Context ignored or not passed.

**Severity**: High

**Recommendation**:
```go
func fetchSessionHistory(ctx context.Context, url string) (*History, error) {
    ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
    defer cancel()

    req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
    if err != nil {
        return nil, err
    }
    // ...
}
```

---

### 8. Use sync.Once for One-Time Initialization

**Description**: Use `sync.Once` for thread-safe lazy initialization.

**Search Pattern**:
```bash
grep -rn "sync\.Once" cli/ --include="*.go"
```

**Pass Criteria**: Singleton initialization uses `sync.Once`. No race conditions.

**Fail Criteria**: Double-checked locking or other error-prone patterns.

**Severity**: Medium

**Recommendation**:
```go
var (
    debugOnce   sync.Once
    debugLogger *log.Logger
)

func InitDebugLogger(path string) error {
    var initErr error
    debugOnce.Do(func() {
        // Initialization runs exactly once
        f, err := os.Create(path)
        if err != nil {
            initErr = err
            return
        }
        debugLogger = log.New(f, "", log.LstdFlags)
    })
    return initErr
}
```

---

### 9. Avoid Goroutine Leaks

**Description**: Every goroutine must have a clear exit condition.

**Search Pattern**:
```bash
grep -rn "go func" cli/ --include="*.go"
```

**Pass Criteria**: Goroutines have exit conditions (channel close, context cancel, timeout).

**Fail Criteria**: Goroutines block forever on channel reads or have no exit path.

**Severity**: High

**Recommendation**:
```go
go func() {
    for {
        select {
        case msg, ok := <-ch:
            if !ok {
                return  // Channel closed, exit
            }
            process(msg)
        case <-ctx.Done():
            return  // Context cancelled, exit
        }
    }
}()
```

---

### 10. Use WaitGroup for Goroutine Coordination

**Description**: Use `sync.WaitGroup` to wait for multiple goroutines to complete.

**Search Pattern**:
```bash
grep -rn "sync\.WaitGroup" cli/ --include="*.go"
```

**Pass Criteria**: Parallel operations use WaitGroup. All goroutines are waited on.

**Fail Criteria**: Main goroutine exits before workers complete. Race conditions.

**Severity**: Medium

**Recommendation**:
```go
func processAll(items []Item) {
    var wg sync.WaitGroup

    for _, item := range items {
        wg.Add(1)
        go func(it Item) {
            defer wg.Done()
            process(it)
        }(item)  // Pass item to avoid closure capture
    }

    wg.Wait()  // Block until all complete
}
```
