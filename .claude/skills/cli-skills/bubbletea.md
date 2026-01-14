# Bubbletea TUI Patterns

Best practices for building Terminal User Interfaces with Bubbletea and Lipgloss in LlamaFarm.

## Checklist

### 1. Implement the Model Interface Correctly

**Description**: All Bubbletea models must implement `Init()`, `Update()`, and `View()` methods.

**Search Pattern**:
```bash
grep -rn "func (m.*) Init()" cli/ --include="*.go"
grep -rn "func (m.*) Update(msg tea.Msg)" cli/ --include="*.go"
grep -rn "func (m.*) View() string" cli/ --include="*.go"
```

**Pass Criteria**: All three methods implemented with correct signatures.

**Fail Criteria**: Missing methods or incorrect signatures that prevent compilation.

**Severity**: High

**Recommendation**:
```go
type myModel struct {
    // State fields
    width  int
    height int
    err    error
}

func (m myModel) Init() tea.Cmd {
    // Return initial commands (can be nil)
    return tea.Batch(doAsyncWork(), tea.EnterAltScreen)
}

func (m myModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
    // Handle messages, return updated model and commands
    switch msg := msg.(type) {
    case tea.WindowSizeMsg:
        m.width = msg.Width
        m.height = msg.Height
    }
    return m, nil
}

func (m myModel) View() string {
    // Return rendered string (never modify state here)
    return "Hello, World!"
}
```

---

### 2. Handle Window Size Messages

**Description**: Always handle `tea.WindowSizeMsg` to support terminal resizing.

**Search Pattern**:
```bash
grep -rn "tea.WindowSizeMsg" cli/ --include="*.go"
```

**Pass Criteria**: All TUI models handle window size changes and adjust layouts accordingly.

**Fail Criteria**: Fixed-size layouts that break on terminal resize.

**Severity**: High

**Recommendation**:
```go
func (m myModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
    switch msg := msg.(type) {
    case tea.WindowSizeMsg:
        m.width = msg.Width
        m.height = msg.Height

        // Update child components
        m.viewport.Width = msg.Width
        m.viewport.Height = msg.Height - footerHeight - headerHeight

        // Protect against negative dimensions
        if m.viewport.Height < 1 {
            m.viewport.Height = 1
        }

        m.textarea.SetWidth(msg.Width - 2)
    }
    return m, nil
}
```

---

### 3. Use Message Types for State Changes

**Description**: Define custom message types for async operations and state updates.

**Search Pattern**:
```bash
grep -rn "type.*Msg struct" cli/ --include="*.go"
```

**Pass Criteria**: Custom message types for each distinct state change or async result.

**Fail Criteria**: Using raw values or string messages for state changes.

**Severity**: Medium

**Recommendation**:
```go
// Define message types
type responseMsg struct{ content string }
type errorMsg struct{ err error }
type streamDone struct{}
type serverHealthMsg struct{ health *HealthPayload }

// Use in Update
func (m myModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
    switch msg := msg.(type) {
    case responseMsg:
        m.content = msg.content
    case errorMsg:
        m.err = msg.err
    case streamDone:
        m.loading = false
    }
    return m, nil
}

// Create commands that return messages
func fetchDataCmd() tea.Cmd {
    return func() tea.Msg {
        data, err := fetchData()
        if err != nil {
            return errorMsg{err: err}
        }
        return responseMsg{content: data}
    }
}
```

---

### 4. Use tea.Batch for Multiple Commands

**Description**: Combine multiple commands with `tea.Batch` when returning from Update.

**Search Pattern**:
```bash
grep -rn "tea.Batch" cli/ --include="*.go"
```

**Pass Criteria**: Multiple concurrent commands combined with `tea.Batch`.

**Fail Criteria**: Returning only one command when multiple are needed.

**Severity**: Medium

**Recommendation**:
```go
func (m myModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
    var cmds []tea.Cmd

    // Update child components
    var vpCmd tea.Cmd
    m.viewport, vpCmd = m.viewport.Update(msg)
    cmds = append(cmds, vpCmd)

    var taCmd tea.Cmd
    m.textarea, taCmd = m.textarea.Update(msg)
    cmds = append(cmds, taCmd)

    switch msg := msg.(type) {
    case tea.KeyMsg:
        if msg.String() == "enter" {
            cmds = append(cmds, sendMessageCmd(m.textarea.Value()))
            cmds = append(cmds, thinkingAnimationCmd())
        }
    }

    return m, tea.Batch(cmds...)
}
```

---

### 5. Keep View() Pure and Side-Effect Free

**Description**: The `View()` method should only render state, never modify it.

**Search Pattern**:
```bash
grep -rn "func (m.*) View()" cli/ --include="*.go" -A20
```

**Pass Criteria**: View only reads from model fields, never assigns to them.

**Fail Criteria**: View modifies state, causing rendering bugs or infinite loops.

**Severity**: High

**Recommendation**:
```go
// Good
func (m myModel) View() string {
    var b strings.Builder
    b.WriteString(m.header())
    b.WriteString(m.viewport.View())
    b.WriteString(m.footer())
    return b.String()
}

// Avoid - modifying state in View
func (m myModel) View() string {
    m.renderCount++  // BAD: modifying state
    return fmt.Sprintf("Rendered %d times", m.renderCount)
}
```

---

### 6. Use Lipgloss Styles Consistently

**Description**: Define styles as package-level variables or model fields for consistency.

**Search Pattern**:
```bash
grep -rn "lipgloss.NewStyle()" cli/ --include="*.go"
```

**Pass Criteria**: Styles defined once and reused, not created in every render.

**Fail Criteria**: Creating new styles in every `View()` call, causing performance issues.

**Severity**: Medium

**Recommendation**:
```go
// Define styles at package level or in model initialization
var (
    headerStyle = lipgloss.NewStyle().
        Bold(true).
        Foreground(lipgloss.Color("86"))

    errorStyle = lipgloss.NewStyle().
        Foreground(lipgloss.Color("9"))

    hintStyle = lipgloss.NewStyle().
        Foreground(lipgloss.Color("240"))
)

// Use in View
func (m myModel) View() string {
    return headerStyle.Render("Title") + "\n" +
           m.content + "\n" +
           hintStyle.Render("Press q to quit")
}
```

---

### 7. Handle Keyboard Events Properly

**Description**: Use `tea.KeyMsg` with proper key string comparisons.

**Search Pattern**:
```bash
grep -rn 'msg.String() ==' cli/ --include="*.go"
```

**Pass Criteria**: Key handling uses `msg.String()` for readable keys or `msg.Type` for special keys.

**Fail Criteria**: Inconsistent key handling that misses edge cases.

**Severity**: Medium

**Recommendation**:
```go
func (m myModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
    switch msg := msg.(type) {
    case tea.KeyMsg:
        switch msg.String() {
        case "ctrl+c", "q":
            return m, tea.Quit
        case "ctrl+t":
            m.toggleMode()
        case "enter":
            return m, m.submitInput()
        case "up":
            m.navigateHistory(-1)
        case "down":
            m.navigateHistory(1)
        case "esc":
            if m.menuActive {
                m.menuActive = false
            } else if m.loading {
                m.cancelOperation()
            }
        }
    }
    return m, nil
}
```

---

### 8. Implement Cancellation Support

**Description**: Long-running operations should be cancellable via Escape or Ctrl+C.

**Search Pattern**:
```bash
grep -rn "Cancel()" cli/ --include="*.go"
```

**Pass Criteria**: Streaming operations can be cancelled, UI provides feedback.

**Fail Criteria**: Operations run to completion with no way to stop them.

**Severity**: High

**Recommendation**:
```go
type myModel struct {
    cancelFunc context.CancelFunc
    loading    bool
}

func (m myModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
    switch msg := msg.(type) {
    case tea.KeyMsg:
        if msg.String() == "esc" && m.loading {
            if m.cancelFunc != nil {
                m.cancelFunc()
            }
            m.loading = false
            m.messages = append(m.messages, "Operation cancelled")
            return m, nil
        }
    }
    return m, nil
}

func (m *myModel) startAsyncOperation() tea.Cmd {
    ctx, cancel := context.WithCancel(context.Background())
    m.cancelFunc = cancel
    m.loading = true

    return func() tea.Msg {
        result, err := doWork(ctx)
        if err != nil {
            return errorMsg{err: err}
        }
        return responseMsg{content: result}
    }
}
```

---

### 9. Use Viewport for Scrollable Content

**Description**: Use the viewport component for content that may exceed terminal height.

**Search Pattern**:
```bash
grep -rn "viewport.Model" cli/ --include="*.go"
```

**Pass Criteria**: Long content uses viewport with proper height calculation.

**Fail Criteria**: Content overflow without scrolling capability.

**Severity**: Medium

**Recommendation**:
```go
import "github.com/charmbracelet/bubbles/viewport"

type myModel struct {
    viewport viewport.Model
    ready    bool
}

func newModel() myModel {
    vp := viewport.New(80, 20)
    vp.SetContent("Initial content")
    return myModel{viewport: vp}
}

func (m myModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
    var cmd tea.Cmd

    switch msg := msg.(type) {
    case tea.WindowSizeMsg:
        headerHeight := 3
        footerHeight := 2
        m.viewport.Width = msg.Width
        m.viewport.Height = msg.Height - headerHeight - footerHeight
        m.ready = true
    }

    m.viewport, cmd = m.viewport.Update(msg)
    return m, cmd
}
```

---

### 10. Handle Auto-Scrolling Correctly

**Description**: Auto-scroll to bottom for new content, but respect user scroll position.

**Search Pattern**:
```bash
grep -rn "GotoBottom\|AtBottom" cli/ --include="*.go"
```

**Pass Criteria**: New content scrolls to bottom only if user was already at bottom.

**Fail Criteria**: Auto-scroll interrupts user reading previous content.

**Severity**: Medium

**Recommendation**:
```go
func (m myModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
    switch msg := msg.(type) {
    case responseMsg:
        // Check if user was at bottom before adding content
        wasAtBottom := m.viewport.AtBottom()

        // Add new content
        m.content += msg.content
        m.viewport.SetContent(m.content)

        // Only auto-scroll if user was following along
        if wasAtBottom || m.justStartedResponse {
            m.viewport.GotoBottom()
        }
    }
    return m, nil
}
```

---

### 11. Use Spinner for Loading States

**Description**: Show spinner animation during async operations for user feedback.

**Search Pattern**:
```bash
grep -rn "spinner.Model" cli/ --include="*.go"
```

**Pass Criteria**: Loading states show animated feedback.

**Fail Criteria**: UI appears frozen during long operations.

**Severity**: Low

**Recommendation**:
```go
import "github.com/charmbracelet/bubbles/spinner"

type myModel struct {
    spinner spinner.Model
    loading bool
}

func newModel() myModel {
    s := spinner.New()
    s.Spinner = spinner.Dot
    s.Style = lipgloss.NewStyle().Foreground(lipgloss.Color("205"))
    return myModel{spinner: s}
}

func (m myModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
    if m.loading {
        var cmd tea.Cmd
        m.spinner, cmd = m.spinner.Update(msg)
        return m, cmd
    }
    return m, nil
}

func (m myModel) View() string {
    if m.loading {
        return m.spinner.View() + " Loading..."
    }
    return m.content
}
```

---

### 12. Separate Overlay Components

**Description**: Overlay components (menus, dialogs) should be separate models for reusability.

**Search Pattern**:
```bash
grep -rn "type.*MenuModel\|type.*ToastModel" cli/ --include="*.go"
```

**Pass Criteria**: Overlays are self-contained models with their own Update/View.

**Fail Criteria**: Overlay logic mixed into main model, making it complex.

**Severity**: Medium

**Recommendation**:
```go
// internal/tui/toast.go
type ToastModel struct {
    message   string
    visible   bool
    timestamp time.Time
}

func (m ToastModel) Update(msg tea.Msg) (ToastModel, tea.Cmd) {
    switch msg := msg.(type) {
    case ShowToastMsg:
        m.message = msg.Message
        m.visible = true
        return m, tea.Tick(3*time.Second, func(t time.Time) tea.Msg {
            return HideToastMsg{}
        })
    case HideToastMsg:
        m.visible = false
    }
    return m, nil
}

// In main model
func (m myModel) View() string {
    content := m.mainContent()
    if toast := m.toast.View(); toast != "" {
        content += "\n" + toast
    }
    return content
}
```
