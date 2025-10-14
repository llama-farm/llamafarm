package tui

import (
    "fmt"
    "strings"

    tea "github.com/charmbracelet/bubbletea"
    "github.com/charmbracelet/lipgloss"
    "github.com/atotto/clipboard"
)

type MenuTab int

const (
    ContextTab MenuTab = iota
    CommandsTab
    ConfigTab
)

type MenuState int

const (
    NormalState MenuState = iota
    ProjectSelectState
)

type QuickMenuModel struct {
    active    bool
    activeTab MenuTab
    cursorPos int
    width     int
    height    int
    menuState MenuState

    config           *Config
    currentProject   string
    currentNamespace string
    currentDB        string
    currentStrategy  string
    currentModel     string
    devMode          bool

    projects   []ProjectItem
    databases  []DatabaseItem
    strategies []StrategyItem
    models     []ModelItem
    commands   []CommandItem

    // Map of database name to its strategies
    databaseStrategies map[string][]StrategyItem

    projectCursorPos int

    baseStyle      lipgloss.Style
    focusedStyle   lipgloss.Style
    dimmedStyle    lipgloss.Style
    headerStyle    lipgloss.Style
    hintStyle      lipgloss.Style
    borderStyle    lipgloss.Style
    tabStyle       lipgloss.Style
    activeTabStyle lipgloss.Style
    // Accent colors
    accentColor    lipgloss.Color
    activeColor    lipgloss.Color
    hintDimColor   lipgloss.Color
}

type ProjectItem struct {
    Name      string
    Namespace string
    IsActive  bool
}

type DatabaseItem struct {
    Name     string
    DocCount int
    IsActive bool
}

type StrategyItem struct {
    Name     string
    IsActive bool
}

type ModelItem struct {
    Name     string
    Provider string
    IsActive bool
}

type CommandItem struct {
    Command     string
    Description string
}

type Config struct {
    Name      string
    Namespace string
}

func NewQuickMenuModel(config *Config) QuickMenuModel {
    m := QuickMenuModel{
        active:           false,
        activeTab:        ContextTab,
        cursorPos:        0,
        config:           config,
        currentProject:   config.Name,
        currentNamespace: config.Namespace,
        devMode:          false,
        menuState:        NormalState,
    }

    m.baseStyle = lipgloss.NewStyle()
    m.focusedStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("86")).Bold(true)
    m.dimmedStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("240"))
    m.headerStyle = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("86"))
    m.hintStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("240"))
    m.borderStyle = lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(lipgloss.Color("86")).Padding(1, 2)
    // Tabs: pill style (active has background), no underline borders
    m.tabStyle = lipgloss.NewStyle().Padding(0, 2).Foreground(lipgloss.Color("245"))
    m.activeTabStyle = lipgloss.NewStyle().Padding(0, 2).Foreground(lipgloss.Color("#000000")).Background(lipgloss.Color("86")).Bold(true)
    m.accentColor = lipgloss.Color("86")
    m.activeColor = lipgloss.Color("39")
    m.hintDimColor = lipgloss.Color("240")

    m.commands = []CommandItem{
        {Command: "lf init", Description: "Scaffold a project and generate llamafarm.yaml"},
        {Command: "lf start", Description: "Launch server + RAG services and open the dev chat UI"},
        {Command: "lf chat", Description: "Send single prompts, preview REST calls, manage sessions"},
        {Command: "lf models", Description: "List available models and manage multi-model configurations"},
        {Command: "lf datasets", Description: "Create, upload, process, and delete datasets"},
        {Command: "lf rag", Description: "Query documents and access RAG maintenance tools"},
        {Command: "lf projects", Description: "List projects by namespace"},
        {Command: "lf version", Description: "Print CLI version/build info"},
    }

    // Initialize with empty slices - will be populated via UpdateData method
    m.databases = []DatabaseItem{}
    m.strategies = []StrategyItem{}
    m.models = []ModelItem{}
    m.projects = []ProjectItem{
        {Name: config.Name, Namespace: config.Namespace, IsActive: true},
    }

    return m
}

func (m QuickMenuModel) Update(msg tea.Msg) (QuickMenuModel, tea.Cmd) {
    if !m.active {
        switch msg := msg.(type) {
        case tea.WindowSizeMsg:
            m.width = msg.Width
            m.height = msg.Height
        case SwitchModeMsg:
            m.devMode = msg.DevMode
        case SwitchDatabaseMsg:
            m.currentDB = msg.DatabaseName
            for i := range m.databases {
                m.databases[i].IsActive = (m.databases[i].Name == msg.DatabaseName)
            }
            // Update strategies to show only those for the selected database
            m.updateStrategiesForCurrentDatabase()
            // Reset strategy selection to the default for the new database
            if len(m.strategies) > 0 {
                m.currentStrategy = m.strategies[0].Name
                m.strategies[0].IsActive = true
            }
        case SwitchModelMsg:
            m.currentModel = msg.ModelName
            for i := range m.models {
                m.models[i].IsActive = (m.models[i].Name == msg.ModelName)
            }
        case SwitchProjectMsg:
            m.currentProject = msg.ProjectName
            m.currentNamespace = msg.Namespace
            for i := range m.projects {
                m.projects[i].IsActive = (m.projects[i].Name == msg.ProjectName && m.projects[i].Namespace == msg.Namespace)
            }
        case SwitchStrategyMsg:
            m.currentStrategy = msg.StrategyName
            for i := range m.strategies {
                m.strategies[i].IsActive = (m.strategies[i].Name == msg.StrategyName)
            }
        }
        return m, nil
    }

    switch msg := msg.(type) {
    case tea.WindowSizeMsg:
        m.width = msg.Width
        m.height = msg.Height
        return m, nil
    case SwitchModeMsg:
        m.devMode = msg.DevMode
        return m, nil
    case SwitchDatabaseMsg:
        m.currentDB = msg.DatabaseName
        for i := range m.databases {
            m.databases[i].IsActive = (m.databases[i].Name == msg.DatabaseName)
        }
        // Update strategies to show only those for the selected database
        m.updateStrategiesForCurrentDatabase()
        // Reset strategy selection to the default for the new database
        if len(m.strategies) > 0 {
            m.currentStrategy = m.strategies[0].Name
            m.strategies[0].IsActive = true
        }
        return m, nil
    case SwitchModelMsg:
        m.currentModel = msg.ModelName
        for i := range m.models {
            m.models[i].IsActive = (m.models[i].Name == msg.ModelName)
        }
        return m, nil
    case SwitchProjectMsg:
        m.currentProject = msg.ProjectName
        m.currentNamespace = msg.Namespace
        for i := range m.projects {
            m.projects[i].IsActive = (m.projects[i].Name == msg.ProjectName && m.projects[i].Namespace == msg.Namespace)
        }
        return m, nil
    case SwitchStrategyMsg:
        m.currentStrategy = msg.StrategyName
        for i := range m.strategies {
            m.strategies[i].IsActive = (m.strategies[i].Name == msg.StrategyName)
        }
        return m, nil
    case tea.KeyMsg:
        if m.menuState == ProjectSelectState {
            return m.handleProjectSelection(msg)
        }
        switch msg.String() {
        case "esc":
            m.active = false
            return m, nil
        case "ctrl+c":
            return m, tea.Quit
        case "tab":
            m.activeTab = (m.activeTab + 1) % 3
            m.cursorPos = 0
            return m, nil
        case "shift+tab":
            m.activeTab = (m.activeTab - 1 + 3) % 3
            m.cursorPos = 0
            return m, nil
        case "up", "k":
            // Wrap-around navigation
            maxPos := m.getMaxCursorPos()
            if m.cursorPos > 0 {
                m.cursorPos--
            } else {
                m.cursorPos = maxPos
            }
            return m, nil
        case "down", "j":
            // Wrap-around navigation
            maxPos := m.getMaxCursorPos()
            if m.cursorPos < maxPos {
                m.cursorPos++
            } else {
                m.cursorPos = 0
            }
            return m, nil
        case "enter":
            return m, m.handleSelection()
        case "ctrl+t", "cmd+t":
            m.devMode = !m.devMode
            return m, switchModeCmd(m.devMode)
        }
    }
    return m, nil
}

func (m QuickMenuModel) handleProjectSelection(msg tea.KeyMsg) (QuickMenuModel, tea.Cmd) {
    switch msg.String() {
    case "esc":
        m.menuState = NormalState
        return m, nil
    case "up", "k":
        if m.projectCursorPos > 0 {
            m.projectCursorPos--
        } else if len(m.projects) > 0 {
            m.projectCursorPos = len(m.projects) - 1
        }
        return m, nil
    case "down", "j":
        if m.projectCursorPos < len(m.projects)-1 {
            m.projectCursorPos++
        } else if len(m.projects) > 0 {
            m.projectCursorPos = 0
        }
        return m, nil
    case "enter":
        selectedProject := m.projects[m.projectCursorPos]
        m.menuState = NormalState
        return m, switchProjectCmd(selectedProject.Name, selectedProject.Namespace)
    }
    return m, nil
}

func (m QuickMenuModel) getMaxCursorPos() int {
    switch m.activeTab {
    case ContextTab:
        // 0: PROJECT MODE, 1: DEV MODE, 2: Project line, 3..dbs, then strategies
        // Max cursor index: 2 + len(dbs) + len(strategies)
        return 2 + len(m.databases) + len(m.strategies)
    case CommandsTab:
        return len(m.commands) - 1
    case ConfigTab:
        // Allow cursor to land on embedding model display row
        return len(m.models)
    default:
        return 0
    }
}

func (m QuickMenuModel) handleSelection() tea.Cmd {
    switch m.activeTab {
    case ContextTab:
        if m.cursorPos == 0 {
            // PROJECT MODE row: if currently in DEV, allow switching back to PROJECT
            if m.devMode {
                m.devMode = false
                return switchModeCmd(false)
            }
            return nil
        }
        if m.cursorPos == 1 {
            // Switch to DEV mode when selecting DEV MODE row
            if !m.devMode {
                m.devMode = true
                return switchModeCmd(true)
            }
            return nil
        }
        if m.cursorPos == 2 {
            // Enter project selection state
            m.menuState = ProjectSelectState
            // Initialize cursor to current project
            m.projectCursorPos = m.findCurrentProjectIndex()
            return nil
        }
        // Databases start at cursor index 3
        dbIndex := m.cursorPos - 3
        if dbIndex >= 0 && len(m.databases) > 0 && dbIndex < len(m.databases) {
            selectedDB := m.databases[dbIndex]
            return switchDatabaseCmd(selectedDB.Name)
        }
        // Strategy selection follows databases
        // Strategies start at cursor index 3+len(dbs)
        stratIndex := m.cursorPos - (3 + len(m.databases))
        if stratIndex >= 0 && len(m.strategies) > 0 && stratIndex < len(m.strategies) {
            selected := m.strategies[stratIndex]
            return switchStrategyCmd(selected.Name)
        }
    case CommandsTab:
        if len(m.commands) > 0 && m.cursorPos < len(m.commands) {
            cmd := m.commands[m.cursorPos].Command
            clipboard.WriteAll(cmd)
            return showToastCmd("Copied: " + cmd)
        }
    case ConfigTab:
        if len(m.models) > 0 && m.cursorPos < len(m.models) {
            selectedModel := m.models[m.cursorPos]
            return switchModelCmd(selectedModel.Name)
        }
    }
    return nil
}

func (m QuickMenuModel) handleSelectionAndRun() tea.Cmd {
    if m.activeTab == CommandsTab && m.cursorPos < len(m.commands) {
        cmd := m.commands[m.cursorPos].Command
        m.active = false
        return executeCommandCmd(cmd)
    }
    return m.handleSelection()
}

func (m QuickMenuModel) findCurrentProjectIndex() int {
    for i, proj := range m.projects {
        if proj.IsActive {
            return i
        }
    }
    return 0
}

func (m QuickMenuModel) View() string {
    if !m.active {
        return ""
    }
    menuWidth := 60
    var content strings.Builder
    header := m.headerStyle.Render("🦙 LlamaFarm Quick Menu")
    closeHint := m.hintStyle.Render("[ESC to close]")
    headerLine := lipgloss.JoinHorizontal(lipgloss.Left, header, strings.Repeat(" ", menuWidth-lipgloss.Width(header)-lipgloss.Width(closeHint)-4), closeHint)
    content.WriteString(headerLine + "\n")
    content.WriteString(strings.Repeat("─", menuWidth-4) + "\n\n")
    content.WriteString(m.renderTabBar() + "\n\n")
    switch m.activeTab {
    case ContextTab:
        content.WriteString(m.renderContextTab())
    case CommandsTab:
        content.WriteString(m.renderCommandsTab())
    case ConfigTab:
        content.WriteString(m.renderConfigTab())
    }
    content.WriteString("\n" + strings.Repeat("─", menuWidth-4) + "\n")
    // Footer boxed styling with fixed inner width to avoid wrap glitches
    innerWidth := menuWidth - 6 // account for outer padding/borders
    footerBox := lipgloss.NewStyle().
        Border(lipgloss.RoundedBorder()).
        BorderForeground(m.accentColor).
        Padding(0, 1).
        Width(innerWidth)
    content.WriteString(footerBox.Render(m.renderFooter()))
    box := m.borderStyle.Width(menuWidth).Render(content.String())
    return m.positionMenu(box)
}

func (m QuickMenuModel) renderTabBar() string {
    tabs := []string{"Context", "Commands", "Config"}
    var pieces []string
    sep := lipgloss.NewStyle().Foreground(lipgloss.Color("238")).Render("|")
    for i, tab := range tabs {
        if i > 0 {
            pieces = append(pieces, sep)
        }
        if MenuTab(i) == m.activeTab {
            pieces = append(pieces, m.activeTabStyle.Render(tab))
        } else {
            pieces = append(pieces, m.tabStyle.Render(tab))
        }
    }
    // Left-align to avoid wrapping when the active pill is wider
    return strings.Join(pieces, " ")
}

func (m QuickMenuModel) renderContextTab() string {
    if m.menuState == ProjectSelectState {
        return m.renderProjectSelector()
    }
    var s strings.Builder
    itemIndex := 0
    // PROJECT MODE line (shows selected state when not in DEV)
    projCursor := "  "
    if m.cursorPos == itemIndex {
        projCursor = "→ "
    }
    projActive := " "
    if !m.devMode {
        projActive = "✓"
    }
    projLine := fmt.Sprintf("%s%s 🤖 PROJECT MODE", projCursor, projActive)
    if m.cursorPos == itemIndex {
        // Cursor highlight takes precedence
        projLine = m.focusedStyle.Render(projLine)
    } else if !m.devMode {
        // Visualize currently selected mode even when not focused
        projLine = lipgloss.NewStyle().Foreground(m.activeColor).Bold(true).Render(projLine)
    }
    s.WriteString(projLine + "\n")
    itemIndex++
    // DEV MODE line (actionable: pressing Enter switches)
    devCursor := "  "
    if m.cursorPos == itemIndex {
        devCursor = "→ "
    }
    devActive := " "
    if m.devMode {
        devActive = "✓"
    }
    devLine := fmt.Sprintf("%s%s 🔧 DEV MODE", devCursor, devActive)
    if m.cursorPos == itemIndex {
        devLine = m.focusedStyle.Render(devLine)
    } else if m.devMode {
        // Visualize currently selected mode even when not focused
        devLine = lipgloss.NewStyle().Foreground(m.activeColor).Bold(true).Render(devLine)
    }
    s.WriteString(devLine + "\n\n")
    itemIndex++
    projCursor2 := "  "
    if m.cursorPos == itemIndex {
        projCursor2 = "→ "
    }
    currentProjectText := m.currentProject
    if len(m.projects) == 0 {
        currentProjectText = "(none)"
    }
    projectLine := fmt.Sprintf("%sProject:  %s", projCursor2, currentProjectText)
    if len(m.projects) > 1 {
        projectLine += "    [▼]"
    }
    if m.cursorPos == itemIndex {
        projectLine = m.focusedStyle.Render(projectLine)
    }
    s.WriteString(projectLine + "\n")
    nsText := m.currentNamespace
    if nsText == "" {
        nsText = "(none)"
    }
    s.WriteString(fmt.Sprintf("  Namespace: %s\n\n", nsText))
    itemIndex++
    s.WriteString("Database:\n")
    if len(m.databases) == 0 {
        s.WriteString("  (no databases)\n")
    }
    for _, db := range m.databases {
        cursor := "  "
        if itemIndex == m.cursorPos {
            cursor = "→ "
        }
        active := " "
        if db.IsActive {
            active = "✓"
        }
        status := ""
        if db.IsActive {
            status = " [active]"
        }
        // Highlight active database name in accent color
        name := db.Name
        if db.IsActive {
            name = lipgloss.NewStyle().Foreground(m.activeColor).Bold(true).Render(name)
        }
        line := fmt.Sprintf("%s%s %-20s (%d docs)%s", cursor, active, name, db.DocCount, status)
        if itemIndex == m.cursorPos {
            line = m.focusedStyle.Render(line)
        }
        s.WriteString(line + "\n")
        itemIndex++
    }
    s.WriteString("\n")
    s.WriteString("Retrieval Strategy:\n")
    if len(m.strategies) == 0 {
        s.WriteString("  (no strategies)\n")
    }
    for idx, strategy := range m.strategies {
        cursor := "  "
        // Strategies start at cursor index 3 + len(databases)
        stratIndex := 3 + len(m.databases) + idx
        if stratIndex == m.cursorPos {
            cursor = "→ "
        }
        active := " "
        if strategy.IsActive {
            active = "✓"
        }
        name := strategy.Name
        if strategy.IsActive {
            name = lipgloss.NewStyle().Foreground(m.activeColor).Bold(true).Render(name)
        }
        line := fmt.Sprintf("%s%s %s", cursor, active, name)
        if strategy.IsActive && stratIndex != m.cursorPos {
            line = lipgloss.NewStyle().Foreground(m.activeColor).Bold(true).Render(line)
        }
        if stratIndex == m.cursorPos {
            line = m.focusedStyle.Render(line)
        }
        s.WriteString(line + "\n")
    }
    return s.String()
}

func (m QuickMenuModel) renderProjectSelector() string {
    var s strings.Builder
    s.WriteString("Select Project:\n\n")
    for i, proj := range m.projects {
        cursor := "  "
        if i == m.projectCursorPos {
            cursor = "→ "
        }
        active := " "
        if proj.IsActive {
            active = "✓"
        }
        status := ""
        if proj.IsActive {
            status = " [current]"
        }
        line := fmt.Sprintf("%s%s %s/%s%s", cursor, active, proj.Name, proj.Namespace, status)
        if i == m.projectCursorPos {
            line = m.focusedStyle.Render(line)
        }
        s.WriteString(line + "\n")
    }
    s.WriteString("\n")
    s.WriteString(m.hintStyle.Render("[Press Enter to switch, Esc to cancel]"))
    return s.String()
}

func (m QuickMenuModel) renderCommandsTab() string {
    var s strings.Builder
    hint := m.hintStyle.Render("CLI Commands (Enter: copy)")
    s.WriteString(hint + "\n")
    for i, cmd := range m.commands {
        cursor := "  "
        if i == m.cursorPos {
            cursor = "→ "
        }
        cmdText := cmd.Command
        if i == m.cursorPos {
            cmdText = m.focusedStyle.Render(cmdText)
        } else {
            cmdText = lipgloss.NewStyle().Foreground(lipgloss.Color("86")).Render(cmdText)
        }
        descLines := wrapText(cmd.Description, 45)
        // Tighter spacing: single line if possible
        s.WriteString(fmt.Sprintf("%s%-18s %s\n", cursor, cmdText, descLines[0]))
        for j := 1; j < len(descLines); j++ {
            s.WriteString(fmt.Sprintf("                     %s\n", descLines[j]))
        }
        // Minimal gap between commands
    }
    return s.String()
}

func (m QuickMenuModel) renderConfigTab() string {
    var s strings.Builder
    s.WriteString("Inference Model:\n")
    for i, model := range m.models {
        cursor := "  "
        if i == m.cursorPos {
            cursor = "→ "
        }
        active := " "
        if model.IsActive {
            active = "✓"
        }
        line := fmt.Sprintf("%s%s %s (%s)", cursor, active, model.Name, model.Provider)
        if i == m.cursorPos {
            line = m.focusedStyle.Render(line)
        }
        s.WriteString(line + "\n")
    }
    s.WriteString("\n")
    s.WriteString("Embedding Model:\n")
    // Show embedding model as selected styling to match inference model section
    itemIndex := len(m.models) // embedding row index
    embCursor := "  "
    if m.cursorPos == itemIndex { // allow focus on this informational row
        embCursor = "→ "
    }
    embLine := fmt.Sprintf("%s✓ nomic-embed-text", embCursor)
    if m.cursorPos == itemIndex {
        embLine = m.focusedStyle.Render(embLine)
    } else {
        embLine = lipgloss.NewStyle().Foreground(m.activeColor).Bold(true).Render(embLine)
    }
    s.WriteString(embLine + "\n")
    return s.String()
}

func (m QuickMenuModel) renderFooter() string {
    var shortcuts []string
    if m.menuState == ProjectSelectState {
        shortcuts = []string{"↑↓: select", "Enter: switch", "Esc: cancel"}
    } else {
        switch m.activeTab {
        case CommandsTab:
            shortcuts = []string{"↑↓: scroll", "Enter: copy", "Tab: next", "ESC: close"}
        default:
            shortcuts = []string{"↑↓: navigate", "Enter: select", "Tab: sections", "ESC: close"}
        }
    }
    return m.hintStyle.Render(strings.Join(shortcuts, "  "))
}

func (m QuickMenuModel) positionMenu(content string) string {
    contentLines := strings.Split(content, "\n")
    contentHeight := len(contentLines)
    contentWidth := 0
    for _, line := range contentLines {
        w := lipgloss.Width(line)
        if w > contentWidth {
            contentWidth = w
        }
    }
    // When height is zero, render without vertical padding (above input)
    topPadding := 0
    if m.height > 0 {
        topPadding = (m.height - contentHeight) / 2
        if topPadding < 0 {
            topPadding = 0
        }
    }
    leftPadding := (m.width - contentWidth) / 2
    if leftPadding < 0 {
        leftPadding = 0
    }
    var result strings.Builder
    for i := 0; i < topPadding; i++ {
        result.WriteString("\n")
    }
    for _, line := range contentLines {
        result.WriteString(strings.Repeat(" ", leftPadding))
        result.WriteString(line)
        result.WriteString("\n")
    }
    return result.String()
}

func wrapText(text string, width int) []string {
    if len(text) <= width {
        return []string{text}
    }
    var lines []string
    words := strings.Fields(text)
    currentLine := ""
    for _, word := range words {
        if len(currentLine)+len(word)+1 <= width {
            if currentLine == "" {
                currentLine = word
            } else {
                currentLine += " " + word
            }
        } else {
            if currentLine != "" {
                lines = append(lines, currentLine)
            }
            currentLine = word
        }
    }
    if currentLine != "" {
        lines = append(lines, currentLine)
    }
    return lines
}

type SwitchModeMsg struct{ DevMode bool }
func switchModeCmd(devMode bool) tea.Cmd { return func() tea.Msg { return SwitchModeMsg{DevMode: devMode} } }

type SwitchProjectMsg struct{ ProjectName, Namespace string }
func switchProjectCmd(projectName, namespace string) tea.Cmd {
    return func() tea.Msg { return SwitchProjectMsg{ProjectName: projectName, Namespace: namespace} }
}

type SwitchDatabaseMsg struct{ DatabaseName string }
func switchDatabaseCmd(dbName string) tea.Cmd { return func() tea.Msg { return SwitchDatabaseMsg{DatabaseName: dbName} } }

type SwitchModelMsg struct{ ModelName string }
func switchModelCmd(modelName string) tea.Cmd { return func() tea.Msg { return SwitchModelMsg{ModelName: modelName} } }

type ExecuteCommandMsg struct{ Command string }
func executeCommandCmd(command string) tea.Cmd { return func() tea.Msg { return ExecuteCommandMsg{Command: command} } }

type ShowToastMsg struct{ Message string }
func showToastCmd(message string) tea.Cmd { return func() tea.Msg { return ShowToastMsg{Message: message} } }

type SwitchStrategyMsg struct{ StrategyName string }
func switchStrategyCmd(name string) tea.Cmd { return func() tea.Msg { return SwitchStrategyMsg{StrategyName: name} } }

func (m *QuickMenuModel) Toggle() { m.active = !m.active; if m.active { m.activeTab = ContextTab; m.cursorPos = 0; m.menuState = NormalState } }
func (m *QuickMenuModel) Open()   { m.active = true; m.activeTab = ContextTab; m.cursorPos = 0; m.menuState = NormalState }
func (m *QuickMenuModel) Close()  { m.active = false; m.menuState = NormalState }
func (m QuickMenuModel) IsActive() bool { return m.active }

// UpdateMenuDataMsg is sent when menu data needs to be refreshed from configuration
type UpdateMenuDataMsg struct {
	Models     []ModelItem
	Databases  []DatabaseItem
	Strategies []StrategyItem
}

// SetData updates the menu with real configuration data
func (m *QuickMenuModel) SetData(models []ModelItem, databases []DatabaseItem, databaseStrategies map[string][]StrategyItem, currentModel, currentDB, currentStrategy string) {
	m.models = models
	m.databases = databases
	m.databaseStrategies = databaseStrategies
	m.currentModel = currentModel
	m.currentDB = currentDB
	m.currentStrategy = currentStrategy

	// Update strategies to show only those for the current database
	m.updateStrategiesForCurrentDatabase()
}

// updateStrategiesForCurrentDatabase updates the strategies slice to only show strategies for the current database
func (m *QuickMenuModel) updateStrategiesForCurrentDatabase() {
	if m.databaseStrategies != nil && m.currentDB != "" {
		if strategies, ok := m.databaseStrategies[m.currentDB]; ok {
			m.strategies = strategies
		} else {
			m.strategies = []StrategyItem{}
		}
	} else {
		m.strategies = []StrategyItem{}
	}
}


