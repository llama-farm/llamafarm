package cmd

import (
	"context"
	"fmt"
	"hash/fnv"
	"io"
	"llamafarm-cli/cmd/config"
	uitk "llamafarm-cli/internal/tui"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/term"
	"github.com/google/uuid"
)

var (
	farmerPrompt     = "🌾 Farmer:"
	serverPrompt     = "📡 Server:"
	ollamaHostPrompt = "🐏 Ollama:"
	projectPrompt    = "📁 Project:"
	sessionPrompt    = "🆔"
)

// getAssistantLabel returns the appropriate label based on the current chat mode
func (m chatModel) getAssistantLabel() string {
	if m.currentMode == ModeProject {
		return projectPrompt
	}
	return farmerPrompt
}

// renderMarkdown is disabled for now - Glamour doesn't work well in TUI environments
// It detects we're not in a TTY and falls back to ASCII-only mode regardless of config
func renderMarkdown(content string, width int) string {
	// TODO: Implement proper markdown rendering for TUI
	// For now, just return the content as-is
	return content
}

const gap = "\n\n"

// overrides provided by dev command
var designerPreferredPort int
var designerForced bool

var lastTranscriptKey string

var chatCtx = &ChatSessionContext{
	ServerURL:        serverURL,
	Namespace:        "llamafarm",
	ProjectID:        "project_seed",
	SessionMode:      SessionModeDev,
	SessionNamespace: namespace,
	SessionProject:   projectID,
	Temperature:      temperature,
	MaxTokens:        maxTokens,
	HTTPClient:       getHTTPClient(),
}

// fetchAvailableModels is now defined in models_shared.go

// runChatSessionTUI starts the Bubble Tea TUI for chat.
func runChatSessionTUI(mode SessionMode, projectInfo *config.ProjectInfo, serverHealth *HealthPayload) {
	// Update session context with project info first
	if projectInfo != nil {
		chatCtx.SessionNamespace = projectInfo.Namespace
		chatCtx.SessionProject = projectInfo.Project
		if mode == SessionModeProject {
			chatCtx.Namespace = projectInfo.Namespace
			chatCtx.ProjectID = projectInfo.Project
		}
	}
	chatCtx.ServerURL = serverURL
	chatCtx.HTTPClient = getHTTPClient()
	chatCtx.SessionMode = mode

	// Load existing session context to restore session ID if available
	// This needs to happen AFTER we set SessionNamespace/SessionProject
	// so we read from the correct location
	if chatCtx.SessionMode == SessionModeDev {
		if existingContext, err := readSessionContext(chatCtx); err == nil && existingContext != nil {
			chatCtx.SessionID = existingContext.SessionID
			logDebug(fmt.Sprintf("Restored dev mode session ID: %s", chatCtx.SessionID))
		}
	} else if chatCtx.SessionMode == SessionModeProject {
		if existingContext, err := readSessionContext(chatCtx); err == nil && existingContext != nil {
			chatCtx.SessionID = existingContext.SessionID
			logDebug(fmt.Sprintf("Restored project mode session ID: %s", chatCtx.SessionID))
		}
	}

	m := newChatModel(projectInfo, serverHealth)
	p := tea.NewProgram(m)
	m.program = p

	// Enable TUI mode for output routing
	SetTUIMode(p)
	defer ClearTUIMode()

	if _, err := p.Run(); err != nil {
		// Use the output API instead of direct stderr write
		OutputError("Error running TUI: %v\n", err)
	}
}

type ChatMode int

const (
	ModeDev     ChatMode = iota // Chat with llamafarm/project_seed for help
	ModeProject                 // Chat with user's project to test
)

type ModeContext struct {
	Mode              ChatMode
	SessionID         string
	Messages          []Message
	History           []string
	Model             string // Currently selected model name
	Database          string // Currently selected database
	RetrievalStrategy string // Currently selected retrieval strategy
}

// ModelInfo is now defined in models_shared.go

type chatModel struct {
	transcript     string
	serverHealth   *HealthPayload
	projectInfo    *config.ProjectInfo
	spin           spinner.Model
	messages       []Message
	thinking       bool
	printing       bool
	thinkFrame     int
	history        []string
	histIndex      int
	width          int
	status         string
	err            error
	viewport       viewport.Model
	textarea       textarea.Model
	program        *tea.Program
	streamCh       chan tea.Msg
	designerStatus string
	designerURL    string
	// Mode switching state
	currentMode    ChatMode
	devModeCtx     *ModeContext
	projectModeCtx *ModeContext
	// Model switching state
	availableModels []ModelInfo
	currentModel    string
	// RAG database/strategy state
	availableDatabases *DatabasesResponse
	currentDatabase    string
	currentStrategy    string
	// Overlay menu and toast
	quickMenu  uitk.QuickMenuModel
	toast      uitk.ToastModel
	termHeight int
	menuActive bool
	// Controller decouples data/state updates from the UI model
	controller *Controller
}

// removed: old bottom menu state

type (
	streamDone struct{}
)

type responseMsg struct{ content string }
type errorMsg struct{ err error }
type tickMsg struct{}

type designerReadyMsg struct{ url string }
type designerErrorMsg struct{ err error }
type serverHealthMsg struct{ health *HealthPayload }
type modeSwitchMsg struct{ mode ChatMode }

func newChatModel(projectInfo *config.ProjectInfo, serverHealth *HealthPayload) chatModel {
	var devMessages []Message

	ta := textarea.New()
	ta.Placeholder = "Send a message..."
	ta.Focus()

	ta.Prompt = "> "

	ta.SetWidth(30)
	ta.SetHeight(1)

	// Remove cursor line styling
	ta.FocusedStyle.CursorLine = lipgloss.NewStyle()

	ta.ShowLineNumbers = false

	vp := viewport.New(30, 5)

	ta.KeyMap.InsertNewline.SetEnabled(false)

	s := spinner.New()
	s.Spinner = spinner.Dot
	s.Style = lipgloss.NewStyle().Foreground(lipgloss.Color("11"))

	// Build DEV mode history regardless of current session mode
	var devHistory SessionHistory
	var devUserChatMessages []string
	var devSessionID string
	{
		// Read DEV session context from disk: session storage is keyed to user's project
		// while chat target for DEV is llamafarm/project_seed
		devCtxForRead := &ChatSessionContext{
			ServerURL:        chatCtx.ServerURL,
			Namespace:        "llamafarm",
			ProjectID:        "project_seed",
			SessionMode:      SessionModeDev,
			SessionNamespace: chatCtx.SessionNamespace,
			SessionProject:   chatCtx.SessionProject,
			HTTPClient:       chatCtx.HTTPClient,
		}
		if existingContext, err := readSessionContext(devCtxForRead); err == nil && existingContext != nil {
			devSessionID = existingContext.SessionID
		}
		if devSessionID != "" {
			devHistory = fetchSessionHistory(chatCtx.ServerURL, "llamafarm", "project_seed", devSessionID)
			for _, msg := range devHistory.Messages {
				if msg.Role == "user" {
					devUserChatMessages = append(devUserChatMessages, msg.Content.ChatMessage)
				}
				devMessages = append(devMessages, Message{Role: msg.Role, Content: msg.Content.ChatMessage})
			}
			logDebug(fmt.Sprintf("Restored DEV history (session %s): %d messages", devSessionID, len(devHistory.Messages)))
		}
		if len(devMessages) == 0 {
			devMessages = append(devMessages, Message{Role: "client", Content: "Send a message or type '/help' for commands."})
		}
	}

	// Fetch initial greeting for project_seed (disabled)
	// if greeting := fetchInitialGreeting(chatCtx); greeting != "" {
	// 	messages = append(messages, Message{Role: "assistant", Content: greeting})
	// }

	width, _, _ := term.GetSize(uintptr(os.Stdout.Fd()))

	// Always include server status as a client message in both modes
	devMessages = append(devMessages, Message{Role: "client", Content: renderServerStatusProblems(serverHealth)})

	// Initialize mode contexts - build DEV context
	devCtx := &ModeContext{
		Mode:      ModeDev,
		SessionID: devSessionID,
		Messages:  devMessages,
		History:   devUserChatMessages,
	}

	// Project mode context - try to restore session or create new one
	var projectSessionID string
	var projectHistory []string
	var projectMessages []Message

	if projectInfo != nil {
		// Try to load existing project session
		projectChatCtx := &ChatSessionContext{
			ServerURL:        chatCtx.ServerURL,
			Namespace:        projectInfo.Namespace,
			ProjectID:        projectInfo.Project,
			SessionMode:      SessionModeProject,
			SessionNamespace: projectInfo.Namespace,
			SessionProject:   projectInfo.Project,
			HTTPClient:       chatCtx.HTTPClient,
		}
		if existingContext, err := readSessionContext(projectChatCtx); err == nil && existingContext != nil {
			projectSessionID = existingContext.SessionID
			logDebug(fmt.Sprintf("Restored project mode session ID: %s", projectSessionID))
		} else {
			// Create new session for project mode
			projectSessionID = uuid.New().String()
			logDebug(fmt.Sprintf("Created new project mode session ID: %s", projectSessionID))
		}

		// Fetch and render project session history using the project's namespace/project
		projHist := fetchSessionHistory(chatCtx.ServerURL, projectInfo.Namespace, projectInfo.Project, projectSessionID)
		for _, msg := range projHist.Messages {
			if msg.Role == "user" {
				projectHistory = append(projectHistory, msg.Content.ChatMessage)
			}
			projectMessages = append(projectMessages, Message{Role: msg.Role, Content: msg.Content.ChatMessage})
		}
	} else {
		// No project info, still create a session ID for future use
		projectSessionID = uuid.New().String()
	}

	if len(projectMessages) == 0 {
		projectMessages = []Message{{Role: "client", Content: "Send a message or type '/help' for commands."}}
	}
	// Add server status to project messages as well
	projectMessages = append(projectMessages, Message{Role: "client", Content: renderServerStatusProblems(serverHealth)})

	// Fetch available models and databases for project mode
	var availableModels []ModelInfo
	var availableDatabases *DatabasesResponse
	var availableDatasets []DatasetBrief
	var availablePrompts []config.Prompt
	var currentModel string
	var currentDatabase string
	var currentStrategy string

	if projectInfo != nil {
		// Fetch models
		availableModels = fetchAvailableModels(chatCtx.ServerURL, projectInfo.Namespace, projectInfo.Project)
		if len(availableModels) > 0 {
			// Find default model or use first
			for _, m := range availableModels {
				if m.IsDefault {
					currentModel = m.Name
					break
				}
			}
			if currentModel == "" {
				currentModel = availableModels[0].Name
			}
		}

		// Fetch databases and retrieval strategies
		availableDatabases = fetchAvailableDatabases(chatCtx.ServerURL, projectInfo.Namespace, projectInfo.Project)
		// Fetch dataset names for commands menu
		availableDatasets = fetchAvailableDatasets(chatCtx.ServerURL, projectInfo.Namespace, projectInfo.Project)
		// Load prompts from project config file on disk (best effort)
		if cfg, err := config.LoadConfig(getEffectiveCWD()); err == nil && cfg != nil {
			availablePrompts = cfg.Prompts
		}
		if availableDatabases != nil && len(availableDatabases.Databases) > 0 {
			// Find default database
			for _, db := range availableDatabases.Databases {
				if db.IsDefault {
					currentDatabase = db.Name
					// Find default strategy for this database
					for _, strategy := range db.RetrievalStrategies {
						if strategy.IsDefault {
							currentStrategy = strategy.Name
							break
						}
					}
					break
				}
			}
			// Fallback to first database/strategy if no default
			if currentDatabase == "" {
				currentDatabase = availableDatabases.Databases[0].Name
				if len(availableDatabases.Databases[0].RetrievalStrategies) > 0 {
					currentStrategy = availableDatabases.Databases[0].RetrievalStrategies[0].Name
				}
			}
		}
	}

	projectCtx := &ModeContext{
		Mode:              ModeProject,
		SessionID:         projectSessionID,
		Messages:          projectMessages,
		History:           projectHistory,
		Model:             currentModel,
		Database:          currentDatabase,
		RetrievalStrategy: currentStrategy,
	}

	// Choose initial mode and state
	initialMode := ModeDev
	initialMessages := devMessages
	initialHistory := devUserChatMessages
	if chatCtx.SessionMode == SessionModeProject && projectInfo != nil {
		initialMode = ModeProject
		initialMessages = projectMessages
		initialHistory = projectHistory
	}

	// Initialize viewport content with initial mode messages
	vp.SetContent(renderChatContent(chatModel{messages: initialMessages}))

	// Initialize overlay Quick Menu and toast
	menuCfg := &uitk.Config{}
	if projectInfo != nil {
		menuCfg.Name = projectInfo.Project
		menuCfg.Namespace = projectInfo.Namespace
	}
	// Attach CLI version for menu header
	menuCfg.Version = formatVersionForDisplay(Version)
	qm := uitk.NewQuickMenuModel(menuCfg)

	// Populate menu with real configuration data
	if projectInfo != nil {
		// Convert models to menu format
		menuModels := make([]uitk.ModelItem, 0, len(availableModels))
		for _, m := range availableModels {
			menuModels = append(menuModels, uitk.ModelItem{
				Name:        m.Name,
				Provider:    m.Provider,
				IsActive:    m.Name == currentModel,
				Description: m.Description,
			})
		}

		// Convert databases and strategies to menu format
		menuDatabases := []uitk.DatabaseItem{}
		databaseStrategies := make(map[string][]uitk.StrategyItem)

		if availableDatabases != nil {
			for _, db := range availableDatabases.Databases {
				// For now, show doc count as 0 - would need separate API call for actual counts
				menuDatabases = append(menuDatabases, uitk.DatabaseItem{
					Name:     db.Name,
					DocCount: 0,
					IsActive: db.Name == currentDatabase,
				})

				// Build strategy list for this specific database
				dbStrategies := []uitk.StrategyItem{}
				for _, strat := range db.RetrievalStrategies {
					dbStrategies = append(dbStrategies, uitk.StrategyItem{
						Name:     strat.Name,
						IsActive: (db.Name == currentDatabase && strat.Name == currentStrategy),
					})
				}
				databaseStrategies[db.Name] = dbStrategies
			}
		}

		qm.SetData(menuModels, menuDatabases, databaseStrategies, currentModel, currentDatabase, currentStrategy)
		// Provide datasets and (future) prompts into the menu for Commands tab
		if len(availableDatasets) > 0 {
			// Convert to names and details for the menu
			names := make([]string, 0, len(availableDatasets))
			for _, d := range availableDatasets {
				names = append(names, d.Name)
			}
			qm.Datasets = names
			// Also stash detailed lines into prompt-like descriptions by reusing prompts field later if desired
			// For now, embed dataset detail strings into the menu hint style inside render
		}
		if len(availablePrompts) > 0 {
			pr := make([]uitk.PromptItem, 0, len(availablePrompts))
			for _, p := range availablePrompts {
				// Prefer Content, fallback to Prompt (back-compat)
				content := p.Content
				if strings.TrimSpace(content) == "" {
					content = p.Prompt
				}
				// First line: role (not truncated); default to "system" if missing
				role := p.Role
				if strings.TrimSpace(role) == "" {
					role = "system"
				}
				name := fmt.Sprintf("role: %s", role)
				// Second line: prompt: <content preview>
				preview := strings.TrimSpace(content)
				// Truncate later to give more context (UTF-8 safe)
				const maxPreview = 1000
				if len([]rune(preview)) > maxPreview {
					preview = string([]rune(preview)[:maxPreview]) + "..."
				}
				desc := fmt.Sprintf("prompt: %s", preview)
				pr = append(pr, uitk.PromptItem{Name: name, Description: desc})
			}
			qm.Prompts = pr
		}
	}

	toast := uitk.NewToastModel()

	ctrl := NewController(State{CurrentDatabase: currentDatabase, CurrentStrategy: currentStrategy, ServerHealth: serverHealth})

	return chatModel{
		serverHealth:       serverHealth,
		projectInfo:        projectInfo,
		spin:               s,
		messages:           initialMessages,
		thinking:           false,
		printing:           false,
		history:            initialHistory,
		histIndex:          len(initialHistory),
		designerStatus:     "starting…",
		textarea:           ta,
		viewport:           vp,
		width:              width,
		currentMode:        initialMode,
		devModeCtx:         devCtx,
		projectModeCtx:     projectCtx,
		availableModels:    availableModels,
		currentModel:       currentModel,
		availableDatabases: availableDatabases,
		currentDatabase:    currentDatabase,
		currentStrategy:    currentStrategy,
		quickMenu:          qm,
		toast:              toast,
		controller:         ctrl,
	}
}

// Helper methods for mode management
func (m *chatModel) saveCurrentModeState() {
	ctx := m.getCurrentModeContext()
	ctx.Messages = m.messages
	ctx.History = m.history
	ctx.Database = m.currentDatabase
	ctx.RetrievalStrategy = m.currentStrategy
}

func (m *chatModel) getCurrentModeContext() *ModeContext {
	if m.currentMode == ModeDev {
		return m.devModeCtx
	}
	return m.projectModeCtx
}

func (m *chatModel) restoreModeState(mode ChatMode) {
	var ctx *ModeContext
	if mode == ModeDev {
		ctx = m.devModeCtx
	} else {
		ctx = m.projectModeCtx
	}

	m.messages = ctx.Messages
	m.history = ctx.History
	m.currentDatabase = ctx.Database
	m.currentStrategy = ctx.RetrievalStrategy
	m.histIndex = len(ctx.History)
}

func (m *chatModel) switchMode(newMode ChatMode) {
	// Save current state
	m.saveCurrentModeState()

	// Switch mode
	m.currentMode = newMode

	// Restore new mode state
	m.restoreModeState(newMode)

	// Update chat context
	if newMode == ModeDev {
		chatCtx.Namespace = "llamafarm"
		chatCtx.ProjectID = "project_seed"
		chatCtx.SessionID = m.devModeCtx.SessionID
		chatCtx.SessionMode = SessionModeDev
	} else {
		if m.projectInfo != nil {
			chatCtx.Namespace = m.projectInfo.Namespace
			chatCtx.ProjectID = m.projectInfo.Project
			chatCtx.SessionID = m.projectModeCtx.SessionID
			chatCtx.SessionMode = SessionModeProject
			// Restore model for project mode
			if m.projectModeCtx.Model != "" {
				m.currentModel = m.projectModeCtx.Model
				chatCtx.Model = m.currentModel
			}
		}
	}

	// Save session context for the new mode
	_ = writeSessionContext(chatCtx, chatCtx.SessionID)

	config := &ServiceOrchestrationConfig{
		ServerURL:   serverURL,
		PrintStatus: true,
		ServiceNeeds: map[string]ServiceRequirement{
			"server": ServiceRequired,
			"rag":    ServiceOptional, // Start async, don't wait
		},
		DefaultTimeout: 45 * time.Second,
	}
	health, _ := checkServerHealth(serverURL)
	m.serverHealth = FilterHealthForOptionalServices(health, config, chatCtx.SessionMode)

	// Return switch message
	chatMsg := ""
	if newMode == ModeDev {
		chatMsg = "🦙 Switched to DEV mode - Chat with LlamaFarm Assistant"
	} else {
		chatMsg = fmt.Sprintf("🎯 Switched to PROJECT mode - Testing %s/%s", chatCtx.Namespace, chatCtx.ProjectID)
	}

	shouldAppend := true
	if len(m.messages) > 0 {
		lastMsg := m.messages[len(m.messages)-1]
		if lastMsg.Role == "client" && lastMsg.Content == chatMsg {
			shouldAppend = false
		}
	}
	if shouldAppend {
		m.messages = append(m.messages, Message{Role: "client", Content: chatMsg})
	}
}

// switchModel switches to a different model in PROJECT mode
func (m *chatModel) switchModel(newModel string) {
	oldModel := m.currentModel
	m.currentModel = newModel

	// Update the mode context
	m.projectModeCtx.Model = newModel

	// Update global chat context
	chatCtx.Model = newModel

	// Save session context (preserves session ID)
	_ = writeSessionContext(chatCtx, chatCtx.SessionID)

	// Get model info for display
	modelInfo := m.getModelInfo(newModel)
	var modelDesc string
	if modelInfo.Description != "" {
		modelDesc = fmt.Sprintf("\n%s", modelInfo.Description)
	}

	// Add switch notification to chat
	msg := fmt.Sprintf("🔄 Switched model: %s → %s%s",
		oldModel,
		newModel,
		modelDesc)
	m.messages = append(m.messages, Message{Role: "client", Content: msg})
}

// getNextModel returns the next model in the list (cycles)
func (m *chatModel) getNextModel() string {
	if len(m.availableModels) == 0 {
		return m.currentModel
	}
	for i, model := range m.availableModels {
		if model.Name == m.currentModel {
			nextIdx := (i + 1) % len(m.availableModels)
			return m.availableModels[nextIdx].Name
		}
	}
	return m.currentModel
}

// isValidModel checks if a model name exists in available models
func (m *chatModel) isValidModel(name string) bool {
	for _, model := range m.availableModels {
		if model.Name == name {
			return true
		}
	}
	return false
}

// getModelInfo returns model info for a given name
func (m *chatModel) getModelInfo(name string) ModelInfo {
	for _, model := range m.availableModels {
		if model.Name == name {
			return model
		}
	}
	// Fallback to preserve label when model details aren't found
	return ModelInfo{Name: name}
}

// Database/Strategy switching methods
func (m *chatModel) switchDatabase(newDatabase string) {
	oldDatabase := m.currentDatabase
	m.currentDatabase = newDatabase

	// Update the mode context
	m.projectModeCtx.Database = newDatabase

	// Check if current strategy is valid for new database before resetting
	oldStrategy := m.currentStrategy
	strategyValidForNewDB := false

	noStrategiesForDB := false
	if m.availableDatabases != nil {
		for _, db := range m.availableDatabases.Databases {
			if db.Name == newDatabase {
				if len(db.RetrievalStrategies) == 0 {
					noStrategiesForDB = true
				}
				// Check if current strategy exists in new database
				if oldStrategy != "" {
					for _, strategy := range db.RetrievalStrategies {
						if strategy.Name == oldStrategy {
							strategyValidForNewDB = true
							break
						}
					}
				}

				// If old strategy isn't valid for new database, find a new one
				if !strategyValidForNewDB {
					m.currentStrategy = ""
					m.projectModeCtx.RetrievalStrategy = ""

					// Find default strategy for this database
					for _, strategy := range db.RetrievalStrategies {
						if strategy.IsDefault {
							m.currentStrategy = strategy.Name
							m.projectModeCtx.RetrievalStrategy = strategy.Name
							break
						}
					}
					// If no default, use first strategy
					if m.currentStrategy == "" && len(db.RetrievalStrategies) > 0 {
						m.currentStrategy = db.RetrievalStrategies[0].Name
						m.projectModeCtx.RetrievalStrategy = m.currentStrategy
					}
				}
				break
			}
		}
	}

	m.messages = append(m.messages, Message{
		Role:    "client",
		Content: fmt.Sprintf("Switched from database '%s' to '%s' with strategy '%s'", oldDatabase, newDatabase, m.currentStrategy),
	})

	// Notify if the selected database has no retrieval strategies
	if noStrategiesForDB {
		m.messages = append(m.messages, Message{Role: "client", Content: fmt.Sprintf("Database '%s' has no retrieval strategies configured.", newDatabase)})
	}
}

func (m *chatModel) switchStrategy(newStrategy string) {
	oldStrategy := m.currentStrategy
	m.currentStrategy = newStrategy

	// Update the mode context
	m.projectModeCtx.RetrievalStrategy = newStrategy

	m.messages = append(m.messages, Message{
		Role:    "client",
		Content: fmt.Sprintf("Switched retrieval strategy from '%s' to '%s'", oldStrategy, newStrategy),
	})
}

func (m *chatModel) isValidDatabase(name string) bool {
	if m.availableDatabases == nil {
		return false
	}
	for _, db := range m.availableDatabases.Databases {
		if db.Name == name {
			return true
		}
	}
	return false
}

func (m *chatModel) isValidStrategy(name string) bool {
	if m.availableDatabases == nil {
		return false
	}
	// Check if strategy exists in current database
	for _, db := range m.availableDatabases.Databases {
		if db.Name == m.currentDatabase {
			for _, strategy := range db.RetrievalStrategies {
				if strategy.Name == name {
					return true
				}
			}
			return false
		}
	}
	return false
}

func (m chatModel) Init() tea.Cmd {
	// Kick off spinner and designer background start
	startDesigner := func() tea.Msg {
		// Determine preferred port and forced
		pref := 7724
		forced := false
		if designerPreferredPort > 0 {
			pref = designerPreferredPort
			forced = designerForced
		} else if v := strings.TrimSpace(os.Getenv("LF_DESIGNER_PORT")); v != "" {
			if p, err := strconv.Atoi(v); err == nil && p > 0 {
				pref = p
				forced = true
			}
		}
		url, err := StartDesignerInBackground(context.Background(), DesignerLaunchOptions{PreferredPort: pref, Forced: forced})
		if err != nil {
			return designerErrorMsg{err: err}
		}
		return designerReadyMsg{url: url}
	}
	return tea.Batch(m.spin.Tick, startDesigner, updateServerHealthCmd(m))
}

func updateServerHealthCmd(m chatModel) tea.Cmd {
	return func() tea.Msg {
		health, _ := checkServerHealth(serverURL)
		return serverHealthMsg{health: health}
	}
}

func (m chatModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var (
		tiCmd tea.Cmd
		vpCmd tea.Cmd
		cmd   tea.Cmd
		cmds  []tea.Cmd
	)

	// Route messages to quick menu (it ignores most when inactive)
	m.quickMenu, cmd = m.quickMenu.Update(msg)
	if cmd != nil {
		cmds = append(cmds, cmd)
	}

	// Toggle textarea focus based on overlay activity and lock input when active
	if m.quickMenu.IsActive() && !m.menuActive {
		m.textarea.Blur()
		m.menuActive = true
	}
	if !m.quickMenu.IsActive() && m.menuActive {
		m.textarea.Focus()
		m.menuActive = false
	}

	// Only update textarea when menu is not active
	if !m.quickMenu.IsActive() {
		m.textarea, tiCmd = m.textarea.Update(msg)
	}

	m.viewport, vpCmd = m.viewport.Update(msg)

	// Route all messages to toast
	m.toast, cmd = m.toast.Update(msg)
	if cmd != nil {
		cmds = append(cmds, cmd)
	}

	// Forward all messages to the spinner so it processes its own TickMsgs
	m.spin, cmd = m.spin.Update(msg)

	cmds = append(cmds, vpCmd, tiCmd, cmd)

	headerHeight := lipgloss.Height(renderInfoBar(m))
	footerHeight := lipgloss.Height(renderChatInput(m))

	if m.serverHealth != nil && m.serverHealth.Status != "healthy" {
		logDebug(fmt.Sprintf("Checking latest server health. Last: %v", m.serverHealth))
		cmds = append(cmds, updateServerHealthCmd(m))
	}

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		// CRITICAL: Prevent negative viewport height that causes slice bounds panic
		// Note: headerHeight is now predictable (1 line) due to compact status bar design
		newHeight := msg.Height - footerHeight - headerHeight
		if newHeight < 1 {
			newHeight = 1 // Minimum viable height to prevent panic
		}

		m.viewport.Width = msg.Width
		m.viewport.Height = newHeight // Now guaranteed positive

		// Also protect textarea width calculation
		newWidth := msg.Width - 2
		if newWidth < 10 {
			newWidth = 10
		}
		m.textarea.SetWidth(newWidth)
		m.width = msg.Width
		m.termHeight = msg.Height

	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c":
			m.status = "👋 You have left the pasture. Safe travels, little llama!"
			return m, tea.Quit

		case "ctrl+t":
			// If overlay is active, let overlay handle ctrl+t and return accumulated cmds
			if m.quickMenu.IsActive() {
				return m, tea.Batch(cmds...)
			}
			// Toggle between modes
			newMode := ModeProject
			if m.currentMode == ModeProject {
				newMode = ModeDev
			}
			m.switchMode(newMode)
			m.refreshViewportBottom()
			return m, nil

		case "ctrl+k":
			// Cycle models (PROJECT mode only)
			if m.currentMode == ModeProject && len(m.availableModels) > 0 {
				nextModel := m.getNextModel()
				m.switchModel(nextModel)
				m.refreshViewportBottom()
			}
			return m, nil

		case "tab":
			// If overlay already active, let it handle Tab for tab switching; otherwise ignore
			if m.quickMenu.IsActive() {
				return m, tea.Batch(cmds...)
			}
			return m, nil

			// removed cmd+r menu opener

		case "esc":
			// No-op here; overlay handles its own ESC
			return m, tea.Batch(cmds...)

		case "up":
			// If overlay is active, let it handle navigation
			if m.quickMenu.IsActive() {
				return m, tea.Batch(cmds...)
			}
			// Navigate history
			logDebug(fmt.Sprintf("Up arrow pressed. Current history: %+v", m.history))
			if m.histIndex > 0 {
				m.histIndex--
				m.textarea.SetValue(m.history[m.histIndex])
				m.textarea.CursorEnd()
			}

		case "down":
			// If overlay is active, let it handle navigation
			if m.quickMenu.IsActive() {
				return m, tea.Batch(cmds...)
			}
			// Navigate history
			if m.histIndex < len(m.history)-1 {
				m.histIndex++
				m.textarea.SetValue(m.history[m.histIndex])
				m.textarea.CursorEnd()
			} else {
				m.histIndex = len(m.history)
				m.textarea.SetValue("")
			}
		case "enter":
			// If overlay is active, let it handle selection and return any commands it emitted
			if m.quickMenu.IsActive() {
				return m, tea.Batch(cmds...)
			}
			m.err = nil
			msg := strings.TrimSpace(m.textarea.Value())
			if msg == "" || m.thinking {
				break
			}

			lower := strings.ToLower(msg)
			// Slash commands
			if strings.HasPrefix(lower, "/") {
				fields := strings.Fields(lower)
				cmd := fields[0]
				switch cmd {
				case "/help":
					m.messages = append(m.messages, Message{Role: "client", Content: "Commands:\n  /help - Show this help\n  /mode [dev|project] - Switch mode\n  /model [name] - Switch model (PROJECT mode)\n  /database [name] - Switch RAG database (PROJECT mode)\n  /strategy [name] - Switch retrieval strategy (PROJECT mode)\n  /clear - Clear conversation\n  /launch designer - Open designer\n  /menu - Open Quick Menu\n  /exit - Exit\n\nHotkeys:\n  Ctrl+T - Toggle DEV/PROJECT mode\n  Ctrl+K - Cycle models"})
					m.textarea.SetValue("")
				case "/mode":
					if len(fields) < 2 {
						m.messages = append(m.messages, Message{Role: "client", Content: "Usage: /mode [dev|project]"})
						m.textarea.SetValue("")
						return m, nil
					}
					modeArg := fields[1]
					var newMode ChatMode
					switch modeArg {
					case "dev":
						newMode = ModeDev
					case "project":
						newMode = ModeProject
					default:
						m.messages = append(m.messages, Message{Role: "client", Content: "Unknown mode. Use: /mode [dev|project]"})
						m.textarea.SetValue("")
						return m, nil
					}
					if newMode == m.currentMode {
						m.messages = append(m.messages, Message{Role: "client", Content: fmt.Sprintf("Already in %s mode", modeArg)})
						m.textarea.SetValue("")
						return m, nil
					}
					m.switchMode(newMode)
					m.textarea.SetValue("")
					m.refreshViewportBottom()
				case "/model":
					if m.currentMode != ModeProject {
						m.messages = append(m.messages, Message{
							Role:    "client",
							Content: "Model switching only available in PROJECT mode. Use Ctrl+T to switch.",
						})
						m.textarea.SetValue("")
						break
					}

					if len(fields) < 2 {
						// Show current model and available models
						var msg strings.Builder
						msg.WriteString(fmt.Sprintf("Current model: %s\n\nAvailable models:", m.currentModel))
						for _, model := range m.availableModels {
							marker := ""
							if model.Name == m.currentModel {
								marker = " (current)"
							}
							msg.WriteString(fmt.Sprintf("\n  • %s - %s%s", model.Name, model.Description, marker))
						}
						msg.WriteString("\n\nUsage: /model <name> or press Ctrl+K to cycle")
						m.messages = append(m.messages, Message{Role: "client", Content: msg.String()})
						m.textarea.SetValue("")
						break
					}

					modelName := fields[1]
					if !m.isValidModel(modelName) {
						m.messages = append(m.messages, Message{
							Role:    "client",
							Content: fmt.Sprintf("Unknown model '%s'. Type '/model' to see available models.", modelName),
						})
						m.textarea.SetValue("")
						break
					}

					m.switchModel(modelName)
					m.textarea.SetValue("")
					m.refreshViewportBottom()
				case "/launch":
					if len(fields) < 2 {
						m.messages = append(m.messages, Message{Role: "client", Content: "Usage: /launch <component>. Components: designer"})
						m.textarea.SetValue("")
						break
					}
					target := fields[1]
					if target != "designer" {
						m.messages = append(m.messages, Message{Role: "client", Content: fmt.Sprintf("Unknown component '%s'. Try: /launch designer", target)})
						m.textarea.SetValue("")
						break
					}
					if strings.TrimSpace(m.designerURL) == "" || m.designerStatus != "ready" {
						m.messages = append(m.messages, Message{Role: "client", Content: "Designer is not running yet."})
						m.textarea.SetValue("")
						break
					}
					m.textarea.SetValue("")
					return m, openURL(m.designerURL)
				case "/exit", "/quit":
					m.status = "👋 You have left the pasture. Safe travels, little llama!"
					return m, tea.Quit
				case "/clear":
					// Get current mode context
					ctx := m.getCurrentModeContext()

					// Delete server-side session for current mode
					if ctx.SessionID != "" {
						// Determine namespace/project for current mode
						var namespace, projectID string
						if m.currentMode == ModeDev {
							namespace = "llamafarm"
							projectID = "project_seed"
						} else if m.projectInfo != nil {
							namespace = m.projectInfo.Namespace
							projectID = m.projectInfo.Project
						}

						if namespace != "" && projectID != "" {
							deleteURL := fmt.Sprintf("%s/v1/projects/%s/%s/chat/sessions/%s",
								strings.TrimSuffix(chatCtx.ServerURL, "/"),
								namespace,
								projectID,
								ctx.SessionID)
							req, err := http.NewRequest("DELETE", deleteURL, nil)
							if err == nil {
								resp, err := chatCtx.HTTPClient.Do(req)
								if err == nil {
									resp.Body.Close()
									logDebug(fmt.Sprintf("Deleted server session %s", ctx.SessionID))
								}
							}
						}

						// Generate new session ID for current mode
						ctx.SessionID = uuid.New().String()

						// Update global chatCtx session ID
						chatCtx.SessionID = ctx.SessionID

						// Save new session context
						_ = writeSessionContext(chatCtx, ctx.SessionID)
						logDebug(fmt.Sprintf("Created new dev mode session ID: %s", ctx.SessionID))
					}

					// Clear local state for current mode
					ctx.Messages = []Message{{Role: "client", Content: "Session cleared. New session started."}}
					ctx.History = []string{}

					// Update model state
					m.transcript = ""
					m.messages = ctx.Messages
					m.history = ctx.History
					m.textarea.SetValue("")
					m.setViewportContent()
					m.thinking = false
					m.printing = false
				case "/menu":
					// Back-compat: open the new overlay and hint about Tab
					m.quickMenu.Open()
					if m.termHeight > 0 {
						var setSize tea.Cmd
						m.quickMenu, setSize = m.quickMenu.Update(tea.WindowSizeMsg{Width: m.width, Height: m.termHeight})
						if setSize != nil {
							// ignore setSize here; the menu will render with size in View()
						}
					}
					m.messages = append(m.messages, Message{Role: "client", Content: "Opening Quick Menu."})
					m.textarea.SetValue("")
					return m, nil
				case "/database":
					if m.currentMode != ModeProject {
						m.messages = append(m.messages, Message{
							Role:    "client",
							Content: "Database switching only available in PROJECT mode. Use Ctrl+T to switch.",
						})
						m.textarea.SetValue("")
						break
					}

					if len(fields) < 2 {
						// Show available databases
						var msg strings.Builder
						msg.WriteString("Current database: ")
						if m.currentDatabase != "" {
							msg.WriteString(m.currentDatabase)
						} else {
							msg.WriteString("(none)")
						}
						msg.WriteString("\n\nAvailable databases:")

						if m.availableDatabases != nil && len(m.availableDatabases.Databases) > 0 {
							for _, db := range m.availableDatabases.Databases {
								marker := ""
								if db.Name == m.currentDatabase {
									marker = " (current)"
								} else if db.IsDefault {
									marker = " (default)"
								}
								msg.WriteString(fmt.Sprintf("\n  • %s [%s]%s", db.Name, db.Type, marker))
							}
							msg.WriteString("\n\nUsage: /database <name> or press Tab to open Quick Menu")
						} else {
							msg.WriteString("\n  No databases configured")
						}

						m.messages = append(m.messages, Message{Role: "client", Content: msg.String()})
						m.textarea.SetValue("")
						break
					}

					dbName := fields[1]
					if !m.isValidDatabase(dbName) {
						m.messages = append(m.messages, Message{
							Role:    "client",
							Content: fmt.Sprintf("Unknown database '%s'. Type '/database' to see available databases.", dbName),
						})
						m.textarea.SetValue("")
						break
					}

					m.switchDatabase(dbName)
					m.textarea.SetValue("")
					m.refreshViewportBottom()

				case "/strategy":
					if m.currentMode != ModeProject {
						m.messages = append(m.messages, Message{
							Role:    "client",
							Content: "Strategy switching only available in PROJECT mode. Use Ctrl+T to switch.",
						})
						m.textarea.SetValue("")
						break
					}

					if len(fields) < 2 {
						// Show available strategies for current database
						var msg strings.Builder
						msg.WriteString("Current strategy: ")
						if m.currentStrategy != "" {
							msg.WriteString(m.currentStrategy)
						} else {
							msg.WriteString("(none)")
						}
						msg.WriteString(fmt.Sprintf("\nDatabase: %s", m.currentDatabase))
						msg.WriteString("\n\nAvailable strategies:")

						if m.availableDatabases != nil {
							for _, db := range m.availableDatabases.Databases {
								if db.Name == m.currentDatabase {
									if len(db.RetrievalStrategies) > 0 {
										for _, strategy := range db.RetrievalStrategies {
											marker := ""
											if strategy.Name == m.currentStrategy {
												marker = " (current)"
											} else if strategy.IsDefault {
												marker = " (default)"
											}
											msg.WriteString(fmt.Sprintf("\n  • %s [%s]%s", strategy.Name, strategy.Type, marker))
										}
										msg.WriteString("\n\nUsage: /strategy <name> or press Tab to open Quick Menu")
									} else {
										msg.WriteString("\n  No strategies configured for this database")
									}
									break
								}
							}
						}

						m.messages = append(m.messages, Message{Role: "client", Content: msg.String()})
						m.textarea.SetValue("")
						break
					}

					strategyName := fields[1]
					if !m.isValidStrategy(strategyName) {
						m.messages = append(m.messages, Message{
							Role:    "client",
							Content: fmt.Sprintf("Unknown strategy '%s' for database '%s'. Type '/strategy' to see available strategies.", strategyName, m.currentDatabase),
						})
						m.textarea.SetValue("")
						break
					}

					m.switchStrategy(strategyName)
					m.textarea.SetValue("")
					m.refreshViewportBottom()

				default:
					m.messages = append(m.messages, Message{Role: "client", Content: fmt.Sprintf("Unknown command '%s'. All commands must start with '/'. Type '/help' for available commands.", cmd)})
					m.textarea.SetValue("")
				}
				return m, nil
			}

			m.history = append(m.history, msg)
			m.histIndex = len(m.history)
			m.messages = append(m.messages, Message{Role: "user", Content: msg})
			m.textarea.SetValue("")
			m.thinking = true
			m.printing = true
			// Update chatCtx with current selections (PROJECT mode)
			if m.currentMode == ModeProject {
				if m.currentModel != "" {
					chatCtx.Model = m.currentModel
				}
				if m.currentDatabase != "" {
					chatCtx.RAGDatabase = m.currentDatabase
					chatCtx.RAGEnabled = true
				}
				if m.currentStrategy != "" {
					chatCtx.RAGRetrievalStrategy = m.currentStrategy
				}
			}
			// Start channel-based streaming - important for showing progress
			chunks, errs, _ := startChatStream(m.messages, chatCtx)
			ch := make(chan tea.Msg, 32)
			m.streamCh = ch
			go func() {
				var builder strings.Builder
				for {
					select {
					case s, ok := <-chunks:
						logDebug(fmt.Sprintf("STREAM CHUNK: %v", s))
						if !ok {
							logDebug(fmt.Sprintf("CHANNEL CLOSED: %v", builder.String()))
							ch <- responseMsg{content: builder.String()}
							ch <- streamDone{}
							close(ch)
							return
						}
						builder.WriteString(s)
						ch <- responseMsg{content: builder.String()}
					case e, ok := <-errs:
						if ok && e != nil {
							logDebug(fmt.Sprintf("STREAM ERROR: %v", e))
							ch <- errorMsg{err: e}
						}
					}
				}
			}()
			cmds = append(cmds, listen(m.streamCh), thinkingCmd())
		}

	case responseMsg:
		if m.err != nil {
			m.err = nil
			break
		}

		logDebug(fmt.Sprintf("RESPONSE MSG: %v", msg.content))
		m.thinking = false
		m.printing = true
		if len(m.messages) == 0 || (len(m.messages) > 0 && m.messages[len(m.messages)-1].Role != "assistant") {
			m.messages = append(m.messages, Message{Role: "assistant", Content: msg.content})
		} else {
			// Update last assistant line
			if len(m.messages) > 0 {
				m.messages[len(m.messages)-1] = Message{Role: "assistant", Content: msg.content}
			} else {
				m.messages = append(m.messages, Message{Role: "assistant", Content: msg.content})
			}
		}

		m.setViewportContent()

		if m.streamCh != nil {
			cmds = append(cmds, listen(m.streamCh))
		}

	case errorMsg:
		m.thinking = false
		m.err = msg.err
		m.messages = append(m.messages, Message{Role: "error", Content: fmt.Sprintf("Error: %v", msg.err)})
		if m.streamCh != nil {
			cmds = append(cmds, listen(m.streamCh))
		}

	case tickMsg:
		if m.thinking {
			m.thinkFrame = (m.thinkFrame + 1) % 3
			cmds = append(cmds, thinkingCmd())
		}

	case streamDone:
		if len(m.messages) > 0 {
			logDebug(fmt.Sprintf("STREAM DONE: %v", m.messages[len(m.messages)-1]))
		} else {
			logDebug("STREAM DONE: no messages")
		}
		m.printing = false
		m.streamCh = nil

	case designerReadyMsg:
		m.designerStatus = "ready"
		m.designerURL = msg.url

	case designerErrorMsg:
		m.designerStatus = fmt.Sprintf("error: %v", msg.err)

	case serverHealthMsg:
		// Delegate to controller to update state and emit a unified StateUpdateMsg
		return m, m.controller.UpdateServerHealth(msg.health)

	case StateUpdateMsg:
		// Apply shared state changes from controller
		m.serverHealth = msg.NewState.ServerHealth
		if msg.NewState.CurrentDatabase != "" {
			m.currentDatabase = msg.NewState.CurrentDatabase
		}
		if msg.NewState.CurrentStrategy != "" {
			m.currentStrategy = msg.NewState.CurrentStrategy
		}
		// Update Help tab summary when health updates
		if m.serverHealth != nil {
			m.quickMenu.RAGHealthSummary = formatRAGHealthSummary(m.serverHealth)
		}
		if strings.TrimSpace(msg.Notice) != "" {
			m.messages = append(m.messages, Message{Role: "client", Content: msg.Notice})
		}

	case uitk.SwitchModeMsg:
		// Toggle between DEV and PROJECT based on devMode flag
		if msg.DevMode {
			if m.currentMode != ModeDev {
				m.switchMode(ModeDev)
				m.refreshViewportBottom()
			}
		} else {
			if m.currentMode != ModeProject {
				m.switchMode(ModeProject)
				m.refreshViewportBottom()
			}
		}

	case uitk.SwitchDatabaseMsg:
		if m.currentMode == ModeProject && msg.DatabaseName != "" {
			return m, m.controller.SwitchDatabase(msg.DatabaseName, m.availableDatabases)
		}

	case uitk.SwitchModelMsg:
		if m.currentMode == ModeProject && msg.ModelName != "" {
			m.switchModel(msg.ModelName)
			m.refreshViewportBottom()
		}

	case uitk.SwitchProjectMsg:
		// TODO: implement real project switch; for now, reflect in UI only
		m.messages = append(m.messages, Message{Role: "client", Content: fmt.Sprintf("Selected project: %s/%s", msg.Namespace, msg.ProjectName)})

	case uitk.SwitchStrategyMsg:
		if m.currentMode == ModeProject && msg.StrategyName != "" {
			m.switchStrategy(msg.StrategyName)
			m.refreshViewportBottom()
		}

	case uitk.CycleModelMsg:
		// Ensure PROJECT mode first
		if m.currentMode != ModeProject {
			m.switchMode(ModeProject)
			m.refreshViewportBottom()
		}
		if len(m.availableModels) > 0 {
			next := m.getNextModel()
			old := m.currentModel
			m.switchModel(next)
			m.refreshViewportBottom()
			cmds = append(cmds, func() tea.Msg { return uitk.ShowToastMsg{Message: fmt.Sprintf("Switched model: %s → %s", old, next)} })
		} else {
			cmds = append(cmds, func() tea.Msg { return uitk.ShowToastMsg{Message: "No models available to cycle"} })
		}

	case uitk.ExecuteCommandMsg:
		// For now, just echo the command and toast; future: wire to runner
		m.messages = append(m.messages, Message{Role: "client", Content: fmt.Sprintf("$ %s", msg.Command)})
		cmds = append(cmds, tea.Printf("Executing: %s", msg.Command))
		// Show toast confirmation
		cmds = append(cmds, func() tea.Msg { return uitk.ShowToastMsg{Message: "Running: " + msg.Command} })
		m.quickMenu.Close()

		if m.serverHealth != nil && m.serverHealth.Status != "healthy" {
			// Schedule a non-blocking re-check after 5 seconds
			cmds = append(cmds, tea.Tick(5*time.Second, func(time.Time) tea.Msg {
				return updateServerHealthCmd(m)()
			}))
		}

	case uitk.InsertChatInputMsg:
		if msg.EnsureDev && m.currentMode != ModeDev {
			m.switchMode(ModeDev)
			m.refreshViewportBottom()
		}
		if msg.EnsureProject && m.currentMode != ModeProject {
			m.switchMode(ModeProject)
			m.refreshViewportBottom()
		}
		m.textarea.SetValue(msg.Text)
		if msg.AutoSend {
			// Emulate Enter key handling: add message, clear input, and trigger processing
			m.err = nil
			val := strings.TrimSpace(msg.Text)
			if val != "" {
				m.messages = append(m.messages, Message{Role: "client", Content: val})
				m.textarea.SetValue("")
				return m, func() tea.Msg { return tea.KeyMsg{Type: tea.KeyEnter} }
			}
		}

	case TUIMessageMsg:
		// Handle output messages routed through the messaging API
		formattedContent := FormatMessage(msg.Message)

		if msg.Message.Type == ProgressMessage {
			// For progress messages, find and remove the most recent progress message,
			// then add the updated progress message at the bottom (most recent position)
			// This keeps progress updates always visible at the bottom of the chat
			foundProgressIdx := -1

			// Search backwards through all messages to find the most recent progress message
			for i := len(m.messages) - 1; i >= 0; i-- {
				if m.messages[i].Role == "client" && strings.HasPrefix(m.messages[i].Content, "🔄") {
					foundProgressIdx = i
					break
				}
			}

			if foundProgressIdx >= 0 {
				// Remove the old progress message by slicing it out
				m.messages = append(m.messages[:foundProgressIdx], m.messages[foundProgressIdx+1:]...)
			}

			// Always add the new progress message at the bottom (most recent position)
			m.messages = append(m.messages, Message{Role: "client", Content: formattedContent})
		} else {
			// For non-progress messages, add normally
			m.messages = append(m.messages, Message{Role: "client", Content: formattedContent})
		}
	}

	m.transcript = computeTranscript(m)
	m.refreshViewportBottom()

	return m, tea.Batch(cmds...)
}

func listen(ch <-chan tea.Msg) tea.Cmd {
	return func() tea.Msg {
		msg, ok := <-ch
		logDebug(fmt.Sprintf("LISTEN MSG: %v", msg))
		if !ok {
			fmt.Println("LISTEN DONE")
			return streamDone{}
		}
		return msg
	}
}

func renderServerStatusProblems(health *HealthPayload) string {
	var b strings.Builder

	if health == nil {
		return ""
	}

	prettyPrintHealthProblems(&b, *health)

	return b.String()
}

func computeTranscript(m chatModel) string {
	var b strings.Builder

	key := computeTranscriptKey(m)
	if lastTranscriptKey == key {
		b.WriteString(m.transcript)
	} else {
		baseStyle := lipgloss.NewStyle()
		for _, message := range m.messages {
			var line string
			switch message.Role {
			case "assistant":
				// Render Markdown content with ANSI styling
				renderedContent := renderMarkdown(message.Content, m.width-len(m.getAssistantLabel())-4)
				// Don't use lipgloss.Render on the rendered content to preserve ANSI codes
				labelStyle := baseStyle.Foreground(lipgloss.Color("11"))
				line = labelStyle.Render(m.getAssistantLabel()) + " " + renderedContent + "\n"
			case "user":
				style := baseStyle.Foreground(lipgloss.Color("#ccc"))
				line = style.Bold(true).Render("> ") + style.Render(message.Content)
			case "error":
				line = baseStyle.Foreground(lipgloss.Color("9")).Render(message.Content)
			case "client":
				line = baseStyle.Foreground(lipgloss.Color("#666666")).Render(message.Content)
			}

			b.WriteString(line + "\n")
		}
		lastTranscriptKey = key
	}

	return b.String()
}

func computeTranscriptKey(m chatModel) string {
	h := fnv.New64a()
	if len(m.messages) == 0 {
		return "empty"
	}
	msg := m.messages[len(m.messages)-1]
	io.WriteString(h, msg.Role)
	io.WriteString(h, msg.Content)
	return fmt.Sprintf("%x", h.Sum64())
}

// formatRAGHealthSummary builds a compact health line from the server health payload.
// It uses the existing /health API response and extracts:
// - RAG overall status
// - Task system status
// - Memory usage in MB
func formatRAGHealthSummary(h *HealthPayload) string {
	if h == nil {
		return "RAG: UNKNOWN  |  Task System: UNKNOWN  |  Memory: N/A"
	}
	rag := findRAGComponent(h)
	if rag == nil {
		return "RAG: UNKNOWN  |  Task System: UNKNOWN  |  Memory: N/A"
	}

	ragStatus := strings.ToUpper(strings.TrimSpace(rag.Status))

	taskStatus := "UNKNOWN"
	memoryText := "N/A"

	if rag.Details != nil {
		// Extract task system status
		if checksRaw, ok := rag.Details["checks"]; ok {
			if checks, ok := checksRaw.(map[string]interface{}); ok {
				if tsRaw, ok := checks["task_system"]; ok {
					if ts, ok := tsRaw.(map[string]interface{}); ok {
						if st, ok := ts["status"].(string); ok && strings.TrimSpace(st) != "" {
							taskStatus = strings.ToUpper(st)
						}
					}
				}
			}
		}
		// Extract memory usage (MB)
		if metricsRaw, ok := rag.Details["metrics"]; ok {
			if metrics, ok := metricsRaw.(map[string]interface{}); ok {
				if memVal, ok := metrics["memory_mb"]; ok {
					switch v := memVal.(type) {
					case float64:
						memoryText = fmt.Sprintf("%.0f MB", v)
					case float32:
						memoryText = fmt.Sprintf("%.0f MB", v)
					case int:
						memoryText = fmt.Sprintf("%d MB", v)
					case int64:
						memoryText = fmt.Sprintf("%d MB", v)
					case string:
						memoryText = v + " MB"
					default:
						memoryText = fmt.Sprintf("%v MB", v)
					}
				}
			}
		}
	}

	return fmt.Sprintf("RAG: %s  |  Task System: %s  |  Memory: %s", ragStatus, taskStatus, memoryText)
}

func renderChatContent(m chatModel) string {
	var b strings.Builder

	b.WriteString(m.transcript)

	if m.thinking {
		dots := m.thinkFrame + 1
		thinkingText := m.getAssistantLabel() + " " + m.spin.View() + "Thinking" + strings.Repeat(".", dots)
		wrappedThinking := lipgloss.NewStyle().Foreground(lipgloss.Color("11")).Width(m.width - 2).Render(thinkingText)
		b.WriteString(wrappedThinking + gap)
	}

	// Overlay is drawn from View() so it stays on top consistently

	return b.String()
}

// setViewportContent updates the viewport with the current chat rendering.
func (m *chatModel) setViewportContent() {
	m.viewport.SetContent(lipgloss.NewStyle().Width(m.viewport.Width).Render(renderChatContent(*m)))
}

// refreshViewportBottom updates the viewport and scrolls to the bottom.
func (m *chatModel) refreshViewportBottom() {
	m.setViewportContent()
	m.viewport.GotoBottom()
}

func renderChatInput(m chatModel) string {
	var b strings.Builder

	b.WriteString(gap)

	cbStyle := lipgloss.NewStyle().
		MarginBottom(1).
		BorderStyle(lipgloss.NormalBorder()).
		BorderForeground(lipgloss.Color("63"))

	b.WriteString(cbStyle.Render(m.textarea.View()))

	// Combined helper text with mode-specific shortcut
	var modeHint string
	if m.currentMode == ModeDev {
		modeHint = "Ctrl+T: test project"
	} else {
		modeHint = "Ctrl+T: dev help | Ctrl+K: cycle models"
	}
	helpText := fmt.Sprintf("/help for commands | Up/Down: history | %s", modeHint)

	b.WriteString("\n")
	wrappedHelp := lipgloss.NewStyle().Faint(true).Width(m.width - 2).Render(helpText)
	b.WriteString(wrappedHelp)
	b.WriteString("\n")

	return b.String()
}

func renderInfoBar(m chatModel) string {
	// Mode-specific colors and emojis
	var modeEmoji, modeLabel, bgColor string
	var currentSessionID string

	if m.currentMode == ModeDev {
		modeEmoji = "🦙"
		modeLabel = "DEV MODE"
		bgColor = "#28a745" // Green
		currentSessionID = m.devModeCtx.SessionID
	} else {
		modeEmoji = "🎯"
		modeLabel = "PROJECT MODE"
		bgColor = "#027ffd" // Blue
		currentSessionID = m.projectModeCtx.SessionID
	}

	// Project info
	var project string
	if m.currentMode == ModeDev {
		project = "llamafarm/project_seed"
	} else if m.projectInfo != nil {
		project = fmt.Sprintf("%s/%s", m.projectInfo.Namespace, m.projectInfo.Project)
	} else {
		project = "unknown/unknown"
	}

	// Model info (PROJECT MODE only)
	var modelInfo string
	if m.currentMode == ModeProject && m.currentModel != "" {
		// Find model details
		var modelDetails string
		for _, model := range m.availableModels {
			if model.Name == m.currentModel {
				modelDetails = model.Model
				break
			}
		}
		if modelDetails != "" {
			modelInfo = fmt.Sprintf(" | Model: %s (%s)", m.currentModel, modelDetails)
		} else {
			modelInfo = fmt.Sprintf(" | Model: %s", m.currentModel)
		}
	}

	// Session info (truncate to 8 chars for compactness)
	var session string
	if currentSessionID != "" {
		if len(currentSessionID) > 8 {
			session = currentSessionID[:8]
		} else {
			session = currentSessionID
		}
	} else {
		session = "none"
	}

	// Server status (just icon + simple host)
	statusIcon := iconForStatus(func() string {
		if m.serverHealth != nil {
			return m.serverHealth.Status
		}
		return "degraded"
	}())

	// Extract just the host from serverURL for compactness
	serverHost := serverURL
	if strings.HasPrefix(serverHost, "http://") {
		serverHost = strings.TrimPrefix(serverHost, "http://")
	} else if strings.HasPrefix(serverHost, "https://") {
		serverHost = strings.TrimPrefix(serverHost, "https://")
	}

	// Build compact status line with mode indicator
	statusLine := fmt.Sprintf("%s %s: %s%s | Session: %s | Status: %s | %s",
		modeEmoji, modeLabel, project, modelInfo, session, statusIcon, serverHost)

	// Apply single-line styling with mode-specific background color
	style := lipgloss.NewStyle().
		Width(m.width).
		Background(lipgloss.Color(bgColor)).
		Foreground(lipgloss.Color("#ffffff")).
		PaddingLeft(1).
		PaddingRight(1)

	// Truncate if too long for terminal width
	if lipgloss.Width(statusLine) > m.width-2 { // -2 for padding
		maxLen := m.width - 5 // -5 for padding and "..."
		if maxLen > 0 {
			statusLine = statusLine[:maxLen] + "..."
		}
	}

	return style.Render(statusLine)
}

// removed: old bottom menu panel

func (m chatModel) View() string {
	var b strings.Builder
	// Dim the background when the menu is active
	if m.quickMenu.IsActive() {
		dim := lipgloss.NewStyle().Faint(true)
		b.WriteString(dim.Render(m.viewport.View()))
	} else {
		b.WriteString(m.viewport.View())
	}

	// When menu is active, draw overlay sized to terminal
	if m.quickMenu.IsActive() {
		// Give the overlay a consistent height by passing terminal height
		m.quickMenu, _ = m.quickMenu.Update(tea.WindowSizeMsg{Width: m.width, Height: m.termHeight})
		b.WriteString("\n")
		b.WriteString(m.quickMenu.View())
	}

	if m.quickMenu.IsActive() {
		// Dim the input area and prevent cursor from showing
		dim := lipgloss.NewStyle().Faint(true)
		// Render input without focus cursor
		shadow := m
		shadow.textarea.Blur()
		b.WriteString(dim.Render(renderChatInput(shadow)))
	} else {
		b.WriteString(renderChatInput(m))
	}
	// Always draw the status bar at the very bottom (no dimming)
	b.WriteString(renderInfoBar(m))

	// Toast on top-right
	if v := m.toast.View(); v != "" {
		b.WriteString("\n")
		b.WriteString(v)
	}

	return b.String()
}

func thinkingCmd() tea.Cmd {
	return tea.Tick(250*time.Millisecond, func(time.Time) tea.Msg { return tickMsg{} })
}

func openURL(url string) tea.Cmd {
	return func() tea.Msg {
		var cmd *exec.Cmd
		switch runtime.GOOS {
		case "darwin":
			cmd = exec.Command("open", url)
		case "linux":
			cmd = exec.Command("xdg-open", url)
		case "windows":
			cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", url)
		default:
			return errorMsg{err: fmt.Errorf("unsupported platform for opening urls: %s", runtime.GOOS)}
		}
		if err := cmd.Start(); err != nil {
			return errorMsg{err: fmt.Errorf("failed to open url %s: %v", url, err)}
		}
		return nil
	}
}
