package cmd

import (
	"encoding/json"
	"fmt"
	"hash/fnv"
	"io"
	"os"
	"os/exec"
	"runtime"
	"sort"
	"strings"
	"time"

	uitk "github.com/llamafarm/cli/internal/tui"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/term"
	"github.com/llamafarm/cli/cmd/config"
	"github.com/llamafarm/cli/cmd/orchestrator"
	"github.com/llamafarm/cli/cmd/utils"
	"github.com/llamafarm/cli/cmd/version"
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

// renderAssistantContent processes assistant message content, applying gray styling to <think> tags
func renderAssistantContent(content string, width int) string {
	// Check if content contains <think> tags
	if !strings.Contains(content, "<think>") && !strings.Contains(content, "</think>") {
		// No think tags, render normally
		return renderMarkdown(content, width)
	}

	// Parse and style content with think tags
	var result strings.Builder
	grayStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("#666666"))

	// Simple state machine to track if we're inside a think tag
	inThinkTag := false
	remainder := content

	for len(remainder) > 0 {
		if inThinkTag {
			// Look for closing tag
			if idx := strings.Index(remainder, "</think>"); idx >= 0 {
				// Content inside think tag - render in gray
				thinkContent := remainder[:idx]
				result.WriteString(grayStyle.Render(thinkContent))
				remainder = remainder[idx+8:] // Skip past </think>
				inThinkTag = false
			} else {
				// No closing tag found, render rest in gray
				result.WriteString(grayStyle.Render(remainder))
				remainder = ""
			}
		} else {
			// Look for opening tag
			if idx := strings.Index(remainder, "<think>"); idx >= 0 {
				// Content before think tag - render normally
				beforeThink := remainder[:idx]
				if beforeThink != "" {
					result.WriteString(renderMarkdown(beforeThink, width))
				}
				remainder = remainder[idx+7:] // Skip past <think>
				inThinkTag = true
			} else {
				// No opening tag found, render rest normally
				result.WriteString(renderMarkdown(remainder, width))
				remainder = ""
			}
		}
	}

	return result.String()
}

// renderToolCall renders a tool call as a styled bordered block
func renderToolCall(toolCall ToolCallItem, width int) string {
	// Parse arguments JSON for pretty display
	var args map[string]any
	if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args); err != nil {
		// If not valid JSON, show raw
		args = map[string]any{"raw": toolCall.Function.Arguments}
	}

	// Build content
	var contentLines []string
	contentLines = append(contentLines, lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("11")).Render("🔧 Tool Call"))
	contentLines = append(contentLines, "")
	contentLines = append(contentLines, lipgloss.NewStyle().Foreground(lipgloss.Color("86")).Render("Tool: ")+toolCall.Function.Name)
	// Truncate long IDs
	displayID := toolCall.ID
	if len(displayID) > 12 {
		displayID = displayID[:12] + "..."
	}
	contentLines = append(contentLines, lipgloss.NewStyle().Foreground(lipgloss.Color("240")).Render("ID: ")+displayID)

	if len(args) > 0 {
		contentLines = append(contentLines, "")
		contentLines = append(contentLines, lipgloss.NewStyle().Foreground(lipgloss.Color("86")).Render("Arguments:"))

		// Sort keys to ensure deterministic ordering (prevents flickering during streaming)
		keys := make([]string, 0, len(args))
		for k := range args {
			keys = append(keys, k)
		}
		sort.Strings(keys)

		// Iterate over sorted keys
		for _, k := range keys {
			v := args[k]
			// Truncate long values
			valStr := fmt.Sprintf("%v", v)
			if len(valStr) > 60 {
				valStr = valStr[:60] + "..."
			}
			contentLines = append(contentLines, fmt.Sprintf("  %s: %v",
				lipgloss.NewStyle().Foreground(lipgloss.Color("39")).Render(k),
				valStr))
		}
	}

	blockContent := strings.Join(contentLines, "\n")

	// Calculate box width (limit to terminal width)
	boxWidth := width - 10
	if boxWidth < 40 {
		boxWidth = 40
	}
	if boxWidth > 80 {
		boxWidth = 80
	}

	// Create styled box
	boxStyle := lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("86")).
		Padding(0, 1).
		Width(boxWidth)

	return "\n" + boxStyle.Render(blockContent) + "\n"
}

const gap = "\n\n"

// overrides provided by dev command
var designerPreferredPort int
var designerForced bool

var lastTranscriptKey string

// fetchAvailableModels is now defined in models_shared.go

// runChatSessionTUI starts the Bubble Tea TUI for chat.
func runChatSessionTUI(mode SessionMode, projectInfo *config.ProjectInfo) {
	m := newChatModel(projectInfo, mode)
	p := tea.NewProgram(m, tea.WithAltScreen(), tea.WithMouseCellMotion())
	m.program = p

	// Enable TUI mode for output routing
	utils.SetTUIMode(p)
	defer utils.ClearTUIMode()

	if _, err := p.Run(); err != nil {
		// Use the output API instead of direct stderr write
		utils.OutputError("Error running TUI: %v\n", err)
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
	serverHealth   *orchestrator.HealthPayload
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
	// Track first render to ensure initial scroll to bottom
	isFirstRender bool
	// Track if we just started a new response (should auto-scroll)
	justStartedResponse bool
	// Track tool calls during streaming
	pendingToolCalls []*ToolCall
	// Track if user intentionally cancelled (to suppress error messages)
	intentionallyCancelled bool
	// ChatManager instances for each mode
	devChatManager     *ChatManager
	projectChatManager *ChatManager
	currentChatManager *ChatManager
}

// removed: old bottom menu state

type (
	streamDone struct{}
)

type responseMsg struct{ content string }
type toolCallMsg struct{ toolCall *ToolCall }
type errorMsg struct{ err error }
type tickMsg struct{}

type serverHealthMsg struct{ health *orchestrator.HealthPayload }

func newChatModel(projectInfo *config.ProjectInfo, initialMode SessionMode) chatModel {
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

	// Determine session namespace/project for storage
	sessionNamespace := namespace
	sessionProject := projectID
	if projectInfo != nil {
		sessionNamespace = projectInfo.Namespace
		sessionProject = projectInfo.Project
	}

	// Create DEV mode ChatManager
	devCfg := &ChatConfig{
		ServerURL:        serverURL,
		Namespace:        "llamafarm",
		ProjectID:        "project_seed",
		SessionMode:      SessionModeDev,
		SessionNamespace: sessionNamespace,
		SessionProject:   sessionProject,
		RAGEnabled:       true,
	}
	devChatManager, err := NewChatManager(devCfg)
	if err != nil {
		utils.LogDebug(fmt.Sprintf("Failed to create dev manager: %v", err))
	}

	// Build DEV mode history
	var devHistory []Message
	var devUserChatMessages []string
	if devChatManager != nil {
		devHistory, _ = devChatManager.FetchHistory()
		for _, msg := range devHistory {
			if msg.Role == "user" {
				devUserChatMessages = append(devUserChatMessages, msg.Content)
			}
		}
		utils.LogDebug(fmt.Sprintf("Restored DEV history (session %s): %d messages", devChatManager.GetSessionID(), len(devHistory)))
	}

	devMessages = devHistory
	if len(devMessages) == 0 {
		devMessages = append(devMessages, Message{Role: "client", Content: "Send a message or type '/help' for commands."})
	}

	sm, _ := orchestrator.NewServiceManager(serverURL)
	serverHealth, _ := sm.GetServerHealth()

	// Fetch initial greeting for project_seed (disabled)
	// if greeting := fetchInitialGreeting(chatCtx); greeting != "" {
	// 	messages = append(messages, Message{Role: "assistant", Content: greeting})
	// }

	width, _, _ := term.GetSize(uintptr(os.Stdout.Fd()))

	// Always include server status as a client message in both modes
	devMessages = append(devMessages, Message{Role: "client", Content: renderServerStatusProblems(serverHealth)})

	// Initialize mode contexts - build DEV context
	devSessionID := ""
	if devChatManager != nil {
		devSessionID = devChatManager.GetSessionID()
	}
	devCtx := &ModeContext{
		Mode:      ModeDev,
		SessionID: devSessionID,
		Messages:  devMessages,
		History:   devUserChatMessages,
	}

	// Project mode context - try to restore session or create new one
	var projectHistory []Message
	var projectUserChatMessages []string
	var projectMessages []Message
	var projectChatManager *ChatManager

	// Fetch available models and databases for project mode
	var availableModels []ModelInfo
	var availableDatabases *DatabasesResponse
	var availableDatasets []DatasetBrief
	var availablePrompts []config.PromptSet
	var currentModel string
	var currentDatabase string
	var currentStrategy string

	// Collect startup warnings for display in TUI
	var startupWarnings []string

	if projectInfo != nil {
		// Fetch models
		availableModels = fetchAvailableModels(serverURL, projectInfo.Namespace, projectInfo.Project)
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
		var dbWarning string
		availableDatabases, dbWarning = fetchAvailableDatabases(serverURL, projectInfo.Namespace, projectInfo.Project)
		if dbWarning != "" {
			startupWarnings = append(startupWarnings, dbWarning)
		}
		// Fetch dataset names for commands menu
		var dsWarning string
		availableDatasets, dsWarning = fetchAvailableDatasets(serverURL, projectInfo.Namespace, projectInfo.Project)
		if dsWarning != "" {
			startupWarnings = append(startupWarnings, dsWarning)
		}
		// Load prompts from project config file on disk (best effort)
		if cfg, err := config.LoadConfig(utils.GetEffectiveCWD()); err == nil && cfg != nil {
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

		// Create PROJECT mode ChatManager
		projectChatCfg := &ChatConfig{
			ServerURL:            serverURL,
			Namespace:            projectInfo.Namespace,
			ProjectID:            projectInfo.Project,
			SessionMode:          SessionModeProject,
			SessionNamespace:     projectInfo.Namespace,
			SessionProject:       projectInfo.Project,
			Model:                currentModel,
			RAGEnabled:           true,
			RAGDatabase:          currentDatabase,
			RAGRetrievalStrategy: currentStrategy,
		}
		projectChatManager, err = NewChatManager(projectChatCfg)
		if err != nil {
			utils.LogDebug(fmt.Sprintf("Failed to create project manager: %v", err))
		}

		// Build PROJECT mode history
		if projectChatManager != nil {
			projectHistory, _ = projectChatManager.FetchHistory()
			for _, msg := range projectHistory {
				if msg.Role == "user" {
					projectUserChatMessages = append(projectUserChatMessages, msg.Content)
				}
			}
			utils.LogDebug(fmt.Sprintf("Restored PROJECT history (session %s): %d messages", projectChatManager.GetSessionID(), len(projectHistory)))
		}
	}

	projectMessages = projectHistory
	if len(projectMessages) == 0 {
		projectMessages = []Message{{Role: "client", Content: "Send a message or type '/help' for commands."}}
	}
	// Add server status to project messages as well
	projectMessages = append(projectMessages, Message{Role: "client", Content: renderServerStatusProblems(serverHealth)})
	// Add any startup warnings
	for _, warning := range startupWarnings {
		projectMessages = append(projectMessages, Message{Role: "client", Content: fmt.Sprintf("⚠️  %s", warning)})
	}

	projectCtx := &ModeContext{
		Mode:              ModeProject,
		SessionID:         "",
		Messages:          projectMessages,
		History:           projectUserChatMessages,
		Model:             currentModel,
		Database:          currentDatabase,
		RetrievalStrategy: currentStrategy,
	}
	if projectChatManager != nil {
		projectCtx.SessionID = projectChatManager.GetSessionID()
	}

	// Choose initial mode and state
	initialMessages := devMessages
	initialHistory := devUserChatMessages
	currentManager := devChatManager
	if initialMode == SessionModeProject && projectInfo != nil {
		initialMessages = projectMessages
		initialHistory = projectUserChatMessages
		currentManager = projectChatManager
	}

	// Initialize viewport content with initial mode messages and scroll to bottom
	vp.SetContent(renderChatContent(chatModel{messages: initialMessages}))
	vp.GotoBottom()

	// Initialize overlay Quick Menu and toast
	menuCfg := &uitk.Config{}
	if projectInfo != nil {
		menuCfg.Name = projectInfo.Project
		menuCfg.Namespace = projectInfo.Namespace
	}
	// Attach CLI version for menu header
	menuCfg.Version = version.FormatVersionForDisplay(version.CurrentVersion)
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
				// Use first message from the prompt set
				if len(p.Messages) == 0 {
					continue // Skip prompts with no messages
				}

				content := p.Messages[0].Content
				role := p.Messages[0].Role

				// Default to "system" if missing
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

	// Determine current mode from initialMode parameter
	var chatMode ChatMode
	if initialMode == SessionModeDev {
		chatMode = ModeDev
	} else {
		chatMode = ModeProject
	}

	return chatModel{
		serverHealth:       serverHealth,
		projectInfo:        projectInfo,
		spin:               s,
		messages:           initialMessages,
		thinking:           false,
		printing:           false,
		history:            initialHistory,
		histIndex:          len(initialHistory),
		designerStatus:     "ready",
		designerURL:        serverURL,
		textarea:           ta,
		viewport:           vp,
		width:              width,
		currentMode:        chatMode,
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
		isFirstRender:      true,
		devChatManager:     devChatManager,
		projectChatManager: projectChatManager,
		currentChatManager: currentManager,
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

	// Switch to the appropriate manager
	if newMode == ModeDev {
		m.currentChatManager = m.devChatManager
	} else {
		m.currentChatManager = m.projectChatManager
		// Restore model for project mode
		if m.projectModeCtx.Model != "" {
			m.currentModel = m.projectModeCtx.Model
			// Update the manager's config
			if m.currentChatManager != nil {
				m.currentChatManager.UpdateConfig(func(cfg *ChatConfig) {
					cfg.Model = m.currentModel
				})
			}
		}
	}

	// config := &orchestrator.ServiceOrchestrationConfig{
	// 	ServerURL:   serverURL,
	// 	PrintStatus: true,
	// 	ServiceNeeds: map[string]orchestrator.ServiceRequirement{
	// 		"universal-runtime": orchestrator.ServiceOptional, // Start async, don't wait
	// 		"server":            orchestrator.ServiceRequired,
	// 		"rag":               orchestrator.ServiceOptional, // Start async, don't wait
	// 	},
	// 	DefaultTimeout: 45 * time.Second,
	// }
	// health, _ := orchestrator.CheckServerHealth(serverURL)
	// m.serverHealth = FilterHealthForOptionalServices(health, config, chatCtx.SessionMode)

	// Return switch message
	chatMsg := ""
	if newMode == ModeDev {
		chatMsg = "🦙 Switched to DEV mode - Chat with LlamaFarm Assistant"
	} else {
		var ns, proj string
		if m.projectChatManager != nil {
			cfg := m.projectChatManager.GetConfig()
			ns = cfg.Namespace
			proj = cfg.ProjectID
		}
		chatMsg = fmt.Sprintf("🎯 Switched to PROJECT mode - Testing %s/%s", ns, proj)
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

	// Update the manager's config
	if m.projectChatManager != nil {
		m.projectChatManager.UpdateConfig(func(cfg *ChatConfig) {
			cfg.Model = newModel
		})
	}

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
	// Kick off spinner and server health check
	return tea.Batch(m.spin.Tick, updateServerHealthCmd(m))
}

func updateServerHealthCmd(m chatModel) tea.Cmd {
	return func() tea.Msg {
		// health, _ := orchestrator.CheckServerHealth(serverURL)
		// return serverHealthMsg{health: health}
		return serverHealthMsg{health: nil}
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

	// Only pass non-keyboard events to viewport to prevent interference with textarea
	// Viewport should handle mouse events and window size, but not keyboard input
	shouldUpdateViewport := true
	if _, ok := msg.(tea.KeyMsg); ok {
		// Don't pass keyboard events to viewport - they're for textarea input
		// This prevents spacebar triggering page-down, etc.
		shouldUpdateViewport = false
	}

	// Track viewport position before update to detect user scrolling
	wasAtBottomBeforeUpdate := m.viewport.AtBottom()

	if shouldUpdateViewport {
		m.viewport, vpCmd = m.viewport.Update(msg)
	}

	// If viewport was at bottom but user scrolled up (via mouse), stop auto-scrolling
	// This allows breaking free from following streaming responses
	if wasAtBottomBeforeUpdate && !m.viewport.AtBottom() && m.printing {
		m.justStartedResponse = false
	}

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
		utils.LogDebug(fmt.Sprintf("Checking latest server health. Last: %v", m.serverHealth))
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

		// On first render, ensure content is fully loaded before scrolling to bottom
		if m.isFirstRender {
			m.isFirstRender = false
			// Compute transcript and set viewport content first
			m.transcript = computeTranscript(m)
			m.setViewportContent()
			// Now scroll to bottom with fully loaded content
			m.viewport.GotoBottom()
		}

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
			// If overlay is active, let it handle ESC
			if m.quickMenu.IsActive() {
				return m, tea.Batch(cmds...)
			}
			// Cancel active stream if one is in progress
			if (m.thinking || m.printing) && m.currentChatManager != nil {
				m.intentionallyCancelled = true // Mark as intentional to suppress context.Canceled errors
				m.currentChatManager.Cancel()
				m.thinking = false
				m.printing = false
				m.streamCh = nil
				m.justStartedResponse = false
				m.messages = append(m.messages, Message{Role: "client", Content: "⚠️ Operation aborted"})
				m.refreshViewportBottom()
			}
			return m, tea.Batch(cmds...)

		case "up":
			// If overlay is active, let it handle navigation
			if m.quickMenu.IsActive() {
				return m, tea.Batch(cmds...)
			}
			// Navigate history
			utils.LogDebug(fmt.Sprintf("Up arrow pressed. Current history: %+v", m.history))
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

			// Process command or message using shared handler
			wasCommand, cmd := m.processCommandOrMessage(msg)
			if wasCommand {
				if cmd != nil {
					return m, cmd
				}
				m.refreshViewportBottom()
				break
			}

			// Regular message - start chat stream
			cmds = append(cmds, m.startChatStreamForMessage(msg))
		}

	case toolCallMsg:
		// Skip processing tool call if user intentionally cancelled
		if m.intentionallyCancelled {
			utils.LogDebug("Skipping toolCallMsg due to intentional cancellation")
			break
		}

		// Tool calls are added as assistant messages with structured data
		utils.LogDebug(fmt.Sprintf("TOOL CALL MSG: %s (ID: %s)", msg.toolCall.Name, msg.toolCall.ID))

		// Create or update the last assistant message with this tool call
		if len(m.messages) > 0 && m.messages[len(m.messages)-1].Role == "assistant" && len(m.messages[len(m.messages)-1].ToolCalls) > 0 {
			// Append to existing assistant message with tool calls
			lastMsg := &m.messages[len(m.messages)-1]
			lastMsg.ToolCalls = append(lastMsg.ToolCalls, ToolCallItem{
				ID:   msg.toolCall.ID,
				Type: "function",
				Function: ToolCallFunction{
					Name:      msg.toolCall.Name,
					Arguments: msg.toolCall.Arguments,
				},
			})
		} else {
			// Create new assistant message with tool call
			m.messages = append(m.messages, Message{
				Role:    "assistant",
				Content: "",
				ToolCalls: []ToolCallItem{{
					ID:   msg.toolCall.ID,
					Type: "function",
					Function: ToolCallFunction{
						Name:      msg.toolCall.Name,
						Arguments: msg.toolCall.Arguments,
					},
				}},
			})
		}

		// Store tool call for potential execution
		m.pendingToolCalls = append(m.pendingToolCalls, msg.toolCall)
		utils.LogDebug(fmt.Sprintf("Stored tool call: %s (ID: %s)", msg.toolCall.Name, msg.toolCall.ID))

		// Auto-scroll to show tool call
		if m.justStartedResponse || m.viewport.AtBottom() {
			m.refreshViewportBottom()
		} else {
			m.setViewportContent()
		}

		if m.streamCh != nil {
			cmds = append(cmds, listen(m.streamCh))
		}

	case responseMsg:
		if m.err != nil {
			m.err = nil
			break
		}

		// Skip processing response if user intentionally cancelled
		if m.intentionallyCancelled {
			utils.LogDebug("Skipping responseMsg due to intentional cancellation")
			break
		}

		utils.LogDebug(fmt.Sprintf("RESPONSE MSG: %v", msg.content))
		m.thinking = false
		m.printing = true

		// Check if viewport is at bottom before updating content
		wasAtBottom := m.viewport.AtBottom()

		// Check if last message is a tool call (don't update it, create new message)
		lastIsToolCall := false
		if len(m.messages) > 0 {
			lastMsg := m.messages[len(m.messages)-1]
			lastIsToolCall = lastMsg.Role == "assistant" && len(lastMsg.ToolCalls) > 0
		}

		if len(m.messages) == 0 ||
			(len(m.messages) > 0 && m.messages[len(m.messages)-1].Role != "assistant") ||
			lastIsToolCall {
			// Create new message if:
			// - No messages yet
			// - Last message is not assistant (it's user/client/error)
			// - Last message is a tool call (don't overwrite it)
			m.messages = append(m.messages, Message{Role: "assistant", Content: msg.content})
		} else {
			// Update last assistant message (it's regular streaming content)
			if len(m.messages) > 0 {
				m.messages[len(m.messages)-1] = Message{Role: "assistant", Content: msg.content}
			} else {
				m.messages = append(m.messages, Message{Role: "assistant", Content: msg.content})
			}
		}

		// Auto-scroll during streaming if:
		// 1. This is a fresh response (just sent a message), OR
		// 2. Viewport was already at bottom (following along)
		// This allows users to scroll up to read previous messages, but ensures
		// new responses are visible when you just sent a message
		if m.justStartedResponse || wasAtBottom {
			m.refreshViewportBottom()
		} else {
			m.setViewportContent()
		}

		if m.streamCh != nil {
			cmds = append(cmds, listen(m.streamCh))
		}

	case errorMsg:
		m.thinking = false
		m.err = msg.err
		// Don't show error if user intentionally cancelled and this is a context cancellation error
		if !(m.intentionallyCancelled && strings.Contains(msg.err.Error(), "context canceled")) {
			m.messages = append(m.messages, Message{Role: "error", Content: fmt.Sprintf("Error: %v", msg.err)})
		}
		m.justStartedResponse = false    // Reset flag on error
		m.intentionallyCancelled = false // Reset cancellation flag
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
			utils.LogDebug(fmt.Sprintf("STREAM DONE: %v", m.messages[len(m.messages)-1]))
		} else {
			utils.LogDebug("STREAM DONE: no messages")
		}
		m.printing = false
		m.streamCh = nil
		m.intentionallyCancelled = false // Reset cancellation flag
		m.justStartedResponse = false    // Reset flag after streaming is complete

		// Check if there are any CLI tools to execute
		if len(m.pendingToolCalls) > 0 {
			utils.LogDebug(fmt.Sprintf("Processing %d pending tool calls", len(m.pendingToolCalls)))

			// Execute CLI tools and collect results
			var toolResults []Message
			// Create a context from current manager for tool execution
			var toolCtx *ChatSessionContext
			if m.currentChatManager != nil {
				cfg := m.currentChatManager.GetConfig()
				toolCtx = &ChatSessionContext{
					ServerURL:        cfg.ServerURL,
					Namespace:        cfg.Namespace,
					ProjectID:        cfg.ProjectID,
					SessionID:        m.currentChatManager.GetSessionID(),
					SessionMode:      cfg.SessionMode,
					SessionNamespace: cfg.SessionNamespace,
					SessionProject:   cfg.SessionProject,
					HTTPClient:       utils.GetHTTPClient(),
				}
			}
			for _, tc := range m.pendingToolCalls {
				if strings.HasPrefix(tc.Name, "cli.") {
					utils.LogDebug(fmt.Sprintf("Executing CLI tool: %s", tc.Name))

					// Execute the tool
					if toolResult, err := ExecuteToolCall(tc, toolCtx); err != nil {
						// Tool execution failed - send error as tool result

						errMsg := fmt.Sprintf("Tool execution failed: %v", err)
						utils.LogDebug(errMsg)
						toolResults = append(toolResults, Message{
							Role:       "tool",
							Content:    errMsg,
							ToolCallID: tc.ID,
						})
						// Also show error to user
						m.messages = append(m.messages, Message{Role: "error", Content: fmt.Sprintf("❌ %s", errMsg)})
					} else if toolResult != nil {
						// Tool executed successfully
						utils.LogDebug(fmt.Sprintf("Tool executed successfully: %s", toolResult.Content))
						toolResults = append(toolResults, *toolResult)
						// Show success to user
						m.messages = append(m.messages, Message{Role: "client", Content: fmt.Sprintf("✅ %s", toolResult.Content)})
					}
				}
			}

			// Clear pending tool calls
			m.pendingToolCalls = nil

			// If we have tool results, send them back to continue the conversation
			if len(toolResults) > 0 {
				utils.LogDebug(fmt.Sprintf("Sending %d tool results back to API", len(toolResults)))
				m.messages = append(m.messages, toolResults...)
				m.setViewportContent()
				m.refreshViewportBottom()

				// Only send the tool result messages - the server already has the assistant
				// message with tool calls in its session history
				m.thinking = true
				m.printing = true
				m.justStartedResponse = true
				ch := make(chan tea.Msg, 32)
				m.streamCh = ch
				go func() {
					var builder strings.Builder
					err := m.currentChatManager.StreamMessages(toolResults, func(chunk StreamChunk) error {
						switch chunk.Type {
						case ChunkTypeContent:
							builder.WriteString(chunk.Content)
							ch <- responseMsg{content: builder.String()}
						case ChunkTypeToolCall:
							// Send any accumulated content first
							if builder.Len() > 0 {
								ch <- responseMsg{content: builder.String()}
							}
							// Pass the structured tool call
							ch <- toolCallMsg{toolCall: chunk.ToolCall}
							// Reset builder for subsequent content
							builder.Reset()
						case ChunkTypeError:
							ch <- errorMsg{err: chunk.Error}
						case ChunkTypeDone:
							if builder.Len() > 0 {
								ch <- responseMsg{content: builder.String()}
							}
							ch <- streamDone{}
						}
						return nil
					})

					if err != nil {
						ch <- errorMsg{err: err}
					}
					close(ch)
				}()
				cmds = append(cmds, tea.Batch(listen(m.streamCh), thinkingCmd()))
			}
		}

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
			// m.quickMenu.RAGHealthSummary = formatRAGHealthSummary(m.serverHealth)
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
		// Echo the command in the chat
		commandStr := strings.TrimSpace(msg.Command)
		m.messages = append(m.messages, Message{Role: "client", Content: fmt.Sprintf("$ %s", commandStr)})

		// Special-case: run upgrade inside the current process
		if strings.HasPrefix(commandStr, "lf version upgrade") {
			cmds = append(cmds, func() tea.Msg {
				if err := version.PerformUpgrade(version.UpgradeOpts{}); err != nil {
					return uitk.ShowToastMsg{Message: "Upgrade failed: " + err.Error()}
				}
				return uitk.ShowToastMsg{Message: "Upgrade completed successfully"}
			})
			// Ensure menu closes to avoid inconsistent UI state
			m.quickMenu.Close()
			break
		}

		// Default: just show a toast for now
		cmds = append(cmds, tea.Printf("Executing: %s", commandStr))
		cmds = append(cmds, func() tea.Msg { return uitk.ShowToastMsg{Message: "Running: " + commandStr} })
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
			// Directly process the command/message
			m.err = nil
			val := strings.TrimSpace(msg.Text)
			if val == "" || m.thinking {
				break
			}

			// Process command or message using shared logic
			wasCommand, cmd := m.processCommandOrMessage(val)
			if wasCommand {
				if cmd != nil {
					return m, cmd
				}
				m.refreshViewportBottom()
				break
			}

			// Regular message - start chat stream
			cmds = append(cmds, m.startChatStreamForMessage(val))
		}

	case utils.TUIMessageMsg:
		// Handle output messages routed through the messaging API
		formattedContent := utils.FormatMessage(msg.Message)

		if msg.Message.Type == utils.ProgressMessage {
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
		m.transcript = computeTranscript(m)
		m.refreshViewportBottom()
		return m, tea.Batch(cmds...)
	}

	m.transcript = computeTranscript(m)
	m.setViewportContent()

	return m, tea.Batch(cmds...)
}

func listen(ch <-chan tea.Msg) tea.Cmd {
	return func() tea.Msg {
		msg, ok := <-ch
		utils.LogDebug(fmt.Sprintf("LISTEN MSG: %v", msg))
		if !ok {
			fmt.Println("LISTEN DONE")
			return streamDone{}
		}
		return msg
	}
}

func renderServerStatusProblems(health *orchestrator.HealthPayload) string {
	var b strings.Builder

	if health == nil {
		return ""
	}

	// prettyPrintHealthProblems(&b, *health)

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
				// Check if this message has tool calls
				if len(message.ToolCalls) > 0 {
					// Render each tool call
					for _, toolCall := range message.ToolCalls {
						line += renderToolCall(toolCall, m.width)
					}
				} else if message.Content != "" {
					// Render content with think tag support
					renderedContent := renderAssistantContent(message.Content, m.width-len(m.getAssistantLabel())-4)
					// Don't use lipgloss.Render on the rendered content to preserve ANSI codes
					labelStyle := baseStyle.Foreground(lipgloss.Color("11"))
					line = labelStyle.Render(m.getAssistantLabel()) + " " + renderedContent + "\n"
				}
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
	helpText := fmt.Sprintf("/help for commands | Up/Down: history | Esc: cancel | %s", modeHint)

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
	statusIcon := utils.IconForStatus(func() string {
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

// processCommandOrMessage processes a slash command or returns false for regular messages.
// Returns (wasCommand bool, cmd tea.Cmd). If wasCommand is true, cmd may be nil or a tea.Cmd to execute.
func (m *chatModel) processCommandOrMessage(msg string) (wasCommand bool, cmd tea.Cmd) {
	lower := strings.ToLower(msg)
	// Slash commands
	if strings.HasPrefix(lower, "/") {
		fields := strings.Fields(lower)
		cmdName := fields[0]
		switch cmdName {
		case "/help":
			m.messages = append(m.messages, Message{Role: "client", Content: "Commands:\n  /help - Show this help\n  /mode [dev|project] - Switch mode\n  /model [name] - Switch model (PROJECT mode)\n  /database [name] - Switch RAG database (PROJECT mode)\n  /strategy [name] - Switch retrieval strategy (PROJECT mode)\n  /clear - Clear conversation\n  /launch designer - Open designer\n  /menu - Open Quick Menu\n  /exit - Exit\n  To check version and upgrades run \"lf version\"\n\nHotkeys:\n  Ctrl+T - Toggle DEV/PROJECT mode\n  Ctrl+K - Cycle models\n  Esc - Cancel current operation"})
			m.textarea.SetValue("")
		case "/mode":
			if len(fields) < 2 {
				m.messages = append(m.messages, Message{Role: "client", Content: "Usage: /mode [dev|project]"})
				m.textarea.SetValue("")
				return true, nil
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
				return true, nil
			}
			if newMode == m.currentMode {
				m.messages = append(m.messages, Message{Role: "client", Content: fmt.Sprintf("Already in %s mode", modeArg)})
				m.textarea.SetValue("")
				return true, nil
			}
			m.switchMode(newMode)
			m.textarea.SetValue("")
		case "/model":
			if m.currentMode != ModeProject {
				m.messages = append(m.messages, Message{
					Role:    "client",
					Content: "Model switching only available in PROJECT mode. Use Ctrl+T to switch.",
				})
				m.textarea.SetValue("")
				return true, nil
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
				return true, nil
			}

			modelName := fields[1]
			if !m.isValidModel(modelName) {
				m.messages = append(m.messages, Message{
					Role:    "client",
					Content: fmt.Sprintf("Unknown model '%s'. Type '/model' to see available models.", modelName),
				})
				m.textarea.SetValue("")
				return true, nil
			}

			m.switchModel(modelName)
			m.textarea.SetValue("")
		case "/launch":
			if len(fields) < 2 {
				m.messages = append(m.messages, Message{Role: "client", Content: "Usage: /launch <component>. Components: designer"})
				m.textarea.SetValue("")
				return true, nil
			}
			target := fields[1]
			if target != "designer" {
				m.messages = append(m.messages, Message{Role: "client", Content: fmt.Sprintf("Unknown component '%s'. Try: /launch designer", target)})
				m.textarea.SetValue("")
				return true, nil
			}
			// Designer is served by the server at root URL
			m.textarea.SetValue("")
			designerURL := serverURL
			if m.currentChatManager != nil {
				designerURL = m.currentChatManager.GetConfig().ServerURL
			}
			return true, openURL(designerURL)
		case "/exit", "/quit":
			m.status = "👋 You have left the pasture. Safe travels, little llama!"
			return true, tea.Quit
		case "/clear":
			// Clear session using the manager
			if m.currentChatManager != nil {
				if err := m.currentChatManager.ClearSession(); err != nil {
					utils.LogDebug(fmt.Sprintf("Failed to clear session: %v", err))
				}
			}

			// Get current mode context
			ctx := m.getCurrentModeContext()

			// Update session ID in context
			if m.currentChatManager != nil {
				ctx.SessionID = m.currentChatManager.GetSessionID()
				utils.LogDebug(fmt.Sprintf("Created new session ID: %s", ctx.SessionID))
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
		case "/database":
			if m.currentMode != ModeProject {
				m.messages = append(m.messages, Message{
					Role:    "client",
					Content: "Database switching only available in PROJECT mode. Use Ctrl+T to switch.",
				})
				m.textarea.SetValue("")
				return true, nil
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
				return true, nil
			}

			dbName := fields[1]
			if !m.isValidDatabase(dbName) {
				m.messages = append(m.messages, Message{
					Role:    "client",
					Content: fmt.Sprintf("Unknown database '%s'. Type '/database' to see available databases.", dbName),
				})
				m.textarea.SetValue("")
				return true, nil
			}

			m.switchDatabase(dbName)
			m.textarea.SetValue("")
		case "/strategy":
			if m.currentMode != ModeProject {
				m.messages = append(m.messages, Message{
					Role:    "client",
					Content: "Strategy switching only available in PROJECT mode. Use Ctrl+T to switch.",
				})
				m.textarea.SetValue("")
				return true, nil
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
				return true, nil
			}

			strategyName := fields[1]
			if !m.isValidStrategy(strategyName) {
				m.messages = append(m.messages, Message{
					Role:    "client",
					Content: fmt.Sprintf("Unknown strategy '%s' for database '%s'. Type '/strategy' to see available strategies.", strategyName, m.currentDatabase),
				})
				m.textarea.SetValue("")
				return true, nil
			}

			m.switchStrategy(strategyName)
			m.textarea.SetValue("")
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
			// Check for updates and reflect status in the menu
			if info, err := version.MaybeCheckForUpgrade(true); err == nil && info != nil {
				if info.UpdateAvailable {
					m.quickMenu.SetUpdateAvailable(info.LatestVersion)
				} else {
					m.quickMenu.SetUpToDate()
				}
			}
			m.textarea.SetValue("")
		default:
			m.messages = append(m.messages, Message{Role: "client", Content: fmt.Sprintf("Unknown command '%s'. All commands must start with '/'. Type '/help' for available commands.", cmdName)})
			m.textarea.SetValue("")
		}
		return true, nil
	}
	return false, nil
}

// startChatStreamForMessage starts a chat stream for a regular (non-command) message.
func (m *chatModel) startChatStreamForMessage(msg string) tea.Cmd {
	m.history = append(m.history, msg)
	m.histIndex = len(m.history)
	m.messages = append(m.messages, Message{Role: "user", Content: msg})
	m.textarea.SetValue("")
	m.thinking = true
	m.printing = true
	m.justStartedResponse = true     // Mark that we're starting a new response
	m.intentionallyCancelled = false // Reset cancellation flag for new request
	// Scroll to bottom when user sends a message - ensures they see the response
	m.refreshViewportBottom()

	// Update manager config with current selections (PROJECT mode)
	if m.currentMode == ModeProject && m.currentChatManager != nil {
		m.currentChatManager.UpdateConfig(func(cfg *ChatConfig) {
			if m.currentModel != "" {
				cfg.Model = m.currentModel
			}
			if m.currentDatabase != "" {
				cfg.RAGDatabase = m.currentDatabase
				cfg.RAGEnabled = true
			}
			if m.currentStrategy != "" {
				cfg.RAGRetrievalStrategy = m.currentStrategy
			}
		})
	}

	// Start channel-based streaming using ChatManager
	ch := make(chan tea.Msg, 32)
	m.streamCh = ch

	// Only send the new user message, not the entire history
	// The server manages history via the session
	newMessages := []Message{m.messages[len(m.messages)-1]}

	go func() {
		var builder strings.Builder
		err := m.currentChatManager.StreamMessages(newMessages, func(chunk StreamChunk) error {
			switch chunk.Type {
			case ChunkTypeContent:
				builder.WriteString(chunk.Content)
				ch <- responseMsg{content: builder.String()}
			case ChunkTypeToolCall:
				// Send any accumulated content first
				if builder.Len() > 0 {
					ch <- responseMsg{content: builder.String()}
				}
				// Pass the structured tool call
				ch <- toolCallMsg{toolCall: chunk.ToolCall}
				// Reset builder for subsequent content
				builder.Reset()
			case ChunkTypeError:
				ch <- errorMsg{err: chunk.Error}
			case ChunkTypeDone:
				if builder.Len() > 0 {
					ch <- responseMsg{content: builder.String()}
				}
				ch <- streamDone{}
			}
			return nil
		})

		if err != nil {
			ch <- errorMsg{err: err}
		}
		close(ch)
	}()

	return tea.Batch(listen(m.streamCh), thinkingCmd())
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
