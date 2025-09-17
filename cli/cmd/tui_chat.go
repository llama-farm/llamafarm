package cmd

import (
	"bufio"
	"context"
	"fmt"
	"hash/fnv"
	"io"
	"llamafarm-cli/cmd/config"
	"os"
	"os/exec"
	"path/filepath"
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
)

var (
	farmerPrompt     = "🌾 Farmer:"
	serverPrompt     = "📡 Server:"
	ollamaHostPrompt = "🐏 Ollama:"
	projectPrompt    = "📁 Project:"
	sessionPrompt    = "🆔"
)

const gap = "\n\n"

// overrides provided by dev command
var designerPreferredPort int
var designerForced bool

var lastTranscriptKey string

var chatCtx = &ChatSessionContext{
	ServerURL:   serverURL,
	Namespace:   "llamafarm",
	ProjectID:   "project-seed",
	Temperature: temperature,
	MaxTokens:   maxTokens,
	HTTPClient:  getHTTPClient(),
}

// runChatSessionTUI starts the Bubble Tea TUI for chat.
func runChatSessionTUI(projectInfo *config.ProjectInfo, serverHealth *HealthPayload) {
	m := newChatModel(projectInfo, serverHealth)
	p := tea.NewProgram(m)
	m.program = p
	if _, err := p.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Error running TUI: %v\n", err)
	}
}

type chatModel struct {
	transcript     string
	serverHealth   *HealthPayload
	projectInfo    *config.ProjectInfo
	spin           spinner.Model
	messages       []ChatMessage
	thinking       bool
	printing       bool
	thinkFrame     int
	history        []string
	histIndex      int
	historyPath    string
	width          int
	height         int
	status         string
	err            error
	viewport       viewport.Model
	textarea       textarea.Model
	program        *tea.Program
	streamCh       chan tea.Msg
	designerStatus string
	designerURL    string
}

type (
	streamDone struct{}
)

type responseMsg struct{ content string }
type errorMsg struct{ err error }
type tickMsg struct{}

type designerReadyMsg struct{ url string }
type designerErrorMsg struct{ err error }
type serverHealthMsg struct{ health *HealthPayload }

func newChatModel(projectInfo *config.ProjectInfo, serverHealth *HealthPayload) chatModel {
	messages := []ChatMessage{{Role: "client", Content: "Send a message or type '/help' for commands."}}

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
	vp.SetContent(renderChatContent(chatModel{messages: messages}))

	ta.KeyMap.InsertNewline.SetEnabled(false)

	s := spinner.New()
	s.Spinner = spinner.Dot
	s.Style = lipgloss.NewStyle().Foreground(lipgloss.Color("11"))
	hPath := getHistoryPath()
	h := loadHistory(hPath)

	width, _, _ := term.GetSize(uintptr(os.Stdout.Fd()))

	messages = append(messages, ChatMessage{Role: "client", Content: renderServerStatusProblems(serverHealth)})
	// transcript = append(transcript, problems)

	return chatModel{
		serverHealth:   serverHealth,
		projectInfo:    projectInfo,
		spin:           s,
		messages:       messages,
		thinking:       false,
		printing:       false,
		history:        h,
		histIndex:      len(h),
		historyPath:    hPath,
		designerStatus: "starting…",
		textarea:       ta,
		viewport:       vp,
		width:          width,
	}
}

func getHistoryPath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	dir := filepath.Join(home, ".llamafarm")
	_ = os.MkdirAll(dir, 0700)
	return filepath.Join(dir, "history")
}

func loadHistory(path string) []string {
	if path == "" {
		return nil
	}
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()
	var out []string
	s := bufio.NewScanner(f)
	for s.Scan() {
		line := strings.TrimSpace(s.Text())
		if line != "" {
			out = append(out, line)
		}
	}
	return out
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

	m.textarea, tiCmd = m.textarea.Update(msg)
	m.viewport, vpCmd = m.viewport.Update(msg)

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
		m.viewport.Width = msg.Width
		m.textarea.SetWidth(msg.Width - 2)
		m.viewport.Height = msg.Height - footerHeight - headerHeight
		m.width = msg.Width

	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c":
			m.status = "👋 You have left the pasture. Safe travels, little llama!"
			return m, tea.Quit

		case "up":
			if m.histIndex > 0 {
				m.histIndex--
				m.textarea.SetValue(m.history[m.histIndex])
				m.textarea.CursorEnd()
			}

		case "down":
			if m.histIndex < len(m.history)-1 {
				m.histIndex++
				m.textarea.SetValue(m.history[m.histIndex])
				m.textarea.CursorEnd()
			} else {
				m.histIndex = len(m.history)
				m.textarea.SetValue("")
			}

		case "enter":
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
					m.messages = append(m.messages, ChatMessage{Role: "client", Content: "Commands: /help, /launch designer, clear, exit"})
					m.textarea.SetValue("")
				case "/launch":
					if len(fields) < 2 {
						m.messages = append(m.messages, ChatMessage{Role: "client", Content: "Usage: /launch <component>. Components: designer"})
						m.textarea.SetValue("")
						break
					}
					target := fields[1]
					if target != "designer" {
						m.messages = append(m.messages, ChatMessage{Role: "client", Content: fmt.Sprintf("Unknown component '%s'. Try: /launch designer", target)})
						m.textarea.SetValue("")
						break
					}
					if strings.TrimSpace(m.designerURL) == "" || m.designerStatus != "ready" {
						m.messages = append(m.messages, ChatMessage{Role: "client", Content: "Designer is not running yet."})
						m.textarea.SetValue("")
						break
					}
					cmds = append(cmds, openURL(m.designerURL))
					m.textarea.SetValue("")
				default:
					m.messages = append(m.messages, ChatMessage{Role: "client", Content: fmt.Sprintf("Unknown command '%s'. Type '/help' for available commands.", cmd)})
					m.textarea.SetValue("")
				}
				break
			}

			if lower == "exit" || lower == "quit" {
				m.status = "👋 You have left the pasture. Safe travels, little llama!"
				return m, tea.Quit
			}

			if lower == "clear" {
				m.transcript = ""
				m.messages = nil
				m.textarea.SetValue("")
				m.viewport.SetContent(lipgloss.NewStyle().Width(m.viewport.Width).Render(renderChatContent(m)))
				m.thinking = false
				m.printing = false
				break
			}

			// persist history
			if m.historyPath != "" {
				f, err := os.OpenFile(m.historyPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0600)
				if err == nil {
					fmt.Fprintln(f, msg)
					f.Close()
				}
			}

			m.history = append(m.history, msg)
			m.histIndex = len(m.history)
			m.messages = append(m.messages, ChatMessage{Role: "user", Content: msg})
			m.textarea.SetValue("")
			m.thinking = true
			m.printing = true
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
			m.messages = append(m.messages, ChatMessage{Role: "assistant", Content: msg.content})
		} else {
			// Update last assistant line
			if len(m.messages) > 0 {
				m.messages[len(m.messages)-1] = ChatMessage{Role: "assistant", Content: msg.content}
			} else {
				m.messages = append(m.messages, ChatMessage{Role: "assistant", Content: msg.content})
			}
		}

		m.viewport.SetContent(lipgloss.NewStyle().Width(m.viewport.Width).Render(renderChatContent(m)))

		if m.streamCh != nil {
			cmds = append(cmds, listen(m.streamCh))
		}

	case errorMsg:
		m.thinking = false
		m.err = msg.err
		m.messages = append(m.messages, ChatMessage{Role: "error", Content: fmt.Sprintf("Error: %v", msg.err)})
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
		m.serverHealth = msg.health

		if m.serverHealth != nil && m.serverHealth.Status != "healthy" {
			// Schedule a non-blocking re-check after 5 seconds
			cmds = append(cmds, tea.Tick(5*time.Second, func(time.Time) tea.Msg {
				return updateServerHealthCmd(m)()
			}))
		}
	}

	m.transcript = computeTranscript(m)
	m.viewport.SetContent(lipgloss.NewStyle().Width(m.viewport.Width).Render(renderChatContent(m)))
	m.viewport.GotoBottom()

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
				line = baseStyle.Foreground(lipgloss.Color("11")).Render(farmerPrompt) + " " + message.Content + "\n"
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
		thinkingText := farmerPrompt + " " + m.spin.View() + "Thinking" + strings.Repeat(".", dots)
		wrappedThinking := lipgloss.NewStyle().Foreground(lipgloss.Color("11")).Width(m.width - 2).Render(thinkingText)
		b.WriteString(wrappedThinking + gap)
	}

	return b.String()
}

func renderChatInput(m chatModel) string {
	var b strings.Builder

	b.WriteString(gap)

	cbStyle := lipgloss.NewStyle().
		MarginBottom(1).
		BorderStyle(lipgloss.NormalBorder()).
		BorderForeground(lipgloss.Color("63"))

	b.WriteString(cbStyle.Render(m.textarea.View()))
	helpText := "Type '/help' for commands. Up/Down for history."
	b.WriteString("\n")
	wrappedHelp := lipgloss.NewStyle().Faint(true).Width(m.width - 2).Render(helpText)
	b.WriteString(wrappedHelp)
	b.WriteString("\n")

	return b.String()
}

func renderInfoBar(m chatModel) string {
	headerW := m.width
	headerStyle := lipgloss.NewStyle().
		Width(headerW).
		Background(lipgloss.Color("#027ffd")).
		Foreground(lipgloss.AdaptiveColor{Light: "236", Dark: "248"}).
		PaddingLeft(1)

	// Left/middle parts (already rendered strings)
	var left string
	if m.projectInfo != nil {
		left = fmt.Sprintf("%s %s/%s", projectPrompt, m.projectInfo.Namespace, m.projectInfo.Project)
	} else {
		left = fmt.Sprintf("%s unknown/unknown", projectPrompt)
	}
	mid := ""
	if sessionID != "" {
		mid = fmt.Sprintf(" (%s %s)", sessionPrompt, sessionID)
	}
	leftRendered := lipgloss.NewStyle().Render(left + mid)

	// Right (server status)
	right := fmt.Sprintf("%s %s", iconForStatus(func() string {
		if m.serverHealth != nil {
			return m.serverHealth.Status
		}
		return "degraded"
	}()), serverURL)

	// If headerStyle has padding/borders, subtract them
	frameW, _ := headerStyle.GetFrameSize()
	contentW := headerW - frameW

	// Give the right part the remaining width and align it
	avail := contentW - lipgloss.Width(leftRendered)
	if avail < 1 {
		avail = 1
	}

	rightWithStyle := lipgloss.NewStyle().
		Background(lipgloss.Color("#141e47")).
		Foreground(lipgloss.Color("#ffffff")).
		Padding(0, 1).
		Render(right)

	rightRendered := lipgloss.NewStyle().
		Width(avail).
		Align(lipgloss.Right).
		Render(rightWithStyle)

	// Join and render the full header line
	line := lipgloss.JoinHorizontal(lipgloss.Top, leftRendered, rightRendered)
	return headerStyle.Render(line)
}

func (m chatModel) View() string {
	var b strings.Builder

	b.WriteString(m.viewport.View())
	b.WriteString(renderChatInput(m))
	b.WriteString(renderInfoBar(m))

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
