package cmd

import (
	"bufio"
	"fmt"
	"llamafarm-cli/cmd/config"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

var (
	farmerPrompt  = "🌾 Farmer:"
	serverPrompt  = "📡 Server:"
	projectPrompt = "📁 Project:"
	sessionPrompt = "🆔 Session:"
)

var chatCtx = &ChatSessionContext{
	ServerURL:   serverURL,
	Namespace:   "llamafarm",
	ProjectID:   "project-seed",
	Temperature: temperature,
	MaxTokens:   maxTokens,
	HTTPClient:  getHTTPClient(),
}

// runChatSessionTUI starts the Bubble Tea TUI for chat.
func runChatSessionTUI(projectInfo *config.ProjectInfo) {
	m := newChatModel(projectInfo)
	p := tea.NewProgram(m)
	m.program = p
	if _, err := p.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Error running TUI: %v\n", err)
	}
}

type chatModel struct {
	projectInfo *config.ProjectInfo
	input       textinput.Model
	spin        spinner.Model
	transcript  []string
	messages    []ChatMessage
	thinking    bool
	printing    bool
	thinkFrame  int
	history     []string
	histIndex   int
	historyPath string
	width       int
	height      int
	status      string
	err         error
	program     *tea.Program
	streamCh    chan tea.Msg
}

type (
	streamDone struct{}
)

type responseMsg struct{ content string }
type errorMsg struct{ err error }
type tickMsg struct{}

func newChatModel(projectInfo *config.ProjectInfo) chatModel {
	in := textinput.New()
	in.Placeholder = "Type a message"
	in.Prompt = "You> "
	in.Focus()
	in.CharLimit = 0
	// Set a sensible initial width so the placeholder isn't truncated
	in.Width = 60

	s := spinner.New()
	s.Spinner = spinner.Dot
	s.Style = lipgloss.NewStyle().Foreground(lipgloss.Color("11"))
	hPath := getHistoryPath()
	h := loadHistory(hPath)
	return chatModel{
		projectInfo: projectInfo,
		input:       in,
		spin:        s,
		transcript:  []string{},
		messages:    []ChatMessage{},
		thinking:    false,
		printing:    false,
		history:     h,
		histIndex:   len(h),
		historyPath: hPath,
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
	return m.spin.Tick
}

func (m chatModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		w := max(msg.Width-2, 10)
		m.input.Width = w
		return m, nil
	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c":
			m.status = "👋 You have left the pasture. Safe travels, little llama!"
			return m, tea.Quit

		case "up":
			if m.histIndex > 0 {
				m.histIndex--
				m.input.SetValue(m.history[m.histIndex])
				m.input.CursorEnd()
			}
			return m, nil

		case "down":
			if m.histIndex < len(m.history)-1 {
				m.histIndex++
				m.input.SetValue(m.history[m.histIndex])
				m.input.CursorEnd()
			} else {
				m.histIndex = len(m.history)
				m.input.SetValue("")
			}
			return m, nil

		case "enter":
			msg := strings.TrimSpace(m.input.Value())
			if msg == "" {
				return m, nil
			}

			lower := strings.ToLower(msg)
			if lower == "exit" || lower == "quit" {
				m.status = "👋 You have left the pasture. Safe travels, little llama!"
				return m, tea.Quit
			}

			if lower == "clear" {
				m.transcript = nil
				m.messages = nil
				m.input.SetValue("")
				m.thinking = false
				m.printing = false
				return m, nil
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
			m.transcript = append(m.transcript, lipgloss.NewStyle().Bold(true).Render("You:")+" "+msg)
			m.messages = []ChatMessage{{Role: "user", Content: msg}}
			m.input.SetValue("")
			m.thinking = true
			m.printing = true
			// Start channel-based streaming
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
			return m, tea.Batch(listen(m.streamCh), thinkingCmd())
		}
	case responseMsg:
		logDebug(fmt.Sprintf("RESPONSE MSG: %v", msg.content))
		m.thinking = false
		m.printing = true
		if len(m.messages) == 0 || (len(m.messages) > 0 && m.messages[len(m.messages)-1].Role != "assistant") {
			m.transcript = append(m.transcript, lipgloss.NewStyle().Foreground(lipgloss.Color("11")).Render(farmerPrompt)+" "+msg.content)
		} else {
			// Update last assistant line
			if len(m.transcript) > 0 {
				m.transcript[len(m.transcript)-1] = lipgloss.NewStyle().Foreground(lipgloss.Color("11")).Render(farmerPrompt) + " " + msg.content
			} else {
				m.transcript = append(m.transcript, lipgloss.NewStyle().Foreground(lipgloss.Color("11")).Render(farmerPrompt)+" "+msg.content)
			}
		}
		// Keep a single assistant message representing the latest full content
		if len(m.messages) > 0 && m.messages[len(m.messages)-1].Role == "assistant" {
			m.messages[len(m.messages)-1] = ChatMessage{Role: "assistant", Content: msg.content}
		} else {
			m.messages = append(m.messages, ChatMessage{Role: "assistant", Content: msg.content})
		}
		if m.streamCh != nil {
			return m, listen(m.streamCh)
		}
		return m, nil

	case errorMsg:
		m.thinking = false
		m.transcript = append(m.transcript, lipgloss.NewStyle().Foreground(lipgloss.Color("9")).Render(fmt.Sprintf("Error: %v", msg.err)))
		if m.streamCh != nil {
			return m, listen(m.streamCh)
		}
		return m, nil

	case tickMsg:
		if m.thinking {
			m.thinkFrame = (m.thinkFrame + 1) % 3
			return m, thinkingCmd()
		}
		return m, nil

	case streamDone:
		logDebug(fmt.Sprintf("STREAM DONE: %v", m.transcript))
		m.printing = false
		m.streamCh = nil
		return m, nil
	}

	var cmds []tea.Cmd
	var cmd tea.Cmd
	m.input, cmd = m.input.Update(msg)
	cmds = append(cmds, cmd)
	m.spin, cmd = m.spin.Update(msg)
	cmds = append(cmds, cmd)
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

func (m chatModel) View() string {
	var b strings.Builder
	serverLine := serverPrompt + " " + serverURL
	wrappedServer := lipgloss.NewStyle().Foreground(lipgloss.Color("13")).Width(m.width - 2).Render(serverLine)
	b.WriteString(wrappedServer + "\n")

	projectLine := projectPrompt + " " + m.projectInfo.Namespace + "/" + m.projectInfo.Project
	wrappedProject := lipgloss.NewStyle().Foreground(lipgloss.Color("13")).Width(m.width - 2).Render(projectLine)
	b.WriteString(wrappedProject + "\n")

	if m.err != nil {
		errorText := fmt.Sprintf("We had some trouble: %v", m.err)
		wrappedError := lipgloss.NewStyle().Width(m.width - 2).Render(errorText)
		return "\n" + wrappedError + "\n\n"
	}

	if sessionID != "" {
		sessionLine := sessionPrompt + " " + sessionID
		wrappedSession := lipgloss.NewStyle().Foreground(lipgloss.Color("13")).Width(m.width - 2).Render(sessionLine)
		b.WriteString(wrappedSession + "\n")
	}
	b.WriteString("\n")
	for _, line := range m.transcript {
		// Wrap the line to fit within the terminal width
		wrappedLine := lipgloss.NewStyle().Width(m.width - 2).Render(line)
		b.WriteString(wrappedLine + "\n\n")
	}
	if m.thinking {
		dots := m.thinkFrame + 1
		thinkingText := farmerPrompt + " " + m.spin.View() + "Thinking" + strings.Repeat(".", dots)
		wrappedThinking := lipgloss.NewStyle().Foreground(lipgloss.Color("11")).Width(m.width - 2).Render(thinkingText)
		b.WriteString(wrappedThinking + "\n\n")
	}

	if !m.thinking && !m.printing {
		b.WriteString(m.input.View())
		b.WriteString("\n")
		helpText := "Type 'exit' to quit, 'clear' to reset. Up/Down for history."
		wrappedHelp := lipgloss.NewStyle().Faint(true).Width(m.width - 2).Render(helpText)
		b.WriteString(wrappedHelp)
	}
	return b.String()
}

func thinkingCmd() tea.Cmd {
	return tea.Tick(250*time.Millisecond, func(time.Time) tea.Msg { return tickMsg{} })
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
