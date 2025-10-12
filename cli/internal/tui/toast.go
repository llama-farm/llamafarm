package tui

import (
    "time"

    tea "github.com/charmbracelet/bubbletea"
    "github.com/charmbracelet/lipgloss"
)

type ToastModel struct {
    message   string
    visible   bool
    timestamp time.Time
    width     int
}

type HideToastMsg struct{}

func NewToastModel() ToastModel { return ToastModel{visible: false} }

func (m ToastModel) Update(msg tea.Msg) (ToastModel, tea.Cmd) {
    switch msg := msg.(type) {
    case ShowToastMsg:
        m.message = msg.Message
        m.visible = true
        m.timestamp = time.Now()
        return m, tea.Tick(3*time.Second, func(t time.Time) tea.Msg { return HideToastMsg{} })
    case HideToastMsg:
        m.visible = false
        return m, nil
    case tea.WindowSizeMsg:
        m.width = msg.Width
        return m, nil
    }
    return m, nil
}

func (m ToastModel) View() string {
    if !m.visible { return "" }
    toastStyle := lipgloss.NewStyle().Foreground(lipgloss.Color("230")).Background(lipgloss.Color("86")).Padding(0, 2).MarginRight(2).Bold(true)
    toast := toastStyle.Render(m.message)
    _ = toast // width computed below
    return lipgloss.PlaceHorizontal(m.width, lipgloss.Right, toast)
}


