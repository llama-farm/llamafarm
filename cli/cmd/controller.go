package cmd

import (
	tea "github.com/charmbracelet/bubbletea"
)

// State represents the minimal shared application state decoupled from the UI model.
type State struct {
	CurrentDatabase string
	CurrentStrategy string
	ServerHealth    *HealthPayload
}

// StateUpdateMsg is emitted by the controller to notify the UI of state changes.
type StateUpdateMsg struct {
	NewState State
	Notice   string
}

// Controller owns data/state updates and produces Tea messages for the UI.
type Controller struct {
	state State
}

func NewController(initial State) *Controller {
	return &Controller{state: initial}
}

// SwitchDatabase updates the selected database and resolves a compatible strategy.
func (c *Controller) SwitchDatabase(newDatabase string, available *DatabasesResponse) tea.Cmd {
	oldDatabase := c.state.CurrentDatabase
	newStrategy := c.state.CurrentStrategy
	noStrategiesForDB := false

	if available != nil {
		for _, db := range available.Databases {
			if db.Name == newDatabase {
				if len(db.RetrievalStrategies) == 0 {
					noStrategiesForDB = true
				}
				// Validate current strategy for this database
				strategyValid := false
				if newStrategy != "" {
					for _, s := range db.RetrievalStrategies {
						if s.Name == newStrategy {
							strategyValid = true
							break
						}
					}
				}
				if !strategyValid {
					// Try default, else first
					newStrategy = ""
					for _, s := range db.RetrievalStrategies {
						if s.IsDefault {
							newStrategy = s.Name
							break
						}
					}
					if newStrategy == "" && len(db.RetrievalStrategies) > 0 {
						newStrategy = db.RetrievalStrategies[0].Name
					}
				}
				break
			}
		}
	}

	c.state.CurrentDatabase = newDatabase
	c.state.CurrentStrategy = newStrategy

	notice := "Switched from database '" + oldDatabase + "' to '" + newDatabase + "' with strategy '" + newStrategy + "'"
	if noStrategiesForDB {
		notice += "\nDatabase '" + newDatabase + "' has no retrieval strategies configured."
	}

	return func() tea.Msg { return StateUpdateMsg{NewState: c.state, Notice: notice} }
}

// UpdateServerHealth stores the latest server health snapshot and notifies the UI.
func (c *Controller) UpdateServerHealth(h *HealthPayload) tea.Cmd {
	c.state.ServerHealth = h
	return func() tea.Msg { return StateUpdateMsg{NewState: c.state} }
}
