package cmd

import (
	"testing"
)

func TestValidateAddonName(t *testing.T) {
	tests := []struct {
		name      string
		addonName string
		wantError bool
	}{
		{"valid lowercase", "stt", false},
		{"valid with hyphen", "my-addon", false},
		{"valid with underscore", "my_addon", false},
		{"valid alphanumeric", "addon123", false},
		{"empty name", "", true},
		{"uppercase", "STT", true},
		{"special chars", "addon@test", true},
		{"spaces", "my addon", true},
		{"too long", "a123456789012345678901234567890123456789012345678901", true},
		{"path traversal attempt", "../addon", true},
		{"with slash", "addon/test", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateAddonName(tt.addonName)
			if (err != nil) != tt.wantError {
				t.Errorf("validateAddonName(%q) error = %v, wantError %v", tt.addonName, err, tt.wantError)
			}
		})
	}
}

func TestResolveDependencies(t *testing.T) {
	// Build an in-memory registry for testing (no global state needed)
	registry := &AddonRegistryStore{
		addons: map[string]*AddonDefinition{
			"base": {
				Name:         "base",
				Dependencies: []string{},
			},
			"mid": {
				Name:         "mid",
				Dependencies: []string{"base"},
			},
			"top": {
				Name:         "top",
				Dependencies: []string{"mid"},
			},
			"circular1": {
				Name:         "circular1",
				Dependencies: []string{"circular2"},
			},
			"circular2": {
				Name:         "circular2",
				Dependencies: []string{"circular1"},
			},
		},
	}

	tests := []struct {
		name         string
		addonName    string
		installed    []string
		wantOrder    []string
		wantError    bool
		errorPattern string
	}{
		{
			name:      "no dependencies",
			addonName: "base",
			installed: []string{},
			wantOrder: []string{"base"},
			wantError: false,
		},
		{
			name:      "single dependency",
			addonName: "mid",
			installed: []string{},
			wantOrder: []string{"base", "mid"},
			wantError: false,
		},
		{
			name:      "nested dependencies",
			addonName: "top",
			installed: []string{},
			wantOrder: []string{"base", "mid", "top"},
			wantError: false,
		},
		{
			name:      "skip already installed",
			addonName: "top",
			installed: []string{"base"},
			wantOrder: []string{"mid", "top"},
			wantError: false,
		},
		{
			name:      "already installed",
			addonName: "base",
			installed: []string{"base"},
			wantOrder: []string{},
			wantError: false,
		},
		{
			name:         "circular dependency",
			addonName:    "circular1",
			installed:    []string{},
			wantError:    true,
			errorPattern: "circular dependency",
		},
		{
			name:         "unknown addon",
			addonName:    "unknown",
			installed:    []string{},
			wantError:    true,
			errorPattern: "unknown addon",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Setup state
			state := &AddonsState{
				Version:         "1",
				InstalledAddons: make(map[string]*InstalledAddon),
			}
			for _, name := range tt.installed {
				state.InstalledAddons[name] = &InstalledAddon{Name: name}
			}

			// Run test
			order, err := resolveDependencies(
				registry,
				tt.addonName,
				state,
				make(map[string]bool),
				make(map[string]bool),
			)

			// Check error
			if (err != nil) != tt.wantError {
				t.Errorf("resolveDependencies() error = %v, wantError %v", err, tt.wantError)
				return
			}

			// Check order
			if !tt.wantError {
				if len(order) != len(tt.wantOrder) {
					t.Errorf("resolveDependencies() order length = %d, want %d", len(order), len(tt.wantOrder))
					return
				}
				for i, name := range order {
					if name != tt.wantOrder[i] {
						t.Errorf("resolveDependencies() order[%d] = %s, want %s", i, name, tt.wantOrder[i])
					}
				}
			}
		})
	}
}

func TestGetPlatformString(t *testing.T) {
	// Test that platform string is generated correctly
	platform := getPlatformString()

	// Should be in format: os-arch
	if len(platform) == 0 {
		t.Error("getPlatformString() returned empty string")
	}

	// Should not contain uppercase or spaces
	for _, c := range platform {
		if c >= 'A' && c <= 'Z' {
			t.Errorf("getPlatformString() contains uppercase: %s", platform)
		}
		if c == ' ' {
			t.Errorf("getPlatformString() contains space: %s", platform)
		}
	}

	// Should contain a hyphen
	found := false
	for _, c := range platform {
		if c == '-' {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("getPlatformString() missing hyphen separator: %s", platform)
	}
}

func TestAddonsState(t *testing.T) {
	state := &AddonsState{
		Version:         "1",
		InstalledAddons: make(map[string]*InstalledAddon),
	}

	// Test IsAddonInstalled
	if state.IsAddonInstalled("stt") {
		t.Error("IsAddonInstalled() should return false for uninstalled addon")
	}

	// Test MarkInstalled
	state.MarkInstalled("stt", "1.0.0", "universal-runtime", "macos-arm64")
	if !state.IsAddonInstalled("stt") {
		t.Error("IsAddonInstalled() should return true after MarkInstalled")
	}

	addon := state.InstalledAddons["stt"]
	if addon.Name != "stt" {
		t.Errorf("addon.Name = %s, want stt", addon.Name)
	}
	if addon.Version != "1.0.0" {
		t.Errorf("addon.Version = %s, want 1.0.0", addon.Version)
	}
	if addon.Component != "universal-runtime" {
		t.Errorf("addon.Component = %s, want universal-runtime", addon.Component)
	}
	if addon.Platform != "macos-arm64" {
		t.Errorf("addon.Platform = %s, want macos-arm64", addon.Platform)
	}

	// Test MarkUninstalled
	state.MarkUninstalled("stt")
	if state.IsAddonInstalled("stt") {
		t.Error("IsAddonInstalled() should return false after MarkUninstalled")
	}
}
