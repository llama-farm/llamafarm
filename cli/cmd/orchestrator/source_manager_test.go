package orchestrator

import (
	"os"
	"testing"

	"github.com/llamafarm/cli/cmd/version"
)

func TestGetCurrentCLIVersion(t *testing.T) {
	tests := []struct {
		name            string
		versionVar      string
		envVar          string
		expectedVersion string
		expectError     bool
	}{
		{
			name:            "LF_VERSION_REF override takes priority",
			versionVar:      "v1.0.0",
			envVar:          "v2.0.0",
			expectedVersion: "v2.0.0",
			expectError:     false,
		},
		{
			name:            "Dev build uses main branch",
			versionVar:      "dev",
			envVar:          "",
			expectedVersion: "main",
			expectError:     false,
		},
		{
			name:            "Empty version uses main branch",
			versionVar:      "",
			envVar:          "",
			expectedVersion: "main",
			expectError:     false,
		},
		{
			name:            "Release version is used directly",
			versionVar:      "v1.2.3",
			envVar:          "",
			expectedVersion: "v1.2.3",
			expectError:     false,
		},
		{
			name:            "Version with spaces is trimmed",
			versionVar:      "  v1.2.3  ",
			envVar:          "",
			expectedVersion: "v1.2.3",
			expectError:     false,
		},
		{
			name:            "Custom branch via LF_VERSION_REF",
			versionVar:      "v1.0.0",
			envVar:          "feat-custom-branch",
			expectedVersion: "feat-custom-branch",
			expectError:     false,
		},
		{
			name:            "Commit SHA via LF_VERSION_REF",
			versionVar:      "v1.0.0",
			envVar:          "abc123def456abc123def456abc123def456abc1",
			expectedVersion: "abc123def456abc123def456abc123def456abc1",
			expectError:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Save original values
			originalVersion := version.CurrentVersion
			originalEnv := os.Getenv("LF_VERSION_REF")

			// Set test values
			version.CurrentVersion = tt.versionVar
			if tt.envVar != "" {
				os.Setenv("LF_VERSION_REF", tt.envVar)
			} else {
				os.Unsetenv("LF_VERSION_REF")
			}

			// Restore after test
			defer func() {
				version.CurrentVersion = originalVersion
				if originalEnv != "" {
					os.Setenv("LF_VERSION_REF", originalEnv)
				} else {
					os.Unsetenv("LF_VERSION_REF")
				}
			}()

			// Create a minimal SourceManager (we only need to test GetCurrentCLIVersion)
			sm := &SourceManager{}
			version, err := sm.GetCurrentCLIVersion()

			if tt.expectError && err == nil {
				t.Errorf("expected error but got none")
			}
			if !tt.expectError && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if version != tt.expectedVersion {
				t.Errorf("expected version %q, got %q", tt.expectedVersion, version)
			}
		})
	}
}

func TestGetCurrentCLIVersionIntegration(t *testing.T) {
	// This test verifies the actual behavior with the real Version variable
	// It should use "dev" by default in tests, mapping to "main"

	sm := &SourceManager{}
	version, err := sm.GetCurrentCLIVersion()

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// In tests, Version should be "dev" (default), which maps to "main"
	if version != "main" {
		t.Errorf("expected default version to be 'main', got %q", version)
	}
}
