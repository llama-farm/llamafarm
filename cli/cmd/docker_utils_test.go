package cmd

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDockerUtils_WithFakeDocker(t *testing.T) {
	// create a fake docker executable in a temp dir and prepend to PATH
	dir, err := os.MkdirTemp("", "fakedocker")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(dir)

	script := `#!/bin/sh
arg1="$1"
# handle docker --version
if [ "$arg1" = "--version" ]; then
  echo "Docker version fake"
  exit 0
fi
# handle docker ps ...
if [ "$arg1" = "ps" ]; then
  has_a=0
  for a in "$@"; do
    if [ "$a" = "-a" ]; then
      has_a=1
    fi
  done
  if [ $has_a -eq 1 ]; then
    printf "foo\nbar\n"
  else
    printf "bar\n"
  fi
  exit 0
fi
# handle docker pull <image>
if [ "$arg1" = "pull" ]; then
  echo "Pulled $2"
  exit 0
fi
# unknown
exit 1
`

	path := filepath.Join(dir, "docker")
	if err := os.WriteFile(path, []byte(script), 0755); err != nil {
		t.Fatalf("failed to write fake docker: %v", err)
	}

	oldPath := os.Getenv("PATH")
	defer os.Setenv("PATH", oldPath)
	if err := os.Setenv("PATH", dir+string(os.PathListSeparator)+oldPath); err != nil {
		t.Fatalf("failed to set PATH: %v", err)
	}

	// ensureDockerAvailable should succeed
	if err := ensureDockerAvailable(); err != nil {
		t.Fatalf("ensureDockerAvailable failed with fake docker: %v", err)
	}

	// containerExists should see 'foo' and 'bar' in ps -a output
	if !containerExists("foo") {
		t.Fatalf("expected containerExists to find 'foo'")
	}
	if !containerExists("bar") {
		t.Fatalf("expected containerExists to find 'bar'")
	}
	if containerExists("baz") {
		t.Fatalf("did not expect containerExists to find 'baz'")
	}

	// isContainerRunning should only see 'bar' in running list
	if !isContainerRunning("bar") {
		t.Fatalf("expected isContainerRunning to find 'bar'")
	}
	if isContainerRunning("foo") {
		t.Fatalf("did not expect isContainerRunning to find 'foo' in running list")
	}

	// pullImage should succeed
	if err := pullImage("ghcr.io/example/image:latest"); err != nil {
		t.Fatalf("pullImage failed: %v", err)
	}
}

func TestResolveImageTag(t *testing.T) {
	// Save original version and environment
	originalVersion := Version
	defer func() { Version = originalVersion }()

	// Clear environment variables
	os.Unsetenv("LF_IMAGE_TAG")
	os.Unsetenv("LF_SERVER_IMAGE_TAG")
	os.Unsetenv("LF_DESIGNER_IMAGE_TAG")

	tests := []struct {
		name              string
		version           string
		component         string
		defaultTag        string
		globalOverride    string
		componentOverride string
		expectedTag       string
	}{
		{
			name:        "semantic version",
			version:     "v1.0.0",
			component:   "server",
			defaultTag:  "latest",
			expectedTag: "v1.0.0",
		},
		{
			name:        "prerelease version",
			version:     "v1.0.0-rc1",
			component:   "server",
			defaultTag:  "latest",
			expectedTag: "v1.0.0-rc1",
		},
		{
			name:        "prerelease with build metadata",
			version:     "v1.0.0-beta.1+build.123",
			component:   "server",
			defaultTag:  "latest",
			expectedTag: "v1.0.0-beta.1+build.123",
		},
		{
			name:        "dev version",
			version:     "dev",
			component:   "server",
			defaultTag:  "fallback",
			expectedTag: "latest",
		},
		{
			name:        "non-version string",
			version:     "custom-build",
			component:   "server",
			defaultTag:  "fallback",
			expectedTag: "fallback",
		},
		{
			name:        "empty version",
			version:     "",
			component:   "server",
			defaultTag:  "fallback",
			expectedTag: "fallback",
		},
		{
			name:           "global override takes precedence",
			version:        "v1.0.0",
			component:      "server",
			defaultTag:     "latest",
			globalOverride: "custom-global",
			expectedTag:    "custom-global",
		},
		{
			name:              "component override takes precedence over global",
			version:           "v1.0.0",
			component:         "server",
			defaultTag:        "latest",
			globalOverride:    "custom-global",
			componentOverride: "custom-component",
			expectedTag:       "custom-component",
		},
		{
			name:              "component override takes precedence over version",
			version:           "v1.0.0",
			component:         "designer",
			defaultTag:        "latest",
			componentOverride: "custom-designer",
			expectedTag:       "custom-designer",
		},
		{
			name:        "semantic version without v prefix",
			version:     "1.0.0",
			component:   "server",
			defaultTag:  "latest",
			expectedTag: "v1.0.0",
		},
		{
			name:        "prerelease version without v prefix",
			version:     "2.1.0-rc1",
			component:   "server",
			defaultTag:  "latest",
			expectedTag: "v2.1.0-rc1",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Set up version
			Version = tt.version

			// Set up environment variables
			if tt.globalOverride != "" {
				os.Setenv("LF_IMAGE_TAG", tt.globalOverride)
				defer os.Unsetenv("LF_IMAGE_TAG")
			}
			if tt.componentOverride != "" {
				envVar := "LF_" + strings.ToUpper(tt.component) + "_IMAGE_TAG"
				os.Setenv(envVar, tt.componentOverride)
				defer os.Unsetenv(envVar)
			}

			// Test the function
			result := resolveImageTag(tt.component, tt.defaultTag)
			if result != tt.expectedTag {
				t.Errorf("resolveImageTag(%q, %q) = %q, want %q", tt.component, tt.defaultTag, result, tt.expectedTag)
			}
		})
	}
}

func TestGetImageURL(t *testing.T) {
	// Save original version
	originalVersion := Version
	defer func() { Version = originalVersion }()

	// Clear environment variables
	os.Unsetenv("LF_IMAGE_TAG")
	os.Unsetenv("LF_SERVER_IMAGE_TAG")
	os.Unsetenv("LF_DESIGNER_IMAGE_TAG")
	os.Unsetenv("LF_RAG_IMAGE_TAG")

	tests := []struct {
		name              string
		version           string
		component         string
		globalOverride    string
		componentOverride string
		expectedURL       string
		shouldError       bool
		expectedError     string
	}{
		{
			name:        "server with semantic version",
			version:     "v1.2.3",
			component:   "server",
			expectedURL: "ghcr.io/llama-farm/llamafarm/server:v1.2.3",
		},
		{
			name:        "designer with dev version",
			version:     "dev",
			component:   "designer",
			expectedURL: "ghcr.io/llama-farm/llamafarm/designer:latest",
		},
		{
			name:        "rag with prerelease version",
			version:     "v2.0.0-alpha.1",
			component:   "rag",
			expectedURL: "ghcr.io/llama-farm/llamafarm/rag:v2.0.0-alpha.1",
		},
		{
			name:        "version without v prefix gets normalized",
			version:     "1.0.0",
			component:   "server",
			expectedURL: "ghcr.io/llama-farm/llamafarm/server:v1.0.0",
		},
		{
			name:           "global override applies",
			version:        "v1.0.0",
			component:      "server",
			globalOverride: "custom-global",
			expectedURL:    "ghcr.io/llama-farm/llamafarm/server:custom-global",
		},
		{
			name:              "component override takes precedence",
			version:           "v1.0.0",
			component:         "designer",
			globalOverride:    "custom-global",
			componentOverride: "custom-designer",
			expectedURL:       "ghcr.io/llama-farm/llamafarm/designer:custom-designer",
		},
		{
			name:              "component override without global",
			version:           "v1.0.0",
			component:         "rag",
			componentOverride: "custom-rag",
			expectedURL:       "ghcr.io/llama-farm/llamafarm/rag:custom-rag",
		},
		{
			name:          "unknown component returns error",
			version:       "v1.0.0",
			component:     "unknown",
			shouldError:   true,
			expectedError: "unknown component 'unknown'",
		},
		{
			name:          "invalid component returns error",
			version:       "v1.0.0",
			component:     "invalid-component",
			shouldError:   true,
			expectedError: "unknown component 'invalid-component'",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Set up version
			Version = tt.version

			// Set up environment variables
			if tt.globalOverride != "" {
				os.Setenv("LF_IMAGE_TAG", tt.globalOverride)
				defer os.Unsetenv("LF_IMAGE_TAG")
			}
			if tt.componentOverride != "" {
				envVar := "LF_" + strings.ToUpper(tt.component) + "_IMAGE_TAG"
				os.Setenv(envVar, tt.componentOverride)
				defer os.Unsetenv(envVar)
			}

			// Test the function
			result, err := getImageURL(tt.component)

			if tt.shouldError {
				if err == nil {
					t.Errorf("getImageURL(%q) expected error but got none", tt.component)
					return
				}
				if !strings.Contains(err.Error(), tt.expectedError) {
					t.Errorf("getImageURL(%q) error = %q, want error containing %q", tt.component, err.Error(), tt.expectedError)
				}
				return
			}

			if err != nil {
				t.Errorf("getImageURL(%q) unexpected error: %v", tt.component, err)
				return
			}

			if result != tt.expectedURL {
				t.Errorf("getImageURL(%q) = %q, want %q", tt.component, result, tt.expectedURL)
			}
		})
	}
}
