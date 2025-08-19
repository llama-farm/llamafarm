package cmd

import (
	"os"
	"path/filepath"
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
