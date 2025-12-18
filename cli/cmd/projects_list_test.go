package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/llamafarm/cli/cmd/utils"
)

func writeProjectConfig(t *testing.T, dir, ns, name string) string {
	t.Helper()
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatalf("failed to create project dir: %v", err)
	}
	content := fmt.Sprintf("version: v1\nname: %s\nnamespace: %s\n", name, ns)
	cfgPath := filepath.Join(dir, "llamafarm.yaml")
	if err := os.WriteFile(cfgPath, []byte(content), 0644); err != nil {
		t.Fatalf("failed to write config: %v", err)
	}
	return cfgPath
}

func TestDiscoverProjectsSortsAndMarksCurrent(t *testing.T) {
	root := t.TempDir()
	nsDir := filepath.Join(root, "default")

	alphaDir := filepath.Join(nsDir, "alpha")
	betaDir := filepath.Join(nsDir, "beta")

	alphaCfg := writeProjectConfig(t, alphaDir, "default", "alpha")
	betaCfg := writeProjectConfig(t, betaDir, "default", "beta")

	older := time.Now().Add(-2 * time.Hour)
	newer := time.Now().Add(-1 * time.Hour)
	if err := os.Chtimes(alphaCfg, older, older); err != nil {
		t.Fatalf("failed to set alpha mtime: %v", err)
	}
	if err := os.Chtimes(betaCfg, newer, newer); err != nil {
		t.Fatalf("failed to set beta mtime: %v", err)
	}

	utils.OverrideCwd = betaDir
	defer func() { utils.OverrideCwd = "" }()

	projects, warnings, err := discoverProjects(root)
	if err != nil {
		t.Fatalf("discoverProjects error: %v", err)
	}
	if len(warnings) != 0 {
		t.Fatalf("expected no warnings, got %v", warnings)
	}
	if len(projects) != 2 {
		t.Fatalf("expected 2 projects, got %d", len(projects))
	}

	sortProjectsByModTime(projects)

	if projects[0].Name != "beta" || !projects[0].IsCurrent {
		t.Fatalf("expected beta to be first and marked current, got %+v", projects[0])
	}
	if projects[1].Name != "alpha" || projects[1].IsCurrent {
		t.Fatalf("expected alpha second and not current, got %+v", projects[1])
	}
}

func TestDiscoverProjectsHandlesMissingAndInvalid(t *testing.T) {
	root := t.TempDir()
	nsDir := filepath.Join(root, "default")

	// Missing config
	missingDir := filepath.Join(nsDir, "missing")
	if err := os.MkdirAll(missingDir, 0755); err != nil {
		t.Fatalf("failed to make missing dir: %v", err)
	}

	// Invalid config (no name)
	invalidDir := filepath.Join(nsDir, "invalid")
	if err := os.MkdirAll(invalidDir, 0755); err != nil {
		t.Fatalf("failed to make invalid dir: %v", err)
	}
	invalidCfg := filepath.Join(invalidDir, "llamafarm.yaml")
	if err := os.WriteFile(invalidCfg, []byte("version: v1\nnamespace: default\n"), 0644); err != nil {
		t.Fatalf("failed to write invalid config: %v", err)
	}

	// Valid config
	validDir := filepath.Join(nsDir, "valid")
	writeProjectConfig(t, validDir, "default", "valid")

	projects, warnings, err := discoverProjects(root)
	if err != nil {
		t.Fatalf("discoverProjects error: %v", err)
	}
	if len(projects) != 1 {
		t.Fatalf("expected 1 valid project, got %d", len(projects))
	}
	if projects[0].Name != "valid" || projects[0].Namespace != "default" {
		t.Fatalf("unexpected project data: %+v", projects[0])
	}
	if len(warnings) < 2 {
		t.Fatalf("expected warnings for missing and invalid configs, got %v", warnings)
	}
}

func TestFindCurrentProjectWalksParents(t *testing.T) {
	root := t.TempDir()
	cfgPath := writeProjectConfig(t, root, "ns1", "proj1")
	now := time.Now()
	if err := os.Chtimes(cfgPath, now, now); err != nil {
		t.Fatalf("failed to touch cfg: %v", err)
	}

	nestedDir := filepath.Join(root, "nested", "deep")
	if err := os.MkdirAll(nestedDir, 0755); err != nil {
		t.Fatalf("failed to make nested dir: %v", err)
	}

	info, path := findCurrentProject(nestedDir)
	if info == nil {
		t.Fatalf("expected to find project info")
	}
	if info.Namespace != "ns1" || info.Project != "proj1" {
		t.Fatalf("unexpected project info: %+v", info)
	}
	if path != filepath.Clean(root) {
		t.Fatalf("expected path %s, got %s", filepath.Clean(root), path)
	}
}
