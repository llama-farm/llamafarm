package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadSchemaRef_YAML(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "lf-config-schema-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	configPath := filepath.Join(tempDir, "llamafarm.yaml")
	content := `version: v1
name: demo
namespace: default
schema: schemas/person.py::Person
runtime:
  models:
    - name: default
      provider: ollama
      model: llama3
`
	if err := os.WriteFile(configPath, []byte(content), 0644); err != nil {
		t.Fatalf("failed to write config: %v", err)
	}

	got, err := LoadSchemaRef(tempDir)
	if err != nil {
		t.Fatalf("LoadSchemaRef returned error: %v", err)
	}
	if got != "schemas/person.py::Person" {
		t.Fatalf("expected schema ref %q, got %q", "schemas/person.py::Person", got)
	}
}

func TestLoadSchemaRef_MissingSchemaReturnsEmpty(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "lf-config-schema-empty-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	configPath := filepath.Join(tempDir, "llamafarm.yaml")
	content := `version: v1
name: demo
namespace: default
runtime:
  models:
    - name: default
      provider: ollama
      model: llama3
`
	if err := os.WriteFile(configPath, []byte(content), 0644); err != nil {
		t.Fatalf("failed to write config: %v", err)
	}

	got, err := LoadSchemaRef(tempDir)
	if err != nil {
		t.Fatalf("LoadSchemaRef returned error: %v", err)
	}
	if got != "" {
		t.Fatalf("expected empty schema ref, got %q", got)
	}
}

func TestLoadSchemaRef_NoConfig(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "lf-config-schema-missing-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	if _, err := LoadSchemaRef(tempDir); err == nil {
		t.Fatalf("expected error when no config file exists")
	}
}
