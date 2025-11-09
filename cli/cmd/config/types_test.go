package config

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

func writeTempConfig(t *testing.T, content string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "llamafarm.yaml")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("failed to write temp config: %v", err)
	}
	return path
}

func writeTempConfigDir(t *testing.T, content string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "llamafarm.yaml")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("failed to write temp config: %v", err)
	}
	return dir
}

func writeTempTOMLConfig(t *testing.T, content string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "llamafarm.toml")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("failed to write temp TOML config: %v", err)
	}
	return path
}

func writeTempTOMLConfigDir(t *testing.T, content string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "llamafarm.toml")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("failed to write temp TOML config: %v", err)
	}
	return dir
}

func writeTempJSONConfig(t *testing.T, content string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "llamafarm.json")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("failed to write temp JSON config: %v", err)
	}
	return path
}

func writeTempJSONConfigDir(t *testing.T, content string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "llamafarm.json")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("failed to write temp JSON config: %v", err)
	}
	return dir
}

func TestGetProjectInfo(t *testing.T) {
	// Explicit fields only
	cfg := &LlamaFarmConfig{Name: "shop", Namespace: "acme"}
	pi, err := cfg.GetProjectInfo()
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if pi.Namespace != "acme" || pi.Project != "shop" {
		t.Fatalf("unexpected project info: %+v", pi)
	}

	// Negative: both missing
	cfg = &LlamaFarmConfig{}
	if _, err := cfg.GetProjectInfo(); err == nil {
		t.Fatalf("expected error when both name and namespace are missing")
	}

	// Negative: explicit with slash in name
	cfg = &LlamaFarmConfig{Name: "acme/shop", Namespace: "acme"}
	if _, err := cfg.GetProjectInfo(); err == nil {
		t.Fatalf("expected error for explicit fields when name contains slash")
	}
}

func TestGetServerConfig_Strict(t *testing.T) {
	// No config file, expect error if namespace/project missing
	_, err := GetServerConfig("", "", "", "")
	if err == nil {
		t.Fatalf("expected error when namespace/project missing")
	}

	// With config file containing explicit fields
	dir := writeTempConfigDir(t, "name: shop\nnamespace: acme\nversion: v1\n")

	// First test LoadConfig directly to isolate the issue
	config, err := LoadConfig(dir)
	if err != nil {
		t.Fatalf("LoadConfig failed: %v", err)
	}
	if config.Name != "shop" || config.Namespace != "acme" {
		t.Fatalf("LoadConfig parsed wrong values: name=%s, namespace=%s", config.Name, config.Namespace)
	}

	// Then test GetServerConfig
	sc, err := GetServerConfig(dir, "", "", "")
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if sc.URL != "http://localhost:8000" || sc.Namespace != "acme" || sc.Project != "shop" {
		t.Fatalf("unexpected server config: %+v", sc)
	}
}

func TestLoadTOMLConfig(t *testing.T) {
	// Test loading a TOML config file from directory
	tomlContent := `version = "v1"
name = "test-project"
namespace = "test-org"
`
	dir := writeTempTOMLConfigDir(t, tomlContent)

	// Test loading the TOML config
	config, err := LoadConfig(dir)
	if err != nil {
		t.Fatalf("failed to load TOML config: %v", err)
	}

	// Verify the parsed content
	if config.Version != "v1" {
		t.Errorf("expected version 'v1', got '%s'", config.Version)
	}
	if config.Name != "test-project" {
		t.Errorf("expected name 'test-project', got '%s'", config.Name)
	}
	if config.Namespace != "test-org" {
		t.Errorf("expected namespace 'test-org', got '%s'", config.Namespace)
	}
}

func TestLoadJSONConfig(t *testing.T) {
	// Test loading a JSON config file from directory
	jsonContent := `{
		"version": "v1",
		"name": "test_project_json",
		"namespace": "test-org-json",
		"datasets": [
			{
				"name": "test_dataset",
				"database": "main_database",
				"data_processing_strategy": "universal_processor",
				"files": []
			}
		],
		"prompts": [
			{
				"name": "test_prompt",
				"messages": [
					{
						"role": "system",
						"content": "You are a helpful AI assistant."
					}
				]
			}
		],
		"runtime": {
			"models": [
				{
					"name": "default",
					"description": "Default model",
					"provider": "ollama",
					"model": "llama3.2"
				}
			]
		}
	}`
	dir := writeTempJSONConfigDir(t, jsonContent)

	// Test loading the JSON config
	config, err := LoadConfig(dir)
	if err != nil {
		t.Fatalf("failed to load JSON config: %v", err)
	}

	// Verify the parsed content
	if config.Version != "v1" {
		t.Errorf("expected version 'v1', got '%s'", config.Version)
	}
	if config.Name != "test_project_json" {
		t.Errorf("expected name 'test_project_json', got '%s'", config.Name)
	}
	if config.Namespace != "test-org-json" {
		t.Errorf("expected namespace 'test-org-json', got '%s'", config.Namespace)
	}
}

func TestFindConfigFile(t *testing.T) {
	// Test that FindConfigFile works correctly
	dir := t.TempDir()

	// Create a YAML config file
	yamlContent := "version: v1\nname: test-project\nnamespace: test-org\n"
	yamlPath := filepath.Join(dir, "llamafarm.yaml")
	if err := os.WriteFile(yamlPath, []byte(yamlContent), 0o644); err != nil {
		t.Fatalf("failed to write YAML file: %v", err)
	}

	// Test that FindConfigFile finds the YAML file
	found, err := FindConfigFile(dir)
	if err != nil {
		t.Fatalf("failed to find config file: %v", err)
	}
	if found != yamlPath {
		t.Errorf("expected to find %s, got %s", yamlPath, found)
	}
}

func TestInvalidTOMLConfig(t *testing.T) {
	// Test error handling for invalid TOML
	invalidTOML := `version = "v1"
name = "test"
[invalid
missing closing bracket`
	dir := writeTempTOMLConfigDir(t, invalidTOML)

	_, err := LoadConfig(dir)
	if err == nil {
		t.Fatal("expected error for invalid TOML, but got none")
	}
}

func TestInvalidJSONConfig(t *testing.T) {
	// Test error handling for invalid JSON
	invalidJSON := `{"version": "v1", "name": "test", invalid json}`
	dir := writeTempJSONConfigDir(t, invalidJSON)

	_, err := LoadConfig(dir)
	if err == nil {
		t.Fatal("expected error for invalid JSON, but got none")
	}
}

func TestInvalidYAMLConfig(t *testing.T) {
	// Test error handling for invalid YAML
	invalidYAML := `version: v1
name: test
invalid yaml: [ unclosed bracket`
	dir := writeTempConfigDir(t, invalidYAML)

	_, err := LoadConfig(dir)
	if err == nil {
		t.Fatal("expected error for invalid YAML, but got none")
	}
}

func TestIsConfigFile(t *testing.T) {
	// Test that IsConfigFile correctly identifies config files
	tests := []struct {
		filePath string
		expected bool
	}{
		{"llamafarm.yaml", true},
		{"llamafarm.yml", true},
		{"llamafarm.toml", true},
		{"llamafarm.json", true},
		{"other.yaml", false},
		{"config.toml", false},
		{"llamafarm.txt", false},
		{"llamafarm.yaml.bak", false},
		{"", false},
		{"llamafarm", false},
	}

	for _, test := range tests {
		result := IsConfigFile(test.filePath)
		if result != test.expected {
			t.Errorf("IsConfigFile(%q) = %v, expected %v", test.filePath, result, test.expected)
		}
	}
}

// TestAtomicWriteFile verifies that atomic writes prevent readers from seeing partial writes
func TestAtomicWriteFile(t *testing.T) {
	dir := t.TempDir()
	testFile := filepath.Join(dir, "test.yaml")

	// Write initial content
	initialContent := []byte("initial content")
	if err := atomicWriteFile(testFile, initialContent, 0644); err != nil {
		t.Fatalf("Failed to write initial content: %v", err)
	}

	// Verify initial content
	data, err := os.ReadFile(testFile)
	if err != nil {
		t.Fatalf("Failed to read file: %v", err)
	}
	if string(data) != string(initialContent) {
		t.Errorf("Initial content mismatch: got %q, want %q", string(data), string(initialContent))
	}

	// Test concurrent writes and reads to ensure no partial reads
	var wg sync.WaitGroup
	errChan := make(chan error, 100)

	// Start multiple writers
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			content := []byte("content from writer " + string(rune('0'+id)))
			if err := atomicWriteFile(testFile, content, 0644); err != nil {
				errChan <- err
			}
		}(i)
	}

	// Start multiple readers
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			data, err := os.ReadFile(testFile)
			if err != nil {
				errChan <- err
				return
			}
			// Verify that we always read complete content (starts with "content from writer" or "initial content")
			content := string(data)
			if content != "initial content" && len(content) < len("content from writer ") {
				errChan <- fmt.Errorf("partial read detected: got %q (len %d)", content, len(content))
			}
		}()
	}

	wg.Wait()
	close(errChan)

	// Check for errors
	for err := range errChan {
		if err != nil {
			t.Errorf("Concurrent operation error: %v", err)
		}
	}

	// Verify file still exists and is valid
	if _, err := os.Stat(testFile); err != nil {
		t.Errorf("File does not exist after concurrent operations: %v", err)
	}
}

// TestSaveConfigAtomic verifies that SaveConfig uses atomic writes
func TestSaveConfigAtomic(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "llamafarm.yaml")

	cfg := &LlamaFarmConfig{
		Name:      "test-project",
		Namespace: "test-namespace",
		Version:   "v1",
	}

	// Save config
	if err := SaveConfig(cfg, configPath); err != nil {
		t.Fatalf("Failed to save config: %v", err)
	}

	// Verify config can be loaded
	loadedCfg, err := LoadConfigFile(configPath)
	if err != nil {
		t.Fatalf("Failed to load config: %v", err)
	}

	if loadedCfg.Name != cfg.Name || loadedCfg.Namespace != cfg.Namespace {
		t.Errorf("Loaded config doesn't match: got {name: %s, namespace: %s}, want {name: %s, namespace: %s}",
			loadedCfg.Name, loadedCfg.Namespace, cfg.Name, cfg.Namespace)
	}

	// Test concurrent saves and loads
	var wg sync.WaitGroup
	errChan := make(chan error, 50)

	// Multiple writers
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			cfg := &LlamaFarmConfig{
				Name:      "test-project",
				Namespace: "test-namespace",
				Version:   "v1",
			}
			if err := SaveConfig(cfg, configPath); err != nil {
				errChan <- err
			}
		}(i)
	}

	// Multiple readers
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			cfg, err := LoadConfigFile(configPath)
			if err != nil {
				errChan <- err
				return
			}
			// Verify we always get valid config with required fields
			if cfg.Name == "" || cfg.Namespace == "" {
				errChan <- err
			}
		}()
	}

	wg.Wait()
	close(errChan)

	// Check for errors
	for err := range errChan {
		if err != nil {
			t.Errorf("Concurrent save/load error: %v", err)
		}
	}
}
