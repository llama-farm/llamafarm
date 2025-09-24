package config

import (
	"fmt"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

// GetRuntimeModels returns all runtime models
func (c *LlamaFarmConfig) GetRuntimeModels() []RuntimeModel {
	return c.RuntimeModels
}

// GetDefaultModel returns the default model name
func (c *LlamaFarmConfig) GetDefaultModel() string {
	return c.DefaultModel
}

// GetRuntime returns legacy runtime config if available
func (c *LlamaFarmConfig) GetRuntime() *RuntimeConfig {
	return c.Runtime
}

// GetRuntimeModel returns a specific runtime model by name
func (c *LlamaFarmConfig) GetRuntimeModel(name string) (*RuntimeModel, error) {
	for _, model := range c.RuntimeModels {
		if model.Name == name {
			return &model, nil
		}
	}
	return nil, fmt.Errorf("model '%s' not found", name)
}

// SetDefaultModel sets the default model
func (c *LlamaFarmConfig) SetDefaultModel(modelName string) error {
	// Verify the model exists
	found := false
	for _, model := range c.RuntimeModels {
		if model.Name == modelName {
			found = true
			break
		}
	}
	if !found {
		return fmt.Errorf("model '%s' not found in runtime_models", modelName)
	}
	c.DefaultModel = modelName
	return nil
}

// SaveToFile saves the configuration back to a file
func (c *LlamaFarmConfig) SaveToFile(filePath string) error {
	// Marshal to YAML (most common format)
	data, err := yaml.Marshal(c)
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	// Ensure directory exists
	dir := filepath.Dir(filePath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	// Write to file
	if err := os.WriteFile(filePath, data, 0644); err != nil {
		return fmt.Errorf("failed to write config file: %w", err)
	}

	return nil
}