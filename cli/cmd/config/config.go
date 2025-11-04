package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	toml "github.com/pelletier/go-toml/v2"
	yaml "gopkg.in/yaml.v2"
)

// Config file constants (searched in this order)
var (
	// SupportedLlamaFarmConfigFiles lists all supported llamafarm config file names
	SupportedLlamaFarmConfigFiles = []string{
		"llamafarm.yaml",
		"llamafarm.yml",
		"llamafarm.toml",
		"llamafarm.json",
	}
)

// LoadConfig loads a llamafarm config file from the specified directory
func LoadConfig(configDir string) (*LlamaFarmConfig, error) {
	// configDir should always be a directory path
	if configDir == "" {
		return nil, fmt.Errorf("config directory is required")
	}

	// Search for config files in the specified directory
	foundFile, err := FindConfigFile(configDir)
	if err != nil {
		return nil, fmt.Errorf("no llamafarm config file (yaml/toml/json) found in %s", configDir)
	}

	return LoadConfigFile(foundFile)
}

// LoadConfigFile loads a specific llamafarm config file
func LoadConfigFile(configFile string) (*LlamaFarmConfig, error) {
	return loadConfigFile(configFile)
}

// loadConfigFile loads and parses a config file with the given extension
func loadConfigFile(filePath string) (*LlamaFarmConfig, error) {
	// Read and parse the config file based on its extension
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file %s: %w", filePath, err)
	}

	fileExt := strings.ToLower(filepath.Ext(filePath))

	var config LlamaFarmConfig
	switch strings.ToLower(fileExt) {
	case ".yaml", ".yml":
		if err := yaml.Unmarshal(data, &config); err != nil {
			return nil, fmt.Errorf("failed to parse YAML config file %s: %w", filePath, err)
		}
	case ".toml":
		// TOML support temporarily disabled due to dependency issues
		// For now, skip TOML files and let YAML/JSON take precedence
		if err := toml.Unmarshal(data, &config); err != nil {
			return nil, fmt.Errorf("failed to parse TOML config file %s: %w", filePath, err)
		}
	case ".json":
		if err := json.Unmarshal(data, &config); err != nil {
			return nil, fmt.Errorf("failed to parse JSON config file %s: %w", filePath, err)
		}
	default:
		return nil, fmt.Errorf("unsupported config file extension: %s", fileExt)
	}

	return &config, nil
}

// FindConfigFile searches for llamafarm config files (yaml/toml/json) in the specified directory
func FindConfigFile(searchPath string) (string, error) {
	// Search for config files in the specified directory
	if searchPath == "" {
		return "", fmt.Errorf("search path is required")
	}

	for _, configFile := range SupportedLlamaFarmConfigFiles {
		fullPath := filepath.Join(searchPath, configFile)
		if _, err := os.Stat(fullPath); err == nil {
			return fullPath, nil
		}
	}
	return "", fmt.Errorf("no llamafarm config file (yaml/toml/json) found in %s", searchPath)
}

// IsConfigFile checks if the given file path is a llamafarm config file
func IsConfigFile(filePath string) bool {
	baseName := filepath.Base(filePath)

	for _, configFile := range SupportedLlamaFarmConfigFiles {
		if baseName == configFile {
			return true
		}
	}
	return false
}

// SaveConfig saves a llamafarm.yaml configuration file
func SaveConfig(config *LlamaFarmConfig, configPath string) error {
	// If no path specified, save to llamafarm.yaml in current directory
	if configPath == "" {
		configPath = "llamafarm.yaml"
	}

	data, err := yaml.Marshal(config)
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	// Create directory if it doesn't exist
	if dir := filepath.Dir(configPath); dir != "." {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return fmt.Errorf("failed to create directory: %w", err)
		}
	}

	if err := os.WriteFile(configPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write config file: %w", err)
	}

	return nil
}

// FindDatasetByName finds a dataset by name in the configuration
func (c *LlamaFarmConfig) FindDatasetByName(name string) (*LlamaFarmConfigDatasetsElem, int) {
	for i, dataset := range c.Datasets {
		if dataset.Name == name {
			return &dataset, i
		}
	}
	return nil, -1
}

// AddDataset adds a new dataset to the configuration
func (c *LlamaFarmConfig) AddDataset(dataset LlamaFarmConfigDatasetsElem) error {
	// Check if dataset with same name already exists
	if existing, _ := c.FindDatasetByName(dataset.Name); existing != nil {
		return fmt.Errorf("dataset with name '%s' already exists", dataset.Name)
	}

	c.Datasets = append(c.Datasets, dataset)
	return nil
}

// RemoveDataset removes a dataset from the configuration
func (c *LlamaFarmConfig) RemoveDataset(name string) error {
	_, index := c.FindDatasetByName(name)
	if index == -1 {
		return fmt.Errorf("dataset with name '%s' not found", name)
	}

	// Remove the dataset at the specified index
	c.Datasets = append(c.Datasets[:index], c.Datasets[index+1:]...)
	return nil
}

// ProjectInfo represents extracted namespace and project information
type ProjectInfo struct {
	Namespace string
	Project   string
}

// GetProjectInfo extracts namespace and project from the config name field
func (c *LlamaFarmConfig) GetProjectInfo() (*ProjectInfo, error) {
	name := strings.TrimSpace(c.Name)
	ns := strings.TrimSpace(c.Namespace)

	if name == "" || ns == "" {
		return nil, fmt.Errorf("both 'name' (project) and 'namespace' are required in llamafarm.yaml")
	}
	if strings.Contains(name, "/") {
		return nil, fmt.Errorf("'name' must be a project id without '/', got: %s", c.Name)
	}
	return &ProjectInfo{Namespace: ns, Project: name}, nil
}

// ServerConfig represents server connection configuration
type ServerConfig struct {
	URL       string
	Namespace string
	Project   string
}

// GetServerConfig returns server configuration with defaults applied
func GetServerConfig(configPath string, serverURL string, namespace string, project string) (*ServerConfig, error) {
	// Load configuration if available
	config, err := LoadConfig(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}

	// Apply defaults
	finalServerURL := serverURL
	if finalServerURL == "" {
		finalServerURL = "http://localhost:8000"
	}

	finalNamespace := namespace
	finalProject := project

	// Extract from config if not provided via flags
	if config != nil && (finalNamespace == "" || finalProject == "") {
		projectInfo, err := config.GetProjectInfo()
		if err != nil {
			// Don't silently ignore config errors - return them so users know what's wrong
			return nil, fmt.Errorf("failed to extract project info from config: %w", err)
		}
		if finalNamespace == "" {
			finalNamespace = projectInfo.Namespace
		}
		if finalProject == "" {
			finalProject = projectInfo.Project
		}
	}

	// Validate required fields
	if finalNamespace == "" {
		return nil, fmt.Errorf("namespace is required (provide via --namespace or set 'name' and 'namespace' in llamafarm.yaml)")
	}
	if finalProject == "" {
		return nil, fmt.Errorf("project is required (provide via --project or set 'name' and 'namespace' in llamafarm.yaml)")
	}

	return &ServerConfig{
		URL:       finalServerURL,
		Namespace: finalNamespace,
		Project:   finalProject,
	}, nil
}
