package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v2"
)

// LlamaFarmConfig represents the complete llamafarm.yaml configuration
type LlamaFarmConfig struct {
	Version  string     `yaml:"version"`
	Name     string     `yaml:"name,omitempty"`
	Prompts  []Prompt   `yaml:"prompts,omitempty"`
	RAG      RAGConfig  `yaml:"rag,omitempty"`
	Datasets []Dataset  `yaml:"datasets,omitempty"`
	Models   []Model    `yaml:"models,omitempty"`
}

// Dataset represents a dataset configuration
type Dataset struct {
	Name   string   `yaml:"name"`
	Parser string   `yaml:"parser,omitempty"`
	Files  []string `yaml:"files"`
}

// Prompt represents a prompt configuration
type Prompt struct {
	Name        string `yaml:"name,omitempty"`
	Prompt      string `yaml:"prompt"`
	Description string `yaml:"description,omitempty"`
}

// RAGConfig represents the RAG configuration
type RAGConfig struct {
	Description         string                            `yaml:"description,omitempty"`
	Parsers             map[string]ParserConfig           `yaml:"parsers,omitempty"`
	Embedders           map[string]EmbedderConfig         `yaml:"embedders,omitempty"`
	VectorStores        map[string]VectorStoreConfig      `yaml:"vector_stores,omitempty"`
	RetrievalStrategies map[string]RetrievalStrategyConfig `yaml:"retrieval_strategies,omitempty"`
	Defaults            DefaultsConfig                    `yaml:"defaults,omitempty"`

	// Legacy fields for backwards compatibility
	Parser      ParserConfig      `yaml:"parser,omitempty"`
	Embedder    EmbedderConfig    `yaml:"embedder,omitempty"`
	VectorStore VectorStoreConfig `yaml:"vector_store,omitempty"`
}

// ParserConfig represents a parser configuration
type ParserConfig struct {
	Type           string                 `yaml:"type"`
	Config         map[string]interface{} `yaml:"config,omitempty"`
	FileExtensions []string               `yaml:"file_extensions,omitempty"`
	MimeTypes      []string               `yaml:"mime_types,omitempty"`
	Priority       int                    `yaml:"priority,omitempty"`
}

// EmbedderConfig represents an embedder configuration
type EmbedderConfig struct {
	Type   string                 `yaml:"type"`
	Config map[string]interface{} `yaml:"config,omitempty"`
}

// VectorStoreConfig represents a vector store configuration
type VectorStoreConfig struct {
	Type   string                 `yaml:"type"`
	Config map[string]interface{} `yaml:"config,omitempty"`
}

// RetrievalStrategyConfig represents a retrieval strategy configuration
type RetrievalStrategyConfig struct {
	Type        string                 `yaml:"type"`
	Config      map[string]interface{} `yaml:"config,omitempty"`
	Description string                 `yaml:"description,omitempty"`
}

// DefaultsConfig represents default component selections
type DefaultsConfig struct {
	Parser            string `yaml:"parser"`
	Embedder          string `yaml:"embedder"`
	VectorStore       string `yaml:"vector_store"`
	RetrievalStrategy string `yaml:"retrieval_strategy"`
}

// Model represents a model configuration
type Model struct {
	Provider string `yaml:"provider"`
	Model    string `yaml:"model"`
}

// LoadConfig loads a llamafarm.yaml configuration file
func LoadConfig(configPath string) (*LlamaFarmConfig, error) {
	// If no path specified, look for llamafarm.yaml in current directory
	if configPath == "" {
		if _, err := os.Stat("llamafarm.yaml"); err == nil {
			configPath = "llamafarm.yaml"
		} else if _, err := os.Stat("llamafarm.yml"); err == nil {
			configPath = "llamafarm.yml"
		} else {
			return nil, fmt.Errorf("no llamafarm.yaml file found in current directory")
		}
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var config LlamaFarmConfig
	if err := yaml.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}

	return &config, nil
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
func (c *LlamaFarmConfig) FindDatasetByName(name string) (*Dataset, int) {
	for i, dataset := range c.Datasets {
		if dataset.Name == name {
			return &dataset, i
		}
	}
	return nil, -1
}

// AddDataset adds a new dataset to the configuration
func (c *LlamaFarmConfig) AddDataset(dataset Dataset) error {
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
	if c.Name == "" {
		return nil, fmt.Errorf("project name not set in configuration")
	}

	// Split name by forward slash to get namespace/project
	parts := strings.Split(c.Name, "/")
	if len(parts) != 2 {
		return nil, fmt.Errorf("project name must be in format 'namespace/project', got: %s", c.Name)
	}

	namespace := strings.TrimSpace(parts[0])
	project := strings.TrimSpace(parts[1])

	if namespace == "" || project == "" {
		return nil, fmt.Errorf("namespace and project cannot be empty in name: %s", c.Name)
	}

	return &ProjectInfo{
		Namespace: namespace,
		Project:   project,
	}, nil
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
	var config *LlamaFarmConfig
	var err error

	if configPath != "" {
		config, err = LoadConfig(configPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load config: %w", err)
		}
	} else {
		// Try to load from default locations
		config, _ = LoadConfig("")
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
		if err == nil {
			if finalNamespace == "" {
				finalNamespace = projectInfo.Namespace
			}
			if finalProject == "" {
				finalProject = projectInfo.Project
			}
		}
	}

	// Validate required fields
	if finalNamespace == "" {
		return nil, fmt.Errorf("namespace is required (provide via --namespace flag or set 'name' in llamafarm.yaml)")
	}
	if finalProject == "" {
		return nil, fmt.Errorf("project is required (provide via --project flag or set 'name' in llamafarm.yaml)")
	}

	return &ServerConfig{
		URL:       finalServerURL,
		Namespace: finalNamespace,
		Project:   finalProject,
	}, nil
}

// GetServerConfigLenient returns server configuration, allowing empty namespace/project.
// It attempts to populate from config if available but will not error if missing.
func GetServerConfigLenient(configPath string, serverURL string, namespace string, project string) (*ServerConfig, error) {
    // Load configuration if available
    var cfg *LlamaFarmConfig
    var err error
    if configPath != "" {
        cfg, err = LoadConfig(configPath)
        if err != nil {
            return nil, fmt.Errorf("failed to load config: %w", err)
        }
    } else {
        cfg, _ = LoadConfig("")
    }

    finalServerURL := serverURL
    if finalServerURL == "" {
        finalServerURL = "http://localhost:8000"
    }

    finalNamespace := namespace
    finalProject := project

    if cfg != nil && (finalNamespace == "" || finalProject == "") {
        if projectInfo, err := cfg.GetProjectInfo(); err == nil {
            if finalNamespace == "" {
                finalNamespace = projectInfo.Namespace
            }
            if finalProject == "" {
                finalProject = projectInfo.Project
            }
        }
    }

    return &ServerConfig{
        URL:       finalServerURL,
        Namespace: finalNamespace,
        Project:   finalProject,
    }, nil
}