package config

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

// LlamaFarmConfig represents the complete llamafarm configuration
type LlamaFarmConfig struct {
	Version   string         `yaml:"version" toml:"version" json:"version"`
	Name      string         `yaml:"name" toml:"name" json:"name"`
	Namespace string         `yaml:"namespace" toml:"namespace" json:"namespace"`
	Runtime   RuntimeConfig  `yaml:"runtime,omitempty" toml:"runtime,omitempty" json:"runtime,omitempty"`
	Prompts   []Prompt       `yaml:"prompts,omitempty" toml:"prompts,omitempty" json:"prompts,omitempty"`
	RAG       RAGConfig      `yaml:"rag,omitempty" toml:"rag,omitempty" json:"rag,omitempty"`
	Datasets  []Dataset      `yaml:"datasets,omitempty" toml:"datasets,omitempty" json:"datasets,omitempty"`
	MCP       *MCPConfig     `yaml:"mcp,omitempty" toml:"mcp,omitempty" json:"mcp,omitempty"`

	// Legacy field for backwards compatibility
	Models    []Model        `yaml:"models,omitempty" toml:"models,omitempty" json:"models,omitempty"`
}

// RuntimeConfig represents runtime configuration
type RuntimeConfig struct {
	DefaultModel string  `yaml:"default_model,omitempty" toml:"default_model,omitempty" json:"default_model,omitempty"`
	Models       []Model `yaml:"models,omitempty" toml:"models,omitempty" json:"models,omitempty"`
}

// Dataset represents a dataset configuration
type Dataset struct {
	Name                   string   `yaml:"name" toml:"name" json:"name"`
	DataProcessingStrategy string   `yaml:"data_processing_strategy" toml:"data_processing_strategy" json:"data_processing_strategy"`
	Database               string   `yaml:"database" toml:"database" json:"database"`
	Files                  []string `yaml:"files" toml:"files" json:"files"`
}

// PromptMessage represents a single message in a prompt set
type PromptMessage struct {
	Role    string `yaml:"role,omitempty" json:"role,omitempty"`
	Content string `yaml:"content" json:"content"`
}

// Prompt represents a named prompt set configuration
type Prompt struct {
	Name     string          `yaml:"name" toml:"name" json:"name"`
	Messages []PromptMessage `yaml:"messages" toml:"messages" json:"messages"`
}

// RAGConfig represents the RAG configuration (maps to Rag in Python)
type RAGConfig struct {
	DefaultDatabase           string                    `yaml:"default_database,omitempty" toml:"default_database,omitempty" json:"default_database,omitempty"`
	Databases                 []Database                `yaml:"databases,omitempty" toml:"databases,omitempty" json:"databases,omitempty"`
	DataProcessingStrategies  []DataProcessingStrategy  `yaml:"data_processing_strategies,omitempty" toml:"data_processing_strategies,omitempty" json:"data_processing_strategies,omitempty"`
}

// Database represents a database configuration
type Database struct {
	Name                     string              `yaml:"name" toml:"name" json:"name"`
	Type                     string              `yaml:"type" toml:"type" json:"type"`
	Config                   map[string]interface{} `yaml:"config,omitempty" toml:"config,omitempty" json:"config,omitempty"`
	EmbeddingStrategies      []EmbeddingStrategy `yaml:"embedding_strategies,omitempty" toml:"embedding_strategies,omitempty" json:"embedding_strategies,omitempty"`
	RetrievalStrategies      []RetrievalStrategy `yaml:"retrieval_strategies,omitempty" toml:"retrieval_strategies,omitempty" json:"retrieval_strategies,omitempty"`
	DefaultEmbeddingStrategy string              `yaml:"default_embedding_strategy,omitempty" toml:"default_embedding_strategy,omitempty" json:"default_embedding_strategy,omitempty"`
	DefaultRetrievalStrategy string              `yaml:"default_retrieval_strategy,omitempty" toml:"default_retrieval_strategy,omitempty" json:"default_retrieval_strategy,omitempty"`
}

// EmbeddingStrategy represents an embedding strategy configuration
type EmbeddingStrategy struct {
	Name      string                 `yaml:"name" toml:"name" json:"name"`
	Type      string                 `yaml:"type" toml:"type" json:"type"`
	Config    map[string]interface{} `yaml:"config,omitempty" toml:"config,omitempty" json:"config,omitempty"`
	Condition string                 `yaml:"condition,omitempty" toml:"condition,omitempty" json:"condition,omitempty"`
	Priority  int                    `yaml:"priority,omitempty" toml:"priority,omitempty" json:"priority,omitempty"`
}

// RetrievalStrategy represents a retrieval strategy configuration
type RetrievalStrategy struct {
	Name    string                 `yaml:"name" toml:"name" json:"name"`
	Type    string                 `yaml:"type" toml:"type" json:"type"`
	Config  map[string]interface{} `yaml:"config,omitempty" toml:"config,omitempty" json:"config,omitempty"`
	Default bool                   `yaml:"default,omitempty" toml:"default,omitempty" json:"default,omitempty"`
}

// DataProcessingStrategy represents a data processing strategy
type DataProcessingStrategy struct {
	Name        string      `yaml:"name" toml:"name" json:"name"`
	Description string      `yaml:"description,omitempty" toml:"description,omitempty" json:"description,omitempty"`
	Parsers     []Parser    `yaml:"parsers" toml:"parsers" json:"parsers"`
	Extractors  []Extractor `yaml:"extractors,omitempty" toml:"extractors,omitempty" json:"extractors,omitempty"`
}

// Parser represents a parser configuration
type Parser struct {
	Type                string                 `yaml:"type,omitempty" toml:"type,omitempty" json:"type,omitempty"`
	Config              map[string]interface{} `yaml:"config,omitempty" toml:"config,omitempty" json:"config,omitempty"`
	FileExtensions      []string               `yaml:"file_extensions,omitempty" toml:"file_extensions,omitempty" json:"file_extensions,omitempty"`
	FileIncludePatterns []string               `yaml:"file_include_patterns,omitempty" toml:"file_include_patterns,omitempty" json:"file_include_patterns,omitempty"`
	Priority            int                    `yaml:"priority,omitempty" toml:"priority,omitempty" json:"priority,omitempty"`
	MimeTypes           []string               `yaml:"mime_types,omitempty" toml:"mime_types,omitempty" json:"mime_types,omitempty"`
	FallbackParser      string                 `yaml:"fallback_parser,omitempty" toml:"fallback_parser,omitempty" json:"fallback_parser,omitempty"`
}

// Extractor represents an extractor configuration
type Extractor struct {
	Type                string                 `yaml:"type" toml:"type" json:"type"`
	Config              map[string]interface{} `yaml:"config,omitempty" toml:"config,omitempty" json:"config,omitempty"`
	RequiredFor         []string               `yaml:"required_for,omitempty" toml:"required_for,omitempty" json:"required_for,omitempty"`
	Condition           string                 `yaml:"condition,omitempty" toml:"condition,omitempty" json:"condition,omitempty"`
	FileIncludePatterns []string               `yaml:"file_include_patterns,omitempty" toml:"file_include_patterns,omitempty" json:"file_include_patterns,omitempty"`
	Priority            int                    `yaml:"priority,omitempty" toml:"priority,omitempty" json:"priority,omitempty"`
}

// Model represents a model configuration
type Model struct {
	Name               string                 `yaml:"name" toml:"name" json:"name"`
	Description        string                 `yaml:"description,omitempty" toml:"description,omitempty" json:"description,omitempty"`
	Provider           string                 `yaml:"provider" toml:"provider" json:"provider"`
	Model              string                 `yaml:"model" toml:"model" json:"model"`
	BaseURL            string                 `yaml:"base_url,omitempty" toml:"base_url,omitempty" json:"base_url,omitempty"`
	APIKey             string                 `yaml:"api_key,omitempty" toml:"api_key,omitempty" json:"api_key,omitempty"`
	InstructorMode     string                 `yaml:"instructor_mode,omitempty" toml:"instructor_mode,omitempty" json:"instructor_mode,omitempty"`
	PromptFormat       string                 `yaml:"prompt_format,omitempty" toml:"prompt_format,omitempty" json:"prompt_format,omitempty"`
	ModelAPIParameters map[string]interface{} `yaml:"model_api_parameters,omitempty" toml:"model_api_parameters,omitempty" json:"model_api_parameters,omitempty"`
	ProviderConfig     map[string]interface{} `yaml:"provider_config,omitempty" toml:"provider_config,omitempty" json:"provider_config,omitempty"`
	Prompts            []string               `yaml:"prompts,omitempty" toml:"prompts,omitempty" json:"prompts,omitempty"`
}

// MCPConfig represents Model Context Protocol configuration
type MCPConfig struct {
	Servers []MCPServer `yaml:"servers,omitempty" toml:"servers,omitempty" json:"servers,omitempty"`
}

// MCPServer represents an MCP server configuration
type MCPServer struct {
	Name      string            `yaml:"name" toml:"name" json:"name"`
	Transport string            `yaml:"transport" toml:"transport" json:"transport"`
	Command   string            `yaml:"command,omitempty" toml:"command,omitempty" json:"command,omitempty"`
	Args      []string          `yaml:"args,omitempty" toml:"args,omitempty" json:"args,omitempty"`
	Env       map[string]string `yaml:"env,omitempty" toml:"env,omitempty" json:"env,omitempty"`
	BaseURL   string            `yaml:"base_url,omitempty" toml:"base_url,omitempty" json:"base_url,omitempty"`
	Headers   map[string]string `yaml:"headers,omitempty" toml:"headers,omitempty" json:"headers,omitempty"`
}
