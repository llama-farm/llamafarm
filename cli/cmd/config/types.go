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
	Version       string         `yaml:"version" toml:"version"`
	Name          string         `yaml:"name,omitempty" toml:"name,omitempty"`
	Namespace     string         `yaml:"namespace,omitempty" toml:"namespace,omitempty"`
	Prompts       []Prompt       `yaml:"prompts,omitempty" toml:"prompts,omitempty"`
	RAG           RAGConfig      `yaml:"rag,omitempty" toml:"rag,omitempty"`
	Datasets      []Dataset      `yaml:"datasets,omitempty" toml:"datasets,omitempty"`
	Models        []Model        `yaml:"models,omitempty" toml:"models,omitempty"`
	DefaultModel  string         `yaml:"default_model,omitempty" toml:"default_model,omitempty"`
	RuntimeModels []RuntimeModel `yaml:"runtime_models,omitempty" toml:"runtime_models,omitempty"`
	Runtime       *RuntimeConfig `yaml:"runtime,omitempty" toml:"runtime,omitempty"` // Legacy
}

// Dataset represents a dataset configuration
type Dataset struct {
	Name   string   `yaml:"name" toml:"name"`
	Parser string   `yaml:"parser,omitempty" toml:"parser,omitempty"`
	Files  []string `yaml:"files" toml:"files"`
}

// Prompt represents a prompt configuration
type Prompt struct {
	Name        string `yaml:"name,omitempty" toml:"name,omitempty"`
	Prompt      string `yaml:"prompt" toml:"prompt"`
	Description string `yaml:"description,omitempty" toml:"description,omitempty"`
}

// RAGConfig represents the RAG configuration
type RAGConfig struct {
	Description         string                             `yaml:"description,omitempty" toml:"description,omitempty"`
	Parsers             map[string]ParserConfig            `yaml:"parsers,omitempty" toml:"parsers,omitempty"`
	Embedders           map[string]EmbedderConfig          `yaml:"embedders,omitempty" toml:"embedders,omitempty"`
	VectorStores        map[string]VectorStoreConfig       `yaml:"vector_stores,omitempty" toml:"vector_stores,omitempty"`
	RetrievalStrategies map[string]RetrievalStrategyConfig `yaml:"retrieval_strategies,omitempty" toml:"retrieval_strategies,omitempty"`
	Defaults            DefaultsConfig                     `yaml:"defaults,omitempty" toml:"defaults,omitempty"`

	// Legacy fields for backwards compatibility
	Parser      ParserConfig      `yaml:"parser,omitempty" toml:"parser,omitempty"`
	Embedder    EmbedderConfig    `yaml:"embedder,omitempty" toml:"embedder,omitempty"`
	VectorStore VectorStoreConfig `yaml:"vector_store,omitempty" toml:"vector_store,omitempty"`
}

// ParserConfig represents a parser configuration
type ParserConfig struct {
	Type           string                 `yaml:"type" toml:"type"`
	Config         map[string]interface{} `yaml:"config,omitempty" toml:"config,omitempty"`
	FileExtensions []string               `yaml:"file_extensions,omitempty" toml:"file_extensions,omitempty"`
	MimeTypes      []string               `yaml:"mime_types,omitempty" toml:"mime_types,omitempty"`
	Priority       int                    `yaml:"priority,omitempty" toml:"priority,omitempty"`
}

// EmbedderConfig represents an embedder configuration
type EmbedderConfig struct {
	Type   string                 `yaml:"type" toml:"type"`
	Config map[string]interface{} `yaml:"config,omitempty" toml:"config,omitempty"`
}

// VectorStoreConfig represents a vector store configuration
type VectorStoreConfig struct {
	Type   string                 `yaml:"type" toml:"type"`
	Config map[string]interface{} `yaml:"config,omitempty" toml:"config,omitempty"`
}

// RetrievalStrategyConfig represents a retrieval strategy configuration
type RetrievalStrategyConfig struct {
	Type        string                 `yaml:"type" toml:"type"`
	Config      map[string]interface{} `yaml:"config,omitempty" toml:"config,omitempty"`
	Description string                 `yaml:"description,omitempty" toml:"description,omitempty"`
}

// DefaultsConfig represents default component selections
type DefaultsConfig struct {
	Parser            string `yaml:"parser" toml:"parser"`
	Embedder          string `yaml:"embedder" toml:"embedder"`
	VectorStore       string `yaml:"vector_store" toml:"vector_store"`
	RetrievalStrategy string `yaml:"retrieval_strategy" toml:"retrieval_strategy"`
}

// Model represents a model configuration
type Model struct {
	Provider string `yaml:"provider" toml:"provider"`
	Model    string `yaml:"model" toml:"model"`
}

// RuntimeModel represents a named runtime model configuration
type RuntimeModel struct {
	Name           string                 `yaml:"name" toml:"name" json:"name"`
	Provider       string                 `yaml:"provider" toml:"provider" json:"provider"`
	Model          string                 `yaml:"model" toml:"model" json:"model"`
	BaseURL        string                 `yaml:"base_url,omitempty" toml:"base_url,omitempty" json:"base_url,omitempty"`
	APIKey         string                 `yaml:"api_key,omitempty" toml:"api_key,omitempty" json:"api_key,omitempty"`
	InstructorMode string                 `yaml:"instructor_mode,omitempty" toml:"instructor_mode,omitempty" json:"instructor_mode,omitempty"`
	Parameters     map[string]interface{} `yaml:"parameters,omitempty" toml:"parameters,omitempty" json:"parameters,omitempty"`
}

// RuntimeConfig represents legacy runtime configuration
type RuntimeConfig struct {
	Provider       string  `yaml:"provider" toml:"provider" json:"provider"`
	Model          string  `yaml:"model" toml:"model" json:"model"`
	BaseURL        string  `yaml:"base_url,omitempty" toml:"base_url,omitempty" json:"base_url,omitempty"`
	APIKey         string  `yaml:"api_key,omitempty" toml:"api_key,omitempty" json:"api_key,omitempty"`
	InstructorMode string  `yaml:"instructor_mode,omitempty" toml:"instructor_mode,omitempty" json:"instructor_mode,omitempty"`
	Temperature    float64 `yaml:"temperature,omitempty" toml:"temperature,omitempty" json:"temperature,omitempty"`
	MaxTokens      int     `yaml:"max_tokens,omitempty" toml:"max_tokens,omitempty" json:"max_tokens,omitempty"`
}
