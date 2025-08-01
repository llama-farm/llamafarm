# LlamaFarm Configuration Module

A Python module for loading and validating LlamaFarm configuration files with full type safety.

## Features

- ✅ **Schema-driven types** - Auto-generated TypedDict classes from JSON schema
- ✅ **Multiple formats** - Support for YAML (`.yaml`, `.yml`) and TOML (`.toml`) files
- ✅ **Type safety** - Full IDE autocomplete and static analysis support
- ✅ **Validation** - Runtime validation against JSON schema
- ✅ **Flexible discovery** - Automatic config file discovery in target directories
- ✅ **Error handling** - Comprehensive error messages and graceful fallbacks

## Quick Start

```python
from config import load_config, LlamaFarmConfig

# Load configuration with full type safety
config: LlamaFarmConfig = load_config()

# Access typed configuration values
version = config["version"]  # Literal["v1"]
rag_config = config["rag"]   # RAGConfig
models = config["models"]    # List[ModelConfig]
```

## Configuration Files

The module looks for configuration files in this order:
1. `llamafarm.yaml`
2. `llamafarm.yml`
3. `llamafarm.toml`

### Example Configuration

```yaml
version: v1

prompts:
  - name: "customer_support"
    prompt: "You are a helpful customer support assistant..."
    description: "Default customer support prompt"

rag:
  parser:
    type: CustomerSupportCSVParser
    config:
      content_fields: ["question", "answer"]
      metadata_fields: ["category", "timestamp"]

  embedder:
    type: OllamaEmbedder
    config:
      model: "mxbai-embed-large"
      batch_size: 32

  vector_store:
    type: ChromaStore
    config:
      collection_name: "customer_support_kb"
      persist_directory: "./data/vector_store"

models:
  - provider: "local"
    model: "llama3.1:8b"
  - provider: "openai"
    model: "gpt-4"
```

## API Reference

### `load_config()`

Load configuration from file or directory.

```python
def load_config(
    config_path: Optional[Path] = None,
    directory: Optional[Path] = None,
    validate: bool = True
) -> LlamaFarmConfig:
    """
    Load LlamaFarm configuration from file or directory.

    Args:
        config_path: Path to specific config file
        directory: Directory to search for config files
        validate: Whether to validate against schema

    Returns:
        Typed configuration dictionary

    Raises:
        ConfigError: If config file not found or invalid
    """
```

### Type Definitions

All types are auto-generated from `schema.yaml`:

- `LlamaFarmConfig` - Main configuration TypedDict
- `RAGConfig` - RAG configuration section
- `ModelConfig` - Individual model configuration
- `PromptConfig` - Individual prompt configuration
- `ParserConfig`, `EmbedderConfig`, `VectorStoreConfig` - Component configs

## Type Generation

The types are automatically generated from the JSON schema to ensure they stay in sync:

```bash
# Generate types from schema.yaml
python generate_types.py
```

This creates `config_types.py` with all the TypedDict classes based on your schema.

## Installation

```bash
# Install dependencies
pip install PyYAML jsonschema tomli

# For development
pip install pytest
```

## Testing

```bash
# Run all tests
cd tests && python run_tests.py

# Or with pytest directly
pytest tests/ -v
```

## Error Handling

The module provides comprehensive error handling:

```python
from config import load_config, ConfigError

try:
    config = load_config()
except ConfigError as e:
    print(f"Configuration error: {e}")
    # Handle error appropriately
```

## Integration Examples

### RAG Module Usage
```python
config = load_config()
rag = config["rag"]

parser_type = rag["parser"]["type"]
embedder_model = rag["embedder"]["config"]["model"]
collection_name = rag["vector_store"]["config"]["collection_name"]
```

### Model Manager Usage
```python
config = load_config()
models = config["models"]

# Filter by provider
local_models = [m for m in models if m["provider"] == "local"]
cloud_models = [m for m in models if m["provider"] != "local"]
```

### Prompt Manager Usage
```python
config = load_config()
prompts = config.get("prompts", [])

# Create lookup
prompt_lookup = {p["name"]: p for p in prompts if "name" in p}

# Access specific prompt
if "customer_support" in prompt_lookup:
    prompt_text = prompt_lookup["customer_support"]["prompt"]
```

## Development

### Project Structure

```
config/
├── __init__.py              # Package exports
├── loader.py                # Main configuration loader
├── config_types.py          # Auto-generated types (DO NOT EDIT)
├── generate_types.py        # Type generation script
├── schema.yaml              # JSON schema definition
├── pyproject.toml           # Dependencies and build config
├── README.md                # This documentation
├── example.py               # Usage examples
└── tests/                   # Comprehensive test suite
    ├── sample_config.yaml   # Test configurations
    ├── sample_config.toml
    ├── test_config_loader.py
    ├── test_edge_cases.py
    └── test_integration.py
```

### Adding New Configuration Options

1. Update `schema.yaml` with new fields
2. Run `python generate_types.py` to regenerate types
3. Update tests to cover new functionality
4. Update documentation

### Running Tests

```bash
# Run all tests
cd tests && python run_tests.py

# Run specific test categories
pytest tests/test_config_loader.py -v
pytest tests/test_edge_cases.py -v
pytest tests/test_integration.py -v
```

## License

This module is part of the LlamaFarm project.