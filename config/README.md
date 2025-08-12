# LlamaFarm Configuration Module

A Python module for loading and validating LlamaFarm configuration files with full type safety.

## Features

- ✅ **Schema-driven types** - Auto-generated Pydantic models from JSON schema using `datamodel-code-generator`
- ✅ **Multiple formats** - Support for YAML (`.yaml`, `.yml`) and TOML (`.toml`) files
- ✅ **Type safety** - Full IDE autocomplete and static analysis support with Pydantic
- ✅ **Validation** - Runtime validation against JSON schema with detailed error messages
- ✅ **Flexible discovery** - Automatic config file discovery in target directories
- ✅ **Error handling** - Comprehensive error messages and graceful fallbacks

## Quick Start

```python
from config import load_config
from config.config_types import LlamaFarmConfig

# Load configuration with full type safety
config: LlamaFarmConfig = load_config()

# Access typed configuration values
version = config["version"]  # Literal["v1"]
rag_config = config["rag"]   # Dict[str, Any]
models = config["models"]    # List[Model]
datasets = config["datasets"]  # List[Dataset]
```

## Configuration Files

The module looks for configuration files in this order:
1. `llamafarm.yaml`
2. `llamafarm.yml`
3. `llamafarm.toml`

### Example Configuration

```yaml
version: v1
name: "my-namespace/my-project"

prompts:
  - name: "customer_support"
    prompt: "You are a helpful customer support assistant..."
    description: "Default customer support prompt"

rag:
  parsers: ["pdf", "text", "csv"]
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

datasets:
  - name: "training_data"
    rag_strategy: "auto"
    files: ["file_hash_1", "file_hash_2"]

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

All types are auto-generated from `schema.yaml` using `datamodel-code-generator`:

- `LlamafarmConfig` - Main configuration Pydantic model
- `Dataset` - Dataset configuration with name, RAG strategy, and file hashes
- `Model` - Model configuration with provider and model name
- `Prompt` - Prompt configuration with name, text, and description
- `Provider` - Enum for valid model providers (openai, anthropic, google, local, custom)
- `Version` - Enum for config version (currently "v1")

## Type Generation

The types are automatically generated from the JSON schema using `datamodel-code-generator`:

```bash
# Generate Pydantic models from schema.yaml
./generate-types.sh
```

This creates `config_types.py` with all the Pydantic model classes based on your schema.

## Installation

```bash
# Install with uv (recommended)
uv add llamafarm-config

# Or with pip
pip install PyYAML jsonschema tomli

# For development
uv sync  # Install all dependencies including dev tools
```

## Testing

```bash
# Run all tests with uv
uv run pytest tests/ -v

# Or run the test script
cd tests && python run_tests.py

# Run with coverage
uv run pytest tests/ --cov=config --cov-report=term-missing
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

# Access supported parsers
parsers = rag.get("parsers", [])  # List of parser types

# Access embedder configuration
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

### Dataset Management Usage
```python
config = load_config()
datasets = config["datasets"]

# Filter datasets by strategy
auto_datasets = [d for d in datasets if d["rag_strategy"] == "auto"]

# Access dataset files
for dataset in datasets:
    name = dataset["name"]
    files = dataset["files"]  # List of file hashes
    print(f"Dataset {name} contains {len(files)} files")
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
├── helpers /
    ├── loader.py            # Main configuration loader
    └── generator.py         # Generate config files
├── datamodel.py             # Auto-generated Pydantic models (DO NOT EDIT)
├── schema.yaml              # JSON schema definition
├── pyproject.toml           # Dependencies and build config with uv
├── uv.lock                  # Lock file for reproducible builds
├── README.md                # This documentation
└── tests/                   # Comprehensive test suite
    ├── sample_config.yaml   # Test configurations
    ├── sample_config.toml
    ├── test_config_loader.py
    ├── test_edge_cases.py
    └── test_integration.py
```

### Adding New Configuration Options

1. Update `schema.yaml` with new fields
2. Run `./generate-types.sh` to regenerate types
3. Update tests to cover new functionality
4. Update documentation and examples

### Running Tests

```bash
# Run all tests with uv
uv run pytest tests/ -v

# Run all tests (legacy method)
cd tests && python run_tests.py

# Run specific test categories
uv run pytest tests/test_config_loader.py -v
uv run pytest tests/test_edge_cases.py -v
uv run pytest tests/test_integration.py -v

# Run with coverage
uv run pytest tests/ --cov=config --cov-report=html
```

### Dependencies and Tools

The project uses modern Python tooling:

- **uv** - Fast Python package manager and project manager
- **datamodel-code-generator** - Generates Pydantic models from JSON schema
- **Pydantic** - Runtime validation and serialization with type hints
- **Ruff** - Fast Python linter and formatter
- **pytest** - Modern testing framework

### Development Dependencies

```toml
[project.optional-dependencies]
dev = [
    "datamodel-code-generator>=0.27.3",  # Type generation
    "pytest>=7.0.0",                     # Testing framework
    "pytest-cov>=4.0.0",                 # Coverage reporting
    "mypy>=1.0.0",                       # Static type checking
    "ruff>=0.12.7"                       # Linting and formatting
]
```

## License

This module is part of the LlamaFarm project.