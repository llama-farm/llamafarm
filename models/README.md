# LlamaFarm Models System

A comprehensive, strategy-based model management system for LlamaFarm that provides unified interfaces for cloud APIs, local model applications, fine-tuning, and model repositories.

## 🎯 Overview

The Models System is designed to provide a flexible, extensible framework for managing AI models across different providers and deployment scenarios. It supports everything from simple API calls to complex fine-tuning workflows, with built-in fallback mechanisms and intelligent routing.

## 🚀 Features

- **Multi-Provider Support**: OpenAI, Anthropic, Ollama, Groq, and more
- **Strategy-Based Configuration**: Pre-configured strategies for common use cases
- **Automatic Fallback**: Seamless failover between cloud and local models
- **Fine-Tuning Support**: PyTorch and LlamaFactory integration
- **Repository Management**: HuggingFace and ModelScope support
- **Component Factory System**: Extensible architecture for adding new providers
- **CLI Interface**: Comprehensive command-line tools for all operations
- **Type-Safe Configuration**: JSON Schema validation for all configurations

## 📦 Installation

### Using uv (Recommended)
```bash
uv sync
```

### Using pip
```bash
pip install -e .
```

## 🎮 Quick Start

### 1. Using Pre-Configured Strategies

```bash
# List available strategies
uv run python cli.py strategies list

# Use a strategy
uv run python cli.py generate --strategy local_development --prompt "Hello, world!"
```

### 2. Python API

```python
from models.core import ModelManager

# Initialize with a strategy
manager = ModelManager(strategy="cloud_production")

# Generate text
response = manager.generate("What is the capital of France?")
print(response)
```

### 3. Custom Configuration

```yaml
# custom_strategy.yaml
strategies:
  - name: my_custom_strategy
    description: My custom model configuration
    components:
      cloud_api:
        type: openai_compatible
        config:
          provider: openai
          api_key: ${OPENAI_API_KEY}
          default_model: gpt-4o-mini
```

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- [Architecture Guide](docs/ARCHITECTURE.md) - System design and component overview
- [Strategy Configuration](docs/STRATEGIES.md) - How to configure and use strategies
- [CLI Reference](docs/CLI.md) - Complete command-line interface documentation
- [Integration Guide](docs/INTEGRATION.md) - Integrating with your application
- [Training Guide](docs/TRAINING_OVERVIEW.md) - Fine-tuning and training workflows

### Component-Specific Docs
- [OpenAI Compatible API](docs/components/openai_compatible_api.md)
- [Ollama Integration](docs/components/ollama_app.md)
- [PyTorch Fine-Tuner](docs/components/pytorch_fine_tuner.md)
- [LlamaFactory Fine-Tuner](docs/components/llamafactory_fine_tuner.md)

## 🏗️ Architecture

```
models/
├── components/          # Modular components
│   ├── cloud_apis/     # Cloud API integrations
│   ├── model_apps/     # Local model applications
│   ├── fine_tuners/    # Fine-tuning implementations
│   └── repositories/   # Model repository managers
├── core/               # Core system modules
│   ├── model_manager.py
│   ├── strategy_manager.py
│   └── config_loader.py
├── docs/               # Documentation
├── tests/              # Test suite
└── cli.py             # CLI interface
```

## 🎯 Available Strategies

### Production Ready
- **`cloud_production`**: High-availability OpenAI setup
- **`multi_provider`**: Load-balanced multi-cloud configuration
- **`hybrid_fallback`**: Cloud primary with local fallback

### Development
- **`local_development`**: Ollama-based local setup
- **`privacy_first`**: Completely offline configuration

### Specialized
- **`code_generation`**: Optimized for code tasks
- **`fast_inference`**: Ultra-low latency with Groq
- **`fine_tuning_pipeline`**: Complete training workflow

## 🧪 Testing

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test categories
uv run pytest tests/test_components.py -v
uv run pytest tests/test_strategies.py -v
uv run pytest tests/test_integration.py -v
uv run pytest tests/test_e2e.py -v

# Run with coverage
uv run pytest tests/ --cov=. --cov-report=html --cov-report=term

# Run specific test
uv run pytest tests/test_components.py::test_openai_compatible -v

# Run with markers
uv run pytest -m "not slow" tests/  # Skip slow tests
uv run pytest -m integration tests/  # Only integration tests
```

### Test Organization

```
tests/
├── test_cli_mock.py         # CLI tests with mocked components
├── test_components.py       # Component unit tests
├── test_strategies.py       # Strategy loading and validation
├── test_integration.py      # Integration tests
├── test_e2e.py             # End-to-end tests
├── test_models.py          # Model manager tests
├── test_simple.py          # Basic functionality tests
├── test_configs/           # Test configuration files
└── test_strategies/        # Test strategy files
```

### Writing Tests

```python
# Example test
def test_strategy_loading():
    """Test that strategies load correctly."""
    from models.core import StrategyManager
    
    manager = StrategyManager()
    strategy = manager.get_strategy("local_development")
    
    assert strategy is not None
    assert strategy["name"] == "local_development"
    assert "components" in strategy
```

### Test Coverage

Current coverage targets:
- Core modules: > 80%
- Components: > 70%
- CLI: > 60%
- Overall: > 70%

## 🔧 Configuration

### Environment Variables
```bash
# Required for cloud providers
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GROQ_API_KEY="your-key"

# Optional configurations
export OLLAMA_BASE_URL="http://localhost:11434"
export HF_TOKEN="your-huggingface-token"
```

### Schema Validation

All configurations are validated against the schema defined in `schema.yaml`. This ensures type safety and catches configuration errors early.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

See [LICENSE](../LICENSE) file for details.

## 🔗 Links

- [LlamaFarm Documentation](https://llama-farm.github.io/llamafarm/)
- [API Reference](docs/CLI.md)
- [Examples](docs/STRATEGY_EXAMPLES.md)