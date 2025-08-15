# Project Architecture Guide

Comprehensive guide to the LlamaFarm Models system architecture, component design, and development patterns.

## Table of Contents
- [Overview](#overview)
- [Project Layout](#project-layout)
- [Core Concepts](#core-concepts)
- [Component System](#component-system)
- [Factory Pattern](#factory-pattern)
- [Data Flow](#data-flow)
- [Extension Points](#extension-points)
- [Development Guide](#development-guide)

## Overview

LlamaFarm Models uses a **component-based architecture** with a **factory pattern** for extensibility. The system is designed around:

1. **Components**: Modular, pluggable units (cloud APIs, local apps, fine-tuners)
2. **Strategies**: Configurations that compose components
3. **Factory**: Dynamic component instantiation
4. **Managers**: High-level orchestration

```
┌──────────────────────────────────────────────────┐
│                    CLI Interface                  │
├──────────────────────────────────────────────────┤
│                  Strategy Manager                 │
├──────────────────────────────────────────────────┤
│                   Model Manager                   │
├──────────────────────────────────────────────────┤
│                 Component Factory                 │
├──────────────────────────────────────────────────┤
│  Cloud APIs │ Model Apps │ Fine-Tuners │ Repos   │
└──────────────────────────────────────────────────┘
```

## Project Layout

```
models/
├── cli.py                    # CLI entry point
├── __init__.py              # Package initialization
├── setup.py                 # Package setup
├── pyproject.toml           # Project configuration
├── schema.yaml              # Configuration schema
├── default_strategies.yaml  # Pre-configured strategies
├── model_catalog.yaml       # Model definitions
│
├── core/                    # Core system modules
│   ├── __init__.py
│   ├── model_manager.py     # High-level model operations
│   ├── strategy_manager.py  # Strategy loading and management
│   └── config_loader.py     # Configuration utilities
│
├── components/              # Component implementations
│   ├── __init__.py
│   ├── base.py             # Base component classes
│   ├── factory.py          # Component factory
│   ├── setup_manager.py    # Component setup utilities
│   │
│   ├── cloud_apis/         # Cloud API integrations
│   │   └── openai_compatible/
│   │       ├── __init__.py
│   │       ├── openai_compatible_api.py
│   │       ├── defaults.json
│   │       └── schema.json
│   │
│   ├── model_apps/         # Local model applications
│   │   └── ollama/
│   │       ├── __init__.py
│   │       ├── ollama_app.py
│   │       ├── defaults.json
│   │       └── schema.json
│   │
│   ├── fine_tuners/        # Training implementations
│   │   ├── pytorch/
│   │   │   ├── __init__.py
│   │   │   ├── pytorch_fine_tuner.py
│   │   │   ├── defaults.json
│   │   │   └── schema.json
│   │   └── llamafactory/
│   │       └── ...
│   │
│   ├── model_repositories/ # Model storage
│   │   └── huggingface/
│   │       ├── __init__.py
│   │       └── huggingface_repository.py
│   │
│   └── converters/         # Format converters
│       ├── __init__.py
│       ├── base.py
│       ├── ollama_converter.py
│       └── gguf_converter.py
│
├── tests/                  # Test suite
│   ├── test_components.py
│   ├── test_strategies.py
│   ├── test_integration.py
│   └── test_configs/
│
└── docs/                   # Documentation
    ├── GETTING_STARTED.md
    ├── CLI.md
    ├── STRATEGIES.md
    ├── TRAINING.md
    └── ARCHITECTURE.md     # This file
```

## Core Concepts

### 1. Components

Components are the building blocks of the system. Each component:
- Has a specific responsibility
- Implements a common interface
- Is independently configurable
- Can be replaced or extended

#### Component Types

| Type | Purpose | Examples |
|------|---------|----------|
| `cloud_api` | Cloud model APIs | OpenAI, Anthropic, Groq |
| `model_app` | Local model apps | Ollama, llama.cpp, vLLM |
| `fine_tuner` | Training systems | PyTorch, LlamaFactory |
| `repository` | Model storage | HuggingFace, ModelScope |
| `converter` | Format conversion | GGUF, Ollama formats |

### 2. Strategies

Strategies are YAML configurations that:
- Define which components to use
- Configure component parameters
- Set up fallback chains
- Define routing rules
- Specify constraints

```yaml
strategies:
  - name: example_strategy
    components:
      cloud_api: { ... }    # Component configs
      model_app: { ... }
    fallback_chain: [...]   # Fallback order
    constraints: { ... }    # Resource limits
```

### 3. Factory Pattern

The component factory provides:
- Dynamic component instantiation
- Type registration
- Dependency injection
- Configuration validation

### 4. Managers

High-level orchestrators:
- **StrategyManager**: Loads and manages strategies
- **ModelManager**: Coordinates model operations
- **SetupManager**: Handles component setup

## Component System

### Base Component Class

All components inherit from a base class:

```python
# components/base.py
class BaseComponent:
    """Base class for all components."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validate_config()
    
    def validate_config(self):
        """Validate component configuration."""
        pass
    
    def setup(self):
        """Setup component resources."""
        pass
    
    def cleanup(self):
        """Cleanup component resources."""
        pass
```

### Component Interface

Each component type has a specific interface:

```python
# Cloud API Interface
class BaseCloudAPI(BaseComponent):
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        raise NotImplementedError
    
    def list_models(self) -> List[str]:
        """List available models."""
        raise NotImplementedError

# Fine-Tuner Interface
class BaseFineTuner(BaseComponent):
    def train(self, dataset: str, **kwargs) -> str:
        """Train model on dataset."""
        raise NotImplementedError
    
    def evaluate(self, model: str, dataset: str) -> Dict:
        """Evaluate model performance."""
        raise NotImplementedError
```

### Component Implementation

Example implementation:

```python
# components/cloud_apis/openai_compatible/openai_compatible_api.py
class OpenAICompatibleAPI(BaseCloudAPI):
    """OpenAI-compatible API implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = self._create_client()
    
    def _create_client(self):
        """Create API client."""
        return OpenAI(
            api_key=self.config.get('api_key'),
            base_url=self.config.get('base_url')
        )
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.config.get('default_model'),
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content
```

### Component Configuration

Each component has configuration files:

```json
// components/cloud_apis/openai_compatible/defaults.json
{
  "provider": "openai",
  "base_url": null,
  "api_key": "${OPENAI_API_KEY}",
  "default_model": "gpt-3.5-turbo",
  "timeout": 60,
  "max_retries": 3
}
```

```json
// components/cloud_apis/openai_compatible/schema.json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "provider": {
      "type": "string",
      "enum": ["openai", "groq", "together", "deepseek"]
    },
    "api_key": {
      "type": "string"
    },
    "default_model": {
      "type": "string"
    }
  },
  "required": ["provider", "api_key", "default_model"]
}
```

## Factory Pattern

### Component Factory

The factory manages component lifecycle:

```python
# components/factory.py
class ComponentFactory:
    """Factory for creating components."""
    
    _registry = {}
    
    @classmethod
    def register(cls, component_type: str, component_class: Type):
        """Register a component type."""
        cls._registry[component_type] = component_class
    
    @classmethod
    def create(cls, component_type: str, config: Dict) -> BaseComponent:
        """Create a component instance."""
        if component_type not in cls._registry:
            raise ValueError(f"Unknown component type: {component_type}")
        
        component_class = cls._registry[component_type]
        return component_class(config)
```

### Registration

Components self-register on import:

```python
# components/cloud_apis/openai_compatible/__init__.py
from .openai_compatible_api import OpenAICompatibleAPI
from components.factory import ComponentFactory

# Register component
ComponentFactory.register("openai_compatible", OpenAICompatibleAPI)
```

### Dynamic Loading

Components are loaded dynamically:

```python
# core/model_manager.py
class ModelManager:
    def __init__(self, strategy: Dict):
        self.components = {}
        
        # Load components from strategy
        for comp_type, comp_config in strategy['components'].items():
            component = ComponentFactory.create(
                comp_config['type'],
                comp_config.get('config', {})
            )
            self.components[comp_type] = component
```

## Data Flow

### Request Flow

1. **CLI receives command** → Parses arguments
2. **Strategy Manager** → Loads strategy configuration
3. **Model Manager** → Coordinates components
4. **Component Factory** → Creates component instances
5. **Component** → Executes operation
6. **Response** → Returns through chain

### Example: Text Generation

```
User Input → CLI
    ↓
cli.py: parse_args()
    ↓
StrategyManager: get_strategy("production")
    ↓
ModelManager: __init__(strategy)
    ↓
ComponentFactory: create("openai_compatible", config)
    ↓
OpenAICompatibleAPI: generate(prompt)
    ↓
OpenAI API Call
    ↓
Response → User
```

### Fallback Chain

When primary fails:

```
Primary Component (fails)
    ↓
ModelManager: handle_fallback()
    ↓
Fallback Component 1 (try)
    ↓ (fails)
Fallback Component 2 (try)
    ↓ (succeeds)
Response → User
```

## Extension Points

### Adding a New Component Type

1. **Define interface** in `components/base.py`:
```python
class BaseNewComponent(BaseComponent):
    def do_something(self): pass
```

2. **Create implementation** in `components/new_type/`:
```python
class MyNewComponent(BaseNewComponent):
    def do_something(self):
        return "Implementation"
```

3. **Register with factory**:
```python
ComponentFactory.register("my_new_component", MyNewComponent)
```

4. **Add to schema** in `schema.yaml`:
```yaml
new_component:
  type: object
  properties:
    type:
      enum: [my_new_component]
```

### Adding a New Provider

1. **Create provider directory**:
```
components/cloud_apis/new_provider/
├── __init__.py
├── new_provider_api.py
├── defaults.json
└── schema.json
```

2. **Implement provider class**:
```python
class NewProviderAPI(BaseCloudAPI):
    def generate(self, prompt: str, **kwargs):
        # Implementation
```

3. **Register provider**:
```python
ComponentFactory.register("new_provider", NewProviderAPI)
```

### Adding a New Strategy

1. **Create strategy YAML**:
```yaml
strategies:
  - name: new_strategy
    description: "New use case"
    components:
      cloud_api:
        type: openai_compatible
        config: { ... }
```

2. **Validate strategy**:
```bash
uv run python cli.py validate-config new_strategy.yaml
```

3. **Test strategy**:
```bash
uv run python cli.py test --strategy-file new_strategy.yaml
```

## Development Guide

### Code Organization

- **One component per directory**: Keep related files together
- **Consistent naming**: `component_name.py`, `defaults.json`, `schema.json`
- **Self-contained**: Components should not depend on each other
- **Configuration-driven**: Behavior controlled by config, not code

### Testing Components

```python
# tests/test_components.py
def test_openai_component():
    config = {
        "provider": "openai",
        "api_key": "test-key",
        "default_model": "gpt-3.5-turbo"
    }
    
    component = ComponentFactory.create("openai_compatible", config)
    assert component is not None
    assert component.config["provider"] == "openai"
```

### Error Handling

```python
class ComponentError(Exception):
    """Base exception for component errors."""
    pass

class ConfigurationError(ComponentError):
    """Configuration-related errors."""
    pass

class ConnectionError(ComponentError):
    """Connection-related errors."""
    pass
```

### Logging

```python
import logging

logger = logging.getLogger(__name__)

class MyComponent(BaseComponent):
    def operation(self):
        logger.debug("Starting operation")
        try:
            result = self._do_work()
            logger.info("Operation successful")
            return result
        except Exception as e:
            logger.error(f"Operation failed: {e}")
            raise
```

### Configuration Management

```python
class ConfigLoader:
    @staticmethod
    def load_config(path: str) -> Dict:
        """Load configuration from file."""
        with open(path) as f:
            if path.endswith('.yaml'):
                return yaml.safe_load(f)
            elif path.endswith('.json'):
                return json.load(f)
    
    @staticmethod
    def merge_configs(base: Dict, override: Dict) -> Dict:
        """Merge configurations with overrides."""
        return {**base, **override}
```

## Best Practices

### 1. Component Design

- **Single Responsibility**: Each component does one thing well
- **Interface Compliance**: Implement all required methods
- **Configuration Validation**: Validate config in `__init__`
- **Resource Management**: Clean up in `cleanup()`

### 2. Error Handling

- **Graceful Degradation**: Fall back when possible
- **Informative Errors**: Include context in error messages
- **Retry Logic**: Implement retries for transient failures
- **Logging**: Log errors with appropriate levels

### 3. Performance

- **Lazy Loading**: Load resources only when needed
- **Caching**: Cache expensive operations
- **Async Operations**: Use async for I/O operations
- **Resource Pooling**: Reuse connections and clients

### 4. Security

- **No Hardcoded Secrets**: Use environment variables
- **Input Validation**: Validate all inputs
- **Secure Defaults**: Safe default configurations
- **Audit Logging**: Log security-relevant events

## Navigation Tips

### Finding Components

```bash
# List all components
ls -la components/*/

# Find specific component
find components -name "*openai*"

# Search component code
grep -r "class.*API" components/
```

### Understanding Flow

1. Start at `cli.py` for command handling
2. Follow to `core/strategy_manager.py` for strategy loading
3. Check `core/model_manager.py` for orchestration
4. Look in `components/` for implementations

### Key Files

| File | Purpose |
|------|---------|
| `cli.py` | CLI entry point and command parsing |
| `schema.yaml` | Configuration schema definition |
| `default_strategies.yaml` | Pre-configured strategies |
| `core/model_manager.py` | Main orchestration logic |
| `components/factory.py` | Component instantiation |
| `components/base.py` | Base classes and interfaces |

### Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Trace component creation
ComponentFactory.create = debug_wrapper(ComponentFactory.create)

# Inspect strategy
print(json.dumps(strategy, indent=2))
```

This architecture provides flexibility, extensibility, and maintainability while keeping the codebase organized and testable.