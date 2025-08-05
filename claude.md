# LlamaFarm AI Coding Assistant Guide

## Project Overview
LlamaFarm is a modular, enterprise-ready AI/ML framework that enables building production AI applications with local control. It's designed as an open-source project prioritizing extensibility, testability, and configuration-driven development.

## Key Architecture Principles

### 1. Modular Design
- Each module (rag/, models/, prompts/, config/) operates independently
- Modules communicate through well-defined interfaces
- Factory patterns enable component extensibility
- Plugin architecture allows community contributions

### 2. Strategy-Based Configuration
- **Strategy System**: Predefined configurations for specific use cases (simple, legal, customer_support, etc.)
- **Traditional Config**: Direct component configuration still supported
- **YAML-First**: All configurations use YAML format with extensive commenting
- **Environment Variables**: Secrets and environment-specific settings: `${API_KEY}`
- **Pydantic Validation**: All configurations validated with strong typing
- **Hierarchical Overrides**: Strategy + overrides + environment variables

### 3. Local-First Philosophy
- Prioritize local execution (Ollama, local models)
- Cloud providers as optional enhancements
- Data privacy and cost control by default
- No vendor lock-in

## Development Standards

### Python Best Practices
```python
# Always use type hints
def process_document(content: str, config: Dict[str, Any]) -> ProcessedDocument:
    """Process document with given configuration.
    
    Args:
        content: Raw document content
        config: Processing configuration
        
    Returns:
        ProcessedDocument with extracted data
    """
    # Implementation
```

### Package Management with UV
```bash
# Always use UV for Python dependencies
uv add package-name
uv sync
uv run python script.py
uv run pytest
```

### Testing is Mandatory
- Write tests BEFORE committing code
- Mock external dependencies (APIs, databases)
- Run tests: `uv run pytest`
- Maintain >80% coverage
- Test configuration validation

### Code Organization Pattern
```
module_name/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── base.py          # Abstract base classes
│   ├── factory.py       # Component factories
│   └── registry.py      # Component registry
├── models/
│   ├── __init__.py
│   └── config.py        # Pydantic models
├── implementations/
│   ├── __init__.py
│   ├── provider1.py     # Specific implementations
│   └── provider2.py
├── config/
│   ├── __init__.py
│   ├── schema.py        # Configuration schemas
│   └── examples/        # Example configs
├── tests/
│   ├── __init__.py
│   ├── conftest.py      # Shared fixtures
│   ├── test_core.py
│   └── test_config.py
├── cli.py               # Click-based CLI
└── README.md            # Module documentation
```

## Configuration Management

### Schema Definition
```python
# Always define configuration with Pydantic
from pydantic import BaseModel, Field

class ComponentConfig(BaseModel):
    """Component configuration schema."""
    
    type: str = Field(..., description="Component type")
    name: str = Field(..., description="Component name")
    enabled: bool = Field(True, description="Enable component")
    api_key: Optional[str] = Field(None, description="API key")
    
    class Config:
        extra = "forbid"  # Prevent typos in configs
```

### Strategy-Based Configuration (Recommended)
```yaml
# config/examples/production.yaml
version: "v1"

# Use predefined strategy for quick setup
strategy: "production"

# Override specific settings if needed
strategy_overrides:
  components:
    embedder:
      config:
        batch_size: 64
    vector_store:
      config:
        persist_directory: "./prod_chroma_db"
```

### Traditional Component Configuration
```yaml
# config/examples/traditional.yaml
version: "v1"

# Direct component configuration (still supported)
parser:
  type: "PDFParser"
  config:
    extract_images: true
    ocr_enabled: false

embedder:
  type: "OllamaEmbedder"
  config:
    model: "nomic-embed-text"
    api_base: "http://localhost:11434"

vector_store:
  type: "ChromaStore"
  config:
    persist_directory: "./chroma_db"
    collection_name: "documents"

retrieval_strategy:
  type: "HybridUniversalStrategy"
  config:
    top_k: 10
```

## Common Implementation Patterns

### Factory Pattern (Required)
```python
from typing import Dict, Any, Type
from abc import ABC, abstractmethod

class BaseComponent(ABC):
    """Base class for all components."""
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """Process data."""
        pass

class ComponentFactory:
    """Factory for creating components."""
    
    _registry: Dict[str, Type[BaseComponent]] = {}
    
    @classmethod
    def register(cls, name: str, component_class: Type[BaseComponent]):
        """Register a component type."""
        cls._registry[name] = component_class
    
    @classmethod
    def create(cls, config: Dict[str, Any]) -> BaseComponent:
        """Create component from configuration."""
        component_type = config.get("type")
        if component_type not in cls._registry:
            raise ValueError(f"Unknown component type: {component_type}")
        
        component_class = cls._registry[component_type]
        return component_class(**config.get("config", {}))

# Register implementations
ComponentFactory.register("type1", Type1Component)
ComponentFactory.register("type2", Type2Component)
```

### CLI Pattern (Required)
```python
import click
from rich.console import Console
from rich.table import Table

console = Console()

@click.group()
@click.option('--config', '-c', type=click.Path(exists=True))
@click.pass_context
def cli(ctx, config):
    """Module CLI interface."""
    ctx.ensure_object(dict)
    if config:
        ctx.obj['config'] = load_config(config)

@cli.command()
@click.option('--format', type=click.Choice(['json', 'yaml', 'table']))
def list_components(format):
    """List available components."""
    components = ComponentFactory.list_components()
    
    if format == 'table':
        table = Table(title="Available Components")
        table.add_column("Type", style="cyan")
        table.add_column("Description", style="green")
        
        for comp in components:
            table.add_row(comp['type'], comp['description'])
        
        console.print(table)
    else:
        # JSON/YAML output
        click.echo(format_output(components, format))
```

### Error Handling Pattern
```python
class ModuleError(Exception):
    """Base exception for module."""
    pass

class ConfigurationError(ModuleError):
    """Configuration-related errors."""
    pass

class ProviderError(ModuleError):
    """Provider-specific errors."""
    pass

def safe_operation(func):
    """Decorator for safe operations."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ExternalAPIError as e:
            logger.error(f"External API error: {e}")
            # Try fallback or return graceful error
        except Exception as e:
            logger.exception(f"Unexpected error in {func.__name__}")
            raise ModuleError(f"Operation failed: {e}")
    return wrapper
```

## Testing Requirements

### Test Structure
```python
import pytest
from unittest.mock import patch, MagicMock

class TestComponent:
    """Test component functionality."""
    
    @pytest.fixture
    def config(self):
        """Component configuration."""
        return {
            "type": "TestComponent",
            "config": {
                "setting": "value"
            }
        }
    
    @pytest.fixture
    def component(self, config):
        """Create component instance."""
        return ComponentFactory.create(config)
    
    def test_initialization(self, component):
        """Test component initialization."""
        assert component is not None
        assert component.setting == "value"
    
    @patch('external.api.call')
    def test_external_call(self, mock_api, component):
        """Test external API calls."""
        mock_api.return_value = {"status": "success"}
        result = component.process("data")
        assert result["status"] == "success"
        mock_api.assert_called_once()
    
    def test_configuration_validation(self):
        """Test invalid configuration."""
        with pytest.raises(ConfigurationError):
            ComponentFactory.create({"type": "Invalid"})
```

### Running Tests
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=module_name --cov-report=html

# Run specific test
uv run pytest tests/test_component.py -v

# Run with markers
uv run pytest -m "not integration"
```

## Extensibility Guidelines

### Adding New Components
1. Define configuration schema (Pydantic model)
2. Implement base class interface
3. Register with factory
4. Add CLI commands
5. Write comprehensive tests
6. Document usage and configuration

### Plugin Development
```python
# plugins/custom_parser.py
from rag.core.base import BaseParser

class CustomParser(BaseParser):
    """Custom document parser."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Custom initialization
    
    def parse(self, content: bytes) -> Document:
        """Parse document content."""
        # Implementation
        return Document(...)

# Register plugin
from rag.core.factory import ParserFactory
ParserFactory.register("custom", CustomParser)
```

## Performance Considerations

### Optimization Priorities
1. Configuration parsing (cache parsed configs)
2. Minimize external API calls (batch when possible)
3. Implement connection pooling
4. Use async where appropriate
5. Profile before optimizing

### Resource Management
```python
class ResourceManager:
    """Manage limited resources."""
    
    def __init__(self, max_connections: int = 10):
        self.semaphore = asyncio.Semaphore(max_connections)
    
    async def acquire(self):
        async with self.semaphore:
            yield
```

## Security Best Practices

### API Key Management
```python
# Never hardcode keys
api_key = os.getenv("API_KEY")  # Bad if no validation

# Better approach
from pydantic import SecretStr

class APIConfig(BaseModel):
    api_key: SecretStr = Field(..., description="API key")
    
    @validator('api_key')
    def validate_api_key(cls, v):
        if not v:
            raise ValueError("API key is required")
        return v
```

### Data Privacy
- Process data locally when possible
- Anonymize before sending to external APIs
- Clear temporary files after processing
- Support data retention policies

## Debugging and Logging

### Logging Standards
```python
import logging
from rich.logging import RichHandler

# Module-level logger
logger = logging.getLogger(__name__)

# Rich handler for development
if os.getenv("ENVIRONMENT") == "development":
    logger.addHandler(RichHandler(rich_tracebacks=True))

# Structured logging
logger.info("Processing document", extra={
    "document_id": doc_id,
    "size": len(content),
    "parser": parser_type
})
```

### Debug Mode
```python
@click.option('--debug', is_flag=True, help='Enable debug mode')
def command(debug):
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
```

## Common Tasks

### Running the Development Environment
```bash
# Set up environment
uv sync
uv run pre-commit install

# Start local services
docker-compose up -d
ollama serve

# Run development server
uv run python server/main.py --reload

# Run tests continuously
uv run ptw
```

### Creating a New Module
```bash
# Create module structure
mkdir -p new_module/{core,models,config,tests,examples}

# Initialize module
touch new_module/__init__.py
touch new_module/cli.py

# Set up tests
touch new_module/tests/conftest.py
touch new_module/tests/test_core.py

# Create configuration
touch new_module/config/schema.py
mkdir new_module/config/examples
```

### Debugging Common Issues

1. **Configuration Errors**
   ```python
   # Validate configuration before use
   try:
       config = ConfigSchema(**raw_config)
   except ValidationError as e:
       logger.error(f"Invalid configuration: {e}")
       # Show specific validation errors
   ```

2. **API Failures**
   ```python
   # Implement retry logic
   @retry(max_attempts=3, backoff=exponential_backoff)
   def call_api(endpoint: str, **kwargs):
       # API call implementation
   ```

3. **Memory Issues**
   ```python
   # Process large files in chunks
   def process_large_file(path: Path, chunk_size: int = 1024 * 1024):
       with open(path, 'rb') as f:
           while chunk := f.read(chunk_size):
               yield process_chunk(chunk)
   ```

## Integration Examples

### Using Multiple Modules Together

#### Strategy-Based Approach (Recommended)
```python
from rag import RAGSystem
from models import ModelManager
from prompts import PromptManager

# Load with strategies for quick setup
rag = RAGSystem.from_strategy("customer_support")
models = ModelManager.from_strategy("local_llm")
prompts = PromptManager.from_strategy("professional")

# Process query with strategy-optimized components
query = "How do I reset my password?"
context = rag.search(query)  # Uses strategy-configured retrieval
prompt = prompts.build("support_response", query=query, context=context)
response = models.generate(prompt)  # Uses strategy-configured model
```

#### Traditional Configuration Approach
```python
from rag import RAGSystem
from models import ModelManager
from prompts import PromptManager

# Load configurations
rag_config = load_config("config/rag.yaml")
model_config = load_config("config/models.yaml")
prompt_config = load_config("config/prompts.yaml")

# Initialize systems
rag = RAGSystem(rag_config)
models = ModelManager(model_config)
prompts = PromptManager(prompt_config)

# Process query
query = "What is the weather?"
context = rag.search(query)
prompt = prompts.build("qa_prompt", query=query, context=context)
response = models.generate(prompt)
```

## Contribution Guidelines

### Before Contributing
1. Read existing code in the module
2. Follow established patterns
3. Write tests first (TDD)
4. Update documentation
5. Run quality checks

### Quality Checklist
```bash
# Format code
uv run black .
uv run isort .

# Type checking
uv run mypy .

# Linting
uv run ruff check .

# Tests
uv run pytest

# Documentation
uv run mkdocs build
```

### Pull Request Standards
- Clear description of changes
- Tests for new features
- Documentation updates
- Configuration examples
- No breaking changes without discussion

## Remember
- **Configuration drives everything** - make it configurable
- **Test everything** - no code without tests
- **Document clearly** - others will extend your work
- **Handle errors gracefully** - external services fail
- **Think modular** - components should be independent
- **Use UV** - for all Python package management
- **Local first** - prioritize local execution
- **Be explicit** - clear is better than clever