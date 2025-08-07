# LlamaFarm Prompts System

A strategy-based prompt management system for LLMs, providing dynamic template selection, multi-framework support, and extensible architecture.

## Overview

The Prompts System provides:
- 🎯 **Strategy-Based Configuration**: Pre-configured strategies for common use cases
- 🔄 **Dynamic Template Selection**: Automatic template selection based on context
- 🔧 **Multi-Framework Support**: Works with LangChain, LangGraph, native APIs
- 📊 **Performance Optimization**: Caching, token optimization, parallel processing
- 🎨 **Extensible Architecture**: Easy to add new templates and strategies

## Quick Start

### Prerequisites

This system uses [UV](https://docs.astral.sh/uv/) for Python dependency management. Install UV first:

```bash
# Install UV (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

### Setup

```bash
# Install dependencies with UV
uv sync

# Optional: Run full automated setup
uv run python setup.py
```

### Using the CLI

```bash
# List available strategies
uv run python -m prompts.core.cli.strategy_cli strategy list

# Show strategy details
uv run python -m prompts.core.cli.strategy_cli strategy show simple_qa

# Execute a strategy
uv run python -m prompts.core.cli.strategy_cli strategy execute simple_qa -q "What is machine learning?"

# Get strategy recommendations
uv run python -m prompts.core.cli.strategy_cli strategy recommend --use-case "customer_service"

# Run interactive demo
uv run python -m prompts.core.cli.strategy_cli demo --interactive
```

### Using in Code

```python
from prompts.strategies import StrategyManager

# Initialize manager
manager = StrategyManager()

# Execute a strategy
result = manager.execute_strategy(
    strategy_name="simple_qa",
    inputs={
        "query": "What is machine learning?",
        "context": []
    }
)

print(result)  # Generated prompt
```

## Architecture

### Directory Structure

```
prompts/
├── schema.yaml                 # Top-level schema definition
├── default_strategies.yaml     # Pre-configured strategies
├── schemas/                    # Component schemas
│   ├── templates.json         # Template schema
│   └── strategies.json        # Strategy schema
├── strategies/                # Strategy system
│   ├── __init__.py
│   ├── config.py             # Configuration models
│   ├── loader.py             # Strategy loader
│   ├── manager.py            # Strategy manager
│   └── examples/             # Example strategies
├── templates/                 # Template definitions
│   ├── basic/                # Basic templates
│   │   └── qa_basic/
│   │       ├── schema.json   # Template schema
│   │       ├── defaults.json # Default config
│   │       └── template.jinja2
│   ├── advanced/             # Advanced templates
│   ├── chat/                 # Conversational templates
│   ├── domain_specific/      # Specialized templates
│   └── agentic/              # Agent templates
├── demos/                     # Demonstration scripts
│   ├── demo1_simple_qa.py
│   ├── demo2_customer_support.py
│   ├── demo3_code_assistant.py
│   ├── demo4_rag_research.py
│   └── demo5_advanced_reasoning.py
├── tests/                     # Test suite
└── prompts/                   # Core modules
    ├── cli_strategy.py        # CLI with strategy support
    └── core/                  # Core components
```

## Strategies

### Pre-configured Strategies

1. **simple_qa** - Basic question answering
2. **customer_support** - Customer service interactions
3. **professional_assistant** - Formal business communication
4. **analytical_reasoning** - Complex problem-solving
5. **code_assistant** - Programming help and code review
6. **rag_qa** - Retrieval-augmented generation
7. **creative_writing** - Creative content generation
8. **research_assistant** - Academic research support

### Strategy Configuration

```yaml
strategy_name:
  name: "Human-readable name"
  description: "What this strategy does"
  use_cases: ["primary", "use", "cases"]
  
  templates:
    default:
      template: "template_id"
      config:
        temperature: 0.7
    
    specialized:
      - condition:
          query_type: "technical"
        template: "technical_template"
        priority: 20
  
  selection_rules:
    - name: "rule_name"
      condition:
        expression: "python expression"
      template: "template_to_use"
      priority: 30
  
  global_config:
    temperature: 0.7
    max_tokens: 1000
```

## Templates

### Template Structure

Each template has:
- `schema.json` - Template definition and metadata
- `defaults.json` - Default configuration values
- `template.jinja2` - The actual prompt template

### Creating Templates

1. Create directory: `templates/category/template_name/`
2. Add `schema.json`:

```json
{
  "template_id": "unique_id",
  "name": "Template Name",
  "description": "What this template does",
  "category": "basic|advanced|chat|domain_specific|agentic",
  "inputs": {
    "query": {
      "type": "string",
      "required": true,
      "description": "The user's query"
    }
  },
  "outputs": {
    "prompt": {
      "type": "string",
      "description": "Generated prompt"
    }
  },
  "frameworks": ["langchain", "native"],
  "use_cases": ["qa", "support"]
}
```

3. Add `defaults.json`:

```json
{
  "config": {
    "temperature_hint": 0.7,
    "max_tokens_hint": 500
  },
  "framework_specific": {
    "langchain": {
      "prompt_class": "PromptTemplate"
    }
  }
}
```

4. Add `template.jinja2`:

```jinja2
{% if context %}
Context:
{% for doc in context %}
- {{ doc.content }}
{% endfor %}
{% endif %}

Question: {{ query }}
Answer:
```

## Examples

### Customer Support with Dynamic Templates

```python
# Automatically selects appropriate template based on query type
result = manager.execute_strategy(
    strategy_name="customer_support",
    inputs={
        "message": "I'm getting an error when logging in",
        "history": [...],
        "context": []
    },
    context={
        "query_type": "technical"  # Triggers technical support template
    }
)
```

### Code Analysis with Specialized Templates

```python
# Different templates for debug vs optimization
result = manager.execute_strategy(
    strategy_name="code_assistant",
    inputs={
        "code": buggy_code,
        "language": "python",
        "analysis_type": "debug"
    },
    context={
        "query_type": "debug"  # Triggers comprehensive bug analysis
    }
)
```

### Strategy Recommendations

```python
# Find best strategies for your use case
recommendations = manager.recommend_strategies(
    use_case="customer_service",
    performance="balanced",
    complexity="moderate"
)
```

## Advanced Features

### Input/Output Transforms

```yaml
templates:
  default:
    template: "template_id"
    input_transforms:
      - input: "message"
        transform: "lowercase"
    output_transforms:
      - transform: "extract_code"
```

### Selection Rules

```yaml
selection_rules:
  - name: "error_detection"
    condition:
      expression: "'error' in context.get('query', '').lower()"
    template: "error_handling_template"
    priority: 100
    stop_on_match: true
```

### Performance Optimization

```yaml
optimization:
  caching: true              # Cache rendered prompts
  compression: true          # Compress long contexts
  token_optimization: true   # Optimize token usage
  parallel_processing: true  # Process multiple requests
```

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run tests with coverage
uv run pytest --cov=prompts --cov-report=html

# Run specific test modules
uv run pytest tests/test_strategies.py -v
uv run pytest tests/test_template_structure.py -v
uv run pytest tests/test_cli_strategy.py -v
```

## Development

### Setup Development Environment

```bash
# Install all dependencies (dev + test)
uv sync --extra dev --extra test
```

### Common Development Tasks

```bash
# Format code
uv run black .
uv run isort .

# Run linting
uv run ruff check .
uv run mypy prompts --ignore-missing-imports

# Run tests
uv run pytest
uv run pytest --cov=prompts

# Add new dependencies
uv add package-name

# Add development dependencies
uv add --group dev package-name

# Update all dependencies
uv sync --upgrade
```

### Running Demos

```bash
# Run all demos
uv run python demos/run_all_demos.py

# Run individual demos
uv run python demos/demo1_simple_qa.py
uv run python demos/demo2_customer_support.py
# ... etc

# Clean cache files
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

## Contributing

1. **Add Templates**: Create new templates in appropriate categories
2. **Create Strategies**: Design strategies for specific use cases
3. **Write Tests**: Ensure comprehensive test coverage
4. **Update Docs**: Keep documentation current

### Development Workflow

```bash
# 1. Set up development environment
uv sync --extra dev --extra test

# 2. Make your changes
# ... edit files ...

# 3. Format your code
uv run black .
uv run isort .

# 4. Run tests
uv run pytest

# 5. Check code quality
uv run ruff check .
uv run mypy prompts --ignore-missing-imports
```

## Integration

### With LangChain

```python
from langchain import PromptTemplate
from prompts.strategies import StrategyManager

manager = StrategyManager()
prompt_text = manager.execute_strategy("simple_qa", inputs)
langchain_prompt = PromptTemplate.from_template(prompt_text)
```

### With Native APIs

```python
prompt = manager.execute_strategy("code_assistant", inputs)
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3
)
```

## Best Practices

1. **Choose the Right Strategy**: Use recommendations to find suitable strategies
2. **Configure Appropriately**: Adjust temperature and tokens for your use case
3. **Test Thoroughly**: Validate with various inputs before production
4. **Monitor Usage**: Track execution statistics for optimization
5. **Extend Carefully**: Follow existing patterns when adding features

## Troubleshooting

- **Template Not Found**: Ensure template is registered and path is correct
- **Strategy Load Error**: Check YAML syntax in strategy files
- **Selection Not Working**: Enable verbose logging to debug rules
- **Performance Issues**: Enable caching and optimization features

## Migration from v0.1.0

If you're upgrading from the template-only system:

### Key Changes
1. **Template Structure**: Templates now use modular structure (schema.json, defaults.json, template.jinja2)
2. **Configuration**: Strategy-based instead of direct template configuration
3. **CLI**: New strategy-focused commands replace template-only commands

### Migration Steps
1. Convert YAML templates to new modular structure
2. Create strategies for your common use cases
3. Update code to use StrategyManager instead of direct template calls
4. Use `uv run` for all commands instead of direct Python execution

## Changelog

### v0.2.0 (Current)
- **Strategy-Based Architecture**: Complete redesign for better organization
- **Dynamic Template Selection**: Automatic template selection based on context
- **Performance Optimization**: Caching, compression, token optimization
- **Comprehensive CLI**: New strategy management commands
- **UV Integration**: All commands now use UV for dependency management

### v0.1.0
- Initial release with basic template management
- Jinja2 templating support
- Basic CLI operations

## License

Part of the LlamaFarm project. See main repository for license details.