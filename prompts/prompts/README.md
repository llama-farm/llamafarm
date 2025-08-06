# Prompts System Package Structure

This directory contains the implementation of the LlamaFarm Prompts System.

## Directory Organization

### 🔒 Core (Internal Components)
Located in `core/` - These are internal implementation details:

- **cli/** - Command-line interfaces
  - `legacy_cli.py` - Original template-based CLI
  - `strategy_cli.py` - New strategy-based CLI
  
- **engines/** - Core processing engines
  - `prompt_system.py` - Main orchestration engine
  - `template_engine.py` - Jinja2 template rendering
  - `strategy_engine.py` - Strategy selection logic
  - `template_registry.py` - Template registration
  - `global_prompt_manager.py` - System-wide prompts
  
- **models/** - Data models (Pydantic)
  - `template.py` - Template definitions
  - `strategy.py` - Strategy definitions
  - `config.py` - Configuration models
  - `context.py` - Context models
  
- **loaders/** - Configuration and template loaders
  - `config_loader.py` - Load configurations
  - `template_loader.py` - Load templates
  - `config_builder.py` - Build configurations

### 🔌 Frameworks (Extensible)
Located in `frameworks/` - Add new framework integrations here:

- `langgraph_integration.py` - LangGraph workflow support
- Future: `langchain_adapter.py`, `llamaindex_adapter.py`, etc.

## Extension Points

To extend the system:

1. **Add a new framework**: Create adapter in `frameworks/`
2. **Add custom loaders**: Extend classes in `core/loaders/`
3. **Add new models**: Extend models in `core/models/`

## Import Examples

```python
# Import core components
from prompts.core.engines import PromptSystem, TemplateEngine

# Import models
from prompts.core.models import PromptTemplate, StrategyConfig

# Import framework adapters
from prompts.frameworks import LangGraphWorkflowManager
```

## Note on Extensibility

- **DO NOT** modify files in `core/` unless contributing to the core system
- **DO** extend the system by:
  - Adding files to `frameworks/`
  - Creating new strategies in `../strategies/`
  - Adding templates to `../templates/`
  - Using the public APIs exposed in `__init__.py`