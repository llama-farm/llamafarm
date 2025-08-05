# LlamaFarm Prompts System - Architecture Guide

## System Architecture

The Prompts System follows a **strategy-based architecture** similar to the RAG system:

```
┌─────────────────────────────────────────────────────┐
│                   CLI Interface                     │
│              (cli_strategy.py)                      │
├─────────────────────────────────────────────────────┤
│                Strategy Manager                     │
│         (Orchestration & Execution)                 │
├─────────────────────────────────────────────────────┤
│    Strategy Loader    │    Template Engine          │
│  (YAML Configuration)  │  (Jinja2 Rendering)        │
├─────────────────────────────────────────────────────┤
│              Strategy Configurations                │
│        (default_strategies.yaml + examples/)        │
├─────────────────────────────────────────────────────┤
│                Template Storage                     │
│      (Modular templates with schema/defaults)       │
└─────────────────────────────────────────────────────┘
```

## Core Components

### 1. Strategy System (`strategies/`)

#### Strategy Configuration Model (`config.py`)
```python
class StrategyConfig:
    name: str                    # Strategy name
    description: str             # What it does
    use_cases: List[str]        # Common use cases
    templates: TemplatesConfig   # Template configurations
    selection_rules: List[Rule]  # Dynamic selection
    global_config: GlobalConfig  # Global settings
    optimization: OptConfig      # Performance settings
```

#### Strategy Loader (`loader.py`)
- Loads strategies from YAML files
- Validates configurations
- Supports both file and directory loading
- Provides strategy search and filtering

#### Strategy Manager (`manager.py`)
- High-level interface for strategy execution
- Template selection based on context
- Input/output transformations
- Execution statistics tracking

### 2. Template System (`templates/`)

#### Template Structure
```
templates/
└── category/
    └── template_name/
        ├── schema.json      # Template definition
        ├── defaults.json    # Default configuration
        └── template.jinja2  # Prompt template
```

#### Schema Definition (`schema.json`)
```json
{
  "template_id": "unique_identifier",
  "name": "Human-readable name",
  "category": "basic|advanced|chat|...",
  "inputs": {
    "variable_name": {
      "type": "string|array|object",
      "required": true|false,
      "description": "Variable description",
      "validation": {...}
    }
  },
  "outputs": {...},
  "frameworks": ["langchain", "native", ...],
  "use_cases": ["qa", "support", ...]
}
```

### 3. Schema System (`schemas/`)

- `templates.json` - JSON Schema for template validation
- `strategies.json` - JSON Schema for strategy validation

## Key Concepts

### Strategy-Based Configuration

Strategies are pre-configured combinations of:
- **Templates**: Which prompts to use
- **Selection Rules**: When to use each template
- **Global Config**: System-wide settings
- **Optimizations**: Performance tuning

### Dynamic Template Selection

Templates are selected based on:
1. **Selection Rules**: Explicit conditions with priorities
2. **Specialized Templates**: Context-based conditions
3. **Fallback Templates**: When primary fails
4. **Default Template**: Always available

### Input/Output Transforms

**Input Transforms**:
- `lowercase`, `uppercase`, `trim`
- `truncate`, `clean_whitespace`
- `escape_html`

**Output Transforms**:
- `parse_json`, `extract_code`
- `clean_markdown`, `validate_format`

## Configuration Flow

```
1. Load Strategy (YAML)
    ↓
2. Parse & Validate Configuration
    ↓
3. Execute with Context
    ↓
4. Select Template (Rules → Specialized → Default)
    ↓
5. Apply Input Transforms
    ↓
6. Render Template (Jinja2)
    ↓
7. Apply Global Config
    ↓
8. Apply Output Transforms
    ↓
9. Return Final Prompt
```

## Extension Points

### Adding New Templates

1. Create directory structure
2. Define `schema.json` with inputs/outputs
3. Configure `defaults.json` for frameworks
4. Write `template.jinja2` with variables

### Creating Strategies

1. Define in YAML format
2. Configure template mappings
3. Add selection rules
4. Set optimization parameters

### Framework Integration

Adapters for different frameworks:
- **LangChain**: `LangChainAdapter`
- **LangGraph**: `LangGraphAdapter`
- **Native**: `NativeAdapter`
- **LlamaIndex**: `LlamaIndexAdapter`

## Performance Considerations

### Caching
- Template compilation cached
- Strategy configurations cached
- Execution results cacheable

### Optimization Settings
```yaml
optimization:
  caching: true              # Enable result caching
  compression: true          # Compress large contexts
  token_optimization: true   # Minimize token usage
  parallel_processing: true  # Parallel execution
```

### Resource Management
- Lazy loading of templates
- Connection pooling for external services
- Batch processing support

## Testing Architecture

### Unit Tests
- Strategy configuration validation
- Template selection logic
- Transform functions
- CLI commands

### Integration Tests
- End-to-end workflow
- Multi-strategy execution
- Framework adapters
- Error handling

### Test Organization
```
tests/
├── test_strategies.py         # Strategy system tests
├── test_template_structure.py # Template structure tests
├── test_cli_strategy.py       # CLI functionality tests
└── fixtures/                  # Test data
```

## CLI Architecture

### Command Structure
```
cli_strategy.py
├── strategy/
│   ├── list       # List strategies
│   ├── show       # Show details
│   ├── execute    # Run strategy
│   ├── recommend  # Get recommendations
│   ├── create     # Create new
│   └── validate   # Validate config
├── template/
│   └── usage      # Template usage
├── stats          # System statistics
└── demo           # Run demonstrations
```

## Best Practices

### Strategy Design
1. Start with clear use cases
2. Define sensible defaults
3. Add specialized templates for edge cases
4. Use selection rules for automatic routing
5. Test with diverse inputs

### Template Design
1. Keep templates focused
2. Use clear variable names
3. Provide comprehensive examples
4. Document all inputs/outputs
5. Consider token efficiency

### Performance
1. Enable caching for production
2. Use appropriate batch sizes
3. Monitor execution statistics
4. Profile before optimizing
5. Consider async operations

## Comparison with RAG System

| Feature | RAG System | Prompts System |
|---------|-----------|----------------|
| Strategy Config | ✓ | ✓ |
| Component Selection | Parsers, Embedders, Stores | Templates |
| Dynamic Selection | Retrieval strategies | Template selection rules |
| Transforms | Document processing | Input/output transforms |
| CLI | Component management | Strategy execution |
| Extensibility | Plugins for components | Templates & strategies |

## Future Enhancements

1. **Multi-modal Templates**: Image/audio prompts
2. **Chain Templates**: Sequential prompt chains
3. **A/B Testing**: Built-in experimentation
4. **Version Control**: Template versioning
5. **Metrics & Analytics**: Detailed usage analytics
6. **Cloud Sync**: Remote strategy storage