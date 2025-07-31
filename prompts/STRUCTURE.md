# Directory Structure

```
prompts/
├── README.md                    # Complete system documentation
├── pyproject.toml              # Python package configuration
├── setup_and_demo.sh           # Setup and demonstration script
├── generate_config.py          # Configuration generation utility
├── demo.py                     # Python API demonstration
│
├── prompts/                    # Core system code
│   ├── models/                 # Data models (templates, strategies, config)
│   ├── core/                   # Core engines (template, strategy, registry)
│   ├── utils/                  # Utilities (loaders, builders, helpers)
│   ├── integrations/           # External integrations (LangGraph)
│   └── cli.py                  # Command-line interface
│
├── templates/                  # Individual template files by category
│   ├── basic/                  # Simple Q&A and text generation
│   ├── chat/                   # Conversational templates
│   ├── few_shot/               # Example-based learning templates
│   ├── advanced/               # Complex reasoning templates
│   └── domain_specific/        # Specialized domain templates
│
├── strategies/                 # Strategy documentation
├── config/                     # Generated configuration files
├── test_data/                  # Sample data and test contexts  
├── tests/                      # Test suite
└── uv.lock                     # Dependency lock file
```

## Key Files

- **README.md**: Complete system documentation with examples
- **setup_and_demo.sh**: One-command setup and demonstration
- **pyproject.toml**: Package configuration and dependencies
- **generate_config.py**: Builds configuration from individual templates
- **demo.py**: Python API usage examples

## Getting Started

1. Run `./setup_and_demo.sh` for complete setup and demonstration
2. See `README.md` for detailed documentation and usage examples
3. Use `uv run python -m prompts.cli --help` for CLI reference