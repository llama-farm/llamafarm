# Prompt Selection Strategies

This directory contains documentation for prompt selection strategies used by the LlamaFarm Prompts System.

## Overview

Strategies determine which prompt template to use for a given query and context. The system supports multiple strategy types that can be combined and configured to optimize prompt selection for different use cases.

## Available Strategy Types

### 1. Static Strategy
Always returns the same template regardless of input.
- **Use Cases**: Testing, fixed workflows, default fallback
- **Configuration**: Simple template ID specification

### 2. Rule-Based Strategy  
Uses predefined rules to select templates based on query patterns.
- **Use Cases**: Keyword routing, conditional logic, pattern matching
- **Configuration**: Rules with conditions, priorities, and fallbacks

### 3. Context-Aware Strategy
Selects templates based on context information like domain, user role, and complexity.
- **Use Cases**: Domain-specific routing, role-based selection, adaptive complexity
- **Configuration**: Context rules with priority ordering

## Creating Custom Strategies

1. **Define Strategy Configuration**: Add to your prompt config file
2. **Test Strategy**: Use CLI testing commands
3. **Validate Performance**: Monitor strategy effectiveness

## Testing Strategies

```bash
# Test strategy with sample data
uv run python -m prompts.cli strategy test context_aware_strategy --test-file test_data/sample_contexts.json

# Test specific strategy logic
uv run python -m prompts.cli strategy debug your_strategy --input "test input"

# Compare strategies
uv run python -m prompts.cli strategy compare strategy1 strategy2
```

## Best Practices

- Always define fallback strategies
- Set clear priority levels
- Test strategies before deployment
- Monitor strategy performance
- Document strategy logic and use cases

For detailed examples and configuration options, see the main README.md file.