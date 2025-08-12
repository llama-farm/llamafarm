# Migration Guide: From Old Configs to New Strategies

This guide helps you migrate from the old configuration format (found in `demo_configs/`) to the new strategy-based format.

## Key Changes

### 1. Structure Change
- **Old**: Flat provider-based configuration
- **New**: Component-based strategies with explicit types

### 2. Top-Level Format
```yaml
# Old Format
{
  "name": "Config Name",
  "providers": { ... },
  "default_provider": "provider_name"
}

# New Format
strategies:
  strategy_name:
    name: Strategy Name
    description: Strategy description
    components: { ... }
```

## Migration Examples

### Basic OpenAI Config

**Old Format** (`demo_basic_config.yaml`):
```json
{
  "providers": {
    "openai_gpt4": {
      "type": "cloud",
      "provider": "openai",
      "model": "gpt-4-turbo-preview",
      "api_key": "${OPENAI_API_KEY}"
    }
  }
}
```

**New Format** (`basic_openai.yaml`):
```yaml
strategies:
  basic_openai:
    name: Basic OpenAI Configuration
    components:
      cloud_api:
        type: openai_compatible
        config:
          provider: openai
          api_key: ${OPENAI_API_KEY}
          default_model: gpt-4-turbo-preview
```

### Ollama Local Config

**Old Format** (`demo_ollama_config.yaml`):
```json
{
  "providers": {
    "ollama_llama3": {
      "type": "local",
      "provider": "ollama",
      "model": "llama3:latest",
      "base_url": "http://localhost:11434"
    }
  }
}
```

**New Format** (`ollama_local_models.yaml`):
```yaml
strategies:
  ollama_local:
    name: Ollama Local Models
    components:
      model_app:
        type: ollama
        config:
          base_url: http://localhost:11434
          default_model: llama3:latest
```

## Component Type Mapping

| Old Provider Type | New Component Type | Component Name |
|------------------|-------------------|----------------|
| `cloud` + `openai` | `openai_compatible` | `cloud_api` |
| `cloud` + `anthropic` | `openai_compatible` | `cloud_api` |
| `local` + `ollama` | `ollama` | `model_app` |
| `local` + `huggingface` | `huggingface` | `model_app` |
| `local` + `vllm` | `vllm` | `model_app` |
| `local` + `tgi` | `tgi` | `model_app` |

## New Features Available

### 1. Fallback Chains
```yaml
fallback_chain:
  - provider: cloud_api
    model: gpt-4
    conditions: [api_healthy]
  - provider: cloud_api
    model: gpt-3.5-turbo
```

### 2. Routing Rules
```yaml
routing_rules:
  - pattern: "code|debug"
    provider: cloud_api
    model: codellama-70b
  - pattern: ".*"
    provider: model_app
    model: llama3
```

### 3. Constraints
```yaml
constraints:
  max_tokens_per_request: 4096
  requires_gpu: true
  max_daily_cost_usd: 100
```

### 4. Monitoring
```yaml
monitoring:
  track_usage: true
  track_costs: true
  alert_thresholds:
    daily_cost_usd: 100
```

## Migration Steps

1. **Identify Component Type**: Map your old provider to the new component type
2. **Create Strategy Structure**: Use the new YAML format with `strategies:` at the top
3. **Move Configuration**: Transfer settings to the `config:` section
4. **Add Enhancements**: Add routing rules, fallback chains, constraints as needed
5. **Validate**: Run `python example_strategies/validate_strategies.py`

## Testing Your Migration

```python
from core import StrategyManager

# Load your migrated strategy
manager = StrategyManager(strategies_file="your_strategy.yaml")
strategy = manager.get_strategy("your_strategy_name")

# Validate
errors = manager.validate_strategy("your_strategy_name")
if errors:
    print(f"Validation errors: {errors}")
else:
    print("Strategy is valid!")
```

## Common Issues

### Issue: Missing API Keys
- **Solution**: Use environment variables with `${VAR_NAME}` syntax

### Issue: Invalid Component Type
- **Solution**: Check the component type mapping table above

### Issue: Fallback Not Working
- **Solution**: Ensure fallback providers are defined in components

### Issue: Model Not Found
- **Solution**: Use `default_model` in config, not top-level `model`

## Getting Help

- See example strategies in `/models/example_strategies/`
- Check the schema at `/models/schema.yaml`
- Review default strategies at `/models/default_strategies.yaml`
- Run validation: `python example_strategies/validate_strategies.py`