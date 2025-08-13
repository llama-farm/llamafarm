# Example Strategies

This directory contains example strategy configurations demonstrating various use cases and deployment patterns for the LlamaFarm Models system.

## Strategy Format

All strategies follow the schema defined in `/models/schema.yaml` and use the new component-based architecture.

## Available Examples

### 1. Basic Strategies

#### `basic_openai.yaml`
- Simple OpenAI GPT-4 configuration
- Single cloud provider setup
- Basic routing and monitoring
- Good starting point for cloud-based deployments

### 2. Local Model Strategies

#### `ollama_local_models.yaml`
- Multiple Ollama models with automatic selection
- Model-specific routing rules
- Fallback chain for resilience
- Optimized for local development

#### `local_inference_engines.yaml`
- Configurations for HuggingFace Transformers, vLLM, and TGI
- Production-grade local inference setups
- Hybrid approach combining multiple engines
- GPU optimization settings

### 3. Production Strategies

#### `production_cloud.yaml`
- Enterprise-grade configuration
- Multi-provider failover (OpenAI + Anthropic)
- Rate limiting and cost controls
- Comprehensive monitoring and alerting
- Retry strategies and error handling

### 4. Specialized Strategies

#### `multi_model_specialized.yaml`
- Task-specific model routing
- Different models for code, creative writing, math, etc.
- Combines cloud and local providers
- Optimized for diverse workloads

### 5. Fine-Tuning Strategies

#### `fine_tuning_workflows.yaml`
- PyTorch LoRA fine-tuning setup
- LlamaFactory comprehensive training
- Multi-stage training pipelines (pre-training → SFT → RLHF)
- Model export and deployment configurations

## Usage

### Loading a Strategy

```python
from core import ModelManager

# Load a specific strategy
manager = ModelManager.from_strategy("production_cloud")

# Generate text
response = manager.generate("Hello, world!")
```

### Via CLI

```bash
# Use a strategy for generation
python cli.py --strategy production_cloud generate "Your prompt here"

# List available strategies
python cli.py list-strategies

# Get strategy details
python cli.py info --strategy multi_model_specialized
```

### Validating Strategies

```python
from core import StrategyManager

manager = StrategyManager()

# Load custom strategy file
manager.load_strategy_file("example_strategies/production_cloud.yaml")

# Validate strategy
errors = manager.validate_strategy("production_cloud")
if errors:
    print(f"Validation errors: {errors}")
```

## Environment Variables

Most strategies use environment variables for sensitive data:

- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic Claude API key
- `GROQ_API_KEY` - Groq API key
- `TOGETHER_API_KEY` - Together.ai API key
- `HF_TOKEN` - HuggingFace token for private models
- `OLLAMA_BASE_URL` - Ollama server URL (default: http://localhost:11434)
- `METRICS_ENDPOINT` - Monitoring metrics endpoint
- `WANDB_API_KEY` - Weights & Biases API key for training metrics

## Customization

These strategies are templates that can be customized:

1. **Copy the strategy file** to your project
2. **Modify components** to match your needs
3. **Adjust routing rules** for your use cases
4. **Configure constraints** based on your hardware
5. **Set monitoring** according to your observability stack

## Best Practices

1. **Start Simple**: Begin with basic strategies and add complexity as needed
2. **Test Locally**: Use `ollama_local_models.yaml` for development
3. **Add Fallbacks**: Always configure fallback chains for production
4. **Monitor Costs**: Set budget constraints for cloud providers
5. **Validate Changes**: Use the validation tools before deployment
6. **Environment Separation**: Use different strategies for dev/staging/prod

## Migration from Old Configs

If you have old configuration files from `demo_configs/`, you can convert them:

1. Identify the provider type and settings
2. Map to appropriate component types
3. Convert provider-specific settings to component configs
4. Add routing rules and fallback chains as needed
5. Validate using the schema

## Support

For more information, see:
- Main documentation: `/models/README.md`
- Schema definition: `/models/schema.yaml`
- Default strategies: `/models/default_strategies.yaml`
- CLI usage: `python cli.py --help`