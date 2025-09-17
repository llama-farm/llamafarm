# Example Strategies

This directory contains comprehensive example strategy configurations for LlamaFarm's model management system. Each file demonstrates different use cases and patterns.

## 📁 Strategy Files

### Core Examples (10 Categories)
1. **01_basic_openai.yaml** - Simple OpenAI GPT configurations with fallback chains
   - `basic_openai_gpt4` - GPT-4 for general tasks
   - `openai_gpt35_fast` - Fast GPT-3.5 for simple queries
   - `openai_with_fallback` - Automatic GPT-4 to GPT-3.5 fallback

2. **02_ollama_local.yaml** - Local model deployment using Ollama
   - `ollama_llama3_chat` - Llama 3 conversational AI
   - `ollama_mistral_instruct` - Mistral for instructions
   - `ollama_code_specialist` - Code-optimized models
   - `ollama_embedding_models` - Embeddings for RAG

3. **03_anthropic_claude.yaml** - Claude model configurations
   - `claude_opus_advanced` - Complex reasoning tasks
   - `claude_sonnet_balanced` - Balanced performance/cost
   - `claude_haiku_fast` - Quick responses
   - `claude_with_vision` - Vision capabilities

4. **04_multi_cloud_providers.yaml** - Multi-cloud strategies
   - `multi_cloud_failover` - Automatic provider failover
   - `cost_optimized_routing` - Cost-based routing
   - `specialized_model_routing` - Task-based model selection

5. **05_fine_tuning_workflows.yaml** - Model training configurations
   - `pytorch_lora_training` - LoRA fine-tuning
   - `llamafactory_qlora` - QLoRA with LlamaFactory
   - `full_model_finetuning` - Complete model training
   - `continuous_pretraining` - Domain adaptation

6. **06_huggingface_models.yaml** - Hugging Face deployments
   - `hf_transformers_local` - Local transformers
   - `hf_text_generation` - Various generation models
   - `hf_specialized_models` - Task-specific models
   - `hf_quantized_models` - Efficient quantized models

7. **07_rag_pipelines.yaml** - RAG configurations
   - `basic_rag_pipeline` - Simple embeddings + generation
   - `local_rag_ollama` - Fully local RAG
   - `hybrid_search_rag` - Dense + sparse retrieval
   - `multimodal_rag` - Text and image understanding

8. **08_production_deployments.yaml** - Production patterns
   - `production_api_gateway` - API gateway with monitoring
   - `high_availability_cluster` - Multi-region HA
   - `auto_scaling_deployment` - Load-based scaling
   - `blue_green_deployment` - Zero-downtime updates

9. **09_specialized_domains.yaml** - Domain-specific configurations
   - `medical_ai_assistant` - Medical domain with safety
   - `legal_document_analyzer` - Legal analysis
   - `financial_analysis` - Financial reporting
   - `education_tutor` - Adaptive tutoring

10. **10_development_testing.yaml** - Development setups
    - `mock_testing` - Mock models for testing
    - `a_b_testing` - A/B testing configurations
    - `benchmark_suite` - Performance benchmarking
    - `debug_verbose` - Verbose debugging

### Additional Examples
- **local_inference_engines.yaml** - vLLM, TGI, and hybrid local inference

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