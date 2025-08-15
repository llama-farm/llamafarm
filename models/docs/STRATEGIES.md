# Strategies Guide

Complete guide to understanding and configuring strategies in LlamaFarm.

## Table of Contents
- [What are Strategies?](#what-are-strategies)
- [Strategy Structure](#strategy-structure)
- [Components](#components)
- [Advanced Features](#advanced-features)
- [Pre-Configured Strategies](#pre-configured-strategies)
- [Creating Custom Strategies](#creating-custom-strategies)
- [Best Practices](#best-practices)

## What are Strategies?

Strategies are configuration blueprints that define how LlamaFarm interacts with AI models. They combine:
- Model providers (cloud APIs, local models)
- Fallback chains for reliability
- Routing rules for optimization
- Resource constraints and limits
- Performance optimizations

Think of strategies as "recipes" for different use cases - development, production, training, etc.

## Strategy Structure

### Basic Structure

```yaml
version: "2.0.0"

strategies:
  - name: strategy_name           # Unique identifier
    description: "Purpose"         # Human-readable description
    
    components:                    # Core components configuration
      cloud_api: {...}
      model_app: {...}
      fine_tuner: {...}
      repository: {...}
    
    fallback_chain: [...]         # Fallback options
    routing_rules: [...]          # Conditional routing
    optimization: {...}           # Performance settings
    monitoring: {...}             # Observability config
    constraints: {...}            # Resource limits
```

### Full Example

```yaml
strategies:
  - name: production_hybrid
    description: "Production setup with cloud primary and local fallback"
    
    components:
      cloud_api:
        type: openai_compatible
        config:
          provider: openai
          api_key: ${OPENAI_API_KEY}
          default_model: gpt-4o-mini
          timeout: 30
          max_retries: 3
      
      model_app:
        type: ollama
        config:
          base_url: http://localhost:11434
          default_model: llama3.2:3b
          auto_start: true
    
    fallback_chain:
      - name: primary_cloud
        type: cloud_api
      - name: local_backup
        type: model_app
    
    optimization:
      cache_enabled: true
      batch_size: 10
      timeout_seconds: 30
    
    constraints:
      max_tokens: 4096
      max_cost_per_request: 0.10
```

## Components

### cloud_api
Connects to cloud-based AI services.

```yaml
cloud_api:
  type: openai_compatible        # Provider type
  config:
    provider: openai              # openai, anthropic, groq, etc.
    api_key: ${OPENAI_API_KEY}    # API authentication
    default_model: gpt-4o-mini    # Default model to use
    
    # Optional settings
    base_url: null                # Custom endpoint
    timeout: 60                   # Request timeout (seconds)
    max_retries: 3               # Retry attempts
    retry_delay: 1               # Delay between retries
    
    # Model-specific overrides
    models:
      gpt-4o:
        max_tokens: 4096
        temperature: 0.7
      gpt-3.5-turbo:
        max_tokens: 2048
```

#### Supported Providers
- **openai**: GPT-4, GPT-3.5, DALL-E
- **anthropic**: Claude 3 models
- **groq**: Fast inference with Llama, Mixtral
- **together**: Open-source models
- **deepseek**: Code-specialized models
- **mistral**: Mistral and Codestral models

### model_app
Manages local model applications.

```yaml
model_app:
  type: ollama                   # Application type
  config:
    base_url: http://localhost:11434
    default_model: llama3.2:3b
    auto_start: true              # Start automatically if not running
    
    # Models to manage
    models:
      - name: llama3.2:3b
        pull_on_start: true       # Download if missing
      - name: mistral:7b
        pull_on_start: false
      - name: codellama:13b
        pull_on_start: false
    
    # Resource settings
    gpu_layers: -1                # Number of layers on GPU (-1 = all)
    cpu_threads: 8               # CPU threads to use
    context_size: 4096           # Context window size
```

#### Supported Applications
- **ollama**: Easy local model management
- **llamacpp**: Direct llama.cpp integration
- **vllm**: High-performance serving
- **tgi**: Text Generation Inference

### fine_tuner
Configures model training and fine-tuning.

```yaml
fine_tuner:
  type: pytorch                  # Fine-tuning framework
  config:
    # Base model configuration
    base_model:
      huggingface_id: meta-llama/Llama-2-7b-hf
      cache_dir: ./model_cache
      torch_dtype: float16
    
    # Training method
    method:
      type: lora                 # lora, qlora, full
      r: 16                      # LoRA rank
      alpha: 32                  # LoRA alpha
      dropout: 0.1               # Dropout rate
      target_modules:            # Modules to adapt
        - q_proj
        - v_proj
    
    # Training parameters
    training:
      batch_size: 4
      learning_rate: 2e-4
      num_epochs: 3
      gradient_accumulation_steps: 4
      warmup_ratio: 0.1
```

#### Supported Fine-Tuners
- **pytorch**: Direct PyTorch/Transformers
- **llamafactory**: Advanced fine-tuning framework
- **custom**: Custom training scripts

### repository
Manages model storage and distribution.

```yaml
repository:
  type: huggingface              # Repository type
  config:
    token: ${HF_TOKEN}           # Authentication
    cache_dir: ./hf_cache        # Local cache
    private: false               # Repository visibility
    
    # Upload settings
    push_to_hub: true
    hub_model_id: username/model-name
    hub_strategy: every_save     # every_save, checkpoint, end
```

## Advanced Features

### Fallback Chains

Define multiple fallback options for reliability:

```yaml
fallback_chain:
  - name: Primary OpenAI
    type: openai_compatible
    config:
      provider: openai
      default_model: gpt-4o
      
  - name: Secondary Groq
    type: openai_compatible
    config:
      provider: groq
      default_model: llama-3.1-70b-versatile
      
  - name: Local Fallback
    type: ollama
    config:
      default_model: llama3.2:3b
```

### Routing Rules

Route requests based on conditions:

```yaml
routing_rules:
  - condition: input.length > 2000
    target: long_context_model
    priority: high
    
  - condition: input.contains("code")
    target: code_specialist
    priority: normal
    
  - condition: input.requires_vision
    target: multimodal_model
    priority: high
```

### Optimization Settings

```yaml
optimization:
  # Caching
  cache_enabled: true            # Enable response caching
  cache_ttl: 3600               # Cache time-to-live (seconds)
  
  # Batching
  batch_size: 10                # Batch size for processing
  batch_timeout: 100            # Max wait for batch (ms)
  
  # Performance
  timeout_seconds: 60           # Overall timeout
  parallel_requests: 5          # Concurrent requests
  
  # Retry policy
  retry_policy:
    max_attempts: 3
    backoff_multiplier: 2.0
    max_backoff: 30
```

### Monitoring Configuration

```yaml
monitoring:
  # Logging
  log_level: INFO               # DEBUG, INFO, WARNING, ERROR
  log_format: json              # json, text
  
  # Metrics
  metrics_enabled: true
  metrics_port: 9090
  
  # Tracing
  trace_enabled: true
  trace_sample_rate: 0.1
  
  # Alerts
  alerts:
    - type: latency
      threshold: 5000           # ms
    - type: error_rate
      threshold: 0.05           # 5%
```

### Constraints

Define resource and policy limits:

```yaml
constraints:
  # Token limits
  max_tokens: 4096
  max_input_tokens: 3000
  max_output_tokens: 1096
  
  # Cost controls
  max_cost_per_request: 0.10
  daily_budget: 100.0
  monthly_budget: 3000.0
  
  # Model restrictions
  allowed_models:
    - gpt-4o-mini
    - gpt-3.5-turbo
  blocked_models:
    - gpt-4-turbo-preview
  
  # Hardware requirements
  requires_gpu: false
  min_gpu_memory: 8            # GB
  
  # Privacy settings
  privacy_mode: local_only     # public, private, local_only
```

## Pre-Configured Strategies

### Development Strategies

#### local_development
```yaml
- name: local_development
  description: "Pure local development with Ollama"
  # Uses only local models, no external APIs
  # Perfect for offline development
```

#### hybrid_fallback
```yaml
- name: hybrid_fallback
  description: "Cloud primary with local fallback"
  # Tries cloud first, falls back to local
  # Good balance of performance and reliability
```

### Production Strategies

#### cloud_production
```yaml
- name: cloud_production
  description: "High-availability cloud setup"
  # OpenAI with retries and monitoring
  # For production workloads
```

#### multi_provider
```yaml
- name: multi_provider
  description: "Load-balanced across providers"
  # Distributes across OpenAI, Anthropic, Groq
  # Maximum reliability and cost optimization
```

### Specialized Strategies

#### code_generation
```yaml
- name: code_generation
  description: "Optimized for coding tasks"
  # Uses DeepSeek Coder, GPT-4, Codestral
  # Best for programming assistance
```

#### fine_tuning_pipeline
```yaml
- name: fine_tuning_pipeline
  description: "Complete training workflow"
  # PyTorch with LoRA, HuggingFace integration
  # For model customization
```

#### privacy_first
```yaml
- name: privacy_first
  description: "All processing stays local"
  # No external API calls
  # For sensitive data
```

## Creating Custom Strategies

### Step 1: Define Requirements

Identify your needs:
- Budget constraints?
- Privacy requirements?
- Performance needs?
- Reliability requirements?

### Step 2: Choose Components

Select appropriate components:
```yaml
components:
  # Pick what you need
  cloud_api: {...}     # If using cloud
  model_app: {...}     # If using local
  fine_tuner: {...}    # If training
  repository: {...}    # If storing models
```

### Step 3: Add Resilience

Configure fallbacks:
```yaml
fallback_chain:
  - name: primary
    # Your primary option
  - name: secondary
    # Backup option
  - name: emergency
    # Last resort
```

### Step 4: Optimize

Add performance settings:
```yaml
optimization:
  cache_enabled: true
  batch_size: 10
  # Tune for your use case
```

### Step 5: Set Limits

Define constraints:
```yaml
constraints:
  max_tokens: 2048
  max_cost_per_request: 0.05
  # Protect against runaway costs
```

## Best Practices

### 1. Start Simple
Begin with a pre-configured strategy and customize:
```yaml
# Start with local_development
# Add cloud_api when needed
# Add fallbacks for production
```

### 2. Use Environment Variables
Never hardcode sensitive data:
```yaml
api_key: ${OPENAI_API_KEY}  # Good
api_key: "sk-..."            # Bad!
```

### 3. Test Fallbacks
Ensure fallbacks actually work:
```bash
# Test primary fails gracefully
uv run python cli.py test --strategy your_strategy
```

### 4. Monitor Costs
Set budget constraints:
```yaml
constraints:
  daily_budget: 10.0
  max_cost_per_request: 0.01
```

### 5. Version Control
Track strategy changes:
```yaml
version: "2.0.0"
# Update version when modifying
```

### 6. Document Purpose
Always include descriptions:
```yaml
- name: my_strategy
  description: "Why this exists and what it does"
```

## Validation

Validate your strategies:

```bash
# Validate syntax
uv run python cli.py validate-config my_strategies.yaml

# Test functionality
uv run python cli.py test --strategy-file my_strategies.yaml --strategy my_strategy
```

## Examples

### Minimal Strategy
```yaml
strategies:
  - name: minimal
    description: "Bare minimum configuration"
    components:
      cloud_api:
        type: openai_compatible
        config:
          provider: openai
          default_model: gpt-3.5-turbo
```

### Complex Production Strategy
```yaml
strategies:
  - name: production_complex
    description: "Full production setup with all features"
    
    components:
      cloud_api:
        type: openai_compatible
        config:
          provider: openai
          api_key: ${OPENAI_API_KEY}
          default_model: gpt-4o-mini
      
      model_app:
        type: ollama
        config:
          base_url: http://localhost:11434
          default_model: llama3.2:3b
    
    fallback_chain:
      - name: primary
        type: cloud_api
      - name: groq_fast
        type: openai_compatible
        config:
          provider: groq
          default_model: llama-3.1-8b-instant
      - name: local
        type: model_app
    
    routing_rules:
      - condition: urgent
        target: groq_fast
      - condition: complex
        target: primary
      - condition: private
        target: local
    
    optimization:
      cache_enabled: true
      batch_size: 20
      parallel_requests: 5
    
    monitoring:
      log_level: INFO
      metrics_enabled: true
      trace_enabled: true
    
    constraints:
      max_tokens: 4096
      daily_budget: 100.0
      privacy_mode: private
```