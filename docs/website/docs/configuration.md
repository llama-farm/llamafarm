---
sidebar_position: 3
title: Configuration
---

# Configuration Guide

LlamaFarm uses a simple, declarative YAML configuration system that makes deploying AI as easy as defining your infrastructure as code.

## Configuration Philosophy

LlamaFarm follows these principles:

1. **Configuration over Code** - Define what you want, not how to build it
2. **Sensible Defaults** - Start simple, customize as needed
3. **Progressive Complexity** - Add features incrementally
4. **Environment Aware** - Different configs for dev/staging/prod

## Basic Configuration

A minimal LlamaFarm configuration:

```yaml title="llamafarm.yaml"
models:
  - name: assistant
    type: llama2-7b

deploy:
  local: true
```

This configuration:
- Downloads and loads Llama 2 7B model
- Runs it locally on available hardware
- Exposes a REST API on port 8080

## Configuration Structure

### Root Level

```yaml
# Model definitions
models: []

# Pipeline configuration  
pipeline: []

# Deployment settings
deploy: {}

# Global settings
settings: {}
```

### Models Section

Define AI models to use:

```yaml
models:
  # Local model example
  - name: local-llama
    type: llama2-13b
    device: cuda  # or cpu, auto
    quantization: int8  # int4, int8, float16
    cache_dir: ./models
    
  # Cloud API example
  - name: cloud-gpt
    type: openai
    model: gpt-4
    api_key: ${OPENAI_API_KEY}  # Environment variable
    
  # Custom model
  - name: custom-bert
    type: custom
    path: ./my-fine-tuned-model
    tokenizer: bert-base-uncased
```

### Pipeline Section

Define how data flows through your models:

```yaml
pipeline:
  # Simple generation
  - input: user_query
  - generate:
      model: local-llama
      max_tokens: 500
      temperature: 0.7
  - output: response
  
  # Complex pipeline with embeddings
  - input: documents
  - embed:
      model: sentence-transformers
      batch_size: 32
  - store:
      type: chromadb
      collection: knowledge-base
  - retrieve:
      top_k: 5
  - generate:
      model: local-llama
      context: "{retrieved_docs}"
      prompt: "Answer based on context: {query}"
  - output: answer
```

### Deploy Section

Configure deployment targets:

```yaml
deploy:
  # Local deployment
  local:
    enable: true
    port: 8080
    workers: 4
    gpu_layers: 35
    
  # Cloud deployment
  aws:
    region: us-east-1
    instance_type: g4dn.xlarge
    min_instances: 1
    max_instances: 10
    
  # Edge deployment  
  edge:
    devices:
      - name: jetson-1
        ip: 192.168.1.100
      - name: rpi-cluster
        ips: [192.168.1.101, 192.168.1.102]
        
  # Kubernetes
  kubernetes:
    namespace: llamafarm
    replicas: 3
    resources:
      requests:
        memory: "8Gi"
        cpu: "4"
      limits:
        memory: "16Gi"
        nvidia.com/gpu: 1
```

## Advanced Features

### Multi-Model Routing

Route requests to different models based on criteria:

```yaml
models:
  - name: fast-model
    type: llama2-7b
    tags: ["fast", "general"]
    
  - name: accurate-model  
    type: llama2-70b
    tags: ["accurate", "slow"]
    
pipeline:
  - route:
      by: complexity
      rules:
        - if: token_count < 100
          use: fast-model
        - if: domain == "medical"
          use: accurate-model
        - default: fast-model
```

### Model Cascading

Use fallback models for reliability:

```yaml
pipeline:
  - generate:
      model: primary-model
      fallback:
        - model: backup-model
          when: timeout > 5s
        - model: local-model
          when: error
      retry:
        attempts: 3
        backoff: exponential
```

### Load Balancing

Distribute load across multiple instances:

```yaml
deploy:
  load_balancer:
    strategy: round_robin  # or least_connections, weighted
    health_check:
      endpoint: /health
      interval: 30s
      timeout: 5s
    endpoints:
      - url: http://node1:8080
        weight: 2
      - url: http://node2:8080
        weight: 1
```

### Monitoring & Logging

Configure observability:

```yaml
settings:
  monitoring:
    prometheus:
      enable: true
      port: 9090
      
    logging:
      level: info  # debug, info, warning, error
      format: json
      outputs:
        - console
        - file: ./logs/llamafarm.log
        - elasticsearch:
            host: localhost:9200
            index: llamafarm-logs
            
    tracing:
      enable: true
      jaeger:
        endpoint: http://localhost:14268
```

### Security

Configure security settings:

```yaml
settings:
  security:
    api_keys:
      enable: true
      keys:
        - name: production
          key: ${PROD_API_KEY}
          rate_limit: 1000/hour
          
    ssl:
      enable: true
      cert: ./certs/cert.pem
      key: ./certs/key.pem
      
    cors:
      origins: ["https://myapp.com"]
      methods: ["GET", "POST"]
      
    input_validation:
      max_length: 10000
      block_patterns: ["<script>", "DROP TABLE"]
```

## Environment Variables

Use environment variables for sensitive data:

```yaml
models:
  - name: openai
    api_key: ${OPENAI_API_KEY}
    
  - name: anthropic
    api_key: ${ANTHROPIC_API_KEY:-default_value}  # With default
    
settings:
  database:
    url: ${DATABASE_URL}
    password: ${DB_PASSWORD}
```

## Configuration Profiles

Use different configs for different environments:

```yaml title="llamafarm.dev.yaml"
extends: llamafarm.yaml

models:
  - name: assistant
    type: llama2-7b
    quantization: int4  # Faster for development
    
deploy:
  local:
    debug: true
    hot_reload: true
```

```yaml title="llamafarm.prod.yaml"
extends: llamafarm.yaml

models:
  - name: assistant
    type: llama2-70b
    quantization: float16  # Better quality
    
deploy:
  aws:
    auto_scale: true
    min_instances: 3
```

## Validation

LlamaFarm validates configurations before deployment:

```bash
# Validate configuration
llamafarm validate

# Validate specific file
llamafarm validate -f llamafarm.prod.yaml

# Show computed configuration
llamafarm config show --resolved
```

## Examples

### RAG Application

```yaml
models:
  - name: embedder
    type: sentence-transformers
    model: all-MiniLM-L6-v2
    
  - name: generator
    type: llama2-13b
    device: cuda
    
pipeline:
  - ingest:
      source: ./documents
      chunk_size: 512
      overlap: 50
      
  - embed:
      model: embedder
      
  - store:
      type: chromadb
      persist: ./db
      
  - retrieve:
      model: embedder
      top_k: 5
      
  - generate:
      model: generator
      template: |
        Context: {context}
        Question: {question}
        Answer:
```

### Multi-Language Support

```yaml
models:
  - name: translator
    type: m2m100
    
  - name: multilingual
    type: xlm-roberta
    
pipeline:
  - detect_language:
      model: multilingual
      
  - translate:
      model: translator
      target: english
      when: language != "en"
      
  - process:
      model: llama2
      
  - translate_back:
      model: translator
      target: ${detected_language}
```

### A/B Testing

```yaml
experiments:
  - name: prompt-test
    variants:
      - name: variant-a
        model: llama2
        prompt: "You are a helpful assistant."
        weight: 50
        
      - name: variant-b  
        model: llama2
        prompt: "You are a knowledgeable expert."
        weight: 50
        
    metrics:
      - response_quality
      - user_satisfaction
      - response_time
```

## Best Practices

1. **Start Simple** - Begin with minimal config and add features as needed
2. **Use Version Control** - Track configuration changes in Git
3. **Separate Environments** - Use different configs for dev/staging/prod
4. **Secure Secrets** - Never commit API keys, use environment variables
5. **Monitor Everything** - Enable logging and metrics from the start
6. **Document Changes** - Add comments explaining non-obvious configurations

## Next Steps

- Explore [Model Configuration](./models) for detailed model options
- Learn about [Deployment Strategies](./deployment) 
- See [Examples](./examples) for complete configurations