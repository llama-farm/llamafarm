---
title: Configuration
sidebar_label: Configuration
slug: /configuration
---

Declarative YAML configuration for models, pipelines, deployment, and settings.

## Philosophy

1. Configuration over code
2. Sensible defaults
3. Progressive complexity
4. Environment aware

## Basic configuration

```yaml title="llamafarm.yaml"
models:
  - name: assistant
    type: llama2-7b

deploy:
  local: true
```

## Structure

```yaml
models: [] # Model definitions
pipeline: [] # Dataflow through components
deploy: {} # Deployment targets
settings: {} # Observability, security, etc.
```

## Models section

```yaml
models:
  - name: local-llama
    type: llama2-13b
    device: cuda
    quantization: int8
    cache_dir: ./models

  - name: cloud-gpt
    type: openai
    model: gpt-4
    api_key: ${OPENAI_API_KEY}
```

## Pipeline section

```yaml
pipeline:
  - input: user_query
  - generate:
      model: local-llama
      max_tokens: 500
      temperature: 0.7
  - output: response
```

## Deploy section

```yaml
deploy:
  local:
    enable: true
    port: 8080
    workers: 4
  kubernetes:
    namespace: llamafarm
    replicas: 3
```

## Environment variables

```yaml
settings:
  database:
    url: ${DATABASE_URL}
    password: ${DB_PASSWORD}
```

## Profiles

```yaml title="llamafarm.dev.yaml"
extends: llamafarm.yaml
models:
  - name: assistant
    type: llama2-7b
    quantization: int4

deploy:
  local:
    debug: true
    hot_reload: true
```

## Validation

```bash
llamafarm validate
llamafarm validate -f llamafarm.prod.yaml
llamafarm config show --resolved
```

## Best practices

- Start simple, add features incrementally
- Use version control for config
- Separate dev/staging/prod
- Secure secrets with env vars
- Enable logging and metrics
- Document non-obvious choices

See: Example configs for complete scenarios.
