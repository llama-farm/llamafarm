---
title: Providers
sidebar_label: Providers
slug: /models/providers
toc_min_heading_level: 2
toc_max_heading_level: 3
---

Configure and use model providers (local and cloud).

## Local runtimes

```yaml
models:
  - name: local-llama
    type: llama2-13b
    device: cuda # or cpu, auto
    quantization: int8
    cache_dir: ./models
```

## Cloud APIs

```yaml
models:
  - name: cloud-gpt
    type: openai
    model: gpt-4
    api_key: ${OPENAI_API_KEY}

  - name: anthropic
    type: anthropic
    model: claude-3-haiku
    api_key: ${ANTHROPIC_API_KEY}
```

## Multi-provider routing

Use tags and routing rules to steer requests.

```yaml
models:
  - name: fast-model
    type: llama2-7b
    tags: ['fast', 'general']

  - name: accurate-model
    type: llama2-70b
    tags: ['accurate', 'slow']

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

## Fallbacks and cascading

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

## Secrets and env vars

```yaml
models:
  - name: openai
    api_key: ${OPENAI_API_KEY}
```

For deployment targets (Docker/Kubernetes), see the Deployment section.
