---
title: Examples
sidebar_label: Examples
slug: /configuration/example-configs
toc_min_heading_level: 2
toc_max_heading_level: 3
---

Reference configurations for common setups.

## RAG application

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

## Multi-language

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

## A/B testing

```yaml
experiments:
  - name: prompt-test
    variants:
      - name: variant-a
        model: llama2
        prompt: 'You are a helpful assistant.'
        weight: 50
      - name: variant-b
        model: llama2
        prompt: 'You are a knowledgeable expert.'
        weight: 50
    metrics:
      - response_quality
      - user_satisfaction
      - response_time
```
