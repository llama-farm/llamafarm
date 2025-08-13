---
title: Models
sidebar_label: Models
slug: /models
---

High-level overview of the Models system: strategy, architecture, and quick commands.

## Vision & strategy

The Models system provides a config-driven, CLI-first framework for fine-tuning, managing, and deploying custom models.

- Configuration-driven workflows (JSON/YAML)
- Method-agnostic: LoRA, QLoRA, full fine-tune, adapters, prefix tuning
- Dataset-centric: tools to create and validate datasets from your data
- Production-ready: versioning, evaluation, deployment, monitoring

## Architecture overview

```
models/
├── methods/              # Fine-tuning methods (lora, qlora, full_finetune, adapters, ...)
├── datasets/             # Creation, formats, quality control, augmentation
├── config_examples/      # Example configurations
├── registry/             # Model & adapter registry
├── evaluation/           # Benchmarks, metrics, QA
├── deployment/           # Production deployment configs
└── utils/                # Helpers & tests
```

## Core CLI

```bash
# Dataset creation
llamafarm models create-dataset --source ./rag/data --format alpaca --output ./datasets/rag_qa.json

# Training
llamafarm models train --config ./config_examples/lora_basic.json --dataset ./datasets/rag_qa.json

# Evaluation
llamafarm models evaluate --model ./trained_models/lora_v1 --benchmark hellaswag

# Deployment
llamafarm models deploy --model ./trained_models/lora_v1 --target kubernetes --replicas 2

# Registry
llamafarm models list --type adapters
llamafarm models register --model ./trained_models/lora_v1 --name "rag_enhancement_v1"
llamafarm models rollback --name "rag_enhancement_v1" --version "1.0.0"
```

## Method selection (quick guide)

- Limited GPU memory: prefer QLoRA or LoRA
- Maximum quality: full fine-tuning with large datasets
- Multi-domain: adapters/registry for hot-swapping

See also: Providers and Adapters pages for runtime integration details.
