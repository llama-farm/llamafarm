---
title: Adapters
sidebar_label: Adapters
slug: /models/adapters
toc_min_heading_level: 2
toc_max_heading_level: 3
---

Adapters provide a stable interface for model I/O and enable hot-swapping specialized capabilities at runtime.

## Why adapters

- Normalize inputs/outputs across providers
- Swap domain-specialized behavior without retraining the base model
- Reduce deployment friction by decoupling runtime from training artifacts

## Hot-swappable adapters

```bash
# Switch adapters in production without restart
llamafarm models swap-adapter --adapter medical_specialist --target production
```

## Registry & versioning

Track adapter metadata, versions, and performance.

```bash
# Register
llamafarm models register \
  --name "medical_qa_specialist" \
  --version "1.2.0" \
  --base-model "llama-2-7b" \
  --method "lora" \
  --dataset "medical_qa_v2" \
  --performance-metrics "./eval_results.json"

# List and pull
llamafarm models list --filter domain=medical
llamafarm models pull medical_qa_specialist:1.1.0

# Promote
llamafarm models promote medical_qa_specialist:1.2.0 --env production
```

## Implementation notes

- Memory optimization: gradient checkpointing, mixed precision, quantization
- Performance: batching, caching, compilation
- Security: provenance, RBAC, audit logging
