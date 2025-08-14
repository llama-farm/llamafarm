---
title: Architecture
sidebar_label: Architecture
slug: /overview/architecture
toc_min_heading_level: 2
toc_max_heading_level: 3
---

Modular, configuration-driven architecture.

```text
llamafarm/
├── rag/              # Document processing and retrieval
│   ├── core/         # Base classes and interfaces
│   ├── parsers/      # Document parsers (PDF, CSV, etc.)
│   ├── embedders/    # Text embedding models
│   ├── stores/       # Vector database integrations
│   └── retrieval/    # Retrieval strategies
│
├── models/           # Model management
│   ├── providers/    # LLM provider integrations
│   ├── config/       # Configuration system
│   ├── monitoring/   # Usage tracking and analytics
│   └── optimization/ # Cost and performance optimization
│
├── prompts/          # Prompt engineering
│   ├── templates/    # Prompt template library
│   ├── strategies/   # Template selection strategies
│   ├── evaluation/   # Response evaluation tools
│   └── agents/       # Multi-agent coordination
│
└── training/         # Fine-tuning (coming soon)
    ├── datasets/     # Dataset management
    ├── trainers/     # Training implementations
    └── evaluation/   # Model evaluation
```
