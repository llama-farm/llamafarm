---
title: Templates
sidebar_label: Templates
slug: /prompts/templates
toc_min_heading_level: 2
toc_max_heading_level: 3
---

Manage, test, and evolve prompt templates.

## Basic usage

```bash
# List all available templates
uv run python -m prompts.cli template list

# Execute a simple query
uv run python -m prompts.cli execute "What is machine learning?" --show-details

# Test a specific template
uv run python -m prompts.cli template test qa_basic \
  --variables '{"query":"What is AI?", "context":[{"title":"AI Guide", "content":"AI is..."}]}'
```

## Template modification patterns

### 1) Adding context-aware behavior

```python
# Original template (basic)
{
  "template": "Answer the question: {{query}}\n\nAnswer:"
}

# Enhanced with context awareness
{
  "template": "{% if context %}Based on the following context:\n{% for doc in context %}[{{doc.title}}]: {{doc.content}}\n{% endfor %}{% endif %}\n\nQuestion: {{query}}\n\nProvide a detailed answer{% if context %} using the provided context{% endif %}:"
}
```

### 2) Multi-language support

```bash
# Create language-specific template variants
cp templates/basic/qa_basic.json templates/basic/qa_basic_es.json
# Edit for Spanish and add language detection in your strategy
```

### 3) Dynamic template selection

```python
# beginner_explanation.json
{
  "template": "Explain {{topic}} in simple terms, as if explaining to someone new to the field:\n\n"
}
# expert_analysis.json
{
  "template": "Provide an in-depth technical analysis of {{topic}}, including:\n- Advanced concepts\n- Current research\n- Technical implications\n\n"
}
# Use with dynamic selection
uv run python -m prompts.cli execute "Explain quantum computing" \
  --template beginner_explanation \
  --variables '{"topic": "quantum computing", "user_level": "beginner"}'
```

## Integration examples

### RAG pipeline

```python
import asyncio
from prompts.core.prompt_system import PromptSystem
from prompts.models.config import PromptConfig

async def rag_query(question: str, retrieved_docs: list):
    config = PromptConfig.from_file('config/default_prompts.json')
    prompt_system = PromptSystem(config)
    result = prompt_system.execute_prompt(
        query=question,
        variables={"context": retrieved_docs, "source_citations": True},
        template_override="qa_detailed"
    )
    return result.rendered_prompt
```

### A/B testing templates

```bash
cp templates/basic/qa_basic.json templates/basic/qa_basic_v2.json
uv run python -m prompts.cli execute "Explain machine learning" --template qa_basic > response_a.txt
uv run python -m prompts.cli execute "Explain machine learning" --template qa_basic_v2 > response_b.txt
uv run python -m prompts.cli evaluate "$(cat response_a.txt)" \
  --query "Explain machine learning" \
  --criteria "clarity,completeness" \
  --output-format score > score_a.txt
uv run python -m prompts.cli evaluate "$(cat response_b.txt)" \
  --query "Explain machine learning" \
  --criteria "clarity,completeness" \
  --output-format score > score_b.txt
```
