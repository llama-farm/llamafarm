---
title: Evaluation
sidebar_label: Evaluation
slug: /prompts/evaluation
toc_min_heading_level: 2
toc_max_heading_level: 3
---

Judge outputs and iterate to better prompts.

## Criteria

Define objective metrics and rubrics. Example CLI usage:

```bash
# Evaluate AI responses
uv run python -m prompts.cli evaluate "AI is machine learning" \
  --query "What is AI?" \
  --criteria "accuracy,clarity,completeness" \
  --output-format detailed
```

## A/B testing

Compare prompt variants with the same inputs.

```bash
# Create variant templates
cp templates/basic/qa_basic.json templates/basic/qa_basic_v2.json

# Run A/B
uv run python -m prompts.cli execute "Explain machine learning" --template qa_basic > response_a.txt
uv run python -m prompts.cli execute "Explain machine learning" --template qa_basic_v2 > response_b.txt

# Evaluate both
uv run python -m prompts.cli evaluate "$(cat response_a.txt)" \
  --query "Explain machine learning" \
  --criteria "clarity,completeness" \
  --output-format score > score_a.txt
uv run python -m prompts.cli evaluate "$(cat response_b.txt)" \
  --query "Explain machine learning" \
  --criteria "clarity,completeness" \
  --output-format score > score_b.txt
```

## Automation

Wire into CI with your preferred runner. Examples:

- Validate templates and run evaluation on PRs
- Fail builds on regression thresholds (e.g., clarity score drop)
