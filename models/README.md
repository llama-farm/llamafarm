# Models & Runtime

This package contains helpers for selecting and invoking the runtime defined in `llamafarm.yaml`. Today the project focuses on inference (chat completions) rather than fine-tuning.

## What Exists Today
- Runtime adapters for the server and CLI (`openai`-compatible hosts, Ollama).
- Agent handler wiring (simple chat, structured chat, RAG-aware prompts).
- Type definitions generated from the configuration schema.

## Roadmap
Fine-tuning, evaluation harnesses, and advanced strategy management are planned but not implemented yet. Track progress in issues/roadmap before relying on files under `docs/` for production use—they are placeholders for future work.

## Extending Providers
1. Add the provider enum entry in `config/schema.yaml` (`runtime.provider`).
2. Regenerate types (`config/generate-types.sh`).
3. Implement the provider in the server runtime service and CLI resolver.
4. Document usage in the docs (`docs/website/docs/models/index.md`).

## Tests
```bash
uv run pytest tests/
```

## See Also
- Top-level docs: `docs/website/docs/models/index.md`
- Configuration reference: `docs/website/docs/configuration/index.md`
