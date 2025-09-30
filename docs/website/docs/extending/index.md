---
title: Extending LlamaFarm
sidebar_position: 5
---

# Extending LlamaFarm

LlamaFarm is designed to be extended. This guide shows how to add new runtime providers, vector stores, parsers, extractors, and CLI commands while keeping schema validation intact.

## Extend Runtimes

1. **Update the schema**: add your provider to `runtime.provider` in `config/schema.yaml`.
2. **Regenerate types**:
   ```bash
   cd config
   ./generate-types.sh
   cd ..
   ```
3. **Implement routing**:
   - Server: update runtime selection (e.g., `server/services/runtime_service.py`) to handle the new provider and base URL.
   - CLI: ensure `config/datamodel.py` exposes your provider; if special flags are needed, add them under `cli/cmd`.
4. **Document usage**: add a section to this guide and the Configuration doc showing sample `llamafarm.yaml`.

Example (vLLM running with OpenAI-compatible API):

```yaml
runtime:
  provider: openai
  model: mistral-small
  base_url: http://localhost:8000/v1
  api_key: sk-test
```

No code changes required—just point `base_url` to your gateway.

## Extend RAG Components

### Add a Vector Store

1. Implement a store class (Python) that matches the existing store interface.
2. Register it with the RAG service so the server can instantiate it.
3. Add the store name to `rag/schema.yaml` under `databaseDefinition.type`.
4. Regenerate types and document configuration fields.

### Add a Parser or Extractor

1. Implement the parser/extractor (Python) with the required `process` signature.
2. Register it in the ingestion pipeline.
3. Append the new enum to `rag/schema.yaml` (`parsers` or `extractors` definitions) and define its config schema.
4. Regenerate types and update docs with usage examples.

## Extend the CLI

`lf` is built with Cobra.

1. Create a new file under `cli/cmd` (e.g., `backup.go`).
2. Define a `var myCmd = &cobra.Command{...}` and add it in `init()` with `rootCmd.AddCommand(myCmd)` or attach it to a namespace (datasets/rag).
3. Follow existing patterns for config resolution (`config.GetServerConfig`), auto-starting the server (`ensureServerAvailable`), and output formatting.
4. Write tests in `cli/cmd/..._test.go` if behaviour is complex.
5. Document the command in the CLI reference.

## Testing Extensions

- **Schema validation**: run `uv run --group test python -m pytest config/tests`.
- **CLI**: `cd cli && go test ./...`.
- **Server**: execute relevant pytest suites (e.g., `tests/test_project_chat_orchestrator.py`).
- **Docs**: update this page and run `nx build docs` to ensure navigation stays intact.

## Contribution Checklist

- [ ] Update schemas and regenerate types.
- [ ] Add or update server/CLI logic.
- [ ] Write tests covering new behaviour.
- [ ] Document configuration and usage.
- [ ] Note changes in `README.md` or example configs if appropriate.

Have questions? Open a [discussion](https://github.com/llama-farm/llamafarm/discussions) or join the [Discord](https://discord.gg/RrAUXTCVNF). We’re excited to see what you build.
