## Why

The edge runtime currently cannot operate without network access. Its model-loading path (`common/llamafarm_common/model_utils.py`) always consults the HuggingFace API before falling back to the local cache, and the llama.cpp binary loader (`packages/llamafarm-llama/src/llamafarm_llama/_binary.py`) silently downloads from GitHub on first use. Both behaviors make the runtime unusable in air-gapped deployments — notably llamadrone / arc, where Raspberry Pi devices run the edge runtime in a Docker container with no internet connectivity. The companion change `feat-cli-models-path` (just landed) gives deployment tooling a way to place model files and llama.cpp binaries on the device, but the runtime itself still assumes network access. This change closes the loop so the runtime can consume the canonical `<target-root>/<alias>/` layout and honor a strict offline mode.

## What Changes

- Add a `LLAMAFARM_OFFLINE=1` environment variable that forces the runtime into strict offline mode. When set, any path that would have made a network call instead raises a clear error with actionable remediation (pointing at `lf models pull` or `lf runtime binary pull`). The runtime also propagates `HF_HUB_OFFLINE=1` so transitive `huggingface_hub` and `transformers` calls honor it.
- Add a `LLAMAFARM_MODEL_DIR` environment variable that enables flat-directory model loading. When set, the runtime looks for models under `$LLAMAFARM_MODEL_DIR/<alias>/` first, falling back to the HuggingFace cache only when offline mode is not enabled.
- Extend `common/llamafarm_common/model_utils.py` with offline-mode guards and alias-directory lookup helpers. Fail loudly rather than silently when offline and the model is missing.
- Extend `packages/llamafarm-llama/src/llamafarm_llama/_binary.py` with offline-mode guards. Fail loudly rather than silently when offline and the binary is missing.
- Update `runtimes/edge/models/gguf_language_model.py` to call the new alias-directory resolver before the existing HF-cache path. Transformers-based models in the universal runtime (`from_pretrained`) already honor `HF_HUB_OFFLINE=1` transitively — no code changes there beyond the env-var propagation.
- Log the runtime's resolved mode (`offline` vs `online`) and `LLAMAFARM_MODEL_DIR` value once on startup so operators can verify deployment state.
- Add documentation for offline operation, the canonical flat-directory layout, and a Docker compose example that pairs with `lf models path` from the companion change.

## Capabilities

### New Capabilities
- `runtime-offline`: Strict offline mode for the edge runtime (and any other runtime that shares `model_utils.py`). Defines env-var-driven behavior, the lookup precedence for model files, and the failure semantics when files are missing in offline mode.
- `runtime-model-dir`: Flat-directory model layout support. Defines how the runtime discovers model files in `$LLAMAFARM_MODEL_DIR/<alias>/` using format-sniffing rather than specific filenames, and how this tier fits into the overall resolution order.

### Modified Capabilities

*(None — this change introduces new behavior rather than modifying an existing capability spec. The canonical layout convention lives in `feat-cli-models-path`'s `models-path` capability as `target` paths; this change simply consumes that convention from the runtime side.)*

## Impact

- **Runtime code** (Python): `common/llamafarm_common/model_utils.py`, `packages/llamafarm-llama/src/llamafarm_llama/_binary.py`, `runtimes/edge/models/gguf_language_model.py`, `runtimes/edge/server.py` (or wherever startup logging lives).
- **Environment variables**: Two new vars (`LLAMAFARM_OFFLINE`, `LLAMAFARM_MODEL_DIR`) that deployment tooling sets; no config schema change.
- **No CLI changes**: `feat-cli-models-path` already provides the tools that populate the on-device layout. This change is pure runtime.
- **No lemonade runtime changes**: Out of scope for this change.
- **Universal runtime**: Touched only to propagate `HF_HUB_OFFLINE=1` into the startup environment; deeper GGUF-loader parity with the edge runtime can follow in a separate change if needed.
- **Documentation**: New offline-operation guide in `docs/website/docs/models/` or extending the edge runtime docs; Docker compose example that bind-mounts `/opt/llamafarm/models` into the edge runtime container with the two env vars set.
- **Downstream consumers**: llamadrone / arc can finally cut the network tether on their Pi deployments using the end-to-end flow: `lf models pull` → `lf models path --format json` → ansible → bind-mounted container → offline runtime.
