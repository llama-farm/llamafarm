---
title: Offline operation
sidebar_position: 10
---

# Offline operation

LlamaFarm runtimes can run fully offline — no network calls to HuggingFace,
no runtime downloads of llama.cpp binaries, no silent retries. This is the
deployment pattern used by llamadrone / arc on air-gapped Raspberry Pi
devices, and it works for any runtime that imports `llamafarm_common`.

Offline mode is env-var driven. There are two flags:

| Variable | What it does |
|---|---|
| `LLAMAFARM_OFFLINE=1` | Strict offline mode. The runtime fails loudly if a model or llama.cpp binary is missing, with a remediation message pointing at the right `lf` CLI command. Also propagates `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` so transitive `huggingface_hub`/`transformers` calls honor offline mode. |
| `LLAMAFARM_MODEL_DIR=/opt/llamafarm/models` | Flat-directory model layout root. When set, the runtime looks for models under `$LLAMAFARM_MODEL_DIR/<alias>/` first, before falling back to the HuggingFace cache. |

The two can be used together, separately, or not at all. The default
behavior (neither set) is unchanged from prior releases.

## Canonical on-device layout

The layout that `lf models path` (from the CLI) emits as `target` paths:

```
$LLAMAFARM_MODEL_DIR/
├── manifest.json                  ← written by your ops tooling, not the runtime
├── qwen3-1.7b/                    ← alias from runtime.models[].name
│   ├── model.Q4_K_M.gguf          ← main weights
│   └── mmproj.f16.gguf            ← optional multimodal projector
├── smollm-135m/
│   └── model.Q8_0.gguf
└── yolo11n/
    └── yolo11n.pt                 ← vision model (when runtime supports it)
```

The runtime discovers files by format sniffing — extension plus GGUF magic
bytes for `.gguf` files — rather than requiring specific filenames. Both the
canonical names above (`model.<QUANT>.gguf`, `mmproj.<precision>.gguf`) and
HF-preserved filenames (`Qwen3-1.7B-Q4_K_M.gguf`, `mmproj-qwen-f16.gguf`)
work identically. When multiple weights-candidate files exist in the same
alias directory, the quantization preference order `Q4_K_M > Q4_K > Q5_K_M >
Q5_K > Q8_0 > ...` is applied.

### How the alias is determined per runtime

**Edge runtime** — the alias is auto-derived from the incoming `model` field
of the API request by stripping the `org/` prefix and `:quant` suffix. All of
these request values map to the same alias directory `Qwen3-0.6B-GGUF`:

| Request `model` | Derived alias |
|---|---|
| `Qwen/Qwen3-0.6B-GGUF:Q4_K_M` | `Qwen3-0.6B-GGUF` |
| `Qwen/Qwen3-0.6B-GGUF:Q8_0` | `Qwen3-0.6B-GGUF` |
| `Qwen/Qwen3-0.6B-GGUF` | `Qwen3-0.6B-GGUF` |
| `Qwen3-0.6B-GGUF` | `Qwen3-0.6B-GGUF` |

This means API clients don't have to change anything — an operator places
`/opt/llamafarm/models/Qwen3-0.6B-GGUF/model.Q4_K_M.gguf` on the Pi, and
every request form above finds it. The trade-off: `foo/my-model` and
`bar/my-model` collide on the same alias directory. If you need to
disambiguate, send distinct base names in your request model_ids.

**Universal runtime** — if you're using the universal runtime's GGUF models,
pass `alias=<name>` explicitly when constructing `GGUFLanguageModel`, or
leave it unset to keep the legacy HF-cache-first behavior. Auto-derivation
is edge-specific for now.

## Resolution order

For each model alias, the runtime resolves its weights file in this order,
first match wins:

```
  1. $LLAMAFARM_MODEL_DIR/<alias>/  (format-sniffed)
          │
          ▼
  2. HuggingFace cache             (existing behavior)
          │
          ▼
  3. Network download              (skipped when LLAMAFARM_OFFLINE=1)
```

In strict offline mode, step 3 is removed. If no tier matches, the runtime
raises `FileNotFoundError` with a multi-line message naming the alias, the
paths that were tried, and the `lf` command that would make the file
available.

### What about absolute paths in `runtime.models[].model`?

Absolute filesystem paths (e.g. `runtime.models[0].model: /data/custom.gguf`)
are **not** resolved through the `LLAMAFARM_MODEL_DIR` tier — they flow
through the legacy `get_gguf_file_path` entry point, which handles
`.gguf`-suffixed inputs via a safe-directory basename lookup under
`~/.llamafarm/models/` or `$GGUF_MODELS_DIR/`. This preserves existing
behavior for projects that reference hand-placed files by absolute path;
nothing about this handling changes with `LLAMAFARM_MODEL_DIR`.

If you want your hand-placed GGUF to be discovered via the canonical
`<alias>/` layout, either (a) move it under `$LLAMAFARM_MODEL_DIR/<alias>/`
and reference it by alias, or (b) move it under `$GGUF_MODELS_DIR/` with
its basename matching the `runtime.models[].model` value and continue
referencing by filename.

## End-to-end workflow with `lf models path`

The companion workflow on the CLI side (from `feat-cli-models-path`):

```bash
# 1. On a build host with internet, populate the HF cache
lf models pull

# 2. Get a transport plan telling you where the files live and where they
#    should go on the device
lf models path --format json --target-root /opt/llamafarm/models

# 3. Your ops tooling (ansible, packer, rsync) copies the files per the plan
#    Example: the Ansible playbook snippet from the lf-models docs
```

See [`lf models path`](../cli/lf-models.md#lf-models-path) for the full flag
reference and example Ansible playbook.

## Docker compose example

```yaml
# docker-compose.yml on an air-gapped edge device
services:
  llamafarm-edge:
    image: llamafarm/edge-runtime:latest
    ports:
      - "11540:11540"
    environment:
      LLAMAFARM_OFFLINE: "1"                         # strict offline, no retries
      LLAMAFARM_MODEL_DIR: /models                   # flat-dir layout
      HF_HUB_OFFLINE: "1"                            # belt-and-suspenders
      TRANSFORMERS_OFFLINE: "1"                      # belt-and-suspenders
      LD_LIBRARY_PATH: /opt/llamafarm/bin            # where llama.cpp lives
    volumes:
      - /opt/llamafarm/models:/models:ro             # bind-mount from host
      - /opt/llamafarm/bin:/opt/llamafarm/bin:ro     # llama.cpp binary + deps
```

Note that `LLAMAFARM_OFFLINE=1` automatically sets the two `HF_*OFFLINE`
variables for you, so in practice you only need `LLAMAFARM_OFFLINE` and
`LLAMAFARM_MODEL_DIR` in the compose file. The others are shown above for
clarity.

The `/opt/llamafarm/bin` directory is populated on the host via:

```bash
lf runtime binary pull \
  --platform linux/arm64 \
  --accelerator cpu \
  --export /opt/llamafarm/bin
```

See [`lf runtime binary`](../cli/lf-runtime.md) for details.

## Startup verification

On startup, the runtime emits a single structured log line showing the
resolved mode:

```
INFO  llamafarm_offline_mode  mode=offline
                              model_dir=/opt/llamafarm/models
                              hf_hub_offline=1
                              transformers_offline=1
```

This is your grep-able verification that the deployment configuration was
picked up correctly. If you see `mode=online` when you expected offline, the
env var was not inherited by the runtime process.

## Troubleshooting

**Error: `Model 'qwen3-1.7b' not available in offline mode.`**

```
Model 'qwen3-1.7b' not available in offline mode.
  Tried: /opt/llamafarm/models/qwen3-1.7b/
  Tried: /root/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B-GGUF
  To fix: run 'lf models pull Qwen/Qwen3-1.7B-GGUF' on a host with internet,
          then sync the files to this host
  Note: If the build host has internet, you can also use
          'lf models path --ensure' to pull before emitting a plan.
```

The error names the alias, both places it looked, and the command that
would populate them. On the build host, either:

- Run `lf models pull Qwen/Qwen3-1.7B-GGUF` to cache the model, then
  re-sync the files to the device, OR
- Run `lf models path --ensure --format json` to pull and emit in one
  shot, then ship the files per the plan.

**Error: `llama.cpp binary not available in offline mode for linux/arm64`**

The runtime couldn't find the llama.cpp shared library in any cached
location. Fix with:

```bash
lf runtime binary pull --platform linux/arm64 --accelerator cpu --export /opt/llamafarm/bin
```

Then sync `/opt/llamafarm/bin/` to the device.

**Warning: `skipping <path>: .gguf extension but missing GGUF magic bytes`**

Something with a `.gguf` extension exists in your alias directory but its
first four bytes are not `GGUF`. Usually this means:

- The file is a partial download (truncated) — delete it and re-sync.
- The file is a text file or symlink target that isn't resolvable.
- The file is from a corrupt copy step.

The runtime refuses to hand this file to llama-cpp because loading it would
produce an opaque crash later.

**Warning: `LLAMAFARM_MODEL_DIR=<path> does not exist on disk`**

The root directory named by `LLAMAFARM_MODEL_DIR` does not exist. The
runtime logs this as a warning and falls through to the HF cache. If you
expected offline mode to fail here, you also need `LLAMAFARM_OFFLINE=1`;
otherwise the cache fallback is doing its job.

**I see `mode=online` but I set `LLAMAFARM_OFFLINE=1`**

The env var isn't reaching the runtime process. In a Docker container,
check that the `environment:` block in `docker-compose.yml` actually
contains the variable and that `docker compose config` shows it in the
resolved configuration. In an ansible-managed unit file, check that the
`Environment=` line is present in the final rendered unit and that you ran
`systemctl daemon-reload`.

## Related

- [`lf models path`](../cli/lf-models.md#lf-models-path) — build-host-side transport plan emitter
- [`lf runtime binary pull`](../cli/lf-runtime.md) — cross-platform llama.cpp binary fetcher
