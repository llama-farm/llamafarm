---
title: lf runtime
sidebar_position: 9
---

# `lf runtime`

Manage shared runtime components that the LlamaFarm server, edge runtime, and
universal runtime all depend on. Today this is scoped to llama.cpp binaries;
future subcommands may manage other shared infrastructure.

These commands are primarily for **deployment pipelines** — especially
air-gapped or Dockerfile builds where you cannot rely on first-run downloads.

## Synopsis

```
lf runtime binary pull [flags]
lf runtime binary path [flags]
```

## `lf runtime binary pull`

Download the pinned llama.cpp shared library and its dependency files for a
target platform and accelerator into the local LlamaFarm cache.

```bash
# Current host, best accelerator (default)
lf runtime binary pull

# Linux ARM64 build for a Raspberry Pi (from a macOS dev host)
lf runtime binary pull --platform linux/arm64 --accelerator cpu

# Fetch and materialize a flat directory for Ansible/Packer pickup
lf runtime binary pull --platform linux/arm64 --export /tmp/lf-bin

# Pin a specific llama.cpp release
lf runtime binary pull --version b7800
```

**Flags:**
- `--platform <os>/<arch>` — Target OS/arch (e.g. `linux/arm64`, `darwin/arm64`). Defaults to the current host.
- `--accelerator <backend>` — Compute backend: `cpu`, `cuda`, `metal`, `vulkan`, `rocm`. Defaults to the best supported backend for the target platform.
- `--version <tag>` — llama.cpp version tag. Defaults to the version baked into this CLI (from `llama-cpp-version.txt`).
- `--export <dir>` — After download, copy the binary and all dependency libraries into this directory as a flat layout (preserves symlinks). Size on disk is typically 50–200 MB.

**Supported platform/accelerator combinations:**

| OS       | Arch  | Accelerator            |
|----------|-------|------------------------|
| darwin   | arm64 | metal                  |
| darwin   | amd64 | cpu                    |
| linux    | amd64 | cpu, vulkan            |
| linux    | arm64 | cpu                    |
| windows  | amd64 | cpu, cuda, vulkan      |

Linux `cuda` and `rocm` fall back to `vulkan` because upstream llama.cpp no
longer ships prebuilt CUDA/ROCm Linux binaries.

## `lf runtime binary path`

Print the absolute path to the cached main library for the specified target.
Exits non-zero with a remediation message if the binary has not been
downloaded for that target yet.

```bash
lf runtime binary path
lf runtime binary path --platform linux/arm64 --accelerator cpu
```

Flags match `lf runtime binary pull` except there is no `--export`.

## Example: Dockerfile integration

Pre-fetch the llama.cpp binary at image build time instead of on first
container start. Note that **models are not baked into the image** — they
are mounted from the host filesystem at runtime.

```dockerfile
FROM ubuntu:24.04 AS base
# ... base setup ...

# Fetch the llama.cpp binary for the target platform into a flat dir.
# The dir is then copied into the image at a known location.
RUN lf runtime binary pull \
      --platform linux/arm64 \
      --accelerator cpu \
      --export /opt/llamafarm/bin

ENV LD_LIBRARY_PATH=/opt/llamafarm/bin
ENV LLAMAFARM_OFFLINE=1

# Models are mounted at runtime, not baked in:
VOLUME /opt/llamafarm/models
```

## Example: Ansible integration

```yaml
- name: Pull llama.cpp binary for target
  command: >
    lf runtime binary pull
      --platform linux/arm64
      --accelerator cpu
      --export /tmp/lf-bin
  delegate_to: localhost

- name: Sync binary dir to device
  synchronize:
    src: /tmp/lf-bin/
    dest: /opt/llamafarm/bin/
    rsync_opts: ["--checksum"]
```

## See Also

- [`lf models path`](./lf-models.md#lf-models-path) — companion command for model file transport plans
- [`lf deploy`](./lf-start.md) — full deployment workflow
