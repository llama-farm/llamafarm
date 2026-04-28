# Install LlamaFarm CLI

Recommended install instructions now live in the main README and docs site.

## Quick Reference
- macOS / Linux:
  ```bash
  curl -fsSL https://raw.githubusercontent.com/llama-farm/llamafarm/main/install.sh | bash
  ```
- Windows: download `lf.exe` from the [latest release](https://github.com/llama-farm/llamafarm/releases/latest) and add it to your PATH.

## Bundled binaries
- CLI release archives now include a bundled `llama.cpp` runtime under `llama-cpp/<os>-<arch>/`, next to the `lf` executable.
- PyApp bundles stage the same runtime inside `llamafarm_llama/_bundled/<os>-<arch>/`.
- On first run, LlamaFarm checks those bundled paths before attempting any network download. CPU bundles ship first; GPU-specific bundle variants remain a TODO.

For source builds and advanced usage, see:
- `README.md` (Quickstart)
- `docs/website/docs/cli/index.md` (CLI reference)
