#!/usr/bin/env python3
"""Pre-download llama.cpp binary for Docker builds.

This script is called during Docker build to pre-download the llama.cpp
binary. If the binary is not available for the current platform, it will
skip gracefully (the binary can be downloaded at runtime instead).
"""

import importlib.util
from pathlib import Path


def main():
    # Import _binary module directly to avoid triggering auto-download
    # from llamafarm_llama.__init__ (which calls get_lib_path() on import)
    # In Docker: /app/packages/llamafarm-llama/src/llamafarm_llama/_binary.py
    # Locally: relative to this script
    binary_path = Path("/app/packages/llamafarm-llama/src/llamafarm_llama/_binary.py")
    if not binary_path.exists():
        # Fallback for local development
        binary_path = (
            Path(__file__).resolve().parents[3]
            / "packages/llamafarm-llama/src/llamafarm_llama/_binary.py"
        )

    spec = importlib.util.spec_from_file_location("_binary", binary_path)
    _binary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_binary)

    BINARY_MANIFEST = _binary.BINARY_MANIFEST
    _should_build_from_source = _binary._should_build_from_source
    LLAMA_CPP_VERSION = _binary.LLAMA_CPP_VERSION
    _get_cache_dir = _binary._get_cache_dir
    download_binary = _binary.download_binary
    get_platform_key = _binary.get_platform_key

    platform_key = get_platform_key()
    print(f"Platform: {platform_key}")
    print(f"llama.cpp version: {LLAMA_CPP_VERSION}")

    # Check if binary is available for this platform or can be built
    if platform_key not in BINARY_MANIFEST and not _should_build_from_source(platform_key):
        # Check CPU fallback
        system, machine, _ = platform_key
        cpu_key = (system, machine, "cpu")
        if cpu_key not in BINARY_MANIFEST:
            print(
                f"WARNING: No pre-built llama.cpp binary available for {platform_key}"
            )
            print("The binary will be downloaded at runtime if needed.")
            print("Skipping pre-download step.")
            return

    cache_dir = _get_cache_dir() / LLAMA_CPP_VERSION
    print(f"Cache directory: {cache_dir}")

    try:
        download_binary(cache_dir)
        print("llama.cpp binary pre-downloaded successfully")
    except RuntimeError as e:
        if "404" in str(e) or "Not Found" in str(e):
            print(f"WARNING: Binary not found at release URL: {e}")
            print("The binary may not be published for this platform.")
            print("Skipping pre-download step.")
        else:
            # Re-raise other errors
            raise


if __name__ == "__main__":
    main()
