#!/usr/bin/env python3
"""
Download llama.cpp binary for the current platform.

Usage:
    python scripts/download_binary.py
    python scripts/download_binary.py --backend cpu
    python scripts/download_binary.py --output ./lib/
"""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llamafarm_llama._binary import (
    download_binary,
    get_platform_key,
    get_binary_info,
    LLAMA_CPP_VERSION,
    BINARY_MANIFEST,
)


def main():
    parser = argparse.ArgumentParser(description="Download llama.cpp binary")
    parser.add_argument(
        "--backend",
        choices=["cpu", "cuda11", "cuda12", "metal", "vulkan"],
        help="Force specific backend",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory (default: cache)",
    )
    parser.add_argument(
        "--list-platforms",
        action="store_true",
        help="List available platforms",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show current binary info",
    )
    args = parser.parse_args()

    if args.list_platforms:
        print(f"llama.cpp version: {LLAMA_CPP_VERSION}")
        print("\nAvailable platforms:")
        for key in sorted(BINARY_MANIFEST.keys()):
            manifest = BINARY_MANIFEST[key]
            print(f"  {key}: {manifest['artifact'].format(version=LLAMA_CPP_VERSION)}")
        return

    if args.info:
        info = get_binary_info()
        print("Binary Info:")
        for k, v in info.items():
            print(f"  {k}: {v}")
        return

    # Get platform key
    platform_key = get_platform_key(backend_override=args.backend)
    print(f"Platform: {platform_key}")

    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        from llamafarm_llama._binary import _get_cache_dir

        output_dir = _get_cache_dir() / LLAMA_CPP_VERSION

    print(f"Output: {output_dir}")

    # Download
    lib_path = download_binary(output_dir, platform_key=platform_key)
    print(f"\nDownloaded: {lib_path}")


if __name__ == "__main__":
    main()
