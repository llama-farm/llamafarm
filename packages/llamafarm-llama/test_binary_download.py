#!/usr/bin/env python3
"""
Test script for debugging llama.cpp binary download and backend loading.

Usage:
    # Run locally
    python test_binary_download.py

    # Run in Docker (Linux)
    docker run --rm -v $(pwd):/app -w /app python:3.12-slim bash -c "
        pip install cffi &&
        python test_binary_download.py
    "
"""

import os
import platform
import shutil
import sys
from pathlib import Path

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    print("=" * 60)
    print("llama.cpp Binary Download Test")
    print("=" * 60)
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version}")
    print()

    # Clear cache to force fresh download
    from llamafarm_llama._binary import LLAMA_CPP_VERSION, _get_cache_dir

    cache_dir = _get_cache_dir()
    version_dir = cache_dir / LLAMA_CPP_VERSION

    print(f"Cache directory: {cache_dir}")
    print(f"Version directory: {version_dir}")

    if version_dir.exists():
        print(f"Clearing existing cache at {version_dir}...")
        shutil.rmtree(version_dir)

    print()
    print("=" * 60)
    print("Step 1: Download binary")
    print("=" * 60)

    from llamafarm_llama._binary import _get_lib_name, download_binary, get_platform_key

    platform_key = get_platform_key()
    print(f"Platform key: {platform_key}")
    print(f"Library name: {_get_lib_name()}")
    print()

    try:
        lib_path = download_binary(version_dir, platform_key)
        print(f"\nDownloaded to: {lib_path}")
    except Exception as e:
        print(f"ERROR downloading: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print()
    print("=" * 60)
    print("Step 2: List extracted files")
    print("=" * 60)

    if version_dir.exists():
        files = list(version_dir.iterdir())
        print(f"Found {len(files)} files in {version_dir}:")
        for f in sorted(files):
            if f.is_symlink():
                target = os.readlink(f)
                print(f"  {f.name} -> {target} (symlink)")
            else:
                size = f.stat().st_size
                print(f"  {f.name} ({size:,} bytes)")
    else:
        print(f"ERROR: Version directory does not exist: {version_dir}")
        return 1

    print()
    print("=" * 60)
    print("Step 3: Test library loading")
    print("=" * 60)

    import cffi

    lib_name = _get_lib_name()
    lib_file = version_dir / lib_name

    if not lib_file.exists():
        print(f"ERROR: Library not found: {lib_file}")
        return 1

    # Resolve symlinks
    actual_lib = lib_file
    while actual_lib.is_symlink():
        target = os.readlink(actual_lib)
        actual_lib = (actual_lib.parent / target).resolve()

    print(f"Library: {lib_file}")
    print(f"Actual file: {actual_lib}")
    print(f"Size: {actual_lib.stat().st_size:,} bytes")

    # Set up environment
    system = platform.system()
    if system == "Linux":
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{version_dir}:{ld_path}" if ld_path else str(version_dir)
        print(f"Set LD_LIBRARY_PATH: {os.environ['LD_LIBRARY_PATH']}")
    elif system == "Windows":
        path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{version_dir};{path}"
        try:
            os.add_dll_directory(str(version_dir))
        except AttributeError:
            pass

    # Try loading the library
    ffi = cffi.FFI()
    ffi.cdef("""
        void llama_backend_init(void);
        void ggml_backend_load_all(void);
    """)

    print()
    print("Loading libllama...")
    try:
        lib = ffi.dlopen(str(lib_file))
        print("  OK - libllama loaded")
    except Exception as e:
        print(f"  ERROR loading libllama: {e}")
        return 1

    print()
    print("Calling llama_backend_init()...")
    try:
        lib.llama_backend_init()
        print("  OK - backend initialized")
    except Exception as e:
        print(f"  ERROR: {e}")
        return 1

    print()
    print("Calling ggml_backend_load_all() from libllama...")
    try:
        lib.ggml_backend_load_all()
        print("  OK - backends loaded from libllama")
        return 0
    except Exception as e:
        print(f"  Not in libllama: {e}")

    # Try loading from libggml
    print()
    print("Looking for libggml...")

    ggml_candidates = [
        version_dir / "libggml.so.0",
        version_dir / "libggml.so",
        version_dir / "ggml.dll",
    ]

    # Also search for any libggml file
    ggml_candidates.extend(version_dir.glob("libggml*"))

    ggml_lib_path = None
    for candidate in ggml_candidates:
        if candidate.exists() and candidate.is_file():
            ggml_lib_path = candidate
            break

    if ggml_lib_path:
        print(f"Found: {ggml_lib_path}")

        # Resolve symlinks
        actual_ggml = ggml_lib_path
        while actual_ggml.is_symlink():
            target = os.readlink(actual_ggml)
            actual_ggml = (actual_ggml.parent / target).resolve()
        print(f"Actual file: {actual_ggml}")

        print()
        print("Loading libggml...")
        try:
            ggml_ffi = cffi.FFI()
            ggml_ffi.cdef("void ggml_backend_load_all(void);")
            ggml_lib = ggml_ffi.dlopen(str(ggml_lib_path))
            print("  OK - libggml loaded")
        except Exception as e:
            print(f"  ERROR loading libggml: {e}")
            return 1

        print()
        print("Calling ggml_backend_load_all() from libggml...")
        try:
            ggml_lib.ggml_backend_load_all()
            print("  OK - backends loaded from libggml")
            return 0
        except Exception as e:
            print(f"  ERROR: {e}")
            return 1
    else:
        print("  No libggml found")
        print("  Available files:")
        for f in sorted(version_dir.iterdir()):
            print(f"    {f.name}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
