#!/usr/bin/env python3
"""
Nuitka Build Script for LlamaFarm Services

Compiles server, rag, and runtime components into standalone executables.
Native dependencies (llama.cpp) are downloaded by the CLI.

Usage:
    python build/nuitka/build.py --component server
    python build/nuitka/build.py --component rag
    python build/nuitka/build.py --component runtime
    python build/nuitka/build.py --all

Output:
    dist/
    ├── llamafarm-server-{platform}-{arch}
    ├── llamafarm-rag-{platform}-{arch}
    └── llamafarm-runtime-{platform}-{arch}
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Project root is 2 levels up from this file
PROJECT_ROOT = Path(__file__).parent.parent.parent
DIST_DIR = PROJECT_ROOT / "dist" / "nuitka"


def find_python_with_nuitka() -> str:
    """Find a Python interpreter that has nuitka installed."""
    # Try common Python commands in order of preference
    candidates = [
        sys.executable,
        "python3.12",
        "python3.11",
        "python3",
        "python",
        # Also try conda paths
        "/opt/homebrew/Caskroom/miniconda/base/bin/python",
    ]

    for python in candidates:
        try:
            # Check if nuitka module is importable
            result = subprocess.run(
                [python, "-c", "import nuitka; print('found')"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and "found" in result.stdout:
                # Get version via CLI
                version_result = subprocess.run(
                    [python, "-m", "nuitka", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                version = version_result.stdout.split("\n")[0] if version_result.returncode == 0 else "unknown"
                print(f"Found nuitka {version} in {python}")
                return python
        except (subprocess.SubprocessError, FileNotFoundError):
            continue

    print("ERROR: nuitka not found in any Python interpreter")
    print("Install with: pip install nuitka")
    sys.exit(1)


def get_platform_suffix() -> str:
    """Get platform-architecture suffix for binary naming."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    # Normalize architecture names
    if machine in ("x86_64", "amd64"):
        arch = "x86_64"
    elif machine in ("arm64", "aarch64"):
        arch = "arm64"
    else:
        arch = machine

    # Normalize OS names
    if system == "darwin":
        os_name = "macos"
    elif system == "windows":
        os_name = "windows"
    else:
        os_name = "linux"

    return f"{os_name}-{arch}"


def get_common_nuitka_args() -> list[str]:
    """Get common Nuitka arguments for all builds."""
    return [
        "--standalone",
        "--onefile",
        "--assume-yes-for-downloads",
        # Python optimizations
        "--python-flag=no_site",
        "--python-flag=no_warnings",
        # Disable dev/build tools
        "--nofollow-import-to=pytest",
        "--nofollow-import-to=setuptools",
        "--nofollow-import-to=pip",
        "--nofollow-import-to=wheel",
        # Heavy optional dependencies (lazy-loaded at runtime if needed)
        "--nofollow-import-to=fitz",  # PyMuPDF - PDF rendering
        "--nofollow-import-to=pymupdf",
        "--nofollow-import-to=frontend",  # PyMuPDF frontend
        # Enable useful plugins
        "--enable-plugin=anti-bloat",
        # Show progress
        "--show-progress",
        "--show-memory",
    ]


def build_server(output_dir: Path, python: str) -> Path:
    """Build the server component."""
    print("\n" + "=" * 60)
    print("Building: llamafarm-server")
    print("=" * 60)

    output_name = f"llamafarm-server-{get_platform_suffix()}"
    output_path = output_dir / output_name

    # Server-specific includes
    server_dir = PROJECT_ROOT / "server"

    args = [
        python,
        "-m",
        "nuitka",
        *get_common_nuitka_args(),
        # Output configuration
        f"--output-dir={output_dir}",
        f"--output-filename={output_name}",
        # Server-specific data (seeds must be at root level, next to main.py)
        f"--include-data-dir={server_dir / 'seeds'}=seeds",
        # Include all server subpackages
        "--include-package=api",
        "--include-package=core",
        "--include-package=agents",
        "--include-package=services",
        # Shared packages (config imports as 'config', common as 'llamafarm_common')
        "--include-package=config",
        "--include-package=llamafarm_common",
        # Celery and related packages (have dynamic imports)
        "--include-package=celery",
        "--include-package=kombu",
        "--include-package=amqp",
        "--include-package=billiard",
        "--include-package=vine",
        # FastAPI/uvicorn plugins
        "--enable-plugin=no-qt",
        # Entry point
        str(server_dir / "main.py"),
    ]

    print(f"Running: {' '.join(args[:10])}...")
    result = subprocess.run(args, cwd=PROJECT_ROOT, check=False)

    if result.returncode != 0:
        print(f"ERROR: Server build failed with code {result.returncode}")
        sys.exit(1)

    print(f"SUCCESS: Built {output_path}")
    return output_path


def build_rag(output_dir: Path, python: str) -> Path:
    """Build the RAG component."""
    print("\n" + "=" * 60)
    print("Building: llamafarm-rag")
    print("=" * 60)

    output_name = f"llamafarm-rag-{get_platform_suffix()}"
    output_path = output_dir / output_name

    rag_dir = PROJECT_ROOT / "rag"

    args = [
        python,
        "-m",
        "nuitka",
        *get_common_nuitka_args(),
        # Output configuration
        f"--output-dir={output_dir}",
        f"--output-filename={output_name}",
        # Include all RAG subpackages
        "--include-package=core",
        "--include-package=components",
        "--include-package=embedders",
        "--include-package=stores",
        "--include-package=utils",
        "--include-package=retrieval",
        # Shared packages
        "--include-package=config",
        "--include-package=llamafarm_common",
        # Celery and related packages (have dynamic imports)
        "--include-package=celery",
        "--include-package=kombu",
        "--include-package=amqp",
        "--include-package=billiard",
        "--include-package=vine",
        # No Qt needed
        "--enable-plugin=no-qt",
        # Entry point
        str(rag_dir / "main.py"),
    ]

    print(f"Running: {' '.join(args[:10])}...")
    result = subprocess.run(args, cwd=PROJECT_ROOT, check=False)

    if result.returncode != 0:
        print(f"ERROR: RAG build failed with code {result.returncode}")
        sys.exit(1)

    print(f"SUCCESS: Built {output_path}")
    return output_path


def build_runtime(output_dir: Path, python: str) -> Path:
    """Build the Universal Runtime component."""
    print("\n" + "=" * 60)
    print("Building: llamafarm-runtime")
    print("=" * 60)

    output_name = f"llamafarm-runtime-{get_platform_suffix()}"
    output_path = output_dir / output_name

    runtime_dir = PROJECT_ROOT / "runtimes" / "universal"

    args = [
        python,
        "-m",
        "nuitka",
        *get_common_nuitka_args(),
        # Output configuration
        f"--output-dir={output_dir}",
        f"--output-filename={output_name}",
        # Include all runtime subpackages
        "--include-package=models",
        "--include-package=utils",
        "--include-package=core",
        "--include-package=routers",
        # Shared packages
        "--include-package=llamafarm_common",
        "--include-package=llamafarm_llama",
        "--include-package=config",
        # PyTorch is optional - will be loaded dynamically if available
        "--nofollow-import-to=torch",
        "--nofollow-import-to=torchvision",
        "--nofollow-import-to=torchaudio",
        # OCR backends are optional
        "--nofollow-import-to=easyocr",
        "--nofollow-import-to=paddleocr",
        "--nofollow-import-to=surya",
        # FastAPI plugin
        "--enable-plugin=no-qt",
        # Entry point
        str(runtime_dir / "server.py"),
    ]

    print(f"Running: {' '.join(args[:10])}...")
    result = subprocess.run(args, cwd=PROJECT_ROOT, check=False)

    if result.returncode != 0:
        print(f"ERROR: Runtime build failed with code {result.returncode}")
        sys.exit(1)

    print(f"SUCCESS: Built {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Build LlamaFarm components with Nuitka")
    parser.add_argument(
        "--component",
        choices=["server", "rag", "runtime"],
        help="Component to build",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Build all components",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DIST_DIR,
        help="Output directory for built binaries",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean output directory before building",
    )

    args = parser.parse_args()

    if not args.component and not args.all:
        parser.error("Either --component or --all is required")

    # Find Python with nuitka installed
    python = find_python_with_nuitka()

    # Setup output directory
    output_dir = args.output_dir
    if args.clean and output_dir.exists():
        print(f"Cleaning {output_dir}...")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build components
    built = []

    if args.all or args.component == "server":
        built.append(build_server(output_dir, python))

    if args.all or args.component == "rag":
        built.append(build_rag(output_dir, python))

    if args.all or args.component == "runtime":
        built.append(build_runtime(output_dir, python))

    # Summary
    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print(f"Platform: {get_platform_suffix()}")
    print(f"Output directory: {output_dir}")
    print("Built binaries:")
    for path in built:
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  - {path.name} ({size_mb:.1f} MB)")
        else:
            print(f"  - {path.name} (not found - check .build directory)")


if __name__ == "__main__":
    main()
