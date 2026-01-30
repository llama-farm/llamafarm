#!/usr/bin/env python3
"""
PyApp Build Script for LlamaFarm Server (POC)

Packages the server component as a standalone binary using PyApp.
Builds a "fat wheel" containing server + config + common packages,
then compiles a PyApp binary that embeds the wheel and a Python distribution.

Usage:
    python tools/pyapp/build.py
    python tools/pyapp/build.py --python-version 3.12
    python tools/pyapp/build.py --no-embed-python

Output:
    dist/pyapp/llamafarm-server-{platform}-{arch}
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DIST_DIR = PROJECT_ROOT / "dist" / "pyapp"
CACHE_DIR = Path(__file__).parent / ".cache"

PYAPP_VERSION = "0.26.0"
PYAPP_SOURCE_URL = (
    f"https://github.com/ofek/pyapp/releases/download/v{PYAPP_VERSION}/source.tar.gz"
)


def get_platform_suffix() -> str:
    """Get platform-architecture suffix for binary naming."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if machine in ("x86_64", "amd64"):
        arch = "x86_64"
    elif machine in ("arm64", "aarch64"):
        arch = "arm64"
    else:
        arch = machine

    if system == "darwin":
        os_name = "macos"
    elif system == "windows":
        os_name = "windows"
    else:
        os_name = "linux"

    return f"{os_name}-{arch}"


def check_prerequisites() -> None:
    """Verify required tools are installed."""
    # Check cargo
    try:
        result = subprocess.run(
            ["cargo", "--version"], capture_output=True, text=True, check=True
        )
        print(f"Found {result.stdout.strip()}")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("ERROR: Rust/Cargo not found. Install from https://rustup.rs/")
        sys.exit(1)

    # Check uv
    try:
        result = subprocess.run(
            ["uv", "--version"], capture_output=True, text=True, check=True
        )
        print(f"Found uv {result.stdout.strip()}")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("ERROR: uv not found. Install from https://docs.astral.sh/uv/")
        sys.exit(1)


def generate_config_types() -> None:
    """Generate config datamodel types from schema."""
    config_dir = PROJECT_ROOT / "config"
    generate_script = config_dir / "generate_types.py"

    if not generate_script.exists():
        print("WARNING: config/generate_types.py not found, skipping type generation")
        return

    print("Generating config types...")
    subprocess.run(
        [sys.executable, str(generate_script)],
        cwd=config_dir,
        check=True,
    )
    print("Config types generated.")


def build_fat_wheel(output_dir: Path) -> Path:
    """Build a single wheel containing server + config + common packages.

    Creates a temporary build directory with symlinks to source packages
    and a generated pyproject.toml, then builds a wheel using uv.

    Returns the path to the built wheel.
    """
    print("\n" + "=" * 60)
    print("Building fat wheel")
    print("=" * 60)

    build_dir = output_dir / "_build"
    wheel_dir = output_dir / "_wheels"

    # Clean previous builds
    for d in (build_dir, wheel_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    # Copy the pyproject.toml template
    template = Path(__file__).parent / "pyproject.server.toml"
    shutil.copy2(template, build_dir / "pyproject.toml")

    # Create the server package with all subpackages nested inside it.
    #
    # The codebase mixes two import styles:
    #   - Bare:     from api.main import llama_farm_api    (in main.py)
    #   - Prefixed: from server.services.xxx import yyy    (in routers)
    #
    # To support both, we nest api/core/agents/services INSIDE the server
    # package (so server.services.xxx resolves), and use a __main__.py shim
    # that adds the server package directory to sys.path (so bare api.xxx
    # also resolves).
    server_pkg = build_dir / "server"
    server_pkg.mkdir()
    (server_pkg / "__init__.py").touch()
    shutil.copy2(PROJECT_ROOT / "server" / "main.py", server_pkg / "main.py")
    shutil.copytree(
        PROJECT_ROOT / "server" / "seeds",
        server_pkg / "seeds",
    )

    # Create __main__.py shim so `python -m server` works with both
    # bare and prefixed imports
    main_shim = server_pkg / "__main__.py"
    main_shim.write_text(
        '"""PyApp entry point for llamafarm-server.\n'
        "\n"
        "Adds the server package directory to sys.path so that bare imports\n"
        "(from api.main import ...) work alongside prefixed imports\n"
        "(from server.services.xxx import ...).\n"
        '"""\n'
        "import os\n"
        "import sys\n"
        "\n"
        "# Allow bare imports (from api.xxx, from core.xxx, etc.)\n"
        "sys.path.insert(0, os.path.dirname(__file__))\n"
        "\n"
        "# Signal PyApp mode for runtime detection\n"
        'os.environ["LLAMAFARM_PYAPP"] = "1"\n'
        "\n"
        "# Import main module — this executes module-level setup\n"
        "# (logging, PID file, seed copying, FastAPI app creation)\n"
        "from server import main  # noqa: F401\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    import uvicorn\n"
        "    from server.core.settings import settings\n"
        "\n"
        "    uvicorn.run(\n"
        '        main.app,\n'
        "        host=settings.HOST,\n"
        "        port=settings.PORT,\n"
        "        reload=False,\n"
        "        log_config=None,\n"
        "        access_log=False,\n"
        "    )\n"
    )

    # Symlink server subpackages INSIDE the server package
    for pkg in ("api", "core", "agents", "services", "context_providers", "tools"):
        src = PROJECT_ROOT / "server" / pkg
        if src.exists():
            os.symlink(src, server_pkg / pkg)
        else:
            print(f"WARNING: server/{pkg} not found, skipping")

    # Symlink config package (the config/ repo dir IS the package)
    os.symlink(PROJECT_ROOT / "config", build_dir / "config")

    # Ensure config/helpers/ has __init__.py for hatchling discovery.
    # The repo uses setuptools with explicit package listing, but hatchling
    # needs __init__.py to recognize subpackages.
    helpers_init = PROJECT_ROOT / "config" / "helpers" / "__init__.py"
    if not helpers_init.exists():
        print("Creating config/helpers/__init__.py for package discovery")
        helpers_init.touch()
        # Track that we created this so we can note it
        print("NOTE: Created config/helpers/__init__.py in the source tree.")
        print("      Consider committing this file.")

    # Symlink common package
    os.symlink(
        PROJECT_ROOT / "common" / "llamafarm_common",
        build_dir / "llamafarm_common",
    )

    # Symlink observability package (shared monorepo package)
    os.symlink(PROJECT_ROOT / "observability", build_dir / "observability")

    # Build the wheel
    print(f"Building wheel in {build_dir}...")
    result = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheel_dir)],
        cwd=build_dir,
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("ERROR: Wheel build failed")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)

    print(result.stdout)

    # Find the built wheel
    wheels = list(wheel_dir.glob("*.whl"))
    if not wheels:
        print("ERROR: No wheel file found after build")
        sys.exit(1)

    wheel_path = wheels[0]
    size_mb = wheel_path.stat().st_size / (1024 * 1024)
    print(f"Built wheel: {wheel_path.name} ({size_mb:.1f} MB)")
    return wheel_path


def download_pyapp_source(cache_dir: Path) -> Path:
    """Download and extract the PyApp source release.

    Returns the path to the extracted source directory.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / f"pyapp-v{PYAPP_VERSION}-source.tar.gz"
    source_dir = cache_dir / f"pyapp-v{PYAPP_VERSION}"

    if source_dir.exists():
        print(f"Using cached PyApp source: {source_dir}")
        return source_dir

    if not archive_path.exists():
        print(f"Downloading PyApp v{PYAPP_VERSION} source...")
        urlretrieve(PYAPP_SOURCE_URL, archive_path)
        print("Download complete.")

    print("Extracting PyApp source...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=cache_dir)

    # PyApp extracts to a directory named "pyapp-vX.Y.Z"
    # Check common extraction patterns
    for candidate in (
        cache_dir / f"pyapp-v{PYAPP_VERSION}",
        cache_dir / f"pyapp-{PYAPP_VERSION}",
        cache_dir / "pyapp",
    ):
        if candidate.exists() and (candidate / "Cargo.toml").exists():
            if candidate != source_dir:
                candidate.rename(source_dir)
            print(f"PyApp source ready: {source_dir}")
            return source_dir

    # If none matched, look for any directory with Cargo.toml
    for item in cache_dir.iterdir():
        if item.is_dir() and (item / "Cargo.toml").exists():
            item.rename(source_dir)
            print(f"PyApp source ready: {source_dir}")
            return source_dir

    print("ERROR: Could not find PyApp source after extraction")
    print(f"Contents of {cache_dir}:")
    for item in cache_dir.iterdir():
        print(f"  {item.name}")
    sys.exit(1)


def build_pyapp_binary(
    source_dir: Path,
    wheel_path: Path,
    output_dir: Path,
    python_version: str = "3.12",
    embed_python: bool = True,
) -> Path:
    """Build the PyApp binary with the embedded wheel.

    Returns the path to the built binary.
    """
    print("\n" + "=" * 60)
    print("Building PyApp binary")
    print("=" * 60)

    output_name = f"llamafarm-server-{get_platform_suffix()}"
    if platform.system() == "Windows":
        output_name += ".exe"

    # PyApp configuration via environment variables
    env = os.environ.copy()
    env.update(
        {
            # Project: embed the fat wheel
            "PYAPP_PROJECT_PATH": str(wheel_path),
            # Entry point: run server/__main__.py (shim that handles imports)
            "PYAPP_EXEC_MODULE": "server",
            # Python version
            "PYAPP_PYTHON_VERSION": python_version,
            # Embed Python distribution in the binary (no download on first run)
            "PYAPP_DISTRIBUTION_EMBED": "true" if embed_python else "",
            # Use uv for faster dependency installation
            "PYAPP_UV_ENABLED": "true",
            # Expose management commands for add-on support
            "PYAPP_EXPOSE_PIP": "true",
            "PYAPP_EXPOSE_PYTHON": "true",
            # Allow pip to read env vars and config at runtime
            # (needed for custom package indexes for private add-ons)
            "PYAPP_PIP_ALLOW_CONFIG": "true",
            # Management command name
            "PYAPP_SELF_COMMAND": "self",
        }
    )

    # Remove empty values (PyApp treats presence of env var as truthy)
    env = {k: v for k, v in env.items() if v}

    print("PyApp configuration:")
    for key in sorted(env):
        if key.startswith("PYAPP_"):
            val = env[key]
            # Truncate long paths for readability
            if len(val) > 80:
                val = f"...{val[-60:]}"
            print(f"  {key}={val}")

    print("\nRunning cargo build (this may take a while on first run)...")
    result = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=source_dir,
        env=env,
        check=False,
    )

    if result.returncode != 0:
        print(f"ERROR: cargo build failed with code {result.returncode}")
        sys.exit(1)

    # Find the built binary
    target_dir = source_dir / "target" / "release"
    candidates = [
        target_dir / "pyapp",
        target_dir / "pyapp.exe",
    ]
    binary_path = None
    for candidate in candidates:
        if candidate.exists():
            binary_path = candidate
            break

    if binary_path is None:
        print("ERROR: Built binary not found")
        print(f"Contents of {target_dir}:")
        for item in sorted(target_dir.iterdir()):
            if item.is_file() and not item.name.startswith("."):
                print(f"  {item.name} ({item.stat().st_size / 1024 / 1024:.1f} MB)")
        sys.exit(1)

    # Copy to output directory with platform-specific name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    shutil.copy2(binary_path, output_path)

    # Make executable on Unix
    if platform.system() != "Windows":
        output_path.chmod(0o755)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSUCCESS: Built {output_path} ({size_mb:.1f} MB)")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build LlamaFarm server with PyApp (POC)"
    )
    parser.add_argument(
        "--python-version",
        default="3.12",
        help="Python version for the embedded distribution (default: 3.12)",
    )
    parser.add_argument(
        "--no-embed-python",
        action="store_true",
        help="Don't embed Python in the binary (download on first run instead)",
    )
    parser.add_argument(
        "--skip-types",
        action="store_true",
        help="Skip config type generation (if already generated)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DIST_DIR,
        help="Output directory for the built binary (default: dist/pyapp/)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean all build artifacts before building",
    )

    args = parser.parse_args()

    print("LlamaFarm Server — PyApp Build (POC)")
    print("=" * 60)
    print(f"Platform: {get_platform_suffix()}")
    print(f"Python version: {args.python_version}")
    print(f"Embed Python: {not args.no_embed_python}")
    print(f"Output: {args.output_dir}")
    print()

    # Step 0: Clean if requested
    if args.clean:
        for d in (args.output_dir, CACHE_DIR):
            if d.exists():
                print(f"Cleaning {d}...")
                shutil.rmtree(d)

    # Step 1: Check prerequisites
    check_prerequisites()

    # Step 2: Generate config types
    if not args.skip_types:
        generate_config_types()

    # Step 3: Build the fat wheel
    wheel_path = build_fat_wheel(args.output_dir)

    # Step 4: Download PyApp source
    source_dir = download_pyapp_source(CACHE_DIR)

    # Step 5: Build PyApp binary
    binary_path = build_pyapp_binary(
        source_dir=source_dir,
        wheel_path=wheel_path,
        output_dir=args.output_dir,
        python_version=args.python_version,
        embed_python=not args.no_embed_python,
    )

    # Summary
    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print(f"Binary: {binary_path}")
    print(f"Size: {binary_path.stat().st_size / 1024 / 1024:.1f} MB")
    print()
    print("Quick start:")
    print(f"  {binary_path}")
    print()
    print("Management commands:")
    print(f"  {binary_path.name} self pip install <addon-package>")
    print(f"  {binary_path.name} self pip list")
    print(f"  {binary_path.name} self python -c 'import fastapi; print(fastapi.__version__)'")
    print(f"  {binary_path.name} self restore  # reinstall from scratch")


if __name__ == "__main__":
    main()
