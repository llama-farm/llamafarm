#!/usr/bin/env python3
"""
PyApp Build Script for LlamaFarm Components.

Packages LlamaFarm components (server, rag, runtime) as standalone binaries
using PyApp. Builds a "fat wheel" containing the component + internal packages,
then compiles a PyApp binary that embeds the wheel and a Python distribution.

Usage:
    python tools/pyapp/build.py                         # Build server (default)
    python tools/pyapp/build.py --component rag
    python tools/pyapp/build.py --component runtime
    python tools/pyapp/build.py --component all         # Build all components
    python tools/pyapp/build.py --version 0.0.26        # Explicit release version
    python tools/pyapp/build.py --python-version 3.12
    python tools/pyapp/build.py --no-embed-python

Version:
    --version sets the wheel version embedded in the binary. If omitted,
    the version is derived from git: exact tags produce release versions
    (v0.0.26 -> 0.0.26), otherwise a dev version (0.0.0.dev0+g<sha>).

Output:
    dist/pyapp/llamafarm-{component}-{platform}-{arch}
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path
from urllib.request import urlretrieve

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DIST_DIR = PROJECT_ROOT / "dist" / "pyapp"
CACHE_DIR = Path(__file__).parent / ".cache"

PYAPP_VERSION = "0.26.0"
PYAPP_SOURCE_URL = (
    f"https://github.com/ofek/pyapp/releases/download/v{PYAPP_VERSION}/source.tar.gz"
)


# ============================================================================
# Component configurations
# ============================================================================
#
# Each component defines how to assemble its fat wheel:
#   source_dir:       component source relative to PROJECT_ROOT
#   pkg_name:         the wrapper package name in the wheel
#   binary_prefix:    output binary name prefix (suffixed with platform-arch)
#   exec_module:      PYAPP_EXEC_MODULE value
#   modules:          .py files to copy into the wrapper package
#   data_dirs:        directories to copy into the wrapper package
#   inner_packages:   subpackages to symlink INTO the wrapper package
#   root_symlinks:    packages to symlink at the build root level
#                     (list of (name, path-relative-to-PROJECT_ROOT) tuples)
#   needs_config:     whether this component bundles the config package
#   pip_extra_args:   extra pip args for PyApp (e.g. PyTorch CPU index)
#   python_version:   minimum Python version for this component
#
# Fat wheel pyproject.toml generation (deps read from source pyproject files):
#   wheel_name:       [project].name in the generated pyproject.toml
#   requires_python:  [project].requires-python
#   wheel_packages:   [tool.hatch.build.targets.wheel].packages
#   force_include:    [tool.hatch.build.targets.wheel.force-include]
#   bundled_packages: internal packages whose deps are merged in
#                     (dict mapping PyPI name -> dir relative to PROJECT_ROOT)
#   exclude_deps:     dep names to filter out (internal refs, dev-only pkgs)
#   extra_deps:       additional deps only needed in PyApp builds

COMPONENTS: dict[str, dict] = {
    "server": {
        "source_dir": "server",
        "pkg_name": "server",
        "binary_prefix": "llamafarm-server",
        "exec_module": "server",
        "modules": ["main.py"],
        "data_dirs": ["seeds"],
        "inner_packages": [
            "api", "core", "agents", "services", "context_providers", "tools",
        ],
        "root_symlinks": [
            ("config", "config"),
            ("llamafarm_common", "common/llamafarm_common"),
            ("observability", "observability"),
        ],
        "needs_config": True,
        "pip_extra_args": "",
        "python_version": "3.12",
        # Fat wheel configuration (previously in pyproject.server.toml)
        "wheel_name": "llamafarm-server",
        "requires_python": ">=3.12",
        "wheel_packages": ["server", "config", "llamafarm_common", "observability"],
        "force_include": {
            "server/seeds": "server/seeds",
            "config/templates": "config/templates",
            "config/schema.yaml": "config/schema.yaml",
            "config/schema.deref.yaml": "config/schema.deref.yaml",
        },
        "bundled_packages": {
            "llamafarm-config": "config",
            "llamafarm-common": "common",
        },
        "exclude_deps": [
            "llamafarm-config", "llamafarm-common",
            "dotenv", "pre-commit", "ruff",
        ],
        "extra_deps": [],
    },
    "rag": {
        "source_dir": "rag",
        "pkg_name": "rag",
        "binary_prefix": "llamafarm-rag",
        "exec_module": "rag",
        "modules": ["main.py", "celery_app.py", "api.py"],
        "data_dirs": [],
        "inner_packages": [
            "core", "components", "tasks", "utils", "cli",
        ],
        "root_symlinks": [
            ("config", "config"),
            ("llamafarm_common", "common/llamafarm_common"),
            ("observability", "observability"),
        ],
        "needs_config": True,
        "pip_extra_args": "",
        "python_version": "3.12",
        # Fat wheel configuration (previously in pyproject.rag.toml)
        "wheel_name": "llamafarm-rag",
        "requires_python": ">=3.11",
        "wheel_packages": ["rag", "config", "llamafarm_common", "observability"],
        "force_include": {
            "config/templates": "config/templates",
            "config/schema.yaml": "config/schema.yaml",
            "config/schema.deref.yaml": "config/schema.deref.yaml",
        },
        "bundled_packages": {
            "llamafarm-config": "config",
            "llamafarm-common": "common",
        },
        "exclude_deps": [
            "llamafarm-config", "llamafarm-common",
            "pytest", "pytest-cov", "pytest-asyncio", "pytest-mock",
        ],
        "extra_deps": [],
    },
    "runtime": {
        "source_dir": "runtimes/universal",
        "pkg_name": "runtime",
        "binary_prefix": "llamafarm-runtime",
        "exec_module": "runtime",
        "modules": ["server.py", "download_model.py", "state.py"],
        "data_dirs": ["config"],
        "inner_packages": [
            "models", "utils", "core", "routers", "api_types", "services",
        ],
        "root_symlinks": [
            ("llamafarm_common", "common/llamafarm_common"),
            ("llamafarm_llama", "packages/llamafarm-llama/src/llamafarm_llama"),
        ],
        "needs_config": False,
        "pip_extra_args": "",
        "python_version": "3.12",
        # Fat wheel configuration (previously in pyproject.runtime.toml)
        "wheel_name": "llamafarm-runtime",
        "requires_python": ">=3.10",
        "wheel_packages": ["runtime", "llamafarm_common", "llamafarm_llama"],
        "force_include": {
            "runtime/config": "runtime/config",
        },
        "bundled_packages": {
            "llamafarm-common": "common",
            "llamafarm-llama": "packages/llamafarm-llama",
        },
        "exclude_deps": ["llamafarm-common", "llamafarm-llama"],
        "extra_deps": [
            "pywin32>=306; sys_platform == 'win32'",
        ],
    },
}


# ============================================================================
# __main__.py shim templates
# ============================================================================
#
# Each component needs a __main__.py that:
#   1. Adds the wrapper package dir to sys.path (for bare imports)
#   2. Sets LLAMAFARM_PYAPP=1 for runtime detection
#   3. Imports the component's main module (triggering module-level setup)
#   4. Starts the appropriate server/worker

MAIN_SHIMS: dict[str, str] = {
    "server": '''\
"""PyApp entry point for llamafarm-server.

Adds the server package directory to sys.path so that bare imports
(from api.main import ...) work alongside prefixed imports
(from server.services.xxx import ...).
"""
import os
import sys

# Allow bare imports (from api.xxx, from core.xxx, etc.)
sys.path.insert(0, os.path.dirname(__file__))

# Signal PyApp mode for runtime detection
os.environ["LLAMAFARM_PYAPP"] = "1"

# Import main module — this executes module-level setup
# (logging, PID file, seed copying, FastAPI app creation)
from server import main  # noqa: F401

if __name__ == '__main__':
    import uvicorn
    from server.core.settings import settings

    uvicorn.run(
        main.app,
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        log_config=None,
        access_log=False,
    )
''',
    "rag": '''\
"""PyApp entry point for llamafarm-rag.

Adds the rag package directory to sys.path so that bare imports
(from core.xxx, from celery_app import ...) work alongside prefixed
imports (from rag.core.xxx import ...).
"""
import os
import sys

# Allow bare imports (from core.xxx, from celery_app, etc.)
sys.path.insert(0, os.path.dirname(__file__))

# Signal PyApp mode for runtime detection
os.environ["LLAMAFARM_PYAPP"] = "1"

# Import main module — this executes module-level setup
# (logging, PID file, Celery worker configuration)
from rag import main  # noqa: F401

if __name__ == '__main__':
    main.main()
''',
    "runtime": '''\
"""PyApp entry point for llamafarm-runtime.

Adds the runtime package directory to sys.path so that bare imports
(from models import ..., from state import ...) work alongside prefixed
imports (from runtime.models import ...).
"""
import os
import sys

# Allow bare imports (from models.xxx, from routers.xxx, etc.)
sys.path.insert(0, os.path.dirname(__file__))

# Signal PyApp mode for runtime detection
os.environ["LLAMAFARM_PYAPP"] = "1"

# Import server module — this executes module-level setup
# (logging, device detection, model loaders, FastAPI app creation)
from runtime import server  # noqa: F401

if __name__ == '__main__':
    import uvicorn
    from llamafarm_common.pidfile import write_pid

    write_pid("universal-runtime")

    port = int(os.getenv("LF_RUNTIME_PORT", os.getenv("PORT", "11540")))
    host = os.getenv("LF_RUNTIME_HOST", os.getenv("HOST", "127.0.0.1"))

    uvicorn.run(
        server.app,
        host=host,
        port=port,
        log_config=None,
        access_log=False,
        ws_ping_interval=30.0,
        ws_ping_timeout=60.0,
    )
''',
}


# ============================================================================
# Shared utilities
# ============================================================================


def get_version_from_git() -> str:
    """Derive a PEP 440 version from git.

    On a release tag (v0.0.26): returns "0.0.26".
    Otherwise: returns "0.0.0.dev0+g<short-sha>" for dev builds.
    """
    # Try exact tag first (release builds)
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--exact-match"],
            capture_output=True,
            text=True,
            check=True,
            cwd=PROJECT_ROOT,
        )
        tag = result.stdout.strip()
        return tag.lstrip("v")
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Fall back to short SHA (dev builds)
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=PROJECT_ROOT,
        )
        sha = result.stdout.strip()
        return f"0.0.0.dev0+g{sha}"
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "0.0.0.dev0"


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


# ============================================================================
# Dependency collection (DRY — reads from source pyproject.toml files)
# ============================================================================

# Regex to extract the package name from a PEP 508 dependency string.
# Matches the leading name portion before any extras, version, or markers.
_DEP_NAME_RE = re.compile(r"^([A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?)")


def _normalize_dep_name(dep_str: str) -> str:
    """Normalize a dependency string to a comparable package name.

    Handles extras (pkg[extra]), version specifiers (>=1.0), environment
    markers (; sys_platform == ...), and URL references (pkg @ url).
    """
    m = _DEP_NAME_RE.match(dep_str.strip())
    name = m.group(1) if m else dep_str.strip()
    return re.sub(r"[-_.]+", "_", name).lower()


def collect_pyapp_dependencies(component: str) -> list[str]:
    """Collect production dependencies for a PyApp fat wheel.

    Reads the component's main pyproject.toml and the pyproject.toml of each
    bundled internal package, filters out dev-only and internal-package
    references, and returns a deduplicated list.
    """
    cfg = COMPONENTS[component]
    exclude = {_normalize_dep_name(d) for d in cfg.get("exclude_deps", [])}

    deps: list[str] = []

    # 1. Read component's own production dependencies
    main_pyproject = PROJECT_ROOT / cfg["source_dir"] / "pyproject.toml"
    with open(main_pyproject, "rb") as f:
        data = tomllib.load(f)
    for dep in data.get("project", {}).get("dependencies", []):
        if _normalize_dep_name(dep) not in exclude:
            deps.append(dep)

    # 2. Read bundled internal package dependencies
    for _pkg_ref, pkg_dir in cfg.get("bundled_packages", {}).items():
        pkg_pyproject = PROJECT_ROOT / pkg_dir / "pyproject.toml"
        if not pkg_pyproject.exists():
            continue
        with open(pkg_pyproject, "rb") as f:
            pkg_data = tomllib.load(f)
        for dep in pkg_data.get("project", {}).get("dependencies", []):
            if _normalize_dep_name(dep) not in exclude:
                deps.append(dep)

    # 3. Add PyApp-specific extras (e.g. pywin32 for Windows)
    deps.extend(cfg.get("extra_deps", []))

    # Deduplicate by normalized name, keeping first occurrence
    seen: set[str] = set()
    unique: list[str] = []
    for dep in deps:
        name = _normalize_dep_name(dep)
        if name not in seen:
            seen.add(name)
            unique.append(dep)

    return unique


def generate_pyproject_toml(component: str, version: str) -> str:
    """Generate a pyproject.toml for a PyApp fat wheel build.

    Collects dependencies from source pyproject.toml files and produces
    a self-contained pyproject.toml that hatchling can build.
    """
    cfg = COMPONENTS[component]
    deps = collect_pyapp_dependencies(component)

    deps_str = "\n".join(f"    {dep!r}," for dep in deps)
    pkgs_str = "\n".join(f"    {pkg!r}," for pkg in cfg["wheel_packages"])

    lines = [
        "# Auto-generated by tools/pyapp/build.py — do not edit.",
        "# Dependencies are collected from source pyproject.toml files at build time.",
        "",
        "[build-system]",
        'requires = ["hatchling"]',
        'build-backend = "hatchling.build"',
        "",
        "[project]",
        f'name = {cfg["wheel_name"]!r}',
        f'version = "{version}"',
        f'requires-python = {cfg["requires_python"]!r}',
        "dependencies = [",
        deps_str,
        "]",
        "",
        "[tool.hatch.build.targets.wheel]",
        "packages = [",
        pkgs_str,
        "]",
    ]

    force_include = cfg.get("force_include", {})
    if force_include:
        lines.append("")
        lines.append("[tool.hatch.build.targets.wheel.force-include]")
        for src, dest in force_include.items():
            lines.append(f'{src!r} = {dest!r}')

    lines.append("")  # trailing newline
    return "\n".join(lines)


# ============================================================================
# Fat wheel builder
# ============================================================================


def build_fat_wheel(component: str, output_dir: Path, version: str) -> Path:
    """Build a single wheel containing a component + its internal packages.

    Creates a temporary build directory with symlinks to source packages
    and a pyproject.toml template, then builds a wheel using uv.

    Returns the path to the built wheel.
    """
    cfg = COMPONENTS[component]
    source_dir = PROJECT_ROOT / cfg["source_dir"]

    print("\n" + "=" * 60)
    print(f"Building fat wheel for {component}")
    print("=" * 60)

    build_dir = output_dir / f"_build_{component}"
    wheel_dir = output_dir / f"_wheels_{component}"

    # Clean previous builds
    for d in (build_dir, wheel_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    # Generate the pyproject.toml from source dependency files
    pyproject_path = build_dir / "pyproject.toml"
    pyproject_path.write_text(generate_pyproject_toml(component, version))
    print(f"Wheel version: {version}")

    # Create the wrapper package with all subpackages nested inside it.
    #
    # LlamaFarm components use bare imports (from core.xxx, from models.xxx).
    # We nest subpackages INSIDE the wrapper package, then the __main__.py
    # shim adds the wrapper dir to sys.path so bare imports resolve.
    pkg_name = cfg["pkg_name"]
    pkg_dir = build_dir / pkg_name
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").touch()

    # Copy top-level modules into the wrapper package
    for module in cfg["modules"]:
        src = source_dir / module
        if src.exists():
            shutil.copy2(src, pkg_dir / module)
        else:
            print(f"WARNING: {cfg['source_dir']}/{module} not found, skipping")

    # Copy data directories into the wrapper package
    for data_dir in cfg["data_dirs"]:
        src = source_dir / data_dir
        if src.exists():
            shutil.copytree(src, pkg_dir / data_dir)
        else:
            print(f"WARNING: {cfg['source_dir']}/{data_dir} not found, skipping")

    # Write the __main__.py shim
    shim_content = MAIN_SHIMS[component]
    (pkg_dir / "__main__.py").write_text(shim_content)

    # Symlink subpackages INSIDE the wrapper package
    for pkg in cfg["inner_packages"]:
        src = source_dir / pkg
        if src.exists():
            os.symlink(src, pkg_dir / pkg)
        else:
            print(f"WARNING: {cfg['source_dir']}/{pkg} not found, skipping")

    # Symlink external packages at the build root level
    for name, rel_path in cfg["root_symlinks"]:
        src = PROJECT_ROOT / rel_path
        dest = build_dir / name
        if src.exists():
            os.symlink(src, dest)
        else:
            print(f"WARNING: {rel_path} not found, skipping")

    # If this component bundles the config package, ensure config/helpers/
    # has __init__.py for hatchling discovery (setuptools config doesn't need
    # it but hatchling does).
    if cfg["needs_config"]:
        helpers_init = PROJECT_ROOT / "config" / "helpers" / "__init__.py"
        if not helpers_init.exists():
            print("Creating config/helpers/__init__.py for package discovery")
            helpers_init.touch()
            print("NOTE: Created config/helpers/__init__.py in the source tree.")
            print("      Consider committing this file.")

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


# ============================================================================
# PyApp source management
# ============================================================================


def download_pyapp_source(cache_dir: Path) -> Path:
    """Download and extract the PyApp source release.

    Returns the path to the extracted source directory.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / f"pyapp-v{PYAPP_VERSION}-source.tar.gz"
    source_dir = cache_dir / f"pyapp-v{PYAPP_VERSION}"

    if source_dir.exists() and (source_dir / "Cargo.toml").exists():
        print(f"Using cached PyApp source: {source_dir}")
        return source_dir

    # Clean up incomplete cache (e.g. CI restoring only target/ subdirectory)
    if source_dir.exists():
        shutil.rmtree(source_dir)

    if not archive_path.exists():
        print(f"Downloading PyApp v{PYAPP_VERSION} source...")
        urlretrieve(PYAPP_SOURCE_URL, archive_path)
        print("Download complete.")

    print("Extracting PyApp source...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=cache_dir, filter="data")

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


# ============================================================================
# PyApp binary builder
# ============================================================================


def build_pyapp_binary(
    component: str,
    source_dir: Path,
    wheel_path: Path,
    output_dir: Path,
    python_version: str = "3.12",
    embed_python: bool = True,
) -> Path:
    """Build the PyApp binary with the embedded wheel.

    Returns the path to the built binary.
    """
    cfg = COMPONENTS[component]

    print("\n" + "=" * 60)
    print(f"Building PyApp binary for {component}")
    print("=" * 60)

    output_name = f"{cfg['binary_prefix']}-{get_platform_suffix()}"
    if platform.system() == "Windows":
        output_name += ".exe"

    # PyApp configuration via environment variables
    env = os.environ.copy()
    env.update(
        {
            # Project: embed the fat wheel
            "PYAPP_PROJECT_PATH": str(wheel_path),
            # Entry point: run {pkg}/__main__.py (shim that handles imports)
            "PYAPP_EXEC_MODULE": cfg["exec_module"],
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
            # Extra pip/uv args for dependency installation
            "PYAPP_PIP_EXTRA_ARGS": cfg["pip_extra_args"],
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


# ============================================================================
# Build orchestration
# ============================================================================


def build_component(
    component: str,
    output_dir: Path,
    version: str,
    python_version: str,
    embed_python: bool,
    skip_types: bool,
) -> Path:
    """Build a single component end-to-end.

    Returns the path to the built binary.
    """
    cfg = COMPONENTS[component]

    print(f"\n{'#' * 60}")
    print(f"# Building: {component}")
    print(f"{'#' * 60}")

    # Generate config types if this component bundles config
    if cfg["needs_config"] and not skip_types:
        generate_config_types()

    # Build the fat wheel
    wheel_path = build_fat_wheel(component, output_dir, version)

    # Download PyApp source (cached across components)
    source_dir = download_pyapp_source(CACHE_DIR)

    # Build PyApp binary
    binary_path = build_pyapp_binary(
        component=component,
        source_dir=source_dir,
        wheel_path=wheel_path,
        output_dir=output_dir,
        python_version=python_version,
        embed_python=embed_python,
    )

    return binary_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build LlamaFarm components with PyApp"
    )
    parser.add_argument(
        "--component",
        choices=[*COMPONENTS.keys(), "all"],
        default="server",
        help="Component to build (default: server)",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Version for the built wheel (default: auto-detect from git)",
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

    # Determine which components to build
    if args.component == "all":
        components = list(COMPONENTS.keys())
    else:
        components = [args.component]

    # Resolve version
    version = args.version or get_version_from_git()

    print("LlamaFarm — PyApp Build")
    print("=" * 60)
    print(f"Component(s): {', '.join(components)}")
    print(f"Version: {version}")
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

    # Step 2: Build each component
    results: list[tuple[str, Path]] = []
    for component in components:
        binary_path = build_component(
            component=component,
            output_dir=args.output_dir,
            version=version,
            python_version=args.python_version,
            embed_python=not args.no_embed_python,
            skip_types=args.skip_types,
        )
        results.append((component, binary_path))
        # Skip type generation for subsequent components (only needed once)
        args.skip_types = True

    # Summary
    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    for component, binary_path in results:
        size_mb = binary_path.stat().st_size / (1024 * 1024)
        print(f"  {component}: {binary_path} ({size_mb:.1f} MB)")
    print()
    print("Quick start:")
    for _, binary_path in results:
        print(f"  {binary_path}")
    print()
    print("Management commands:")
    binary_name = results[0][1].name
    print(f"  {binary_name} self pip install <addon-package>")
    print(f"  {binary_name} self pip list")
    print(f"  {binary_name} self python -c 'import sys; print(sys.version)'")
    print(f"  {binary_name} self restore  # reinstall from scratch")


if __name__ == "__main__":
    main()
