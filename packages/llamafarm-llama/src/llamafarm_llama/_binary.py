"""
Download and manage pre-built llama.cpp binaries from GitHub releases.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import shlex
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from importlib import metadata

logger = logging.getLogger(__name__)

# Pin to specific llama.cpp release
LLAMA_CPP_VERSION = "b7376"
LLAMA_CPP_REPO = "ggml-org/llama.cpp"

# Binary URLs from llama.cpp GitHub releases
# Format: https://github.com/ggml-org/llama.cpp/releases/download/{version}/{artifact}
# Note: All releases are .zip format; paths vary by platform
BINARY_MANIFEST: dict[tuple[str, str, str], dict] = {
    # Linux x86_64
    ("linux", "x86_64", "cpu"): {
        "artifact": "llama-{version}-bin-ubuntu-x64.zip",
        "lib": "build/bin/libllama.so",
        "sha256": None,  # Populated at release time
    },
    ("linux", "x86_64", "vulkan"): {
        "artifact": "llama-{version}-bin-ubuntu-vulkan-x64.zip",
        "lib": "build/bin/libllama.so",
        "sha256": None,
    },
    # Linux ARM64 (LlamaFarm provided)
    ("linux", "arm64", "cpu"): {
        "artifact": "https://github.com/llama-farm/llamafarm/releases/download/{version}/llama-b7376-bin-linux-arm64.zip",
        "lib": "bin/libllama.so",
        "sha256": None,
    },
    # macOS
    ("darwin", "arm64", "metal"): {
        "artifact": "llama-{version}-bin-macos-arm64.zip",
        "lib": "build/bin/libllama.dylib",
        "sha256": None,
    },
    ("darwin", "x86_64", "cpu"): {
        "artifact": "llama-{version}-bin-macos-x64.zip",
        "lib": "build/bin/libllama.dylib",
        "sha256": None,
    },
    # Windows
    ("win32", "amd64", "cpu"): {
        "artifact": "llama-{version}-bin-win-cpu-x64.zip",
        "lib": "llama.dll",  # Windows: library is in root
        "sha256": None,
    },
    ("win32", "amd64", "cuda12"): {
        "artifact": "llama-{version}-bin-win-cuda12.4-x64.zip",
        "lib": "llama.dll",
        "sha256": None,
    },
    ("win32", "amd64", "cuda11"): {
        "artifact": "llama-{version}-bin-win-cuda11.7-x64.zip",
        "lib": "llama.dll",
        "sha256": None,
    },
    ("win32", "amd64", "vulkan"): {
        "artifact": "llama-{version}-bin-win-vulkan-x64.zip",
        "lib": "llama.dll",
        "sha256": None,
    },
}


def _should_build_from_source(platform_key: tuple[str, str, str]) -> bool:
    """Return True when llama.cpp should be built from source."""
    return False



def _build_from_source(dest_dir: Path, version: str, backend: str) -> Path:
    """Build llama.cpp from source and install it into dest_dir."""
    source_url = f"https://github.com/{LLAMA_CPP_REPO}/archive/refs/tags/{version}.zip"

    dest_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        archive_path = tmpdir_path / f"llama.cpp-{version}.zip"

        logger.info(f"Downloading llama.cpp source {version} for source build...")
        req = Request(source_url, headers={"User-Agent": "llamafarm-llama"})
        try:
            with urlopen(req, timeout=300) as response:
                archive_path.write_bytes(response.read())
        except (URLError, HTTPError) as e:
            raise RuntimeError(f"Failed to download source archive {source_url}: {e}") from e

        extract_dir = tmpdir_path / "source"
        extract_dir.mkdir()
        _safe_extract_zip(archive_path, extract_dir)

        source_root = next(
            (p for p in extract_dir.iterdir() if p.is_dir()),
            None,
        )
        if source_root is None:
            raise RuntimeError("Failed to locate llama.cpp source directory after extraction")

        build_dir = source_root / "build"
        cmake_args = [
            "cmake",
            "-S",
            str(source_root),
            "-B",
            str(build_dir),
            "-DBUILD_SHARED_LIBS=ON",
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_EXAMPLES=OFF",
        ]

        if backend.startswith("cuda"):
            cmake_args.append("-DGGML_CUDA=ON")
        elif backend == "vulkan":
            cmake_args.append("-DGGML_VULKAN=ON")

        extra_args = os.environ.get("LLAMAFARM_LLAMA_CMAKE_ARGS")
        if extra_args:
            cmake_args.extend(shlex.split(extra_args))

        logger.info(f"Configuring llama.cpp build (backend={backend})...")
        subprocess.run(cmake_args, check=True)

        build_cmd = ["cmake", "--build", str(build_dir), "--config", "Release"]
        jobs = os.environ.get("LLAMAFARM_LLAMA_BUILD_JOBS")
        if jobs:
            build_cmd.extend(["--parallel", jobs])
        else:
            build_cmd.append("--parallel")

        logger.info("Building llama.cpp from source...")
        subprocess.run(build_cmd, check=True)

        lib_name = _get_lib_name()
        candidates = list(build_dir.rglob(lib_name))
        if not candidates:
            raise RuntimeError(f"Could not find {lib_name} in build output")

        lib_src = next((c for c in candidates if c.parent.name == "bin"), candidates[0])
        lib_dest = dest_dir / lib_name

        _extract_with_symlinks(lib_src, lib_dest)
        _copy_dependencies(build_dir, dest_dir)

        logger.info(f"Installed llama.cpp from source to: {lib_dest}")
        return lib_dest


def get_platform_key(backend_override: Optional[str] = None) -> tuple[str, str, str]:
    """Detect current platform and best available backend."""
    system = platform.system().lower()
    if system == "windows":
        system = "win32"

    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        machine = "x86_64" if system != "win32" else "amd64"
    elif machine == "arm64" or machine == "aarch64":
        machine = "arm64"

    # Allow backend override via environment
    if backend_override is None:
        backend_override = os.environ.get("LLAMAFARM_BACKEND")

    if backend_override:
        return (system, machine, backend_override)

    # Detect best backend
    backend = _detect_backend(system, machine)
    return (system, machine, backend)


def _detect_backend(system: str, machine: str) -> str:
    """Detect the best available GPU backend."""
    if system == "darwin" and machine == "arm64":
        return "metal"  # Apple Silicon always has Metal

    # Check for CUDA
    if _has_cuda():
        cuda_version = _get_cuda_version()
        return "cuda12" if cuda_version >= 12 else "cuda11"

    # Check for Vulkan
    if _has_vulkan():
        return "vulkan"

    return "cpu"


def _has_cuda() -> bool:
    """Check if CUDA is available."""
    import subprocess

    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL, timeout=5)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check environment variables
    if os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME"):
        return True

    # Check common paths
    cuda_paths = ["/usr/local/cuda", "/usr/lib/cuda", "/opt/cuda"]
    for path in cuda_paths:
        if Path(path).exists():
            return True

    return False


def _get_cuda_version() -> int:
    """Get major CUDA version."""
    import subprocess

    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        # Map driver version to CUDA version (approximate)
        driver = float(output.strip().split(".")[0])
        if driver >= 525:
            return 12
        return 11
    except Exception:
        return 11


def _has_vulkan() -> bool:
    """Check if Vulkan is available."""
    if os.environ.get("VULKAN_SDK"):
        return True
    if Path("/usr/share/vulkan").exists():
        return True
    return False


def get_lib_path() -> Path:
    """Get path to libllama, downloading if necessary."""
    # Check for bundled binary (platform wheel)
    bundled = Path(__file__).parent / "lib" / _get_lib_name()
    if bundled.exists():
        logger.debug(f"Using bundled binary: {bundled}")
        return bundled

    # Check for cached download
    cache_dir = _get_cache_dir()
    cached = cache_dir / LLAMA_CPP_VERSION / _get_lib_name()
    if cached.exists():
        logger.debug(f"Using cached binary: {cached}")
        return cached

    # Download
    logger.info(f"Downloading llama.cpp {LLAMA_CPP_VERSION}...")
    return download_binary(cache_dir / LLAMA_CPP_VERSION)


def _get_lib_name() -> str:
    """Get platform-specific library name."""
    system = platform.system().lower()
    if system == "darwin":
        return "libllama.dylib"
    elif system == "windows":
        return "llama.dll"
    else:
        return "libllama.so"


def _safe_extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Safely extract a zip file, preventing Zip Slip path traversal attacks.

    Validates that all extracted paths stay within the destination directory.
    """
    dest_dir = dest_dir.resolve()

    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.namelist():
            # Resolve the target path
            member_path = (dest_dir / member).resolve()

            # Ensure the resolved path is within dest_dir
            try:
                member_path.relative_to(dest_dir)
            except ValueError:
                raise RuntimeError(
                    f"Zip Slip detected: {member!r} would extract outside target directory"
                )

        # All paths validated, safe to extract
        z.extractall(dest_dir)


def _get_cache_dir() -> Path:
    """Get cache directory for downloaded binaries."""
    if os.environ.get("LLAMAFARM_CACHE_DIR"):
        return Path(os.environ["LLAMAFARM_CACHE_DIR"])

    # Platform-specific cache
    system = platform.system().lower()
    if system == "darwin":
        return Path.home() / "Library" / "Caches" / "llamafarm-llama"
    elif system == "windows":
        local_app_data = os.environ.get("LOCALAPPDATA", str(Path.home()))
        return Path(local_app_data) / "llamafarm-llama" / "cache"
    else:
        xdg_cache = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
        return Path(xdg_cache) / "llamafarm-llama"


def _extract_with_symlinks(src_path: Path, dest_path: Path) -> None:
    """Extract a file, following symlink chains and preserving symlinks.

    Handles cases where libllama.dylib -> libllama.0.dylib -> libllama.0.0.7376.dylib
    by following the chain, copying the actual file, and recreating symlinks.
    """
    dest_dir = dest_path.parent

    # Follow symlink chain to find the actual file
    current = src_path
    symlink_chain: list[tuple[Path, str]] = []  # (symlink_path, target_name)

    while current.is_symlink() or (current.exists() and current.stat().st_size < 100):
        if current.is_symlink():
            # It's a real symlink
            target = os.readlink(current)
            logger.debug(f"Following symlink: {current.name} -> {target}")
            symlink_chain.append((current, target))
            # Resolve relative to symlink's directory
            current = (current.parent / target).resolve()
        elif current.exists() and current.stat().st_size < 100:
            # Might be a text file containing symlink target (some extractors do this)
            try:
                target = current.read_text().strip()
                if target and not target.startswith("/") and len(target) < 256:
                    logger.debug(f"Following text symlink: {current.name} -> {target}")
                    symlink_chain.append((current, target))
                    potential_target = current.parent / target
                    if potential_target.exists():
                        current = potential_target
                        continue
            except Exception:
                pass
            # Not a symlink reference, treat as actual file
            break
        else:
            break

    # Verify we found an actual file
    if not current.exists():
        raise RuntimeError(f"Could not resolve symlink chain from {src_path}")

    if current.stat().st_size < 1000:
        raise RuntimeError(
            f"Resolved file {current} is too small ({current.stat().st_size} bytes), "
            "likely not a valid library"
        )

    logger.debug(f"Found actual library: {current} ({current.stat().st_size} bytes)")

    # Copy the actual file first - use the final target name
    if symlink_chain:
        # Copy actual file with its real name
        actual_dest = dest_dir / current.name
        if not actual_dest.exists():
            shutil.copy2(current, actual_dest)
            logger.debug(f"Copied actual file: {current.name}")

        # Recreate symlink chain in reverse order (from innermost to outermost)
        for symlink_src, target in reversed(symlink_chain):
            symlink_dest = dest_dir / symlink_src.name
            if symlink_dest.exists() or symlink_dest.is_symlink():
                symlink_dest.unlink()
            symlink_dest.symlink_to(target)
            logger.debug(f"Created symlink: {symlink_src.name} -> {target}")
    else:
        # No symlinks, just copy the file directly
        shutil.copy2(current, dest_path)
        logger.debug(f"Copied file: {current.name} -> {dest_path.name}")


def download_binary(
    dest_dir: Path, platform_key: Optional[tuple[str, str, str]] = None
) -> Path:
    """Download the appropriate llama.cpp binary."""
    if platform_key is None:
        platform_key = get_platform_key()

    version = os.environ.get("LLAMAFARM_LLAMA_VERSION", LLAMA_CPP_VERSION)

    # For Linux ARM64, we need the LlamaFarm package version to construct the URL
    if platform_key == ("linux", "arm64", "cpu"):
        try:
            package_version = metadata.version("llamafarm-llama")
            # If dev version, fallback or handle appropriately. For now assume v0.0.1 for dev
            if "dev" in package_version or "0.1.0" in package_version: # 0.1.0 is the pyproject default
                 version = "v0.0.1"
            else:
                 version = f"v{package_version}"
        except metadata.PackageNotFoundError:
            version = "v0.0.1"


    if platform_key not in BINARY_MANIFEST:
        if _should_build_from_source(platform_key):
            logger.warning(
                f"No pre-built binary for {platform_key}; building from source instead."
            )
            return _build_from_source(dest_dir, version, platform_key[2])
        # Try falling back to CPU
        system, machine, _ = platform_key
        cpu_key = (system, machine, "cpu")
        if cpu_key in BINARY_MANIFEST:
            logger.warning(f"No binary for {platform_key}, falling back to CPU")
            platform_key = cpu_key
        else:
            raise RuntimeError(f"No pre-built binary available for {platform_key}")

    manifest = BINARY_MANIFEST[platform_key]
    if platform_key == ("linux", "arm64", "cpu"):
         # Use full URL from manifest for our custom builds
         url = manifest["artifact"].format(version=version)
         artifact = url.split("/")[-1]
    else:
         artifact = manifest["artifact"].format(version=version)
         url = f"https://github.com/{LLAMA_CPP_REPO}/releases/download/{version}/{artifact}"

    print(f"Downloading llama.cpp {version} for {platform_key}...")
    print(f"  URL: {url}")

    dest_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        archive_path = tmpdir_path / artifact

        # Download with progress
        try:
            req = Request(url, headers={"User-Agent": "llamafarm-llama"})
            with urlopen(req, timeout=300) as response:
                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0
                chunk_size = 8192

                with open(archive_path, "wb") as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = downloaded * 100 // total_size
                            print(
                                f"\r  Progress: {pct}% ({downloaded}/{total_size} bytes)",
                                end="",
                                flush=True,
                            )
                print()  # Newline after progress
        except (URLError, HTTPError) as e:
            raise RuntimeError(f"Failed to download {url}: {e}") from e

        # Verify checksum if available
        if manifest.get("sha256"):
            actual = hashlib.sha256(archive_path.read_bytes()).hexdigest()
            if actual != manifest["sha256"]:
                raise RuntimeError(
                    f"Checksum mismatch: expected {manifest['sha256']}, got {actual}"
                )

        # Extract (all llama.cpp releases are .zip format)
        extract_dir = tmpdir_path / "extracted"
        extract_dir.mkdir()

        if artifact.endswith(".zip"):
            _safe_extract_zip(archive_path, extract_dir)
        else:
            raise RuntimeError(f"Unknown archive format: {artifact}")

        # Find the library file
        lib_path = manifest["lib"]
        lib_src = extract_dir / lib_path

        # Handle nested directories - search for the library
        if not lib_src.exists():
            # Try to find it
            lib_name = Path(lib_path).name
            candidates = list(extract_dir.rglob(lib_name))
            if candidates:
                lib_src = candidates[0]
                logger.debug(f"Found library at: {lib_src}")
            else:
                raise RuntimeError(
                    f"Could not find {lib_name} in archive. Contents: {list(extract_dir.rglob('*'))}"
                )

        # Handle symlinks: newer llama.cpp releases use symlinks like:
        # libllama.dylib -> libllama.0.dylib -> libllama.0.0.7376.dylib
        # We need to follow the chain and copy the actual file + recreate symlinks
        lib_dest = dest_dir / _get_lib_name()
        _extract_with_symlinks(lib_src, lib_dest)

        # Copy any additional dependencies (Windows DLLs, CUDA libs, etc.)
        _copy_dependencies(extract_dir, dest_dir)

        print(f"  Installed to: {lib_dest}")
        return lib_dest


def _copy_dependencies(src_dir: Path, dest_dir: Path):
    """Copy additional runtime dependencies (ggml libs, CUDA libs, etc.)."""
    lib_name = _get_lib_name()
    system = platform.system().lower()

    # For versioned libraries, copy the actual versioned file, not the symlinks
    # Note: Linux uses libfoo.so.0.0.0 format, macOS uses libfoo.0.0.0.dylib
    patterns = [
        "*.dll",
        "*.metal",  # Metal shader source (required for macOS GPU acceleration)
    ]

    if system == "darwin":
        # macOS: version before extension (libggml.0.0.0.dylib)
        patterns.extend([
            "libggml*.*.*.*dylib",
        ])
    else:
        # Linux: version after extension (libggml.so.0.0.0)
        # Also include unversioned .so files for backend loading
        patterns.extend([
            "libggml*.so.*",      # Versioned: libggml.so.0.0.0
            "libggml*.so",        # Unversioned: libggml.so, libggml-cpu.so
            "ggml-*.so",          # Backend plugins: ggml-cpu.so, ggml-cuda.so
            "libcublas*.so.*",
            "libcudart*.so.*",
            "libcublasLt*.so.*",
        ])

    for pattern in patterns:
        for f in src_dir.rglob(pattern):
            # Skip symlinks (small text files) and the main library
            if f.is_file() and f.name != lib_name and f.stat().st_size > 100:
                dest = dest_dir / f.name
                if not dest.exists():
                    shutil.copy2(f, dest)
                    logger.debug(f"Copied dependency: {f.name}")

    # Create symlinks for versioned dylibs on macOS/Linux
    # e.g., libggml.0.9.4.dylib needs libggml.0.dylib and libggml.dylib symlinks
    if platform.system().lower() != "windows":
        _create_version_symlinks(dest_dir)


def _create_version_symlinks(dest_dir: Path):
    """Create symlinks for versioned libraries.

    Newer llama.cpp releases use versioned libraries:
    - macOS: libggml.0.9.4.dylib with symlinks libggml.0.dylib -> libggml.0.9.4.dylib
    - Linux: libggml.so.0.9.4 with symlinks libggml.so.0 -> libggml.so.0.9.4

    We need to recreate these symlinks.
    """
    import re

    system = platform.system().lower()

    if system == "darwin":
        # macOS: libfoo.MAJOR.MINOR.PATCH.dylib
        for lib_file in dest_dir.glob("*.dylib"):
            match = re.match(r"^(lib[\w-]+)\.(\d+)\.(\d+)\.(\d+)\.dylib$", lib_file.name)
            if match:
                base_name = match.group(1)  # e.g., "libggml" or "libggml-base"
                major = match.group(2)  # e.g., "0"

                # Create libggml.0.dylib -> libggml.0.9.4.dylib
                major_symlink = dest_dir / f"{base_name}.{major}.dylib"
                if not major_symlink.exists():
                    major_symlink.symlink_to(lib_file.name)
                    logger.debug(f"Created symlink: {major_symlink.name} -> {lib_file.name}")

                # Create libggml.dylib -> libggml.0.dylib
                base_symlink = dest_dir / f"{base_name}.dylib"
                if not base_symlink.exists():
                    base_symlink.symlink_to(major_symlink.name)
                    logger.debug(f"Created symlink: {base_symlink.name} -> {major_symlink.name}")
    else:
        # Linux: libfoo.so.MAJOR.MINOR.PATCH
        for lib_file in dest_dir.iterdir():
            if not lib_file.is_file():
                continue
            # Match libggml.so.0.0.0 or libggml-base.so.0.0.0
            match = re.match(r"^(lib[\w-]+)\.so\.(\d+)\.(\d+)\.(\d+)$", lib_file.name)
            if match:
                base_name = match.group(1)  # e.g., "libggml" or "libggml-base"
                major = match.group(2)  # e.g., "0"

                # Create libggml.so.0 -> libggml.so.0.0.0
                major_symlink = dest_dir / f"{base_name}.so.{major}"
                if not major_symlink.exists():
                    major_symlink.symlink_to(lib_file.name)
                    logger.debug(f"Created symlink: {major_symlink.name} -> {lib_file.name}")

                # Create libggml.so -> libggml.so.0
                base_symlink = dest_dir / f"{base_name}.so"
                if not base_symlink.exists():
                    base_symlink.symlink_to(major_symlink.name)
                    logger.debug(f"Created symlink: {base_symlink.name} -> {major_symlink.name}")


def clear_cache():
    """Clear the download cache."""
    cache_dir = _get_cache_dir()
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"Cleared cache: {cache_dir}")


def get_binary_info() -> dict:
    """Get information about the current binary configuration."""
    platform_key = get_platform_key()
    lib_path = None

    # Check bundled
    bundled = Path(__file__).parent / "lib" / _get_lib_name()
    if bundled.exists():
        lib_path = bundled
        source = "bundled"
    else:
        # Check cached
        cache_dir = _get_cache_dir()
        cached = cache_dir / LLAMA_CPP_VERSION / _get_lib_name()
        if cached.exists():
            lib_path = cached
            source = "cached"
        else:
            source = "not_downloaded"

    return {
        "version": LLAMA_CPP_VERSION,
        "platform_key": platform_key,
        "lib_path": str(lib_path) if lib_path else None,
        "lib_name": _get_lib_name(),
        "source": source,
        "cache_dir": str(_get_cache_dir()),
    }
