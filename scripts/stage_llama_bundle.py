#!/usr/bin/env python3
"""Stage a bundled llama.cpp binary for a target platform."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import platform
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LLAMA_BINARY_PATH = (
    PROJECT_ROOT
    / "packages"
    / "llamafarm-llama"
    / "src"
    / "llamafarm_llama"
    / "_binary.py"
)


_LLAMA_BINARY_MODULE = None


def _load_llama_binary_module():
    """Lazily import _binary.py from the llamafarm-llama package.

    The module is loaded on first call so any future top-level side effect
    in _binary.py only runs when the staging script actually needs it,
    not at import time of this script. Subsequent calls reuse the cached
    module.
    """
    global _LLAMA_BINARY_MODULE
    if _LLAMA_BINARY_MODULE is not None:
        return _LLAMA_BINARY_MODULE
    spec = importlib.util.spec_from_file_location(
        "llamafarm_llama_binary", LLAMA_BINARY_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec from {LLAMA_BINARY_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _LLAMA_BINARY_MODULE = module
    return module


PLATFORM_KEYS: dict[str, tuple[str, str, str]] = {
    "linux-x86_64": ("linux", "x86_64", "cpu"),
    "linux-arm64": ("linux", "arm64", "cpu"),
    "darwin-arm64": ("darwin", "arm64", "metal"),
    "windows-x86_64": ("win32", "amd64", "cpu"),
}


def detect_host_platform() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "windows":
        system = "windows"
    elif system not in ("linux", "darwin"):
        raise ValueError(f"unsupported host OS for llama bundle staging: {system}")

    if machine in ("x86_64", "amd64"):
        machine = "x86_64"
    elif machine in ("arm64", "aarch64"):
        machine = "arm64"
    else:
        raise ValueError(
            f"unsupported host architecture for llama bundle staging: {machine}"
        )

    platform_slug = f"{system}-{machine}"
    if platform_slug not in PLATFORM_KEYS:
        raise ValueError(
            f"unsupported host platform for llama bundle staging: {platform_slug}"
        )
    return platform_slug


def normalize_platform(value: str) -> str:
    if value == "host":
        return detect_host_platform()
    if value not in PLATFORM_KEYS:
        raise ValueError(f"unsupported platform: {value}")
    return value


def lib_name_for_system(system: str) -> str:
    if system == "darwin":
        return "libllama.dylib"
    if system in ("windows", "win32"):
        return "llama.dll"
    return "libllama.so"


def create_version_symlinks(dest_dir: Path, system: str) -> None:
    if system == "windows":
        return

    if system == "darwin":
        for lib_file in dest_dir.glob("*.dylib"):
            match = re.match(
                r"^(lib[\w-]+)\.(\d+)\.(\d+)\.(\d+)\.dylib$", lib_file.name
            )
            if not match:
                continue
            base_name = match.group(1)
            major = match.group(2)
            major_symlink = dest_dir / f"{base_name}.{major}.dylib"
            if not major_symlink.exists():
                major_symlink.symlink_to(lib_file.name)
            base_symlink = dest_dir / f"{base_name}.dylib"
            if not base_symlink.exists():
                base_symlink.symlink_to(major_symlink.name)
        return

    for lib_file in dest_dir.iterdir():
        if not lib_file.is_file():
            continue
        match = re.match(r"^(lib[\w-]+)\.so\.(\d+)\.(\d+)\.(\d+)$", lib_file.name)
        if not match:
            continue
        base_name = match.group(1)
        major = match.group(2)
        major_symlink = dest_dir / f"{base_name}.so.{major}"
        if not major_symlink.exists():
            major_symlink.symlink_to(lib_file.name)
        base_symlink = dest_dir / f"{base_name}.so"
        if not base_symlink.exists():
            base_symlink.symlink_to(major_symlink.name)


def copy_dependencies_for_system(
    src_dir: Path,
    dest_dir: Path,
    system: str,
    main_lib: str,
) -> None:
    patterns = [
        "*.dll",
        "*.metal",
    ]

    if system == "darwin":
        patterns.extend([
            "libggml*.*.*.*dylib",
            "libmtmd*.*.*.*dylib",
        ])
    else:
        patterns.extend([
            "libggml*.so.*",
            "libggml*.so",
            "ggml-*.so",
            "libmtmd*.so.*",
            "libmtmd*.so",
            "libcublas*.so.*",
            "libcudart*.so.*",
            "libcublasLt*.so.*",
        ])

    for pattern in patterns:
        for candidate in src_dir.rglob(pattern):
            if (
                not candidate.is_file()
                or candidate.name == main_lib
                or candidate.stat().st_size <= 100
            ):
                continue
            dest = dest_dir / candidate.name
            if not dest.exists():
                shutil.copy2(candidate, dest)

    create_version_symlinks(dest_dir, system)


def build_download_spec(platform_slug: str) -> tuple[str, str, dict]:
    llama_binary = _load_llama_binary_module()
    platform_key = PLATFORM_KEYS[platform_slug]
    manifest = llama_binary.BINARY_MANIFEST[platform_key]
    version = llama_binary.LLAMA_CPP_VERSION
    if platform_key == ("linux", "arm64", "cpu"):
        llamafarm_version = llama_binary._get_llamafarm_release_version()
        url = manifest["artifact"].format(
            version=version,
            llamafarm_version=llamafarm_version,
        )
        artifact = url.split("/")[-1]
    else:
        artifact = manifest["artifact"].format(version=version)
        url = (
            f"https://github.com/{llama_binary.LLAMA_CPP_REPO}"
            f"/releases/download/{version}/{artifact}"
        )
    return artifact, url, manifest


def download_archive(url: str, archive_path: Path) -> None:
    req = Request(url, headers={"User-Agent": "llamafarm-bundle-stager"})
    # Trusted GitHub release URL; HTTPS-only.
    with urlopen(req, timeout=300) as response:  # noqa: S310
        archive_path.write_bytes(response.read())


def _try_fetch_remote_sha256(url: str) -> str | None:
    """Best-effort fetch of a sidecar SHA256.

    GitHub release artifacts sometimes ship `<artifact>.sha256` next to the
    archive. When present, we verify against it to harden the supply chain.
    Missing sidecars (the common case for upstream llama.cpp) just return None
    so the caller logs the locally-computed hash and proceeds — TLS + GitHub's
    cert chain remain the underlying integrity guarantee for those.
    """
    sha_url = url + ".sha256"
    req = Request(sha_url, headers={"User-Agent": "llamafarm-bundle-stager"})
    try:
        # Trusted GitHub release URL; HTTPS-only.
        with urlopen(req, timeout=30) as response:  # noqa: S310
            text = response.read().decode("utf-8", errors="replace").strip()
    except (HTTPError, URLError, OSError):
        return None
    if not text:
        return None
    # Format is typically "<hex>  <filename>" — take the first hex chunk.
    candidate = text.split()[0].strip()
    if len(candidate) == 64 and all(c in "0123456789abcdefABCDEF" for c in candidate):
        return candidate.lower()
    return None


def verify_archive_checksum(url: str, archive_path: Path) -> str:
    """Compute the archive's SHA256 and (when available) verify a remote one.

    Returns the locally-computed digest. Raises RuntimeError if a remote
    sidecar checksum is fetched and disagrees with the local one.
    """
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    remote = _try_fetch_remote_sha256(url)
    if remote is None:
        print(
            f"sha256({archive_path.name}) = {digest} "
            "(no upstream sidecar; relying on TLS for integrity)",
            file=sys.stderr,
        )
        return digest
    if remote != digest:
        raise RuntimeError(
            f"checksum mismatch for {archive_path.name}: "
            f"expected {remote}, got {digest}"
        )
    print(
        f"sha256({archive_path.name}) = {digest} (verified against upstream sidecar)",
        file=sys.stderr,
    )
    return digest


def stage_bundle(platform_slug: str, destination_root: Path) -> Path:
    llama_binary = _load_llama_binary_module()
    dest_dir = destination_root / platform_slug
    destination_root.mkdir(parents=True, exist_ok=True)
    system = PLATFORM_KEYS[platform_slug][0]
    main_lib = lib_name_for_system(system)
    artifact, url, manifest = build_download_spec(platform_slug)

    last_error: Exception | None = None
    for attempt in range(1, 4):
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                archive_path = tmpdir_path / artifact
                extract_dir = tmpdir_path / "extracted"
                extract_dir.mkdir(parents=True, exist_ok=True)

                download_archive(url, archive_path)
                verify_archive_checksum(url, archive_path)

                if artifact.endswith(".zip"):
                    llama_binary._safe_extract_zip(archive_path, extract_dir)
                elif artifact.endswith(".tar.gz") or artifact.endswith(".tgz"):
                    llama_binary._safe_extract_tarball(archive_path, extract_dir)
                else:
                    raise RuntimeError(f"unsupported archive format: {artifact}")

                lib_src = extract_dir / manifest["lib"]
                if not lib_src.exists():
                    candidates = list(extract_dir.rglob(main_lib))
                    if not candidates:
                        raise RuntimeError(
                            f"could not find {main_lib} in extracted archive "
                            f"for {platform_slug}"
                        )
                    lib_src = candidates[0]

                llama_binary._extract_with_symlinks(lib_src, dest_dir / main_lib)
                copy_dependencies_for_system(extract_dir, dest_dir, system, main_lib)
            return dest_dir
        except (HTTPError, URLError, OSError, RuntimeError) as exc:
            last_error = exc
            if attempt == 3:
                break
            delay_seconds = 2 ** (attempt - 1)
            print(
                f"Attempt {attempt} failed staging {platform_slug}: {exc}. "
                f"Retrying in {delay_seconds}s...",
                file=sys.stderr,
            )
            time.sleep(delay_seconds)

    assert last_error is not None
    raise RuntimeError(
        f"failed to stage {platform_slug} after 3 attempts"
    ) from last_error


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage bundled llama.cpp binaries")
    parser.add_argument(
        "--platform",
        default="host",
        help="Target platform slug or 'host'",
    )
    parser.add_argument(
        "--destination-root",
        required=True,
        type=Path,
        help="Root directory that will receive <platform>/ files",
    )
    args = parser.parse_args()

    platform_slug = normalize_platform(args.platform)
    dest_dir = stage_bundle(platform_slug, args.destination_root.resolve())
    print(dest_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
