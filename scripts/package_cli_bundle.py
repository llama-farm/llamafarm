#!/usr/bin/env python3
"""Package an `lf` binary and bundled llama.cpp files into a release archive."""

from __future__ import annotations

import argparse
import hashlib
import platform
import shutil
import stat
import tarfile
import tempfile
import zipfile
from pathlib import Path

SUPPORTED_PLATFORMS = {
    "linux-x86_64",
    "linux-arm64",
    "darwin-arm64",
    "windows-x86_64",
}


def detect_host_platform() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "windows":
        system = "windows"
    elif system not in ("linux", "darwin"):
        raise ValueError(f"unsupported host OS for CLI packaging: {system}")

    if machine in ("x86_64", "amd64"):
        machine = "x86_64"
    elif machine in ("arm64", "aarch64"):
        machine = "arm64"
    else:
        raise ValueError(f"unsupported host architecture for CLI packaging: {machine}")

    platform_slug = f"{system}-{machine}"
    if platform_slug not in SUPPORTED_PLATFORMS:
        raise ValueError(
            f"unsupported host platform for CLI packaging: {platform_slug}"
        )
    return platform_slug


def normalize_platform(value: str) -> str:
    if value == "host":
        return detect_host_platform()
    if value not in SUPPORTED_PLATFORMS:
        raise ValueError(f"unsupported platform: {value}")
    return value


def archive_suffix(platform_slug: str) -> str:
    if platform_slug.startswith("windows-"):
        return ".zip"
    return ".tar.gz"


def default_binary_name(platform_slug: str) -> str:
    if platform_slug.startswith("windows-"):
        return "lf.exe"
    return "lf"


def resolve_binary_path(binary_path: Path, platform_slug: str) -> Path:
    if binary_path.exists():
        return binary_path
    if platform_slug.startswith("windows-"):
        exe_path = binary_path.with_suffix(".exe")
        if exe_path.exists():
            return exe_path
    raise FileNotFoundError(f"CLI binary not found: {binary_path}")


def write_zip_archive(stage_dir: Path, archive_path: Path) -> None:
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in sorted(stage_dir.rglob("*")):
            if item.is_dir():
                continue
            arcname = item.relative_to(stage_dir)
            info = zipfile.ZipInfo.from_file(item, arcname)
            if item.stat().st_mode & stat.S_IXUSR:
                info.external_attr = 0o755 << 16
            with item.open("rb") as src, zf.open(info, "w") as dst:
                shutil.copyfileobj(src, dst)


def write_tar_archive(stage_dir: Path, archive_path: Path) -> None:
    with tarfile.open(archive_path, "w:gz") as tf:
        for item in sorted(stage_dir.iterdir()):
            tf.add(item, arcname=item.name, recursive=True)


def write_sha256(path: Path) -> Path:
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    checksum_path.write_text(f"{checksum}  {path.name}\n")
    return checksum_path


def package_bundle(
    binary_path: Path,
    platform_slug: str,
    llama_root: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    binary_path = resolve_binary_path(binary_path.resolve(), platform_slug)
    llama_dir = (llama_root / platform_slug).resolve()
    if not llama_dir.exists():
        raise FileNotFoundError(f"staged llama.cpp directory not found: {llama_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"lf-{platform_slug}{archive_suffix(platform_slug)}"

    with tempfile.TemporaryDirectory() as tmpdir:
        stage_dir = Path(tmpdir)
        staged_binary = stage_dir / default_binary_name(platform_slug)
        shutil.copy2(binary_path, staged_binary)
        staged_binary.chmod(0o755)

        staged_llama_dir = stage_dir / "llama-cpp" / platform_slug
        staged_llama_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(llama_dir, staged_llama_dir, symlinks=True)

        if archive_path.suffix == ".zip":
            write_zip_archive(stage_dir, archive_path)
        else:
            write_tar_archive(stage_dir, archive_path)

    checksum_path = write_sha256(archive_path)
    return archive_path, checksum_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Package an lf release bundle")
    parser.add_argument(
        "--binary",
        required=True,
        type=Path,
        help="Path to the built CLI binary",
    )
    parser.add_argument(
        "--platform",
        default="host",
        help="Target platform slug or 'host'",
    )
    parser.add_argument(
        "--llama-root",
        required=True,
        type=Path,
        help=(
            "Directory containing one <platform>/ subdir per target. The "
            "bundle for --platform foo is read from <llama-root>/foo/. "
            "Typically the same value passed to "
            "stage_llama_bundle.py --destination-root."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Archive output directory",
    )
    args = parser.parse_args()

    platform_slug = normalize_platform(args.platform)
    archive_path, checksum_path = package_bundle(
        args.binary,
        platform_slug,
        args.llama_root.resolve(),
        args.output_dir.resolve(),
    )
    print(archive_path)
    print(checksum_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
