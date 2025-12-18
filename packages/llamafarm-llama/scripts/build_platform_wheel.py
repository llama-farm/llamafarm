#!/usr/bin/env python3
"""
Build a platform-specific wheel with bundled llama.cpp binary.

Usage:
    python scripts/build_platform_wheel.py --platform linux-x64-cpu
    python scripts/build_platform_wheel.py --platform macos-arm64-metal
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


PLATFORM_MAP = {
    "linux-x64-cpu": ("linux", "x86_64", "cpu"),
    "linux-x64-cuda11": ("linux", "x86_64", "cuda11"),
    "linux-x64-cuda12": ("linux", "x86_64", "cuda12"),
    "linux-x64-vulkan": ("linux", "x86_64", "vulkan"),
    "macos-arm64-metal": ("darwin", "arm64", "metal"),
    "macos-x64-cpu": ("darwin", "x86_64", "cpu"),
    "win-x64-cpu": ("win32", "amd64", "cpu"),
    "win-x64-cuda12": ("win32", "amd64", "cuda12"),
    "win-x64-vulkan": ("win32", "amd64", "vulkan"),
}


def build_wheel(platform_str: str, output_dir: Path):
    """Build a wheel for the specified platform with bundled binary."""
    from llamafarm_llama._binary import download_binary

    if platform_str not in PLATFORM_MAP:
        print(f"Unknown platform: {platform_str}")
        print(f"Available: {', '.join(PLATFORM_MAP.keys())}")
        sys.exit(1)

    platform_key = PLATFORM_MAP[platform_str]
    print(f"Building wheel for {platform_key}")

    # Get the lib directory
    lib_dir = Path(__file__).parent.parent / "src" / "llamafarm_llama" / "lib"
    lib_dir.mkdir(exist_ok=True)

    # Clear any existing binaries
    for f in lib_dir.glob("*"):
        if f.name != ".gitkeep":
            if f.is_dir():
                shutil.rmtree(f)
            else:
                f.unlink()

    # Download for target platform
    print(f"Downloading binary for {platform_str}...")
    download_binary(lib_dir, platform_key=platform_key)

    # Build the wheel
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building wheel...")
    subprocess.check_call(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(output_dir)],
        cwd=Path(__file__).parent.parent,
    )

    # Find the built wheel
    wheels = list(output_dir.glob("*.whl"))
    if not wheels:
        print("No wheel built!")
        sys.exit(1)

    wheel_path = wheels[0]
    print(f"Built: {wheel_path}")

    # Clean up lib directory
    for f in lib_dir.glob("*"):
        if f.name != ".gitkeep":
            if f.is_dir():
                shutil.rmtree(f)
            else:
                f.unlink()

    return wheel_path


def main():
    parser = argparse.ArgumentParser(description="Build platform-specific wheel")
    parser.add_argument(
        "--platform",
        required=True,
        choices=list(PLATFORM_MAP.keys()),
        help="Target platform",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist"),
        help="Output directory (default: dist)",
    )
    args = parser.parse_args()

    build_wheel(args.platform, args.output)


if __name__ == "__main__":
    main()
