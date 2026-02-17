#!/usr/bin/env python3
"""
Build addon wheel bundles for distribution.

Creates platform-specific tar.gz files containing pre-built wheels for each addon.

Usage:
    python tools/build_addon_wheels.py --addon stt --platform macos-arm64
    python tools/build_addon_wheels.py --addon all --platform all
"""

import argparse
import platform
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import yaml


def load_addon_specs() -> dict:
    """Load addon specifications from individual YAML files in addons/registry/."""
    registry_dir = Path(__file__).parent.parent / "addons" / "registry"

    if not registry_dir.exists():
        raise FileNotFoundError(f"Addon registry directory not found at {registry_dir}")

    specs = {}

    # Load all .yaml files in the registry directory
    for yaml_file in sorted(registry_dir.glob("*.yaml")):
        try:
            with open(yaml_file) as f:
                addon_data = yaml.safe_load(f)

            if not addon_data or "name" not in addon_data:
                print(f"Warning: Skipping invalid addon file {yaml_file.name}")
                continue

            addon_name = addon_data["name"]
            specs[addon_name] = {
                "packages": addon_data.get("packages", []),
            }

        except Exception as e:
            print(f"Warning: Failed to load addon from {yaml_file.name}: {e}")
            continue

    if not specs:
        raise RuntimeError(f"No valid addons found in {registry_dir}")

    return specs


def load_platforms() -> list[str]:
    """Load platform list from platforms.yaml."""
    platforms_file = Path(__file__).parent.parent / "addons" / "platforms.yaml"

    if not platforms_file.exists():
        raise FileNotFoundError(f"Platforms file not found at {platforms_file}")

    with open(platforms_file) as f:
        data = yaml.safe_load(f)

    # Return only enabled platforms
    return [p["name"] for p in data.get("platforms", []) if p.get("enabled", True)]


ADDON_SPECS = load_addon_specs()
PLATFORMS = load_platforms()


def get_host_platform() -> str:
    """Detect the current host platform in our naming convention."""
    machine = platform.machine().lower()
    if sys.platform == "darwin":
        arch = "arm64" if machine == "arm64" else "x86_64"
        return f"macos-{arch}"
    elif sys.platform == "linux":
        if machine in ("aarch64", "arm64"):
            arch = "arm64"
        elif machine in ("x86_64", "amd64"):
            arch = "x86_64"
        else:
            return "unknown"
        return f"linux-{arch}"
    elif sys.platform == "win32":
        return "windows-x86_64"
    return "unknown"


def build_addon_wheels(addon_name: str, target_platform: str, output_dir: Path):
    """Build wheels for an addon."""
    spec = ADDON_SPECS[addon_name]

    # Skip meta-addons (no packages)
    if not spec["packages"]:
        print(f"Skipping {addon_name} (meta-addon with no packages)")
        return

    # Validate that the target platform matches the host, since pip download
    # fetches wheels for the current host regardless of the target label
    host = get_host_platform()
    if host != target_platform:
        raise RuntimeError(
            f"Cannot build for {target_platform} on {host}; "
            f"pip downloads wheels for the host platform"
        )

    print(f"Building {addon_name} for {target_platform}...")

    # Create temp directory for wheels
    wheels_dir = output_dir / f"{addon_name}-{target_platform}-wheels"
    wheels_dir.mkdir(parents=True, exist_ok=True)

    # Download wheels
    for package in spec["packages"]:
        print(f"  Downloading {package}...")
        result = subprocess.run(
            [
                "pip",
                "download",
                "--dest",
                str(wheels_dir),
                "--only-binary=:all:",
                package,
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"Error downloading {package}:")
            print(result.stderr)
            raise RuntimeError(f"Failed to download {package}")

    # Verify we have wheels
    wheel_files = list(wheels_dir.glob("*.whl"))
    if not wheel_files:
        raise RuntimeError(f"No wheel files found in {wheels_dir}")

    print(f"  Downloaded {len(wheel_files)} wheel(s)")

    # Create tar.gz
    tarball_path = output_dir / f"{addon_name}-wheels-{target_platform}.tar.gz"
    print(f"  Creating {tarball_path.name}...")
    with tarfile.open(tarball_path, "w:gz") as tar:
        for wheel_file in wheels_dir.iterdir():
            tar.add(wheel_file, arcname=wheel_file.name)

    print(f"✓ Created {tarball_path}")

    # Clean up temp dir
    shutil.rmtree(wheels_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--addon", required=True, help="Addon name or 'all' for all addons"
    )
    parser.add_argument(
        "--platform", required=True, help="Platform name or 'all' for all platforms"
    )
    parser.add_argument("--output", default="dist/addons", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which addons to build
    if args.addon == "all":
        addons = list(ADDON_SPECS.keys())
    else:
        if args.addon not in ADDON_SPECS:
            print(f"Error: Unknown addon '{args.addon}'")
            print(f"Available addons: {', '.join(ADDON_SPECS.keys())}")
            return 1
        addons = [args.addon]

    # Determine which platforms to build
    if args.platform == "all":
        platforms = PLATFORMS
    else:
        if args.platform not in PLATFORMS:
            print(f"Error: Unknown platform '{args.platform}'")
            print(f"Available platforms: {', '.join(PLATFORMS)}")
            return 1
        platforms = [args.platform]

    # Build all combinations
    failures = 0
    for addon in addons:
        for plat in platforms:
            try:
                build_addon_wheels(addon, plat, output_dir)
            except Exception as e:
                print(f"✗ Failed to build {addon} for {plat}: {e}")
                failures += 1

    if failures:
        print(f"\nBuild finished with {failures} failure(s)")
        return 1

    print("\nBuild complete!")

    return 0


if __name__ == "__main__":
    exit(main() or 0)
