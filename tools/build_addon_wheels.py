#!/usr/bin/env python3
"""
Build addon wheel bundles for distribution.

Creates platform-specific tar.gz files containing pre-built wheels for each addon.

Usage:
    python tools/build_addon_wheels.py --addon stt --platform macos-arm64
    python tools/build_addon_wheels.py --addon all --platform all
"""

import argparse
import logging
import platform
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Dict, Set

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        print("Error: TOML parsing requires Python 3.11+ or 'pip install tomli'")
        sys.exit(1)

import yaml


def normalize_package_name(name: str) -> str:
    """Normalize package name: lowercase, replace hyphens with underscores."""
    return re.sub(r"[-_.]+", "_", name.lower())


def parse_uv_lock_packages(uv_lock_path: Path) -> Set[str]:
    """Parse uv.lock file to extract package names from base install."""
    if not uv_lock_path.exists():
        logging.warning(f"uv.lock file not found at {uv_lock_path}")
        return set()
    
    try:
        with open(uv_lock_path, 'rb') as f:
            lock_data = tomllib.load(f)
        
        packages = set()
        for package in lock_data.get("package", []):
            package_name = package.get("name")
            if package_name:
                packages.add(normalize_package_name(package_name))
        
        logging.info(f"Parsed {len(packages)} packages from uv.lock")
        return packages
    except Exception as e:
        logging.warning(f"Failed to parse uv.lock: {e}")
        return set()


def parse_pyproject_extras(pyproject_path: Path) -> Set[str]:
    """Parse pyproject.toml to extract packages from optional-dependencies and dependency-groups."""
    if not pyproject_path.exists():
        logging.warning(f"pyproject.toml file not found at {pyproject_path}")
        return set()
    
    try:
        with open(pyproject_path, 'rb') as f:
            project_data = tomllib.load(f)
        
        packages = set()
        
        # Parse optional-dependencies (gpu, tts, tts-mlx, speech, etc.)
        optional_deps = project_data.get("project", {}).get("optional-dependencies", {})
        for extra_name, deps in optional_deps.items():
            if extra_name in ["gpu", "tts", "tts-mlx", "speech"]:
                for dep in deps:
                    # Extract package name from dependency spec (e.g., "torch>=2.0.0" -> "torch")
                    package_name = re.split(r'[<>=!]', dep.strip())[0].strip()
                    packages.add(normalize_package_name(package_name))
        
        # Parse dependency-groups (dev group)
        dependency_groups = project_data.get("dependency-groups", {})
        dev_deps = dependency_groups.get("dev", [])
        for dep in dev_deps:
            package_name = re.split(r'[<>=!]', dep.strip())[0].strip()
            packages.add(normalize_package_name(package_name))
        
        logging.info(f"Parsed {len(packages)} packages from pyproject.toml extras/groups")
        return packages
    except Exception as e:
        logging.warning(f"Failed to parse pyproject.toml: {e}")
        return set()


def get_base_exclusion_set() -> Set[str]:
    """Get combined set of packages to exclude from base install."""
    repo_root = Path(__file__).parent.parent
    uv_lock_path = repo_root / "runtimes" / "universal" / "uv.lock"
    pyproject_path = repo_root / "runtimes" / "universal" / "pyproject.toml"
    
    # Get packages from uv.lock
    uv_packages = parse_uv_lock_packages(uv_lock_path)
    
    # Get packages from pyproject.toml extras/groups
    extra_packages = parse_pyproject_extras(pyproject_path)
    
    # Combine both sets
    all_excluded = uv_packages | extra_packages
    
    logging.info(f"Total base packages to exclude: {len(all_excluded)}")
    logging.debug(f"Excluded packages: {sorted(all_excluded)}")
    
    return all_excluded


def extract_package_name_from_wheel(wheel_filename: str) -> str:
    """Extract normalized package name from wheel filename."""
    # Wheel filename format: {name}-{version}-{tags}.whl
    # Split on '-' and take first part as package name
    parts = wheel_filename.split('-')
    if parts:
        return normalize_package_name(parts[0])
    return normalize_package_name(wheel_filename.replace('.whl', ''))


def get_addon_keep_packages(addon_name: str, addon_spec: Dict) -> Set[str]:
    """Get packages that should always be kept for this addon."""
    keep_packages = set()
    
    # Add packages from addon's packages list (always keep)
    for package in addon_spec.get("packages", []):
        # Handle URL-based packages (e.g., spaCy model URLs)
        if package.startswith("http"):
            # For URL packages, try to extract package name from URL
            url_parts = package.split("/")
            if url_parts:
                filename = url_parts[-1]
                if filename.endswith(".whl"):
                    package_name = extract_package_name_from_wheel(filename)
                    keep_packages.add(package_name)
        else:
            # Regular package spec
            package_name = re.split(r'[<>=!]', package.strip())[0].strip()
            keep_packages.add(normalize_package_name(package_name))
    
    # Add packages from keep_packages list if present
    for package in addon_spec.get("keep_packages", []):
        package_name = re.split(r'[<>=!]', package.strip())[0].strip()
        keep_packages.add(normalize_package_name(package_name))
    
    logging.info(f"Addon {addon_name} keep packages: {sorted(keep_packages)}")
    return keep_packages


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
                "keep_packages": addon_data.get("keep_packages", []),
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


def build_addon_wheels(addon_name: str, target_platform: str, output_dir: Path, no_exclude: bool = False):
    """Build wheels for an addon."""
    spec = ADDON_SPECS[addon_name]

    # Skip meta-addons (no packages)
    if not spec["packages"]:
        logging.info(f"Skipping {addon_name} (meta-addon with no packages)")
        return

    # Validate that the target platform matches the host, since pip download
    # fetches wheels for the current host regardless of the target label
    host = get_host_platform()
    if host != target_platform:
        raise RuntimeError(
            f"Cannot build for {target_platform} on {host}; "
            f"pip downloads wheels for the host platform"
        )

    logging.info(f"Building {addon_name} for {target_platform}...")

    # Create temp directory for wheels
    wheels_dir = output_dir / f"{addon_name}-{target_platform}-wheels"
    wheels_dir.mkdir(parents=True, exist_ok=True)

    # Download wheels
    for package in spec["packages"]:
        logging.info(f"  Downloading {package}...")
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
            logging.error(f"Error downloading {package}:")
            logging.error(result.stderr)
            raise RuntimeError(f"Failed to download {package}")

    # Get all wheel files
    all_wheel_files = list(wheels_dir.glob("*.whl"))
    if not all_wheel_files:
        raise RuntimeError(f"No wheel files found in {wheels_dir}")

    logging.info(f"  Downloaded {len(all_wheel_files)} wheel(s) before filtering")

    # Filter wheels if exclusion is enabled
    if no_exclude:
        logging.info("  Exclusion disabled via --no-exclude flag")
        final_wheel_files = all_wheel_files
    else:
        # Get base packages to exclude
        base_excluded = get_base_exclusion_set()
        
        # Get packages to keep for this addon
        addon_keep = get_addon_keep_packages(addon_name, spec)
        
        final_wheel_files = []
        excluded_count = 0
        
        for wheel_file in all_wheel_files:
            wheel_name = wheel_file.name
            package_name = extract_package_name_from_wheel(wheel_name)
            
            # Decide whether to keep this wheel
            if package_name in addon_keep:
                final_wheel_files.append(wheel_file)
                logging.debug(f"    Keeping {wheel_name} (addon-specific package)")
            elif package_name in base_excluded:
                logging.info(f"    Excluding {wheel_name} (base package: {package_name})")
                wheel_file.unlink()  # Delete the excluded wheel
                excluded_count += 1
            else:
                final_wheel_files.append(wheel_file)
                logging.debug(f"    Keeping {wheel_name} (not in base exclusion set)")
        
        logging.info(f"  Excluded {excluded_count} wheel(s), keeping {len(final_wheel_files)} wheel(s)")

    # Create tar.gz with remaining wheels
    tarball_path = output_dir / f"{addon_name}-wheels-{target_platform}.tar.gz"
    logging.info(f"  Creating {tarball_path.name}...")
    
    if not final_wheel_files:
        logging.warning(f"  No wheels remaining after filtering for {addon_name}")
        # Still create an empty tarball for consistency
    
    with tarfile.open(tarball_path, "w:gz") as tar:
        for wheel_file in final_wheel_files:
            if wheel_file.exists():  # Double-check file still exists
                tar.add(wheel_file, arcname=wheel_file.name)

    # Show final tarball size
    tarball_size_mb = tarball_path.stat().st_size / (1024 * 1024)
    logging.info(f"✓ Created {tarball_path.name} ({tarball_size_mb:.1f} MB)")

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
    parser.add_argument(
        "--no-exclude", 
        action="store_true", 
        help="Disable filtering of base packages (for debugging)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )

    if args.no_exclude:
        logging.info("Base package exclusion is DISABLED")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which addons to build
    if args.addon == "all":
        addons = list(ADDON_SPECS.keys())
    else:
        if args.addon not in ADDON_SPECS:
            logging.error(f"Unknown addon '{args.addon}'")
            logging.error(f"Available addons: {', '.join(ADDON_SPECS.keys())}")
            return 1
        addons = [args.addon]

    # Determine which platforms to build
    if args.platform == "all":
        platforms = PLATFORMS
    else:
        if args.platform not in PLATFORMS:
            logging.error(f"Unknown platform '{args.platform}'")
            logging.error(f"Available platforms: {', '.join(PLATFORMS)}")
            return 1
        platforms = [args.platform]

    # Build all combinations
    failures = 0
    for addon in addons:
        for plat in platforms:
            try:
                build_addon_wheels(addon, plat, output_dir, no_exclude=args.no_exclude)
            except Exception as e:
                logging.error(f"✗ Failed to build {addon} for {plat}: {e}")
                failures += 1

    if failures:
        logging.error(f"\nBuild finished with {failures} failure(s)")
        return 1

    logging.info("\nBuild complete!")

    return 0


if __name__ == "__main__":
    exit(main() or 0)
