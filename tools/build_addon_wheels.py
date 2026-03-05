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
        raise RuntimeError(
            "TOML parsing requires Python 3.11+ or 'pip install tomli'"
        ) from None

import yaml


def normalize_package_name(name: str) -> str:
    """Normalize package name: lowercase, replace hyphens with underscores, strip extras."""
    # Strip PEP 508 extras brackets (e.g., "uvicorn[standard]" -> "uvicorn")
    name = re.sub(r"\[.*?\]", "", name)
    return re.sub(r"[-_.]+", "_", name.lower())


def parse_uv_lock_packages(uv_lock_path: Path, pyproject_path: Path) -> Set[str]:
    """
    Parse uv.lock file to extract package names reachable from base dependencies only.
    Builds a dependency graph and only includes packages reachable from 
    [project].dependencies.
    """
    if not uv_lock_path.exists():
        return set()
    
    if not pyproject_path.exists():
        return set()
    
    try:
        # Load uv.lock to build dependency graph
        with open(uv_lock_path, 'rb') as f:
            lock_data = tomllib.load(f)
        
        # Load pyproject.toml to get base dependencies
        with open(pyproject_path, 'rb') as f:
            project_data = tomllib.load(f)
        
        # Build package dependency graph from uv.lock
        dependency_graph = {}
        for package in lock_data.get("package", []):
            package_name = package.get("name")
            if package_name:
                normalized_name = normalize_package_name(package_name)
                deps = []
                for dep in package.get("dependencies", []):
                    # uv.lock deps can be dicts (e.g., {"name": "numpy", "marker": "..."})
                    # or strings (e.g., "numpy>=1.24")
                    if isinstance(dep, dict):
                        dep_name = dep.get("name", "")
                    else:
                        dep_name = re.split(r'[<>=!;]', dep.strip())[0].strip()
                    if dep_name:
                        deps.append(normalize_package_name(dep_name))
                dependency_graph[normalized_name] = deps
        
        # Get base dependencies from pyproject.toml [project].dependencies
        base_deps = set()
        project_deps = project_data.get("project", {}).get("dependencies", [])
        for dep in project_deps:
            # Extract package name from dependency spec
            dep_name = re.split(r'[<>=!;]', dep.strip())[0].strip()
            if dep_name:
                base_deps.add(normalize_package_name(dep_name))
        
        # Walk the dependency graph transitively from base dependencies
        base_reachable = set()
        
        def walk_dependencies(package_name: str, visited: Set[str]):
            """Recursively walk dependencies to find all reachable packages."""
            if package_name in visited:
                return  # Avoid cycles
            visited.add(package_name)
            base_reachable.add(package_name)
            
            # Walk transitive dependencies
            for dep in dependency_graph.get(package_name, []):
                walk_dependencies(dep, visited)
        
        # Start walk from each base dependency
        for base_dep in base_deps:
            walk_dependencies(base_dep, set())
        
        count = len(base_reachable)
        logging.info(f"Parsed {count} base-reachable packages from uv.lock")
        logging.debug(f"Base dependencies: {sorted(base_deps)}")
        logging.debug(f"Base-reachable packages: {sorted(base_reachable)}")
        return base_reachable
        
    except (KeyError, TypeError, tomllib.TOMLDecodeError) as e:
        logging.warning(f"Failed to parse uv.lock and dependency graph: {e}")
        return set()


def parse_pyproject_extras(pyproject_path: Path) -> Set[str]:
    """Parse pyproject.toml to extract packages from GPU extra only."""
    if not pyproject_path.exists():
        logging.warning(f"pyproject.toml file not found at {pyproject_path}")
        return set()
    
    try:
        with open(pyproject_path, 'rb') as f:
            project_data = tomllib.load(f)
        
        packages = set()
        
        # Only include packages from the 'gpu' extra (torch etc. installed by CLI)
        # Remove tts, tts-mlx, speech (addon-managed) and dev (not base-installed)
        optional_deps = project_data.get("project", {}).get("optional-dependencies", {})
        gpu_deps = optional_deps.get("gpu", [])
        for dep in gpu_deps:
            # Extract package name from dependency spec 
            package_name = re.split(r'[<>=!]', dep.strip())[0].strip()
            if package_name:
                packages.add(normalize_package_name(package_name))
        
        logging.info(f"Parsed {len(packages)} packages from pyproject.toml gpu extra")
        return packages
    except (KeyError, TypeError, tomllib.TOMLDecodeError) as e:
        logging.warning(f"Failed to parse pyproject.toml: {e}")
        return set()


def get_base_exclusion_set(no_exclude: bool = False) -> Set[str]:
    """Get combined set of packages to exclude from base install."""
    if no_exclude:
        logging.info("Base package exclusion disabled via --no-exclude flag")
        return set()
    
    repo_root = Path(__file__).parent.parent
    uv_lock_path = repo_root / "runtimes" / "universal" / "uv.lock"
    pyproject_path = repo_root / "runtimes" / "universal" / "pyproject.toml"
    
    # Check if required files exist
    if not uv_lock_path.exists():
        raise FileNotFoundError(
            f"uv.lock file not found at {uv_lock_path}. "
            f"Run 'uv sync' to generate it or use --no-exclude to disable filtering."
        )
    
    if not pyproject_path.exists():
        raise FileNotFoundError(
            f"pyproject.toml file not found at {pyproject_path}. "
            f"Check the project structure or use --no-exclude to disable filtering."
        )
    
    # Get packages from uv.lock (base-reachable only)
    uv_packages = parse_uv_lock_packages(uv_lock_path, pyproject_path)
    
    # Get packages from pyproject.toml extras (GPU only)
    extra_packages = parse_pyproject_extras(pyproject_path)
    
    # Combine both sets
    all_excluded = uv_packages | extra_packages
    
    # Fail loudly if exclusion set is empty (would reproduce 4GB problem)
    if not all_excluded:
        raise RuntimeError(
            "Base exclusion set is empty - this would include all packages and "
            "reproduce the original 4GB addon problem. Check that uv.lock and "
            "pyproject.toml are valid, or use --no-exclude if this is intentional."
        )
    
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


def build_addon_wheels(
    addon_name: str,
    target_platform: str,
    output_dir: Path,
    base_excluded: Set[str],
    no_exclude: bool = False
):
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
                msg = f"    Excluding {wheel_name} (base package: {package_name})"
                logging.info(msg)
                wheel_file.unlink()  # Delete the excluded wheel
                excluded_count += 1
            else:
                final_wheel_files.append(wheel_file)
                logging.debug(f"    Keeping {wheel_name} (not in base exclusion set)")
        
        kept_count = len(final_wheel_files)
        msg = f"  Excluded {excluded_count} wheel(s), keeping {kept_count} wheel(s)"
        logging.info(msg)

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

    # Compute base exclusion set once (avoid re-parsing per addon/platform)
    base_excluded = get_base_exclusion_set(no_exclude=args.no_exclude)

    # Build all combinations
    failures = 0
    for addon in addons:
        for plat in platforms:
            try:
                build_addon_wheels(addon, plat, output_dir, base_excluded, no_exclude=args.no_exclude)
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
