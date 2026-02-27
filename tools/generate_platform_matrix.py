#!/usr/bin/env python3
"""
Generate GitHub Actions matrix JSON from platforms.yaml.

This script reads addons/platforms.yaml and outputs a JSON matrix
that can be used in GitHub Actions workflows.

Usage:
    python tools/generate_platform_matrix.py
    python tools/generate_platform_matrix.py --addons stt,tts
"""

import argparse
import json
from pathlib import Path

import yaml


def load_platforms():
    """Load platform configurations from platforms.yaml."""
    platforms_file = Path(__file__).parent.parent / "addons" / "platforms.yaml"

    if not platforms_file.exists():
        raise FileNotFoundError(f"Platforms file not found at {platforms_file}")

    with open(platforms_file) as f:
        data = yaml.safe_load(f)

    # Return only enabled platforms
    return [p for p in data.get("platforms", []) if p.get("enabled", True)]


def discover_addons():
    """Discover addon names from the registry directory.

    Only returns addons that have packages (i.e., need wheel builds).
    """
    registry_dir = Path(__file__).parent.parent / "addons" / "registry"
    addons = []
    for yaml_file in sorted(registry_dir.glob("*.yaml")):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
        if data and data.get("name") and data.get("packages"):
            addons.append(data["name"])
    return addons


def generate_matrix(addons=None, platforms=None):
    """Generate a GitHub Actions matrix."""
    all_platforms = load_platforms()

    # Filter platforms if specified
    if platforms:
        platform_names = platforms.split(",")
        all_platforms = [p for p in all_platforms if p["name"] in platform_names]

    # Discover addons from registry if not specified
    addons = discover_addons() if not addons else addons.split(",")

    # Build matrix
    matrix = {
        "include": [
            {
                "platform": platform["name"],
                "runner": platform["runner"],
                "addon": addon,
            }
            for platform in all_platforms
            for addon in addons
        ]
    }

    return matrix


def main():
    parser = argparse.ArgumentParser(
        description="Generate GitHub Actions matrix from platforms.yaml"
    )
    parser.add_argument(
        "--addons", help='Comma-separated list of addons (e.g., "stt,tts")'
    )
    parser.add_argument(
        "--platforms", help='Comma-separated list of platforms (e.g., "macos-arm64,linux-x86_64")'
    )
    args = parser.parse_args()

    matrix = generate_matrix(addons=args.addons, platforms=args.platforms)
    print(json.dumps(matrix))


if __name__ == "__main__":
    main()
