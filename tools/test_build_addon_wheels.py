#!/usr/bin/env python3
"""
Unit tests for build_addon_wheels.py

Run with: python3 tools/test_build_addon_wheels.py
"""

import re
import tempfile
from pathlib import Path


def normalize_package_name(name: str) -> str:
    """Normalize package name: lowercase, replace hyphens with underscores."""
    return re.sub(r"[-_.]+", "_", name.lower())


def extract_package_name_from_wheel(wheel_filename: str) -> str:
    """Extract normalized package name from wheel filename."""
    parts = wheel_filename.split('-')
    if parts:
        return normalize_package_name(parts[0])
    return normalize_package_name(wheel_filename.replace('.whl', ''))


def test_normalize_package_name():
    """Test package name normalization."""
    test_cases = [
        ("torch", "torch"),
        ("torch-audio", "torch_audio"), 
        ("PyTorch", "pytorch"),
        ("scikit-learn", "scikit_learn"),
        ("pillow", "pillow"),
        ("opencv-python-headless", "opencv_python_headless"),
        ("en_core_web_sm", "en_core_web_sm"),
    ]
    
    print("Testing normalize_package_name:")
    all_passed = True
    for input_name, expected in test_cases:
        result = normalize_package_name(input_name)
        passed = result == expected
        all_passed = all_passed and passed
        status = "✓" if passed else "✗"
        print(f"  {status} {input_name} -> {result} (expected: {expected})")
    
    return all_passed


def test_extract_package_name_from_wheel():
    """Test wheel filename parsing."""
    test_cases = [
        ("torch-2.0.0-cp310-cp310-linux_x86_64.whl", "torch"),
        ("scikit_learn-1.3.0-cp310-cp310-linux_x86_64.whl", "scikit_learn"),
        ("en_core_web_sm-3.8.0-py3-none-any.whl", "en_core_web_sm"),
        ("opencv_python_headless-4.8.0-cp310-cp310-linux_x86_64.whl", "opencv_python_headless"),
        ("kokoro-0.9.4-py3-none-any.whl", "kokoro"),
    ]
    
    print("\nTesting extract_package_name_from_wheel:")
    all_passed = True
    for wheel_name, expected in test_cases:
        result = extract_package_name_from_wheel(wheel_name)
        passed = result == expected
        all_passed = all_passed and passed
        status = "✓" if passed else "✗"
        print(f"  {status} {wheel_name} -> {result} (expected: {expected})")
    
    return all_passed


def test_filtering_logic():
    """Test the filtering logic with sample data."""
    print("\nTesting filtering logic:")
    
    # Sample base packages (would be excluded)
    base_excluded = {
        "torch", "transformers", "numpy", "pillow", "fastapi", "uvicorn", "pydantic"
    }
    
    # Sample addon keep packages (TTS example)
    addon_keep = {
        "kokoro", "misaki", "spacy", "pydub", "av", "pocket_tts"
    }
    
    # Sample wheel files
    test_wheels = [
        ("torch-2.0.0-cp310-cp310-linux_x86_64.whl", False, "base package"),
        ("transformers-4.35.0-py3-none-any.whl", False, "base package"), 
        ("kokoro-0.9.4-py3-none-any.whl", True, "addon-specific"),
        ("misaki-0.9.0-py3-none-any.whl", True, "addon-specific"),
        ("spacy-3.8.11-py3-none-any.whl", True, "addon-specific"),
        ("some_other_package-1.0.0-py3-none-any.whl", True, "not in base"),
    ]
    
    all_passed = True
    for wheel_name, should_keep, reason in test_wheels:
        package_name = extract_package_name_from_wheel(wheel_name)
        
        # Apply filtering logic
        if package_name in addon_keep:
            keep = True
            actual_reason = "addon-specific"
        elif package_name in base_excluded:
            keep = False
            actual_reason = "base package"
        else:
            keep = True
            actual_reason = "not in base"
        
        passed = keep == should_keep
        all_passed = all_passed and passed
        
        status = "✓" if passed else "✗"
        action = "KEEP" if keep else "EXCLUDE"
        print(f"  {status} {action} {wheel_name} ({actual_reason})")
    
    return all_passed


if __name__ == "__main__":
    print("Running unit tests for build_addon_wheels.py...\n")
    
    test1 = test_normalize_package_name()
    test2 = test_extract_package_name_from_wheel()
    test3 = test_filtering_logic()
    
    all_passed = test1 and test2 and test3
    
    print(f"\n{'✓ All tests passed!' if all_passed else '✗ Some tests failed!'}")
    exit(0 if all_passed else 1)