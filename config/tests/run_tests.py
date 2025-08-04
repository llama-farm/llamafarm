#!/usr/bin/env python3
"""
Test runner for the LlamaFarm configuration module.
"""

import sys
from pathlib import Path


def main():
    """Run all tests for the configuration module."""
    test_dir = Path(__file__).parent
    config_dir = test_dir.parent

    # Add config directory to Python path
    sys.path.insert(0, str(config_dir))

    print("🧪 Running LlamaFarm Configuration Module Tests")
    print("=" * 50)

    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("❌ pytest is not installed. Please install it:")
        print("   pip install pytest")
        return 1

    # Run pytest with the tests directory
    args = [
        str(test_dir),
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "-x",  # Stop on first failure
    ]

    print(f"Running: pytest {' '.join(args)}")
    print()

    # Run the tests
    exit_code = pytest.main(args)

    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code: {exit_code}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
