#!/usr/bin/env python3
"""
Legacy test script for the configuration loader.
This script is kept for backward compatibility but the comprehensive
test suite is now located in the tests/ directory.
"""

import sys
from pathlib import Path

def main():
    """Run the comprehensive test suite."""
    print("🔄 Redirecting to comprehensive test suite...")
    print("The main tests are now located in the tests/ directory.")
    print()

    # Import and run the new test runner
    tests_dir = Path(__file__).parent / "tests"
    sys.path.insert(0, str(tests_dir))

    try:
        from run_tests import main as run_main_tests
        return run_main_tests()
    except ImportError:
        print("❌ Could not import the new test runner.")
        print("Please run tests directly with:")
        print("   cd config/tests && python run_tests.py")
        print("   or")
        print("   pytest config/tests/")
        return 1


if __name__ == "__main__":
    sys.exit(main())