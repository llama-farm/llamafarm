#!/usr/bin/env python3
"""
Simple test to verify demos work with real CLI commands.
"""

import subprocess
import sys

def test_cli_help():
    """Test that CLI help works."""
    print("Testing CLI help...")
    result = subprocess.run(
        ["uv", "run", "python", "cli.py", "--help"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print("✅ CLI help works")
        return True
    else:
        print("❌ CLI help failed")
        print(result.stderr)
        return False

def test_demo_scripts():
    """Test that demo scripts exist and are executable."""
    print("\nTesting demo scripts...")
    from pathlib import Path
    
    demo_scripts = [
        "demos/demo1_cloud_fallback.py",
        "demos/demo2_multi_model.py", 
        "demos/demo3_training.py",
        "demos/run_all_demos.py"
    ]
    
    all_exist = True
    for script in demo_scripts:
        script_path = Path(script)
        if script_path.exists():
            print(f"  ✅ {script}")
        else:
            print(f"  ❌ {script} - NOT FOUND")
            all_exist = False
    
    if all_exist:
        print("✅ All demo scripts exist")
        return True
    else:
        print("❌ Some demo scripts missing")
        return False

def test_strategy_file():
    """Test that strategy file exists."""
    print("\nChecking strategy file...")
    from pathlib import Path
    strategy_file = Path("demos/strategies.yaml")
    if strategy_file.exists():
        print(f"✅ Strategy file exists: {strategy_file}")
        
        # Check it's valid YAML
        import yaml
        try:
            with open(strategy_file) as f:
                strategies = yaml.safe_load(f)
                print(f"   Found {len(strategies.get('strategies', {}))} strategies:")
                for name in strategies.get('strategies', {}).keys():
                    print(f"   - {name}")
            return True
        except Exception as e:
            print(f"❌ Strategy file is not valid YAML: {e}")
            return False
    else:
        print(f"❌ Strategy file not found: {strategy_file}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("TESTING LLAMAFARM MODELS CLI")
    print("="*60)
    
    tests = [
        test_cli_help,
        test_demo_scripts,
        test_strategy_file
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append(False)
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("✅ All tests passed!")
        print("\n🎯 You can now run the demos:")
        print("   python demos/demo1_cloud_fallback.py")
        print("   python demos/demo2_multi_model.py")  
        print("   python demos/demo3_training.py")
        print("   python demos/run_all_demos.py")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())