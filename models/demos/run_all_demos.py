#!/usr/bin/env python3
"""
Run All LlamaFarm Demos
Showcases the complete model management capabilities.

This script uses ONLY CLI commands - no hardcoding!
"""

import subprocess
import sys
from pathlib import Path
import time

def run_command(cmd: str, check: bool = True):
    """Run a CLI command and display output."""
    print(f"\n🔧 Running: {cmd}")
    print("-" * 60)
    
    result = subprocess.run(
        cmd, 
        shell=True, 
        capture_output=False,  # Show output directly
        text=True,
        check=check
    )
    
    return result.returncode == 0

def print_banner(title: str, description: str = ""):
    """Print a formatted banner."""
    width = 70
    print("\n" + "=" * width)
    print(f"{title:^{width}}")
    if description:
        print("-" * width)
        for line in description.split("\n"):
            if line.strip():
                print(f"{line:^{width}}")
    print("=" * width)

def main():
    """Run all demonstrations."""
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║           🦙 LLAMAFARM MODELS - COMPLETE DEMO SUITE        ║
    ╠════════════════════════════════════════════════════════════╣
    ║                                                            ║
    ║  Welcome to the LlamaFarm Models demonstration suite!     ║
    ║                                                            ║
    ║  This will showcase:                                      ║
    ║  1. Cloud API Fallback   - Resilient API management       ║
    ║  2. Multi-Model Strategy - Task-optimized routing         ║
    ║  3. Model Training       - Fine-tuning pipeline           ║
    ║                                                            ║
    ║  Everything is driven by strategies.yaml configuration    ║
    ║  NO HARDCODING - pure CLI + strategies!                   ║
    ║                                                            ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Check if running in automated mode
    auto_mode = "--auto" in sys.argv
    quick_mode = "--quick" in sys.argv
    
    if not auto_mode:
        input("\nPress Enter to begin the demo suite...")
    
    # Auto-setup all requirements before running demos
    print("\n📦 Setting up all components...")
    if not run_command("uv run python cli.py setup demos/strategies.yaml --auto --verbose", check=False):
        print("⚠️  Some components may not have installed, but continuing with demos...")
    
    # Track timing
    suite_start = time.time()
    
    # Demo 1: Cloud Fallback
    print_banner("DEMO 1: CLOUD API FALLBACK", 
                "Automatic failover from cloud to local models")
    
    if not auto_mode:
        input("\nPress Enter to start Demo 1...")
    
    if not run_command("uv run python cli.py demo fallback --verbose", check=False):
        print("⚠️  Demo 1 had issues, continuing...")
    
    time.sleep(2)  # Brief pause between demos
    
    # Demo 2: Multi-Model
    print_banner("DEMO 2: MULTI-MODEL OPTIMIZATION",
                "Different models for different tasks")
    
    if not auto_mode:
        input("\nPress Enter to start Demo 2...")
    
    if not run_command("uv run python cli.py demo multi-model --verbose", check=False):
        print("⚠️  Demo 2 had issues, continuing...")
    
    time.sleep(2)
    
    # Demo 3: Training (optional, as it takes longer)
    print_banner("DEMO 3: MODEL FINE-TUNING",
                "Train and deploy a specialized model")
    
    if not auto_mode:
        response = input("\nDemo 3 involves model training (5-10 minutes). Run it? [y/N]: ")
        if response.lower() != 'y':
            print("Skipping Demo 3...")
        else:
            quick_flag = "--quick" if quick_mode else ""
            if not run_command(f"uv run python cli.py demo training --verbose {quick_flag}", check=False):
                print("⚠️  Demo 3 had issues, continuing...")
    elif not quick_mode:
        print("\n⏭️  Skipping training demo in auto mode (use --quick to include)")
    else:
        if not run_command("uv run python cli.py demo training --verbose --quick", check=False):
            print("⚠️  Demo 3 had issues, continuing...")
    
    # Summary
    elapsed = time.time() - suite_start
    
    print_banner("🎉 DEMO SUITE COMPLETE!", 
                f"Total time: {elapsed/60:.1f} minutes")
    
    print("\n📚 What we demonstrated:")
    print("   ✅ Automatic API fallback for reliability")
    print("   ✅ Task-based model routing for efficiency")
    print("   ✅ Fine-tuning pipeline for specialization")
    print("   ✅ Model conversion for deployment")
    print("   ✅ Strategy-driven configuration")
    
    print("\n🚀 Next Steps:")
    print("   1. Explore demos/strategies.yaml to see configurations")
    print("   2. Try individual demos with your own prompts")
    print("   3. Create custom strategies for your use cases")
    print("   4. Fine-tune models on your own datasets")
    
    print("\n📖 Documentation:")
    print("   • README.md           - Overview and setup")
    print("   • docs/STRATEGIES.md  - Strategy configuration guide")
    print("   • docs/TRAINING.md    - Fine-tuning guide")
    print("   • docs/DEPLOYMENT.md  - Deployment options")
    
    print("\n💬 CLI Commands to Try:")
    print("   uv run python cli.py --help")
    print("   uv run python cli.py list-strategies")
    print("   uv run python cli.py demo --help")
    print("   uv run python cli.py train --help")
    print("   uv run python cli.py convert --help")
    
    print("\n" + "=" * 70)
    print("Thank you for trying LlamaFarm Models! 🦙")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()