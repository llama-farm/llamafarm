#!/usr/bin/env python3
"""
🚀 LlamaFarm Models - Quick Setup Script

This script ensures all components are properly installed for LlamaFarm Models.
Run this after cloning the repository to get everything ready.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description=None):
    """Run a command and return success status."""
    if description:
        print(f"\n📋 {description}")
    print(f"$ {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0 and result.stderr:
        print(f"❌ Error: {result.stderr}")
    
    return result.returncode == 0

def main():
    """Main setup function."""
    print("="*70)
    print("🦙 LLAMAFARM MODELS - COMPLETE SETUP")
    print("="*70)
    print("\nThis script will set up everything you need to use LlamaFarm Models:")
    print("  • Install Python dependencies")
    print("  • Set up Ollama (if available)")
    print("  • Install GGUF converter for model exports")
    print("  • Verify all components")
    
    # Check Python version
    print("\n1️⃣ Checking Python version...")
    if sys.version_info < (3, 10):
        print(f"❌ Python 3.10+ required, you have {sys.version}")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Check if uv is installed
    print("\n2️⃣ Checking for uv package manager...")
    if not run_command("uv --version", "Checking uv"):
        print("❌ uv not found. Please install it:")
        print("   curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)
    print("✅ uv is installed")
    
    # Install Python dependencies
    print("\n3️⃣ Installing Python dependencies...")
    if not run_command("uv sync", "Installing dependencies with uv"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    print("✅ Python dependencies installed")
    
    # Set up environment file if it doesn't exist
    print("\n4️⃣ Setting up environment...")
    env_file = Path("../.env")
    env_example = Path("../.env.example")
    
    if not env_file.exists() and env_example.exists():
        print("Creating .env file from template...")
        import shutil
        shutil.copy(env_example, env_file)
        print("✅ Created .env file - please add your API keys if you have them")
    elif env_file.exists():
        print("✅ .env file already exists")
    else:
        print("⚠️  No .env.example found - you may need to create .env manually")
    
    # Run automatic setup for all strategies
    print("\n5️⃣ Setting up all components...")
    print("This will install:")
    print("  • Ollama (if not installed)")
    print("  • GGUF converter (llama.cpp)")
    print("  • Python dependencies for conversion")
    
    if run_command(
        "uv run python cli.py setup demos/strategies.yaml --auto --verbose",
        "Running automatic setup"
    ):
        print("✅ All components installed successfully")
    else:
        print("⚠️  Some components may have failed to install")
        print("   This is okay - they'll be installed when needed")
    
    # Test with mock model
    print("\n6️⃣ Testing with mock model...")
    if run_command(
        'uv run python cli.py info --strategy mock_development',
        "Testing mock model"
    ):
        print("✅ Mock model working correctly")
    else:
        print("❌ Mock model test failed")
    
    # Final instructions
    print("\n" + "="*70)
    print("✅ SETUP COMPLETE!")
    print("="*70)
    print("\n🎉 You're ready to use LlamaFarm Models!")
    print("\n🚀 Quick Start Commands:")
    print("\n1. Test without API keys (mock model):")
    print("   uv run python demos/demo_mock_model.py")
    print("\n2. List available strategies:")
    print("   uv run python cli.py list-strategies")
    print("\n3. Run interactive chat:")
    print("   uv run python cli.py chat")
    print("\n4. Run all demos:")
    print("   uv run python demos/run_all_demos.py")
    print("\n📚 Documentation:")
    print("   • README.md - Complete guide")
    print("   • docs/CLI.md - CLI reference")
    print("   • docs/STRATEGIES.md - Strategy system")
    print("\n💡 Tips:")
    print("   • Add API keys to .env for cloud models")
    print("   • Start Ollama with 'ollama serve' for local models")
    print("   • Use mock models for development (no setup needed!)")

if __name__ == "__main__":
    main()