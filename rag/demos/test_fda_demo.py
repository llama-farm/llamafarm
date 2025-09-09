#!/usr/bin/env python3
"""
Non-interactive test of FDA letters demo
Tests the core functionality without requiring user input
"""

import os
import sys
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Constants
FDA_LETTERS_PATH = Path(__file__).parent / "static_samples" / "fda_letters"
STRATEGY_FILE = Path(__file__).parent / "fda_letters_strategy.yaml"
DB_NAME = "fda_letters_db"
DATA_PROCESSING_STRATEGY = "fda_letters_processing"
# Combined strategy name format: processing_strategy_database_name
STRATEGY_NAME = f"{DATA_PROCESSING_STRATEGY}_{DB_NAME}"

def run_cli_command(command, description=""):
    """Run a CLI command and show results."""
    full_command = f"uv run python cli.py {command}"
    print(f"\n{'='*60}")
    print(f"Test: {description}")
    print(f"Command: {full_command}")
    print(f"{'='*60}")
    
    result = subprocess.run(
        full_command,
        shell=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    
    if result.returncode == 0:
        print(f"✅ SUCCESS")
        if result.stdout:
            print("Output (first 500 chars):")
            print(result.stdout[:500])
    else:
        print(f"❌ FAILED (return code: {result.returncode})")
        if result.stderr:
            print("Error:")
            print(result.stderr[:500])
    
    return result.returncode == 0

def main():
    print("\n" + "="*80)
    print("FDA LETTERS DEMO - NON-INTERACTIVE TEST")
    print("="*80)
    
    # Check files exist
    print(f"\n✓ Strategy file exists: {STRATEGY_FILE.exists()}")
    print(f"✓ FDA letters path exists: {FDA_LETTERS_PATH.exists()}")
    
    pdf_files = list(FDA_LETTERS_PATH.glob("*.pdf"))
    print(f"✓ Found {len(pdf_files)} PDF files")
    
    # Test 1: List available strategies
    success = run_cli_command(
        f"--strategy-file {STRATEGY_FILE} strategies",
        "List available strategies"
    )
    
    # Test 2: Check database info
    run_cli_command(
        f"--strategy-file {STRATEGY_FILE} info",
        "Check database info"
    )
    
    # Test 3: Ingest a single file (quick test)
    first_pdf = pdf_files[0] if pdf_files else None
    if first_pdf:
        success = run_cli_command(
            f"--strategy-file {STRATEGY_FILE} ingest --strategy {STRATEGY_NAME} {first_pdf}",
            f"Ingest single file: {first_pdf.name}"
        )
        
        if success:
            print("\n✅ Single file ingestion successful!")
        else:
            print("\n⚠️  Ingestion had issues - checking dependencies...")
            
            # Check if Ollama is running
            ollama_check = subprocess.run(
                "ollama list",
                shell=True,
                capture_output=True,
                text=True
            )
            
            if ollama_check.returncode != 0:
                print("❌ Ollama is not running! Start it with: ollama serve")
            else:
                print("✅ Ollama is running")
                # Check for nomic-embed-text model
                if "nomic-embed-text" not in ollama_check.stdout:
                    print("⚠️  nomic-embed-text model not found. Pull it with: ollama pull nomic-embed-text")
                else:
                    print("✅ nomic-embed-text model is available")
    
    # Test 4: Simple search (if ingestion worked)
    if success:
        run_cli_command(
            f"--strategy-file {STRATEGY_FILE} search --strategy {STRATEGY_NAME} 'FDA'",
            "Test basic search"
        )
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    
    if success:
        print("\n✅ Core functionality is working!")
        print("You can now run the interactive demo:")
        print("  uv run python demos/demo_fda_letters_interactive.py")
    else:
        print("\n⚠️  Some issues detected. Please check:")
        print("  1. Ollama is running (ollama serve)")
        print("  2. nomic-embed-text model is available (ollama pull nomic-embed-text)")
        print("  3. Dependencies are installed (uv pip install -r requirements.txt)")

if __name__ == "__main__":
    main()