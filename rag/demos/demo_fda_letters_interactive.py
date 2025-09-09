#!/usr/bin/env python3
"""
FDA Letters Interactive Demo
============================
Interactive demonstration of processing FDA regulatory letters using the new RAG system.
Uses LlamaIndex PDF parser with advanced metadata extraction and the CLI interface.

This demo showcases:
1. Strategy validation
2. Document ingestion with progress tracking
3. Metadata extraction patterns
4. Various search strategies
5. Interactive exploration
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Constants
FDA_LETTERS_PATH = Path(__file__).parent / "static_samples" / "fda_letters"
STRATEGY_FILE = Path(__file__).parent / "fda_letters_strategy.yaml"
DB_NAME = "fda_letters_db"
DATA_PROCESSING_STRATEGY = "fda_letters_processing"
# Combined strategy name format: processing_strategy_database_name
STRATEGY_NAME = f"{DATA_PROCESSING_STRATEGY}_{DB_NAME}"

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")

def print_step(step_num: int, text: str):
    """Print a step indicator."""
    print(f"{Colors.CYAN}{Colors.BOLD}Step {step_num}: {text}{Colors.ENDC}")

def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✅ {text}{Colors.ENDC}")

def print_info(text: str):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.ENDC}")

def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.ENDC}")

def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}❌ {text}{Colors.ENDC}")

def wait_for_enter(prompt: str = "Press Enter to continue..."):
    """Wait for user to press Enter."""
    # Check if running in non-interactive mode
    if os.environ.get('DEMO_MODE') == 'automated':
        print(f"\n{Colors.YELLOW}[Automated mode - continuing...]{Colors.ENDC}")
        time.sleep(0.5)  # Small delay for readability
    else:
        input(f"\n{Colors.YELLOW}{prompt}{Colors.ENDC}")

def run_cli_command(command: List[str], capture_output: bool = False) -> subprocess.CompletedProcess:
    """Run a CLI command using uv."""
    # Separate global flags from subcommand and its args
    global_flags = []
    subcommand_and_args = []
    
    # Extract global flags that need to come before subcommand
    i = 0
    while i < len(command):
        if command[i] == "--strategy-file" and i + 1 < len(command):
            global_flags.extend(["--strategy-file", command[i + 1]])
            i += 2
        elif command[i] == "--verbose":
            global_flags.append("--verbose")
            i += 1
        elif command[i] == "--quiet":
            global_flags.append("--quiet")
            i += 1
        else:
            subcommand_and_args.append(command[i])
            i += 1
    
    # Build final command
    full_command = ["uv", "run", "python", "cli.py"] + global_flags + subcommand_and_args
    
    print(f"{Colors.BLUE}Running: {' '.join(full_command)}{Colors.ENDC}")
    
    if capture_output:
        result = subprocess.run(
            full_command,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        if result.returncode == 0:
            print_success("Command completed successfully")
        else:
            print_error(f"Command failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr}")
        return result
    else:
        result = subprocess.run(
            full_command,
            cwd=Path(__file__).parent.parent
        )
        if result.returncode == 0:
            print_success("Command completed successfully")
        else:
            print_error(f"Command failed with return code {result.returncode}")
        return result

def show_sample_files():
    """Display available FDA letter files."""
    print_info("Available FDA Letter Files:")
    pdf_files = list(FDA_LETTERS_PATH.glob("*.pdf"))
    
    if not pdf_files:
        print_error("No PDF files found in FDA letters directory!")
        return []
    
    # Show first 10 and last 5 files
    print(f"\n{Colors.CYAN}Sample files (showing {min(15, len(pdf_files))} of {len(pdf_files)} total):{Colors.ENDC}")
    
    sample_files = pdf_files[:10] + pdf_files[-5:] if len(pdf_files) > 15 else pdf_files
    
    for i, file in enumerate(sample_files, 1):
        file_size = file.stat().st_size / 1024  # Size in KB
        print(f"  {i:2}. {file.name} ({file_size:.1f} KB)")
        
        # Add separator between first 10 and last 5
        if i == 10 and len(pdf_files) > 15:
            print(f"  ... ({len(pdf_files) - 15} more files)")
    
    return pdf_files

def validate_strategy():
    """Check that the strategy configuration is accessible."""
    print_step(1, "Strategy Configuration Check")
    
    print_info(f"Strategy file: {STRATEGY_FILE}")
    print_info(f"Strategy name: {STRATEGY_NAME}")
    print_info(f"Database name: {DB_NAME}")
    
    # Check CLI is available
    result = run_cli_command(
        ["--help"],
        capture_output=True
    )
    
    if result.returncode != 0:
        print_error("CLI not available!")
        sys.exit(1)
    
    # List strategies to verify our strategy file loads
    result = run_cli_command(
        ["--strategy-file", str(STRATEGY_FILE), "strategies"],
        capture_output=True
    )
    
    if result.returncode == 0:
        print_success("Strategy configuration loaded successfully!")
    else:
        print_warning("Could not list strategies, but continuing...")

def ingest_documents(limit: int = None):
    """Ingest FDA letter documents."""
    print_step(2, "Ingesting FDA Letters")
    
    pdf_files = list(FDA_LETTERS_PATH.glob("*.pdf"))
    total_files = len(pdf_files)
    
    if limit:
        print_info(f"Processing first {limit} of {total_files} PDF files")
        files_to_process = pdf_files[:limit]
    else:
        print_info(f"Processing all {total_files} PDF files")
        files_to_process = pdf_files
    
    print_warning("This may take several minutes depending on the number of files...")
    
    # Create a temporary file list for limited processing
    if limit:
        # Process files individually for better control
        for i, file in enumerate(files_to_process, 1):
            print(f"\n{Colors.CYAN}Processing file {i}/{len(files_to_process)}: {file.name}{Colors.ENDC}")
            result = run_cli_command([
                "ingest",
                str(file),
                "--strategy-file", str(STRATEGY_FILE),
                "--strategy", STRATEGY_NAME,
                    "--verbose"
            ])
            
            if result.returncode != 0:
                print_error(f"Failed to process {file.name}")
                continue
    else:
        # Process entire directory
        result = run_cli_command([
            "ingest",
            str(FDA_LETTERS_PATH),
            "--strategy-file", str(STRATEGY_FILE),
            "--strategy", STRATEGY_NAME,
            "--verbose"
        ])
    
    print_success(f"Document ingestion complete!")

def show_extraction_stats():
    """Show statistics about extracted metadata."""
    print_step(3, "Database Information")
    
    print_info("Checking database information...")
    
    # Show database info
    result = run_cli_command([
        "--strategy-file", str(STRATEGY_FILE),
        "info"
    ], capture_output=True)
    
    if result.returncode == 0 and result.stdout:
        print(result.stdout)
    else:
        print_info("Database info will be available after ingestion.")

def run_example_searches():
    """Run example search queries."""
    print_step(4, "Running Example Searches")
    
    # Note: The new schema uses the default_retrieval_strategy from the database config
    # We can't switch retrieval strategies at query time anymore
    print_info("Using default retrieval strategy: hybrid_regulatory_search")
    print_info("(configured in fda_letters_strategy.yaml)")
    
    example_queries = [
        ("Deficiencies Search", "deficiencies"),
        ("Clinical Trials Search", "clinical trials"),  
        ("Manufacturing Search", "manufacturing concerns"),
        ("Safety Search", "safety"),
        ("FDA Approval Search", "approval requirements"),
    ]
    
    for query_name, query_text in example_queries:
        print(f"\n{Colors.CYAN}{Colors.BOLD}{query_name}:{Colors.ENDC}")
        print(f"Query: '{query_text}'")
        
        wait_for_enter(f"Press Enter to run {query_name}...")
        
        result = run_cli_command([
            "search",
            "--strategy", STRATEGY_NAME,
            query_text,
            "--strategy-file", str(STRATEGY_FILE),
            "--top-k", "5",
            "--verbose"
        ])
        
        print("\n" + "-"*60)

def interactive_search():
    """Interactive search session."""
    print_step(5, "Interactive Search Session")
    
    # Skip interactive search in automated mode
    if os.environ.get('DEMO_MODE') == 'automated':
        print_info("Automated mode - skipping interactive search")
        print_info("Running one example search instead...")
        
        # Run a single example search
        run_cli_command([
            "search",
            "--strategy", STRATEGY_NAME,
            "clinical trial safety",
            "--strategy-file", str(STRATEGY_FILE),
            "--top-k", "3"
        ])
        return
    
    print_info("You can now search the FDA letters database interactively.")
    print_info("Using hybrid_regulatory_search retrieval strategy (configured in database)")
    print_info("This combines similarity search with metadata filtering for best results")
    print_info("Type 'quit' to exit the interactive session")
    
    while True:
        print("\n" + "="*60)
        query = input(f"{Colors.CYAN}Enter your search query (or 'quit'): {Colors.ENDC}")
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query.strip():
            continue
        
        # Run search with the configured strategy
        print(f"\n{Colors.BLUE}Searching database...{Colors.ENDC}")
        
        run_cli_command([
            "search",
            "--strategy", STRATEGY_NAME,
            query,
            "--strategy-file", str(STRATEGY_FILE),
            "--top-k", "5",
            "--verbose"
        ])

def show_cli_commands():
    """Display CLI commands for manual use."""
    print_step(6, "CLI Commands for Manual Use")
    
    print_info("You can run these commands manually to work with the FDA letters:")
    
    commands = [
        ("List strategies", f"uv run python cli.py --strategy-file {STRATEGY_FILE} strategies"),
        ("Ingest all documents", f"uv run python cli.py --strategy-file {STRATEGY_FILE} ingest --strategy {STRATEGY_NAME} {FDA_LETTERS_PATH}"),
        ("Basic search", f"uv run python cli.py --strategy-file {STRATEGY_FILE} search --strategy {STRATEGY_NAME} 'your query'"),
        ("Search with different retrieval", f"uv run python cli.py --strategy-file {STRATEGY_FILE} search --strategy {STRATEGY_NAME} --retrieval-strategy metadata_filtered_search 'your query'"),
        ("Show database info", f"uv run python cli.py --strategy-file {STRATEGY_FILE} info"),
    ]
    
    for desc, cmd in commands:
        print(f"\n{Colors.CYAN}{desc}:{Colors.ENDC}")
        print(f"  {cmd}")

def clean_database():
    """Clean up any existing database at the start."""
    print_info("Cleaning up any existing database...")
    
    # Direct cleanup - simpler and more reliable
    import shutil
    db_path = Path("./data/fda_letters_vectordb")
    if db_path.exists():
        try:
            shutil.rmtree(db_path)
            print_success("Existing database cleaned successfully")
        except Exception as e:
            print_warning(f"Could not clean database: {e}")
    else:
        print_info("No existing database found")

def main():
    """Main demo function."""
    print_header("FDA LETTERS PROCESSING DEMO")
    print_info("This demo showcases processing FDA regulatory letters using:")
    print("  • LlamaIndex PDF parser with semantic chunking")
    print("  • Advanced metadata extraction (dates, entities, patterns)")
    print("  • Multiple retrieval strategies")
    print("  • Interactive search capabilities")
    
    # Check if strategy file exists
    if not STRATEGY_FILE.exists():
        print_error(f"Strategy file not found: {STRATEGY_FILE}")
        sys.exit(1)
    
    # Check if FDA letters directory exists
    if not FDA_LETTERS_PATH.exists():
        print_error(f"FDA letters directory not found: {FDA_LETTERS_PATH}")
        sys.exit(1)
    
    # Clean database at START of demo (so we can rerun it)
    print("\n" + "="*60)
    print_step(0, "Database Cleanup")
    clean_database()
    
    # Show available files
    print("\n" + "="*60)
    pdf_files = show_sample_files()
    
    if not pdf_files:
        print_error("No PDF files found to process!")
        sys.exit(1)
    
    wait_for_enter()
    
    # Step 1: Validate strategy
    print("\n" + "="*60)
    validate_strategy()
    wait_for_enter()
    
    # Step 2: Ask about ingestion
    print("\n" + "="*60)
    print_step(2, "Document Ingestion Options")
    
    # Check if running in automated mode
    if os.environ.get('DEMO_MODE') == 'automated':
        print_info("Automated mode - processing ALL documents")
        ingest_documents()  # No limit - process all files!
    else:
        print("How many documents would you like to process?")
        print("  1. Quick demo (first 5 files)")
        print("  2. Medium demo (first 20 files)")
        print(f"  3. Full processing (all {len(pdf_files)} files)")
        print("  4. Skip ingestion (use existing data)")
        
        choice = input(f"{Colors.CYAN}Enter choice (1-4): {Colors.ENDC}").strip()
        
        if choice == "1":
            ingest_documents(limit=5)
        elif choice == "2":
            ingest_documents(limit=20)
        elif choice == "3":
            ingest_documents()
        elif choice == "4":
            print_info("Skipping ingestion, using existing data...")
        else:
            print_warning("Invalid choice, using quick demo (5 files)")
            ingest_documents(limit=5)
    
    wait_for_enter()
    
    # Step 3: Show extraction statistics
    print("\n" + "="*60)
    show_extraction_stats()
    wait_for_enter()
    
    # Step 4: Run example searches
    print("\n" + "="*60)
    run_example_searches()
    wait_for_enter()
    
    # Step 5: Interactive search
    print("\n" + "="*60)
    interactive_search()
    
    # Step 6: Show CLI commands
    print("\n" + "="*60)
    show_cli_commands()
    
    # Conclusion
    print_header("DEMO COMPLETE")
    print_success("FDA Letters processing demo completed successfully!")
    print_info("The vector database has been created and is ready for use.")
    print_info(f"Database location: ./data/fda_letters_vectordb")
    print_info(f"Strategy file: {STRATEGY_FILE}")
    print("\nYou can continue using the CLI commands shown above to work with the data.")
    print_info("\n💡 Note: The database is preserved so you can continue querying it.")
    print_info("The database will be cleaned at the START of the next demo run.")
    print_info("\n📌 To run this demo again:")
    print(f"  {Colors.CYAN}uv run python demos/demo_fda_letters_interactive.py{Colors.ENDC}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n" + Colors.YELLOW + "Demo interrupted by user." + Colors.ENDC)
        sys.exit(0)
    except Exception as e:
        print_error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)