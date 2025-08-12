#!/usr/bin/env python3
"""
Test script to verify all demo CLI commands work correctly.
Run from the demos directory to test commands shown in demo output.
"""

import subprocess
import sys
from pathlib import Path
import time

# Colors
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

def run_command(command: str, description: str = ""):
    """Run a command and show the result."""
    print(f"\n{BOLD}{BLUE}Testing:{RESET} {description or command}")
    print(f"{YELLOW}$ {command}{RESET}")
    
    # Remove any box drawing characters that might have been copied
    command = command.replace("│", "").strip()
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,  # Run from rag directory
            timeout=30
        )
        
        if result.returncode == 0:
            print(f"{GREEN}✅ Success{RESET}")
            # Show first few lines of output
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines[:3]:
                    print(f"  {line}")
                if len(lines) > 3:
                    print(f"  ... ({len(lines)-3} more lines)")
            return True
        else:
            print(f"{RED}❌ Failed{RESET}")
            if result.stderr:
                print(f"  Error: {result.stderr[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"{RED}❌ Timed out{RESET}")
        return False
    except Exception as e:
        print(f"{RED}❌ Error: {e}{RESET}")
        return False


def main():
    """Test all demo commands."""
    print(f"{BOLD}{BLUE}{'='*60}{RESET}")
    print(f"{BOLD}{BLUE}Testing Demo CLI Commands{RESET}")
    print(f"{BOLD}{BLUE}{'='*60}{RESET}")
    
    # Commands from the demo output (with intuitive order)
    commands = [
        # Basic search commands
        ("python cli.py search 'transformer architecture' --strategy research_papers_demo",
         "Search research papers"),
         
        ("python cli.py ingest demos/static_samples/customer_support/support_tickets.csv --strategy customer_support_demo",
         "Ingest customer support data"),
         
        ("python cli.py info --strategy code_documentation_demo",
         "Get code documentation info"),
         
        ("python cli.py --verbose search 'AI breakthrough' --strategy news_analysis_demo",
         "Verbose news search"),
         
        ("python cli.py --quiet search 'revenue growth' --strategy business_reports_demo",
         "Quiet business search"),
         
        # Help commands
        ("python cli.py --help",
         "Show main help"),
         
        ("python cli.py strategies list",
         "List available strategies"),
         
        ("python cli.py extractors",
         "Show available extractors"),
    ]
    
    passed = 0
    failed = 0
    
    for command, description in commands:
        if run_command(command, description):
            passed += 1
        else:
            failed += 1
        time.sleep(1)  # Brief pause between commands
    
    # Summary
    print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")
    print(f"{BOLD}Summary:{RESET}")
    print(f"  {GREEN}Passed: {passed}{RESET}")
    print(f"  {RED}Failed: {failed}{RESET}")
    
    if failed == 0:
        print(f"\n{GREEN}{BOLD}All tests passed! 🎉{RESET}")
    else:
        print(f"\n{RED}{BOLD}Some tests failed. Check the output above.{RESET}")
    print(f"{BOLD}{BLUE}{'='*60}{RESET}\n")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())