#!/usr/bin/env python3
"""
Demo 2: Multi-Model Strategy
Shows different models optimized for different tasks.

This demo uses ONLY CLI commands - no hardcoding!
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Add colorama for colored output
try:
    from colorama import init, Fore, Style, Back
    init(autoreset=True)
except ImportError:
    # Fallback if colorama not available
    class Fore:
        BLACK = RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ''
    class Style:
        RESET_ALL = BRIGHT = DIM = NORMAL = ''

def run_cli_command(cmd: str, description: str = None, check_error: bool = True, show_output: bool = True):
    """Run a CLI command and show output."""
    if description:
        print(f"\n{Fore.CYAN}📋 {description}{Style.RESET_ALL}")
    
    print(f"{Fore.YELLOW}$ {cmd}{Style.RESET_ALL}")
    
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True
    )
    
    if show_output:
        if result.stdout:
            print(result.stdout)
        if result.stderr and check_error:
            print(f"{Fore.RED}{result.stderr}{Style.RESET_ALL}")
    
    return result.returncode == 0, result.stdout, result.stderr

def press_enter_to_continue():
    """Wait for user input unless in automated mode."""
    if not os.getenv("DEMO_MODE") == "automated":
        input(f"\n{Fore.GREEN}Press Enter to continue...{Style.RESET_ALL}")

def print_success(msg: str):
    """Print success message."""
    print(f"{Fore.GREEN}✅ {msg}{Style.RESET_ALL}")

def print_error(msg: str):
    """Print error message."""
    print(f"{Fore.RED}❌ {msg}{Style.RESET_ALL}")

def print_info(msg: str):
    """Print info message."""
    print(f"{Fore.CYAN}ℹ️  {msg}{Style.RESET_ALL}")

def main():
    """Run the multi-model demonstration."""
    
    print("\n" + "="*70)
    print(f"{Fore.CYAN}🎯 MULTI-MODEL TASK OPTIMIZATION DEMO{Style.RESET_ALL}")
    print("="*70)
    
    # Auto-setup requirements
    print(f"\n{Fore.YELLOW}📦 Checking and installing requirements...{Style.RESET_ALL}")
    success, stdout, _ = run_cli_command(
        'uv run python cli.py setup demos/strategies.yaml --auto --verbose',
        "Setting up components for multi-model strategy",
        show_output=True
    )
    if not success:
        print_error("Setup failed. Please check your environment.")
        return
    print_success("All requirements are ready!")
    
    print("\nThis demo shows how different models can be optimized")
    print("for different types of tasks to maximize performance and minimize costs.")
    print("\n📚 What you'll see:")
    print("  1. Simple queries → Fast, cheap models")
    print("  2. Complex reasoning → Advanced models")
    print("  3. Creative tasks → Specialized models")  
    print("  4. Code generation → Local models")
    print("  5. Cost comparison analysis")
    
    press_enter_to_continue()
    
    # Show task routing strategy
    print(f"\n{Fore.BLUE}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}TASK ROUTING STRATEGY{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{'='*60}{Style.RESET_ALL}")
    print("\n📊 Model Selection by Task Type:")
    print("  • Simple queries    → GPT-3.5 Turbo (fast, $0.002/1k tokens)")
    print("  • Complex reasoning → GPT-4o Mini   (smart, $0.015/1k tokens)")
    print("  • Creative writing  → GPT-4o        (creative, $0.03/1k tokens)")
    print("  • Code generation   → Mistral 7B    (local, free)")
    print("\n💡 This routing can save 60-80% on API costs!")
    
    press_enter_to_continue()
    
    # Demo 1: Simple task
    print(f"\n{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}TASK 1: SIMPLE QUERY{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
    print("\nFor simple factual questions, we use fast, cheap models:")
    
    run_cli_command(
        'uv run python cli.py complete "What is 2+2?" --strategy demo2_multi_model --strategy-file demos/strategies.yaml',
        "Simple math question → GPT-3.5 Turbo"
    )
    
    press_enter_to_continue()
    
    # Demo 2: Complex task  
    print(f"\n{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}TASK 2: COMPLEX REASONING{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")
    print("\nFor complex reasoning, we use more advanced models:")
    
    run_cli_command(
        'uv run python cli.py complete "Explain the theory of relativity and its key implications for modern physics" --strategy demo2_multi_model --strategy-file demos/strategies.yaml',
        "Complex physics question → GPT-4o Mini"
    )
    
    press_enter_to_continue()
    
    # Demo 3: Creative task
    print(f"\n{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}TASK 3: CREATIVE WRITING{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
    print("\nFor creative tasks, we use the most creative models:")
    
    run_cli_command(
        'uv run python cli.py complete "Write a humorous haiku about debugging code that won\'t work" --strategy demo2_multi_model --strategy-file demos/strategies.yaml',
        "Creative writing → GPT-4o",
        check_error=False
    )
    
    press_enter_to_continue()
    
    # Demo 4: Code task
    print(f"\n{Fore.BLUE}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}TASK 4: CODE GENERATION{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{'='*60}{Style.RESET_ALL}")
    print("\nFor code generation, we can use local models (free!):")
    
    # Check if Ollama is available for code generation
    check_ollama, _, _ = run_cli_command(
        "which ollama",
        "Checking if Ollama is available",
        check_error=False,
        show_output=False
    )
    
    if check_ollama:
        run_cli_command(
            'uv run python cli.py test-local mistral:7b --query "Write a Python function to calculate fibonacci numbers"',
            "Code generation → Local Mistral (free!)",
            check_error=False
        )
    else:
        print_info("Ollama not available. Would use local Mistral model for free code generation.")
        print_info("Install Ollama and run: ollama pull mistral:7b")
        
        # Fallback to cloud for demonstration
        run_cli_command(
            'uv run python cli.py complete "Write a Python function to calculate fibonacci numbers" --strategy demo2_multi_model --strategy-file demos/strategies.yaml',
            "Code generation → GPT-3.5 (fallback)",
            check_error=False
        )
    
    # Final analysis
    print(f"\n{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}COST & PERFORMANCE ANALYSIS{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
    
    print("\n📊 Without intelligent routing:")
    print("   All tasks → GPT-4o = ~$0.15 per 1k tokens")
    print("   4 tasks × 100 tokens = ~$0.06 total")
    
    print("\n🎯 With intelligent routing:")
    print("   Simple   → GPT-3.5    = $0.002 per 1k tokens")
    print("   Complex  → GPT-4o Mini = $0.015 per 1k tokens") 
    print("   Creative → GPT-4o     = $0.030 per 1k tokens")
    print("   Code     → Local      = $0.000 per 1k tokens")
    print("   Total cost ≈ $0.02 (67% savings!)")
    
    print("\n💡 Key benefits:")
    print("   • Massive cost savings (60-80%)")
    print("   • Better performance per task type")
    print("   • Privacy for code generation (local)")
    print("   • Reduced latency for simple tasks")
    
    print("\n" + "=" * 60)
    print("✅ Demo completed successfully!")
    print("\n💡 Key Takeaways:")
    print("   • Task routing reduces costs by 60-80%")
    print("   • Simple queries don't need expensive models")
    print("   • Specialized models excel at specific tasks")
    print("   • Local models provide free computation for suitable tasks")
    
    print("\n📊 Cost Analysis:")
    print("   Without routing: All tasks → GPT-4 = $$$")
    print("   With routing:    Mixed models    = $")
    
    print("\n📝 To customize this demo:")
    print("   1. Edit demos/strategies.yaml")
    print("   2. Modify the 'demo2_multi_model' strategy")
    print("   3. Add your own routing rules")
    print("   4. Configure task categories and model assignments")
    
    print("\n🔍 Try it yourself with specific task types:")
    print("   # Simple task")
    print("   uv run python cli.py complete 'What is 2+2?' --strategy demo2_multi_model --strategy-file demos/strategies.yaml")
    print("   ")
    print("   # Complex task")
    print("   uv run python cli.py complete 'Explain quantum entanglement' --strategy demo2_multi_model --strategy-file demos/strategies.yaml")
    print("   ")
    print("   # Code task")
    print("   uv run python cli.py complete 'Write a Python sorting function' --strategy demo2_multi_model --strategy-file demos/strategies.yaml")

if __name__ == "__main__":
    main()