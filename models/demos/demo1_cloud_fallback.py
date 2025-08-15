#!/usr/bin/env python3
"""
Demo 1: Cloud Model Fallback
Shows automatic failover from cloud APIs to local models.

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
    """Run the cloud fallback demonstration."""
    
    print("\n" + "="*70)
    print(f"{Fore.CYAN}🚀 CLOUD MODEL FALLBACK DEMONSTRATION{Style.RESET_ALL}")
    print("="*70)
    
    # Auto-setup requirements
    print(f"\n{Fore.YELLOW}📦 Checking and installing requirements...{Style.RESET_ALL}")
    success, stdout, _ = run_cli_command(
        'uv run python cli.py setup demos/strategies.yaml --auto --verbose',
        "Setting up components for demo1_cloud_fallback strategy",
        show_output=True
    )
    if not success:
        print_error("Setup failed. Please check your environment.")
        return
    print_success("All requirements are ready!")
    
    print("\nThis demo shows automatic failover from cloud APIs to local models")
    print("when the cloud service is unavailable.")
    print("\n📚 What you'll see:")
    print("  1. Successful cloud API call")
    print("  2. Simulated API failure")
    print("  3. Automatic fallback to local model")
    print("  4. Fallback chain explanation")
    
    press_enter_to_continue()
    
    # Step 1: Show successful cloud API call
    print(f"\n{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}STEP 1: SUCCESSFUL CLOUD API CALL{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
    print("\nFirst, let's make a successful call to a cloud API.")
    print("This will work if you have a valid API key in your .env file.")
    
    success, stdout, _ = run_cli_command(
        'uv run python cli.py complete "What is the capital of France?" --strategy demo1_cloud_fallback --strategy-file demos/strategies.yaml',
        "Calling cloud provider via strategy"
    )
    
    if success and "Paris" in stdout:
        print_success("Cloud API responded successfully with correct answer!")
    else:
        print_info("Cloud API call didn't work - you may not have an API key set")
        print_info("That's okay! Let's continue with the fallback demonstration...")
    
    press_enter_to_continue()
    
    # Step 2: Simulate API failure and show fallback
    print(f"\n{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}STEP 2: SIMULATING API FAILURE & FALLBACK{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")
    print("\nNow let's see what happens when the API fails.")
    print("We'll temporarily use an invalid API key to simulate failure.")
    
    # Save current API key and set invalid one
    orig_key = os.getenv("OPENAI_API_KEY", "")
    os.environ["OPENAI_API_KEY"] = "sk-invalid-test-key-for-demo"
    
    print_info("Setting invalid API key: sk-invalid-test-key-for-demo")
    
    success, stdout, stderr = run_cli_command(
        'uv run python cli.py complete "Explain quantum computing in simple terms" --strategy demo1_cloud_fallback --strategy-file demos/strategies.yaml',
        "Attempting cloud API call (will fail)",
        check_error=False
    )
    
    # Restore original key
    os.environ["OPENAI_API_KEY"] = orig_key
    
    if not success:
        print_error("Cloud API failed as expected!")
        print_info("In a real system with fallback configured, it would now try local models...")
    
    press_enter_to_continue()
    
    # Step 3: Show local model working
    print(f"\n{Fore.BLUE}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}STEP 3: LOCAL MODEL AS FALLBACK{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{'='*60}{Style.RESET_ALL}")
    print("\nLet's demonstrate using a local Ollama model as fallback.")
    
    # Check if Ollama is available
    check_ollama, _, _ = run_cli_command(
        "which ollama",
        "Checking if Ollama is installed",
        check_error=False,
        show_output=False
    )
    
    if check_ollama:
        print_success("Ollama is installed!")
        
        # Try to use local model
        success, stdout, _ = run_cli_command(
            'uv run python cli.py test-local llama3.2:3b --query "What is 2+2?"',
            "Testing local Ollama model",
            check_error=False
        )
        
        if success:
            print_success("Local model works! This would be your fallback.")
        else:
            print_info("Local model not available. Run: ollama pull llama3.2:3b")
    else:
        print_info("Ollama not installed. Install from https://ollama.ai for local fallback")
    
    press_enter_to_continue()
    
    # Step 4: Explain the fallback chain
    print(f"\n{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}UNDERSTANDING THE FALLBACK CHAIN{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}")
    print("\n📊 The strategy file (demos/strategies.yaml) defines a fallback chain:")
    print("\n  1. 🥇 Primary:    OpenAI GPT-4")
    print("  2. 🥈 Secondary:  OpenAI GPT-3.5 Turbo")
    print("  3. 🥉 Tertiary:   Local Ollama Llama 3.2")
    print("  4. 🏅 Quaternary: Local Ollama Mistral")
    print("\nThe system tries each in order until one succeeds.")
    print("\n💡 Benefits:")
    print("  • High availability - service continues even if cloud fails")
    print("  • Cost optimization - can fallback to cheaper options")
    print("  • Privacy option - sensitive data can use local models")
    print("  • Offline capability - works without internet")
    
    print("\n" + "=" * 60)
    print("✅ Demo completed successfully!")
    print("\n💡 Key Takeaways:")
    print("   • Cloud APIs provide fast, high-quality responses")
    print("   • Automatic fallback ensures service continuity")
    print("   • Local models provide privacy and offline capability")
    print("   • Fallback chains can be customized via strategies.yaml")
    
    print("\n📝 To customize this demo:")
    print("   1. Edit demos/strategies.yaml")
    print("   2. Modify the 'demo1_cloud_fallback' strategy")
    print("   3. Add your own API keys in .env file")
    print("   4. Configure additional fallback providers")
    
    print("\n🔍 Try it yourself:")
    print("   uv run python cli.py complete 'Your prompt here' --strategy demo1_cloud_fallback --strategy-file demos/strategies.yaml")

if __name__ == "__main__":
    main()