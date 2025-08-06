#!/usr/bin/env python3
"""
Demo 2: Customer Support with Dynamic Template Selection

This demo shows:
- Customer support strategy with specialized templates
- Automatic template selection based on query type
- Selection rules in action
- Input/output transformations
- Conversation history handling
"""

import sys
import os
from pathlib import Path
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from demo_helper import setup_demo_environment


def run_demo():
    """Run the demo using the strategy manager."""
    print("=== Demo 2: Customer Support Assistant ===\n")
    
    # Initialize the strategy manager with templates
    manager = setup_demo_environment()
    
    # Example 1: General support query
    print("Example 1: General Support Query")
    print("-" * 50)
    
    result = manager.execute_strategy(
        strategy_name="customer_support",
        inputs={
            "message": "I can't find my order confirmation email",
            "history": [],
            "context": []
        }
    )
    
    print("Generated Prompt:")
    print(result)
    print("\n")
    
    # Example 2: Technical query (triggers specialized template)
    print("Example 2: Technical Support Query")
    print("-" * 50)
    
    result = manager.execute_strategy(
        strategy_name="customer_support",
        inputs={
            "message": "I'm getting an error when trying to login: 'Invalid credentials'",
            "history": [
                {"role": "user", "content": "I forgot my password"},
                {"role": "assistant", "content": "I can help you reset your password. Please click the 'Forgot Password' link on the login page."}
            ],
            "context": []
        },
        context={
            "query_type": "technical"  # This triggers the specialized template
        }
    )
    
    print("Generated Prompt:")
    print(result)
    print("\n")
    
    # Example 3: Complaint handling (triggers different personality)
    print("Example 3: Complaint Handling")
    print("-" * 50)
    
    result = manager.execute_strategy(
        strategy_name="customer_support",
        inputs={
            "message": "This is the third time I've had to contact support about this issue!",
            "history": [],
            "context": []
        },
        context={
            "query_type": "complaint"
        }
    )
    
    print("Generated Prompt:")
    print(result)
    print("\n")
    
    # Example 4: Selection rule triggering
    print("Example 4: Automatic Technical Detection")
    print("-" * 50)
    
    # The word "error" triggers the technical detection rule
    result = manager.execute_strategy(
        strategy_name="customer_support",
        inputs={
            "query": "Why do I see an error message on checkout?",  # Using 'query' to trigger rule
            "message": "Why do I see an error message on checkout?",
            "history": [],
            "context": []
        }
    )
    
    print("Generated Prompt:")
    print(result)
    print("\n")
    
    # Example 5: Input transformations
    print("Example 5: Input Transformations")
    print("-" * 50)
    
    # Create a professional assistant strategy that uses transformations
    professional_result = manager.execute_strategy(
        strategy_name="professional_assistant",
        inputs={
            "message": "   HELLO, I NEED HELP WITH MY ACCOUNT!!!   ",  # Will be trimmed
            "history": [],
            "context": []
        }
    )
    
    print("Original message: '   HELLO, I NEED HELP WITH MY ACCOUNT!!!   '")
    print("After transformation (trimmed):")
    print(professional_result)
    print("\n")
    
    # Example 6: Strategy configuration details
    print("Example 6: Customer Support Strategy Configuration")
    print("-" * 50)
    
    strategy = manager.get_strategy("customer_support")
    print(f"Strategy: {strategy.name}")
    print(f"Templates configured: {len(strategy.templates.specialized) + 2} (default + fallback + specialized)")
    print(f"Selection rules: {len(strategy.selection_rules)}")
    print("\nSpecialized templates:")
    for spec in strategy.templates.specialized:
        print(f"  - Condition: {spec.condition.dict(exclude_none=True)}")
        print(f"    Template: {spec.template}")
        print(f"    Priority: {spec.priority}")
    print("\nSelection rules:")
    for rule in strategy.selection_rules:
        print(f"  - {rule.name}: {rule.condition}")
    print("\n")
    
    # Teaching moment
    print("💡 Key Concepts:")
    print("- Customer support strategy adapts based on query type")
    print("- Specialized templates handle technical queries and complaints differently")
    print("- Selection rules automatically detect patterns (e.g., 'error', 'bug')")
    print("- Input transforms clean and normalize user input")
    print("- System prompts ensure consistent, helpful tone")
    print("- Higher priority rules/templates override lower ones")
    
    # Show template usage
    print("\n📊 Template Usage:")
    usage = manager.get_template_usage()
    for template, strategies in usage.items():
        if "customer_support" in strategies:
            print(f"  - {template}: used by {', '.join(strategies)}")


def main():
    """Main entry point with CLI integration example."""
    print("\nThis demo can also be run via CLI:")
    print("  python -m prompts.cli_strategy demo --strategy customer_support --interactive")
    print("  python -m prompts.cli_strategy strategy execute customer_support -q 'I need help with my order'")
    print("  python -m prompts.cli_strategy strategy show customer_support --show-templates")
    print("\nRunning demo...\n")
    
    run_demo()


if __name__ == "__main__":
    main()