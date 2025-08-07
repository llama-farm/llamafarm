#!/usr/bin/env python3
"""
Demo 1: Simple Question Answering

This demo shows:
- Basic Q&A with the simple_qa strategy
- How strategies automatically select templates
- Working with context documents
- Temperature and token configuration
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
    print("=== Demo 1: Simple Question Answering ===\n")
    
    # Initialize the strategy manager with templates
    manager = setup_demo_environment()
    
    # Example 1: Basic Q&A without context
    print("Example 1: Simple question without context")
    print("-" * 50)
    
    result = manager.execute_strategy(
        strategy_name="simple_qa",
        inputs={
            "query": "What is machine learning?",
            "context": []
        }
    )
    
    print("Generated Prompt:")
    print(result)
    print("\n")
    
    # Example 2: Q&A with context documents
    print("Example 2: Question with context documents")
    print("-" * 50)
    
    context_docs = [
        {
            "title": "Introduction to ML",
            "content": "Machine learning is a subset of artificial intelligence (AI) that enables systems to learn and improve from experience without being explicitly programmed."
        },
        {
            "title": "Types of ML",
            "content": "There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning."
        }
    ]
    
    result = manager.execute_strategy(
        strategy_name="simple_qa",
        inputs={
            "query": "What are the main types of machine learning?",
            "context": context_docs
        }
    )
    
    print("Generated Prompt:")
    print(result)
    print("\n")
    
    # Example 3: Using configuration overrides
    print("Example 3: Custom configuration")
    print("-" * 50)
    
    result = manager.execute_strategy(
        strategy_name="simple_qa",
        inputs={
            "query": "Explain neural networks in simple terms",
            "context": []
        },
        override_config={
            "temperature_hint": 0.9,  # Higher temperature for more creative explanation
            "max_tokens_hint": 200,   # Shorter response
            "context_format": "bullet"  # Use bullet format
        }
    )
    
    print("Generated Prompt:")
    print(result)
    print("\n")
    
    # Example 4: Strategy statistics
    print("Example 4: Execution Statistics")
    print("-" * 50)
    
    stats = manager.get_execution_stats()
    print("Execution stats:", json.dumps(stats, indent=2))
    print("\n")
    
    # Example 5: Strategy details
    print("Example 5: Strategy Configuration")
    print("-" * 50)
    
    strategy = manager.get_strategy("simple_qa")
    print(f"Strategy: {strategy.name}")
    print(f"Description: {strategy.description}")
    print(f"Use cases: {', '.join(strategy.use_cases)}")
    print(f"Performance profile: {strategy.performance_profile.value}")
    print(f"Default template: {strategy.templates.default.template}")
    print(f"Global temperature: {strategy.global_config.temperature}")
    print(f"Global max tokens: {strategy.global_config.max_tokens}")
    print("\n")
    
    # Teaching moment
    print("💡 Key Concepts:")
    print("- Strategies encapsulate prompt templates and configuration")
    print("- The simple_qa strategy uses 'qa_basic' template by default")
    print("- Context documents are automatically formatted")
    print("- Configuration can be overridden per execution")
    print("- Global settings apply to all templates in the strategy")


def main():
    """Main entry point with CLI integration example."""
    print("\nThis demo can also be run via CLI:")
    print("  python -m prompts.cli_strategy demo --strategy simple_qa")
    print("  python -m prompts.cli_strategy strategy execute simple_qa -q 'What is machine learning?'")
    print("\nRunning demo...\n")
    
    run_demo()


if __name__ == "__main__":
    main()