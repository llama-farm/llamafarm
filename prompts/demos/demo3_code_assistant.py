#!/usr/bin/env python3
"""
Demo 3: Code Assistant with Analysis and Generation

This demo shows:
- Code analysis and review
- Different analysis depths based on query type
- Output transformations for code extraction
- Working with programming languages
- Debugging vs optimization workflows
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
    print("=== Demo 3: Code Assistant ===\n")
    
    # Initialize the strategy manager with templates
    manager = setup_demo_environment()
    
    # Example 1: Basic code review
    print("Example 1: Basic Code Review")
    print("-" * 50)
    
    code_snippet = """
def calculate_average(numbers):
    sum = 0
    for num in numbers:
        sum += num
    return sum / len(numbers)
"""
    
    result = manager.execute_strategy(
        strategy_name="code_assistant",
        inputs={
            "code": code_snippet,
            "language": "python",
            "analysis_type": "review"
        }
    )
    
    print("Generated Prompt:")
    print(result)
    print("\n")
    
    # Example 2: Debug mode (triggers comprehensive analysis)
    print("Example 2: Debug Mode - Comprehensive Analysis")
    print("-" * 50)
    
    buggy_code = """
def find_max(lst):
    max_val = lst[0]
    for i in range(len(lst)):
        if lst[i] > max_val:
            max_val = i  # Bug: storing index instead of value
    return max_val
"""
    
    result = manager.execute_strategy(
        strategy_name="code_assistant",
        inputs={
            "code": buggy_code,
            "language": "python",
            "analysis_type": "debug"
        },
        context={
            "query_type": "debug"  # Triggers specialized template
        }
    )
    
    print("Generated Prompt:")
    print(result)
    print("\n")
    
    # Example 3: Performance optimization
    print("Example 3: Performance Optimization")
    print("-" * 50)
    
    slow_code = """
def find_duplicates(arr):
    duplicates = []
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] == arr[j] and arr[i] not in duplicates:
                duplicates.append(arr[i])
    return duplicates
"""
    
    result = manager.execute_strategy(
        strategy_name="code_assistant",
        inputs={
            "code": slow_code,
            "language": "python",
            "analysis_type": "optimize"
        },
        context={
            "query_type": "optimize"  # Triggers performance-focused template
        }
    )
    
    print("Generated Prompt:")
    print(result)
    print("\n")
    
    # Example 4: Code generation request
    print("Example 4: Code Generation")
    print("-" * 50)
    
    generation_result = manager.execute_strategy(
        strategy_name="code_assistant",
        inputs={
            "code": "",  # Empty for generation
            "language": "javascript",
            "analysis_type": "generate",
            "requirements": "Create a function that validates email addresses using regex"
        }
    )
    
    print("Generated Prompt:")
    print(generation_result)
    print("\n")
    
    # Example 5: Multi-language support
    print("Example 5: Multi-Language Analysis")
    print("-" * 50)
    
    java_code = """
public class Calculator {
    public double divide(double a, double b) {
        return a / b;  // Potential division by zero
    }
}
"""
    
    result = manager.execute_strategy(
        strategy_name="code_assistant",
        inputs={
            "code": java_code,
            "language": "java",
            "analysis_type": "review"
        }
    )
    
    print("Generated Prompt:")
    print(result)
    print("\n")
    
    # Example 6: Custom configuration
    print("Example 6: Custom Analysis Configuration")
    print("-" * 50)
    
    result = manager.execute_strategy(
        strategy_name="code_assistant",
        inputs={
            "code": code_snippet,
            "language": "python",
            "analysis_type": "security"
        },
        override_config={
            "analysis_depth": "comprehensive",
            "check_patterns": ["security", "input_validation", "sql_injection"],
            "output_format": "json"  # Request JSON output
        }
    )
    
    print("Generated Prompt:")
    print(result)
    print("\n")
    
    # Show strategy configuration
    print("Strategy Configuration:")
    print("-" * 50)
    
    strategy = manager.get_strategy("code_assistant")
    print(f"Strategy: {strategy.name}")
    print(f"Analysis depth: {strategy.templates.default.config.get('analysis_depth', 'standard')}")
    print(f"Temperature: {strategy.global_config.temperature} (low for accuracy)")
    output_transforms = [t.transform for t in strategy.output_transforms] if strategy.output_transforms else []
    print(f"Output transforms: {output_transforms if output_transforms else 'none'}")
    
    print("\nSpecialized configurations:")
    for spec in strategy.templates.specialized:
        print(f"  - {spec.template}: focus on {spec.config.get('focus', 'general')}")
    
    print("\n")
    
    # Teaching moment
    print("💡 Key Concepts:")
    print("- Code assistant adapts analysis based on the task (debug, optimize, review)")
    print("- Low temperature (0.3) ensures accurate, consistent code analysis")
    print("- Output transform extracts code blocks from responses")
    print("- Different check patterns for different analysis types")
    print("- Supports multiple programming languages")
    print("- System prompts ensure best practices and clear documentation")
    
    # Demonstrate output transformation
    print("\n🔄 Output Transformation Example:")
    print("If the LLM response contains code blocks like:")
    print("```python")
    print("def improved_function():")
    print("    pass")
    print("```")
    print("The extract_code transform will extract just the code.")


def main():
    """Main entry point with CLI integration example."""
    print("\nThis demo can also be run via CLI:")
    print("  python -m prompts.cli_strategy strategy execute code_assistant -q 'Review this Python code' -c '{\"code\": \"def add(a, b): return a + b\"}'")
    print("  python -m prompts.cli_strategy strategy recommend --use-case coding --performance accuracy")
    print("\nRunning demo...\n")
    
    run_demo()


if __name__ == "__main__":
    main()