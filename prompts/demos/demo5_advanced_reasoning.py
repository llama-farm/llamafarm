#!/usr/bin/env python3
"""
Demo 5: Advanced Reasoning and Evaluation

This demo shows:
- Chain of thought reasoning
- Quality evaluation with LLM-as-judge
- Strategy recommendation based on use case
- Performance profiling
- Creating custom strategies
"""

import sys
import os
from pathlib import Path
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from demo_helper import setup_demo_environment
from strategies import StrategyConfig
from strategies.config import TemplatesConfig, TemplateConfig


def run_demo():
    """Run the demo using the strategy manager."""
    print("=== Demo 5: Advanced Reasoning and Evaluation ===\n")
    
    # Initialize the strategy manager with templates
    manager = setup_demo_environment()
    
    # Example 1: Chain of thought reasoning
    print("Example 1: Chain of Thought Problem Solving")
    print("-" * 50)
    
    result = manager.execute_strategy(
        strategy_name="analytical_reasoning",
        inputs={
            "query": "A company's revenue grew by 20% in Q1, 15% in Q2, declined by 5% in Q3, and grew by 10% in Q4. If they started the year with $1M in monthly revenue, what was their total revenue for the year?",
            "context": [],
            "examples": []
        }
    )
    
    print("Generated Prompt:")
    print(result)
    print("\n")
    
    # Example 2: Mathematical problem solving
    print("Example 2: Mathematical Problem Solver")
    print("-" * 50)
    
    result = manager.execute_strategy(
        strategy_name="mathematical_solver",
        inputs={
            "query": "Prove that the sum of any two odd numbers is always even.",
            "context": ["Definition: An odd number can be written as 2n + 1 where n is an integer"],
            "examples": []
        }
    )
    
    print("Generated Prompt:")
    print(result)
    print("\n")
    
    # Example 3: Quality evaluation
    print("Example 3: Quality Evaluation (LLM-as-Judge)")
    print("-" * 50)
    
    content_to_evaluate = """
    Machine learning is a type of AI that allows computers to learn from data.
    It uses algorithms to identify patterns and make decisions.
    There are three main types: supervised, unsupervised, and reinforcement learning.
    """
    
    evaluation_criteria = {
        "accuracy": "Is the information factually correct?",
        "clarity": "Is the explanation clear and easy to understand?",
        "completeness": "Does it cover the essential aspects?",
        "relevance": "Is it relevant to understanding ML basics?"
    }
    
    result = manager.execute_strategy(
        strategy_name="quality_evaluator",
        inputs={
            "content": content_to_evaluate,
            "criteria": evaluation_criteria,
            "rubric": "Score each criterion from 1-10 and provide justification"
        }
    )
    
    print("Generated Evaluation Prompt:")
    print(result)
    print("\n")
    
    # Example 4: Comparison evaluation
    print("Example 4: Comparative Evaluation")
    print("-" * 50)
    
    result = manager.execute_strategy(
        strategy_name="quality_evaluator",
        inputs={
            "content": [
                {"id": "A", "text": "Python is great for beginners due to its simple syntax."},
                {"id": "B", "text": "Python's readability and extensive libraries make it ideal for newcomers to programming."}
            ],
            "criteria": {"clarity": 1.0, "completeness": 1.0},
            "rubric": "Compare both responses and determine which is better"
        },
        context={
            "query_type": "comparison"  # Triggers comparison mode
        }
    )
    
    print("Generated Comparison Prompt:")
    print(result)
    print("\n")
    
    # Example 5: Strategy recommendation
    print("Example 5: Strategy Recommendation")
    print("-" * 50)
    
    print("Finding best strategies for 'customer service' use case...")
    recommendations = manager.recommend_strategies(
        use_case="customer_service",
        performance="balanced",
        complexity="moderate"
    )
    
    print(f"Recommended strategies:")
    for i, strategy in enumerate(recommendations[:3], 1):
        print(f"{i}. {strategy.name}")
        print(f"   - Description: {strategy.description}")
        print(f"   - Performance: {strategy.performance_profile.value}")
        print(f"   - Complexity: {strategy.complexity.value}")
    print("\n")
    
    # Example 6: Creating a custom strategy
    print("Example 6: Creating a Custom Strategy")
    print("-" * 50)
    
    # Create a custom strategy for educational content
    custom_strategy = manager.create_strategy(
        name="Educational Assistant",
        description="Strategy for creating educational content with examples",
        default_template="qa_detailed",
        use_cases=["education", "teaching", "explanation"],
        performance_profile="balanced",
        complexity="moderate"
    )
    
    # Configure the strategy
    custom_strategy.templates.default.config = {
        "include_reasoning": True,
        "response_structure": "structured",
        "include_examples": True
    }
    
    # Add specialized template for exercises
    from strategies.config import SpecializedTemplate, ConditionConfig
    custom_strategy.templates.specialized.append(
        SpecializedTemplate(
            template="qa_detailed",
            condition=ConditionConfig(query_type="exercise"),
            config={
                "include_reasoning": True,
                "response_structure": "step_by_step",
                "include_hints": True
            },
            priority=10
        )
    )
    
    # Save the custom strategy
    # Save to a temporary file for demo purposes
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    manager.save_strategy("educational_assistant", custom_strategy, temp_file.name)
    print(f"✅ Custom strategy saved to: {temp_file.name}")
    temp_file.close()
    
    print(f"Created custom strategy: {custom_strategy.name}")
    print(f"Templates configured: {len(custom_strategy.templates.specialized) + 1}")
    print(f"Use cases: {', '.join(custom_strategy.use_cases)}")
    print("\n")
    
    # Example 7: Performance profiling
    print("Example 7: Performance Analysis")
    print("-" * 50)
    
    # Get execution statistics
    stats = manager.get_execution_stats()
    print("Execution Statistics:")
    for strategy_name, template_stats in stats.items():
        print(f"\n{strategy_name}:")
        for template_id, count in template_stats.items():
            print(f"  - {template_id}: {count} executions")
    
    # Get template usage across strategies
    print("\nTemplate Usage Analysis:")
    template_usage = manager.get_template_usage()
    for template_id, strategies in sorted(template_usage.items())[:5]:
        print(f"  - {template_id}: used by {len(strategies)} strategies")
    
    print("\n")
    
    # Teaching moment
    print("💡 Key Concepts:")
    print("- Chain of thought breaks complex problems into steps")
    print("- Mathematical solver uses very low temperature (0.3) for accuracy")
    print("- Quality evaluator can score content or compare alternatives")
    print("- Strategy recommendation helps find the right tool for the job")
    print("- Custom strategies can be created and saved for specific needs")
    print("- Performance profiling helps optimize strategy selection")
    print("- Different strategies optimize for different goals (speed vs accuracy)")
    
    # Advanced features
    print("\n🚀 Advanced Features:")
    print("- Strategies can have multiple specialized templates")
    print("- Selection rules use expressions for dynamic template choice")
    print("- Global configuration applies to all templates in a strategy")
    print("- Input/output transforms modify data automatically")
    print("- Strategies can be composed and extended")


def main():
    """Main entry point with CLI integration example."""
    print("\nThis demo can also be run via CLI:")
    print("  python -m prompts.cli_strategy strategy execute analytical_reasoning -q 'Solve this complex problem'")
    print("  python -m prompts.cli_strategy strategy create --name 'Custom Strategy' --description 'My custom strategy'")
    print("  python -m prompts.cli_strategy stats")
    print("\nRunning demo...\n")
    
    run_demo()


if __name__ == "__main__":
    main()