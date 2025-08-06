#!/usr/bin/env python3
"""
Master Demo Runner

This script runs all prompt system demos in sequence, showing the full
capabilities of the LlamaFarm prompt management system.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import all demos
from demo1_simple_qa import run_demo as demo1
from demo2_customer_support import run_demo as demo2
from demo3_code_assistant import run_demo as demo3
from demo4_rag_research import run_demo as demo4
from demo5_advanced_reasoning import run_demo as demo5


def print_banner(title):
    """Print a demo banner."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def main():
    """Run all demos."""
    print_banner("LlamaFarm Prompt System - Master Demo")
    
    print("""
Welcome to the LlamaFarm Prompt System demonstration!

This system provides:
✅ Strategy-based prompt management
✅ Dynamic template selection
✅ Input/output transformations
✅ Framework integration (LangChain, native, etc.)
✅ Performance optimization
✅ Extensible architecture

Press Enter to start the demos...
    """)
    input()
    
    demos = [
        ("Demo 1: Simple Question Answering", demo1),
        ("Demo 2: Customer Support Assistant", demo2),
        ("Demo 3: Code Assistant", demo3),
        ("Demo 4: RAG-Enhanced Research", demo4),
        ("Demo 5: Advanced Reasoning & Evaluation", demo5)
    ]
    
    for i, (title, demo_func) in enumerate(demos, 1):
        print_banner(f"{i}/5: {title}")
        
        try:
            demo_func()
        except Exception as e:
            print(f"⚠️  Error in demo: {e}")
            print("Continuing to next demo...")
        
        if i < len(demos):
            print("\n" + "─" * 50)
            print("Press Enter to continue to the next demo...")
            input()
    
    # Summary
    print_banner("Demo Summary")
    
    print("""
🎉 Demo Complete!

You've seen how the LlamaFarm Prompt System provides:

1. **Simple Q&A** - Basic prompting with context support
2. **Customer Support** - Dynamic template selection based on query type
3. **Code Assistant** - Specialized prompts for different coding tasks
4. **RAG Research** - Integration with retrieval systems and citations
5. **Advanced Reasoning** - Complex problem-solving and evaluation

Key Features Demonstrated:
- 📋 Strategy-based configuration
- 🔄 Automatic template selection
- 🔧 Input/output transformations
- 📊 Performance optimization
- 🎯 Use case specialization
- 🔌 Framework integration
- 📈 Execution analytics

Next Steps:
1. Explore the available strategies: `prompts/default_strategies.yaml`
2. Create custom templates: `prompts/templates/`
3. Design your own strategies for specific use cases
4. Integrate with your LLM framework of choice
5. Monitor and optimize performance

For more information:
- Documentation: `prompts/README.md`
- Schema reference: `prompts/schema.yaml`
- Template examples: `prompts/templates/`
- CLI usage: `python -m prompts.cli_strategy --help`

CLI Examples:
  # List all strategies
  python -m prompts.cli_strategy strategy list
  
  # Show strategy details
  python -m prompts.cli_strategy strategy show simple_qa
  
  # Execute a strategy
  python -m prompts.cli_strategy strategy execute simple_qa -q 'Your question here'
  
  # Get strategy recommendations
  python -m prompts.cli_strategy strategy recommend --use-case 'customer_service'
  
  # Run interactive demo
  python -m prompts.cli_strategy demo --interactive
    """)


if __name__ == "__main__":
    main()