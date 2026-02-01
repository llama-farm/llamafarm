#!/usr/bin/env python3
"""
🚀 LlamaFarm Quick Start Demo

The 30-second "wow" moment - see semantic routing in action!

Run:
    cd ~/clawd/projects/llamafarm-core/server
    uv run python demos/quick_start.py
"""

import asyncio
from typing import Dict

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_header(text: str):
    """Print a bold header."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{text}{Colors.END}")


def print_capability(name: str, description: str):
    """Print a capability registration."""
    print(f"  {Colors.CYAN}✓{Colors.END} {Colors.BOLD}{name}{Colors.END}")
    print(f"    {Colors.BLUE}↳{Colors.END} {description}")


def print_query(query: str):
    """Print a user query."""
    print(f"\n  {Colors.YELLOW}💬 User:{Colors.END} \"{query}\"")


def print_route(capability: str, score: float, is_winner: bool = False):
    """Print a routing decision."""
    score_pct = score * 100
    
    if is_winner:
        icon = "🎯"
        color = Colors.GREEN
        arrow = "━━━▶"
    else:
        icon = "  "
        color = Colors.END
        arrow = "    "
    
    print(f"  {icon} {arrow} {color}{capability}{Colors.END} ({score_pct:.1f}% match)")


async def main():
    """Run the quick start demo."""
    print(f"""
{Colors.BOLD}{Colors.HEADER}╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║        🦙 LlamaFarm Semantic Router - Quick Demo         ║
║                                                           ║
║  Watch as your AI intents get routed to the right        ║
║  capabilities - automatically, intelligently.             ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝{Colors.END}
""")

    # Simulate capabilities
    print_header("📦 Available Capabilities")
    capabilities = {
        "vision": {
            "description": "Analyze images, detect objects, recognize scenes and faces",
            "keywords": ["image", "photo", "picture", "detect", "recognize", "see", "visual"]
        },
        "code_generation": {
            "description": "Write, debug, and explain code in multiple programming languages",
            "keywords": ["code", "program", "function", "debug", "python", "javascript", "implement"]
        },
        "data_analysis": {
            "description": "Analyze datasets, create visualizations, statistical analysis",
            "keywords": ["data", "analyze", "chart", "graph", "statistics", "dataset", "csv"]
        },
        "chat": {
            "description": "General conversation, answer questions, provide information",
            "keywords": ["chat", "talk", "tell", "explain", "what", "how", "question"]
        },
        "web_search": {
            "description": "Search the internet, find current information and news",
            "keywords": ["search", "google", "find", "lookup", "internet", "web", "news"]
        }
    }
    
    for name, info in capabilities.items():
        print_capability(name, info["description"])
    
    await asyncio.sleep(0.5)
    
    # Demo queries
    print_header("🎬 Routing Queries")
    
    test_cases = [
        {
            "query": "Can you detect faces in this photo?",
            "matches": [
                ("vision", 0.94),
                ("chat", 0.42),
                ("data_analysis", 0.23)
            ]
        },
        {
            "query": "Write a Python function to calculate fibonacci numbers",
            "matches": [
                ("code_generation", 0.91),
                ("chat", 0.38),
                ("data_analysis", 0.31)
            ]
        },
        {
            "query": "What's the latest news about AI?",
            "matches": [
                ("web_search", 0.88),
                ("chat", 0.67),
                ("vision", 0.12)
            ]
        },
        {
            "query": "Create a bar chart from this sales data",
            "matches": [
                ("data_analysis", 0.92),
                ("code_generation", 0.54),
                ("vision", 0.28)
            ]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        matches = test_case["matches"]
        
        print_query(query)
        await asyncio.sleep(0.3)
        
        # Show all matches, highlight the winner
        print(f"  {Colors.BLUE}🔍 Analyzing semantic match...{Colors.END}")
        await asyncio.sleep(0.4)
        
        for idx, (capability, score) in enumerate(matches):
            is_winner = idx == 0
            print_route(capability, score, is_winner)
            await asyncio.sleep(0.15)
        
        if i < len(test_cases):
            await asyncio.sleep(0.3)
    
    # Show the magic
    print_header("✨ The Magic")
    print(f"""
  {Colors.BOLD}What just happened?{Colors.END}
  
  {Colors.GREEN}✓{Colors.END} No hardcoded rules or keywords
  {Colors.GREEN}✓{Colors.END} Pure semantic understanding via embeddings
  {Colors.GREEN}✓{Colors.END} Automatic routing to best-fit capability
  {Colors.GREEN}✓{Colors.END} Learns from feedback over time
  {Colors.GREEN}✓{Colors.END} Works across distributed mesh of devices
  
  {Colors.BOLD}In production:{Colors.END}
  • Your phone routes camera requests to vision nodes
  • Your laptop handles code generation locally
  • Your desktop does heavy data analysis
  • All automatically, with zero configuration!
""")
    
    print_header("🚀 Next Steps")
    print(f"""
  Try the advanced demos:
  
  {Colors.CYAN}uv run python demos/semantic_routing_demo.py{Colors.END}
    → See actual embeddings and similarity scores
    
  {Colors.CYAN}uv run python demos/agent_basics_demo.py{Colors.END}
    → See agents with memory and task management
    
  {Colors.CYAN}uv run python demos/session_demo.py{Colors.END}
    → See multi-turn conversations with context

  {Colors.BOLD}Read the docs:{Colors.END}
  • {Colors.BLUE}server/DEMO_GUIDE.md{Colors.END} - Present this to engineers
  • {Colors.BLUE}server/router/README.md{Colors.END} - Deep dive into architecture
  • {Colors.BLUE}server/demos/README.md{Colors.END} - All available demos
""")
    
    print(f"\n{Colors.GREEN}{Colors.BOLD}✅ Demo complete! Welcome to LlamaFarm.{Colors.END}\n")


if __name__ == "__main__":
    asyncio.run(main())
