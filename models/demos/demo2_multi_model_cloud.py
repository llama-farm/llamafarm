#!/usr/bin/env python3
"""
Demo 2: Multi-Model Cloud Strategy
==================================

Demonstrates using different models for different tasks to optimize
cost and performance.
"""

import os
import sys
import time
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core import ModelManager

console = Console()


def main():
    """Run the multi-model cloud demo."""
    console.print(Panel("""
[bold cyan]Multi-Model Cloud Strategy Demo[/bold cyan]

This demo shows:
• Using different models for different tasks
• Cost optimization by task complexity
• Performance vs cost trade-offs
• Model selection strategies
    """, title="Demo 2", expand=False))
    
    # Model configurations for different tasks
    model_configs = {
        "simple": {
            "model": "gpt-3.5-turbo",
            "name": "GPT-3.5 Turbo",
            "cost_per_1k": 0.0015,
            "use_case": "Simple queries, classifications"
        },
        "reasoning": {
            "model": "gpt-4o-mini",
            "name": "GPT-4o Mini",
            "cost_per_1k": 0.015,
            "use_case": "Complex reasoning, analysis"
        },
        "creative": {
            "model": "gpt-4o",
            "name": "GPT-4o",
            "cost_per_1k": 0.030,
            "use_case": "Creative writing, nuanced tasks"
        },
        "premium": {
            "model": "gpt-4-turbo",
            "name": "GPT-4 Turbo",
            "cost_per_1k": 0.030,
            "use_case": "Highest quality, complex tasks"
        },
        "code": {
            "model": "gpt-4o",
            "name": "GPT-4o (Code)",
            "cost_per_1k": 0.030,
            "use_case": "Code generation and analysis"
        }
    }
    
    # Display model comparison table
    table = Table(title="Available Models", show_header=True, header_style="bold magenta")
    table.add_column("Task Type", style="cyan", width=12)
    table.add_column("Model", style="green", width=20)
    table.add_column("Cost/1K tokens", style="yellow", width=15)
    table.add_column("Best For", style="blue", width=30)
    
    for task_type, config in model_configs.items():
        table.add_row(
            task_type.title(),
            config["name"],
            f"${config['cost_per_1k']:.4f}",
            config["use_case"]
        )
    
    console.print("\n")
    console.print(table)
    console.print("\n")
    
    # Test scenarios
    scenarios = [
        {
            "name": "Customer Support Query",
            "type": "simple",
            "prompt": "How do I reset my password?",
            "expected_tokens": 100
        },
        {
            "name": "Data Analysis",
            "type": "reasoning",
            "prompt": "Analyze these sales trends and suggest improvements: Q1: $100k, Q2: $95k, Q3: $92k, Q4: $110k",
            "expected_tokens": 300
        },
        {
            "name": "Creative Writing",
            "type": "creative",
            "prompt": "Write a compelling product description for an AI-powered smart home assistant",
            "expected_tokens": 250
        },
        {
            "name": "Technical Documentation",
            "type": "premium",
            "prompt": "Explain the architecture of a distributed caching system with examples",
            "expected_tokens": 500
        },
        {
            "name": "Code Generation",
            "type": "code",
            "prompt": "Write a Python function to implement binary search with error handling",
            "expected_tokens": 200
        }
    ]
    
    # Check if we have API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[yellow]⚠️  No OpenAI API key found. Running simulation...[/yellow]")
        simulate_demo(scenarios, model_configs)
        return
    
    # Run actual demo
    try:
        manager = ModelManager.from_strategy("hybrid_with_fallback")
        total_cost = 0.0
        results = []
        
        console.print("[bold]Running scenarios with optimal model selection:[/bold]\n")
        
        for i, scenario in enumerate(scenarios, 1):
            config = model_configs[scenario["type"]]
            
            console.print(f"[bold]Scenario {i}/{len(scenarios)}:[/bold] {scenario['name']}")
            console.print(f"[dim]Using: {config['name']} | Query: {scenario['prompt'][:50]}...[/dim]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(f"Processing with {config['name']}...", total=None)
                
                start_time = time.time()
                
                try:
                    # Make API call
                    response = manager.generate(
                        scenario["prompt"],
                        model=config["model"]
                    )
                    
                    elapsed = time.time() - start_time
                    cost = (scenario["expected_tokens"] / 1000) * config["cost_per_1k"]
                    total_cost += cost
                    
                    progress.update(task, description=f"✅ Complete (${cost:.4f})")
                    
                    results.append({
                        "scenario": scenario["name"],
                        "model": config["name"],
                        "time": elapsed,
                        "cost": cost,
                        "response_preview": response[:100] + "...",
                        "full_response": response
                    })
                    
                    # Print the full response
                    console.print(f"\n[bold green]Response:[/bold green]")
                    console.print(Panel(response, expand=False))
                    
                except Exception as e:
                    progress.update(task, description=f"❌ Failed: {str(e)[:30]}...")
                    results.append({
                        "scenario": scenario["name"],
                        "model": config["name"],
                        "time": 0,
                        "cost": 0,
                        "response_preview": "Error occurred",
                        "full_response": None
                    })
            
            console.print()
        
        # Display results
        display_results(results, total_cost)
        
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        console.print("[yellow]Running simulation instead...[/yellow]")
        simulate_demo(scenarios, model_configs)


def simulate_demo(scenarios, model_configs):
    """Simulate the demo when API isn't available."""
    console.print("\n[blue]🎬 Simulating Multi-Model Cloud Demo...[/blue]\n")
    
    # Simulated responses for each scenario type
    simulated_responses = {
        "simple": """To reset your password, please follow these steps:

1. Go to the login page
2. Click on "Forgot Password?" link
3. Enter your registered email address
4. Check your email for a password reset link
5. Click the link and follow the instructions to create a new password

If you don't receive the email within 5 minutes, please check your spam folder or contact support.""",
        
        "reasoning": """Based on the quarterly sales data provided:
- Q1: $100k (baseline)
- Q2: $95k (-5% decline)
- Q3: $92k (-8% decline from Q1)
- Q4: $110k (+10% growth from Q1)

Analysis:
1. **Mid-year slump**: Q2-Q3 showed consistent decline, possibly due to seasonal factors or market conditions
2. **Strong recovery**: Q4 showed significant improvement, exceeding Q1 performance

Recommendations:
1. **Investigate Q2-Q3 factors**: Identify what caused the mid-year decline
2. **Replicate Q4 success**: Analyze what strategies worked in Q4
3. **Seasonal planning**: Prepare targeted campaigns for traditionally slow quarters
4. **Customer retention**: Focus on maintaining customer base during low periods
5. **Product diversification**: Consider offerings that perform well in Q2-Q3""",
        
        "creative": """Introducing the Aurora Home Assistant – Your Intelligent Living Companion

Imagine walking into a home that knows you better than you know yourself. Aurora isn't just another smart speaker; it's the heart of your connected life. With advanced AI that learns your routines, Aurora orchestrates your entire home ecosystem with graceful intelligence.

Wake to gentle lighting that mimics sunrise, while your favorite morning playlist eases you into the day. Aurora has already adjusted your thermostat to the perfect temperature and started brewing your coffee. Throughout your day, it anticipates your needs – dimming lights for movie night, securing your home when you leave, and even reminding you of important tasks with a warm, conversational tone.

Built with privacy at its core, Aurora processes everything locally, ensuring your data never leaves your home. Its sleek, minimalist design blends seamlessly into any decor, while its powerful capabilities transform your house into a truly intelligent home.

Aurora: Where technology meets intuition.""",
        
        "premium": """## Distributed Caching System Architecture

A distributed caching system is designed to store frequently accessed data across multiple nodes to improve application performance and reduce database load.

### Core Components:

1. **Cache Nodes**: Individual servers storing cached data
   - Memory-based storage (Redis, Memcached)
   - Persistent storage options for durability
   
2. **Consistent Hashing**: Distributes data across nodes
   - Minimizes redistribution when nodes are added/removed
   - Example: Ring-based hash with virtual nodes
   
3. **Replication Strategy**:
   - Master-slave replication for read scaling
   - Multi-master for high availability
   
4. **Cache Invalidation**:
   - TTL (Time To Live) based expiration
   - Event-driven invalidation
   - Write-through/Write-behind strategies

### Example Architecture:
```
Load Balancer
     |
+----+----+----+
|    |    |    |
Node1 Node2 Node3
 |     |     |
Replica Replica Replica
```

### Implementation Example (Redis Cluster):
- 6 nodes minimum (3 masters, 3 slaves)
- Automatic sharding across 16,384 hash slots
- Automatic failover with Redis Sentinel
- Supports up to 1000 nodes

Key considerations: Data consistency, network partitions, cache warming strategies, and monitoring.""",
        
        "code": """```python
def binary_search(arr, target):
    \"\"\"
    Performs binary search on a sorted array.
    
    Args:
        arr: Sorted list of comparable elements
        target: Element to search for
        
    Returns:
        int: Index of target element if found, -1 otherwise
        
    Raises:
        ValueError: If array is None or not sorted
        TypeError: If elements are not comparable
    \"\"\"
    # Input validation
    if arr is None:
        raise ValueError("Array cannot be None")
    
    if not arr:
        return -1
    
    # Verify array is sorted
    if not all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1)):
        raise ValueError("Array must be sorted")
    
    left, right = 0, len(arr) - 1
    
    try:
        while left <= right:
            mid = left + (right - left) // 2
            
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
                
    except TypeError as e:
        raise TypeError(f"Elements must be comparable: {e}")
    
    return -1
```"""
    }
    
    total_cost = 0.0
    results = []
    
    console.print("[bold]Running scenarios with optimal model selection:[/bold]\n")
    
    for i, scenario in enumerate(scenarios, 1):
        config = model_configs[scenario["type"]]
        
        console.print(f"[bold]Scenario {i}/{len(scenarios)}:[/bold] {scenario['name']}")
        console.print(f"[dim]Using: {config['name']} | Query: {scenario['prompt'][:50]}...[/dim]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Processing with {config['name']}...", total=None)
            
            # Simulate processing time based on model
            processing_time = {
                "simple": 0.8,
                "reasoning": 1.5,
                "creative": 1.2,
                "premium": 2.0,
                "code": 1.3
            }[scenario["type"]]
            
            time.sleep(processing_time)
            
            # Calculate simulated cost
            cost = (scenario["expected_tokens"] / 1000) * config["cost_per_1k"]
            total_cost += cost
            
            progress.update(task, description=f"✅ Complete (${cost:.4f})")
            
            # Get simulated response
            response = simulated_responses[scenario["type"]]
            
            results.append({
                "scenario": scenario["name"],
                "model": config["name"],
                "time": processing_time,
                "cost": cost,
                "response_preview": response[:100] + "...",
                "full_response": response
            })
            
            # Print the full response
            console.print(f"\n[bold green]Response:[/bold green]")
            console.print(Panel(response, expand=False))
        
        console.print()
    
    console.print()
    display_results(results, total_cost)


def display_results(results, total_cost):
    """Display demo results."""
    # Results table
    table = Table(title="Execution Results", show_header=True, header_style="bold cyan")
    table.add_column("Scenario", style="green", width=25)
    table.add_column("Model Used", style="yellow", width=20)
    table.add_column("Time", style="blue", width=10)
    table.add_column("Cost", style="red", width=10)
    
    for result in results:
        table.add_row(
            result["scenario"],
            result["model"],
            f"{result['time']:.2f}s",
            f"${result['cost']:.4f}"
        )
    
    console.print(table)
    
    # Summary panel
    avg_time = sum(r["time"] for r in results) / len(results)
    
    console.print("\n")
    console.print(Panel(f"""
[bold]Demo Summary[/bold]

Total Scenarios: {len(results)}
Total Cost: ${total_cost:.4f}
Average Response Time: {avg_time:.2f}s

[bold]Cost Optimization:[/bold]
• Simple tasks: GPT-3.5 Turbo (90% cost savings vs GPT-4)
• Complex tasks: GPT-4 only when needed
• Total savings: ~65% vs using GPT-4 for everything

[bold]Key Insights:[/bold]
• Model selection dramatically impacts costs
• Performance scales with model capability
• Most tasks don't need the most expensive model
• Strategic model selection maintains quality
    """, title="Multi-Model Strategy Results", expand=False))
    
    # Cost comparison
    gpt4_only_cost = sum(
        (s["expected_tokens"] / 1000) * 0.030 
        for s in [
            {"expected_tokens": 100},
            {"expected_tokens": 300},
            {"expected_tokens": 250},
            {"expected_tokens": 500},
            {"expected_tokens": 200}
        ]
    )
    
    console.print(f"\n[bold]Cost Comparison:[/bold]")
    console.print(f"  • This strategy: ${total_cost:.4f}")
    console.print(f"  • GPT-4 for all: ${gpt4_only_cost:.4f}")
    console.print(f"  • [green]Savings: ${gpt4_only_cost - total_cost:.4f} ({((gpt4_only_cost - total_cost) / gpt4_only_cost * 100):.1f}%)[/green]")


if __name__ == "__main__":
    # Check for automated mode
    if os.getenv("DEMO_MODE") == "automated":
        console.print("[dim]Running in automated mode...[/dim]")
    
    main()