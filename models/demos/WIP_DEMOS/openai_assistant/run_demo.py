#!/usr/bin/env python3
"""
🤖 OpenAI Assistant Demo
========================

This demo showcases using OpenAI's API directly for AI assistance tasks.
No fine-tuning required - demonstrates prompt engineering and API usage.

Key Learning Points:
- Direct API usage vs fine-tuning
- Prompt engineering techniques
- Cost optimization strategies
- Production API integration patterns
"""

import json
import os
import time
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

console = Console()

class OpenAIAssistantDemo:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
    def display_intro(self):
        """Display demo introduction"""
        intro_text = """
🤖 [bold cyan]OpenAI Assistant Demo[/bold cyan]

[bold yellow]Scenario:[/bold yellow] Multi-purpose AI assistant using OpenAI's API
[bold yellow]Advantage:[/bold yellow] No training required - immediate deployment
[bold yellow]Model:[/bold yellow] GPT-4 (via OpenAI API)
[bold yellow]Method:[/bold yellow] Direct API calls with optimized prompts
[bold yellow]Cost:[/bold yellow] Pay-per-use (~$0.03 per 1K tokens)
[bold yellow]Use Cases:[/bold yellow] Content writing, analysis, coding help, Q&A

[bold green]Why this approach:[/bold green]
• No model training or infrastructure setup required
• Access to latest GPT models immediately
• Excellent for prototyping and low-volume use cases
• Built-in safety and content filtering
• Regular model updates from OpenAI

[bold red]Considerations:[/bold red]
• Ongoing API costs (vs one-time training)
• Internet connectivity required
• Data sent to OpenAI servers
• Rate limits and quota management
        """
        
        console.print(Panel(intro_text, title="🚀 Demo Overview", expand=False))

    def check_api_setup(self):
        """Check if OpenAI API is properly configured"""
        console.print("\n[bold blue]🔧 API Configuration Check[/bold blue]")
        
        if not self.api_key:
            console.print("[red]❌ OPENAI_API_KEY not found in environment![/red]")
            console.print("[yellow]Please set your OpenAI API key in .env file:[/yellow]")
            console.print("[cyan]OPENAI_API_KEY=sk-proj-your-key-here[/cyan]")
            return False
            
        console.print(f"[green]✅ API Key found: {self.api_key[:10]}...{self.api_key[-4:]}[/green]")
        console.print(f"[green]✅ Base URL: {self.base_url}[/green]")
        
        # Check if we can import openai
        try:
            console.print("[cyan]🔍 Checking for OpenAI Python package...[/cyan]")
            import openai
            console.print("[green]✅ OpenAI package is available[/green]")
            return True
        except ImportError:
            console.print("[yellow]📦 OpenAI package not found. Installing...[/yellow]")
            try:
                import subprocess
                import sys
                subprocess.run([sys.executable, "-m", "pip", "install", "openai"], check=True)
                console.print("[green]✅ OpenAI package installed successfully[/green]")
                return True
            except Exception as e:
                console.print(f"[red]❌ Failed to install OpenAI package: {e}[/red]")
                return False

    def demonstrate_capabilities(self):
        """Demonstrate different OpenAI API capabilities"""
        console.print("\n[bold blue]🎯 OpenAI API Capabilities Demo[/bold blue]")
        
        # Simulate API calls with different use cases
        use_cases = [
            {
                "name": "Content Writing",
                "prompt": "Write a professional blog post introduction about AI in healthcare",
                "response": """Artificial Intelligence is revolutionizing healthcare at an unprecedented pace, transforming everything from diagnostic accuracy to patient care delivery. As medical professionals increasingly embrace AI-powered tools, we're witnessing remarkable improvements in treatment outcomes, operational efficiency, and personalized medicine approaches.

From machine learning algorithms that can detect cancer in medical imaging with superhuman accuracy to natural language processing systems that streamline clinical documentation, AI is not just supporting healthcare workers—it's empowering them to focus on what matters most: patient care.""",
                "tokens_used": 85,
                "cost": 0.0026
            },
            {
                "name": "Code Generation",
                "prompt": "Create a Python function to calculate fibonacci numbers with memoization",
                "response": """```python
def fibonacci_memo(n, memo={}):
    \"\"\"
    Calculate fibonacci number with memoization for efficiency.
    
    Args:
        n (int): The position in fibonacci sequence
        memo (dict): Memoization cache
    
    Returns:
        int: The fibonacci number at position n
    \"\"\"
    if n in memo:
        return memo[n]
    
    if n <= 1:
        return n
    
    memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
    return memo[n]

# Example usage
print(fibonacci_memo(10))  # Output: 55
```""",
                "tokens_used": 124,
                "cost": 0.0037
            },
            {
                "name": "Data Analysis",
                "prompt": "Analyze this sales data and provide insights: Q1: $45K, Q2: $52K, Q3: $48K, Q4: $61K",
                "response": """**Sales Performance Analysis:**

**Key Findings:**
• Total annual revenue: $206K
• Average quarterly performance: $51.5K
• Growth trajectory shows 35.6% increase from Q1 to Q4

**Quarterly Insights:**
• Q2 showed strong 15.6% growth over Q1
• Q3 experienced a slight 7.7% decline (seasonal dip?)
• Q4 delivered exceptional 27.1% recovery and growth

**Recommendations:**
• Investigate Q3 factors to prevent future dips
• Replicate Q4 success strategies in upcoming quarters
• Target annual growth rate of 25-30% based on Q4 momentum""",
                "tokens_used": 142,
                "cost": 0.0043
            }
        ]
        
        total_cost = 0
        
        for i, use_case in enumerate(use_cases, 1):
            console.print(f"\n[bold yellow]Use Case {i}: {use_case['name']}[/bold yellow]")
            console.print(f"[cyan]Prompt:[/cyan] {use_case['prompt']}")
            
            # Simulate API call with progress
            with Progress(
                SpinnerColumn(),
                TextColumn("[cyan]{task.description}[/cyan]"),
                console=console,
            ) as progress:
                task = progress.add_task("Calling OpenAI API...", total=None)
                time.sleep(1.5)  # Simulate API call time
                progress.update(task, description="Processing response...")
                time.sleep(0.8)
                progress.update(task, description="✅ Response received!")
            
            console.print(f"[green]Response:[/green]")
            console.print(f"[dim]{use_case['response']}[/dim]")
            
            # Show metrics
            console.print(f"\n[yellow]📊 Metrics:[/yellow]")
            console.print(f"  • Tokens used: {use_case['tokens_used']}")
            console.print(f"  • Estimated cost: ${use_case['cost']:.4f}")
            total_cost += use_case['cost']
            
            console.print("\n" + "-" * 80)
        
        # Summary
        console.print(f"\n[bold green]💰 Total Demo Cost: ${total_cost:.4f}[/bold green]")
        console.print(f"[dim]Actual costs would vary based on model and usage patterns[/dim]")

    def show_cost_optimization(self):
        """Show cost optimization strategies"""
        console.print("\n[bold blue]💡 Cost Optimization Strategies[/bold blue]")
        
        strategies_table = Table(title="OpenAI Cost Optimization", show_header=True)
        strategies_table.add_column("Strategy", style="cyan", width=20)
        strategies_table.add_column("Savings", style="green", width=15)
        strategies_table.add_column("Implementation", style="yellow", width=40)
        
        strategies_table.add_row(
            "Use GPT-3.5 for simple tasks",
            "~90% cheaper",
            "Route simple queries to gpt-3.5-turbo instead of gpt-4"
        )
        strategies_table.add_row(
            "Optimize prompt length",
            "30-50% savings",
            "Use concise, specific prompts without unnecessary context"
        )
        strategies_table.add_row(
            "Implement response caching",
            "50-80% savings",
            "Cache common queries and responses in Redis/database"
        )
        strategies_table.add_row(
            "Use max_tokens limits",
            "20-40% savings",
            "Set appropriate max_tokens to prevent overly long responses"
        )
        strategies_table.add_row(
            "Batch similar requests",
            "10-20% savings",
            "Group related queries to share context efficiently"
        )
        
        console.print(strategies_table)

    def show_vs_finetuning(self):
        """Compare OpenAI API vs Fine-tuning approaches"""
        console.print("\n[bold blue]⚖️  OpenAI API vs Fine-Tuning Comparison[/bold blue]")
        
        comparison_table = Table(title="Approach Comparison", show_header=True)
        comparison_table.add_column("Factor", style="cyan", width=20)
        comparison_table.add_column("OpenAI API", style="blue", width=25)
        comparison_table.add_column("Fine-Tuning", style="green", width=25)
        comparison_table.add_column("Winner", style="yellow", width=15)
        
        comparisons = [
            ("Setup Time", "Minutes", "Hours/Days", "OpenAI API"),
            ("Initial Cost", "$0", "$100-1000+", "OpenAI API"),
            ("Ongoing Cost", "$0.03/1K tokens", "Hosting ~$50-200/mo", "Depends on usage"),
            ("Model Quality", "Excellent (GPT-4)", "Good (with data)", "OpenAI API"),
            ("Customization", "Limited (prompts)", "High (domain-specific)", "Fine-tuning"),
            ("Data Privacy", "Sent to OpenAI", "Keep local", "Fine-tuning"),
            ("Latency", "200-800ms", "50-200ms (local)", "Fine-tuning"),
            ("Maintenance", "None", "Model updates, hosting", "OpenAI API"),
            ("Scale to Zero", "Perfect", "Infrastructure costs", "OpenAI API")
        ]
        
        for factor, api_score, ft_score, winner in comparisons:
            comparison_table.add_row(factor, api_score, ft_score, winner)
        
        console.print(comparison_table)
        
        console.print("\n[bold yellow]📋 Decision Guidelines:[/bold yellow]")
        console.print("[green]Choose OpenAI API when:[/green]")
        console.print("• Rapid prototyping or MVP development")
        console.print("• Low to medium volume usage (<100K tokens/month)")
        console.print("• Need latest model capabilities")
        console.print("• Minimal technical team/resources")
        
        console.print("\n[green]Choose Fine-Tuning when:[/green]")
        console.print("• High volume usage (>1M tokens/month)")
        console.print("• Highly specialized domain knowledge needed")
        console.print("• Strict data privacy requirements")
        console.print("• Consistent, predictable workloads")

    def show_production_integration(self):
        """Show production integration patterns"""
        console.print("\n[bold blue]🚀 Production Integration Patterns[/bold blue]")
        
        integration_text = """
[bold yellow]1. Direct Integration:[/bold yellow]
```python
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def call_openai_with_retry(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.7
    )
    return response.choices[0].message.content
```

[bold yellow]2. Caching Layer:[/bold yellow]
```python
import redis
import hashlib

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cached_openai_call(prompt, ttl=3600):
    cache_key = f"openai:{hashlib.md5(prompt.encode()).hexdigest()}"
    cached = redis_client.get(cache_key)
    
    if cached:
        return cached.decode()
    
    response = call_openai_with_retry(prompt)
    redis_client.setex(cache_key, ttl, response)
    return response
```

[bold yellow]3. Rate Limiting:[/bold yellow]
```python
from ratelimit import limits, sleep_and_retry
import requests

@sleep_and_retry
@limits(calls=60, period=60)  # 60 calls per minute
def rate_limited_openai_call(prompt):
    return call_openai_with_retry(prompt)
```

[bold yellow]4. Cost Monitoring:[/bold yellow]
```python
class OpenAICostTracker:
    def __init__(self):
        self.total_tokens = 0
        self.total_cost = 0.0
    
    def track_usage(self, tokens_used, model="gpt-4"):
        costs = {"gpt-4": 0.03, "gpt-3.5-turbo": 0.002}
        cost = (tokens_used / 1000) * costs.get(model, 0.03)
        
        self.total_tokens += tokens_used
        self.total_cost += cost
        
        if self.total_cost > 100:  # Alert at $100
            send_alert(f"OpenAI costs reached ${self.total_cost:.2f}")
```
        """
        
        console.print(Panel(integration_text, title="🔧 Production Code Examples", expand=False))

    def run_demo(self):
        """Run the complete demo"""
        try:
            console.print("\n[bold green]🚀 Starting OpenAI Assistant Demo...[/bold green]")
            console.print("[yellow]📋 This demo shows direct OpenAI API usage - no training required![/yellow]")
            console.print("[yellow]⏱️  Estimated time: 2-3 minutes[/yellow]\n")
            
            self.display_intro()
            time.sleep(1)
            
            # Check API setup
            if not self.check_api_setup():
                console.print("\n[yellow]⚠️  Running in simulation mode due to missing API setup[/yellow]")
            
            if os.getenv("DEMO_MODE") != "automated":
                input("\n[yellow]Press Enter to demonstrate API capabilities...[/yellow]")
            else:
                console.print("\n[dim]Automated mode - proceeding to API demonstration...[/dim]")
                time.sleep(1)
            
            self.demonstrate_capabilities()
            
            if os.getenv("DEMO_MODE") != "automated":
                input("\n[yellow]Press Enter to see cost optimization strategies...[/yellow]")
            else:
                console.print("\n[dim]Showing cost optimization strategies...[/dim]")
                time.sleep(1)
            
            self.show_cost_optimization()
            
            self.show_vs_finetuning()
            
            if os.getenv("DEMO_MODE") != "automated":
                input("\n[yellow]Press Enter to see production integration patterns...[/yellow]")
            else:
                console.print("\n[dim]Showing production integration patterns...[/dim]")
                time.sleep(1)
            
            self.show_production_integration()
            
            console.print("\n[bold green]🎉 OpenAI Assistant demo completed successfully![/bold green]")
            console.print("[cyan]💡 This approach is perfect for rapid prototyping and low-medium volume use cases![/cyan]")
            return True
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Demo interrupted by user[/yellow]")
            return False
        except Exception as e:
            console.print(f"\n[red]Demo error: {e}[/red]")
            return False


def main():
    """Main entry point"""
    demo = OpenAIAssistantDemo()
    demo.run_demo()


if __name__ == "__main__":
    main()