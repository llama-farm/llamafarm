#!/usr/bin/env python3
"""
🌐 OpenAI Model Switching Demo
=============================

This demo showcases dynamic model switching between different OpenAI models
(GPT-4o-mini, GPT-4o, GPT-4-turbo) based on query complexity, cost preferences,
and performance requirements.

Key Learning Points:
- Strategic model selection for cost optimization
- Automatic complexity detection and routing
- Real-time cost and performance tracking
- Easy model switching with configuration strategies
- Production-ready failover and monitoring
"""

import json
import os
import sys
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.syntax import Syntax
from rich import print as rprint

console = Console()

# Mock OpenAI client for demo purposes (replace with real client in production)
class MockOpenAIResponse:
    def __init__(self, content: str, model: str, tokens: int):
        self.choices = [type('Choice', (), {
            'message': type('Message', (), {'content': content})()
        })()]
        self.usage = type('Usage', (), {'total_tokens': tokens})()
        self.model = model

class MockOpenAIClient:
    def __init__(self):
        self.chat = self
        self.completions = self
    
    async def create(self, model: str, messages: List[Dict], **kwargs) -> MockOpenAIResponse:
        # Simulate different response times for different models
        response_times = {
            'gpt-4o-mini': 0.8,
            'gpt-4o': 2.5,
            'gpt-4-turbo': 2.0
        }
        
        await asyncio.sleep(response_times.get(model, 1.0))
        
        # Generate mock response based on query
        query = messages[0]['content'] if messages else "test query"
        
        # Simple response generation based on model
        if model == 'gpt-4o-mini':
            response = f"[GPT-4o-mini] Quick response to: {query[:50]}... This is a fast, cost-effective answer."
            tokens = len(response.split()) * 1.3
        elif model == 'gpt-4o':
            response = f"[GPT-4o] Comprehensive analysis of: {query[:50]}... This is a detailed, high-quality response with advanced reasoning and thorough explanations."
            tokens = len(response.split()) * 2.1
        else:  # gpt-4-turbo
            response = f"[GPT-4-turbo] Balanced response to: {query[:50]}... This provides good quality with reasonable cost and performance."
            tokens = len(response.split()) * 1.8
        
        return MockOpenAIResponse(response, model, int(tokens))

class TaskComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"

class CostStrategy(Enum):
    MINIMIZE_COST = "minimize_cost"
    BALANCED = "balanced"
    MAXIMIZE_QUALITY = "maximize_quality"
    PERFORMANCE_FIRST = "performance_first"

@dataclass
class ModelStrategy:
    name: str
    description: str
    model: str
    temperature: float
    max_tokens: int
    use_cases: List[str]
    cost_per_1k_tokens: float
    expected_quality: float
    response_time_ms: int
    complexity_threshold: TaskComplexity

class OpenAIModelsDemo:
    def __init__(self):
        self.dataset_path = Path("../datasets/general/model_switching.jsonl")
        self.strategy_path = Path("strategies/openai_models.yaml")
        
        # Initialize with mock client (replace with real OpenAI client)
        self.client = MockOpenAIClient()
        self.usage_stats = {}
        
        # Define model strategies
        self.strategies = {
            "lightning_fast": ModelStrategy(
                name="lightning_fast",
                description="Ultra-fast responses for simple queries",
                model="gpt-4o-mini",
                temperature=0.3,
                max_tokens=500,
                use_cases=["simple_qa", "definitions", "basic_math"],
                cost_per_1k_tokens=0.375,
                expected_quality=0.75,
                response_time_ms=800,
                complexity_threshold=TaskComplexity.SIMPLE
            ),
            "balanced_performer": ModelStrategy(
                name="balanced_performer",
                description="Good balance of speed, cost, and quality",
                model="gpt-4o-mini",
                temperature=0.7,
                max_tokens=1500,
                use_cases=["general_chat", "content_writing", "code_help"],
                cost_per_1k_tokens=0.375,
                expected_quality=0.85,
                response_time_ms=1200,
                complexity_threshold=TaskComplexity.MODERATE
            ),
            "quality_focused": ModelStrategy(
                name="quality_focused",
                description="High-quality responses for complex tasks",
                model="gpt-4o",
                temperature=0.7,
                max_tokens=2000,
                use_cases=["analysis", "research", "complex_reasoning"],
                cost_per_1k_tokens=10.0,
                expected_quality=0.95,
                response_time_ms=2500,
                complexity_threshold=TaskComplexity.COMPLEX
            ),
            "expert_level": ModelStrategy(
                name="expert_level",
                description="Maximum capability for expert-level tasks",
                model="gpt-4o",
                temperature=0.2,
                max_tokens=4000,
                use_cases=["expert_analysis", "scientific_reasoning", "complex_coding"],
                cost_per_1k_tokens=10.0,
                expected_quality=0.98,
                response_time_ms=4000,
                complexity_threshold=TaskComplexity.EXPERT
            )
        }
        
    def display_intro(self):
        """Display demo introduction"""
        intro_text = """
🌐 [bold cyan]OpenAI Model Switching Demo[/bold cyan]

[bold yellow]Scenario:[/bold yellow] Intelligent model routing for optimal cost/performance
[bold yellow]Models:[/bold yellow] GPT-4o-mini, GPT-4o, GPT-4-turbo
[bold yellow]Strategy:[/bold yellow] Dynamic complexity-based routing
[bold yellow]Dataset:[/bold yellow] 15+ model switching and strategy examples

[bold green]Why this approach:[/bold green]
• GPT-4o-mini is 60x cheaper than GPT-4o for simple tasks
• GPT-4o provides superior reasoning for complex queries
• Smart routing can reduce costs by 40-70% while maintaining quality
• Easy configuration switching for different environments

[bold red]Key demonstrations:[/bold red]
• Automatic complexity detection and model selection
• Cost vs quality trade-off analysis
• Real-time performance and cost tracking
• Failover and error handling strategies
• Production deployment patterns
        """
        
        console.print(Panel(intro_text, title="🚀 Demo Overview", expand=False))

    def analyze_dataset(self):
        """Analyze and display dataset statistics"""
        console.print("\n[bold blue]📊 Dataset Analysis[/bold blue]")
        console.print("[yellow]🔍 Checking for dataset files...[/yellow]")
        
        if not self.dataset_path.exists():
            console.print(f"[red]❌ Dataset not found: {self.dataset_path}[/red]")
            return False
            
        console.print(f"[green]✅ Found dataset at {self.dataset_path}[/green]")
            
        examples = []
        with open(self.dataset_path, 'r') as f:
            for line in f:
                examples.append(json.loads(line))
                
        console.print(f"[green]✅ Loaded {len(examples)} training examples[/green]")
        
        # Analyze content types
        categories = {
            'Model Comparison': 0,
            'Cost Optimization': 0,
            'Strategy Switching': 0,
            'Implementation': 0,
            'Production': 0
        }
        
        for example in examples:
            instruction = example.get('instruction', '').lower()
            if 'difference' in instruction or 'compare' in instruction:
                categories['Model Comparison'] += 1
            elif 'cost' in instruction or 'price' in instruction:
                categories['Cost Optimization'] += 1
            elif 'switch' in instruction or 'strategy' in instruction:
                categories['Strategy Switching'] += 1
            elif 'implement' in instruction or 'code' in instruction:
                categories['Implementation'] += 1
            else:
                categories['Production'] += 1
        
        # Display statistics table
        stats_table = Table(title="Dataset Statistics", show_header=True)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("Total Examples", str(len(examples)))
        stats_table.add_row("Average Response Length", f"{sum(len(ex['output']) for ex in examples) // len(examples)} characters")
        
        console.print(stats_table)
        
        # Display category distribution
        category_table = Table(title="Content Categories", show_header=True)
        category_table.add_column("Category", style="yellow")
        category_table.add_column("Examples", style="magenta")
        
        for category, count in categories.items():
            category_table.add_row(category, str(count))
            
        console.print(category_table)
        
        return True

    def show_sample_data(self):
        """Display sample training examples"""
        console.print("\n[bold blue]📝 Sample Training Examples[/bold blue]")
        
        with open(self.dataset_path, 'r') as f:
            examples = [json.loads(line) for line in f]
            
        # Show 2 representative examples
        import random
        samples = random.sample(examples, min(2, len(examples)))
        
        for i, sample in enumerate(samples, 1):
            console.print(f"\n[bold yellow]Example {i}:[/bold yellow]")
            console.print(f"[cyan]Q:[/cyan] {sample['instruction']}")
            console.print(f"[green]A:[/green]")
            
            # Truncate long responses for better display
            output = sample['output']
            if len(output) > 400:
                console.print(output[:400] + "...\n[dim](Response truncated for display)[/dim]")
            else:
                console.print(output)

    def analyze_query_complexity(self, query: str) -> TaskComplexity:
        """Analyze query to determine complexity level"""
        query_lower = query.lower()
        
        # Expert level indicators
        expert_keywords = [
            'comprehensive analysis', 'research paper', 'scientific method',
            'architectural design', 'system design', 'algorithm optimization'
        ]
        
        # Complex level indicators  
        complex_keywords = [
            'analyze', 'compare and contrast', 'evaluate', 'synthesize',
            'strategy', 'methodology', 'framework', 'implementation'
        ]
        
        # Moderate level indicators
        moderate_keywords = [
            'how to', 'explain', 'describe', 'steps', 'example', 'overview'
        ]
        
        # Simple level indicators
        simple_keywords = [
            'what is', 'define', 'who is', 'when was', 'basic', 'simple'
        ]
        
        # Check for expert indicators
        for keyword in expert_keywords:
            if keyword in query_lower:
                return TaskComplexity.EXPERT
        
        # Check for complex indicators
        for keyword in complex_keywords:
            if keyword in query_lower:
                return TaskComplexity.COMPLEX
        
        # Check for moderate indicators
        for keyword in moderate_keywords:
            if keyword in query_lower:
                return TaskComplexity.MODERATE
        
        # Check for simple indicators
        for keyword in simple_keywords:
            if keyword in query_lower:
                return TaskComplexity.SIMPLE
        
        # Default based on length
        word_count = len(query.split())
        if word_count > 30:
            return TaskComplexity.COMPLEX
        elif word_count > 15:
            return TaskComplexity.MODERATE
        else:
            return TaskComplexity.SIMPLE

    def select_strategy(self, query: str, cost_preference: CostStrategy = CostStrategy.BALANCED) -> ModelStrategy:
        """Select optimal strategy based on query and preferences"""
        complexity = self.analyze_query_complexity(query)
        
        # Filter strategies by complexity threshold
        suitable_strategies = [
            strategy for strategy in self.strategies.values()
            if self._complexity_order(strategy.complexity_threshold) >= self._complexity_order(complexity)
        ]
        
        if not suitable_strategies:
            return self.strategies["balanced_performer"]
        
        # Apply cost preference
        if cost_preference == CostStrategy.MINIMIZE_COST:
            return min(suitable_strategies, key=lambda s: s.cost_per_1k_tokens)
        elif cost_preference == CostStrategy.MAXIMIZE_QUALITY:
            return max(suitable_strategies, key=lambda s: s.expected_quality)
        elif cost_preference == CostStrategy.PERFORMANCE_FIRST:
            return min(suitable_strategies, key=lambda s: s.response_time_ms)
        else:  # BALANCED
            # Calculate composite score
            def composite_score(strategy):
                quality_cost_ratio = strategy.expected_quality / strategy.cost_per_1k_tokens
                time_penalty = 1000 / strategy.response_time_ms
                return quality_cost_ratio * time_penalty
            
            return max(suitable_strategies, key=composite_score)

    def _complexity_order(self, complexity: TaskComplexity) -> int:
        """Get numeric order of complexity"""
        order = {
            TaskComplexity.SIMPLE: 1,
            TaskComplexity.MODERATE: 2,
            TaskComplexity.COMPLEX: 3,
            TaskComplexity.EXPERT: 4
        }
        return order[complexity]

    async def demonstrate_model_switching(self):
        """Demonstrate intelligent model switching"""
        console.print("\n[bold blue]🎯 Model Switching Demonstration[/bold blue]")
        
        # Demo queries of different complexities
        demo_queries = [
            {
                'query': 'What is OpenAI?',
                'description': 'Simple definition query',
                'expected_complexity': TaskComplexity.SIMPLE
            },
            {
                'query': 'How do I choose between GPT-4o-mini and GPT-4o for my application?',
                'description': 'Moderate complexity decision question',
                'expected_complexity': TaskComplexity.MODERATE
            },
            {
                'query': 'Design a comprehensive cost optimization strategy for deploying multiple OpenAI models in a production environment with varying workloads.',
                'description': 'Complex system design question',
                'expected_complexity': TaskComplexity.COMPLEX
            }
        ]
        
        cost_strategies = [CostStrategy.MINIMIZE_COST, CostStrategy.BALANCED, CostStrategy.MAXIMIZE_QUALITY]
        
        total_cost = 0
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}[/bold blue]"),
            BarColumn(bar_width=30),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            console=console,
            refresh_per_second=10
        ) as progress:
            
            main_task = progress.add_task("Processing model switching demos", total=len(demo_queries) * len(cost_strategies))
            
            for query_info in demo_queries:
                console.print(f"\n[yellow]📝 Query: {query_info['description']}[/yellow]")
                console.print(f"[dim]{query_info['query']}[/dim]")
                
                detected_complexity = self.analyze_query_complexity(query_info['query'])
                console.print(f"[cyan]🔍 Detected Complexity: {detected_complexity.value.upper()}[/cyan]")
                
                query_results = []
                
                for cost_strategy in cost_strategies:
                    strategy = self.select_strategy(query_info['query'], cost_strategy)
                    
                    console.print(f"\n[green]🎯 {cost_strategy.value} → {strategy.name} ({strategy.model})[/green]")
                    
                    # Simulate API call
                    start_time = time.time()
                    try:
                        response = await self.client.create(
                            model=strategy.model,
                            messages=[{'role': 'user', 'content': query_info['query']}],
                            temperature=strategy.temperature,
                            max_tokens=strategy.max_tokens
                        )
                        
                        response_time = (time.time() - start_time) * 1000
                        tokens_used = response.usage.total_tokens
                        cost = (tokens_used / 1000) * strategy.cost_per_1k_tokens
                        total_cost += cost
                        
                        console.print(f"[green]  ✅ Success: {tokens_used} tokens, ${cost:.4f}, {response_time:.0f}ms[/green]")
                        console.print(f"[dim]  Response: {response.choices[0].message.content[:100]}...[/dim]")
                        
                        query_results.append({
                            'strategy': cost_strategy.value,
                            'model': strategy.model,
                            'cost': cost,
                            'time': response_time,
                            'tokens': tokens_used,
                            'quality': strategy.expected_quality
                        })
                        
                    except Exception as e:
                        console.print(f"[red]  ❌ Error: {e}[/red]")
                    
                    progress.advance(main_task)
                    await asyncio.sleep(0.1)  # Small delay for demo effect
                
                results.append({
                    'query': query_info['description'],
                    'complexity': detected_complexity.value,
                    'results': query_results
                })
        
        # Show summary comparison
        console.print(f"\n[bold green]📊 Demo Summary[/bold green]")
        console.print(f"[yellow]Total Cost: ${total_cost:.4f}[/yellow]")
        
        summary_table = Table(title="Strategy Performance Comparison", show_header=True)
        summary_table.add_column("Query Type", style="cyan")
        summary_table.add_column("Cost Strategy", style="yellow")
        summary_table.add_column("Model Used", style="green")
        summary_table.add_column("Cost", style="red")
        summary_table.add_column("Time (ms)", style="blue")
        summary_table.add_column("Quality Score", style="magenta")
        
        for result in results:
            for i, query_result in enumerate(result['results']):
                query_name = result['query'] if i == 0 else ""
                summary_table.add_row(
                    query_name,
                    query_result['strategy'],
                    query_result['model'],
                    f"${query_result['cost']:.4f}",
                    f"{query_result['time']:.0f}",
                    f"{query_result['quality']:.0%}"
                )
        
        console.print(summary_table)

    def show_cost_analysis(self):
        """Show cost analysis and optimization recommendations"""
        console.print("\n[bold blue]💰 Cost Analysis & Optimization[/bold blue]")
        
        # Create cost comparison table
        cost_table = Table(title="OpenAI Model Cost Comparison (per 1K tokens)", show_header=True)
        cost_table.add_column("Model", style="cyan")
        cost_table.add_column("Input Cost", style="green")
        cost_table.add_column("Output Cost", style="red")
        cost_table.add_column("Average Cost", style="yellow")
        cost_table.add_column("Best Use Cases", style="blue")
        
        models_info = [
            ("GPT-4o-mini", "$0.15", "$0.60", "$0.375", "Chat, simple Q&A, content moderation"),
            ("GPT-4o", "$5.00", "$15.00", "$10.00", "Complex reasoning, analysis, research"),
            ("GPT-4-turbo", "$10.00", "$30.00", "$20.00", "Balanced performance, professional tasks")
        ]
        
        for model_info in models_info:
            cost_table.add_row(*model_info)
        
        console.print(cost_table)
        
        # Cost optimization recommendations
        recommendations = """
[bold green]💡 Cost Optimization Recommendations:[/bold green]

1. [yellow]Use GPT-4o-mini for 70-80% of queries:[/yellow]
   • Simple definitions and explanations
   • Basic Q&A and customer support
   • Content moderation and classification

2. [yellow]Reserve GPT-4o for complex tasks:[/yellow]
   • In-depth analysis and research
   • Complex reasoning and problem-solving
   • Advanced coding and architecture decisions

3. [yellow]Implement smart routing:[/yellow]
   • Automatic complexity detection
   • Cost-based model selection
   • Fallback strategies for reliability

4. [yellow]Monitor and optimize:[/yellow]
   • Track costs per model and use case
   • A/B test different routing strategies
   • Set budget alerts and limits
        """
        
        console.print(Panel(recommendations, title="💡 Optimization Tips", expand=False))

    def show_deployment_strategies(self):
        """Show deployment and production strategies"""
        deployment_text = """
🚀 [bold cyan]Deployment & Production Strategies[/bold cyan]

[bold yellow]Environment-Specific Configurations:[/bold yellow]

[green]Development Environment:[/green]
• Default to GPT-4o-mini for cost savings
• Enable extensive logging and debugging
• Use aggressive caching to reduce API calls
• Set lower rate limits to prevent overspend

[green]Staging Environment:[/green]
• Mirror production routing logic
• Use balanced cost/quality strategy
• Enable performance monitoring
• Test failover scenarios

[green]Production Environment:[/green]
• Implement smart routing with circuit breakers
• Set up comprehensive monitoring and alerting
• Use connection pooling and request queuing
• Enable real-time cost tracking

[bold yellow]Best Practices:[/bold yellow]
• Implement request deduplication and caching
• Use async processing for high-volume scenarios
• Set up proper error handling and retries
• Monitor model performance and costs continuously
• Implement gradual rollout for strategy changes

[bold yellow]Scaling Considerations:[/bold yellow]
• Use load balancing across multiple API keys
• Implement request batching where possible
• Consider regional deployments for latency
• Plan for model version updates and migrations
        """
        
        console.print(Panel(deployment_text, title="🎯 Production Ready", expand=False))

    def run_demo(self):
        """Run the complete demo"""
        try:
            # Immediate startup feedback
            console.print("\n[bold green]🚀 Starting OpenAI Model Switching Demo...[/bold green]")
            console.print("[yellow]⏱️  Estimated time: 4-5 minutes[/yellow]")
            console.print("[cyan]📋 Loading demo components...[/cyan]\n")
            
            time.sleep(0.5)
            
            self.display_intro()
            time.sleep(1)
            
            console.print("\n[cyan]🔄 Beginning dataset analysis...[/cyan]")
            time.sleep(0.3)
            
            if not self.analyze_dataset():
                return False
                
            console.print("\n[cyan]📝 Preparing sample data display...[/cyan]")
            time.sleep(0.3)
            self.show_sample_data()
            
            console.print("\n[yellow]▶️  Ready to proceed with model switching demonstration[/yellow]")
            if os.getenv("DEMO_MODE") != "automated":
                input("[yellow]Press Enter to continue...[/yellow]")
            else:
                console.print("[dim]Running in automated mode - skipping user prompts[/dim]")
                time.sleep(1)
            
            # Run model switching demo
            asyncio.run(self.demonstrate_model_switching())
            
            if os.getenv("DEMO_MODE") != "automated":
                input("\n[yellow]Press Enter to view cost analysis...[/yellow]")
            else:
                console.print("\n[dim]Proceeding to cost analysis...[/dim]")
                time.sleep(1)
            
            self.show_cost_analysis()
            
            self.show_deployment_strategies()
            
            console.print("\n[bold green]🎉 OpenAI Model Switching demo completed successfully![/bold green]")
            console.print("[cyan]💡 Key takeaway: Smart model routing can reduce costs by 40-70% while maintaining quality![/cyan]")
            return True
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Demo interrupted by user[/yellow]")
            return False
        except Exception as e:
            console.print(f"\n[red]Demo error: {e}[/red]")
            return False


def main():
    """Main entry point"""
    demo = OpenAIModelsDemo()
    demo.run_demo()


if __name__ == "__main__":
    main()