"""Enhanced CLI with strategy support for the LlamaFarm Prompts system."""

import json
import yaml
from typing import Any, Dict, List, Optional
from datetime import datetime

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.tree import Tree

from pathlib import Path

# Import handling for both package and direct script execution
import sys
prompts_dir = Path(__file__).parent.parent.parent.parent

# Only add to path if needed and not already present
if str(prompts_dir) not in sys.path:
    sys.path.insert(0, str(prompts_dir))

# Now import with absolute imports
from strategies import StrategyManager, StrategyConfig
from prompts.core.engines.template_registry import TemplateRegistry
from prompts.core.engines.template_engine import TemplateEngine

console = Console()

# Global instances
strategy_manager: Optional[StrategyManager] = None
template_registry: Optional[TemplateRegistry] = None


def init_system(strategies_file: Optional[str] = None, strategies_dir: Optional[str] = None):
    """Initialize the strategy system."""
    global strategy_manager, template_registry
    
    template_registry = TemplateRegistry()
    strategy_manager = StrategyManager(
        strategies_file=strategies_file,
        strategies_dir=strategies_dir,
        template_registry=template_registry
    )
    return strategy_manager


@click.group()
@click.option('--strategies-file', default='default_strategies.yaml', help='Path to strategies file')
@click.option('--strategies-dir', help='Directory containing strategy files')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def cli(strategies_file: str, strategies_dir: str, verbose: bool):
    """LlamaFarm Prompts Strategy Management CLI"""
    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    try:
        init_system(strategies_file, strategies_dir)
        console.print(f"[green]✓[/green] Loaded strategy system")
    except Exception as e:
        console.print(f"[red]Error loading system: {e}[/red]")
        raise click.Abort()


# =============================================================================
# STRATEGY COMMANDS
# =============================================================================

@cli.group()
def strategy():
    """Strategy management commands"""
    pass


@strategy.command('list')
@click.option('--format', 'output_format', default='table', type=click.Choice(['table', 'json', 'yaml']))
def list_strategies(output_format):
    """List all available strategies"""
    if not strategy_manager:
        console.print("[red]Strategy manager not initialized[/red]")
        return
    
    strategies = strategy_manager.load_strategies()
    
    if output_format == 'table':
        table = Table(title="Available Strategies")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Use Cases", style="yellow")
        table.add_column("Performance", style="magenta")
        table.add_column("Complexity", style="blue")
        
        for strategy_id, strategy in strategies.items():
            use_cases = ", ".join(strategy.use_cases[:3])
            if len(strategy.use_cases) > 3:
                use_cases += f" (+{len(strategy.use_cases) - 3})"
            
            # Handle both enum and string values
            performance = getattr(strategy.performance_profile, 'value', strategy.performance_profile)
            complexity = getattr(strategy.complexity, 'value', strategy.complexity)
            
            table.add_row(
                strategy_id,
                strategy.name,
                use_cases,
                performance,
                complexity
            )
        
        console.print(table)
    
    elif output_format == 'json':
        data = {sid: s.to_dict() for sid, s in strategies.items()}
        console.print_json(data=data)
    
    else:  # yaml
        data = {sid: s.to_dict() for sid, s in strategies.items()}
        console.print(yaml.dump(data, default_flow_style=False))


@strategy.command('show')
@click.argument('strategy_id')
@click.option('--show-templates', is_flag=True, help='Show template configurations')
def show_strategy(strategy_id, show_templates):
    """Show details of a specific strategy"""
    if not strategy_manager:
        console.print("[red]Strategy manager not initialized[/red]")
        return
    
    strategy = strategy_manager.get_strategy(strategy_id)
    if not strategy:
        console.print(f"[red]Strategy '{strategy_id}' not found[/red]")
        return
    
    # Basic info panel
    info = f"""
[bold]Name:[/bold] {strategy.name}
[bold]Description:[/bold] {strategy.description}
[bold]Performance:[/bold] {strategy.performance_profile.value}
[bold]Complexity:[/bold] {strategy.complexity.value}
[bold]Use Cases:[/bold] {', '.join(strategy.use_cases)}
"""
    console.print(Panel(info, title=f"Strategy: {strategy_id}"))
    
    # Templates tree
    tree = Tree("Templates")
    
    # Default template
    default_node = tree.add(f"[green]default[/green]: {strategy.templates.default.template}")
    if show_templates and strategy.templates.default.config:
        for key, value in strategy.templates.default.config.items():
            default_node.add(f"{key}: {value}")
    
    # Fallback template
    if strategy.templates.fallback:
        fallback_node = tree.add(f"[yellow]fallback[/yellow]: {strategy.templates.fallback.template}")
        if show_templates and strategy.templates.fallback.config:
            for key, value in strategy.templates.fallback.config.items():
                fallback_node.add(f"{key}: {value}")
    
    # Specialized templates
    if strategy.templates.specialized:
        spec_node = tree.add("[blue]specialized[/blue]")
        for spec in strategy.templates.specialized:
            condition_str = json.dumps(spec.condition.dict(exclude_none=True))
            spec_child = spec_node.add(f"{spec.template} (priority: {spec.priority})")
            spec_child.add(f"Condition: {condition_str}")
    
    console.print(tree)
    
    # Selection rules
    if strategy.selection_rules:
        console.print("\n[bold]Selection Rules:[/bold]")
        rules_table = Table()
        rules_table.add_column("Name", style="cyan")
        rules_table.add_column("Condition", style="yellow")
        rules_table.add_column("Template", style="green")
        rules_table.add_column("Priority", style="magenta")
        
        for rule in sorted(strategy.selection_rules, key=lambda r: r.priority, reverse=True):
            rules_table.add_row(
                rule.name,
                str(rule.condition),
                rule.template,
                str(rule.priority)
            )
        
        console.print(rules_table)
    
    # Global config
    if strategy.global_config.system_prompts or strategy.global_config.temperature or strategy.global_config.max_tokens:
        console.print("\n[bold]Global Configuration:[/bold]")
        if strategy.global_config.temperature:
            console.print(f"  Temperature: {strategy.global_config.temperature}")
        if strategy.global_config.max_tokens:
            console.print(f"  Max Tokens: {strategy.global_config.max_tokens}")
        if strategy.global_config.model_preferences:
            console.print(f"  Model Preferences: {', '.join(strategy.global_config.model_preferences)}")


@strategy.command('execute')
@click.argument('strategy_id')
@click.option('--query', '-q', help='Query or input text')
@click.option('--context', '-c', help='Additional context (JSON)')
@click.option('--override', '-o', help='Configuration overrides (JSON)')
@click.option('--show-selection', is_flag=True, help='Show template selection process')
def execute_strategy(strategy_id, query, context, override, show_selection):
    """Execute a strategy with given inputs"""
    if not strategy_manager:
        console.print("[red]Strategy manager not initialized[/red]")
        return
    
    # Prepare inputs
    inputs = {"query": query or ""}
    
    # Parse context
    context_dict = {}
    if context:
        try:
            context_dict = json.loads(context)
        except json.JSONDecodeError:
            console.print("[red]Invalid context JSON[/red]")
            return
    
    # Parse overrides
    override_config = {}
    if override:
        try:
            override_config = json.loads(override)
        except json.JSONDecodeError:
            console.print("[red]Invalid override JSON[/red]")
            return
    
    try:
        # Execute strategy
        with console.status(f"Executing strategy '{strategy_id}'..."):
            result = strategy_manager.execute_strategy(
                strategy_name=strategy_id,
                inputs=inputs,
                context=context_dict,
                override_config=override_config
            )
        
        # Show result
        console.print("\n[bold green]Generated Prompt:[/bold green]")
        syntax = Syntax(result, "text", theme="monokai", line_numbers=True)
        console.print(syntax)
        
        # Show template selection if requested
        if show_selection:
            strategy = strategy_manager.get_strategy(strategy_id)
            template_config = strategy.get_template_for_context({**inputs, **context_dict})
            console.print(f"\n[bold]Selected Template:[/bold] {template_config.template}")
            
    except Exception as e:
        console.print(f"[red]Error executing strategy: {e}[/red]")


@strategy.command('recommend')
@click.option('--use-case', help='Desired use case')
@click.option('--performance', type=click.Choice(['speed', 'accuracy', 'balanced', 'cost_optimized']))
@click.option('--complexity', type=click.Choice(['simple', 'moderate', 'complex']))
def recommend_strategies(use_case, performance, complexity):
    """Get strategy recommendations based on criteria"""
    if not strategy_manager:
        console.print("[red]Strategy manager not initialized[/red]")
        return
    
    recommendations = strategy_manager.recommend_strategies(
        use_case=use_case,
        performance=performance,
        complexity=complexity
    )
    
    if not recommendations:
        console.print("[yellow]No strategies match the criteria[/yellow]")
        return
    
    console.print(f"\n[bold]Recommended Strategies:[/bold]")
    
    for i, strategy in enumerate(recommendations[:5], 1):
        console.print(f"\n{i}. [cyan]{strategy.name}[/cyan]")
        console.print(f"   {strategy.description}")
        console.print(f"   Use cases: {', '.join(strategy.use_cases[:3])}")
        console.print(f"   Performance: {strategy.performance_profile.value}")
        console.print(f"   Complexity: {strategy.complexity.value}")


@strategy.command('create')
@click.option('--name', prompt=True, help='Strategy name')
@click.option('--description', prompt=True, help='Strategy description')
@click.option('--template', prompt=True, help='Default template ID')
@click.option('--output', '-o', help='Output file path')
def create_strategy(name, description, template, output):
    """Create a new strategy interactively"""
    if not strategy_manager:
        console.print("[red]Strategy manager not initialized[/red]")
        return
    
    # Collect use cases
    use_cases = []
    console.print("\nEnter use cases (empty line to finish):")
    while True:
        use_case = click.prompt("Use case", default="", show_default=False)
        if not use_case:
            break
        use_cases.append(use_case)
    
    # Performance profile
    performance = click.prompt(
        "Performance profile",
        type=click.Choice(['speed', 'accuracy', 'balanced', 'cost_optimized']),
        default='balanced'
    )
    
    # Complexity
    complexity = click.prompt(
        "Complexity",
        type=click.Choice(['simple', 'moderate', 'complex']),
        default='moderate'
    )
    
    try:
        # Create strategy
        strategy = strategy_manager.create_strategy(
            name=name,
            description=description,
            default_template=template,
            use_cases=use_cases,
            performance_profile=performance,
            complexity=complexity
        )
        
        # Generate strategy ID
        strategy_id = name.lower().replace(' ', '_')
        
        # Save if output specified
        if output:
            strategy_manager.save_strategy(strategy_id, strategy, output)
            console.print(f"[green]✓[/green] Strategy saved to {output}")
        else:
            # Display YAML
            console.print("\n[bold]Generated Strategy:[/bold]")
            yaml_str = yaml.dump(strategy.to_dict(), default_flow_style=False)
            syntax = Syntax(yaml_str, "yaml", theme="monokai")
            console.print(syntax)
            
    except Exception as e:
        console.print(f"[red]Error creating strategy: {e}[/red]")


@strategy.command('validate')
@click.argument('strategy_file', type=click.Path(exists=True))
def validate_strategy(strategy_file):
    """Validate a strategy configuration file"""
    try:
        with open(strategy_file, 'r') as f:
            data = yaml.safe_load(f)
        
        # Validate each strategy in file
        for strategy_id, strategy_data in data.items():
            if isinstance(strategy_data, dict) and 'templates' in strategy_data:
                try:
                    strategy = StrategyConfig.from_dict(strategy_data)
                    errors = strategy_manager.loader.validate_strategy(strategy)
                    
                    if errors:
                        console.print(f"[red]✗[/red] {strategy_id}: {', '.join(errors)}")
                    else:
                        console.print(f"[green]✓[/green] {strategy_id}: Valid")
                        
                except Exception as e:
                    console.print(f"[red]✗[/red] {strategy_id}: {str(e)}")
                    
    except Exception as e:
        console.print(f"[red]Error reading file: {e}[/red]")


# =============================================================================
# TEMPLATE COMMANDS (Enhanced)
# =============================================================================

@cli.group()
def template():
    """Template management commands"""
    pass


@template.command('usage')
@click.option('--template-id', help='Show usage for specific template')
def template_usage(template_id):
    """Show which strategies use each template"""
    if not strategy_manager:
        console.print("[red]Strategy manager not initialized[/red]")
        return
    
    usage = strategy_manager.get_template_usage()
    
    if template_id:
        # Show specific template usage
        if template_id in usage:
            strategies = usage[template_id]
            console.print(f"\n[bold]Template '{template_id}' is used by:[/bold]")
            for strategy in strategies:
                console.print(f"  • {strategy}")
        else:
            console.print(f"[yellow]Template '{template_id}' is not used by any strategy[/yellow]")
    else:
        # Show all template usage
        table = Table(title="Template Usage")
        table.add_column("Template", style="cyan")
        table.add_column("Used By", style="green")
        table.add_column("Count", style="yellow")
        
        for template_id, strategies in sorted(usage.items()):
            strategy_list = ", ".join(strategies[:3])
            if len(strategies) > 3:
                strategy_list += f" (+{len(strategies) - 3})"
            
            table.add_row(
                template_id,
                strategy_list,
                str(len(strategies))
            )
        
        console.print(table)


# =============================================================================
# SYSTEM COMMANDS
# =============================================================================

@cli.command('stats')
def show_stats():
    """Show system statistics"""
    if not strategy_manager:
        console.print("[red]Strategy manager not initialized[/red]")
        return
    
    strategies = strategy_manager.load_strategies()
    stats = strategy_manager.get_execution_stats()
    
    console.print(Panel("[bold]System Statistics[/bold]"))
    
    # Strategy stats
    console.print(f"\n[bold]Strategies:[/bold] {len(strategies)}")
    
    # Template usage
    template_usage = strategy_manager.get_template_usage()
    console.print(f"[bold]Templates in use:[/bold] {len(template_usage)}")
    
    # Execution stats
    if stats:
        console.print("\n[bold]Execution Statistics:[/bold]")
        table = Table()
        table.add_column("Strategy", style="cyan")
        table.add_column("Template", style="green")
        table.add_column("Executions", style="yellow")
        
        for strategy_name, template_stats in stats.items():
            for template_id, count in template_stats.items():
                table.add_row(strategy_name, template_id, str(count))
        
        console.print(table)
    else:
        console.print("[dim]No executions recorded yet[/dim]")


@cli.command('demo')
@click.option('--strategy', default='simple_qa', help='Strategy to demonstrate')
@click.option('--interactive', is_flag=True, help='Interactive mode')
def run_demo(strategy, interactive):
    """Run a demonstration of the prompt system"""
    if not strategy_manager:
        console.print("[red]Strategy manager not initialized[/red]")
        return
    
    if interactive:
        # Interactive demo
        console.print(f"\n[bold]Interactive Demo - Strategy: {strategy}[/bold]")
        console.print("Type 'quit' to exit\n")
        
        while True:
            query = click.prompt("Query", default="", show_default=False)
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            try:
                result = strategy_manager.execute_strategy(
                    strategy_name=strategy,
                    inputs={"query": query, "context": []}
                )
                
                console.print("\n[bold green]Generated Prompt:[/bold green]")
                console.print(result)
                console.print("\n" + "-" * 50 + "\n")
                
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]\n")
    else:
        # Non-interactive demo
        demo_queries = [
            "What is machine learning?",
            "How do I get started with Python?",
            "Explain quantum computing in simple terms"
        ]
        
        console.print(f"\n[bold]Demo - Strategy: {strategy}[/bold]\n")
        
        for query in demo_queries:
            console.print(f"[bold cyan]Query:[/bold cyan] {query}")
            
            try:
                result = strategy_manager.execute_strategy(
                    strategy_name=strategy,
                    inputs={"query": query, "context": []}
                )
                
                console.print("[bold green]Generated Prompt:[/bold green]")
                console.print(result[:200] + "..." if len(result) > 200 else result)
                console.print("\n" + "-" * 50 + "\n")
                
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]\n")


if __name__ == '__main__':
    cli()