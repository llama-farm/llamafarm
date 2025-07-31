"""Command-line interface for the LlamaFarm Prompts system."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import yaml
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn

from .core.prompt_system import PromptSystem
from .models.config import PromptConfig, GlobalPromptConfig
from .models.context import PromptContext
from .models.template import PromptTemplate, TemplateType, TemplateComplexity, TemplateMetadata
from .models.strategy import PromptStrategy, StrategyType, StrategyRule, RuleOperator


console = Console()

# Global prompt system instance
prompt_system: Optional[PromptSystem] = None
config_path: Optional[str] = None


def load_prompt_system(config_file: str) -> PromptSystem:
    """Load the prompt system from configuration."""
    global prompt_system, config_path
    
    if not Path(config_file).exists():
        console.print(f"[red]Configuration file not found: {config_file}[/red]")
        raise click.Abort()
    
    try:
        config = PromptConfig.from_file(config_file)
        prompt_system = PromptSystem(config)
        config_path = config_file
        return prompt_system
    except Exception as e:
        console.print(f"[red]Error loading configuration: {str(e)}[/red]")
        raise click.Abort()


@click.group()
@click.option(
    '--config', 
    default='config/default_prompts.yaml',
    help='Path to prompts configuration file',
    show_default=True
)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def cli(config: str, verbose: bool):
    """LlamaFarm Prompts Management System CLI"""
    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    # Load prompt system
    load_prompt_system(config)
    console.print(f"[green]✓[/green] Loaded prompts system from {config}")


# =============================================================================
# TEMPLATE MANAGEMENT COMMANDS
# =============================================================================

@cli.group()
def template():
    """Template management commands"""
    pass


@template.command('list')
@click.option('--type', help='Filter by template type')
@click.option('--domain', help='Filter by domain')
@click.option('--complexity', help='Filter by complexity level')
@click.option('--tag', help='Filter by tag')
@click.option('--format', 'output_format', default='table', type=click.Choice(['table', 'json', 'yaml']))
def list_templates(type, domain, complexity, tag, output_format):
    """List all available templates"""
    if not prompt_system:
        console.print("[red]Prompt system not loaded[/red]")
        return
    
    # Build filter
    filter_by = {}
    if type:
        filter_by['type'] = TemplateType(type)
    if domain:
        filter_by['domain'] = domain
    if complexity:
        filter_by['complexity'] = TemplateComplexity(complexity)
    if tag:
        filter_by['tags'] = [tag]
    
    templates = prompt_system.list_templates(filter_by)
    
    if output_format == 'json':
        template_data = [t.to_dict() for t in templates]
        console.print(json.dumps(template_data, indent=2, default=str))
    elif output_format == 'yaml':
        template_data = [t.to_dict() for t in templates]
        console.print(yaml.dump(template_data, default_flow_style=False))
    else:
        # Table format
        table = Table(title=f"Prompt Templates ({len(templates)} found)")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Type", style="yellow")
        table.add_column("Domain", style="blue")
        table.add_column("Complexity", style="magenta")
        table.add_column("Variables", style="white")
        
        for template in templates:
            variables = ", ".join(template.input_variables[:3])
            if len(template.input_variables) > 3:
                variables += f" (+{len(template.input_variables) - 3} more)"
            
            table.add_row(
                template.template_id,
                template.name,
                template.type.value,
                template.metadata.domain,
                template.metadata.complexity.value,
                variables or "None"
            )
        
        console.print(table)


@template.command('show')
@click.argument('template_id')
@click.option('--show-content', is_flag=True, help='Show template content')
def show_template(template_id, show_content):
    """Show details of a specific template"""
    if not prompt_system:
        console.print("[red]Prompt system not loaded[/red]")
        return
    
    template = prompt_system.get_template(template_id)
    if not template:
        console.print(f"[red]Template not found: {template_id}[/red]")
        return
    
    # Basic info
    console.print(Panel(f"[bold]{template.name}[/bold]", title="Template Details"))
    
    info_table = Table(show_header=False, box=None)
    info_table.add_column("Key", style="cyan")
    info_table.add_column("Value", style="white")
    
    info_table.add_row("ID", template.template_id)
    info_table.add_row("Type", template.type.value)
    info_table.add_row("Domain", template.metadata.domain)
    info_table.add_row("Complexity", template.metadata.complexity.value)
    info_table.add_row("Use Case", template.metadata.use_case)
    info_table.add_row("Variables", ", ".join(template.input_variables))
    info_table.add_row("Optional Variables", ", ".join(template.optional_variables))
    info_table.add_row("Tags", ", ".join(template.metadata.tags))
    
    if template.metadata.description:
        info_table.add_row("Description", template.metadata.description)
    
    console.print(info_table)
    
    # Show template content if requested
    if show_content:
        console.print("\\n[bold]Template Content:[/bold]")
        syntax = Syntax(template.template, "jinja2", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title="Template"))


@template.command('search')
@click.argument('query')
@click.option('--limit', default=10, help='Maximum number of results')
def search_templates(query, limit):
    """Search templates by query"""
    if not prompt_system:
        console.print("[red]Prompt system not loaded[/red]")
        return
    
    templates = prompt_system.search_templates(query)[:limit]
    
    if not templates:
        console.print(f"[yellow]No templates found matching '{query}'[/yellow]")
        return
    
    table = Table(title=f"Search Results for '{query}' ({len(templates)} found)")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Type", style="yellow")
    table.add_column("Domain", style="blue")
    table.add_column("Use Case", style="magenta")
    
    for template in templates:
        table.add_row(
            template.template_id,
            template.name,
            template.type.value,
            template.metadata.domain,
            template.metadata.use_case
        )
    
    console.print(table)


@template.command('create')
@click.option('--interactive', '-i', is_flag=True, help='Interactive template creation')
@click.option('--from-file', help='Create template from JSON/YAML file')
def create_template(interactive, from_file):
    """Create a new template"""
    if not prompt_system:
        console.print("[red]Prompt system not loaded[/red]")
        return
    
    if from_file:
        try:
            template = PromptTemplate.from_file(from_file)
            success = prompt_system.add_template(template)
            
            if success:
                console.print(f"[green]✓ Template '{template.template_id}' created successfully[/green]")
            else:
                console.print("[red]Failed to create template[/red]")
        except Exception as e:
            console.print(f"[red]Error creating template: {str(e)}[/red]")
        return
    
    if interactive:
        # Interactive template creation
        console.print("[bold]Interactive Template Creation[/bold]")
        
        template_id = click.prompt("Template ID")
        name = click.prompt("Template Name")
        template_type = click.prompt(
            "Template Type", 
            type=click.Choice([t.value for t in TemplateType])
        )
        domain = click.prompt("Domain", default="general")
        use_case = click.prompt("Use Case")
        
        console.print("\\nEnter template content (end with Ctrl+D or Ctrl+Z):")
        template_content = click.get_text_stream('stdin').read()
        
        input_variables = []
        console.print("\\nEnter input variables (one per line, empty line to finish):")
        while True:
            var = click.prompt("Variable", default="", show_default=False)
            if not var:
                break
            input_variables.append(var)
        
        # Create template
        template = PromptTemplate(
            template_id=template_id,
            name=name,
            type=TemplateType(template_type),
            template=template_content,
            input_variables=input_variables,
            metadata=TemplateMetadata(
                use_case=use_case,
                domain=domain
            )
        )
        
        success = prompt_system.add_template(template)
        
        if success:
            console.print(f"[green]✓ Template '{template_id}' created successfully[/green]")
        else:
            console.print("[red]Failed to create template[/red]")
    else:
        console.print("Use --interactive or --from-file option")


@template.command('test')
@click.argument('template_id')
@click.option('--variables', help='JSON string of test variables')
@click.option('--variables-file', help='JSON/YAML file with test variables')
def test_template(template_id, variables, variables_file):
    """Test a template with sample variables"""
    if not prompt_system:
        console.print("[red]Prompt system not loaded[/red]")
        return
    
    # Get test variables
    test_vars = {}
    
    if variables_file:
        with open(variables_file, 'r') as f:
            if variables_file.endswith('.yaml') or variables_file.endswith('.yml'):
                test_vars = yaml.safe_load(f)
            else:
                test_vars = json.load(f)
    elif variables:
        test_vars = json.loads(variables)
    else:
        # Get template and prompt for variables
        template = prompt_system.get_template(template_id)
        if not template:
            console.print(f"[red]Template not found: {template_id}[/red]")
            return
        
        console.print(f"Enter values for template variables:")
        for var in template.input_variables:
            value = click.prompt(f"{var}")
            test_vars[var] = value
    
    # Test template
    context = PromptContext(query="test query")
    exec_context = prompt_system.test_template(template_id, test_vars, context)
    
    if exec_context.has_errors():
        console.print("[red]Template test failed:[/red]")
        for error in exec_context.errors:
            console.print(f"  • {error}")
    else:
        console.print("[green]✓ Template test successful[/green]")
        console.print("\\n[bold]Rendered Prompt:[/bold]")
        console.print(Panel(exec_context.rendered_prompt, title="Test Result"))


@template.command('validate')
@click.argument('template_id', required=False)
@click.option('--all', is_flag=True, help='Validate all templates')
def validate_templates(template_id, all):
    """Validate template(s)"""
    if not prompt_system:
        console.print("[red]Prompt system not loaded[/red]")
        return
    
    if all:
        validation_results = prompt_system.template_registry.validate_all_templates()
        
        if not validation_results:
            console.print("[green]✓ All templates are valid[/green]")
        else:
            console.print(f"[red]Found validation errors in {len(validation_results)} templates:[/red]")
            for tid, errors in validation_results.items():
                console.print(f"\\n[yellow]{tid}:[/yellow]")
                for error in errors:
                    console.print(f"  • {error}")
    
    elif template_id:
        template = prompt_system.get_template(template_id)
        if not template:
            console.print(f"[red]Template not found: {template_id}[/red]")
            return
        
        errors = prompt_system.validate_template(template)
        
        if not errors:
            console.print(f"[green]✓ Template '{template_id}' is valid[/green]")
        else:
            console.print(f"[red]Validation errors in '{template_id}':[/red]")
            for error in errors:
                console.print(f"  • {error}")
    else:
        console.print("Specify --all or a template ID")


# =============================================================================
# EXECUTION COMMANDS
# =============================================================================

@cli.command('execute')
@click.argument('query')
@click.option('--template', help='Force specific template')
@click.option('--strategy', help='Force specific strategy')
@click.option('--variables', help='JSON string of additional variables')
@click.option('--context-file', help='JSON/YAML file with context data')
@click.option('--show-details', is_flag=True, help='Show execution details')
def execute_prompt(query, template, strategy, variables, context_file, show_details):
    """Execute a prompt query"""
    if not prompt_system:
        console.print("[red]Prompt system not loaded[/red]")
        return
    
    # Prepare context
    context = PromptContext(query=query)
    
    if context_file:
        with open(context_file, 'r') as f:
            if context_file.endswith('.yaml') or context_file.endswith('.yml'):
                context_data = yaml.safe_load(f)
            else:
                context_data = json.load(f)
        
        # Update context with file data
        for key, value in context_data.items():
            if hasattr(context, key):
                setattr(context, key, value)
            else:
                context.add_custom_attribute(key, value)
    
    # Prepare variables
    vars_dict = {}
    if variables:
        vars_dict = json.loads(variables)
    
    # Execute with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Executing prompt...", total=None)
        
        exec_context = prompt_system.execute_prompt(
            query=query,
            context=context,
            variables=vars_dict,
            strategy_override=strategy,
            template_override=template
        )
    
    # Show results
    if exec_context.has_errors():
        console.print("[red]Execution failed:[/red]")
        for error in exec_context.errors:
            console.print(f"  • {error}")
        return
    
    console.print("[green]✓ Execution successful[/green]")
    
    if show_details:
        details_table = Table(show_header=False, box=None)
        details_table.add_column("Key", style="cyan")
        details_table.add_column("Value", style="white")
        
        details_table.add_row("Template Used", exec_context.selected_template_id or "None")
        details_table.add_row("Strategy Used", exec_context.selected_strategy_id or "None")
        details_table.add_row("Execution Time", f"{exec_context.total_duration_ms}ms")
        details_table.add_row("Global Prompts", ", ".join(exec_context.applied_global_prompts))
        
        console.print("\\n[bold]Execution Details:[/bold]")
        console.print(details_table)
    
    console.print("\\n[bold]Rendered Prompt:[/bold]")
    console.print(Panel(exec_context.rendered_prompt, title="Result"))


# =============================================================================
# STRATEGY COMMANDS
# =============================================================================

@cli.group()
def strategy():
    """Strategy management commands"""
    pass


@strategy.command('list')
@click.option('--format', 'output_format', default='table', type=click.Choice(['table', 'json']))
def list_strategies(output_format):
    """List all strategies"""
    if not prompt_system:
        console.print("[red]Prompt system not loaded[/red]")
        return
    
    strategies = prompt_system.strategy_engine.list_strategies()
    
    if output_format == 'json':
        strategy_data = [s.dict() for s in strategies]
        console.print(json.dumps(strategy_data, indent=2, default=str))
    else:
        table = Table(title=f"Prompt Strategies ({len(strategies)} found)")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Type", style="yellow")
        table.add_column("Rules", style="blue")
        table.add_column("Enabled", style="magenta")
        
        for strategy in strategies:
            table.add_row(
                strategy.strategy_id,
                strategy.name,
                strategy.type.value,
                str(len(strategy.rules)),
                "✓" if strategy.enabled else "✗"
            )
        
        console.print(table)


@strategy.command('test')
@click.argument('strategy_id')
@click.option('--test-file', help='JSON/YAML file with test contexts')
def test_strategy(strategy_id, test_file):
    """Test a strategy with sample contexts"""
    if not prompt_system:
        console.print("[red]Prompt system not loaded[/red]")
        return
    
    # Load test contexts
    test_contexts = []
    
    if test_file:
        with open(test_file, 'r') as f:
            if test_file.endswith('.yaml') or test_file.endswith('.yml'):
                test_contexts = yaml.safe_load(f)
            else:
                test_contexts = json.load(f)
    else:
        # Default test contexts
        test_contexts = [
            {"query_type": "qa", "domain": "general"},
            {"query_type": "summary", "domain": "technical"},
            {"user_role": "expert", "complexity_level": "high"},
        ]
    
    # Test strategy
    results = prompt_system.strategy_engine.test_strategy(strategy_id, test_contexts)
    
    if "error" in results:
        console.print(f"[red]{results['error']}[/red]")
        return
    
    # Show results
    console.print(f"[bold]Strategy Test Results: {strategy_id}[/bold]")
    
    summary = results["summary"]
    console.print(f"\\nTotal Tests: {summary['total_tests']}")
    console.print(f"Successful: {summary['successful_selections']}")
    console.print(f"Fallbacks: {summary['fallback_uses']}")
    console.print(f"Errors: {summary['errors']}")
    
    # Show individual test cases
    table = Table(title="Test Cases")
    table.add_column("Case", style="cyan")
    table.add_column("Context", style="white")
    table.add_column("Selected Template", style="green")
    table.add_column("Status", style="yellow")
    
    for test_case in results["test_cases"]:
        context_str = json.dumps(test_case["context"], separators=(',', ':'))[:50] + "..."
        status = "✓" if test_case["success"] else "✗"
        if test_case["success"] and test_case.get("used_fallback"):
            status += " (fallback)"
        
        table.add_row(
            str(test_case["test_case"]),
            context_str,
            test_case.get("selected_template", "None"),
            status
        )
    
    console.print(table)


# =============================================================================
# GLOBAL PROMPTS COMMANDS
# =============================================================================

@cli.group()
def global_prompt():
    """Global prompt management commands"""
    pass


@global_prompt.command('list')
def list_global_prompts():
    """List all global prompts"""
    if not prompt_system:
        console.print("[red]Prompt system not loaded[/red]")
        return
    
    global_prompts = prompt_system.global_prompt_manager.list_global_prompts()
    
    table = Table(title=f"Global Prompts ({len(global_prompts)} found)")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Type", style="yellow")
    table.add_column("Priority", style="blue")
    table.add_column("Applies To", style="magenta")
    table.add_column("Enabled", style="white")
    
    for gp in global_prompts:
        # Determine type
        gp_type = []
        if gp.system_prompt:
            gp_type.append("system")
        if gp.prefix_prompt:
            gp_type.append("prefix")
        if gp.suffix_prompt:
            gp_type.append("suffix")
        
        applies_to = ", ".join(gp.applies_to[:3])
        if len(gp.applies_to) > 3:
            applies_to += f" (+{len(gp.applies_to) - 3})"
        
        table.add_row(
            gp.global_id,
            gp.name,
            ", ".join(gp_type),
            str(gp.priority),
            applies_to,
            "✓" if gp.enabled else "✗"
        )
    
    console.print(table)


@global_prompt.command('create')
@click.option('--id', 'global_id', required=True, help='Global prompt ID')
@click.option('--name', required=True, help='Global prompt name')
@click.option('--system', help='System prompt content')
@click.option('--prefix', help='Prefix prompt content')
@click.option('--suffix', help='Suffix prompt content')
@click.option('--applies-to', multiple=True, help='Template patterns (can specify multiple)')
@click.option('--priority', default=100, help='Priority (lower = higher priority)')
def create_global_prompt(global_id, name, system, prefix, suffix, applies_to, priority):
    """Create a new global prompt"""
    if not prompt_system:
        console.print("[red]Prompt system not loaded[/red]")
        return
    
    # Validate inputs
    if not any([system, prefix, suffix]):
        console.print("[red]At least one of --system, --prefix, or --suffix is required[/red]")
        return
    
    # Create global prompt
    global_prompt = GlobalPromptConfig(
        global_id=global_id,
        name=name,
        system_prompt=system,
        prefix_prompt=prefix,
        suffix_prompt=suffix,
        applies_to=list(applies_to) if applies_to else ["*"],
        priority=priority
    )
    
    # Validate
    errors = prompt_system.global_prompt_manager.validate_global_prompt(global_prompt)
    if errors:
        console.print("[red]Validation errors:[/red]")
        for error in errors:
            console.print(f"  • {error}")
        return
    
    # Add to system
    prompt_system.global_prompt_manager.add_global_prompt(global_prompt)
    console.print(f"[green]✓ Global prompt '{global_id}' created successfully[/green]")


# =============================================================================
# SYSTEM COMMANDS
# =============================================================================

@cli.command('stats')
def show_stats():
    """Show system statistics"""
    if not prompt_system:
        console.print("[red]Prompt system not loaded[/red]")
        return
    
    stats = prompt_system.get_system_stats()
    
    console.print("[bold]System Statistics[/bold]")
    
    stats_table = Table(show_header=False, box=None)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="white")
    
    stats_table.add_row("Total Executions", str(stats["total_executions"]))
    stats_table.add_row("Total Errors", str(stats["total_errors"]))
    stats_table.add_row("Total Fallbacks", str(stats["total_fallbacks"]))
    stats_table.add_row("Error Rate", f"{stats['error_rate']:.2%}")
    stats_table.add_row("Fallback Rate", f"{stats['fallback_rate']:.2%}")
    stats_table.add_row("Templates", str(stats["templates_count"]))
    stats_table.add_row("Strategies", str(stats["strategies_count"]))
    stats_table.add_row("Global Prompts", str(stats["global_prompts_count"]))
    
    console.print(stats_table)


@cli.command('validate-config')
def validate_config():
    """Validate the current configuration"""
    if not prompt_system:
        console.print("[red]Prompt system not loaded[/red]")
        return
    
    errors = prompt_system.config.validate_config()
    
    if not errors:
        console.print("[green]✓ Configuration is valid[/green]")
    else:
        console.print(f"[red]Configuration validation failed ({len(errors)} errors):[/red]")
        for error in errors:
            console.print(f"  • {error}")


# =============================================================================
# TOP-LEVEL UTILITY COMMANDS
# =============================================================================

@cli.command('validate')
@click.option('--all', is_flag=True, help='Validate all components')
@click.option('--templates', is_flag=True, help='Validate templates only')
@click.option('--strategies', is_flag=True, help='Validate strategies only')
@click.option('--config', is_flag=True, help='Validate configuration only')
def validate_system(all, templates, strategies, config):
    """Validate system components"""
    if not prompt_system:
        console.print("[red]Prompt system not loaded[/red]")
        return
    
    if all or not any([templates, strategies, config]):
        # Validate everything
        templates = strategies = config = True
    
    total_errors = 0
    
    if config:
        console.print("[bold]Validating Configuration...[/bold]")
        config_errors = prompt_system.config.validate_config()
        if not config_errors:
            console.print("[green]✓ Configuration is valid[/green]")
        else:
            console.print(f"[red]Configuration errors ({len(config_errors)}):[/red]")
            for error in config_errors:
                console.print(f"  • {error}")
            total_errors += len(config_errors)
        console.print()
    
    if templates:
        console.print("[bold]Validating Templates...[/bold]")
        template_errors = prompt_system.template_registry.validate_all_templates()
        if not template_errors:
            console.print("[green]✓ All templates are valid[/green]")
        else:
            console.print(f"[red]Template errors in {len(template_errors)} templates:[/red]")
            for tid, errors in template_errors.items():
                console.print(f"  {tid}:")
                for error in errors:
                    console.print(f"    • {error}")
                total_errors += len(errors)
        console.print()
    
    if strategies:
        console.print("[bold]Validating Strategies...[/bold]")
        strategies_list = prompt_system.strategy_engine.list_strategies()
        strategy_errors = 0
        for strategy in strategies_list:
            if not strategy.enabled:
                console.print(f"[yellow]⚠️ Strategy '{strategy.strategy_id}' is disabled[/yellow]")
            if not strategy.rules:
                console.print(f"[red]✗ Strategy '{strategy.strategy_id}' has no rules[/red]")
                strategy_errors += 1
        
        if strategy_errors == 0:
            console.print("[green]✓ All strategies are valid[/green]")
        else:
            total_errors += strategy_errors
        console.print()
    
    # Summary
    if total_errors == 0:
        console.print("[green]🎉 System validation passed![/green]")
    else:
        console.print(f"[red]❌ System validation failed with {total_errors} errors[/red]")


@cli.command('test')
@click.option('--template', help='Test specific template')
@click.option('--strategy', help='Test specific strategy')
@click.option('--all-templates', is_flag=True, help='Test all templates')
@click.option('--all-strategies', is_flag=True, help='Test all strategies')
@click.option('--sample-size', default=5, help='Number of test cases per component')
def test_system(template, strategy, all_templates, all_strategies, sample_size):
    """Test system components with sample data"""
    if not prompt_system:
        console.print("[red]Prompt system not loaded[/red]")
        return
    
    if not any([template, strategy, all_templates, all_strategies]):
        console.print("[yellow]Specify what to test: --template, --strategy, --all-templates, or --all-strategies[/yellow]")
        return
    
    test_results = {"passed": 0, "failed": 0, "errors": []}
    
    # Sample test data
    test_queries = [
        "What is machine learning?",
        "Explain the benefits of renewable energy",
        "How does photosynthesis work?",
        "Compare different database types",
        "Analyze the economic impact of AI"
    ]
    
    test_contexts = [
        {"domain": "general", "query_type": "definition"},
        {"domain": "environmental", "query_type": "explanation"},
        {"domain": "science", "query_type": "process"},
        {"domain": "technical", "query_type": "comparison"},
        {"domain": "economic", "query_type": "analysis"}
    ]
    
    if template:
        console.print(f"[bold]Testing Template: {template}[/bold]")
        _test_single_template(prompt_system, template, test_queries[:sample_size], test_results)
    
    elif all_templates:
        console.print("[bold]Testing All Templates...[/bold]")
        templates = prompt_system.list_templates()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Testing {len(templates)} templates...", total=len(templates))
            
            for template_obj in templates:
                _test_single_template(prompt_system, template_obj.template_id, test_queries[:2], test_results)
                progress.advance(task)
    
    if strategy:
        console.print(f"[bold]Testing Strategy: {strategy}[/bold]")
        _test_single_strategy(prompt_system, strategy, test_contexts[:sample_size], test_results)
    
    elif all_strategies:
        console.print("[bold]Testing All Strategies...[/bold]")
        strategies = prompt_system.strategy_engine.list_strategies()
        
        for strategy_obj in strategies:
            _test_single_strategy(prompt_system, strategy_obj.strategy_id, test_contexts[:2], test_results)
    
    # Show results
    console.print(f"\n[bold]Test Results:[/bold]")
    console.print(f"  Passed: [green]{test_results['passed']}[/green]")
    console.print(f"  Failed: [red]{test_results['failed']}[/red]")
    
    if test_results['errors']:
        console.print(f"\n[red]Errors encountered:[/red]")
        for error in test_results['errors'][:10]:  # Show first 10 errors
            console.print(f"  • {error}")
        if len(test_results['errors']) > 10:
            console.print(f"  ... and {len(test_results['errors']) - 10} more errors")


def _test_single_template(prompt_system, template_id: str, queries: List[str], results: dict):
    """Test a single template with multiple queries."""
    for query in queries:
        try:
            context = PromptContext(query=query)
            exec_context = prompt_system.execute_prompt(
                query=query,
                context=context,
                template_override=template_id
            )
            
            if exec_context.has_errors():
                results['failed'] += 1
                results['errors'].extend(exec_context.errors)
            else:
                results['passed'] += 1
                
        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"Template {template_id}: {str(e)}")


def _test_single_strategy(prompt_system, strategy_id: str, contexts: List[dict], results: dict):
    """Test a single strategy with multiple contexts."""
    try:
        strategy_results = prompt_system.strategy_engine.test_strategy(strategy_id, contexts)
        
        if "error" in strategy_results:
            results['failed'] += 1
            results['errors'].append(f"Strategy {strategy_id}: {strategy_results['error']}")
        else:
            summary = strategy_results.get("summary", {})
            results['passed'] += summary.get('successful_selections', 0)
            results['failed'] += summary.get('errors', 0)
            
    except Exception as e:
        results['failed'] += 1
        results['errors'].append(f"Strategy {strategy_id}: {str(e)}")


@cli.command('evaluate')
@click.argument('response_text')
@click.option('--query', required=True, help='Original query that generated the response')
@click.option('--criteria', default='accuracy,relevance,completeness', help='Evaluation criteria (comma-separated)')
@click.option('--context', help='JSON string with context documents')
@click.option('--template', default='llm_judge', help='Evaluation template to use')
@click.option('--output-format', default='detailed', type=click.Choice(['detailed', 'score', 'json']))
def evaluate_response(response_text, query, criteria, context, template, output_format):
    """Evaluate an AI response using built-in evaluation templates"""
    if not prompt_system:
        console.print("[red]Prompt system not loaded[/red]")
        return
    
    # Prepare context documents
    context_docs = []
    if context:
        try:
            context_data = json.loads(context)
            if isinstance(context_data, list):
                context_docs = context_data
            else:
                context_docs = [context_data]
        except json.JSONDecodeError:
            console.print("[red]Invalid JSON in context parameter[/red]")
            return
    
    # Prepare evaluation variables based on template
    variables = {
        "original_query": query,
        "response_to_evaluate": response_text,
        "evaluation_criteria": criteria,
        "context": context_docs
    }
    
    # Add template-specific variables
    if template == "rag_evaluation":
        variables.update({
            "query": query,
            "retrieved_docs": context_docs,
            "generated_response": response_text
        })
    elif template == "response_scoring":
        variables.update({
            "query": query,
            "response": response_text,  
            "scoring_criteria": criteria
        })
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("Evaluating response..."),
            console=console
        ) as progress:
            task = progress.add_task("Processing...", total=None)
            
            exec_context = prompt_system.execute_prompt(
                query=f"Evaluate this response: {response_text[:100]}...",
                variables=variables,
                template_override=template
            )
        
        if exec_context.has_errors():
            console.print("[red]Evaluation failed:[/red]")
            for error in exec_context.errors:
                console.print(f"  • {error}")
            return
        
        # Display results based on format
        if output_format == 'json':
            result = {
                "query": query,
                "response": response_text,
                "criteria": criteria,
                "template_used": template,
                "evaluation": exec_context.rendered_prompt
            }
            console.print(json.dumps(result, indent=2))
            
        elif output_format == 'score':
            # Try to extract score from evaluation (basic heuristic)
            evaluation = exec_context.rendered_prompt.lower()
            if '/10' in evaluation or '/100' in evaluation:
                import re
                scores = re.findall(r'(\d+(?:\.\d+)?)\s*/\s*(\d+)', evaluation)
                if scores:
                    console.print(f"Score: {scores[0][0]}/{scores[0][1]}")
                else:
                    console.print("Score: Could not extract numeric score")
            else:
                console.print("Score: No numeric score found")
                
        else:  # detailed
            console.print(f"[bold]Evaluation Results for Template: {template}[/bold]\n")
            console.print(f"[cyan]Original Query:[/cyan] {query}")
            console.print(f"[cyan]Evaluation Criteria:[/cyan] {criteria}")
            console.print(f"[cyan]Response Length:[/cyan] {len(response_text)} characters\n")
            
            console.print("[bold]Evaluation:[/bold]")
            console.print(Panel(exec_context.rendered_prompt, title="Assessment"))
    
    except Exception as e:
        console.print(f"[red]Evaluation error: {str(e)}[/red]")


@cli.command('benchmark')
@click.option('--templates', is_flag=True, help='Benchmark template performance')
@click.option('--strategies', is_flag=True, help='Benchmark strategy performance')
@click.option('--system', is_flag=True, help='Benchmark overall system performance')
@click.option('--iterations', default=10, help='Number of iterations per test')
@click.option('--output', default='table', type=click.Choice(['table', 'json', 'csv']))
def benchmark_system(templates, strategies, system, iterations, output):
    """Benchmark system performance and generate metrics"""
    if not prompt_system:
        console.print("[red]Prompt system not loaded[/red]")
        return
    
    if not any([templates, strategies, system]):
        # Default to system benchmark
        system = True
    
    benchmark_results = {
        "timestamp": datetime.now().isoformat(),
        "iterations": iterations,
        "results": {}
    }
    
    # Sample queries for benchmarking
    benchmark_queries = [
        "What is artificial intelligence?",
        "Explain machine learning concepts",
        "Compare different programming languages",
        "Analyze economic trends",
        "Summarize scientific research"
    ]
    
    if templates:
        console.print("[bold]Benchmarking Templates...[/bold]")
        template_results = _benchmark_templates(prompt_system, benchmark_queries, iterations)
        benchmark_results["results"]["templates"] = template_results
    
    if strategies:
        console.print("[bold]Benchmarking Strategies...[/bold]") 
        strategy_results = _benchmark_strategies(prompt_system, benchmark_queries, iterations)
        benchmark_results["results"]["strategies"] = strategy_results
    
    if system:
        console.print("[bold]Benchmarking System Performance...[/bold]")
        system_results = _benchmark_system_performance(prompt_system, benchmark_queries, iterations)
        benchmark_results["results"]["system"] = system_results
    
    # Output results
    if output == 'json':
        console.print(json.dumps(benchmark_results, indent=2, default=str))
    elif output == 'csv':
        _output_benchmark_csv(benchmark_results)
    else:
        _output_benchmark_table(benchmark_results)


def _benchmark_templates(prompt_system, queries: List[str], iterations: int) -> dict:
    """Benchmark template performance."""
    templates = prompt_system.list_templates()
    results = {}
    
    with Progress(console=console) as progress:
        task = progress.add_task("Benchmarking templates...", total=len(templates))
        
        for template in templates:
            template_id = template.template_id
            total_time = 0
            success_count = 0
            
            for i in range(iterations):
                query = queries[i % len(queries)]
                start_time = datetime.now()
                
                try:
                    exec_context = prompt_system.execute_prompt(
                        query=query,
                        template_override=template_id
                    )
                    
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds() * 1000
                    
                    if not exec_context.has_errors():
                        total_time += duration
                        success_count += 1
                        
                except Exception:
                    pass
            
            if success_count > 0:
                results[template_id] = {
                    "avg_time_ms": total_time / success_count,
                    "success_rate": success_count / iterations,
                    "total_attempts": iterations
                }
            
            progress.advance(task)
    
    return results


def _benchmark_strategies(prompt_system, queries: List[str], iterations: int) -> dict:
    """Benchmark strategy performance."""
    strategies = prompt_system.strategy_engine.list_strategies()
    results = {}
    
    for strategy in strategies:
        strategy_id = strategy.strategy_id
        total_time = 0
        success_count = 0
        
        for i in range(iterations):
            query = queries[i % len(queries)]
            start_time = datetime.now()
            
            try:
                exec_context = prompt_system.execute_prompt(
                    query=query,
                    strategy_override=strategy_id
                )
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds() * 1000
                
                if not exec_context.has_errors():
                    total_time += duration
                    success_count += 1
                    
            except Exception:
                pass
        
        if success_count > 0:
            results[strategy_id] = {
                "avg_time_ms": total_time / success_count,
                "success_rate": success_count / iterations,
                "total_attempts": iterations
            }
    
    return results


def _benchmark_system_performance(prompt_system, queries: List[str], iterations: int) -> dict:
    """Benchmark overall system performance."""
    total_time = 0
    success_count = 0
    error_count = 0
    
    with Progress(console=console) as progress:
        task = progress.add_task("Running system benchmark...", total=iterations)
        
        for i in range(iterations):
            query = queries[i % len(queries)]
            start_time = datetime.now()
            
            try:
                exec_context = prompt_system.execute_prompt(query=query)
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds() * 1000
                
                if exec_context.has_errors():
                    error_count += 1
                else:
                    success_count += 1
                    total_time += duration
                    
            except Exception:
                error_count += 1
            
            progress.advance(task)
    
    return {
        "total_iterations": iterations,
        "successful_executions": success_count,
        "failed_executions": error_count,
        "avg_execution_time_ms": total_time / max(success_count, 1),
        "success_rate": success_count / iterations,
        "error_rate": error_count / iterations
    }


def _output_benchmark_table(results: dict):
    """Output benchmark results as formatted tables."""
    console.print(f"\n[bold]Benchmark Results ({results['timestamp']})[/bold]")
    console.print(f"Iterations per test: {results['iterations']}\n")
    
    for category, data in results["results"].items():
        if category == "system":
            table = Table(title=f"{category.title()} Performance")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")
            
            table.add_row("Total Iterations", str(data["total_iterations"]))
            table.add_row("Successful Executions", str(data["successful_executions"]))
            table.add_row("Failed Executions", str(data["failed_executions"]))
            table.add_row("Avg Execution Time", f"{data['avg_execution_time_ms']:.2f}ms")
            table.add_row("Success Rate", f"{data['success_rate']:.2%}")
            table.add_row("Error Rate", f"{data['error_rate']:.2%}")
            
        else:
            table = Table(title=f"{category.title()} Performance")
            table.add_column("ID", style="cyan")
            table.add_column("Avg Time (ms)", style="yellow")
            table.add_column("Success Rate", style="green")
            table.add_column("Attempts", style="blue")
            
            for item_id, metrics in data.items():
                table.add_row(
                    item_id,
                    f"{metrics['avg_time_ms']:.2f}",
                    f"{metrics['success_rate']:.2%}",
                    str(metrics['total_attempts'])
                )
        
        console.print(table)
        console.print()


def _output_benchmark_csv(results: dict):
    """Output benchmark results as CSV."""
    import csv
    import io
    
    output = io.StringIO()
    
    # System results
    if "system" in results["results"]:
        writer = csv.writer(output)
        writer.writerow(["Category", "Metric", "Value"])
        
        system_data = results["results"]["system"]
        writer.writerow(["System", "Total Iterations", system_data["total_iterations"]])
        writer.writerow(["System", "Successful Executions", system_data["successful_executions"]])
        writer.writerow(["System", "Failed Executions", system_data["failed_executions"]])
        writer.writerow(["System", "Avg Execution Time (ms)", f"{system_data['avg_execution_time_ms']:.2f}"])
        writer.writerow(["System", "Success Rate", f"{system_data['success_rate']:.4f}"])
        writer.writerow(["System", "Error Rate", f"{system_data['error_rate']:.4f}"])
    
    # Component results
    for category in ["templates", "strategies"]:
        if category in results["results"]:
            if output.tell() > 0:
                output.write("\n")
            
            writer = csv.writer(output)
            writer.writerow([f"{category.title()}", "ID", "Avg Time (ms)", "Success Rate", "Attempts"])
            
            for item_id, metrics in results["results"][category].items():
                writer.writerow([
                    category.title(),
                    item_id,
                    f"{metrics['avg_time_ms']:.2f}",
                    f"{metrics['success_rate']:.4f}",
                    metrics['total_attempts']
                ])
    
    console.print(output.getvalue())


if __name__ == '__main__':
    cli()