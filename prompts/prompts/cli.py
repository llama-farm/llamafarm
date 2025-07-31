"""Command-line interface for the LlamaFarm Prompts system."""

import json
import uuid
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
    default='config/default_prompts.json',
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


if __name__ == '__main__':
    cli()