"""Helper functions for demos."""

import json
from pathlib import Path
from typing import Dict, Any
from prompts.core.engines.template_registry import TemplateRegistry
from prompts.core.engines.template_engine import TemplateEngine
from prompts.core.models.template import PromptTemplate, TemplateType, TemplateMetadata
from strategies import StrategyManager


def setup_demo_environment() -> StrategyManager:
    """Set up the demo environment with loaded templates and demo strategies."""
    # Initialize components
    template_registry = TemplateRegistry()
    template_engine = TemplateEngine()
    
    # Load templates using the new structure
    templates_dir = Path(__file__).parent.parent / "templates"
    template_count = 0
    
    # Iterate through category directories
    for category_dir in templates_dir.iterdir():
        if not category_dir.is_dir() or category_dir.name.startswith('.'):
            continue
            
        # Iterate through template directories within category
        for template_dir in category_dir.iterdir():
            if not template_dir.is_dir() or template_dir.name.startswith('.'):
                continue
                
            # Load template files
            schema_file = template_dir / "schema.json"
            template_file = template_dir / "template.jinja2"
            
            if schema_file.exists() and template_file.exists():
                try:
                    # Load schema
                    with open(schema_file) as f:
                        schema = json.load(f)
                    
                    # Load template content
                    with open(template_file) as f:
                        template_content = f.read()
                    
                    # Create metadata
                    metadata = TemplateMetadata(
                        use_case=schema.get("use_cases", ["general"])[0] if schema.get("use_cases") else "general",
                        description=schema.get("description", "")
                    )
                    
                    # Extract input variables from schema
                    input_vars = list(schema.get("inputs", {}).keys())
                    
                    template = PromptTemplate(
                        template_id=schema.get("template_id", template_dir.name),
                        name=schema.get("name", template_dir.name),
                        template=template_content,
                        type=TemplateType(schema.get("category", category_dir.name)),
                        input_variables=input_vars,
                        metadata=metadata
                    )
                    
                    template_registry.register_template(template)
                    print(f"  + Loaded template: {template.template_id} ({template.type})")
                    template_count += 1
                    
                except Exception as e:
                    print(f"  - Error loading {template_dir}: {e}")
                    continue
    
    print(f"Loaded {template_count} templates.")
    
    # Initialize the strategy manager with loaded templates and demo strategies
    demo_strategies_file = Path(__file__).parent / "demo-strategies.yaml"
    manager = StrategyManager(
        strategies_file=str(demo_strategies_file),
        template_registry=template_registry,
        template_engine=template_engine
    )
    
    # Load available strategies
    strategies = manager.load_strategies()
    print(f"Loaded {len(strategies)} strategies\n")
    
    return manager