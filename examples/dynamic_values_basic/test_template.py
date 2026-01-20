#!/usr/bin/env python3
"""Demo script for the TemplateService.

This script demonstrates the template engine's ability to resolve
dynamic variables in strings and objects.

Run with:
    cd /path/to/llamafarm
    uv run python examples/dynamic_values_basic/test_template.py
"""

import sys
from pathlib import Path

# Add server to path for imports
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "server"))

from services.template_service import TemplateError, TemplateService


def demo_basic_substitution():
    """Demonstrate basic variable substitution."""
    print("=== Basic Substitution ===")

    template = "Hello {{name}}, welcome to {{service}}!"
    variables = {"name": "Alice", "service": "LlamaFarm"}

    result = TemplateService.resolve(template, variables)
    print(f"Template: {template}")
    print(f"Variables: {variables}")
    print(f"Result: {result}")
    print()


def demo_default_values():
    """Demonstrate default value syntax."""
    print("=== Default Values ===")

    template = "User: {{user_name | Guest}}, Role: {{role | user}}"

    # With no variables - uses defaults
    result1 = TemplateService.resolve(template, {})
    print(f"Template: {template}")
    print(f"Variables: {{}}")
    print(f"Result: {result1}")

    # With some variables - mix of provided and defaults
    result2 = TemplateService.resolve(template, {"user_name": "Alice"})
    print(f"Variables: {{'user_name': 'Alice'}}")
    print(f"Result: {result2}")
    print()


def demo_nested_objects():
    """Demonstrate resolution in nested objects."""
    print("=== Nested Objects ===")

    config = {
        "prompt": "You are an assistant for {{company}}.",
        "settings": {
            "temperature": 0.7,
            "model": "{{model_name | gpt-4}}",
        },
        "tools": [
            {"name": "search", "url": "{{api_base}}/search"},
            {"name": "lookup", "url": "{{api_base}}/lookup"},
        ],
    }
    variables = {"company": "Acme Corp", "api_base": "https://api.example.com"}

    result = TemplateService.resolve_object(config, variables)
    print(f"Input config: {config}")
    print(f"Variables: {variables}")
    print(f"Result: {result}")
    print()


def demo_error_handling():
    """Demonstrate error handling for missing variables."""
    print("=== Error Handling ===")

    template = "Hello {{name}}, your ID is {{user_id}}"

    try:
        TemplateService.resolve(template, {"name": "Alice"})
    except TemplateError as e:
        print(f"Template: {template}")
        print(f"Variables: {{'name': 'Alice'}}")
        print(f"Error (expected): {e}")
    print()


def demo_extract_variables():
    """Demonstrate extracting variable info from templates."""
    print("=== Extract Variables ===")

    template = "{{name}} logged in at {{time | now}} from {{location | unknown}}"
    variables = TemplateService.extract_variables(template)

    print(f"Template: {template}")
    print("Extracted variables:")
    for var_name, default in variables:
        if default is not None:
            print(f"  - {var_name} (default: {default})")
        else:
            print(f"  - {var_name} (required)")
    print()


def demo_backwards_compatibility():
    """Demonstrate that static strings pass through unchanged."""
    print("=== Backwards Compatibility ===")

    static_prompts = [
        "You are a helpful assistant.",
        "Answer questions about the codebase.",
        "Use {json: format} for output.",  # Single braces are not templates
    ]

    print("Static prompts (no {{}} markers) pass through unchanged:")
    for prompt in static_prompts:
        result = TemplateService.resolve(prompt, {"unused": "value"})
        assert result == prompt, f"Static prompt was modified: {prompt}"
        print(f"  OK: {prompt[:50]}...")
    print()


if __name__ == "__main__":
    print("LlamaFarm Template Service Demo")
    print("================================\n")

    demo_basic_substitution()
    demo_default_values()
    demo_nested_objects()
    demo_error_handling()
    demo_extract_variables()
    demo_backwards_compatibility()

    print("All demos completed successfully!")
