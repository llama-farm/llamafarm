#!/usr/bin/env python3
"""
Generate configuration files from individual template files.

This script builds the main configuration file by loading all individual
template files and combining them with strategies and global prompts.
"""

import argparse
from pathlib import Path

from prompts.utils.config_builder import build_default_config


def main():
    """Generate configuration files."""
    parser = argparse.ArgumentParser(description="Generate prompt system configuration")
    parser.add_argument(
        "--templates-dir", 
        default="templates",
        help="Directory containing template files"
    )
    parser.add_argument(
        "--output",
        default="config/default_prompts.yaml",
        help="Output configuration file"
    )
    parser.add_argument(
        "--format",
        choices=["json", "yaml"],
        default="yaml",
        help="Output format"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate generated configuration"
    )
    
    args = parser.parse_args()
    
    print(f"🔧 Generating configuration from {args.templates_dir}")
    
    try:
        # Build configuration
        config = build_default_config(
            templates_dir=args.templates_dir,
            output_file=args.output
        )
        
        print(f"✅ Generated configuration: {args.output}")
        
        # Show summary
        templates = config.templates
        strategies = config.strategies
        global_prompts = config.global_prompts
        
        print(f"   📝 Templates: {len(templates)}")
        print(f"   🧠 Strategies: {len(strategies)}")
        print(f"   🌍 Global Prompts: {len(global_prompts)}")
        
        # Show templates by category
        by_category = {}
        for template in templates.values():
            category = template.type.value
            by_category[category] = by_category.get(category, 0) + 1
        
        print("   📊 Templates by category:")
        for category, count in sorted(by_category.items()):
            print(f"      {category}: {count}")
        
        # Validate if requested
        if args.validate:
            print("\n🔍 Validating configuration...")
            errors = config.validate_config()
            
            if not errors:
                print("✅ Configuration is valid")
            else:
                print(f"❌ Configuration has {len(errors)} errors:")
                for error in errors:
                    print(f"   • {error}")
                return 1
        
        print("\n🎯 Configuration ready for use!")
        print(f"   Load with: PromptConfig.from_file('{args.output}')")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error generating configuration: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())