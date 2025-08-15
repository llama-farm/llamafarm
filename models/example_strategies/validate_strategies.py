#!/usr/bin/env python3
"""
Validate all example strategies against the schema.
"""

import sys
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import StrategyManager


def validate_strategy_file(file_path: Path) -> tuple[bool, List[str]]:
    """Validate a single strategy file.
    
    Args:
        file_path: Path to strategy YAML file
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    print(f"\nValidating: {file_path.name}")
    print("-" * 40)
    
    errors = []
    
    try:
        # Load the strategy file
        with open(file_path) as f:
            data = yaml.safe_load(f)
        
        if not data:
            errors.append("Empty file")
            return False, errors
        
        # Check for strategies key
        if 'strategies' not in data:
            errors.append("Missing 'strategies' top-level key")
            return False, errors
        
        strategies = data['strategies']
        
        # Check if strategies is a list (array format)
        if not isinstance(strategies, list):
            errors.append("'strategies' must be an array/list")
            return False, errors
        
        # Validate each strategy
        for i, strategy_config in enumerate(strategies):
            strategy_name = strategy_config.get('name', f'Strategy_{i}')
            print(f"  Strategy: {strategy_name}")
            
            # Check required fields
            required_fields = ['name', 'description', 'components']
            for field in required_fields:
                if field not in strategy_config:
                    errors.append(f"    - Missing required field: {field}")
            
            # Validate components
            if 'components' in strategy_config:
                components = strategy_config['components']
                
                # Check each component
                for comp_name, comp_config in components.items():
                    if 'type' not in comp_config:
                        errors.append(f"    - Component '{comp_name}' missing 'type'")
                    if 'config' not in comp_config:
                        errors.append(f"    - Component '{comp_name}' missing 'config'")
                    
                    # Validate component type
                    valid_types = [
                        'openai_compatible', 'ollama', 'huggingface', 
                        'vllm', 'tgi', 'pytorch', 'llamafactory'
                    ]
                    if 'type' in comp_config and comp_config['type'] not in valid_types:
                        errors.append(f"    - Invalid component type: {comp_config['type']}")
            
            # Validate fallback chain if present
            if 'fallback_chain' in strategy_config:
                chain = strategy_config['fallback_chain']
                if not isinstance(chain, list):
                    errors.append("    - fallback_chain must be a list")
                else:
                    for i, item in enumerate(chain):
                        if not isinstance(item, dict):
                            errors.append(f"    - fallback_chain[{i}] must be a dict")
                        elif 'provider' not in item and 'type' not in item:
                            errors.append(f"    - fallback_chain[{i}] missing provider/type")
            
            # Validate routing rules if present
            if 'routing_rules' in strategy_config:
                rules = strategy_config['routing_rules']
                if not isinstance(rules, list):
                    errors.append("    - routing_rules must be a list")
                else:
                    for i, rule in enumerate(rules):
                        if not isinstance(rule, dict):
                            errors.append(f"    - routing_rules[{i}] must be a dict")
                        elif 'pattern' not in rule:
                            errors.append(f"    - routing_rules[{i}] missing pattern")
                        elif 'provider' not in rule:
                            errors.append(f"    - routing_rules[{i}] missing provider")
        
        if not errors:
            print("  ✅ Valid")
            return True, []
        else:
            print("  ❌ Invalid:")
            for error in errors:
                print(f"    {error}")
            return False, errors
            
    except yaml.YAMLError as e:
        errors.append(f"YAML parsing error: {e}")
        print(f"  ❌ YAML Error: {e}")
        return False, errors
    except Exception as e:
        errors.append(f"Unexpected error: {e}")
        print(f"  ❌ Error: {e}")
        return False, errors


def validate_with_strategy_manager(file_path: Path) -> bool:
    """Validate using StrategyManager to ensure compatibility.
    
    Args:
        file_path: Path to strategy YAML file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Create a temporary StrategyManager with the file
        manager = StrategyManager(strategies_file=file_path)
        
        # Try to load each strategy
        with open(file_path) as f:
            data = yaml.safe_load(f)
        
        if data and 'strategies' in data:
            for strategy_config in data['strategies']:
                strategy_name = strategy_config.get('name', 'unnamed')
                strategy = manager.get_strategy(strategy_name)
                if not strategy:
                    print(f"    - Failed to load strategy: {strategy_name}")
                    return False
        
        return True
        
    except Exception as e:
        print(f"    - StrategyManager validation failed: {e}")
        return False


def main():
    """Validate all strategy files in the example_strategies directory."""
    print("=" * 60)
    print("Example Strategies Validation")
    print("=" * 60)
    
    # Get all YAML files in the directory
    strategy_dir = Path(__file__).parent
    yaml_files = list(strategy_dir.glob("*.yaml"))
    
    if not yaml_files:
        print("No YAML files found in example_strategies/")
        return 1
    
    print(f"\nFound {len(yaml_files)} strategy files to validate")
    
    # Track results
    results = []
    all_valid = True
    
    # Validate each file
    for file_path in sorted(yaml_files):
        is_valid, errors = validate_strategy_file(file_path)
        
        # Also validate with StrategyManager
        if is_valid:
            print("  Testing with StrategyManager...")
            manager_valid = validate_with_strategy_manager(file_path)
            if not manager_valid:
                is_valid = False
                errors.append("Failed StrategyManager validation")
        
        results.append({
            'file': file_path.name,
            'valid': is_valid,
            'errors': errors
        })
        
        if not is_valid:
            all_valid = False
    
    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    valid_count = sum(1 for r in results if r['valid'])
    invalid_count = len(results) - valid_count
    
    print(f"\n✅ Valid strategies: {valid_count}")
    print(f"❌ Invalid strategies: {invalid_count}")
    
    if invalid_count > 0:
        print("\nInvalid files:")
        for result in results:
            if not result['valid']:
                print(f"  - {result['file']}")
                for error in result['errors']:
                    print(f"      {error}")
    
    print("\n" + "=" * 60)
    
    if all_valid:
        print("SUCCESS: All strategies are valid! ✅")
        return 0
    else:
        print("FAILED: Some strategies have validation errors ❌")
        return 1


if __name__ == "__main__":
    sys.exit(main())