#!/usr/bin/env python3
"""Test script to verify YAML configuration loading works."""

import sys
from pathlib import Path

# Add prompts to Python path
sys.path.insert(0, str(Path(__file__).parent))

from prompts.utils.config_loader import load_and_validate_config

def test_yaml_config(config_path: str):
    """Test loading a YAML configuration file."""
    print(f"\n🧪 Testing: {config_path}")
    try:
        config = load_and_validate_config(config_path)
        print(f"✅ Successfully loaded {config_path}")
        
        # Print basic info about the config
        if "name" in config:
            print(f"   Name: {config['name']}")
        
        if "prompts" in config:
            prompts = config["prompts"]
            if "strategy" in prompts:
                print(f"   Strategy: {prompts['strategy']}")
            
            if "templates" in prompts:
                template_count = len(prompts["templates"])
                print(f"   Templates: {template_count}")
                
        return True
        
    except Exception as e:
        print(f"❌ Failed to load {config_path}: {str(e)}")
        return False

def main():
    """Test all YAML configuration files."""
    print("🦙 Testing YAML Configuration Loading for Prompts System")
    print("=" * 60)
    
    # Test all YAML files in config_examples
    config_dir = Path(__file__).parent / "config_examples"
    yaml_files = list(config_dir.glob("**/*.yaml"))
    
    if not yaml_files:
        print("❌ No YAML files found in config_examples directory")
        return False
    
    print(f"Found {len(yaml_files)} YAML configuration files")
    
    success_count = 0
    total_count = len(yaml_files)
    
    for yaml_file in sorted(yaml_files):
        relative_path = yaml_file.relative_to(Path(__file__).parent)
        if test_yaml_config(str(relative_path)):
            success_count += 1
    
    print(f"\n📊 Results: {success_count}/{total_count} configurations loaded successfully")
    
    if success_count == total_count:
        print("🎉 All YAML configurations are valid!")
        return True
    else:
        print(f"⚠️  {total_count - success_count} configurations failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)