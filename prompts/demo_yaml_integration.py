#!/usr/bin/env python3
"""Demonstration of YAML configuration integration with prompts system."""

import sys
from pathlib import Path

# Add prompts to Python path
sys.path.insert(0, str(Path(__file__).parent))

from prompts.utils.config_loader import load_and_validate_config, merge_configs

def demo_yaml_loading():
    """Demonstrate YAML configuration loading."""
    print("🦙 LlamaFarm Prompts System - YAML Integration Demo")
    print("=" * 55)
    
    # Load a simple configuration
    print("\n📋 Loading Simple Q&A Configuration...")
    simple_config = load_and_validate_config("config_examples/basic/simple_qa_config.yaml")
    print(f"✅ Loaded: {simple_config['name']}")
    print(f"   Strategy: {simple_config['prompts']['strategy']}")
    
    # Load a complex configuration
    print("\n🏥 Loading Medical Domain Configuration...")
    medical_config = load_and_validate_config("config_examples/domain_specific/medical_config.yaml")
    print(f"✅ Loaded: {medical_config['name']}")
    print(f"   Domain: {medical_config['prompts']['domain']}")
    print(f"   Compliance: {medical_config['compliance']}")
    
    # Load integration configuration
    print("\n🔗 Loading Full Pipeline Integration...")
    integration_config = load_and_validate_config("config_examples/integration/full_pipeline_config.yaml")
    print(f"✅ Loaded: {integration_config['name']}")
    print(f"   RAG Strategy: {integration_config['retrieval']['strategy']}")
    print(f"   Prompt Strategy: {integration_config['prompts']['strategy']}")
    
    # Demonstrate template loading
    print("\n📝 Template Preview:")
    qa_template = simple_config['prompts']['templates']['qa_basic']
    print(f"   Template Type: {qa_template['type']}")
    print(f"   Input Variables: {qa_template['input_variables']}")
    print("   Template Preview:")
    template_preview = qa_template['template'][:100].replace('\n', '\\n')
    print(f"   \"{template_preview}...\"")
    
    # Demonstrate config merging
    print("\n🔄 Configuration Merging Demo...")
    base_config = {"prompts": {"strategy": "basic", "config": {"timeout": 30}}}
    override_config = {"prompts": {"config": {"timeout": 60, "retries": 3}}}
    merged = merge_configs(base_config, override_config)
    print(f"   Merged timeout: {merged['prompts']['config']['timeout']}")
    print(f"   Added retries: {merged['prompts']['config']['retries']}")
    
    print("\n🎉 YAML Integration Working Perfectly!")
    print("\n📚 Available Configurations:")
    config_dir = Path("config_examples")
    yaml_files = list(config_dir.glob("**/*.yaml"))
    for yaml_file in sorted(yaml_files):
        print(f"   • {yaml_file}")
    
    print(f"\n✨ Total: {len(yaml_files)} YAML configuration examples ready to use")

if __name__ == "__main__":
    try:
        demo_yaml_loading()
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        sys.exit(1)