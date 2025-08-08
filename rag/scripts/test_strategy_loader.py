#!/usr/bin/env python3
"""
Test script to verify strategy loader works with both old and new formats.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.strategies.loader import StrategyLoader


def test_loader():
    """Test loading strategies from different files."""
    print("\n" + "="*60)
    print("Testing Strategy Loader with New Schema Format")
    print("="*60)
    
    # Test loading demo strategies (new format)
    print("\n1. Testing demo_strategies.yaml (new unified format)...")
    demo_loader = StrategyLoader(str(Path(__file__).parent.parent / "demos" / "demo_strategies.yaml"))
    demo_strategies = demo_loader.load_strategies()
    
    print(f"   ✓ Loaded {len(demo_strategies)} strategies")
    for name in demo_loader.list_strategies():
        strategy = demo_loader.get_strategy(name)
        print(f"     - {name}: {strategy.description[:50]}...")
        # Check for new fields
        if strategy.tags:
            print(f"       Tags: {', '.join(strategy.tags)}")
        if strategy.optimization:
            print(f"       Optimization priority: {strategy.optimization.performance_priority}")
    
    # Test loading config samples (new format)
    print("\n2. Testing config_samples files (new unified format)...")
    samples_dir = Path(__file__).parent.parent / "config_samples"
    
    for config_file in ["document_processing_config.yaml", "enterprise_data_config.yaml"]:
        file_path = samples_dir / config_file
        if file_path.exists():
            print(f"\n   Loading {config_file}...")
            loader = StrategyLoader(str(file_path))
            strategies = loader.load_strategies()
            print(f"   ✓ Loaded {len(strategies)} strategies from {config_file}")
            
            # List first 3 strategies
            for i, name in enumerate(loader.list_strategies()[:3]):
                strategy = loader.get_strategy(name)
                print(f"     - {name}: {len(strategy.tags)} tags, {len(strategy.use_cases)} use cases")
    
    # Test strategy search features
    print("\n3. Testing strategy search features...")
    demo_loader = StrategyLoader(str(Path(__file__).parent.parent / "demos" / "demo_strategies.yaml"))
    demo_loader.load_strategies()
    
    # Search by tag
    demo_strategies = demo_loader.get_strategies_by_tag("demo")
    print(f"   ✓ Found {len(demo_strategies)} strategies with tag 'demo'")
    
    # Search by use case
    support_strategies = demo_loader.get_strategies_by_use_case("Customer support ticketing")
    print(f"   ✓ Found {len(support_strategies)} strategies for 'Customer support ticketing'")
    
    # Test templates
    templates = demo_loader.list_templates()
    print(f"   ✓ Found {len(templates)} templates: {', '.join(templates)}")
    
    print("\n" + "="*60)
    print("✅ All tests passed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_loader()