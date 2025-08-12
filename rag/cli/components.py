#!/usr/bin/env python3
"""
CLI command to list all available components in the RAG system.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

def get_component_info(component_dir: Path) -> Tuple[str, List[str]]:
    """Get component name and check if it has required files."""
    required_files = []
    
    # Check for key files
    if (component_dir / "defaults.yaml").exists():
        required_files.append("config")
    if (component_dir / "schema.yaml").exists():
        required_files.append("schema")
    if (component_dir / f"{component_dir.name}.py").exists():
        required_files.append("implementation")
    if (component_dir / f"{component_dir.name}.md").exists():
        required_files.append("docs")
    
    return component_dir.name, required_files

def list_components():
    """List all available components by type."""
    base_dir = Path(__file__).parent / "components"
    
    component_types = {
        "parsers": "📄 Document Parsers",
        "extractors": "🔍 Metadata Extractors", 
        "embedders": "🧠 Embedding Models",
        "stores": "💾 Vector Stores",
        "retrievers": "🎯 Retrieval Strategies"
    }
    
    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'RAG System Components':^60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
    
    for comp_type, display_name in component_types.items():
        comp_dir = base_dir / comp_type
        if not comp_dir.exists():
            continue
            
        # Get all subdirectories that are components
        components = []
        for item in comp_dir.iterdir():
            if item.is_dir() and not item.name.startswith('__'):
                name, features = get_component_info(item)
                components.append((name, features))
        
        if components:
            print(f"{Fore.YELLOW}{display_name}{Style.RESET_ALL}")
            print(f"{Fore.BLUE}{'─' * 40}{Style.RESET_ALL}")
            
            for name, features in sorted(components):
                # Format name for display
                display_name = name.replace('_', ' ').title()
                
                # Check completeness
                if len(features) >= 3:
                    status = f"{Fore.GREEN}✅{Style.RESET_ALL}"
                elif len(features) >= 2:
                    status = f"{Fore.YELLOW}⚠️{Style.RESET_ALL}"
                else:
                    status = f"{Fore.RED}❌{Style.RESET_ALL}"
                
                # Print component
                print(f"  {status} {display_name:<30} [{', '.join(features)}]")
            
            print()
    
    # Print usage hints
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"\n{Fore.GREEN}💡 Usage Tips:{Style.RESET_ALL}")
    print("  • ✅ = Fully implemented component")
    print("  • ⚠️  = Partially implemented component")
    print("  • ❌ = Missing implementation files")
    print()
    print(f"{Fore.GREEN}To use a component in your strategy:{Style.RESET_ALL}")
    print("  1. Reference it in your strategy YAML file")
    print("  2. The system will auto-load it when needed")
    print()
    print(f"{Fore.GREEN}Example strategy configuration:{Style.RESET_ALL}")
    print("""  components:
    parser:
      type: "PDFParser"
    extractors:
      - type: "EntityExtractor"
      - type: "SummaryExtractor"
    embedder:
      type: "OllamaEmbedder"
    vector_store:
      type: "ChromaStore"
    """)

if __name__ == "__main__":
    list_components()