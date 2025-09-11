#!/usr/bin/env python3
"""Script to compile all parser configurations into a registry."""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Any

def compile_parser_registry():
    """Compile all parser configurations into a single registry."""
    
    parsers_dir = Path(__file__).parent
    registry = {
        "version": "1.0",
        "parsers": {},
        "file_extensions": {},
        "mime_types": {},
        "tools": {}
    }
    
    # Scan all subdirectories for config.yaml files
    for subdir in parsers_dir.iterdir():
        if not subdir.is_dir() or subdir.name.startswith('_'):
            continue
        
        config_file = subdir / "config.yaml"
        if not config_file.exists():
            continue
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            if not config or 'parsers' not in config:
                continue
            
            parser_type = subdir.name
            
            for parser in config['parsers']:
                parser_name = parser['name']
                
                # Add to main registry
                registry['parsers'][parser_name] = {
                    **parser,
                    'parser_type': parser_type,
                    'implementation_dir': str(subdir)
                }
                
                # Map file extensions
                for ext in parser.get('supported_extensions', []):
                    if ext not in registry['file_extensions']:
                        registry['file_extensions'][ext] = []
                    registry['file_extensions'][ext].append({
                        'parser': parser_name,
                        'tool': parser['tool'],
                        'priority': 0  # Can be adjusted based on preferences
                    })
                
                # Map MIME types
                for mime in parser.get('mime_types', []):
                    if mime not in registry['mime_types']:
                        registry['mime_types'][mime] = []
                    registry['mime_types'][mime].append({
                        'parser': parser_name,
                        'tool': parser['tool']
                    })
                
                # Map tools
                tool = parser['tool']
                if tool not in registry['tools']:
                    registry['tools'][tool] = []
                registry['tools'][tool].append({
                    'parser': parser_name,
                    'type': parser_type,
                    'capabilities': parser.get('capabilities', [])
                })
                
            print(f"✓ Compiled {len(config['parsers'])} parsers from {parser_type}")
            
        except Exception as e:
            print(f"✗ Failed to compile {config_file}: {e}")
    
    # Save registry as JSON
    registry_file = parsers_dir / "parser_registry.json"
    with open(registry_file, 'w') as f:
        json.dump(registry, f, indent=2)
    print(f"\n✓ Registry saved to {registry_file}")
    
    # Save registry as Python module
    python_file = parsers_dir / "parser_registry.py"
    with open(python_file, 'w') as f:
        f.write('"""Auto-generated parser registry."""\n\n')
        f.write('import json\n\n')
        # Convert to Python representation instead of JSON string
        import pprint
        f.write(f'REGISTRY_DATA = ')
        pp = pprint.PrettyPrinter(indent=2, width=120)
        registry_str = pp.pformat(registry)
        f.write(registry_str)
        f.write('\n\n')
        f.write('class ParserRegistry:\n')
        f.write('    """Parser registry with lookup methods."""\n\n')
        f.write('    def __init__(self):\n')
        f.write('        self.data = REGISTRY_DATA\n\n')
        f.write('    def get_parser(self, name: str) -> dict:\n')
        f.write('        """Get parser configuration by name."""\n')
        f.write('        return self.data["parsers"].get(name)\n\n')
        f.write('    def get_parsers_for_extension(self, ext: str) -> list:\n')
        f.write('        """Get parsers that support a file extension."""\n')
        f.write('        return self.data["file_extensions"].get(ext, [])\n\n')
        f.write('    def get_parsers_for_mime(self, mime: str) -> list:\n')
        f.write('        """Get parsers that support a MIME type."""\n')
        f.write('        return self.data["mime_types"].get(mime, [])\n\n')
        f.write('    def get_parsers_by_tool(self, tool: str) -> list:\n')
        f.write('        """Get all parsers using a specific tool."""\n')
        f.write('        return self.data["tools"].get(tool, [])\n\n')
        f.write('    def list_all_parsers(self) -> list:\n')
        f.write('        """List all available parser names."""\n')
        f.write('        return list(self.data["parsers"].keys())\n\n')
        f.write('registry = ParserRegistry()\n')
    
    print(f"✓ Python module saved to {python_file}")
    
    # Print summary
    print("\n=== Registry Summary ===")
    print(f"Total parsers: {len(registry['parsers'])}")
    print(f"File types supported: {len(registry['file_extensions'])}")
    print(f"MIME types supported: {len(registry['mime_types'])}")
    print(f"Tools integrated: {len(registry['tools'])}")
    
    # List tools and their parsers
    print("\n=== Tools and Parsers ===")
    for tool, parsers in registry['tools'].items():
        parser_names = [p['parser'] for p in parsers]
        print(f"{tool}: {', '.join(parser_names)}")
    
    return registry

if __name__ == "__main__":
    compile_parser_registry()