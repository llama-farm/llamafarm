"""Auto-generated parser registry."""

from typing import Dict, Type, Any
from pathlib import Path
import importlib
import logging

from .base import BaseParser

logger = logging.getLogger(__name__)


class ParserRegistry:
    """Registry for auto-discovering and loading parsers."""
    
    def __init__(self):
        self.parsers: Dict[str, Type[BaseParser]] = {}
        self._discover_parsers()
    
    def _discover_parsers(self):
        """Discover all available parsers."""
        parsers_dir = Path(__file__).parent
        
        for parser_dir in parsers_dir.iterdir():
            if parser_dir.is_dir() and not parser_dir.name.startswith(('__', '.', 'base')):
                try:
                    # Try to import the parser module
                    module_name = f"components.parsers.{parser_dir.name}.parser"
                    module = importlib.import_module(module_name)
                    
                    # Find the parser class
                    for name in dir(module):
                        obj = getattr(module, name)
                        if (isinstance(obj, type) and 
                            issubclass(obj, BaseParser) and 
                            obj != BaseParser):
                            parser_name = parser_dir.name
                            self.parsers[parser_name] = obj
                            logger.debug(f"Registered parser: {parser_name}")
                            break
                            
                except Exception as e:
                    logger.warning(f"Failed to load parser from {parser_dir.name}: {e}")
    
    def get_parser(self, name: str, config: Dict[str, Any] = None) -> BaseParser:
        """Get a parser instance by name.
        
        Args:
            name: Parser name
            config: Parser configuration
            
        Returns:
            Parser instance
        """
        if name not in self.parsers:
            raise ValueError(f"Parser '{name}' not found. Available: {list(self.parsers.keys())}")
        
        return self.parsers[name](config)
    
    def list_parsers(self) -> List[str]:
        """List all available parsers.
        
        Returns:
            List of parser names
        """
        return list(self.parsers.keys())


# Global registry instance
registry = ParserRegistry()
