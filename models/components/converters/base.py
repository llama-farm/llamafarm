"""
Base converter class for model format conversions.
"""

from pathlib import Path
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class ModelConverter(ABC):
    """Base class for model converters."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize converter.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
    @abstractmethod
    def convert(self, input_path: Path, output_path: Path, 
                target_format: str, **kwargs) -> bool:
        """Convert model to target format.
        
        Args:
            input_path: Path to input model
            output_path: Path for output model
            target_format: Target format name
            **kwargs: Additional conversion options
            
        Returns:
            True if successful, False otherwise
        """
        pass
        
    @abstractmethod
    def validate_input(self, input_path: Path) -> bool:
        """Validate input model format.
        
        Args:
            input_path: Path to input model
            
        Returns:
            True if valid, False otherwise
        """
        pass
        
    def get_supported_formats(self) -> list:
        """Get list of supported target formats.
        
        Returns:
            List of format names
        """
        return []