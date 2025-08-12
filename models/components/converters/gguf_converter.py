"""
GGUF format converter for models.
Converts between various formats and GGUF.
"""

import os
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional

from .base import ModelConverter
from .llama_cpp_installer import get_llama_cpp_installer


class GGUFConverter(ModelConverter):
    """Convert models to/from GGUF format."""
    
    # Quantization methods and their descriptions
    QUANTIZATION_METHODS = {
        'q4_0': 'Fastest, lowest quality (4-bit)',
        'q4_1': 'Fast, low quality (4-bit)',
        'q5_0': 'Medium speed and quality (5-bit)',
        'q5_1': 'Medium speed and quality (5-bit)',
        'q8_0': 'Slow, high quality (8-bit)',
        'f16': 'Very slow, best quality (16-bit)',
        'f32': 'Extremely slow, highest quality (32-bit)'
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize GGUF converter.
        
        Args:
            config: Optional configuration with:
                - quantization: Default quantization method
                - llama_cpp_path: Path to llama.cpp installation
        """
        super().__init__(config)
        self.quantization = config.get('quantization', 'q4_0') if config else 'q4_0'
        self.installer = get_llama_cpp_installer()
        self.llama_cpp_path = self.installer.get_install_dir()
        
    def convert(self, input_path: Path, output_path: Path,
                target_format: str = 'gguf', **kwargs) -> bool:
        """Convert model to/from GGUF format.
        
        Args:
            input_path: Path to input model
            output_path: Path for output model
            target_format: Target format ('gguf' or source format if converting from GGUF)
            **kwargs: Additional options:
                - quantization: Quantization method for GGUF
                - vocab_type: Vocabulary type (spm, bpe)
                
        Returns:
            True if successful, False otherwise
        """
        if target_format == 'gguf':
            return self._convert_to_gguf(input_path, output_path, **kwargs)
        else:
            return self._convert_from_gguf(input_path, output_path, target_format, **kwargs)
            
    def _convert_to_gguf(self, input_path: Path, output_path: Path, **kwargs) -> bool:
        """Convert model to GGUF format.
        
        Args:
            input_path: Path to input model (PyTorch/HuggingFace)
            output_path: Path for output GGUF file
            **kwargs: Conversion options
            
        Returns:
            True if successful
        """
        quantization = kwargs.get('quantization', self.quantization)
        
        # Ensure llama.cpp is installed
        if not self.installer.is_installed():
            print("📦 llama.cpp not found. Installing...")
            if not self.installer.install():
                print("❌ Failed to install llama.cpp")
                return False
            
        try:
            # Check if this is a LoRA model
            if (input_path / "adapter_config.json").exists():
                print("Detected LoRA adapter model")
                
            # Use the installer to convert
            temp_gguf = output_path.parent / f"{output_path.stem}_f16.gguf"
            
            print(f"Converting to GGUF format...")
            if not self.installer.convert_to_gguf(input_path, temp_gguf):
                print("❌ Conversion to GGUF failed")
                return False
            
            # Then quantize if needed
            if quantization != 'f16':
                print(f"Quantizing to {quantization}...")
                if not self.installer.quantize_gguf(temp_gguf, output_path, quantization):
                    print("❌ Quantization failed")
                    temp_gguf.unlink(missing_ok=True)
                    return False
                
                # Remove temp file
                temp_gguf.unlink(missing_ok=True)
            else:
                # Just rename
                temp_gguf.rename(output_path)
                
            print(f"✅ Converted to GGUF: {output_path}")
            print(f"   Quantization: {quantization}")
            print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Conversion failed: {e}")
            print(f"Error output: {e.stderr}")
            return False
            
    def _convert_from_gguf(self, input_path: Path, output_path: Path,
                          target_format: str, **kwargs) -> bool:
        """Convert GGUF model to another format.
        
        Args:
            input_path: Path to GGUF file
            output_path: Path for output
            target_format: Target format (pytorch, safetensors)
            **kwargs: Conversion options
            
        Returns:
            True if successful
        """
        print(f"⚠️ Converting from GGUF to {target_format} is limited")
        print("Consider using the original model files instead")
        
        # This is more complex and may require additional tools
        # For now, we'll return False
        return False
        
    def _find_llama_cpp(self) -> Optional[Path]:
        """Find llama.cpp installation.
        
        Returns:
            Path to llama.cpp or None
        """
        possible_paths = [
            Path.home() / "llama.cpp",
            Path("/usr/local/llama.cpp"),
            Path("./llama.cpp"),
            Path("/opt/llama.cpp"),
        ]
        
        for path in possible_paths:
            if path.exists() and (path / "convert.py").exists():
                return path
                
        return None
        
    def _build_llama_cpp(self) -> bool:
        """Build llama.cpp if needed.
        
        Returns:
            True if successful
        """
        if not self.llama_cpp_path:
            return False
            
        try:
            print("Building llama.cpp...")
            os.chdir(self.llama_cpp_path)
            
            # Detect system and build accordingly
            import platform
            system = platform.system()
            
            if system == "Darwin":  # macOS
                # Build with Metal support
                subprocess.run(['make', 'LLAMA_METAL=1'], check=True)
            elif system == "Linux":
                # Build with CUDA if available
                subprocess.run(['make', 'LLAMA_CUBLAS=1'], check=False)
            else:
                # Basic build
                subprocess.run(['make'], check=True)
                
            print("✅ llama.cpp built successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Build failed: {e}")
            return False
            
    def validate_input(self, input_path: Path) -> bool:
        """Validate input model format.
        
        Args:
            input_path: Path to input model
            
        Returns:
            True if valid
        """
        if not input_path.exists():
            return False
            
        # Check for GGUF file
        if input_path.suffix == '.gguf':
            return True
            
        # Check for PyTorch/HuggingFace model
        if input_path.is_dir():
            config_file = input_path / "config.json"
            if config_file.exists():
                # Check for model files
                model_files = list(input_path.glob("*.bin")) + \
                             list(input_path.glob("*.safetensors"))
                return len(model_files) > 0
                
        return False
        
    def get_quantization_info(self) -> Dict[str, str]:
        """Get information about available quantization methods.
        
        Returns:
            Dictionary of method -> description
        """
        return self.QUANTIZATION_METHODS
        
    def estimate_size(self, input_path: Path, quantization: str) -> float:
        """Estimate output size for given quantization.
        
        Args:
            input_path: Path to input model
            quantization: Quantization method
            
        Returns:
            Estimated size in MB
        """
        # Get input size
        if input_path.is_dir():
            total_size = sum(f.stat().st_size for f in input_path.rglob("*") if f.is_file())
        else:
            total_size = input_path.stat().st_size
            
        # Estimate based on quantization
        compression_ratios = {
            'q4_0': 0.25,
            'q4_1': 0.27,
            'q5_0': 0.31,
            'q5_1': 0.33,
            'q8_0': 0.50,
            'f16': 0.50,
            'f32': 1.0
        }
        
        ratio = compression_ratios.get(quantization, 0.3)
        return (total_size * ratio) / (1024 * 1024)  # Return in MB
        
    def get_supported_formats(self) -> list:
        """Get list of supported formats.
        
        Returns:
            List of format names
        """
        return ['gguf', 'pytorch', 'huggingface']