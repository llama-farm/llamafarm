"""
Ollama format converter for models.
Converts PyTorch/HuggingFace models to Ollama format.
"""

import os
import json
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import tempfile

from .base import ModelConverter


class OllamaConverter(ModelConverter):
    """Convert models to Ollama format."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Ollama converter.
        
        Args:
            config: Optional configuration with:
                - quantization: Quantization method (q4_0, q4_1, q5_0, q5_1, q8_0)
                - system_prompt: Default system prompt
                - model_name: Name for the Ollama model
        """
        super().__init__(config)
        self.quantization = config.get('quantization', 'q4_0') if config else 'q4_0'
        self.system_prompt = config.get('system_prompt', 'You are a helpful assistant.') if config else 'You are a helpful assistant.'
        
    def convert(self, input_path: Path, output_path: Path, 
                target_format: str = 'ollama', **kwargs) -> bool:
        """Convert model to Ollama format.
        
        Args:
            input_path: Path to input model (PyTorch or GGUF)
            output_path: Path for output (directory for Ollama model)
            target_format: Should be 'ollama'
            **kwargs: Additional options:
                - model_name: Name for the model in Ollama
                - quantization: Override quantization method
                - push_to_registry: Push to Ollama registry
                
        Returns:
            True if successful, False otherwise
        """
        if target_format != 'ollama':
            raise ValueError(f"This converter only supports 'ollama' format, got {target_format}")
            
        model_name = kwargs.get('model_name', output_path.stem)
        quantization = kwargs.get('quantization', self.quantization)
        
        # Remove model_name and quantization from kwargs to avoid duplicate arguments
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['model_name', 'quantization']}
        
        # Check if input is already GGUF
        if input_path.suffix == '.gguf':
            return self._create_from_gguf(input_path, output_path, model_name, **filtered_kwargs)
        else:
            # Need to convert to GGUF first
            return self._convert_pytorch_to_ollama(input_path, output_path, model_name, quantization, **filtered_kwargs)
            
    def _create_from_gguf(self, gguf_path: Path, output_path: Path, 
                         model_name: str, **kwargs) -> bool:
        """Create Ollama model from GGUF file.
        
        Args:
            gguf_path: Path to GGUF file
            output_path: Output directory
            model_name: Model name for Ollama
            
        Returns:
            True if successful
        """
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create Modelfile
        modelfile_content = self._create_modelfile(gguf_path, **kwargs)
        modelfile_path = output_path / "Modelfile"
        
        with open(modelfile_path, 'w') as f:
            f.write(modelfile_content)
            
        # Create model with Ollama
        try:
            result = subprocess.run(
                ['ollama', 'create', model_name, '-f', str(modelfile_path)],
                capture_output=True,
                text=True,
                check=True
            )
            
            print(f"✅ Model created: {model_name}")
            print(f"Run with: ollama run {model_name}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create Ollama model: {e}")
            print(f"Error output: {e.stderr}")
            return False
        except FileNotFoundError:
            print("❌ Ollama not found. Please install Ollama first.")
            return False
            
    def _convert_pytorch_to_ollama(self, pytorch_path: Path, output_path: Path,
                                   model_name: str, quantization: str, **kwargs) -> bool:
        """Convert PyTorch model to Ollama format.
        
        Args:
            pytorch_path: Path to PyTorch model directory
            output_path: Output directory
            model_name: Model name for Ollama
            quantization: Quantization method
            
        Returns:
            True if successful
        """
        # Import the GGUF converter
        from .gguf_converter import GGUFConverter
        
        # First convert to GGUF using our enhanced GGUF converter
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            gguf_path = temp_path / f"{model_name}.gguf"
            
            # Use GGUF converter which handles LoRA models properly
            gguf_converter = GGUFConverter({'quantization': quantization})
            
            print("Converting PyTorch model to GGUF format...")
            if not gguf_converter.convert(pytorch_path, gguf_path, target_format='gguf'):
                print("❌ Failed to convert to GGUF")
                return False
                
            # Create Ollama model from GGUF
            return self._create_from_gguf(gguf_path, output_path, model_name, **kwargs)
            
    def _convert_to_gguf(self, pytorch_path: Path, gguf_path: Path, 
                        quantization: str) -> bool:
        """Convert PyTorch model to GGUF format.
        
        Args:
            pytorch_path: Path to PyTorch model
            gguf_path: Output GGUF path
            quantization: Quantization method
            
        Returns:
            True if successful
        """
        try:
            # Try to use llama.cpp's convert script
            convert_script = self._find_llama_cpp_convert()
            
            if not convert_script:
                print("❌ llama.cpp convert script not found")
                return False
                
            # Run conversion
            result = subprocess.run(
                ['python', str(convert_script), str(pytorch_path), 
                 '--outfile', str(gguf_path), '--outtype', quantization],
                capture_output=True,
                text=True,
                check=True
            )
            
            print(f"✅ Converted to GGUF: {gguf_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ GGUF conversion failed: {e}")
            print(f"Error output: {e.stderr}")
            return False
            
    def _create_modelfile(self, model_path: Path, **kwargs) -> str:
        """Create Modelfile content for Ollama.
        
        Args:
            model_path: Path to model file
            **kwargs: Additional parameters
            
        Returns:
            Modelfile content as string
        """
        temperature = kwargs.get('temperature', 0.7)
        top_p = kwargs.get('top_p', 0.9)
        top_k = kwargs.get('top_k', 40)
        system_prompt = kwargs.get('system_prompt', self.system_prompt)
        
        modelfile = f"""# Modelfile for converted model
FROM {model_path}

# Model parameters
PARAMETER temperature {temperature}
PARAMETER top_p {top_p}
PARAMETER top_k {top_k}
PARAMETER num_predict 2048
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|end|>"
PARAMETER stop "</s>"

# System prompt
SYSTEM "{system_prompt}"

# Template (for chat models)
TEMPLATE \"\"\"{{{{ if .System }}}}System: {{{{ .System }}}}
{{{{ end }}}}{{{{ if .Prompt }}}}User: {{{{ .Prompt }}}}
{{{{ end }}}}Assistant: \"\"\"
"""
        
        # Add custom parameters if provided
        if 'parameters' in kwargs:
            for key, value in kwargs['parameters'].items():
                modelfile += f"PARAMETER {key} {value}\n"
                
        return modelfile
        
    def _check_llama_cpp(self) -> bool:
        """Check if llama.cpp is available.
        
        Returns:
            True if available
        """
        # Check common locations
        possible_paths = [
            Path.home() / "llama.cpp",
            Path("/usr/local/llama.cpp"),
            Path("./llama.cpp"),
        ]
        
        for path in possible_paths:
            if path.exists():
                return True
                
        # Check if in PATH
        result = subprocess.run(['which', 'llama-cli'], capture_output=True)
        return result.returncode == 0
        
    def _install_llama_cpp(self) -> bool:
        """Install llama.cpp if not available.
        
        Returns:
            True if successful
        """
        try:
            print("Installing llama.cpp...")
            
            # Clone repository
            subprocess.run([
                'git', 'clone', 
                'https://github.com/ggerganov/llama.cpp',
                str(Path.home() / 'llama.cpp')
            ], check=True)
            
            # Build
            os.chdir(Path.home() / 'llama.cpp')
            subprocess.run(['make'], check=True)
            
            print("✅ llama.cpp installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install llama.cpp: {e}")
            return False
            
    def _find_llama_cpp_convert(self) -> Optional[Path]:
        """Find llama.cpp convert.py script.
        
        Returns:
            Path to convert.py or None
        """
        possible_paths = [
            Path.home() / "llama.cpp" / "convert.py",
            Path("/usr/local/llama.cpp") / "convert.py",
            Path("./llama.cpp") / "convert.py",
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
                
        return None
        
    def validate_input(self, input_path: Path) -> bool:
        """Validate input model format.
        
        Args:
            input_path: Path to input model
            
        Returns:
            True if valid
        """
        if not input_path.exists():
            return False
            
        # Check for supported formats
        if input_path.is_dir():
            # Check for PyTorch/HuggingFace model files
            required_files = ['config.json', 'pytorch_model.bin']
            alt_files = ['model.safetensors', 'model-00001-of-*.safetensors']
            
            has_required = all((input_path / f).exists() for f in required_files[:1])
            has_model = any((input_path / f).exists() for f in required_files[1:]) or \
                       any(list(input_path.glob(f)) for f in alt_files)
                       
            return has_required and has_model
            
        elif input_path.suffix == '.gguf':
            return True
            
        return False
        
    def get_supported_formats(self) -> list:
        """Get list of supported target formats.
        
        Returns:
            List of format names
        """
        return ['ollama']