#!/usr/bin/env python3
"""
Llama.cpp installer and manager for model conversion.
Handles installation across different platforms (Mac, Linux, Windows).
"""

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path
import tempfile
import urllib.request
import zipfile
import tarfile

class LlamaCppInstaller:
    """Manages llama.cpp installation and conversion tools."""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.machine = platform.machine().lower()
        self.base_dir = Path.home() / ".llamafarm" / "llama.cpp"
        self.repo_url = "https://github.com/ggerganov/llama.cpp.git"
        
    def get_install_dir(self) -> Path:
        """Get the installation directory for llama.cpp."""
        return self.base_dir
    
    def is_installed(self) -> bool:
        """Check if llama.cpp is installed."""
        convert_script = self.get_convert_script()
        quantize_binary = self.get_quantize_binary()
        return convert_script and convert_script.exists() and quantize_binary and quantize_binary.exists()
    
    def get_convert_script(self) -> Path:
        """Get path to the convert.py script."""
        possible_paths = [
            self.base_dir / "convert_hf_to_gguf.py",  # Current name in repo
            self.base_dir / "convert-hf-to-gguf.py",
            self.base_dir / "convert.py",  # Old name
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        return self.base_dir / "convert_hf_to_gguf.py"  # Default to current name
    
    def get_quantize_binary(self) -> Path:
        """Get path to the quantize binary."""
        # Check multiple possible locations and names
        if self.system == "windows":
            possible_paths = [
                self.base_dir / "build" / "bin" / "llama-quantize.exe",
                self.base_dir / "build" / "bin" / "quantize.exe",
                self.base_dir / "build" / "llama-quantize.exe",
                self.base_dir / "build" / "quantize.exe",
            ]
        else:
            possible_paths = [
                self.base_dir / "build" / "bin" / "llama-quantize",  # Current name
                self.base_dir / "build" / "bin" / "quantize",        # Old name
                self.base_dir / "build" / "llama-quantize",
                self.base_dir / "build" / "quantize",
            ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        # Return the most likely path
        if self.system == "windows":
            return self.base_dir / "build" / "bin" / "llama-quantize.exe"
        else:
            return self.base_dir / "build" / "bin" / "llama-quantize"
    
    def install(self, force: bool = False) -> bool:
        """Install llama.cpp with platform-specific optimizations."""
        if self.is_installed() and not force:
            print("✅ llama.cpp is already installed")
            # Still ensure Python dependencies are installed
            self._install_python_dependencies()
            return True
        
        print(f"Installing llama.cpp for {self.system} ({self.machine})...")
        
        # Create base directory
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Method 1: Try to use pre-built binaries (fastest)
            if self._install_prebuilt():
                print("✅ Installed pre-built llama.cpp")
                return True
        except Exception as e:
            print(f"Pre-built installation failed: {e}, trying source build...")
        
        try:
            # Method 2: Build from source
            if self._install_from_source():
                print("✅ Built llama.cpp from source")
                # Install Python dependencies after successful build
                self._install_python_dependencies()
                return True
        except Exception as e:
            print(f"Source build failed: {e}")
        
        try:
            # Method 3: Minimal Python-only installation
            if self._install_minimal():
                print("✅ Installed minimal Python conversion tools")
                return True
        except Exception as e:
            print(f"Minimal installation failed: {e}")
        
        return False
    
    def _install_prebuilt(self) -> bool:
        """Install pre-built binaries based on platform."""
        # This would download pre-built binaries from releases
        # For now, we'll skip to source build
        return False
    
    def _install_python_dependencies(self) -> bool:
        """Install Python dependencies required for llama.cpp conversion."""
        try:
            import mistral_common
            import torch
            import transformers
            import sentencepiece
            # All dependencies are available
            return True
        except ImportError:
            print("📦 Installing Python dependencies for llama.cpp...")
            try:
                # Use subprocess to install in current environment
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q",
                     "mistral-common", "torch", "transformers", 
                     "sentencepiece", "protobuf", "numpy<2.0.0"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print("✅ Python dependencies installed")
                    return True
                else:
                    print(f"⚠️  Failed to install some dependencies: {result.stderr}")
                    # Continue anyway, some conversions may still work
                    return True
            except Exception as e:
                print(f"⚠️  Could not install Python dependencies: {e}")
                # Continue anyway
                return True
    
    def _install_from_source(self) -> bool:
        """Build llama.cpp from source."""
        # Check for required tools
        if self.system != "windows":
            # Check for make/cmake
            if not shutil.which("make") and not shutil.which("cmake"):
                print("❌ make or cmake not found. Please install build tools.")
                return False
        
        # Clone repository
        if not (self.base_dir / ".git").exists():
            print("Cloning llama.cpp repository...")
            result = subprocess.run(
                ["git", "clone", "--depth", "1", self.repo_url, str(self.base_dir)],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"Git clone failed: {result.stderr}")
                return False
        
        # Build based on platform
        os.chdir(self.base_dir)
        
        if self.system == "darwin":  # macOS
            # Build with CMake and Metal support for Apple Silicon
            print("Building with CMake for macOS...")
            build_dir = self.base_dir / "build"
            build_dir.mkdir(exist_ok=True)
            
            if "arm" in self.machine or "aarch" in self.machine:
                print("Enabling Metal support for Apple Silicon...")
                cmake_args = ["-DLLAMA_METAL=ON"]
            else:
                print("Building for Intel Mac...")
                cmake_args = []
            
            # Configure with CMake
            result = subprocess.run(
                ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"] + cmake_args,
                cwd=build_dir,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"CMake configuration failed: {result.stderr}")
                return False
            
            # Build
            result = subprocess.run(
                ["cmake", "--build", ".", "--config", "Release", "-j"],
                cwd=build_dir,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"Build failed: {result.stderr}")
                return False
            
            return True
                
        elif self.system == "linux":
            # Build with CMake
            print("Building with CMake for Linux...")
            build_dir = self.base_dir / "build"
            build_dir.mkdir(exist_ok=True)
            
            # Check for CUDA
            if shutil.which("nvcc"):
                print("Enabling CUDA support...")
                cmake_args = ["-DLLAMA_CUDA=ON"]
            else:
                print("Building CPU-only version...")
                cmake_args = []
            
            # Configure with CMake
            result = subprocess.run(
                ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"] + cmake_args,
                cwd=build_dir,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"CMake configuration failed: {result.stderr}")
                return False
            
            # Build
            result = subprocess.run(
                ["cmake", "--build", ".", "--config", "Release", "-j"],
                cwd=build_dir,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"Build failed: {result.stderr}")
                return False
            
            return True
                
        elif self.system == "windows":
            # Use CMake on Windows
            print("Building with CMake for Windows...")
            build_dir = self.base_dir / "build"
            build_dir.mkdir(exist_ok=True)
            
            subprocess.run(["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"], 
                         cwd=build_dir, check=True)
            subprocess.run(["cmake", "--build", ".", "--config", "Release"], 
                         cwd=build_dir, check=True)
            return True
        else:
            print(f"Unsupported system: {self.system}")
            return False
    
    def _install_minimal(self) -> bool:
        """Install minimal Python-only conversion tools."""
        print("Installing minimal Python conversion tools...")
        
        # Download just the conversion scripts
        scripts = [
            "convert.py",
            "convert-hf-to-gguf.py",
            "gguf.py",
            "gguf/__init__.py",
            "gguf/constants.py",
            "gguf/gguf_reader.py",
            "gguf/gguf_writer.py",
        ]
        
        base_url = "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/"
        
        for script in scripts:
            script_path = self.base_dir / script
            script_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                url = base_url + script
                print(f"Downloading {script}...")
                urllib.request.urlretrieve(url, script_path)
            except Exception as e:
                print(f"Failed to download {script}: {e}")
                return False
        
        # Create a dummy quantize script for Python-only mode
        quantize_script = self.base_dir / "quantize.py"
        quantize_script.write_text("""#!/usr/bin/env python3
# Minimal quantization script
import sys
print("Note: Binary quantization not available in Python-only mode")
print("The model will be converted but not quantized")
sys.exit(0)
""")
        quantize_script.chmod(0o755)
        
        return True
    
    def convert_to_gguf(self, model_path: Path, output_path: Path, 
                       model_type: str = "llama") -> bool:
        """Convert a PyTorch/HF model to GGUF format."""
        if not self.is_installed():
            print("llama.cpp not installed. Installing...")
            if not self.install():
                print("❌ Failed to install llama.cpp")
                return False
        
        convert_script = self.get_convert_script()
        
        # For LoRA models, we need to merge them first
        if (model_path / "adapter_config.json").exists():
            print("Detected LoRA adapter. Merging with base model...")
            if not self._merge_lora_model(model_path, output_path.parent / "merged"):
                return False
            model_path = output_path.parent / "merged"
        
        # Run conversion
        print(f"Converting {model_path} to GGUF format...")
        
        cmd = [
            sys.executable, str(convert_script),
            str(model_path),
            "--outfile", str(output_path),
            "--outtype", "f16"  # Start with fp16
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Conversion failed: {result.stderr}")
            return False
        
        print(f"✅ Converted to GGUF: {output_path}")
        return True
    
    def _merge_lora_model(self, lora_path: Path, output_path: Path) -> bool:
        """Merge LoRA adapter with base model."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            import json
            
            # Load adapter config to get base model
            with open(lora_path / "adapter_config.json") as f:
                config = json.load(f)
                base_model_name = config.get("base_model_name_or_path")
            
            if not base_model_name:
                print("❌ Cannot determine base model from adapter config")
                return False
            
            print(f"Loading base model: {base_model_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="cpu"
            )
            
            print("Loading LoRA adapter...")
            model = PeftModel.from_pretrained(base_model, str(lora_path))
            
            print("Merging adapter with base model...")
            model = model.merge_and_unload()
            
            # Save merged model
            output_path.mkdir(parents=True, exist_ok=True)
            print(f"Saving merged model to {output_path}")
            model.save_pretrained(output_path)
            
            # Also save tokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(lora_path))
            tokenizer.save_pretrained(output_path)
            
            print("✅ Successfully merged LoRA with base model")
            return True
            
        except Exception as e:
            print(f"❌ Failed to merge LoRA model: {e}")
            return False
    
    def quantize_gguf(self, input_path: Path, output_path: Path, 
                     quantization: str = "q4_0") -> bool:
        """Quantize a GGUF model."""
        quantize_bin = self.get_quantize_binary()
        
        if not quantize_bin or not quantize_bin.exists():
            print("⚠️ Quantize binary not available, skipping quantization")
            # Just copy the file
            shutil.copy2(input_path, output_path)
            return True
        
        print(f"Quantizing to {quantization}...")
        
        cmd = [str(quantize_bin), str(input_path), str(output_path), quantization]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Quantization failed: {result.stderr}")
            return False
        
        print(f"✅ Quantized to {quantization}: {output_path}")
        return True


# Singleton instance
_installer = None

def get_llama_cpp_installer() -> LlamaCppInstaller:
    """Get the singleton llama.cpp installer."""
    global _installer
    if _installer is None:
        _installer = LlamaCppInstaller()
    return _installer


if __name__ == "__main__":
    # Test installation
    installer = get_llama_cpp_installer()
    
    print(f"Platform: {installer.system} ({installer.machine})")
    print(f"Install directory: {installer.base_dir}")
    
    if installer.is_installed():
        print("✅ llama.cpp is installed")
        print(f"Convert script: {installer.get_convert_script()}")
        print(f"Quantize binary: {installer.get_quantize_binary()}")
    else:
        print("❌ llama.cpp not installed")
        if input("Install now? (y/n): ").lower() == 'y':
            if installer.install():
                print("✅ Installation successful")
            else:
                print("❌ Installation failed")