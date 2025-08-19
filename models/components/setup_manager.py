#!/usr/bin/env python3
"""
Setup Manager for LlamaFarm.
Analyzes strategy requirements and installs/downloads necessary tools and models.
"""

import os
import sys
import json
import yaml
import subprocess
import platform
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import urllib.request
from enum import Enum

class ToolType(Enum):
    """Types of tools that can be installed."""
    OLLAMA = "ollama"
    LLAMA_CPP = "llama.cpp"
    VLLM = "vllm"
    TGI = "tgi"  # Text Generation Inference
    LLAMAFILE = "llamafile"
    
class ModelFormat(Enum):
    """Model formats."""
    GGUF = "gguf"
    PYTORCH = "pytorch"
    SAFETENSORS = "safetensors"
    ONNX = "onnx"
    
class SetupManager:
    """Manages setup and installation of tools and models."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.system = platform.system().lower()
        self.machine = platform.machine().lower()
        self.base_dir = Path.home() / ".llamafarm"
        self.base_dir.mkdir(exist_ok=True)
        
        # Load component registry and definitions
        self.component_registry = self._load_component_registry()
        self.component_definitions = {}
        
    def _load_component_registry(self) -> Dict[str, Any]:
        """Load the component registry."""
        registry_path = Path(__file__).parent / "definitions" / "registry.yaml"
        if registry_path.exists():
            with open(registry_path) as f:
                return yaml.safe_load(f)
        return {}
        
    def _load_component_definition(self, component_name: str) -> Dict[str, Any]:
        """Load a component definition from its YAML file."""
        if component_name in self.component_definitions:
            return self.component_definitions[component_name]
            
        # Find the component in the registry
        definition_file = None
        for category in self.component_registry.get('categories', {}).values():
            if component_name in category.get('components', {}):
                definition_file = category['components'][component_name]['definition_file']
                break
                
        if not definition_file:
            return {}
            
        definition_path = Path(__file__).parent / "definitions" / definition_file
        if definition_path.exists():
            with open(definition_path) as f:
                definition = yaml.safe_load(f)
                self.component_definitions[component_name] = definition
                return definition
                
        return {}
        
    def analyze_strategy(self, strategy_path: Path) -> Dict[str, Any]:
        """Analyze a strategy file to determine requirements based on component definitions."""
        requirements = {
            "components": [],
            "models": [],
            "dependencies": set(),
            "hardware": {},
            "estimated_size_gb": 0
        }
        
        # Load strategy file
        with open(strategy_path) as f:
            if strategy_path.suffix == '.yaml':
                strategies = yaml.safe_load(f)
            else:
                strategies = json.load(f)
        
        # Track unique components to avoid duplicates
        seen_components = set()
        
        # Analyze each strategy
        strategies_list = strategies.get('strategies', [])
        for strategy in strategies_list:
            strategy_name = strategy.get('name', 'unknown')
            if self.verbose:
                print(f"Analyzing strategy: {strategy_name}")
            
            # Check components
            components = strategy.get('components', {})
            
            # Map component types to actual component names
            component_mapping = {
                'model_app': 'ollama',           # Default model app (can be overridden by type)
                'cloud_api': 'openai',           # Default cloud API
                'fine_tuner': 'pytorch',         # Default fine tuner
                'ollama': 'ollama',              # Direct reference
                'mock_model': 'mock_model',      # Direct mock reference
                'converters': 'gguf_converter'   # From export settings
            }
            
            # Process each component in the strategy
            for component_type, component_config in components.items():
                # Determine actual component name
                if isinstance(component_config, dict) and 'type' in component_config:
                    component_name = component_config['type']
                else:
                    component_name = component_mapping.get(component_type, component_type)
                
                # Skip if we've already seen this component
                if component_name in seen_components:
                    continue
                    
                seen_components.add(component_name)
                
                if self.verbose:
                    print(f"  Found component: {component_name} (type: {component_type})")
                
                # Load component definition
                definition = self._load_component_definition(component_name)
                if definition:
                    requirements['components'].append({
                        'name': component_name,
                        'type': component_type,
                        'config': component_config,
                        'definition': definition
                    })
                    
                    # Add system dependencies from component definition
                    # NOTE: Python dependencies are handled by pyproject.toml
                    deps = definition.get('setup', {}).get('dependencies', {})
                    for dep in deps.get('system', []):
                        if isinstance(dep, dict):
                            requirements['dependencies'].add(dep['name'])
                        else:
                            requirements['dependencies'].add(dep)
            
            # Check for export/conversion requirements
            export_config = strategy.get('export', {})
            if export_config.get('to_ollama') or export_config.get('to_gguf'):
                # Need converters
                if export_config.get('to_gguf') or export_config.get('to_ollama'):
                    gguf_def = self._load_component_definition('gguf_converter')
                    if gguf_def:
                        requirements['components'].append({
                            'name': 'gguf_converter',
                            'type': 'converter',
                            'config': export_config,
                            'definition': gguf_def
                        })
                        
                if export_config.get('to_ollama'):
                    ollama_conv_def = self._load_component_definition('ollama_converter')  
                    if ollama_conv_def:
                        requirements['components'].append({
                            'name': 'ollama_converter', 
                            'type': 'converter',
                            'config': export_config,
                            'definition': ollama_conv_def
                        })
            
            # Extract model requirements from component configs
            for component in requirements['components']:
                config = component['config']
                definition = component['definition']
                
                # Models from component config
                if 'models' in config:
                    for model in config['models']:
                        if isinstance(model, dict):
                            requirements['models'].append({
                                'name': model.get('name'),
                                'component': component['name'],
                                'pull_on_start': model.get('pull_on_start', False)
                            })
                        else:
                            requirements['models'].append({
                                'name': model,
                                'component': component['name']
                            })
                
                # Default model from component config
                if 'default_model' in config:
                    requirements['models'].append({
                        'name': config['default_model'],
                        'component': component['name'],
                        'default': True
                    })
        
        # Estimate total size using component definitions
        requirements['estimated_size_gb'] = self._estimate_size_from_definitions(requirements)
        
        return requirements
        
    def _estimate_size_from_definitions(self, requirements: Dict[str, Any]) -> float:
        """Estimate total disk space needed using component definitions."""
        total_gb = 0
        
        # Use component definitions for size estimation
        for component in requirements['components']:
            definition = component['definition']
            setup = definition.get('setup', {})
            models_config = setup.get('models', {})
            
            # Check for size estimation rules
            size_rules = models_config.get('size_estimation', {}).get('rules', [])
            
            # Estimate model sizes
            for model in requirements['models']:
                if model.get('component') == component['name']:
                    model_name = model['name'].lower()
                    
                    # Apply size rules from component definition
                    for rule in size_rules:
                        import re
                        if re.search(rule['pattern'], model_name):
                            total_gb += rule['size_gb']
                            break
                    else:
                        # Default fallback estimate
                        total_gb += 2.0
        
        # Add space for tools themselves
        for component in requirements['components']:
            if component['name'] in ['gguf_converter']:
                total_gb += 0.5  # Space for llama.cpp build
                
        return total_gb
    
    def check_component_installed(self, component_name: str) -> bool:
        """Check if a specific component is installed using its definition."""
        definition = self._load_component_definition(component_name)
        if not definition:
            return False
            
        setup = definition.get('setup', {})
        detection = setup.get('detection', {})
        
        # Try command-based detection
        if 'command' in detection:
            try:
                result = subprocess.run(
                    detection['command'].split(),
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    success_pattern = detection.get('success_pattern', '')
                    if success_pattern and success_pattern in result.stdout:
                        return True
                    elif not success_pattern:
                        return True
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
                
        # Try path-based detection
        if 'methods' in detection:
            for method in detection['methods']:
                if 'path_exists' in method:
                    path = Path(method['path_exists']).expanduser()
                    if path.exists():
                        return True
                elif 'command' in method:
                    try:
                        result = subprocess.run(
                            method['command'], 
                            shell=True,
                            capture_output=True,
                            text=True,
                            timeout=10
                        )
                        if result.returncode == 0:
                            success_pattern = method.get('success_pattern', '')
                            if not success_pattern or success_pattern in result.stdout:
                                return True
                    except (subprocess.TimeoutExpired, FileNotFoundError):
                        continue
        
        return False
    
    def install_ollama(self) -> bool:
        """Install Ollama based on platform."""
        print("Installing Ollama...")
        
        if self.system == "darwin":  # macOS
            # Download Ollama for Mac
            print("Downloading Ollama for macOS...")
            download_url = "https://ollama.ai/download/Ollama-darwin.zip"
            
            # Alternative: use brew if available
            if shutil.which('brew'):
                print("Installing via Homebrew...")
                result = subprocess.run(['brew', 'install', 'ollama'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print("✅ Ollama installed via Homebrew")
                    return True
            
            print("Please download Ollama from: https://ollama.ai/download")
            print("Or run: brew install ollama")
            return False
            
        elif self.system == "linux":
            # Install script for Linux
            print("Installing Ollama for Linux...")
            install_script = "curl -fsSL https://ollama.ai/install.sh | sh"
            result = subprocess.run(install_script, shell=True, 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Ollama installed successfully")
                return True
            else:
                print(f"❌ Installation failed: {result.stderr}")
                return False
                
        elif self.system == "windows":
            print("Please download Ollama from: https://ollama.ai/download")
            print("Windows installer available at the link above")
            return False
        
        return False
    
    def install_component(self, component_name: str) -> bool:
        """Install a component using its definition."""
        definition = self._load_component_definition(component_name)
        if not definition:
            print(f"❌ No definition found for component: {component_name}")
            return False
            
        setup = definition.get('setup', {})
        installation = setup.get('installation', {})
        
        # Get platform-specific installation config
        platform_config = installation.get(self.system, {})
        if not platform_config:
            print(f"❌ No installation method for platform: {self.system}")
            return False
            
        # Try primary installation method
        primary = platform_config.get('primary', {})
        if primary.get('method') == 'build_from_source':
            return self._build_from_source(primary, component_name)
        elif primary.get('method') == 'script':
            return self._install_via_script(primary)
        elif primary.get('method') == 'homebrew':
            return self._install_via_homebrew(primary)
        elif primary.get('method') == 'download':
            return self._install_via_download(primary)
        elif primary.get('method') == 'builtin':
            return self._install_builtin(primary, component_name)
        else:
            print(f"❌ Unsupported installation method: {primary.get('method')}")
            return False
            
    def _build_from_source(self, config: Dict[str, Any], component_name: str) -> bool:
        """Build component from source using steps defined in config."""
        steps = config.get('steps', [])
        
        for step in steps:
            step_name = step.get('name', 'Build step')
            command = step.get('command', '')
            working_dir = step.get('working_directory', '~')
            env_vars = step.get('environment_vars', {})
            timeout_mins = step.get('timeout_minutes', 30)
            
            print(f"  {step_name}...")
            
            # Expand user path
            working_dir = Path(working_dir).expanduser()
            
            # Prepare environment
            env = os.environ.copy()
            env.update(env_vars)
            
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    cwd=working_dir,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout_mins * 60
                )
                
                if result.returncode != 0:
                    print(f"❌ Step failed: {step_name}")
                    print(f"Error: {result.stderr}")
                    return False
                    
                print(f"✅ {step_name} completed")
                
            except subprocess.TimeoutExpired:
                print(f"❌ Step timed out: {step_name}")
                return False
            except Exception as e:
                print(f"❌ Step error: {step_name} - {e}")
                return False
        
        # Verify installation
        verify_command = config.get('verify')
        if verify_command:
            try:
                result = subprocess.run(
                    verify_command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    print(f"✅ {component_name} installation verified")
                    return True
                else:
                    print(f"⚠️  {component_name} installation completed but verification failed")
                    return True  # Still consider it successful
            except:
                print(f"⚠️  Could not verify {component_name} installation")
                return True  # Still consider it successful
                
        return True
    
    def _install_via_script(self, config: Dict[str, Any]) -> bool:
        """Install via shell script."""
        command = config.get('command', '')
        verify = config.get('verify', '')
        
        print(f"  Running installation script...")
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print("✅ Installation script completed")
                if verify:
                    return self._verify_installation(verify)
                return True
            else:
                print(f"❌ Installation script failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Installation script timed out")
            return False
        except Exception as e:
            print(f"❌ Installation script error: {e}")
            return False
    
    def _install_via_homebrew(self, config: Dict[str, Any]) -> bool:
        """Install via Homebrew."""
        command = config.get('command', '')
        verify = config.get('verify', '')
        
        # Check if brew is available
        if not shutil.which('brew'):
            print("❌ Homebrew not found")
            return False
            
        print(f"  Installing via Homebrew...")
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode == 0:
                print("✅ Homebrew installation completed")
                if verify:
                    return self._verify_installation(verify)
                return True
            else:
                print(f"❌ Homebrew installation failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Homebrew installation error: {e}")
            return False
    
    def _install_via_download(self, config: Dict[str, Any]) -> bool:
        """Install via direct download."""
        url = config.get('url', '')
        instructions = config.get('instructions', '')
        
        if instructions:
            print(f"⚠️  Manual installation required:")
            print(f"   {instructions}")
            if url:
                print(f"   URL: {url}")
            return False
        else:
            print(f"⚠️  Download method not fully implemented yet")
            print(f"   URL: {url}")
            return False
    
    def _install_builtin(self, config: Dict[str, Any], component_name: str) -> bool:
        """Handle built-in components that don't need installation."""
        description = config.get('description', f'{component_name} is built-in')
        print(f"  {description}")
        return True
    
    def _verify_installation(self, verify_command: str) -> bool:
        """Verify installation using command."""
        try:
            result = subprocess.run(
                verify_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0
        except:
            return False
        
    def install_llama_cpp(self) -> bool:
        """Install llama.cpp using component definition."""
        return self.install_component('gguf_converter')
    
    def download_ollama_model(self, model_name: str) -> bool:
        """Download an Ollama model."""
        print(f"Pulling Ollama model: {model_name}")
        
        result = subprocess.run(['ollama', 'pull', model_name],
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Downloaded {model_name}")
            return True
        else:
            print(f"❌ Failed to download {model_name}: {result.stderr}")
            return False
    
    def download_huggingface_model(self, model_id: str, cache_dir: Optional[Path] = None) -> bool:
        """Download a HuggingFace model."""
        try:
            from huggingface_hub import snapshot_download
            
            if not cache_dir:
                cache_dir = self.base_dir / "models" / "huggingface"
            
            print(f"Downloading HuggingFace model: {model_id}")
            snapshot_download(
                repo_id=model_id,
                cache_dir=str(cache_dir),
                ignore_patterns=["*.msgpack", "*.h5", "*.ot"]  # Skip unnecessary files
            )
            
            print(f"✅ Downloaded {model_id}")
            return True
            
        except ImportError:
            print("❌ huggingface_hub not installed")
            print("Install with: pip install huggingface_hub")
            return False
        except Exception as e:
            print(f"❌ Failed to download {model_id}: {e}")
            return False
    
    def convert_model_format(self, model_path: Path, target_format: ModelFormat,
                           output_path: Optional[Path] = None) -> Optional[Path]:
        """Convert a model to a different format."""
        if target_format == ModelFormat.GGUF:
            from components.converters.gguf_converter import GGUFConverter
            
            converter = GGUFConverter()
            if not output_path:
                output_path = model_path.parent / f"{model_path.stem}.gguf"
            
            if converter.convert(model_path, output_path, target_format='gguf'):
                return output_path
        
        return None
    
    def interactive_setup(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Interactive setup with user prompts."""
        decisions = {
            'install_components': [],
            'download_models': [],
            'skip_items': []
        }
        
        print("\n" + "="*60)
        print("SETUP REQUIREMENTS ANALYSIS")
        print("="*60)
        
        # Show estimated size
        print(f"\n📊 Estimated disk space needed: {requirements['estimated_size_gb']:.1f} GB")
        
        # Check available space
        import shutil
        stat = shutil.disk_usage(self.base_dir)
        available_gb = stat.free / (1024**3)
        print(f"💾 Available disk space: {available_gb:.1f} GB")
        
        if available_gb < requirements['estimated_size_gb']:
            print("⚠️  Warning: Not enough disk space!")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return decisions
        
        # Check components
        print("\n🔧 Required Components:")
        for component in requirements['components']:
            component_name = component['name']
            component_desc = component['definition'].get('description', component_name)
            is_installed = self.check_component_installed(component_name)
            
            status = "✅ Installed" if is_installed else "❌ Not installed"
            print(f"  • {component_name}: {component_desc}")
            print(f"    Status: {status}")
            
            if not is_installed:
                response = input(f"    Install {component_name}? (y/n): ")
                if response.lower() == 'y':
                    decisions['install_components'].append(component)
                else:
                    decisions['skip_items'].append(component_name)
        
        # Check system dependencies
        if requirements['dependencies']:
            print("\n⚙️  System Dependencies:")
            for dep in requirements['dependencies']:
                # Check if system dependency is available
                dep_available = shutil.which(dep) is not None
                status = "✅ Available" if dep_available else "❌ Not available"
                print(f"  • {dep}: {status}")
                
                if not dep_available:
                    print(f"    Please install {dep} via your system package manager")
        
        # Check models
        print("\n🤖 Required Models:")
        for model in requirements['models']:
            print(f"  • {model['name']} (for {model.get('component', 'unknown')})")
            
            # Check if model exists (for now, only Ollama models)
            exists = False
            if model.get('component') == 'ollama':
                try:
                    result = subprocess.run(['ollama', 'list'], 
                                          capture_output=True, text=True, timeout=10)
                    exists = model['name'] in result.stdout if result.returncode == 0 else False
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    exists = False
            
            status = "✅ Available" if exists else "❌ Not available"
            print(f"    Status: {status}")
            
            if not exists:
                response = input(f"    Download {model['name']}? (y/n): ")
                if response.lower() == 'y':
                    decisions['download_models'].append(model)
                else:
                    decisions['skip_items'].append(model['name'])
        
        return decisions
    
    def execute_setup(self, requirements: Dict[str, Any], decisions: Dict[str, Any]) -> bool:
        """Execute the setup based on requirements and decisions."""
        success = True
        
        # Install components
        for component in decisions['install_components']:
            component_name = component['name']
            
            # Check if already installed first
            if self.check_component_installed(component_name):
                print(f"✅ {component_name} is already installed")
                continue
                
            print(f"\n📦 Installing {component_name}...")
            
            # Use the generic component installer
            if self.install_component(component_name):
                print(f"✅ {component_name} installed successfully")
            else:
                print(f"❌ Failed to install {component_name}")
                success = False
        
        # Download models
        for model in decisions['download_models']:
            print(f"\n📥 Downloading {model['name']}...")
            
            if model.get('component') == 'ollama':
                if not self.download_ollama_model(model['name']):
                    success = False
                else:
                    print(f"✅ Downloaded {model['name']}")
            else:
                print(f"⚠️  Download method for {model.get('component', 'unknown')} not implemented yet")
        
        return success
    
    def setup_from_strategy(self, strategy_path: Path, interactive: bool = True) -> bool:
        """Complete setup from a strategy file."""
        print(f"🔍 Analyzing strategy: {strategy_path}")
        
        # Analyze requirements
        requirements = self.analyze_strategy(strategy_path)
        
        if interactive:
            # Get user decisions
            decisions = self.interactive_setup(requirements)
            
            # Confirm
            print("\n" + "="*60)
            print("SETUP SUMMARY")
            print("="*60)
            print(f"Components to install: {[c['name'] for c in decisions['install_components']]}")
            print(f"Models to download: {[m['name'] for m in decisions['download_models']]}")
            print(f"Skipped items: {decisions['skip_items']}")
            
            response = input("\nProceed with setup? (y/n): ")
            if response.lower() != 'y':
                print("Setup cancelled")
                return False
        else:
            # Auto mode - install everything
            decisions = {
                'install_components': requirements['components'],
                'download_models': requirements['models'],
                'skip_items': []
            }
        
        # Execute setup
        print("\n🚀 Starting setup...")
        success = self.execute_setup(requirements, decisions)
        
        if success:
            print("\n✅ Setup completed successfully!")
        else:
            print("\n⚠️  Setup completed with some errors")
        
        return success
    
    def verify_setup(self, strategy_path: Path) -> Dict[str, Any]:
        """Verify that all requirements are met."""
        requirements = self.analyze_strategy(strategy_path)
        
        verification = {
            'ready': True,
            'missing_components': [],
            'missing_models': [],
            'missing_dependencies': [],
            'warnings': []
        }
        
        # Check components
        for component in requirements['components']:
            component_name = component['name']
            if not self.check_component_installed(component_name):
                verification['missing_components'].append(component_name)
                verification['ready'] = False
        
        # Check system dependencies
        for dep in requirements['dependencies']:
            if not shutil.which(dep):
                verification['missing_dependencies'].append(dep)
                verification['ready'] = False
        
        # Check models
        for model in requirements['models']:
            model_available = False
            if model.get('component') == 'ollama':
                try:
                    result = subprocess.run(['ollama', 'list'], 
                                          capture_output=True, text=True, timeout=10)
                    model_available = model['name'] in result.stdout if result.returncode == 0 else False
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    model_available = False
                    
            if not model_available:
                verification['missing_models'].append(model['name'])
                verification['ready'] = False
        
        return verification


def main():
    """CLI entry point for setup manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup tools and models for LlamaFarm")
    parser.add_argument('strategy_file', help='Path to strategy YAML file')
    parser.add_argument('--auto', action='store_true', 
                       help='Automatic mode (no prompts)')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify setup, don\'t install')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    manager = SetupManager(verbose=args.verbose)
    strategy_path = Path(args.strategy_file)
    
    if not strategy_path.exists():
        print(f"❌ Strategy file not found: {strategy_path}")
        sys.exit(1)
    
    if args.verify_only:
        verification = manager.verify_setup(strategy_path)
        
        print("\n" + "="*60)
        print("SETUP VERIFICATION")
        print("="*60)
        
        if verification['ready']:
            print("✅ All requirements are met!")
        else:
            print("❌ Missing components:")
            if verification['missing_tools']:
                print(f"  Tools: {verification['missing_tools']}")
            if verification['missing_models']:
                print(f"  Models: {verification['missing_models']}")
        
        sys.exit(0 if verification['ready'] else 1)
    
    success = manager.setup_from_strategy(strategy_path, interactive=not args.auto)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()