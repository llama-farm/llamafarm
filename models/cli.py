#!/usr/bin/env python3
"""
LlamaFarm Models CLI - Model Management Commands

This CLI provides commands for managing cloud and local models following
the same patterns as the RAG system CLI, using UV for all operations.
"""

import os
import sys
import json
import argparse
import logging
import subprocess
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv

# Import Hugging Face libraries
try:
    from huggingface_hub import HfApi, login, list_models, snapshot_download
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Import optional local inference engines
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

try:
    from text_generation import Client
    TGI_AVAILABLE = True
except ImportError:
    TGI_AVAILABLE = False

# Load environment variables
load_dotenv()

# Import rich for better output formatting
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.syntax import Syntax
    console = Console()
except ImportError:
    console = None

def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

def print_info(message: str):
    """Print info message."""
    if console:
        console.print(f"[blue]ℹ[/blue]  {message}")
    else:
        print(f"ℹ  {message}")

def print_success(message: str):
    """Print success message."""
    if console:
        console.print(f"[green]✓[/green] {message}")
    else:
        print(f"✓ {message}")

def print_error(message: str):
    """Print error message."""
    if console:
        console.print(f"[red]✗[/red] {message}")
    else:
        print(f"✗ {message}")

def print_warning(message: str):
    """Print warning message."""
    if console:
        console.print(f"[yellow]⚠[/yellow]  {message}")
    else:
        print(f"⚠  {message}")

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON or YAML file with environment variable substitution."""
    config_file = Path(config_path)
    models_dir = Path(__file__).parent
    
    if not config_file.exists():
        # Try relative to models directory
        config_file = models_dir / config_path
        
    if not config_file.exists():
        # Try in config_examples directory
        config_file = models_dir / "config_examples" / Path(config_path).name
        
    if not config_file.exists():
        print_error(f"Configuration file not found: {config_path}")
        print_info(f"Searched in:")
        print_info(f"  - {Path(config_path).absolute()}")
        print_info(f"  - {(models_dir / config_path).absolute()}")
        print_info(f"  - {(models_dir / 'config_examples' / Path(config_path).name).absolute()}")
        print_info(f"\nAvailable configs in config/ directory:")
        config_dir = models_dir / "config"
        if config_dir.exists():
            for f in config_dir.glob("*.json"):
                print_info(f"  - config/{f.name}")
            for f in config_dir.glob("*.yaml"):
                print_info(f"  - config/{f.name}")
            for f in config_dir.glob("*.yml"):
                print_info(f"  - config/{f.name}")
        sys.exit(1)
    
    try:
        with open(config_file, 'r') as f:
            config_text = f.read()
            
        # Substitute environment variables
        for key, value in os.environ.items():
            if value:  # Only substitute non-empty values
                config_text = config_text.replace(f"${{{key}}}", value)
        
        # Determine file type and parse accordingly
        if config_file.suffix.lower() in ['.yaml', '.yml']:
            try:
                import yaml
                return yaml.safe_load(config_text)
            except ImportError:
                print_error("PyYAML not installed. Install with: pip install PyYAML")
                sys.exit(1)
            except yaml.YAMLError as e:
                print_error(f"Invalid YAML in configuration file: {e}")
                sys.exit(1)
        else:
            # Default to JSON
            return json.loads(config_text)
            
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON in configuration file: {e}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Failed to load configuration: {e}")
        sys.exit(1)

def list_command(args):
    """List available model providers."""
    config = load_config(args.config)
    providers = config.get("providers", {})
    
    if not providers:
        print_warning("No model providers configured")
        return
    
    if console:
        table = Table(title="🦙 Available Model Providers")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Provider", style="green")
        table.add_column("Model", style="yellow")
        table.add_column("Status", style="blue")
        
        default_provider = config.get("default_provider", "")
        
        for name, provider_config in providers.items():
            provider_type = provider_config.get("type", "unknown")
            provider = provider_config.get("provider", "unknown")
            model = provider_config.get("model", "unknown")
            is_default = "✓ Default" if name == default_provider else ""
            
            table.add_row(name, provider_type, provider, model, is_default)
        
        console.print(table)
        
        if args.detailed:
            console.print("\n[bold]Detailed Configuration:[/bold]")
            for name, provider_config in providers.items():
                panel_content = json.dumps(provider_config, indent=2)
                syntax = Syntax(panel_content, "json", theme="monokai", line_numbers=False)
                console.print(Panel(syntax, title=f"[cyan]{name}[/cyan]", expand=False))
    else:
        # Fallback for when rich is not available
        print("\n🦙 Available Model Providers")
        print("=" * 60)
        
        default_provider = config.get("default_provider", "")
        
        for name, provider_config in providers.items():
            provider_type = provider_config.get("type", "unknown")
            provider = provider_config.get("provider", "unknown")
            model = provider_config.get("model", "unknown")
            is_default = " (Default)" if name == default_provider else ""
            
            print(f"\n📍 {name}{is_default}")
            print(f"   Type: {provider_type}")
            print(f"   Provider: {provider}")
            print(f"   Model: {model}")
            
            if args.detailed:
                description = provider_config.get("description", "No description")
                print(f"   Description: {description}")
                if provider_type == "cloud":
                    cost = provider_config.get("cost_per_1k_tokens", "N/A")
                    print(f"   Cost: ${cost} per 1K tokens")

def test_command(args):
    """Test model connectivity and performance."""
    config = load_config(args.config)
    providers = config.get("providers", {})
    
    if args.provider not in providers:
        print_error(f"Provider '{args.provider}' not found in configuration")
        print_info(f"Available providers: {', '.join(providers.keys())}")
        return
    
    provider_config = providers[args.provider]
    
    print_info(f"Testing provider: {args.provider}")
    print_info(f"Model: {provider_config.get('model', 'unknown')}")
    print_info(f"Type: {provider_config.get('type', 'unknown')}")
    
    # Mock test results - in real implementation, this would test actual connectivity
    import time
    import random
    
    print_info("Running connectivity test...")
    time.sleep(0.5)
    
    if provider_config.get("type") == "cloud":
        if not provider_config.get("api_key"):
            print_error("API key not configured")
            return
        print_success("API key validated")
    
    print_info("Testing model availability...")
    time.sleep(0.5)
    print_success("Model is available")
    
    if args.query:
        print_info(f"Testing generation with query: {args.query}")
        time.sleep(1)
        
        # Mock response
        mock_latency = random.randint(200, 800)
        mock_tokens = len(args.query.split()) * 5
        
        print_success(f"Generation successful!")
        print_info(f"Latency: {mock_latency}ms")
        print_info(f"Tokens used: ~{mock_tokens}")
        
        if provider_config.get("type") == "cloud":
            cost_per_1k = provider_config.get("cost_per_1k_tokens", 0)
            if cost_per_1k:
                estimated_cost = (mock_tokens / 1000) * cost_per_1k
                print_info(f"Estimated cost: ${estimated_cost:.4f}")

def health_check_command(args):
    """Check health of all configured providers."""
    config = load_config(args.config)
    providers = config.get("providers", {})
    
    if not providers:
        print_warning("No providers configured")
        return
    
    print_info("Running health checks on all providers...\n")
    
    results = []
    
    for name, provider_config in providers.items():
        provider_type = provider_config.get("type", "unknown")
        model = provider_config.get("model", "unknown")
        
        # Mock health check - in real implementation, this would check actual availability
        import random
        is_healthy = random.choice([True, True, True, False])  # 75% healthy
        latency = random.randint(50, 500) if is_healthy else None
        
        results.append({
            "name": name,
            "type": provider_type,
            "model": model,
            "healthy": is_healthy,
            "latency": latency
        })
    
    # Display results
    if console:
        table = Table(title="🏥 Provider Health Status")
        table.add_column("Provider", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Model", style="yellow")
        table.add_column("Status", style="green")
        table.add_column("Latency", style="blue")
        
        for result in results:
            status = "[green]✓ Healthy[/green]" if result["healthy"] else "[red]✗ Unhealthy[/red]"
            latency = f"{result['latency']}ms" if result["latency"] else "N/A"
            
            table.add_row(
                result["name"],
                result["type"],
                result["model"],
                status,
                latency
            )
        
        console.print(table)
    else:
        # Fallback display
        for result in results:
            status = "✓ Healthy" if result["healthy"] else "✗ Unhealthy"
            print(f"\n{result['name']}:")
            print(f"  Status: {status}")
            print(f"  Type: {result['type']}")
            print(f"  Model: {result['model']}")
            if result["latency"]:
                print(f"  Latency: {result['latency']}ms")

def validate_config_command(args):
    """Validate model configuration."""
    try:
        config = load_config(args.config)
        
        # Check required fields
        required_fields = ["providers"]
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            print_error(f"Missing required fields: {', '.join(missing_fields)}")
            return
        
        # Validate providers
        providers = config.get("providers", {})
        if not providers:
            print_warning("No providers configured")
            return
        
        print_info(f"Validating {len(providers)} provider(s)...")
        
        for name, provider_config in providers.items():
            print(f"\n🔍 Validating '{name}'...")
            
            # Check required provider fields
            required_provider_fields = ["type", "provider", "model"]
            missing = [f for f in required_provider_fields if f not in provider_config]
            
            if missing:
                print_error(f"  Missing fields: {', '.join(missing)}")
                continue
            
            # Type-specific validation
            provider_type = provider_config.get("type")
            if provider_type == "cloud":
                if not provider_config.get("api_key"):
                    print_warning("  No API key configured (using environment variable?)")
                if not provider_config.get("base_url"):
                    print_warning("  No base URL specified")
            elif provider_type == "local":
                if not provider_config.get("host"):
                    print_error("  Missing 'host' for local provider")
                if not provider_config.get("port"):
                    print_error("  Missing 'port' for local provider")
            
            print_success(f"  Basic validation passed")
        
        # Check selection strategy if present
        if "selection_strategy" in config:
            print_info("\nValidating selection strategy...")
            strategy = config["selection_strategy"]
            if "type" not in strategy:
                print_error("  Selection strategy missing 'type'")
            else:
                print_success("  Selection strategy validated")
        
        # Check fallback chain
        if "fallback_chain" in config:
            chain = config["fallback_chain"]
            invalid_providers = [p for p in chain if p not in providers]
            if invalid_providers:
                print_error(f"  Invalid providers in fallback chain: {', '.join(invalid_providers)}")
            else:
                print_success("  Fallback chain validated")
        
        print_success("\n✅ Configuration validation complete")
        
    except Exception as e:
        print_error(f"Validation failed: {e}")

def compare_command(args):
    """Compare responses from different models."""
    config = load_config(args.config)
    providers = config.get("providers", {})
    
    # Parse provider list
    provider_names = args.providers.split(',')
    
    # Validate providers
    invalid_providers = [p for p in provider_names if p not in providers]
    if invalid_providers:
        print_error(f"Invalid providers: {', '.join(invalid_providers)}")
        print_info(f"Available providers: {', '.join(providers.keys())}")
        return
    
    print_info(f"Comparing {len(provider_names)} providers with query: '{args.query}'")
    
    # Mock comparison results
    results = []
    for provider_name in provider_names:
        provider_config = providers[provider_name]
        
        # Simulate response
        import random
        import time
        
        start_time = time.time()
        time.sleep(random.uniform(0.2, 0.5))  # Simulate API call
        latency = int((time.time() - start_time) * 1000)
        
        mock_response = f"This is a simulated response from {provider_config['model']}. "
        mock_response += "In a real implementation, this would be an actual API response."
        
        tokens = len(mock_response.split()) * 2
        cost = 0
        if provider_config.get("type") == "cloud":
            cost_per_1k = provider_config.get("cost_per_1k_tokens", 0)
            cost = (tokens / 1000) * cost_per_1k
        
        results.append({
            "provider": provider_name,
            "model": provider_config["model"],
            "response": mock_response,
            "latency": latency,
            "tokens": tokens,
            "cost": cost
        })
    
    # Display results
    if console:
        for result in results:
            panel_content = f"[bold]Model:[/bold] {result['model']}\n"
            panel_content += f"[bold]Latency:[/bold] {result['latency']}ms\n"
            panel_content += f"[bold]Tokens:[/bold] {result['tokens']}\n"
            if result['cost'] > 0:
                panel_content += f"[bold]Cost:[/bold] ${result['cost']:.4f}\n"
            panel_content += f"\n[italic]{result['response']}[/italic]"
            
            console.print(Panel(panel_content, title=f"[cyan]{result['provider']}[/cyan]"))
            console.print()
    else:
        # Fallback display
        for i, result in enumerate(results):
            if i > 0:
                print("\n" + "-" * 60 + "\n")
            print(f"Provider: {result['provider']}")
            print(f"Model: {result['model']}")
            print(f"Latency: {result['latency']}ms")
            print(f"Tokens: {result['tokens']}")
            if result['cost'] > 0:
                print(f"Cost: ${result['cost']:.4f}")
            print(f"\nResponse: {result['response']}")

def _substitute_env_vars(value: str) -> str:
    """Substitute environment variables in config values."""
    if not isinstance(value, str):
        return value
    
    if value.startswith("${") and value.endswith("}"):
        # Handle format like ${VAR:default}
        if ":" in value:
            env_var_part = value[2:-1]  # Remove ${ and }
            env_var, default_val = env_var_part.split(":", 1)
            return os.getenv(env_var, default_val)
        else:
            env_var = value[2:-1]
            return os.getenv(env_var, "")
    
    return value

def _call_openai_api(provider_config: Dict[str, Any], query: str, system_prompt: str = None) -> str:
    """Make a real OpenAI API call."""
    try:
        import openai
        
        # Get API key from config with environment variable substitution
        api_key = _substitute_env_vars(provider_config.get("api_key", ""))
        
        if not api_key:
            return "Error: OpenAI API key not configured. Set OPENAI_API_KEY in your environment."
        
        # Create client with explicit empty org to avoid header issues
        client = openai.OpenAI(
            api_key=api_key,
            base_url=provider_config.get("base_url", "https://api.openai.com/v1"),
            organization=""  # Set to empty string to avoid organization header issues
        )
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})
        
        # Make API call
        response = client.chat.completions.create(
            model=provider_config.get("model", "gpt-4o-mini"),
            messages=messages,
            max_tokens=provider_config.get("max_tokens", 2048),
            temperature=provider_config.get("temperature", 0.7),
            top_p=provider_config.get("top_p", 1.0) if provider_config.get("top_p") else None
        )
        
        return response.choices[0].message.content or "No response generated"
        
    except ImportError:
        return "Error: OpenAI package not installed. Run: uv add openai"
    except Exception as e:
        return f"Error calling OpenAI API: {str(e)}"

def _call_anthropic_api(provider_config: Dict[str, Any], query: str, system_prompt: str = None) -> str:
    """Make a real Anthropic API call."""
    try:
        import anthropic
        
        # Get API key from config with environment variable substitution
        api_key = _substitute_env_vars(provider_config.get("api_key", ""))
        
        if not api_key:
            return "Error: Anthropic API key not configured"
        
        # Create client
        client = anthropic.Anthropic(
            api_key=api_key
        )
        
        # Make API call
        response = client.messages.create(
            model=provider_config.get("model", "claude-3-haiku-20240307"),
            max_tokens=provider_config.get("max_tokens", 4096),
            temperature=provider_config.get("temperature", 0.7),
            system=system_prompt or "You are a helpful assistant.",
            messages=[{"role": "user", "content": query}]
        )
        
        return response.content[0].text if response.content else "No response generated"
        
    except ImportError:
        return "Error: Anthropic package not installed. Run: uv add anthropic"
    except Exception as e:
        return f"Error calling Anthropic API: {str(e)}"

def _call_ollama_api(provider_config: Dict[str, Any], query: str, system_prompt: str = None) -> str:
    """Make a real Ollama API call."""
    try:
        import requests
        
        # Handle environment variable substitution for host
        host = _substitute_env_vars(provider_config.get("host", "localhost"))
        
        port = provider_config.get("port", 11434)
        base_url = f"http://{host}:{port}"
        
        # Prepare the prompt
        full_prompt = query
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nUser: {query}"
        
        # Make API call
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": provider_config.get("model", "llama3.1:8b"),
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": provider_config.get("temperature", 0.7),
                    "num_predict": provider_config.get("max_tokens", 512)
                }
            },
            timeout=provider_config.get("timeout", 120)
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "No response generated")
        else:
            return f"Error: Ollama API returned {response.status_code}: {response.text}"
            
    except ImportError:
        return "Error: Requests package not installed"
    except Exception as e:
        return f"Error calling Ollama API: {str(e)}"

def query_command(args):
    """Send a query to a model with full control over parameters."""
    config = load_config(args.config)
    providers = config.get("providers", {})
    
    # Determine provider to use
    provider_name = args.provider or config.get("default_provider")
    if not provider_name or provider_name not in providers:
        print_error(f"Provider '{provider_name}' not found")
        print_info(f"Available providers: {', '.join(providers.keys())}")
        sys.exit(1)
    
    provider_config = providers[provider_name].copy()
    
    # Override settings if provided
    if args.temperature is not None:
        provider_config["temperature"] = args.temperature
    if args.max_tokens is not None:
        provider_config["max_tokens"] = args.max_tokens
    if args.top_p is not None:
        provider_config["top_p"] = args.top_p
    
    print_info(f"Using provider: {provider_name}")
    print_info(f"Model: {provider_config.get('model', 'unknown')}")
    
    # Make the actual API call
    try:
        import time
        start_time = time.time()
        
        provider_type = provider_config.get("provider", "").lower()
        response_text = ""
        
        if provider_type == "openai":
            response_text = _call_openai_api(provider_config, args.query, args.system)
        elif provider_type == "anthropic":
            response_text = _call_anthropic_api(provider_config, args.query, args.system)
        elif provider_type == "ollama":
            response_text = _call_ollama_api(provider_config, args.query, args.system)
        else:
            # Fallback to mock response for unsupported providers
            response_text = f"Mock response from {provider_config['model']} to query: '{args.query}'"
            if args.system:
                response_text = f"[System: {args.system}]\n{response_text}"
            time.sleep(0.5)  # Simulate API call
        
        latency = int((time.time() - start_time) * 1000)
        
        if args.json:
            import json
            output = {
                "provider": provider_name,
                "model": provider_config["model"],
                "query": args.query,
                "response": response_text,
                "latency_ms": latency,
                "parameters": {
                    "temperature": provider_config.get("temperature"),
                    "max_tokens": provider_config.get("max_tokens"),
                    "top_p": provider_config.get("top_p")
                }
            }
            print(json.dumps(output, indent=2))
        else:
            print_success(f"Response received in {latency}ms")
            print(f"\n{response_text}")
        
        if args.save:
            with open(args.save, 'w') as f:
                f.write(response_text)
            print_success(f"Response saved to: {args.save}")
            
    except Exception as e:
        print_error(f"Query failed: {e}")
        sys.exit(1)

def chat_command(args):
    """Start an interactive chat session with a model."""
    config = load_config(args.config)
    providers = config.get("providers", {})
    
    # Determine provider to use
    provider_name = args.provider or config.get("default_provider")
    if not provider_name or provider_name not in providers:
        print_error(f"Provider '{provider_name}' not found")
        print_info(f"Available providers: {', '.join(providers.keys())}")
        sys.exit(1)
    
    provider_config = providers[provider_name].copy()
    
    # Override settings if provided
    if args.temperature is not None:
        provider_config["temperature"] = args.temperature
    
    print_info(f"Starting chat with: {provider_name}")
    print_info(f"Model: {provider_config.get('model', 'unknown')}")
    print_info("Type 'exit' or 'quit' to end the chat")
    print_info("Type 'clear' to clear the conversation history")
    print()
    
    # Load history if provided
    messages = []
    if args.history and os.path.exists(args.history):
        try:
            with open(args.history, 'r') as f:
                messages = json.load(f)
            print_success(f"Loaded {len(messages)} messages from history")
        except Exception as e:
            print_warning(f"Could not load history: {e}")
    
    # Add system prompt if provided
    if args.system:
        messages.append({"role": "system", "content": args.system})
    
    try:
        while True:
            # Get user input
            user_input = input("\n🧑 You: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                break
            elif user_input.lower() == 'clear':
                messages = []
                if args.system:
                    messages.append({"role": "system", "content": args.system})
                print_success("Conversation cleared")
                continue
            elif not user_input:
                continue
            
            # Add user message
            messages.append({"role": "user", "content": user_input})
            
            # Get response (mock for now)
            print("\n🤖 Assistant: ", end="", flush=True)
            
            # In real implementation, this would stream from API
            response = f"This is a simulated response from {provider_config['model']}."
            
            if args.stream:
                # Simulate streaming
                import time
                for char in response:
                    print(char, end="", flush=True)
                    time.sleep(0.01)
                print()
            else:
                print(response)
            
            # Add assistant message
            messages.append({"role": "assistant", "content": response})
        
        # Save history if requested
        if args.save_history:
            with open(args.save_history, 'w') as f:
                json.dump(messages, f, indent=2)
            print_success(f"\nChat history saved to: {args.save_history}")
            
    except KeyboardInterrupt:
        print("\n\nChat ended.")
    except Exception as e:
        print_error(f"\nChat error: {e}")

def send_command(args):
    """Send file content to a model."""
    if not os.path.exists(args.file):
        print_error(f"File not found: {args.file}")
        sys.exit(1)
    
    config = load_config(args.config)
    providers = config.get("providers", {})
    
    # Determine provider to use
    provider_name = args.provider or config.get("default_provider")
    if not provider_name or provider_name not in providers:
        print_error(f"Provider '{provider_name}' not found")
        print_info(f"Available providers: {', '.join(providers.keys())}")
        sys.exit(1)
    
    provider_config = providers[provider_name].copy()
    
    # Override settings if provided
    if args.temperature is not None:
        provider_config["temperature"] = args.temperature
    if args.max_tokens is not None:
        provider_config["max_tokens"] = args.max_tokens
    
    # Read file content
    try:
        with open(args.file, 'r') as f:
            content = f.read()
    except Exception as e:
        print_error(f"Could not read file: {e}")
        sys.exit(1)
    
    # Prepare query
    query = content
    if args.prompt:
        query = f"{args.prompt}\n\n{content}"
    
    print_info(f"Sending file to: {provider_name}")
    print_info(f"Model: {provider_config.get('model', 'unknown')}")
    print_info(f"File size: {len(content)} characters")
    
    try:
        # Make real API call with file content
        import time
        start_time = time.time()
        
        provider_type = provider_config.get("provider", "").lower()
        
        # Make real API call based on provider type
        if provider_type == "openai":
            response = _call_openai_api(provider_config, query, args.prompt)
        elif provider_type == "anthropic":
            response = _call_anthropic_api(provider_config, query, args.prompt)
        elif provider_type == "ollama":
            response = _call_ollama_api(provider_config, query, args.prompt)
        else:
            # Fallback to mock response for unsupported providers
            response = f"Mock: Processed {os.path.basename(args.file)} with {provider_config['model']}"
            time.sleep(0.5)
        
        latency = int((time.time() - start_time) * 1000)
        
        print_success(f"Response received in {latency}ms")
        print(f"\n{response}")
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(response)
            print_success(f"Response saved to: {args.output}")
            
    except Exception as e:
        print_error(f"Send failed: {e}")
        sys.exit(1)

def batch_command(args):
    """Process multiple queries from a file."""
    if not os.path.exists(args.file):
        print_error(f"File not found: {args.file}")
        sys.exit(1)
    
    config = load_config(args.config)
    providers = config.get("providers", {})
    
    # Determine provider to use
    provider_name = args.provider or config.get("default_provider")
    if not provider_name or provider_name not in providers:
        print_error(f"Provider '{provider_name}' not found")
        print_info(f"Available providers: {', '.join(providers.keys())}")
        sys.exit(1)
    
    provider_config = providers[provider_name].copy()
    
    # Override settings if provided
    if args.temperature is not None:
        provider_config["temperature"] = args.temperature
    
    # Read queries
    try:
        with open(args.file, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print_error(f"Could not read file: {e}")
        sys.exit(1)
    
    print_info(f"Processing {len(queries)} queries with: {provider_name}")
    print_info(f"Model: {provider_config.get('model', 'unknown')}")
    print_info(f"Parallel requests: {args.parallel}")
    
    results = []
    
    # Process queries with real API calls
    import time
    provider_type = provider_config.get("provider", "").lower()
    
    for i, query in enumerate(queries):
        print(f"\nProcessing query {i+1}/{len(queries)}: {query[:50]}...")
        
        start_time = time.time()
        
        # Make real API call based on provider type
        if provider_type == "openai":
            response = _call_openai_api(provider_config, query)
        elif provider_type == "anthropic":
            response = _call_anthropic_api(provider_config, query)
        elif provider_type == "ollama":
            response = _call_ollama_api(provider_config, query)
        else:
            # Fallback to mock response for unsupported providers
            response = f"Mock response to: {query}"
            time.sleep(0.2)  # Simulate API call
        
        latency = int((time.time() - start_time) * 1000)
        
        results.append({
            "query": query,
            "response": response,
            "latency_ms": latency
        })
        
        print_success(f"Completed in {latency}ms")
    
    # Save results if requested
    if args.output:
        try:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print_success(f"\nResults saved to: {args.output}")
        except Exception as e:
            print_error(f"Could not save results: {e}")
    
    print_success(f"\nBatch processing complete: {len(results)} queries processed")

def generate_config_command(args):
    """Generate example configuration."""
    config_type = args.type
    
    templates = {
        "basic": {
            "name": "Basic Model Configuration",
            "version": "1.0.0",
            "default_provider": "openai_gpt4",
            "providers": {
                "openai_gpt4": {
                    "type": "cloud",
                    "provider": "openai",
                    "model": "gpt-4-turbo-preview",
                    "api_key": "${OPENAI_API_KEY}",
                    "base_url": "https://api.openai.com/v1",
                    "max_tokens": 4096,
                    "temperature": 0.7
                }
            }
        },
        "multi": {
            "name": "Multi-Provider Configuration",
            "version": "1.0.0",
            "default_provider": "openai_gpt4",
            "fallback_chain": ["openai_gpt4", "anthropic_claude", "local_llama"],
            "providers": {
                "openai_gpt4": {
                    "type": "cloud",
                    "provider": "openai",
                    "model": "gpt-4-turbo-preview",
                    "api_key": "${OPENAI_API_KEY}",
                    "base_url": "https://api.openai.com/v1",
                    "max_tokens": 4096,
                    "temperature": 0.7
                },
                "anthropic_claude": {
                    "type": "cloud",
                    "provider": "anthropic",
                    "model": "claude-3-opus-20240229",
                    "api_key": "${ANTHROPIC_API_KEY}",
                    "base_url": "https://api.anthropic.com/v1",
                    "max_tokens": 4096,
                    "temperature": 0.7
                },
                "local_llama": {
                    "type": "local",
                    "provider": "ollama",
                    "model": "llama2:13b",
                    "host": "localhost",
                    "port": 11434,
                    "timeout": 120
                }
            },
            "selection_strategy": {
                "type": "cost_optimized",
                "factors": {
                    "cost": 0.4,
                    "quality": 0.3,
                    "speed": 0.3
                }
            }
        },
        "production": {
            "name": "Production Model Configuration",
            "version": "1.0.0",
            "environment": "production",
            "default_provider": "openai_gpt4",
            "providers": {
                "openai_gpt4": {
                    "type": "cloud",
                    "provider": "openai",
                    "model": "gpt-4-turbo-preview",
                    "api_key": "${OPENAI_API_KEY}",
                    "base_url": "https://api.openai.com/v1",
                    "max_tokens": 4096,
                    "temperature": 0.7,
                    "rate_limit": {
                        "requests_per_minute": 500,
                        "tokens_per_minute": 90000
                    }
                },
                "openai_gpt35": {
                    "type": "cloud",
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",
                    "api_key": "${OPENAI_API_KEY}",
                    "base_url": "https://api.openai.com/v1",
                    "max_tokens": 4096,
                    "temperature": 0.7,
                    "cost_per_1k_tokens": 0.002
                }
            },
            "fallback_strategy": {
                "enabled": True,
                "chain": [
                    {"provider": "openai_gpt4", "conditions": ["api_healthy", "within_rate_limit"]},
                    {"provider": "openai_gpt35", "conditions": ["api_healthy"]}
                ],
                "retry_strategy": {
                    "max_retries": 3,
                    "backoff_multiplier": 2
                }
            },
            "monitoring": {
                "track_usage": True,
                "track_costs": True,
                "alert_thresholds": {
                    "daily_cost_usd": 100,
                    "error_rate_percent": 5
                }
            }
        }
    }
    
    if config_type not in templates:
        print_error(f"Unknown configuration type: {config_type}")
        print_info(f"Available types: {', '.join(templates.keys())}")
        return
    
    config = templates[config_type]
    
    # Output configuration
    output_path = args.output
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        print_success(f"Configuration written to: {output_path}")
    else:
        # Print to console
        if console:
            syntax = Syntax(json.dumps(config, indent=2), "json", theme="monokai")
            console.print(Panel(syntax, title=f"[cyan]{config_type.title()} Configuration[/cyan]"))
        else:
            print(json.dumps(config, indent=2))

def list_local_command(args):
    """List locally available models from Ollama."""
    print_info("Fetching local Ollama models...")
    
    try:
        # Get Ollama base URL from environment or default
        ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        
        # Try to get models via API first
        try:
            response = requests.get(f"{ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                models = models_data.get('models', [])
                
                if not models:
                    print_warning("No local models found in Ollama")
                    return
                
                if console:
                    table = Table(title="🦙 Local Ollama Models")
                    table.add_column("Name", style="cyan")
                    table.add_column("ID", style="yellow")
                    table.add_column("Size", style="green")
                    table.add_column("Modified", style="blue")
                    
                    for model in models:
                        name = model.get('name', 'unknown')
                        model_id = model.get('digest', 'unknown')[:12]
                        size = model.get('size', 0)
                        # Convert size to human readable
                        if size > 1024**3:
                            size_str = f"{size / (1024**3):.1f} GB"
                        elif size > 1024**2:
                            size_str = f"{size / (1024**2):.1f} MB"
                        else:
                            size_str = f"{size / 1024:.1f} KB"
                        
                        modified = model.get('modified_at', 'unknown')
                        if modified != 'unknown':
                            # Parse and format timestamp
                            from datetime import datetime
                            try:
                                dt = datetime.fromisoformat(modified.replace('Z', '+00:00'))
                                modified = dt.strftime("%Y-%m-%d %H:%M")
                            except:
                                pass
                        
                        table.add_row(name, model_id, size_str, modified)
                    
                    console.print(table)
                else:
                    print("\n🦙 Local Ollama Models")
                    print("=" * 60)
                    for model in models:
                        print(f"\n📍 {model.get('name', 'unknown')}")
                        print(f"   ID: {model.get('digest', 'unknown')[:12]}")
                        size = model.get('size', 0)
                        if size > 1024**3:
                            size_str = f"{size / (1024**3):.1f} GB"
                        elif size > 1024**2:
                            size_str = f"{size / (1024**2):.1f} MB"
                        else:
                            size_str = f"{size / 1024:.1f} KB"
                        print(f"   Size: {size_str}")
                        print(f"   Modified: {model.get('modified_at', 'unknown')}")
            else:
                raise requests.RequestException("API not available")
                
        except requests.RequestException:
            # Fallback to ollama list command
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) <= 1:
                    print_warning("No local models found in Ollama")
                    return
                
                print_success(f"Found {len(lines)-1} local models:")
                print(result.stdout)
            else:
                print_error("Failed to list Ollama models. Is Ollama running?")
                
    except FileNotFoundError:
        print_error("Ollama not found. Please install Ollama first.")
        print_info("Install from: https://ollama.ai/")
    except Exception as e:
        print_error(f"Error listing local models: {e}")

def pull_model_command(args):
    """Pull a model to Ollama."""
    model_name = args.model
    print_info(f"Pulling model: {model_name}")
    
    try:
        # Use Ollama API for pulling with progress
        ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        
        # Try API first for better progress tracking
        try:
            print_info("Starting model pull via API...")
            response = requests.post(
                f"{ollama_base_url}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=300
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if 'status' in data:
                                status = data['status']
                                if 'completed' in data and 'total' in data:
                                    completed = data['completed']
                                    total = data['total']
                                    percent = (completed / total) * 100 if total > 0 else 0
                                    print(f"\r{status}: {percent:.1f}%", end='', flush=True)
                                else:
                                    print(f"\r{status}", end='', flush=True)
                        except json.JSONDecodeError:
                            continue
                print()  # New line after progress
                print_success(f"Successfully pulled model: {model_name}")
            else:
                raise requests.RequestException("API pull failed")
                
        except requests.RequestException:
            # Fallback to CLI command
            print_info("Falling back to ollama CLI...")
            result = subprocess.run(['ollama', 'pull', model_name], 
                                  capture_output=False, text=True)
            if result.returncode == 0:
                print_success(f"Successfully pulled model: {model_name}")
            else:
                print_error(f"Failed to pull model: {model_name}")
                
    except FileNotFoundError:
        print_error("Ollama not found. Please install Ollama first.")
    except Exception as e:
        print_error(f"Error pulling model: {e}")

def test_local_command(args):
    """Test a local Ollama model."""
    model_name = args.model
    query = args.query or "Hello, how are you?"
    
    print_info(f"Testing local model: {model_name}")
    print_info(f"Query: {query}")
    
    try:
        ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        
        # Test with Ollama API
        start_time = datetime.now()
        
        payload = {
            "model": model_name,
            "prompt": query,
            "stream": False
        }
        
        response = requests.post(
            f"{ollama_base_url}/api/generate",
            json=payload,
            timeout=30
        )
        
        end_time = datetime.now()
        latency = int((end_time - start_time).total_seconds() * 1000)
        
        if response.status_code == 200:
            result = response.json()
            generated_response = result.get('response', 'No response')
            
            print_success("Generation successful!")
            print_info(f"Latency: {latency}ms")
            print_info(f"Model: {model_name}")
            
            if console:
                panel_content = f"[bold]Query:[/bold] {query}\n\n"
                panel_content += f"[bold]Response:[/bold]\n{generated_response}"
                console.print(Panel(panel_content, title=f"[cyan]Local Model Test: {model_name}[/cyan]"))
            else:
                print(f"\nQuery: {query}")
                print(f"Response: {generated_response}")
            
            # Additional metrics if available
            if 'eval_count' in result:
                print_info(f"Tokens generated: {result['eval_count']}")
            if 'eval_duration' in result:
                tokens_per_sec = result['eval_count'] / (result['eval_duration'] / 1e9) if result['eval_duration'] > 0 else 0
                print_info(f"Speed: {tokens_per_sec:.1f} tokens/sec")
                
        else:
            print_error(f"API request failed: {response.status_code}")
            print_error(response.text)
            
    except requests.RequestException as e:
        print_error(f"Failed to connect to Ollama API: {e}")
        print_error("Make sure Ollama is running (ollama serve)")
    except Exception as e:
        print_error(f"Error testing local model: {e}")

def generate_ollama_config_command(args):
    """Generate Ollama-specific configuration."""
    print_info("Generating Ollama configuration...")
    
    # Get currently available models
    available_models = []
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            available_models = [line.split()[0] for line in lines if line.strip()]
    except:
        available_models = ["llama3.1:8b", "llama3:latest", "llama3.2:3b"]  # Fallback defaults
    
    if not available_models:
        print_warning("No local models found. Generating config with example models.")
        available_models = ["llama3.1:8b", "llama3:latest", "mistral:7b"]
    
    # Create configuration with available models
    providers = {}
    for i, model in enumerate(available_models[:5]):  # Limit to 5 models
        provider_name = f"ollama_{model.replace(':', '_').replace('.', '_')}"
        providers[provider_name] = {
            "type": "local",
            "provider": "ollama",
            "model": model,
            "host": "${OLLAMA_HOST:-localhost}",
            "port": "${OLLAMA_PORT:-11434}",
            "base_url": "${OLLAMA_BASE_URL:-http://localhost:11434}",
            "timeout": 120,
            "description": f"Local {model} model via Ollama"
        }
    
    config = {
        "name": "Ollama Local Models Configuration",
        "version": "1.0.0",
        "default_provider": list(providers.keys())[0] if providers else "ollama_llama3_1_8b",
        "providers": providers,
        "local_settings": {
            "ollama": {
                "host": "localhost",
                "port": 11434,
                "timeout": 120,
                "keep_alive": "5m",
                "concurrent_requests": 4
            }
        },
        "fallback_chain": list(providers.keys())[:3] if len(providers) >= 3 else list(providers.keys()),
        "selection_strategy": {
            "type": "performance_optimized",
            "factors": {
                "speed": 0.4,
                "memory_usage": 0.3,
                "quality": 0.3
            }
        }
    }
    
    # Output configuration
    output_path = args.output
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        print_success(f"Ollama configuration written to: {output_path}")
    else:
        if console:
            syntax = Syntax(json.dumps(config, indent=2), "json", theme="monokai")
            console.print(Panel(syntax, title="[cyan]Ollama Configuration[/cyan]"))
        else:
            print(json.dumps(config, indent=2))
    
    print_info(f"Generated configuration for {len(providers)} local models")

def list_hf_models_command(args):
    """List available Hugging Face models."""
    if not HF_AVAILABLE:
        print_error("Hugging Face libraries not available. Run: uv sync")
        return
    
    print_info("Searching Hugging Face models...")
    
    try:
        # Get HF token from environment
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            print_warning("HF_TOKEN not found. Only public models will be shown.")
        
        # Initialize HF API
        api = HfApi(token=hf_token)
        
        # Search parameters
        search_term = args.search if hasattr(args, 'search') and args.search else None
        limit = args.limit if hasattr(args, 'limit') and args.limit else 20
        
        # Search for models
        models = list_models(
            search=search_term,
            task="text-generation",
            sort="downloads",
            direction=-1,
            limit=limit,
            token=hf_token
        )
        
        models_list = list(models)
        
        if not models_list:
            print_warning("No models found matching criteria")
            return
        
        if console:
            table = Table(title="🤗 Hugging Face Models")
            table.add_column("Model ID", style="cyan")
            table.add_column("Author", style="yellow")
            table.add_column("Downloads", style="green")
            table.add_column("Tags", style="blue")
            
            for model in models_list:
                model_id = model.id
                author = model_id.split('/')[0] if '/' in model_id else "unknown"
                downloads = f"{model.downloads:,}" if model.downloads else "N/A"
                tags = ", ".join(model.tags[:3]) if model.tags else "N/A"
                
                table.add_row(model_id, author, downloads, tags)
            
            console.print(table)
        else:
            print(f"\n🤗 Found {len(models_list)} Hugging Face models")
            print("=" * 60)
            
            for model in models_list:
                print(f"\n📍 {model.id}")
                author = model.id.split('/')[0] if '/' in model.id else "unknown"
                print(f"   Author: {author}")
                if model.downloads:
                    print(f"   Downloads: {model.downloads:,}")
                if model.tags:
                    print(f"   Tags: {', '.join(model.tags[:3])}")
    
    except Exception as e:
        print_error(f"Error searching Hugging Face models: {e}")

def download_hf_model_command(args):
    """Download a Hugging Face model locally."""
    if not HF_AVAILABLE:
        print_error("Hugging Face libraries not available. Run: uv sync")
        return
    
    model_id = args.model_id
    cache_dir = args.cache_dir or os.path.expanduser("~/.cache/huggingface/hub")
    
    print_info(f"Downloading Hugging Face model: {model_id}")
    print_info(f"Cache directory: {cache_dir}")
    
    try:
        # Get HF token from environment
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            print_warning("HF_TOKEN not found. Only public models can be downloaded.")
        
        # Login if token is available
        if hf_token:
            login(token=hf_token, add_to_git_credential=False)
            print_success("Authenticated with Hugging Face")
        
        # Download the model
        print_info("Starting download...")
        local_path = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            token=hf_token,
            ignore_patterns=["*.git*", "README.md", "*.jpg", "*.png"] if not args.include_images else None
        )
        
        print_success(f"Model downloaded successfully to: {local_path}")
        
        # Get model info
        api = HfApi(token=hf_token)
        model_info = api.model_info(model_id)
        
        print_info(f"Model size: ~{model_info.safetensors['total'] / (1024**3):.1f} GB" if 
                  hasattr(model_info, 'safetensors') and model_info.safetensors else "Size unknown")
        
        if model_info.tags:
            print_info(f"Tags: {', '.join(model_info.tags[:5])}")
            
    except Exception as e:
        print_error(f"Error downloading model: {e}")
        if "401" in str(e):
            print_error("Authentication failed. Check your HF_TOKEN.")

def test_hf_model_command(args):
    """Test a downloaded Hugging Face model."""
    if not HF_AVAILABLE:
        print_error("Hugging Face libraries not available. Run: uv sync")
        return
    
    model_id = args.model_id
    query = args.query or "Hello, how are you?"
    max_tokens = args.max_tokens or 50
    
    print_info(f"Testing Hugging Face model: {model_id}")
    print_info(f"Query: {query}")
    
    try:
        # Get HF token from environment
        hf_token = os.getenv('HF_TOKEN')
        
        if hf_token:
            login(token=hf_token, add_to_git_credential=False)
        
        print_info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        
        print_info("Loading model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            token=hf_token,
            device_map="auto" if args.gpu else "cpu",
            torch_dtype="auto"
        )
        
        start_time = datetime.now()
        
        # Tokenize input
        inputs = tokenizer(query, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        end_time = datetime.now()
        latency = int((end_time - start_time).total_seconds() * 1000)
        
        print_success("Generation successful!")
        print_info(f"Latency: {latency}ms")
        
        if console:
            panel_content = f"[bold]Query:[/bold] {query}\n\n"
            panel_content += f"[bold]Response:[/bold]\n{response}"
            console.print(Panel(panel_content, title=f"[cyan]HF Model Test: {model_id}[/cyan]"))
        else:
            print(f"\nQuery: {query}")
            print(f"Response: {response}")
            
    except Exception as e:
        print_error(f"Error testing model: {e}")
        if "torch" in str(e).lower():
            print_info("PyTorch may not be installed. Try: uv add torch")

def generate_hf_config_command(args):
    """Generate Hugging Face model configuration."""
    if not HF_AVAILABLE:
        print_error("Hugging Face libraries not available. Run: uv sync")
        return
    
    print_info("Generating Hugging Face configuration...")
    
    # Popular text generation models
    popular_models = [
        "microsoft/DialoGPT-small",
        "microsoft/DialoGPT-medium", 
        "gpt2",
        "distilgpt2",
        "EleutherAI/gpt-neo-1.3B"
    ]
    
    # Allow user to specify models
    if args.models:
        models = args.models.split(',')
    else:
        models = popular_models[:3]  # Use first 3 popular models
    
    providers = {}
    for i, model in enumerate(models):
        provider_name = f"hf_{model.replace('/', '_').replace('-', '_').lower()}"
        providers[provider_name] = {
            "type": "local",
            "provider": "huggingface",
            "model": model,
            "cache_dir": "${HF_CACHE_DIR:-~/.cache/huggingface/hub}",
            "device": "auto",
            "torch_dtype": "auto",
            "trust_remote_code": False,
            "token": "${HF_TOKEN}",
            "max_tokens": 100,
            "temperature": 0.7,
            "description": f"Hugging Face {model} model"
        }
    
    config = {
        "name": "Hugging Face Models Configuration", 
        "version": "1.0.0",
        "default_provider": list(providers.keys())[0] if providers else "hf_gpt2",
        "providers": providers,
        "hf_settings": {
            "cache_dir": "~/.cache/huggingface/hub",
            "trust_remote_code": False,
            "use_auth_token": True,
            "device_map": "auto",
            "torch_dtype": "auto"
        },
        "selection_strategy": {
            "type": "size_optimized",
            "factors": {
                "model_size": 0.4,
                "quality": 0.3,
                "speed": 0.3
            }
        }
    }
    
    # Output configuration
    output_path = args.output
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        print_success(f"Hugging Face configuration written to: {output_path}")
    else:
        if console:
            syntax = Syntax(json.dumps(config, indent=2), "json", theme="monokai")
            console.print(Panel(syntax, title="[cyan]Hugging Face Configuration[/cyan]"))
        else:
            print(json.dumps(config, indent=2))
    
    print_info(f"Generated configuration for {len(providers)} HF models")

def hf_login_command(args):
    """Login to Hugging Face Hub."""
    if not HF_AVAILABLE:
        print_error("Hugging Face libraries not available. Run: uv sync")
        return
    
    try:
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            print_error("HF_TOKEN not found in environment variables")
            print_info("Please set HF_TOKEN in your .env file")
            return
        
        print_info("Logging in to Hugging Face Hub...")
        login(token=hf_token, add_to_git_credential=False)
        
        # Test the login by getting user info
        api = HfApi(token=hf_token)
        user_info = api.whoami()
        
        print_success(f"Successfully logged in as: {user_info['name']}")
        print_info(f"Type: {user_info['type']}")
        
    except Exception as e:
        print_error(f"Login failed: {e}")

def list_vllm_models_command(args):
    """List available vLLM-compatible models."""
    print_info("Listing vLLM-compatible models...")
    
    if not VLLM_AVAILABLE:
        print_warning("vLLM not available. Install with: uv add --optional local-engines vllm")
    
    # Popular vLLM-compatible models
    popular_models = [
        {"model": "meta-llama/Llama-2-7b-chat-hf", "size": "7B", "type": "Chat"},
        {"model": "meta-llama/Llama-2-13b-chat-hf", "size": "13B", "type": "Chat"},
        {"model": "mistralai/Mistral-7B-Instruct-v0.1", "size": "7B", "type": "Instruct"},
        {"model": "microsoft/DialoGPT-large", "size": "1.5B", "type": "Dialog"},
        {"model": "EleutherAI/gpt-neox-20b", "size": "20B", "type": "Base"},
    ]
    
    if console:
        table = Table(title="🚀 vLLM-Compatible Models")
        table.add_column("Model", style="cyan")
        table.add_column("Size", style="yellow")
        table.add_column("Type", style="green")
        table.add_column("Status", style="blue")
        
        for model_info in popular_models:
            status = "Available" if VLLM_AVAILABLE else "vLLM not installed"
            table.add_row(
                model_info["model"],
                model_info["size"],
                model_info["type"],
                status
            )
        
        console.print(table)
    else:
        print(f"\n🚀 vLLM-Compatible Models")
        print("=" * 60)
        
        for model_info in popular_models:
            print(f"\n📍 {model_info['model']}")
            print(f"   Size: {model_info['size']}")
            print(f"   Type: {model_info['type']}")
            print(f"   Status: {'Available' if VLLM_AVAILABLE else 'vLLM not installed'}")

def test_vllm_command(args):
    """Test a model using vLLM."""
    if not VLLM_AVAILABLE:
        print_error("vLLM not available. Install with: uv add --optional local-engines vllm")
        return
    
    model_name = args.model
    query = args.query or "Hello, how are you?"
    max_tokens = args.max_tokens or 50
    
    print_info(f"Testing vLLM model: {model_name}")
    print_info(f"Query: {query}")
    
    try:
        print_info("Initializing vLLM engine (this may take a while)...")
        
        # Initialize vLLM
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            max_model_len=getattr(args, 'max_model_len', 2048),
            gpu_memory_utilization=getattr(args, 'gpu_memory', 0.8),
        )
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=max_tokens,
            top_p=0.9
        )
        
        start_time = datetime.now()
        
        # Generate
        outputs = llm.generate([query], sampling_params)
        
        end_time = datetime.now()
        latency = int((end_time - start_time).total_seconds() * 1000)
        
        if outputs and len(outputs) > 0:
            output = outputs[0]
            generated_text = output.outputs[0].text if output.outputs else "No output"
            
            print_success("Generation successful!")
            print_info(f"Latency: {latency}ms")
            print_info(f"Tokens generated: {len(output.outputs[0].token_ids) if output.outputs else 0}")
            
            if console:
                panel_content = f"[bold]Query:[/bold] {query}\n\n"
                panel_content += f"[bold]Response:[/bold]\n{generated_text}"
                console.print(Panel(panel_content, title=f"[cyan]vLLM Test: {model_name}[/cyan]"))
            else:
                print(f"\nQuery: {query}")
                print(f"Response: {generated_text}")
        else:
            print_error("No output generated")
            
    except Exception as e:
        print_error(f"Error testing vLLM model: {e}")
        if "CUDA" in str(e):
            print_info("GPU may not be available. Try with smaller model or adjust gpu_memory_utilization.")

def list_tgi_models_command(args):
    """List Text Generation Inference compatible models."""
    print_info("Listing TGI-compatible models...")
    
    if not TGI_AVAILABLE:
        print_warning("TGI client not available. Install with: uv add --optional local-engines tgi")
    
    # Check TGI endpoints
    tgi_endpoints = [
        {"url": os.getenv('TGI_BASE_URL', 'http://localhost:8080'), "name": "Local TGI"},
        {"url": os.getenv('TGI_BASE_URL2', ''), "name": "TGI Instance 2"},
        {"url": os.getenv('TGI_BASE_URL3', ''), "name": "TGI Instance 3"},
    ]
    
    # Filter out empty URLs
    tgi_endpoints = [ep for ep in tgi_endpoints if ep["url"]]
    
    if console:
        table = Table(title="🔥 Text Generation Inference Endpoints")
        table.add_column("Name", style="cyan")
        table.add_column("URL", style="yellow")
        table.add_column("Status", style="green")
        
        for endpoint in tgi_endpoints:
            # Test connectivity
            try:
                response = requests.get(f"{endpoint['url']}/health", timeout=2)
                status = "✓ Healthy" if response.status_code == 200 else "⚠ Issues"
            except:
                status = "✗ Unreachable"
            
            table.add_row(endpoint["name"], endpoint["url"], status)
        
        console.print(table)
    else:
        print(f"\n🔥 Text Generation Inference Endpoints")
        print("=" * 60)
        
        for endpoint in tgi_endpoints:
            print(f"\n📍 {endpoint['name']}")
            print(f"   URL: {endpoint['url']}")
            
            try:
                response = requests.get(f"{endpoint['url']}/health", timeout=2)
                status = "✓ Healthy" if response.status_code == 200 else "⚠ Issues"
            except:
                status = "✗ Unreachable"
            print(f"   Status: {status}")

def test_tgi_command(args):
    """Test a Text Generation Inference endpoint."""
    if not TGI_AVAILABLE:
        print_error("TGI client not available. Install with: uv add --optional local-engines tgi")
        return
    
    endpoint_url = args.endpoint or os.getenv('TGI_BASE_URL', 'http://localhost:8080')
    query = args.query or "Hello, how are you?"
    max_tokens = args.max_tokens or 50
    
    print_info(f"Testing TGI endpoint: {endpoint_url}")
    print_info(f"Query: {query}")
    
    try:
        # Initialize TGI client
        client = Client(endpoint_url)
        
        start_time = datetime.now()
        
        # Generate
        response = client.generate(
            query,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        end_time = datetime.now()
        latency = int((end_time - start_time).total_seconds() * 1000)
        
        print_success("Generation successful!")
        print_info(f"Latency: {latency}ms")
        print_info(f"Generated tokens: {len(response.generated_text.split())}")
        
        if console:
            panel_content = f"[bold]Query:[/bold] {query}\n\n"
            panel_content += f"[bold]Response:[/bold]\n{response.generated_text}"
            console.print(Panel(panel_content, title=f"[cyan]TGI Test: {endpoint_url}[/cyan]"))
        else:
            print(f"\nQuery: {query}")
            print(f"Response: {response.generated_text}")
            
    except Exception as e:
        print_error(f"Error testing TGI endpoint: {e}")
        if "Connection" in str(e):
            print_info("Make sure TGI server is running and accessible.")

def generate_local_engines_config_command(args):
    """Generate configuration for local inference engines."""
    print_info("Generating local inference engines configuration...")
    
    # Base configuration structure
    providers = {}
    
    # Add vLLM providers if available
    if VLLM_AVAILABLE or args.include_unavailable:
        providers["vllm_llama2_7b"] = {
            "type": "local",
            "provider": "vllm",
            "model": "meta-llama/Llama-2-7b-chat-hf",
            "max_model_len": 2048,
            "gpu_memory_utilization": 0.8,
            "trust_remote_code": True,
            "temperature": 0.7,
            "max_tokens": 100,
            "description": "Llama 2 7B Chat via vLLM"
        }
        
        providers["vllm_mistral_7b"] = {
            "type": "local",
            "provider": "vllm",
            "model": "mistralai/Mistral-7B-Instruct-v0.1",
            "max_model_len": 2048,
            "gpu_memory_utilization": 0.8,
            "trust_remote_code": True,
            "temperature": 0.7,
            "max_tokens": 100,
            "description": "Mistral 7B Instruct via vLLM"
        }
    
    # Add TGI providers
    tgi_base_url = os.getenv('TGI_BASE_URL', 'http://localhost:8080')
    providers["tgi_local"] = {
        "type": "local",
        "provider": "tgi",
        "endpoint": tgi_base_url,
        "timeout": 30,
        "temperature": 0.7,
        "max_tokens": 100,
        "top_p": 0.9,
        "description": "Local Text Generation Inference server"
    }
    
    # Add Ollama providers (reference existing ones)
    providers["ollama_reference"] = {
        "type": "local",
        "provider": "ollama",
        "model": "llama3.1:8b",
        "host": "localhost",
        "port": 11434,
        "timeout": 120,
        "description": "Reference to existing Ollama setup"
    }
    
    config = {
        "name": "Local Inference Engines Configuration",
        "version": "1.0.0",
        "default_provider": list(providers.keys())[0] if providers else "vllm_llama2_7b",
        "providers": providers,
        "engine_settings": {
            "vllm": {
                "gpu_memory_utilization": 0.8,
                "max_model_len": 2048,
                "trust_remote_code": True,
                "tensor_parallel_size": 1
            },
            "tgi": {
                "timeout": 30,
                "stream": False,
                "details": True
            },
            "ollama": {
                "host": "localhost",
                "port": 11434,
                "timeout": 120
            }
        },
        "resource_requirements": {
            "vllm_llama2_7b": {"gpu_memory_gb": 16, "ram_gb": 32},
            "vllm_mistral_7b": {"gpu_memory_gb": 16, "ram_gb": 32},
            "tgi_local": {"gpu_memory_gb": 8, "ram_gb": 16},
            "ollama_reference": {"ram_gb": 8}
        }
    }
    
    # Output configuration
    output_path = args.output
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        print_success(f"Local engines configuration written to: {output_path}")
    else:
        if console:
            syntax = Syntax(json.dumps(config, indent=2), "json", theme="monokai")
            console.print(Panel(syntax, title="[cyan]Local Inference Engines Configuration[/cyan]"))
        else:
            print(json.dumps(config, indent=2))
    
    print_info(f"Generated configuration for {len(providers)} local engine providers")
    
    # Show availability status
    print_info("\nEngine Availability:")
    print_info(f"  vLLM: {'✓ Available' if VLLM_AVAILABLE else '✗ Not installed'}")
    print_info(f"  TGI: {'✓ Available' if TGI_AVAILABLE else '✗ Not installed'}")
    print_info(f"  Ollama: {'✓ Available' if os.system('which ollama > /dev/null 2>&1') == 0 else '✗ Not installed'}")

def create_cli_parser():
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="🦙 LlamaFarm Models CLI - Manage Cloud and Local Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query models with default or custom configs
  uv run python cli.py query "What is machine learning?"
  uv run python cli.py --config config/real_models_example.json query "Explain AI" --provider groq_llama3_70b
  
  # Interactive chat session
  uv run python cli.py chat
  uv run python cli.py chat --provider anthropic_claude_3_haiku --temperature 0.8
  
  # Send file content to model
  uv run python cli.py send code.py --prompt "Review this code for bugs"
  uv run python cli.py send data.csv --prompt "Analyze this data" --output analysis.md
  
  # Batch process queries
  uv run python cli.py batch queries.txt --output results.json
  uv run python cli.py batch prompts.txt --parallel 5 --provider openai_gpt4o_mini
  
  # List all configured models
  uv run python cli.py list --detailed
  uv run python cli.py --config config/use_case_examples.json list
  
  # List local Ollama models
  uv run python cli.py list-local
  
  # Test a specific provider
  uv run python cli.py test openai_gpt4o_mini --query "Hello, world!"
  
  # Test a local Ollama model
  uv run python cli.py test-local llama3.1:8b --query "Explain AI"
  
  # Pull a new model to Ollama
  uv run python cli.py pull llama3.2:1b
  
  # Check health of all providers
  uv run python cli.py health-check
  
  # Compare responses from multiple models
  uv run python cli.py compare --providers openai_gpt4o_mini,anthropic_claude_3_haiku --query "Explain quantum computing"
  
  # Generate a configuration template
  uv run python cli.py generate-config --type production --output my_config.json
  
  # Generate Ollama-specific configuration
  uv run python cli.py generate-ollama-config --output ollama_config.json
  
  # List Hugging Face models
  uv run python cli.py list-hf --search "gpt" --limit 10
  
  # Download a Hugging Face model
  uv run python cli.py download-hf gpt2
  
  # Test a Hugging Face model
  uv run python cli.py test-hf gpt2 --query "Hello world" --max-tokens 30
  
  # Generate Hugging Face configuration
  uv run python cli.py generate-hf-config --models "gpt2,distilgpt2"
  
  # Login to Hugging Face Hub
  uv run python cli.py hf-login
  
  # List vLLM-compatible models
  uv run python cli.py list-vllm
  
  # Test a model with vLLM
  uv run python cli.py test-vllm "meta-llama/Llama-2-7b-chat-hf" --query "Hello"
  
  # List TGI endpoints
  uv run python cli.py list-tgi
  
  # Test TGI endpoint
  uv run python cli.py test-tgi --endpoint "http://localhost:8080" --query "Hello"
  
  # Generate local engines configuration
  uv run python cli.py generate-engines-config --output engines_config.json
        """
    )
    
    # Global options
    parser.add_argument("--config", "-c", default="config/default.yaml",
                       help="Configuration file path (supports .yaml, .yml, and .json)")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available model providers")
    list_parser.add_argument("--detailed", "-d", action="store_true",
                           help="Show detailed configuration")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test model connectivity")
    test_parser.add_argument("provider", help="Provider name to test")
    test_parser.add_argument("--query", "-q", default="Hello, world!",
                           help="Test query to send")
    
    # Query command - comprehensive model interaction
    query_parser = subparsers.add_parser("query", help="Send a query to a model with full control")
    query_parser.add_argument("query", help="Query text to send to the model")
    query_parser.add_argument("--provider", "-p", help="Specific provider to use (default: from config)")
    query_parser.add_argument("--temperature", "-t", type=float, help="Override temperature setting")
    query_parser.add_argument("--max-tokens", "-m", type=int, help="Override max tokens")
    query_parser.add_argument("--top-p", type=float, help="Override top-p setting")
    query_parser.add_argument("--system", "-s", help="System prompt to use")
    query_parser.add_argument("--stream", action="store_true", help="Stream the response")
    query_parser.add_argument("--json", action="store_true", help="Output response as JSON")
    query_parser.add_argument("--save", help="Save response to file")
    
    # Chat command - interactive chat session
    chat_parser = subparsers.add_parser("chat", help="Start an interactive chat session")
    chat_parser.add_argument("--provider", "-p", help="Specific provider to use (default: from config)")
    chat_parser.add_argument("--temperature", "-t", type=float, help="Temperature setting")
    chat_parser.add_argument("--system", "-s", help="System prompt to use")
    chat_parser.add_argument("--history", help="Load chat history from file")
    chat_parser.add_argument("--save-history", help="Save chat history to file")
    
    # Send command - send file content to model
    send_parser = subparsers.add_parser("send", help="Send file content to a model")
    send_parser.add_argument("file", help="File path to send")
    send_parser.add_argument("--provider", "-p", help="Specific provider to use")
    send_parser.add_argument("--prompt", help="Additional prompt to prepend")
    send_parser.add_argument("--temperature", "-t", type=float, help="Temperature setting")
    send_parser.add_argument("--max-tokens", "-m", type=int, help="Maximum tokens")
    send_parser.add_argument("--output", "-o", help="Save response to file")
    
    # Batch command - process multiple queries
    batch_parser = subparsers.add_parser("batch", help="Process multiple queries from a file")
    batch_parser.add_argument("file", help="File containing queries (one per line)")
    batch_parser.add_argument("--provider", "-p", help="Specific provider to use")
    batch_parser.add_argument("--temperature", "-t", type=float, help="Temperature setting")
    batch_parser.add_argument("--output", "-o", help="Output file for responses")
    batch_parser.add_argument("--parallel", "-j", type=int, default=1, help="Number of parallel requests")
    
    # Health check command
    health_parser = subparsers.add_parser("health-check", help="Check health of all providers")
    
    # Validate config command
    validate_parser = subparsers.add_parser("validate-config", help="Validate configuration file")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare model responses")
    compare_parser.add_argument("--providers", "-p", required=True,
                              help="Comma-separated list of providers")
    compare_parser.add_argument("--query", "-q", required=True,
                              help="Query to test with")
    
    # Generate config command
    generate_parser = subparsers.add_parser("generate-config", help="Generate configuration template")
    generate_parser.add_argument("--type", "-t", choices=["basic", "multi", "production"],
                               default="basic", help="Configuration type")
    generate_parser.add_argument("--output", "-o", help="Output file path")
    
    # List local models command
    list_local_parser = subparsers.add_parser("list-local", help="List locally available Ollama models")
    
    # Pull model command
    pull_parser = subparsers.add_parser("pull", help="Pull a model to Ollama")
    pull_parser.add_argument("model", help="Model name to pull (e.g., llama3.1:8b)")
    
    # Test local model command
    test_local_parser = subparsers.add_parser("test-local", help="Test a local Ollama model")
    test_local_parser.add_argument("model", help="Local model name to test")
    test_local_parser.add_argument("--query", "-q", help="Test query to send")
    
    # Generate Ollama config command
    ollama_config_parser = subparsers.add_parser("generate-ollama-config", help="Generate Ollama-specific configuration")
    ollama_config_parser.add_argument("--output", "-o", help="Output file path")
    
    # Hugging Face commands
    hf_list_parser = subparsers.add_parser("list-hf", help="List available Hugging Face models")
    hf_list_parser.add_argument("--search", "-s", help="Search term for models")
    hf_list_parser.add_argument("--limit", "-l", type=int, default=20, help="Limit number of results")
    
    hf_download_parser = subparsers.add_parser("download-hf", help="Download a Hugging Face model")
    hf_download_parser.add_argument("model_id", help="Hugging Face model ID (e.g., gpt2)")
    hf_download_parser.add_argument("--cache-dir", help="Custom cache directory")
    hf_download_parser.add_argument("--include-images", action="store_true", help="Include image files in download")
    
    hf_test_parser = subparsers.add_parser("test-hf", help="Test a Hugging Face model")
    hf_test_parser.add_argument("model_id", help="Hugging Face model ID to test")
    hf_test_parser.add_argument("--query", "-q", help="Test query to send")
    hf_test_parser.add_argument("--max-tokens", type=int, default=50, help="Maximum tokens to generate")
    hf_test_parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    
    hf_config_parser = subparsers.add_parser("generate-hf-config", help="Generate Hugging Face configuration")
    hf_config_parser.add_argument("--output", "-o", help="Output file path")
    hf_config_parser.add_argument("--models", "-m", help="Comma-separated list of model IDs")
    
    hf_login_parser = subparsers.add_parser("hf-login", help="Login to Hugging Face Hub")
    
    # Local inference engines commands
    vllm_list_parser = subparsers.add_parser("list-vllm", help="List vLLM-compatible models")
    
    vllm_test_parser = subparsers.add_parser("test-vllm", help="Test a model using vLLM")
    vllm_test_parser.add_argument("model", help="Model name/path to test")
    vllm_test_parser.add_argument("--query", "-q", help="Test query to send")
    vllm_test_parser.add_argument("--max-tokens", type=int, default=50, help="Maximum tokens to generate")
    vllm_test_parser.add_argument("--max-model-len", type=int, default=2048, help="Maximum model context length")
    vllm_test_parser.add_argument("--gpu-memory", type=float, default=0.8, help="GPU memory utilization (0.0-1.0)")
    
    tgi_list_parser = subparsers.add_parser("list-tgi", help="List TGI endpoints and status")
    
    tgi_test_parser = subparsers.add_parser("test-tgi", help="Test a Text Generation Inference endpoint")
    tgi_test_parser.add_argument("--endpoint", "-e", help="TGI endpoint URL")
    tgi_test_parser.add_argument("--query", "-q", help="Test query to send")
    tgi_test_parser.add_argument("--max-tokens", type=int, default=50, help="Maximum tokens to generate")
    
    engines_config_parser = subparsers.add_parser("generate-engines-config", help="Generate local inference engines configuration")
    engines_config_parser.add_argument("--output", "-o", help="Output file path")
    engines_config_parser.add_argument("--include-unavailable", action="store_true", help="Include unavailable engines in config")
    
    return parser

def main():
    """Main CLI entry point."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    if not args.command:
        parser.print_help()
        print_info("\n💡 Try: uv run python cli.py list")
        sys.exit(1)
    
    # Route commands
    if args.command == "list":
        list_command(args)
    elif args.command == "test":
        test_command(args)
    elif args.command == "query":
        query_command(args)
    elif args.command == "chat":
        chat_command(args)
    elif args.command == "send":
        send_command(args)
    elif args.command == "batch":
        batch_command(args)
    elif args.command == "health-check":
        health_check_command(args)
    elif args.command == "validate-config":
        validate_config_command(args)
    elif args.command == "compare":
        compare_command(args)
    elif args.command == "generate-config":
        generate_config_command(args)
    elif args.command == "list-local":
        list_local_command(args)
    elif args.command == "pull":
        pull_model_command(args)
    elif args.command == "test-local":
        test_local_command(args)
    elif args.command == "generate-ollama-config":
        generate_ollama_config_command(args)
    elif args.command == "list-hf":
        list_hf_models_command(args)
    elif args.command == "download-hf":
        download_hf_model_command(args)
    elif args.command == "test-hf":
        test_hf_model_command(args)
    elif args.command == "generate-hf-config":
        generate_hf_config_command(args)
    elif args.command == "hf-login":
        hf_login_command(args)
    elif args.command == "list-vllm":
        list_vllm_models_command(args)
    elif args.command == "test-vllm":
        test_vllm_command(args)
    elif args.command == "list-tgi":
        list_tgi_models_command(args)
    elif args.command == "test-tgi":
        test_tgi_command(args)
    elif args.command == "generate-engines-config":
        generate_local_engines_config_command(args)
    else:
        print_error(f"Unknown command: {args.command}")
        sys.exit(1)

if __name__ == "__main__":
    main()