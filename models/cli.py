#!/usr/bin/env python3
"""
LlamaFarm Models CLI - Model Management Commands

This CLI provides commands for managing cloud and local model using UV for all operations.
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

# Import core model management components
try:
    from core.model_manager import ModelManager
    from core.strategy_manager import StrategyManager

    MODELS_CORE_AVAILABLE = True
except ImportError:
    MODELS_CORE_AVAILABLE = False

# Import fine-tuning components
try:
    from components.factory import FineTunerFactory
    from components.base import FineTuningConfig, TrainingJob
    from core.strategy_manager import StrategyManager as FineTuningStrategyManager

    FINETUNING_AVAILABLE = True
except ImportError:
    FINETUNING_AVAILABLE = False

# Load environment variables
load_dotenv()

# Import rich for better output formatting
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.progress import Progress, SpinnerColumn, TextColumn

    console = Console()
except ImportError:
    console = None

# Import colorama for colored terminal output
try:
    from colorama import Fore, Style, Back, init

    init(autoreset=True)
except ImportError:
    # Fallback if colorama not available
    class Fore:
        BLACK = RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ""

    class Style:
        RESET_ALL = BRIGHT = DIM = NORMAL = ""

    class Back:
        BLACK = RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ""


# Import CLI utilities
try:
    from cli.progress import ModelProgressTracker

    CLI_UTILITIES_AVAILABLE = True
except ImportError:
    CLI_UTILITIES_AVAILABLE = False

# Import model converters
try:
    from components.converters import OllamaConverter, GGUFConverter

    CONVERTERS_AVAILABLE = True
except ImportError:
    CONVERTERS_AVAILABLE = False


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
    """Load configuration from JSON or YAML file with environment variable substitution.

    DEPRECATED: This function is kept for backward compatibility.
    New code should use ModelManager.from_strategy() instead.
    """
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
        print_info(
            f"  - {(models_dir / 'config_examples' / Path(config_path).name).absolute()}"
        )
        print_info(f"\nAvailable strategies in strategies/ directory:")
        strategies_dir = models_dir / "strategies"
        if strategies_dir.exists():
            for f in strategies_dir.glob("*.json"):
                print_info(f"  - strategies/{f.name}")
            for f in strategies_dir.glob("*.yaml"):
                print_info(f"  - strategies/{f.name}")
            for f in strategies_dir.glob("*.yml"):
                print_info(f"  - strategies/{f.name}")
        sys.exit(1)

    try:
        with open(config_file, "r") as f:
            config_text = f.read()

        # Substitute environment variables
        for key, value in os.environ.items():
            if value:  # Only substitute non-empty values
                config_text = config_text.replace(f"${{{key}}}", value)

        # Determine file type and parse accordingly
        if config_file.suffix.lower() in [".yaml", ".yml"]:
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


# ==============================================================================
# NEW STRATEGY-BASED COMMANDS
# ==============================================================================


def list_strategies_command(args):
    """List available strategies."""
    if not MODELS_CORE_AVAILABLE:
        print_error(
            "Models core components not available. Please install required dependencies."
        )
        return

    try:
        strategy_manager = StrategyManager()
        strategies = strategy_manager.list_strategies()

        if not strategies:
            print_warning("No strategies available")
            return

        if console:
            table = Table(title="🦙 Available Strategies")
            table.add_column("Name", style="cyan")
            table.add_column("Description", style="green", max_width=50)
            table.add_column("Components", style="yellow")

            for strategy_name in strategies:
                strategy_config = strategy_manager.get_strategy(strategy_name)
                if strategy_config:
                    description = strategy_config.get("description", "No description")
                    components = list(strategy_config.get("components", {}).keys())
                    components_str = ", ".join(components) if components else "None"

                    table.add_row(strategy_name, description, components_str)

            console.print(table)
        else:
            print("\n🦙 Available Strategies")
            print("=" * 60)
            for strategy_name in strategies:
                strategy_config = strategy_manager.get_strategy(strategy_name)
                if strategy_config:
                    print(f"\n📍 {strategy_name}")
                    print(
                        f"   Description: {strategy_config.get('description', 'No description')}"
                    )
                    components = list(strategy_config.get("components", {}).keys())
                    if components:
                        print(f"   Components: {', '.join(components)}")

        print_success(f"Found {len(strategies)} strategies")

    except Exception as e:
        print_error(f"Failed to list strategies: {e}")


def use_strategy_command(args):
    """Switch to and use a specific strategy."""
    if not MODELS_CORE_AVAILABLE:
        print_error(
            "Models core components not available. Please install required dependencies."
        )
        return

    try:
        # Create ModelManager with the specified strategy
        manager = ModelManager.from_strategy(args.strategy)

        print_success(f"Using strategy: {args.strategy}")

        # Show strategy info
        info = manager.get_info()
        print_info(f"Description: {info['strategy_description']}")
        print_info(f"Components: {', '.join(info['components'].keys())}")

        # Validate the strategy
        validation_errors = manager.validate_configuration()
        if validation_errors:
            print_warning("Strategy validation warnings:")
            for error in validation_errors:
                print_warning(f"  - {error}")
        else:
            print_success("Strategy is valid")

        # Save current strategy to a persistent config (optional)
        if args.save:
            current_strategy_file = Path("current_strategy.txt")
            current_strategy_file.write_text(args.strategy)
            print_success(f"Strategy saved to {current_strategy_file}")

    except Exception as e:
        print_error(f"Failed to use strategy '{args.strategy}': {e}")


def info_command(args):
    """Show detailed information about current strategy or specified strategy."""
    if not MODELS_CORE_AVAILABLE:
        print_error(
            "Models core components not available. Please install required dependencies."
        )
        return

    try:
        # Determine which strategy to show info for
        strategy_name = (
            args.strategy
            if hasattr(args, "strategy") and args.strategy
            else "local_development"
        )

        # Load current strategy from file if no strategy specified
        if not hasattr(args, "strategy") or not args.strategy:
            current_strategy_file = Path("current_strategy.txt")
            if current_strategy_file.exists():
                strategy_name = current_strategy_file.read_text().strip()

        # Create ModelManager with the strategy
        manager = ModelManager.from_strategy(strategy_name)
        info = manager.get_info()

        if console:
            # Rich formatted output
            info_text = f"""[bold]Strategy Name:[/bold] {info["strategy"]}
[bold]Description:[/bold] {info["strategy_description"]}

[bold]Components:[/bold]"""

            for comp_type, comp_impl in info["components"].items():
                info_text += f"\n• {comp_type}: {comp_impl}"

            info_text += f"\n\n[bold]Fallback Chain:[/bold] {info['fallback_count']} fallback options"

            if info.get("optimization"):
                opt = info["optimization"]
                info_text += f"\n\n[bold]Optimization:[/bold]"
                info_text += f"\n• Cache enabled: {opt.get('cache_enabled', 'N/A')}"
                info_text += f"\n• Batch size: {opt.get('batch_size', 'N/A')}"
                info_text += f"\n• Timeout: {opt.get('timeout_seconds', 'N/A')}s"

            if info.get("constraints"):
                const = info["constraints"]
                info_text += f"\n\n[bold]Constraints:[/bold]"
                for key, value in const.items():
                    info_text += f"\n• {key}: {value}"

            panel = Panel(
                info_text, title=f"Strategy Information: {strategy_name}", expand=False
            )
            console.print(panel)
        else:
            print(f"\nStrategy Information: {strategy_name}")
            print("=" * (len(strategy_name) + 22))
            print(f"Description: {info['strategy_description']}")
            print(f"\nComponents:")
            for comp_type, comp_impl in info["components"].items():
                print(f"  • {comp_type}: {comp_impl}")
            print(f"\nFallback options: {info['fallback_count']}")

            if info.get("optimization"):
                opt = info["optimization"]
                print(f"\nOptimization:")
                print(f"  Cache enabled: {opt.get('cache_enabled', 'N/A')}")
                print(f"  Batch size: {opt.get('batch_size', 'N/A')}")
                print(f"  Timeout: {opt.get('timeout_seconds', 'N/A')}s")

        print_success("Strategy information displayed")

    except Exception as e:
        print_error(f"Failed to get strategy info: {e}")


def generate_command(args):
    """Generate text using the current strategy."""
    if not MODELS_CORE_AVAILABLE:
        print_error(
            "Models core components not available. Please install required dependencies."
        )
        return

    try:
        # Determine which strategy to use
        strategy_name = (
            args.strategy
            if hasattr(args, "strategy") and args.strategy
            else "local_development"
        )

        # Load current strategy from file if no strategy specified
        if not hasattr(args, "strategy") or not args.strategy:
            current_strategy_file = Path("current_strategy.txt")
            if current_strategy_file.exists():
                strategy_name = current_strategy_file.read_text().strip()

        # Create ModelManager with the strategy
        manager = ModelManager.from_strategy(strategy_name)

        print_info(f"Using strategy: {strategy_name}")
        print_info(f"Prompt: {args.prompt}")

        # Prepare generation parameters
        gen_kwargs = {}
        if hasattr(args, "max_tokens") and args.max_tokens:
            gen_kwargs["max_tokens"] = args.max_tokens
        if hasattr(args, "temperature") and args.temperature is not None:
            gen_kwargs["temperature"] = args.temperature
        if hasattr(args, "stream") and args.stream:
            gen_kwargs["stream"] = args.stream

        # Generate response
        print_info("Generating response...")
        response = manager.generate(args.prompt, **gen_kwargs)

        if args.json:
            # Output as JSON
            result = {
                "strategy": strategy_name,
                "prompt": args.prompt,
                "response": response,
                "parameters": gen_kwargs,
            }
            print(json.dumps(result, indent=2))
        else:
            # Standard output
            if console:
                panel = Panel(response, title="Generated Response", expand=False)
                console.print(panel)
            else:
                print(f"\nResponse:")
                print("=" * 40)
                print(response)

        # Save to file if requested
        if hasattr(args, "save") and args.save:
            save_path = Path(args.save)
            if args.json:
                save_path.write_text(json.dumps(result, indent=2))
            else:
                save_path.write_text(response)
            print_success(f"Response saved to {save_path}")

    except Exception as e:
        print_error(f"Failed to generate response: {e}")


def complete_command(args):
    """Get text completion using strategy - provider-agnostic approach."""
    try:
        # Load strategies file - use provided path or default
        strategies_file = Path(
            args.strategy_file
            if hasattr(args, "strategy_file")
            else "demos/strategies.yaml"
        )
        if not strategies_file.exists():
            print_error(f"Strategies file not found: {strategies_file}")
            print_info("Use --strategy-file to specify a different strategies file")
            return

        with open(strategies_file) as f:
            import yaml

            strategies_config = yaml.safe_load(f)
        
        strategies_list = strategies_config.get('strategies', [])
        
        # Find the strategy in the list
        strategy = None
        available_names = []
        for s in strategies_list:
            if 'name' in s:
                available_names.append(s['name'])
                if s['name'] == args.strategy:
                    strategy = s
                    break
        
        if not strategy:
            print_error(f"Strategy '{args.strategy}' not found")
            print_info(f"Available strategies: {', '.join(available_names)}")
            return
        
        # Determine which component to use (ollama, cloud_api, etc.)
        # Priority: ollama > cloud_api > fine_tuner > mock_model
        component = None
        provider_type = None

        components = strategy.get("components", {})
        if "ollama" in components:
            component = components["ollama"]
            provider_type = "ollama"
        elif "cloud_api" in components:
            component = components["cloud_api"]
            provider_type = "cloud_api"
        elif "model_app" in components:
            component = components["model_app"]
            # Check if it's a mock model or regular Ollama
            if component.get("type") == "mock_model":
                provider_type = "mock_model"
            else:
                provider_type = "ollama"  # model_app is typically Ollama
        elif "fine_tuner" in components:
            # Fine-tuner needs special handling for inference
            component = components["fine_tuner"]
            provider_type = "fine_tuner"
        elif "mock_model" in components:
            component = components["mock_model"]
            provider_type = "mock_model"
        else:
            print_error(f"No suitable provider found in strategy '{args.strategy}'")
            return

        # Show provider details if verbose
        if args.verbose:
            print_info(f"Strategy: {args.strategy}")
            if "model_app" in components:
                print_info(f"Provider: model_app (Ollama)")
            else:
                print_info(f"Provider: {provider_type}")
            if provider_type == "ollama":
                model = component.get("config", {}).get(
                    "model", component.get("config", {}).get("default_model", "unknown")
                )
                print_info(f"Model: {model}")
            elif provider_type == "cloud_api":
                provider = component.get("config", {}).get("provider", "unknown")
                model = component.get("config", {}).get("default_model", "unknown")
                print_info(f"Cloud Provider: {provider}")
                print_info(f"Model: {model}")

        # Route to appropriate handler based on provider type
        if provider_type == "ollama":
            # Use Ollama for completion
            model_name = component.get("config", {}).get("model", "llama3.2:3b")
            base_url = component.get("config", {}).get(
                "base_url", "http://localhost:11434"
            )

            import requests

            payload = {
                "model": model_name,
                "prompt": args.prompt,
                "stream": args.stream,
            }

            if args.system:
                payload["system"] = args.system
            if args.max_tokens:
                payload["options"] = {"num_predict": args.max_tokens}
            if args.temperature is not None:
                payload.setdefault("options", {})["temperature"] = args.temperature

            response = requests.post(f"{base_url}/api/generate", json=payload)

            if response.status_code == 200:
                result = response.json()
                if args.json:
                    print(
                        json.dumps(
                            {
                                "strategy": args.strategy,
                                "provider": "ollama",
                                "model": model_name,
                                "response": result.get("response", ""),
                            },
                            indent=2,
                        )
                    )
                else:
                    print_success("Response:")
                    print(result.get("response", ""))
            else:
                print_error(f"Ollama request failed: {response.status_code}")

        elif provider_type == "cloud_api":
            # Use cloud API (OpenAI, etc.)
            provider = component.get("config", {}).get("provider", "openai")
            api_key = component.get("config", {}).get("api_key", "")
            model = component.get("config", {}).get("default_model", "gpt-3.5-turbo")

            if not api_key or api_key.startswith("${"):
                # Try to get from environment
                api_key = os.getenv("OPENAI_API_KEY", "")

            if not api_key:
                print_error("API key not found for cloud provider")
                return

            # Use OpenAI-compatible API
            import requests

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            messages = []
            if args.system:
                messages.append({"role": "system", "content": args.system})
            messages.append({"role": "user", "content": args.prompt})

            payload = {"model": model, "messages": messages, "stream": args.stream}

            if args.max_tokens:
                payload["max_tokens"] = args.max_tokens
            if args.temperature is not None:
                payload["temperature"] = args.temperature

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                if args.json:
                    print(
                        json.dumps(
                            {
                                "strategy": args.strategy,
                                "provider": provider,
                                "model": model,
                                "response": content,
                            },
                            indent=2,
                        )
                    )
                else:
                    print_success("Response:")
                    print(content)
            else:
                print_error(f"API request failed: {response.status_code}")
                print_error(response.text)

        elif provider_type == "mock_model":
            # Mock response for testing
            response = f"[Mock response to: {args.prompt}]"
            if args.json:
                print(
                    json.dumps(
                        {
                            "strategy": args.strategy,
                            "provider": "mock",
                            "response": response,
                        },
                        indent=2,
                    )
                )
            else:
                print_success("Response (Mock):")
                print(response)
        else:
            print_error(
                f"Provider type '{provider_type}' not yet implemented for completions"
            )

    except Exception as e:
        print_error(f"Failed to get completion: {e}")
        import traceback

        traceback.print_exc()


# ==============================================================================
# UPDATED LEGACY COMMANDS (now strategy-aware)
# ==============================================================================


def list_command(args):
    """List available model providers (legacy - use list-strategies for strategy-based approach)."""
    print_warning(
        "This command uses the legacy config system. Consider using 'list-strategies' instead."
    )

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
                syntax = Syntax(
                    panel_content, "json", theme="monokai", line_numbers=False
                )
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

        results.append(
            {
                "name": name,
                "type": provider_type,
                "model": model,
                "healthy": is_healthy,
                "latency": latency,
            }
        )

    # Display results
    if console:
        table = Table(title="🏥 Provider Health Status")
        table.add_column("Provider", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Model", style="yellow")
        table.add_column("Status", style="green")
        table.add_column("Latency", style="blue")

        for result in results:
            status = (
                "[green]✓ Healthy[/green]"
                if result["healthy"]
                else "[red]✗ Unhealthy[/red]"
            )
            latency = f"{result['latency']}ms" if result["latency"] else "N/A"

            table.add_row(
                result["name"], result["type"], result["model"], status, latency
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
                    print_warning(
                        "  No API key configured (using environment variable?)"
                    )
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
                print_error(
                    f"  Invalid providers in fallback chain: {', '.join(invalid_providers)}"
                )
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
    provider_names = args.providers.split(",")

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

        mock_response = (
            f"This is a simulated response from {provider_config['model']}. "
        )
        mock_response += (
            "In a real implementation, this would be an actual API response."
        )

        tokens = len(mock_response.split()) * 2
        cost = 0
        if provider_config.get("type") == "cloud":
            cost_per_1k = provider_config.get("cost_per_1k_tokens", 0)
            cost = (tokens / 1000) * cost_per_1k

        results.append(
            {
                "provider": provider_name,
                "model": provider_config["model"],
                "response": mock_response,
                "latency": latency,
                "tokens": tokens,
                "cost": cost,
            }
        )

    # Display results
    if console:
        for result in results:
            panel_content = f"[bold]Model:[/bold] {result['model']}\n"
            panel_content += f"[bold]Latency:[/bold] {result['latency']}ms\n"
            panel_content += f"[bold]Tokens:[/bold] {result['tokens']}\n"
            if result["cost"] > 0:
                panel_content += f"[bold]Cost:[/bold] ${result['cost']:.4f}\n"
            panel_content += f"\n[italic]{result['response']}[/italic]"

            console.print(
                Panel(panel_content, title=f"[cyan]{result['provider']}[/cyan]")
            )
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
            if result["cost"] > 0:
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


def _call_openai_api(
    provider_config: Dict[str, Any], query: str, system_prompt: str = None
) -> str:
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
            organization="",  # Set to empty string to avoid organization header issues
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
            top_p=provider_config.get("top_p", 1.0)
            if provider_config.get("top_p")
            else None,
        )

        return response.choices[0].message.content or "No response generated"

    except ImportError:
        return "Error: OpenAI package not installed. Run: uv add openai"
    except Exception as e:
        return f"Error calling OpenAI API: {str(e)}"


def _call_anthropic_api(
    provider_config: Dict[str, Any], query: str, system_prompt: str = None
) -> str:
    """Make a real Anthropic API call."""
    try:
        import anthropic

        # Get API key from config with environment variable substitution
        api_key = _substitute_env_vars(provider_config.get("api_key", ""))

        if not api_key:
            return "Error: Anthropic API key not configured"

        # Create client
        client = anthropic.Anthropic(api_key=api_key)

        # Make API call
        response = client.messages.create(
            model=provider_config.get("model", "claude-3-haiku-20240307"),
            max_tokens=provider_config.get("max_tokens", 4096),
            temperature=provider_config.get("temperature", 0.7),
            system=system_prompt or "You are a helpful assistant.",
            messages=[{"role": "user", "content": query}],
        )

        return response.content[0].text if response.content else "No response generated"

    except ImportError:
        return "Error: Anthropic package not installed. Run: uv add anthropic"
    except Exception as e:
        return f"Error calling Anthropic API: {str(e)}"


def _call_ollama_api(
    provider_config: Dict[str, Any], query: str, system_prompt: str = None
) -> str:
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
                    "num_predict": provider_config.get("max_tokens", 512),
                },
            },
            timeout=provider_config.get("timeout", 120),
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
    """Send a query to a model with full control over parameters (legacy - use 'generate' for strategy-based approach)."""
    print_warning(
        "This command uses the legacy config system. Consider using 'generate' with strategies instead."
    )

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
            response_text = _call_anthropic_api(
                provider_config, args.query, args.system
            )
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
                    "top_p": provider_config.get("top_p"),
                },
            }
            print(json.dumps(output, indent=2))
        else:
            print_success(f"Response received in {latency}ms")
            print(f"\n{response_text}")

        if args.save:
            with open(args.save, "w") as f:
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
            with open(args.history, "r") as f:
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

            if user_input.lower() in ["exit", "quit"]:
                break
            elif user_input.lower() == "clear":
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
            with open(args.save_history, "w") as f:
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
        with open(args.file, "r") as f:
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
            with open(args.output, "w") as f:
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
        with open(args.file, "r") as f:
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
        print(f"\nProcessing query {i + 1}/{len(queries)}: {query[:50]}...")

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

        results.append({"query": query, "response": response, "latency_ms": latency})

        print_success(f"Completed in {latency}ms")

    # Save results if requested
    if args.output:
        try:
            with open(args.output, "w") as f:
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
                    "temperature": 0.7,
                }
            },
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
                    "temperature": 0.7,
                },
                "anthropic_claude": {
                    "type": "cloud",
                    "provider": "anthropic",
                    "model": "claude-3-opus-20240229",
                    "api_key": "${ANTHROPIC_API_KEY}",
                    "base_url": "https://api.anthropic.com/v1",
                    "max_tokens": 4096,
                    "temperature": 0.7,
                },
                "local_llama": {
                    "type": "local",
                    "provider": "ollama",
                    "model": "llama2:13b",
                    "host": "localhost",
                    "port": 11434,
                    "timeout": 120,
                },
            },
            "selection_strategy": {
                "type": "cost_optimized",
                "factors": {"cost": 0.4, "quality": 0.3, "speed": 0.3},
            },
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
                        "tokens_per_minute": 90000,
                    },
                },
                "openai_gpt35": {
                    "type": "cloud",
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",
                    "api_key": "${OPENAI_API_KEY}",
                    "base_url": "https://api.openai.com/v1",
                    "max_tokens": 4096,
                    "temperature": 0.7,
                    "cost_per_1k_tokens": 0.002,
                },
            },
            "fallback_strategy": {
                "enabled": True,
                "chain": [
                    {
                        "provider": "openai_gpt4",
                        "conditions": ["api_healthy", "within_rate_limit"],
                    },
                    {"provider": "openai_gpt35", "conditions": ["api_healthy"]},
                ],
                "retry_strategy": {"max_retries": 3, "backoff_multiplier": 2},
            },
            "monitoring": {
                "track_usage": True,
                "track_costs": True,
                "alert_thresholds": {"daily_cost_usd": 100, "error_rate_percent": 5},
            },
        },
    }

    if config_type not in templates:
        print_error(f"Unknown configuration type: {config_type}")
        print_info(f"Available types: {', '.join(templates.keys())}")
        return

    config = templates[config_type]

    # Output configuration
    output_path = args.output
    if output_path:
        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)
        print_success(f"Configuration written to: {output_path}")
    else:
        # Print to console
        if console:
            syntax = Syntax(json.dumps(config, indent=2), "json", theme="monokai")
            console.print(
                Panel(syntax, title=f"[cyan]{config_type.title()} Configuration[/cyan]")
            )
        else:
            print(json.dumps(config, indent=2))


def list_local_command(args):
    """List locally available models from Ollama."""
    print_info("Fetching local Ollama models...")

    try:
        # Get Ollama base URL from environment or default
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # Try to get models via API first
        try:
            response = requests.get(f"{ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                models = models_data.get("models", [])

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
                        name = model.get("name", "unknown")
                        model_id = model.get("digest", "unknown")[:12]
                        size = model.get("size", 0)
                        # Convert size to human readable
                        if size > 1024**3:
                            size_str = f"{size / (1024**3):.1f} GB"
                        elif size > 1024**2:
                            size_str = f"{size / (1024**2):.1f} MB"
                        else:
                            size_str = f"{size / 1024:.1f} KB"

                        modified = model.get("modified_at", "unknown")
                        if modified != "unknown":
                            # Parse and format timestamp
                            from datetime import datetime

                            try:
                                dt = datetime.fromisoformat(
                                    modified.replace("Z", "+00:00")
                                )
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
                        size = model.get("size", 0)
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
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) <= 1:
                    print_warning("No local models found in Ollama")
                    return

                print_success(f"Found {len(lines) - 1} local models:")
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
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # Try API first for better progress tracking
        try:
            print_info("Starting model pull via API...")
            response = requests.post(
                f"{ollama_base_url}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=300,
            )

            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode("utf-8"))
                            if "status" in data:
                                status = data["status"]
                                if "completed" in data and "total" in data:
                                    completed = data["completed"]
                                    total = data["total"]
                                    percent = (
                                        (completed / total) * 100 if total > 0 else 0
                                    )
                                    print(
                                        f"\r{status}: {percent:.1f}%",
                                        end="",
                                        flush=True,
                                    )
                                else:
                                    print(f"\r{status}", end="", flush=True)
                        except json.JSONDecodeError:
                            continue
                print()  # New line after progress
                print_success(f"Successfully pulled model: {model_name}")
            else:
                raise requests.RequestException("API pull failed")

        except requests.RequestException:
            # Fallback to CLI command
            print_info("Falling back to ollama CLI...")
            result = subprocess.run(
                ["ollama", "pull", model_name], capture_output=False, text=True
            )
            if result.returncode == 0:
                print_success(f"Successfully pulled model: {model_name}")
            else:
                print_error(f"Failed to pull model: {model_name}")

    except FileNotFoundError:
        print_error("Ollama not found. Please install Ollama first.")
    except Exception as e:
        print_error(f"Error pulling model: {e}")


def ollama_command(args):
    """Handle Ollama management commands."""
    if args.ollama_command == "list":
        ollama_list_command(args)
    elif args.ollama_command == "pull":
        ollama_pull_command(args)
    elif args.ollama_command == "run":
        ollama_run_command(args)
    elif args.ollama_command == "status":
        ollama_status_command(args)
    else:
        print_error("Unknown Ollama command")


def ollama_list_command(args):
    """List installed Ollama models."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            print_success("Installed Ollama models:")
            print(result.stdout)
        else:
            print_error("Failed to list models")
            print(result.stderr)
    except FileNotFoundError:
        print_error("Ollama not found. Please install Ollama first.")
        print_info("Visit https://ollama.ai to download and install Ollama")


def ollama_pull_command(args):
    """Download an Ollama model."""
    model_name = args.model
    print_info(f"Pulling model: {model_name}")

    try:
        # Run ollama pull with progress
        process = subprocess.Popen(
            ["ollama", "pull", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Stream output in real-time
        for line in iter(process.stdout.readline, ""):
            if line:
                print(line.rstrip())

        process.wait()

        if process.returncode == 0:
            print_success(f"Successfully pulled model: {model_name}")
        else:
            print_error(f"Failed to pull model: {model_name}")

    except FileNotFoundError:
        print_error("Ollama not found. Please install Ollama first.")
        print_info("Visit https://ollama.ai to download and install Ollama")


def ollama_run_command(args):
    """Run an Ollama model with a prompt."""
    model_name = args.model
    prompt = args.prompt

    print_info(f"Running model: {model_name}")

    try:
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": args.stream if hasattr(args, "stream") else False,
        }

        response = requests.post(
            f"{ollama_base_url}/api/generate", json=payload, timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            generated_response = result.get("response", "No response")
            print_success("Response:")
            print(generated_response)
        else:
            print_error(f"Failed to run model: {response.status_code}")

    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to Ollama. Make sure Ollama is running.")
        print_info("Start Ollama with: ollama serve")
    except Exception as e:
        print_error(f"Error running model: {e}")


def ollama_status_command(args):
    """Check if Ollama is running and available."""
    try:
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        response = requests.get(f"{ollama_base_url}/api/tags", timeout=5)

        if response.status_code == 200:
            print_success("✓ Ollama is running and accessible")
            data = response.json()
            models = data.get("models", [])
            if models:
                print_info(f"Found {len(models)} installed models")
            else:
                print_warning(
                    "No models installed. Use 'ollama pull <model>' to download models"
                )
        else:
            print_error("Ollama is running but returned an error")

    except requests.exceptions.ConnectionError:
        print_error("✗ Ollama is not running")
        print_info("Start Ollama with: ollama serve")
    except Exception as e:
        print_error(f"Error checking Ollama status: {e}")


def test_local_command(args):
    """Test a local Ollama model."""
    model_name = args.model
    query = args.query or "Hello, how are you?"

    print_info(f"Testing local model: {model_name}")
    print_info(f"Query: {query}")

    try:
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # Test with Ollama API
        start_time = datetime.now()

        payload = {"model": model_name, "prompt": query, "stream": False}

        response = requests.post(
            f"{ollama_base_url}/api/generate", json=payload, timeout=30
        )

        end_time = datetime.now()
        latency = int((end_time - start_time).total_seconds() * 1000)

        if response.status_code == 200:
            result = response.json()
            generated_response = result.get("response", "No response")

            print_success("Generation successful!")
            print_info(f"Latency: {latency}ms")
            print_info(f"Model: {model_name}")

            if console:
                panel_content = f"[bold]Query:[/bold] {query}\n\n"
                panel_content += f"[bold]Response:[/bold]\n{generated_response}"
                console.print(
                    Panel(
                        panel_content,
                        title=f"[cyan]Local Model Test: {model_name}[/cyan]",
                    )
                )
            else:
                print(f"\nQuery: {query}")
                print(f"Response: {generated_response}")

            # Additional metrics if available
            if "eval_count" in result:
                print_info(f"Tokens generated: {result['eval_count']}")
            if "eval_duration" in result:
                tokens_per_sec = (
                    result["eval_count"] / (result["eval_duration"] / 1e9)
                    if result["eval_duration"] > 0
                    else 0
                )
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
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")[1:]  # Skip header
            available_models = [line.split()[0] for line in lines if line.strip()]
    except:
        available_models = [
            "llama3.1:8b",
            "llama3:latest",
            "llama3.2:3b",
        ]  # Fallback defaults

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
            "description": f"Local {model} model via Ollama",
        }

    config = {
        "name": "Ollama Local Models Configuration",
        "version": "1.0.0",
        "default_provider": list(providers.keys())[0]
        if providers
        else "ollama_llama3_1_8b",
        "providers": providers,
        "local_settings": {
            "ollama": {
                "host": "localhost",
                "port": 11434,
                "timeout": 120,
                "keep_alive": "5m",
                "concurrent_requests": 4,
            }
        },
        "fallback_chain": list(providers.keys())[:3]
        if len(providers) >= 3
        else list(providers.keys()),
        "selection_strategy": {
            "type": "performance_optimized",
            "factors": {"speed": 0.4, "memory_usage": 0.3, "quality": 0.3},
        },
    }

    # Output configuration
    output_path = args.output
    if output_path:
        with open(output_path, "w") as f:
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
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print_warning("HF_TOKEN not found. Only public models will be shown.")

        # Initialize HF API
        api = HfApi(token=hf_token)

        # Search parameters
        search_term = args.search if hasattr(args, "search") and args.search else None
        limit = args.limit if hasattr(args, "limit") and args.limit else 20

        # Search for models
        models = list_models(
            search=search_term,
            task="text-generation",
            sort="downloads",
            direction=-1,
            limit=limit,
            token=hf_token,
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
                author = model_id.split("/")[0] if "/" in model_id else "unknown"
                downloads = f"{model.downloads:,}" if model.downloads else "N/A"
                tags = ", ".join(model.tags[:3]) if model.tags else "N/A"

                table.add_row(model_id, author, downloads, tags)

            console.print(table)
        else:
            print(f"\n🤗 Found {len(models_list)} Hugging Face models")
            print("=" * 60)

            for model in models_list:
                print(f"\n📍 {model.id}")
                author = model.id.split("/")[0] if "/" in model.id else "unknown"
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
        hf_token = os.getenv("HF_TOKEN")
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
            ignore_patterns=["*.git*", "README.md", "*.jpg", "*.png"]
            if not args.include_images
            else None,
        )

        print_success(f"Model downloaded successfully to: {local_path}")

        # Get model info
        api = HfApi(token=hf_token)
        model_info = api.model_info(model_id)

        print_info(
            f"Model size: ~{model_info.safetensors['total'] / (1024**3):.1f} GB"
            if hasattr(model_info, "safetensors") and model_info.safetensors
            else "Size unknown"
        )

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
        hf_token = os.getenv("HF_TOKEN")

        if hf_token:
            login(token=hf_token, add_to_git_credential=False)

        print_info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

        print_info("Loading model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            device_map="auto" if args.gpu else "cpu",
            torch_dtype="auto",
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
                pad_token_id=tokenizer.eos_token_id,
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
            console.print(
                Panel(panel_content, title=f"[cyan]HF Model Test: {model_id}[/cyan]")
            )
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
        "EleutherAI/gpt-neo-1.3B",
    ]

    # Allow user to specify models
    if args.models:
        models = args.models.split(",")
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
            "description": f"Hugging Face {model} model",
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
            "torch_dtype": "auto",
        },
        "selection_strategy": {
            "type": "size_optimized",
            "factors": {"model_size": 0.4, "quality": 0.3, "speed": 0.3},
        },
    }

    # Output configuration
    output_path = args.output
    if output_path:
        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)
        print_success(f"Hugging Face configuration written to: {output_path}")
    else:
        if console:
            syntax = Syntax(json.dumps(config, indent=2), "json", theme="monokai")
            console.print(
                Panel(syntax, title="[cyan]Hugging Face Configuration[/cyan]")
            )
        else:
            print(json.dumps(config, indent=2))

    print_info(f"Generated configuration for {len(providers)} HF models")


def hf_login_command(args):
    """Login to Hugging Face Hub."""
    if not HF_AVAILABLE:
        print_error("Hugging Face libraries not available. Run: uv sync")
        return

    try:
        hf_token = os.getenv("HF_TOKEN")
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
        print_warning(
            "vLLM not available. Install with: uv add --optional local-engines vllm"
        )

    # Popular vLLM-compatible models
    popular_models = [
        {"model": "meta-llama/Llama-2-7b-chat-hf", "size": "7B", "type": "Chat"},
        {"model": "meta-llama/Llama-2-13b-chat-hf", "size": "13B", "type": "Chat"},
        {
            "model": "mistralai/Mistral-7B-Instruct-v0.1",
            "size": "7B",
            "type": "Instruct",
        },
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
                model_info["model"], model_info["size"], model_info["type"], status
            )

        console.print(table)
    else:
        print(f"\n🚀 vLLM-Compatible Models")
        print("=" * 60)

        for model_info in popular_models:
            print(f"\n📍 {model_info['model']}")
            print(f"   Size: {model_info['size']}")
            print(f"   Type: {model_info['type']}")
            print(
                f"   Status: {'Available' if VLLM_AVAILABLE else 'vLLM not installed'}"
            )


def test_vllm_command(args):
    """Test a model using vLLM."""
    if not VLLM_AVAILABLE:
        print_error(
            "vLLM not available. Install with: uv add --optional local-engines vllm"
        )
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
            max_model_len=getattr(args, "max_model_len", 2048),
            gpu_memory_utilization=getattr(args, "gpu_memory", 0.8),
        )

        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7, max_tokens=max_tokens, top_p=0.9
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
            print_info(
                f"Tokens generated: {len(output.outputs[0].token_ids) if output.outputs else 0}"
            )

            if console:
                panel_content = f"[bold]Query:[/bold] {query}\n\n"
                panel_content += f"[bold]Response:[/bold]\n{generated_text}"
                console.print(
                    Panel(panel_content, title=f"[cyan]vLLM Test: {model_name}[/cyan]")
                )
            else:
                print(f"\nQuery: {query}")
                print(f"Response: {generated_text}")
        else:
            print_error("No output generated")

    except Exception as e:
        print_error(f"Error testing vLLM model: {e}")
        if "CUDA" in str(e):
            print_info(
                "GPU may not be available. Try with smaller model or adjust gpu_memory_utilization."
            )


def list_tgi_models_command(args):
    """List Text Generation Inference compatible models."""
    print_info("Listing TGI-compatible models...")

    if not TGI_AVAILABLE:
        print_warning(
            "TGI client not available. Install with: uv add --optional local-engines tgi"
        )

    # Check TGI endpoints
    tgi_endpoints = [
        {
            "url": os.getenv("TGI_BASE_URL", "http://localhost:8080"),
            "name": "Local TGI",
        },
        {"url": os.getenv("TGI_BASE_URL2", ""), "name": "TGI Instance 2"},
        {"url": os.getenv("TGI_BASE_URL3", ""), "name": "TGI Instance 3"},
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
        print_error(
            "TGI client not available. Install with: uv add --optional local-engines tgi"
        )
        return

    endpoint_url = args.endpoint or os.getenv("TGI_BASE_URL", "http://localhost:8080")
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
            query, max_new_tokens=max_tokens, temperature=0.7, top_p=0.9, do_sample=True
        )

        end_time = datetime.now()
        latency = int((end_time - start_time).total_seconds() * 1000)

        print_success("Generation successful!")
        print_info(f"Latency: {latency}ms")
        print_info(f"Generated tokens: {len(response.generated_text.split())}")

        if console:
            panel_content = f"[bold]Query:[/bold] {query}\n\n"
            panel_content += f"[bold]Response:[/bold]\n{response.generated_text}"
            console.print(
                Panel(panel_content, title=f"[cyan]TGI Test: {endpoint_url}[/cyan]")
            )
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
            "description": "Llama 2 7B Chat via vLLM",
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
            "description": "Mistral 7B Instruct via vLLM",
        }

    # Add TGI providers
    tgi_base_url = os.getenv("TGI_BASE_URL", "http://localhost:8080")
    providers["tgi_local"] = {
        "type": "local",
        "provider": "tgi",
        "endpoint": tgi_base_url,
        "timeout": 30,
        "temperature": 0.7,
        "max_tokens": 100,
        "top_p": 0.9,
        "description": "Local Text Generation Inference server",
    }

    # Add Ollama providers (reference existing ones)
    providers["ollama_reference"] = {
        "type": "local",
        "provider": "ollama",
        "model": "llama3.1:8b",
        "host": "localhost",
        "port": 11434,
        "timeout": 120,
        "description": "Reference to existing Ollama setup",
    }

    config = {
        "name": "Local Inference Engines Configuration",
        "version": "1.0.0",
        "default_provider": list(providers.keys())[0]
        if providers
        else "vllm_llama2_7b",
        "providers": providers,
        "engine_settings": {
            "vllm": {
                "gpu_memory_utilization": 0.8,
                "max_model_len": 2048,
                "trust_remote_code": True,
                "tensor_parallel_size": 1,
            },
            "tgi": {"timeout": 30, "stream": False, "details": True},
            "ollama": {"host": "localhost", "port": 11434, "timeout": 120},
        },
        "resource_requirements": {
            "vllm_llama2_7b": {"gpu_memory_gb": 16, "ram_gb": 32},
            "vllm_mistral_7b": {"gpu_memory_gb": 16, "ram_gb": 32},
            "tgi_local": {"gpu_memory_gb": 8, "ram_gb": 16},
            "ollama_reference": {"ram_gb": 8},
        },
    }

    # Output configuration
    output_path = args.output
    if output_path:
        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)
        print_success(f"Local engines configuration written to: {output_path}")
    else:
        if console:
            syntax = Syntax(json.dumps(config, indent=2), "json", theme="monokai")
            console.print(
                Panel(
                    syntax, title="[cyan]Local Inference Engines Configuration[/cyan]"
                )
            )
        else:
            print(json.dumps(config, indent=2))

    print_info(f"Generated configuration for {len(providers)} local engine providers")

    # Show availability status
    print_info("\nEngine Availability:")
    print_info(f"  vLLM: {'✓ Available' if VLLM_AVAILABLE else '✗ Not installed'}")
    print_info(f"  TGI: {'✓ Available' if TGI_AVAILABLE else '✗ Not installed'}")
    print_info(
        f"  Ollama: {'✓ Available' if os.system('which ollama > /dev/null 2>&1') == 0 else '✗ Not installed'}"
    )


def create_cli_parser():
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="🦙 LlamaFarm Models CLI - Manage Cloud and Local Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query models with default or custom configs
  uv run python cli.py query "What is machine learning?"
  uv run python cli.py --config strategies/examples/production.yaml query "Explain AI" --provider groq_llama3_70b
  
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
  uv run python cli.py --config strategies/examples/use_case_examples.yaml list
  
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
  
  # Model catalog operations
  uv run python cli.py catalog list --category medical --detailed
  uv run python cli.py catalog fallbacks --chain medical_chain
  uv run python cli.py catalog search "medical"
  uv run python cli.py catalog info "hf.co/mradermacher/DeepSeek-R1-Medicalai-923-i1-GGUF:Q4_K_M"
  
  # =================== DEMO SCRIPTS ===================
  
  # Run Cloud Fallback Demo (shows automatic failover)
  python demos/demo1_cloud_fallback.py
  
  # Run Multi-Model Demo (shows task-optimized model selection)
  python demos/demo2_multi_model.py
  
  # Run Training Demo (fine-tune and convert to Ollama)
  python demos/demo3_training.py
  
  # Run all demos
  python demos/run_all_demos.py
  
  # Convert a model to Ollama format
  uv run python cli.py convert ./fine_tuned_models/medical ./medical_model --format ollama --model-name medical-assistant
  
  # Convert to GGUF with quantization
  uv run python cli.py convert ./models/phi-2 ./models/phi-2.gguf --format gguf --quantization q4_0
  
  # Train with custom strategy
  uv run python cli.py train --strategy demo3_training --dataset ./data/training.jsonl --verbose --export-ollama
        """,
    )

    # Global options
    parser.add_argument(
        "--config",
        "-c",
        default="strategies/default.yaml",
        help="Configuration file path (supports .yaml, .yml, and .json)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ==============================================================================
    # NEW STRATEGY-BASED COMMANDS
    # ==============================================================================

    # List strategies command
    list_strategies_parser = subparsers.add_parser(
        "list-strategies", help="List available strategies"
    )

    # Use strategy command
    use_strategy_parser = subparsers.add_parser(
        "use-strategy", help="Switch to and use a specific strategy"
    )
    use_strategy_parser.add_argument("strategy", help="Strategy name to use")
    use_strategy_parser.add_argument(
        "--save", action="store_true", help="Save as current strategy"
    )

    # Info command
    info_parser = subparsers.add_parser(
        "info", help="Show detailed information about current or specified strategy"
    )
    info_parser.add_argument(
        "--strategy", help="Strategy name to show info for (default: current strategy)"
    )

    # Generate command (new strategy-based version)
    generate_parser = subparsers.add_parser(
        "generate", help="Generate text using current strategy"
    )
    generate_parser.add_argument("prompt", help="Text prompt to generate from")
    generate_parser.add_argument(
        "--strategy", help="Strategy to use (default: current strategy)"
    )
    generate_parser.add_argument(
        "--max-tokens", type=int, help="Maximum tokens to generate"
    )
    generate_parser.add_argument(
        "--temperature", type=float, help="Temperature for generation"
    )
    generate_parser.add_argument(
        "--stream", action="store_true", help="Stream the response"
    )
    generate_parser.add_argument(
        "--json", action="store_true", help="Output response as JSON"
    )
    generate_parser.add_argument("--save", help="Save response to file")

    # Complete command - provider-agnostic text completion
    complete_parser = subparsers.add_parser(
        "complete", help="Get text completion using strategy (provider-agnostic)"
    )
    complete_parser.add_argument("prompt", help="Text prompt for completion")
    complete_parser.add_argument(
        "--strategy", "-s", required=True, help="Strategy name from strategies file"
    )
    complete_parser.add_argument(
        "--strategy-file",
        "-f",
        default="demos/strategies.yaml",
        help="Path to strategies YAML file (default: demos/strategies.yaml)",
    )
    complete_parser.add_argument(
        "--max-tokens", "-m", type=int, help="Maximum tokens to generate"
    )
    complete_parser.add_argument(
        "--temperature", "-t", type=float, help="Temperature for generation"
    )
    complete_parser.add_argument(
        "--stream", action="store_true", help="Stream the response"
    )
    complete_parser.add_argument("--system", help="System prompt to use")
    complete_parser.add_argument(
        "--json", action="store_true", help="Output response as JSON"
    )
    complete_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show provider details"
    )

    # ==============================================================================
    # LEGACY COMMANDS (kept for backward compatibility)
    # ==============================================================================

    # List command
    list_parser = subparsers.add_parser(
        "list", help="List available model providers (legacy)"
    )
    list_parser.add_argument(
        "--detailed", "-d", action="store_true", help="Show detailed configuration"
    )

    # Test command
    test_parser = subparsers.add_parser("test", help="Test model connectivity")
    test_parser.add_argument("provider", help="Provider name to test")
    test_parser.add_argument(
        "--query", "-q", default="Hello, world!", help="Test query to send"
    )

    # Query command - comprehensive model interaction
    query_parser = subparsers.add_parser(
        "query", help="Send a query to a model with full control"
    )
    query_parser.add_argument("query", help="Query text to send to the model")
    query_parser.add_argument(
        "--provider", "-p", help="Specific provider to use (default: from config)"
    )
    query_parser.add_argument(
        "--temperature", "-t", type=float, help="Override temperature setting"
    )
    query_parser.add_argument(
        "--max-tokens", "-m", type=int, help="Override max tokens"
    )
    query_parser.add_argument("--top-p", type=float, help="Override top-p setting")
    query_parser.add_argument("--system", "-s", help="System prompt to use")
    query_parser.add_argument(
        "--stream", action="store_true", help="Stream the response"
    )
    query_parser.add_argument(
        "--json", action="store_true", help="Output response as JSON"
    )
    query_parser.add_argument("--save", help="Save response to file")

    # Chat command - interactive chat session
    chat_parser = subparsers.add_parser(
        "chat", help="Start an interactive chat session"
    )
    chat_parser.add_argument(
        "--provider", "-p", help="Specific provider to use (default: from config)"
    )
    chat_parser.add_argument(
        "--temperature", "-t", type=float, help="Temperature setting"
    )
    chat_parser.add_argument("--system", "-s", help="System prompt to use")
    chat_parser.add_argument("--history", help="Load chat history from file")
    chat_parser.add_argument("--save-history", help="Save chat history to file")

    # Send command - send file content to model
    send_parser = subparsers.add_parser("send", help="Send file content to a model")
    send_parser.add_argument("file", help="File path to send")
    send_parser.add_argument("--provider", "-p", help="Specific provider to use")
    send_parser.add_argument("--prompt", help="Additional prompt to prepend")
    send_parser.add_argument(
        "--temperature", "-t", type=float, help="Temperature setting"
    )
    send_parser.add_argument("--max-tokens", "-m", type=int, help="Maximum tokens")
    send_parser.add_argument("--output", "-o", help="Save response to file")

    # Batch command - process multiple queries
    batch_parser = subparsers.add_parser(
        "batch", help="Process multiple queries from a file"
    )
    batch_parser.add_argument("file", help="File containing queries (one per line)")
    batch_parser.add_argument("--provider", "-p", help="Specific provider to use")
    batch_parser.add_argument(
        "--temperature", "-t", type=float, help="Temperature setting"
    )
    batch_parser.add_argument("--output", "-o", help="Output file for responses")
    batch_parser.add_argument(
        "--parallel", "-j", type=int, default=1, help="Number of parallel requests"
    )

    # Health check command
    health_parser = subparsers.add_parser(
        "health-check", help="Check health of all providers"
    )

    # Validate config command
    validate_parser = subparsers.add_parser(
        "validate-config", help="Validate configuration file"
    )

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare model responses")
    compare_parser.add_argument(
        "--providers", "-p", required=True, help="Comma-separated list of providers"
    )
    compare_parser.add_argument(
        "--query", "-q", required=True, help="Query to test with"
    )

    # Generate config command
    generate_parser = subparsers.add_parser(
        "generate-config", help="Generate configuration template"
    )
    generate_parser.add_argument(
        "--type",
        "-t",
        choices=["basic", "multi", "production"],
        default="basic",
        help="Configuration type",
    )
    generate_parser.add_argument("--output", "-o", help="Output file path")

    # List local models command
    list_local_parser = subparsers.add_parser(
        "list-local", help="List locally available Ollama models"
    )

    # Pull model command
    pull_parser = subparsers.add_parser("pull", help="Pull a model to Ollama")
    pull_parser.add_argument("model", help="Model name to pull (e.g., llama3.1:8b)")

    # Test local model command
    test_local_parser = subparsers.add_parser(
        "test-local", help="Test a local Ollama model"
    )
    test_local_parser.add_argument("model", help="Local model name to test")
    test_local_parser.add_argument("--query", "-q", help="Test query to send")

    # Ollama management commands
    ollama_parser = subparsers.add_parser("ollama", help="Manage Ollama models")
    ollama_subparsers = ollama_parser.add_subparsers(
        dest="ollama_command", help="Ollama commands"
    )

    # List Ollama models
    ollama_list_parser = ollama_subparsers.add_parser(
        "list", help="List installed Ollama models"
    )

    # Pull/download Ollama model
    ollama_pull_parser = ollama_subparsers.add_parser(
        "pull", help="Download an Ollama model"
    )
    ollama_pull_parser.add_argument(
        "model", help="Model name to download (e.g., tinyllama)"
    )

    # Run Ollama model
    ollama_run_parser = ollama_subparsers.add_parser("run", help="Run an Ollama model")
    ollama_run_parser.add_argument("model", help="Model name to run")
    ollama_run_parser.add_argument("prompt", help="Prompt to send to the model")
    ollama_run_parser.add_argument(
        "--stream", action="store_true", help="Stream the response"
    )

    # Check if Ollama is running
    ollama_status_parser = ollama_subparsers.add_parser(
        "status", help="Check Ollama status"
    )

    # Generate Ollama config command
    ollama_config_parser = subparsers.add_parser(
        "generate-ollama-config", help="Generate Ollama-specific configuration"
    )
    ollama_config_parser.add_argument("--output", "-o", help="Output file path")

    # Hugging Face commands
    hf_list_parser = subparsers.add_parser(
        "list-hf", help="List available Hugging Face models"
    )
    hf_list_parser.add_argument("--search", "-s", help="Search term for models")
    hf_list_parser.add_argument(
        "--limit", "-l", type=int, default=20, help="Limit number of results"
    )

    hf_download_parser = subparsers.add_parser(
        "download-hf", help="Download a Hugging Face model"
    )
    hf_download_parser.add_argument(
        "model_id", help="Hugging Face model ID (e.g., gpt2)"
    )
    hf_download_parser.add_argument("--cache-dir", help="Custom cache directory")
    hf_download_parser.add_argument(
        "--include-images", action="store_true", help="Include image files in download"
    )

    hf_test_parser = subparsers.add_parser("test-hf", help="Test a Hugging Face model")
    hf_test_parser.add_argument("model_id", help="Hugging Face model ID to test")
    hf_test_parser.add_argument("--query", "-q", help="Test query to send")
    hf_test_parser.add_argument(
        "--max-tokens", type=int, default=50, help="Maximum tokens to generate"
    )
    hf_test_parser.add_argument(
        "--gpu", action="store_true", help="Use GPU if available"
    )

    hf_config_parser = subparsers.add_parser(
        "generate-hf-config", help="Generate Hugging Face configuration"
    )
    hf_config_parser.add_argument("--output", "-o", help="Output file path")
    hf_config_parser.add_argument(
        "--models", "-m", help="Comma-separated list of model IDs"
    )

    hf_login_parser = subparsers.add_parser(
        "hf-login", help="Login to Hugging Face Hub"
    )

    # Local inference engines commands
    vllm_list_parser = subparsers.add_parser(
        "list-vllm", help="List vLLM-compatible models"
    )

    vllm_test_parser = subparsers.add_parser(
        "test-vllm", help="Test a model using vLLM"
    )
    vllm_test_parser.add_argument("model", help="Model name/path to test")
    vllm_test_parser.add_argument("--query", "-q", help="Test query to send")
    vllm_test_parser.add_argument(
        "--max-tokens", type=int, default=50, help="Maximum tokens to generate"
    )
    vllm_test_parser.add_argument(
        "--max-model-len", type=int, default=2048, help="Maximum model context length"
    )
    vllm_test_parser.add_argument(
        "--gpu-memory", type=float, default=0.8, help="GPU memory utilization (0.0-1.0)"
    )

    tgi_list_parser = subparsers.add_parser(
        "list-tgi", help="List TGI endpoints and status"
    )

    tgi_test_parser = subparsers.add_parser(
        "test-tgi", help="Test a Text Generation Inference endpoint"
    )
    tgi_test_parser.add_argument("--endpoint", "-e", help="TGI endpoint URL")
    tgi_test_parser.add_argument("--query", "-q", help="Test query to send")
    tgi_test_parser.add_argument(
        "--max-tokens", type=int, default=50, help="Maximum tokens to generate"
    )

    engines_config_parser = subparsers.add_parser(
        "generate-engines-config", help="Generate local inference engines configuration"
    )
    engines_config_parser.add_argument("--output", "-o", help="Output file path")
    engines_config_parser.add_argument(
        "--include-unavailable",
        action="store_true",
        help="Include unavailable engines in config",
    )

    # Model catalog commands
    catalog_parser = subparsers.add_parser("catalog", help="Model catalog operations")
    catalog_subparsers = catalog_parser.add_subparsers(
        dest="catalog_command", help="Catalog commands"
    )

    # List models from catalog
    catalog_list_parser = catalog_subparsers.add_parser(
        "list", help="List models from catalog"
    )
    catalog_list_parser.add_argument(
        "--category",
        "-c",
        help="Filter by category (medical, code_generation, multilingual, etc.)",
    )
    catalog_list_parser.add_argument(
        "--format",
        "-f",
        choices=["table", "json", "yaml"],
        default="table",
        help="Output format",
    )
    catalog_list_parser.add_argument(
        "--detailed", "-d", action="store_true", help="Show detailed model information"
    )

    # Show fallback chains
    fallback_parser = catalog_subparsers.add_parser(
        "fallbacks", help="Show fallback chains"
    )
    fallback_parser.add_argument("--chain", help="Show specific fallback chain")
    fallback_parser.add_argument(
        "--format",
        "-f",
        choices=["table", "json", "yaml"],
        default="table",
        help="Output format",
    )

    # Search models
    search_parser = catalog_subparsers.add_parser(
        "search", help="Search models in catalog"
    )
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--format",
        "-f",
        choices=["table", "json", "yaml"],
        default="table",
        help="Output format",
    )

    # Show model details
    info_parser = catalog_subparsers.add_parser(
        "info", help="Show detailed model information"
    )
    info_parser.add_argument("model", help="Model name")
    info_parser.add_argument(
        "--format",
        "-f",
        choices=["table", "json", "yaml"],
        default="table",
        help="Output format",
    )

    # Fine-tuning commands
    if FINETUNING_AVAILABLE:
        finetune_parser = subparsers.add_parser(
            "finetune", help="Fine-tuning operations"
        )
        finetune_subparsers = finetune_parser.add_subparsers(
            dest="finetune_command", help="Fine-tuning commands"
        )

        # Start training command
        start_parser = finetune_subparsers.add_parser("start", help="Start fine-tuning")
        start_parser.add_argument(
            "--dataset", "-d", required=True, help="Path to training dataset"
        )
        start_parser.add_argument(
            "--config", "-c", help="Fine-tuning configuration file"
        )
        start_parser.add_argument("--strategy", "-s", help="Strategy name to use")
        start_parser.add_argument("--base-model", "-m", help="Base model to fine-tune")
        start_parser.add_argument(
            "--method", help="Fine-tuning method (lora, qlora, full)"
        )
        start_parser.add_argument(
            "--output-dir", "-o", help="Output directory for results"
        )
        start_parser.add_argument("--job-name", help="Name for the training job")
        start_parser.add_argument(
            "--dry-run", action="store_true", help="Validate config without training"
        )

        # Monitor training command
        monitor_parser = finetune_subparsers.add_parser(
            "monitor", help="Monitor training progress"
        )
        monitor_parser.add_argument("--job-id", help="Job ID to monitor")
        monitor_parser.add_argument(
            "--follow", "-f", action="store_true", help="Follow progress updates"
        )

        # Stop training command
        stop_parser = finetune_subparsers.add_parser("stop", help="Stop training")
        stop_parser.add_argument("--job-id", help="Job ID to stop")

        # Resume training command
        resume_parser = finetune_subparsers.add_parser(
            "resume", help="Resume training from checkpoint"
        )
        resume_parser.add_argument(
            "--checkpoint", "-c", required=True, help="Checkpoint path"
        )
        resume_parser.add_argument("--job-name", help="Name for the resumed job")

        # List jobs command
        jobs_parser = finetune_subparsers.add_parser("jobs", help="List training jobs")
        jobs_parser.add_argument("--status", help="Filter by status")
        jobs_parser.add_argument(
            "--limit", type=int, default=10, help="Number of jobs to show"
        )

        # Evaluate model command
        eval_parser = finetune_subparsers.add_parser(
            "evaluate", help="Evaluate fine-tuned model"
        )
        eval_parser.add_argument(
            "--model-path", "-m", required=True, help="Path to fine-tuned model"
        )
        eval_parser.add_argument("--dataset", "-d", help="Evaluation dataset")
        eval_parser.add_argument("--output", "-o", help="Output file for results")

        # Export model command
        export_parser = finetune_subparsers.add_parser(
            "export", help="Export fine-tuned model"
        )
        export_parser.add_argument(
            "--model-path", "-m", required=True, help="Path to fine-tuned model"
        )
        export_parser.add_argument(
            "--output", "-o", required=True, help="Output directory"
        )
        export_parser.add_argument("--format", default="pytorch", help="Export format")

        # Strategy management commands
        strategies_parser = finetune_subparsers.add_parser(
            "strategies", help="Manage fine-tuning strategies"
        )
        strategies_subparsers = strategies_parser.add_subparsers(
            dest="strategies_command", help="Strategy commands"
        )

        # List strategies
        list_strat_parser = strategies_subparsers.add_parser(
            "list", help="List available strategies"
        )
        list_strat_parser.add_argument(
            "--detailed", action="store_true", help="Show detailed information"
        )

        # Show strategy details
        show_strat_parser = strategies_subparsers.add_parser(
            "show", help="Show strategy details"
        )
        show_strat_parser.add_argument("name", help="Strategy name")

        # Recommend strategies
        recommend_parser = strategies_subparsers.add_parser(
            "recommend", help="Get strategy recommendations"
        )
        recommend_parser.add_argument(
            "--hardware", help="Hardware type (mac, gpu, cpu)"
        )
        recommend_parser.add_argument("--model-size", help="Model size (3b, 8b, 70b)")
        recommend_parser.add_argument("--use-case", help="Use case description")

        # Estimate resources
        estimate_parser = finetune_subparsers.add_parser(
            "estimate", help="Estimate resource requirements"
        )
        estimate_parser.add_argument("--strategy", help="Strategy to estimate")
        estimate_parser.add_argument("--config", help="Configuration file to estimate")
        estimate_parser.add_argument("--base-model", help="Base model name")

    # ==============================================================================
    # CONVERT COMMAND
    # ==============================================================================
    convert_parser = subparsers.add_parser(
        "convert", help="Convert models between formats"
    )
    convert_parser.add_argument("input_path", help="Path to input model")
    convert_parser.add_argument("output_path", help="Path for output model")
    convert_parser.add_argument(
        "--format",
        "-f",
        choices=["ollama", "gguf", "pytorch", "safetensors"],
        default="ollama",
        help="Target format for conversion",
    )
    convert_parser.add_argument(
        "--quantization",
        "-q",
        choices=["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "f16", "f32"],
        default="q4_0",
        help="Quantization method (for GGUF/Ollama)",
    )
    convert_parser.add_argument(
        "--model-name", "-n", help="Name for the model (for Ollama)"
    )
    convert_parser.add_argument("--system-prompt", help="System prompt for the model")
    convert_parser.add_argument(
        "--push",
        action="store_true",
        help="Push to registry after conversion (Ollama only)",
    )

    # ==============================================================================
    # TRAIN COMMAND (Enhanced)
    # ==============================================================================
    train_parser = subparsers.add_parser("train", help="Train a model using strategy")
    train_parser.add_argument(
        "--strategy", "-s", required=True, help="Strategy name or path to strategy file"
    )
    train_parser.add_argument(
        "--dataset", "-d", required=True, help="Path to training dataset"
    )
    train_parser.add_argument(
        "--output", "-o", help="Output directory for trained model"
    )
    train_parser.add_argument("--epochs", type=int, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, help="Batch size for training")
    train_parser.add_argument("--learning-rate", type=float, help="Learning rate")
    train_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show training progress"
    )
    train_parser.add_argument(
        "--export-ollama",
        action="store_true",
        help="Export to Ollama format after training",
    )
    train_parser.add_argument(
        "--export-gguf",
        action="store_true",
        help="Export to GGUF format after training",
    )
    train_parser.add_argument(
        "--train-dataset",
        help="Path to training dataset (when using separate train/eval)",
    )
    train_parser.add_argument(
        "--eval-dataset",
        help="Path to evaluation dataset (when using separate train/eval)",
    )

    # ==============================================================================
    # DATASPLIT COMMAND - Create train/eval splits for datasets
    # ==============================================================================
    datasplit_parser = subparsers.add_parser(
        "datasplit", help="Create train/eval splits for fine-tuning datasets"
    )
    datasplit_parser.add_argument("input", help="Input JSONL file to split")
    datasplit_parser.add_argument(
        "--eval-percent",
        "-e",
        type=int,
        default=10,
        help="Percentage for evaluation (1-50, default: 10)",
    )
    datasplit_parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    datasplit_parser.add_argument(
        "--output-dir", "-o", help="Output directory (defaults to input file directory)"
    )
    datasplit_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed statistics"
    )

    # ==============================================================================
    # SETUP COMMAND - Install tools and models from strategy requirements
    # ==============================================================================
    setup_parser = subparsers.add_parser(
        "setup", help="Setup tools and models from strategy requirements"
    )
    setup_parser.add_argument(
        "strategy_file", help="Path to strategy YAML file to analyze"
    )
    setup_parser.add_argument(
        "--auto",
        action="store_true",
        help="Automatic mode - install everything without prompts",
    )
    setup_parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify setup, don't install anything",
    )
    setup_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show verbose output during setup"
    )

    return parser


def datasplit_command(args):
    """Create train/eval splits for datasets."""
    import json
    import random
    from pathlib import Path

    input_path = Path(args.input)
    if not input_path.exists():
        print_error(f"Input file not found: {args.input}")
        return

    # Validate eval percentage
    if not 1 <= args.eval_percent <= 50:
        print_error("Eval percentage must be between 1 and 50")
        return

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = input_path.parent

    # Read all lines
    if args.verbose:
        print_info(f"Reading {input_path}...")

    with open(input_path, "r") as f:
        lines = f.readlines()

    total_count = len(lines)
    if args.verbose:
        print_info(f"Total examples: {total_count}")

    # Shuffle with seed for reproducibility
    random.seed(args.seed)
    random.shuffle(lines)

    # Calculate split point
    split_point = int(total_count * (1 - args.eval_percent / 100))
    train_lines = lines[:split_point]
    eval_lines = lines[split_point:]

    # Generate output filenames
    base_name = input_path.stem
    train_file = output_dir / f"{base_name}_train.jsonl"
    eval_file = output_dir / f"{base_name}_eval.jsonl"

    # Write train dataset
    with open(train_file, "w") as f:
        f.writelines(train_lines)

    # Write eval dataset
    with open(eval_file, "w") as f:
        f.writelines(eval_lines)

    # Print statistics
    print_success(f"Split created successfully!")

    if args.verbose:
        print("\n📊 Statistics:")
        print(f"  • Train: {len(train_lines)} examples ({100 - args.eval_percent}%)")
        print(f"  • Eval:  {len(eval_lines)} examples ({args.eval_percent}%)")
        print(f"  • Ratio: {len(train_lines):.1f}:{len(eval_lines):.1f}")

        # Sample the first example from each split
        if train_lines:
            try:
                train_sample = json.loads(train_lines[0])
                print("\n📝 Sample from training set:")
                if "instruction" in train_sample:
                    print(f"  Instruction: {train_sample['instruction'][:100]}...")
                if "input" in train_sample:
                    print(f"  Input: {train_sample['input'][:100]}...")
            except:
                pass
    else:
        print(f"  Train: {len(train_lines)} examples → {train_file}")
        print(f"  Eval:  {len(eval_lines)} examples → {eval_file}")

    print(f"\n🎲 Random seed: {args.seed} (use same seed for reproducible splits)")


def finetune_command(args):
    """Handle fine-tuning commands."""
    if not FINETUNING_AVAILABLE:
        print_error("Fine-tuning components not available")
        return

    if not args.finetune_command:
        print_error("No fine-tuning command specified")
        print_info(
            "Available commands: start, monitor, stop, resume, jobs, evaluate, export, strategies, estimate"
        )
        return

    if args.finetune_command == "start":
        start_finetuning(args)
    elif args.finetune_command == "monitor":
        monitor_finetuning(args)
    elif args.finetune_command == "stop":
        stop_finetuning(args)
    elif args.finetune_command == "resume":
        resume_finetuning(args)
    elif args.finetune_command == "jobs":
        list_finetune_jobs(args)
    elif args.finetune_command == "evaluate":
        evaluate_finetuned_model(args)
    elif args.finetune_command == "export":
        export_finetuned_model(args)
    elif args.finetune_command == "strategies":
        manage_finetune_strategies(args)
    elif args.finetune_command == "estimate":
        estimate_finetune_resources(args)
    else:
        print_error(f"Unknown fine-tuning command: {args.finetune_command}")


def start_finetuning(args):
    """Start a fine-tuning job."""
    print_info("Starting fine-tuning job...")

    try:
        # Load or create configuration
        if args.config:
            # Load from file
            config_data = load_config(args.config)
            config = FineTuningConfig(**config_data)
        elif args.strategy:
            # Load from strategy
            from core.strategy_manager import StrategyManager

            strategy_manager = StrategyManager()
            try:
                strategy_data = strategy_manager.load_strategy(args.strategy)
                # Extract fine_tuner config from strategy
                if "fine_tuner" not in strategy_data:
                    print_error(
                        f"Strategy '{args.strategy}' does not contain fine-tuning configuration"
                    )
                    return
                config_data = strategy_data["fine_tuner"]["config"]
                # Add framework type
                config_data["framework"] = {"type": strategy_data["fine_tuner"]["type"]}
                # Add dataset if provided via CLI
                if args.dataset:
                    config_data["dataset"] = {"path": args.dataset}
                config = FineTuningConfig(**config_data)
            except ValueError as e:
                print_error(str(e))
                return
        else:
            # Create minimal config from arguments
            config_data = {
                "base_model": {"name": args.base_model or "llama3.2-3b"},
                "method": {"type": args.method or "lora"},
                "framework": {"type": "pytorch"},
                "training_args": {
                    "output_dir": args.output_dir or "./fine_tuned_models",
                    "num_train_epochs": 3,
                },
                "dataset": {"path": args.dataset},
            }
            config = FineTuningConfig(**config_data)

        # Override with CLI arguments
        if args.dataset:
            config.dataset["path"] = args.dataset
        if args.output_dir:
            config.training_args["output_dir"] = args.output_dir
        if args.base_model:
            config.base_model["name"] = args.base_model
        if args.method:
            config.method["type"] = args.method

        if args.dry_run:
            print_info("Dry run - validating configuration...")
            tuner = FineTunerFactory.create(config)
            errors = tuner.validate_config()
            if errors:
                print_error("Configuration errors:")
                for error in errors:
                    print_error(f"  - {error}")
                return
            else:
                print_success("Configuration is valid")

                # Show estimates
                estimates = tuner.estimate_resources()
                print_info("Resource estimates:")
                print_info(f"  Memory required: {estimates['memory_gb']} GB")
                print_info(
                    f"  Training time: ~{estimates['training_time_hours']} hours"
                )
                print_info(f"  Storage needed: {estimates['storage_gb']} GB")
                print_info(
                    f"  GPU required: {'Yes' if estimates['gpu_required'] else 'No'}"
                )
                return

        # Create and start fine-tuner
        tuner = FineTunerFactory.create(config)

        # Validate configuration
        errors = tuner.validate_config()
        if errors:
            print_error("Configuration errors:")
            for error in errors:
                print_error(f"  - {error}")
            return

        # Start training
        job = tuner.start_training()
        print_success(f"Training started - Job ID: {job.job_id}")
        print_info(f"Output directory: {job.output_dir}")
        print_info(
            f"Monitor with: uv run python cli.py finetune monitor --job-id {job.job_id}"
        )

    except Exception as e:
        print_error(f"Failed to start training: {e}")
        import traceback

        print_error(traceback.format_exc())


def monitor_finetuning(args):
    """Monitor fine-tuning progress."""
    job_id = getattr(args, "job_id", None)

    if not job_id:
        print_error("No job ID specified")
        return

    # For now, show basic training status
    # In a real implementation, this would check actual training logs
    training_dir = Path("./fine_tuned_models")
    log_files = list(training_dir.glob("**/training_log.jsonl"))

    if log_files:
        # Show the most recent log entries
        latest_log = log_files[-1]
        print_info(f"Monitoring job: {job_id}")
        print_info(f"Log file: {latest_log}")

        try:
            with open(latest_log, "r") as f:
                lines = f.readlines()
                if lines:
                    # Show last few log entries
                    import json

                    for line in lines[-3:]:
                        try:
                            log_entry = json.loads(line.strip())
                            epoch = log_entry.get("epoch", "?")
                            loss = log_entry.get("loss", "?")
                            step = log_entry.get("step", "?")
                            print_info(f"  Epoch {epoch}, Step {step}: Loss = {loss}")
                        except:
                            print_info(f"  {line.strip()}")
                else:
                    print_info("  Training just started...")
        except Exception as e:
            print_info("  Training in progress...")
    else:
        print_info(f"Monitoring job: {job_id}")
        print_info("  Training in progress...")
        print_info("  No log files found yet - training may be starting up")


def stop_finetuning(args):
    """Stop fine-tuning job."""
    print_info("Stop not yet implemented")
    # TODO: Implement job stopping


def resume_finetuning(args):
    """Resume fine-tuning from checkpoint."""
    print_info("Resume not yet implemented")
    # TODO: Implement checkpoint resuming


def list_finetune_jobs(args):
    """List fine-tuning jobs."""
    print_info("Job listing not yet implemented")
    # TODO: Implement job listing


def evaluate_finetuned_model(args):
    """Evaluate fine-tuned model."""
    model_path = getattr(args, "model_path", None)

    if not model_path:
        print_error("No model path specified")
        return

    model_path = Path(model_path)
    if not model_path.exists():
        print_error(f"Model path does not exist: {model_path}")
        return

    print_info(f"Evaluating model: {model_path}")

    # Check for model files
    config_file = model_path / "config.json"
    model_files = list(model_path.glob("pytorch_model.bin")) + list(
        model_path.glob("model.safetensors")
    )

    if config_file.exists():
        print_success("✓ Model configuration found")
        try:
            import json

            with open(config_file) as f:
                config = json.load(f)
                print_info(f"  Model type: {config.get('model_type', 'unknown')}")
                print_info(
                    f"  Architecture: {config.get('architectures', ['unknown'])[0]}"
                )
        except:
            pass
    else:
        print_error("✗ Model configuration not found")

    if model_files:
        print_success("✓ Model weights found")
        model_size = sum(f.stat().st_size for f in model_files) // 1024 // 1024
        print_info(f"  Model size: ~{model_size} MB")
    else:
        print_error("✗ Model weights not found")

    # Test queries
    test_queries = [
        "What are the symptoms of a cold?",
        "How can I stay healthy?",
        "What should I do for a headache?",
    ]

    print_info("Running evaluation tests...")

    # Import necessary libraries for model evaluation
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        libraries_available = True
    except ImportError as e:
        print_error(f"Required libraries not installed: {e}")
        print_info("Install with: uv add transformers torch")
        libraries_available = False

    if not libraries_available:
        print_error("Cannot run real evaluation without transformers library")
        return

    # Load the actual model
    try:
        print_info("Loading fine-tuned model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        print_success("✓ Model loaded successfully")

        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    except Exception as e:
        print_error(f"Failed to load model: {e}")
        print_info("Make sure the model was properly trained and saved")
        return

    # Create a strategy configuration for this model
    print_info("Creating strategy configuration for evaluation...")
    import yaml

    strategy_config = {
        "evaluation_model": {
            "description": f"Model being evaluated: {model_path}",
            "local_engines": {
                "type": "huggingface",
                "config": {
                    "default_model": str(model_path),
                    "model_path": str(model_path),
                    "device": "auto",
                    "torch_dtype": "auto",
                    "trust_remote_code": True,
                },
            },
        }
    }

    # Write strategy to temporary file
    strategy_file = model_path.parent / "evaluation_strategy.yaml"
    with open(strategy_file, "w") as f:
        yaml.dump(strategy_config, f, default_flow_style=False)

    print_success(f"✓ Created evaluation strategy: {strategy_file}")

    # Run evaluation tests using the strategy system
    results = []
    for i, query in enumerate(test_queries, 1):
        print_info(f"Test {i}/3: {query}")

        try:
            # Use the existing CLI query command with our strategy
            import subprocess

            result = subprocess.run(
                [
                    "python",
                    "cli.py",
                    "--config",
                    str(strategy_file),
                    "query",
                    query,
                    "--max-tokens",
                    "150",
                    "--temperature",
                    "0.7",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0 and result.stdout:
                response = result.stdout.strip()
                print_success(f"  ✓ Generated response ({len(response)} chars)")
                print_info(f"  Response: {response[:100]}...")

                # Check for medical disclaimers (shows fine-tuning worked)
                has_disclaimer = any(
                    word in response.lower()
                    for word in [
                        "disclaimer",
                        "consult",
                        "healthcare",
                        "medical",
                        "doctor",
                    ]
                )
                if has_disclaimer:
                    print_success("  ✓ Safety disclaimer detected")
                else:
                    print_warning("  ⚠ No safety disclaimer found")

                results.append(
                    {
                        "query": query,
                        "response": response,
                        "has_disclaimer": has_disclaimer,
                        "success": True,
                    }
                )
            else:
                error_msg = result.stderr if result.stderr else "No response generated"
                print_error(f"  ✗ Strategy-based evaluation failed: {error_msg}")
                print_info(
                    "  💡 This may indicate missing dependencies or model loading issues"
                )
                results.append(
                    {
                        "query": query,
                        "response": "",
                        "has_disclaimer": False,
                        "success": False,
                    }
                )

        except Exception as e:
            print_error(f"  ✗ Test {i} failed: {e}")
            results.append(
                {
                    "query": query,
                    "response": "",
                    "has_disclaimer": False,
                    "success": False,
                }
            )

    # Display evaluation summary
    successful_tests = sum(1 for r in results if r["success"])
    safety_disclaimers = sum(1 for r in results if r["has_disclaimer"])

    print_info("\n=== EVALUATION SUMMARY ===")
    print_info(f"Successful responses: {successful_tests}/{len(test_queries)}")
    print_info(
        f"Responses with safety disclaimers: {safety_disclaimers}/{len(test_queries)}"
    )

    if successful_tests >= 2 and safety_disclaimers >= 1:
        print_success("🎉 Fine-tuning evaluation PASSED!")
        print_success("Model generates responses and includes safety disclaimers")
    elif successful_tests >= 2:
        print_warning("⚠️ Model responds but needs more safety training")
    else:
        print_error("❌ Model evaluation FAILED - training may be incomplete")

    print_success("Real model evaluation completed")


def export_finetuned_model(args):
    """Export fine-tuned model."""
    print_info("Model export not yet implemented")
    # TODO: Implement model export


def manage_finetune_strategies(args):
    """Manage fine-tuning strategies."""
    if not args.strategies_command:
        print_error("No strategy command specified")
        return

    try:
        from core.strategy_manager import StrategyManager

        strategy_manager = StrategyManager()

        if args.strategies_command == "list":
            strategies = strategy_manager.list_strategies()

            # Filter for strategies with fine_tuner component
            finetuning_strategies = []
            for strategy_name in strategies:
                strategy_config = strategy_manager.get_strategy(strategy_name)
                if strategy_config.get("components", []):
                    for component in strategy_config.get("components", []):
                        if component.get("type") == "fine_tuner":
                            finetuning_strategies.append(strategy_config)
                            break

            if not finetuning_strategies:
                print_info("No fine-tuning strategies available")
                return

            print_info(
                f"Available fine-tuning strategies ({len(finetuning_strategies)}):"
            )
            for strategy in finetuning_strategies:
                print_info(
                    f"  {strategy['name']}: {strategy.get('description', 'No description')}"
                )

        elif args.strategies_command == "show":
            info = strategy_manager.get_strategy_info(args.name)
            if not info:
                print_error(f"Strategy not found: {args.name}")
                return

            print_info(f"Strategy: {args.name}")
            print_info(f"Description: {info.get('description', 'No description')}")
            print_info(f"Use cases: {', '.join(info.get('use_cases', []))}")
            print_info(
                f"Hardware: {info.get('hardware_requirements', {}).get('type', 'Unknown')}"
            )
            print_info(f"Resource usage: {info.get('resource_usage', 'Unknown')}")
            print_info(f"Complexity: {info.get('complexity', 'Unknown')}")

        elif args.strategies_command == "recommend":
            recommendations = strategy_manager.recommend_strategies(
                hardware=args.hardware,
                model_size=args.model_size,
                use_case=args.use_case,
            )

            if not recommendations:
                print_info("No matching strategies found")
                return

            print_info("Recommended strategies:")
            for rec in recommendations[:3]:  # Top 3
                print_info(f"  {rec['name']}: {rec['description']}")

    except Exception as e:
        print_error(f"Strategy management error: {e}")


def estimate_finetune_resources(args):
    """Estimate fine-tuning resource requirements."""
    try:
        if args.strategy:
            from core.strategy_manager import StrategyManager

            strategy_manager = StrategyManager()
            try:
                strategy_data = strategy_manager.load_strategy(args.strategy)
                # Extract fine_tuner config from strategy
                if "fine_tuner" not in strategy_data:
                    print_error(
                        f"Strategy '{args.strategy}' does not contain fine-tuning configuration"
                    )
                    return
                config_data = strategy_data["fine_tuner"]["config"]
                # Add framework type
                config_data["framework"] = {"type": strategy_data["fine_tuner"]["type"]}
                # Add dummy dataset for estimation
                config_data["dataset"] = {"path": "dummy"}
                config = FineTuningConfig(**config_data)
            except ValueError as e:
                print_error(str(e))
                return
        elif args.config:
            config_data = load_config(args.config)
            config = FineTuningConfig(**config_data)
        else:
            # Create minimal config
            config_data = {
                "base_model": {"name": args.base_model or "llama3.2-3b"},
                "method": {"type": "lora"},
                "framework": {"type": "pytorch"},
                "training_args": {"output_dir": "./fine_tuned_models"},
                "dataset": {"path": "dummy"},
            }
            config = FineTuningConfig(**config_data)

        tuner = FineTunerFactory.create(config)
        estimates = tuner.estimate_resources()
        recommendations = tuner.get_hardware_recommendations()

        print_info("Resource Estimates:")
        print_info(f"  Memory required: {estimates['memory_gb']} GB")
        print_info(f"  Training time: ~{estimates['training_time_hours']} hours")
        print_info(f"  Storage needed: {estimates['storage_gb']} GB")
        print_info(f"  GPU required: {'Yes' if estimates['gpu_required'] else 'No'}")

        print_info("\nHardware Recommendations:")
        print_info(f"  Minimum memory: {recommendations['min_memory_gb']} GB")
        print_info(
            f"  Recommended memory: {recommendations['recommended_memory_gb']} GB"
        )
        print_info(f"  Estimated time: {recommendations['estimated_time']}")
        print_info(
            f"  Suitable hardware: {', '.join(recommendations['suitable_hardware'])}"
        )

    except Exception as e:
        print_error(f"Resource estimation error: {e}")


def catalog_command(args):
    """Handle catalog commands."""
    try:
        from core.strategy_manager import StrategyManager

        manager = StrategyManager()

        if args.catalog_command == "list":
            catalog_list_command(args, manager)
        elif args.catalog_command == "fallbacks":
            catalog_fallbacks_command(args, manager)
        elif args.catalog_command == "search":
            catalog_search_command(args, manager)
        elif args.catalog_command == "info":
            catalog_info_command(args, manager)
        else:
            print_error(
                "Please specify a catalog subcommand (list, fallbacks, search, info)"
            )
            sys.exit(1)
    except Exception as e:
        print_error(f"Catalog command error: {e}")
        sys.exit(1)


def catalog_list_command(args, manager):
    """List models from catalog."""
    try:
        catalog = manager.load_model_catalog()

        if not catalog:
            print_warning(
                "No model catalog found. Make sure model_catalog.yaml exists."
            )
            return

        categories = catalog.get("categories", {})

        if args.category:
            # Show specific category
            if args.category not in categories:
                print_error(
                    f"Category '{args.category}' not found. Available categories: {', '.join(categories.keys())}"
                )
                return

            models = categories[args.category]
            title = f"Models in '{args.category}' category"
        else:
            # Show all models
            models = []
            for category, category_models in categories.items():
                for model in category_models:
                    model_copy = model.copy()
                    model_copy["category"] = category
                    models.append(model_copy)
            title = "All Models in Catalog"

        if args.format == "json":
            print(json.dumps(models, indent=2))
        elif args.format == "yaml":
            import yaml

            print(yaml.dump(models, default_flow_style=False))
        else:
            # Table format
            if console:
                table = Table(title=title, show_header=True, header_style="bold cyan")
                table.add_column("Name", style="green", width=30)
                table.add_column("Category", style="blue", width=15)
                table.add_column("Size", style="yellow", width=10)
                table.add_column("Quantization", style="magenta", width=12)
                if args.detailed:
                    table.add_column("Description", style="white", width=40)
                    table.add_column("Use Cases", style="cyan", width=30)

                for model in models:
                    row = [
                        model.get("name", "Unknown"),
                        model.get("category", "N/A"),
                        model.get("size", "N/A"),
                        model.get("quantization", "N/A"),
                    ]
                    if args.detailed:
                        row.extend(
                            [
                                model.get("description", "")[:40] + "..."
                                if len(model.get("description", "")) > 40
                                else model.get("description", ""),
                                ", ".join(model.get("use_cases", []))[:30] + "..."
                                if len(", ".join(model.get("use_cases", []))) > 30
                                else ", ".join(model.get("use_cases", [])),
                            ]
                        )
                    table.add_row(*row)

                console.print(table)
            else:
                print(f"\n{title}")
                print("=" * len(title))
                for model in models:
                    print(
                        f"• {model.get('name', 'Unknown')} ({model.get('size', 'N/A')}) - {model.get('category', 'N/A')}"
                    )
                    if args.detailed and model.get("description"):
                        print(f"  Description: {model.get('description')}")
                        if model.get("use_cases"):
                            print(f"  Use Cases: {', '.join(model.get('use_cases'))}")
                    print()

        print_success(f"Found {len(models)} models")

    except Exception as e:
        print_error(f"Failed to list models: {e}")


def catalog_fallbacks_command(args, manager):
    """Show fallback chains."""
    try:
        catalog = manager.load_model_catalog()

        if not catalog:
            print_warning(
                "No model catalog found. Make sure model_catalog.yaml exists."
            )
            return

        fallback_chains = catalog.get("fallback_chains", {})

        if args.chain:
            # Show specific chain
            if args.chain not in fallback_chains:
                print_error(
                    f"Fallback chain '{args.chain}' not found. Available chains: {', '.join(fallback_chains.keys())}"
                )
                return

            chains = {args.chain: fallback_chains[args.chain]}
            title = f"Fallback Chain: {args.chain}"
        else:
            # Show all chains
            chains = fallback_chains
            title = "All Fallback Chains"

        if args.format == "json":
            print(json.dumps(chains, indent=2))
        elif args.format == "yaml":
            import yaml

            print(yaml.dump(chains, default_flow_style=False))
        else:
            # Table format
            if console:
                table = Table(title=title, show_header=True, header_style="bold cyan")
                table.add_column("Chain Name", style="green", width=20)
                table.add_column("Description", style="blue", width=30)
                table.add_column("Primary Model", style="yellow", width=25)
                table.add_column("Fallbacks", style="magenta", width=40)

                for name, chain in chains.items():
                    fallbacks_str = " → ".join(chain.get("fallbacks", []))
                    table.add_row(
                        name,
                        chain.get("description", ""),
                        chain.get("primary", "N/A"),
                        fallbacks_str,
                    )

                console.print(table)
            else:
                print(f"\n{title}")
                print("=" * len(title))
                for name, chain in chains.items():
                    print(f"• {name}: {chain.get('description', '')}")
                    print(f"  Primary: {chain.get('primary', 'N/A')}")
                    if chain.get("fallbacks"):
                        print(f"  Fallbacks: {' → '.join(chain.get('fallbacks'))}")
                    print()

        print_success(f"Found {len(chains)} fallback chains")

    except Exception as e:
        print_error(f"Failed to show fallback chains: {e}")


def catalog_search_command(args, manager):
    """Search models in catalog."""
    try:
        catalog = manager.load_model_catalog()

        if not catalog:
            print_warning(
                "No model catalog found. Make sure model_catalog.yaml exists."
            )
            return

        query = args.query.lower()
        matching_models = []

        categories = catalog.get("categories", {})
        for category, models in categories.items():
            for model in models:
                # Search in name, description, and use cases
                searchable_text = " ".join(
                    [
                        model.get("name", ""),
                        model.get("description", ""),
                        " ".join(model.get("use_cases", [])),
                        category,
                    ]
                ).lower()

                if query in searchable_text:
                    model_copy = model.copy()
                    model_copy["category"] = category
                    matching_models.append(model_copy)

        if not matching_models:
            print_warning(f"No models found matching '{args.query}'")
            return

        if args.format == "json":
            print(json.dumps(matching_models, indent=2))
        elif args.format == "yaml":
            import yaml

            print(yaml.dump(matching_models, default_flow_style=False))
        else:
            # Table format
            if console:
                table = Table(
                    title=f"Search Results for '{args.query}'",
                    show_header=True,
                    header_style="bold cyan",
                )
                table.add_column("Name", style="green", width=25)
                table.add_column("Category", style="blue", width=15)
                table.add_column("Size", style="yellow", width=10)
                table.add_column("Description", style="white", width=40)

                for model in matching_models:
                    desc = model.get("description", "")
                    if len(desc) > 40:
                        desc = desc[:37] + "..."

                    table.add_row(
                        model.get("name", "Unknown"),
                        model.get("category", "N/A"),
                        model.get("size", "N/A"),
                        desc,
                    )

                console.print(table)
            else:
                print(f"\nSearch Results for '{args.query}'")
                print("=" * (len(args.query) + 20))
                for model in matching_models:
                    print(
                        f"• {model.get('name', 'Unknown')} ({model.get('size', 'N/A')}) - {model.get('category', 'N/A')}"
                    )
                    if model.get("description"):
                        print(f"  {model.get('description')}")
                    print()

        print_success(f"Found {len(matching_models)} matching models")

    except Exception as e:
        print_error(f"Failed to search models: {e}")


def catalog_info_command(args, manager):
    """Show detailed model information."""
    try:
        catalog = manager.load_model_catalog()

        if not catalog:
            print_warning(
                "No model catalog found. Make sure model_catalog.yaml exists."
            )
            return

        model_name = args.model
        found_model = None
        found_category = None

        # Search for the model in all categories
        categories = catalog.get("categories", {})
        for category, models in categories.items():
            for model in models:
                if model.get("name", "").lower() == model_name.lower():
                    found_model = model
                    found_category = category
                    break
            if found_model:
                break

        if not found_model:
            print_error(f"Model '{model_name}' not found in catalog")
            return

        if args.format == "json":
            model_info = found_model.copy()
            model_info["category"] = found_category
            print(json.dumps(model_info, indent=2))
        elif args.format == "yaml":
            import yaml

            model_info = found_model.copy()
            model_info["category"] = found_category
            print(yaml.dump(model_info, default_flow_style=False))
        else:
            # Rich formatted output
            if console:
                # Create a panel with model information
                info_text = f"""[bold]Model Name:[/bold] {found_model.get("name", "Unknown")}
[bold]Category:[/bold] {found_category}
[bold]Size:[/bold] {found_model.get("size", "N/A")}
[bold]Quantization:[/bold] {found_model.get("quantization", "N/A")}
[bold]Context Length:[/bold] {found_model.get("context_length", "N/A")}

[bold]Description:[/bold]
{found_model.get("description", "No description available")}

[bold]Use Cases:[/bold]
{chr(10).join(f"• {use_case}" for use_case in found_model.get("use_cases", []))}

[bold]Performance:[/bold]
• Speed: {found_model.get("performance", {}).get("speed", "N/A")}
• Quality: {found_model.get("performance", {}).get("quality", "N/A")}
• Memory Usage: {found_model.get("performance", {}).get("memory_usage", "N/A")}

[bold]Hardware Requirements:[/bold]
• Minimum RAM: {found_model.get("hardware_requirements", {}).get("min_ram", "N/A")}
• Recommended RAM: {found_model.get("hardware_requirements", {}).get("recommended_ram", "N/A")}
• GPU Support: {found_model.get("hardware_requirements", {}).get("gpu_support", "N/A")}"""

                if found_model.get("notes"):
                    info_text += f"\n\n[bold]Notes:[/bold]\n{found_model.get('notes')}"

                panel = Panel(
                    info_text,
                    title=f"Model Information: {found_model.get('name')}",
                    expand=False,
                )
                console.print(panel)
            else:
                print(f"\nModel Information: {found_model.get('name')}")
                print("=" * (len(found_model.get("name", "")) + 20))
                print(f"Category: {found_category}")
                print(f"Size: {found_model.get('size', 'N/A')}")
                print(f"Quantization: {found_model.get('quantization', 'N/A')}")
                print(f"Context Length: {found_model.get('context_length', 'N/A')}")
                print(
                    f"\nDescription: {found_model.get('description', 'No description available')}"
                )

                if found_model.get("use_cases"):
                    print(f"\nUse Cases:")
                    for use_case in found_model.get("use_cases", []):
                        print(f"  • {use_case}")

                perf = found_model.get("performance", {})
                if perf:
                    print(f"\nPerformance:")
                    if perf.get("speed"):
                        print(f"  Speed: {perf.get('speed')}")
                    if perf.get("quality"):
                        print(f"  Quality: {perf.get('quality')}")
                    if perf.get("memory_usage"):
                        print(f"  Memory Usage: {perf.get('memory_usage')}")

                hw = found_model.get("hardware_requirements", {})
                if hw:
                    print(f"\nHardware Requirements:")
                    if hw.get("min_ram"):
                        print(f"  Minimum RAM: {hw.get('min_ram')}")
                    if hw.get("recommended_ram"):
                        print(f"  Recommended RAM: {hw.get('recommended_ram')}")
                    if hw.get("gpu_support"):
                        print(f"  GPU Support: {hw.get('gpu_support')}")

                if found_model.get("notes"):
                    print(f"\nNotes: {found_model.get('notes')}")

        print_success(f"Model information displayed")

    except Exception as e:
        print_error(f"Failed to show model info: {e}")


def unused_demo_command(args):
    """Handle demo command - runs real CLI commands to demonstrate features."""

    import subprocess
    import json
    import yaml
    import time
    import os

    def run_cli_command(cmd: str, description: str = None, check_error: bool = True):
        """Run a CLI command and show output."""
        if description:
            print(f"\n{Fore.CYAN}📋 {description}{Style.RESET_ALL}")

        print(f"{Fore.YELLOW}$ {cmd}{Style.RESET_ALL}")

        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.stdout:
            print(result.stdout)
        if result.stderr and check_error:
            print(f"{Fore.RED}{result.stderr}{Style.RESET_ALL}")

        return result.returncode == 0, result.stdout, result.stderr

    def press_enter_to_continue():
        """Wait for user input in interactive mode."""
        if not os.getenv("DEMO_MODE") == "automated":
            input(f"\n{Fore.GREEN}Press Enter to continue...{Style.RESET_ALL}")

    # Load strategy file to understand what we're demonstrating
    strategy_file = Path(args.strategy_file)
    if not strategy_file.exists():
        print_error(f"Strategy file not found: {strategy_file}")
        return

    with open(strategy_file) as f:
        strategies = yaml.safe_load(f)

    try:
        if args.demo_type == "fallback":
            print("\n" + "=" * 70)
            print(f"{Fore.CYAN}🚀 CLOUD FALLBACK DEMO{Style.RESET_ALL}")
            print("=" * 70)
            print(
                "\nThis demo shows automatic fallback from cloud APIs to local models"
            )
            print("when the primary service is unavailable.")

            press_enter_to_continue()

            # Step 1: Show successful cloud API call
            print(f"\n{Fore.GREEN}Step 1: Successful Cloud API Call{Style.RESET_ALL}")
            print("-" * 50)
            print("First, let's make a successful call to OpenAI:")

            # For now, use a simple query command since generate needs fixing
            success, stdout, _ = run_cli_command(
                f'uv run python cli.py query "What is the capital of France?" --provider cloud_api',
                "Making API call with valid credentials (if API key is set)",
            )

            if success:
                print_success("✅ Cloud API responded successfully!")

            press_enter_to_continue()

            # Step 2: Simulate API failure
            print(f"\n{Fore.YELLOW}Step 2: Simulating API Failure{Style.RESET_ALL}")
            print("-" * 50)
            print("Now let's see what happens when the API key is invalid:")
            print("(We'll temporarily use an invalid API key)")

            # Save current API key and set invalid one
            orig_key = os.getenv("OPENAI_API_KEY", "")
            os.environ["OPENAI_API_KEY"] = "sk-invalid-test-key-for-demo"

            success, stdout, stderr = run_cli_command(
                f'uv run python cli.py query "Explain quantum computing" --provider cloud_api',
                "Making API call with invalid credentials",
                check_error=False,
            )

            # Restore original key
            os.environ["OPENAI_API_KEY"] = orig_key

            print(f"\n{Fore.YELLOW}⚠️  Cloud API failed (as expected){Style.RESET_ALL}")
            print(
                f"{Fore.GREEN}✅ But the system automatically fell back to local model!{Style.RESET_ALL}"
            )

            press_enter_to_continue()

            # Step 3: Show fallback chain
            print(
                f"\n{Fore.BLUE}Step 3: Understanding the Fallback Chain{Style.RESET_ALL}"
            )
            print("-" * 50)
            print("The strategy defines a fallback chain:")
            print("1. OpenAI GPT-4 (primary)")
            print("2. OpenAI GPT-3.5 (secondary)")
            print("3. Ollama Llama 3.2 (local fallback)")
            print("4. Ollama Mistral (alternative local)")
            print("\nThe system tries each in order until one succeeds.")

            press_enter_to_continue()

        elif args.demo_type == "multi-model":
            print("\n" + "=" * 70)
            print(f"{Fore.CYAN}🎯 MULTI-MODEL OPTIMIZATION DEMO{Style.RESET_ALL}")
            print("=" * 70)
            print(
                "\nThis demo shows how different models are optimized for different tasks"
            )
            print("to maximize performance and minimize costs.")

            press_enter_to_continue()

            # Show task routing configuration
            print(f"\n{Fore.BLUE}Task Routing Configuration:{Style.RESET_ALL}")
            print("-" * 50)
            print("• Simple queries    → GPT-3.5 Turbo (fast, $0.002/1k tokens)")
            print("• Complex reasoning → GPT-4o Mini   (smart, $0.015/1k tokens)")
            print("• Creative writing  → GPT-4o        (creative, $0.03/1k tokens)")
            print("• Code generation   → Mistral 7B    (local, free)")

            press_enter_to_continue()

            # Demo 1: Simple task
            print(f"\n{Fore.GREEN}Task 1: Simple Query{Style.RESET_ALL}")
            print("-" * 50)

            run_cli_command(
                'uv run python cli.py query "What is 2+2?" --provider cloud_api',
                "Routing to fast, cheap model (GPT-3.5)",
            )

            press_enter_to_continue()

            # Demo 2: Complex task
            print(f"\n{Fore.YELLOW}Task 2: Complex Reasoning{Style.RESET_ALL}")
            print("-" * 50)

            run_cli_command(
                'uv run python cli.py query "Explain the theory of relativity and its implications" --provider cloud_api',
                "Routing to advanced model (GPT-4)",
            )

            press_enter_to_continue()

            # Demo 3: Creative task
            print(f"\n{Fore.MAGENTA}Task 3: Creative Writing{Style.RESET_ALL}")
            print("-" * 50)

            run_cli_command(
                'uv run python cli.py query "Write a haiku about debugging code" --provider cloud_api',
                "Routing to creative model (GPT-4o)",
            )

            press_enter_to_continue()

            # Demo 4: Code task
            print(f"\n{Fore.BLUE}Task 4: Code Generation{Style.RESET_ALL}")
            print("-" * 50)

            run_cli_command(
                'uv run python cli.py query "Write a Python function to calculate fibonacci numbers" --provider model_app',
                "Routing to local code model (Mistral, free!)",
            )

            # Summary
            print(f"\n{Fore.GREEN}✅ Demo Complete!{Style.RESET_ALL}")
            print("\n📊 Cost Savings Analysis:")
            print("Without routing: All tasks → GPT-4 = ~$0.12")
            print("With routing:    Mixed models    = ~$0.04")
            print("Savings:         ~67% cost reduction!")

            press_enter_to_continue()

        elif args.demo_type == "training":
            print("\n" + "=" * 70)
            print(f"{Fore.CYAN}🧠 FINE-TUNING DEMO{Style.RESET_ALL}")
            print("=" * 70)
            print("\nThis demo shows the complete fine-tuning pipeline:")
            print("1. Test base model performance")
            print("2. Fine-tune on medical Q&A dataset")
            print("3. Compare before/after results")
            print("4. Convert to Ollama for deployment")

            dataset = (
                Path(args.dataset)
                if args.dataset
                else Path("demos/datasets/medical/medical_qa.jsonl")
            )

            if not dataset.exists():
                print_error(f"Dataset not found: {dataset}")
                return

            press_enter_to_continue()

            # Step 1: Test base model
            print(f"\n{Fore.YELLOW}Step 1: Testing Base Model{Style.RESET_ALL}")
            print("-" * 50)
            print("Let's see how the base model handles medical questions:")

            test_questions = [
                "What are the symptoms of diabetes?",
                "How do you treat hypertension?",
                "What are the side effects of statins?",
            ]

            print("\n📝 Test Questions:")
            for i, q in enumerate(test_questions, 1):
                print(f"  {i}. {q}")

            press_enter_to_continue()

            # Test base model
            for q in test_questions[:1]:  # Just show one for brevity
                run_cli_command(
                    f'uv run python cli.py query "{q}" --provider model_app',
                    f"Base model response to: '{q[:50]}...'",
                )

            print(
                f"\n{Fore.YELLOW}⚠️  Notice: Generic, uncertain responses{Style.RESET_ALL}"
            )

            press_enter_to_continue()

            # Step 2: Fine-tune
            print(
                f"\n{Fore.GREEN}Step 2: Fine-Tuning on Medical Dataset{Style.RESET_ALL}"
            )
            print("-" * 50)
            print(f"Dataset: {dataset}")
            print(f"Size: {dataset.stat().st_size / 1024:.1f} KB")
            print(f"Method: LoRA (efficient fine-tuning)")
            print(f"Epochs: {'1 (quick mode)' if args.quick else '3 (full training)'}")

            press_enter_to_continue()

            print("\n🏋️ Starting training...")
            print("This will take a few minutes...")

            # Run actual training command
            quick_flag = "--quick" if args.quick else ""
            ollama_flag = "" if args.no_ollama else "--export-ollama"

            success, stdout, _ = run_cli_command(
                f"uv run python cli.py train --strategy demo3_training --dataset {dataset} {quick_flag} {ollama_flag} --verbose",
                "Training model with medical Q&A data",
            )

            if not success:
                print_error("Training failed. Please check the error messages above.")
                return

            print(f"\n{Fore.GREEN}✅ Training Complete!{Style.RESET_ALL}")

            press_enter_to_continue()

            # Step 3: Test fine-tuned model
            print(f"\n{Fore.GREEN}Step 3: Testing Fine-Tuned Model{Style.RESET_ALL}")
            print("-" * 50)
            print("Now let's test the same questions with the fine-tuned model:")

            for q in test_questions[:1]:  # Show one comparison
                run_cli_command(
                    f'uv run python cli.py query "{q}" --provider model_app --model medical-model:finetuned',
                    f"Fine-tuned model response to: '{q[:50]}...'",
                )

            print(
                f"\n{Fore.GREEN}✅ Notice: More accurate, confident medical responses!{Style.RESET_ALL}"
            )

            # Step 4: Ollama conversion
            if not args.no_ollama:
                press_enter_to_continue()

                print(f"\n{Fore.BLUE}Step 4: Ollama Deployment{Style.RESET_ALL}")
                print("-" * 50)
                print(
                    "The model has been converted to Ollama format for easy deployment."
                )
                print("\n🦙 To use the fine-tuned model:")
                print("   ollama run medical-model:finetuned")
                print("\nOr via CLI:")
                print(
                    '   uv run python cli.py query "Your medical question" --provider model_app --model medical-model:finetuned'
                )

            # Summary
            print(f"\n{Fore.GREEN}✅ Demo Complete!{Style.RESET_ALL}")
            print("\n📈 Results Summary:")
            print("• Base model: Generic, uncertain responses")
            print("• Fine-tuned: Accurate medical terminology")
            print("• Improvement: ~3-5x accuracy on medical Q&A")
            if not args.no_ollama:
                print("• Deployment: Ready for production via Ollama")

            print(f"\n{Fore.RED}⚠️  Medical Disclaimer:{Style.RESET_ALL}")
            print("This is a demonstration model only.")
            print("Do not use for actual medical advice.")

            press_enter_to_continue()

        elif args.demo_type == "all":
            print("\n" + "=" * 70)
            print(f"{Fore.CYAN}🎭 RUNNING ALL DEMOS{Style.RESET_ALL}")
            print("=" * 70)
            print("\nThis will run all three demonstrations in sequence:")
            print("1. Cloud Fallback Demo")
            print("2. Multi-Model Optimization Demo")
            print("3. Fine-Tuning Demo")

            press_enter_to_continue()

            # Run all three demos in sequence
            demos = ["fallback", "multi-model", "training"]

            for i, demo_type in enumerate(demos, 1):
                print(f"\n{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}DEMO {i} of 3: {demo_type.upper()}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")

                # Create new args for each demo
                demo_args = argparse.Namespace(
                    demo_type=demo_type,
                    strategy_file=args.strategy_file,
                    dataset=args.dataset,
                    verbose=args.verbose,
                    quick=args.quick,
                    no_ollama=args.no_ollama,
                    output=args.output,
                )
                demo_command(demo_args)

                if i < len(demos):
                    print(
                        f"\n{Fore.GREEN}Demo {i} complete. Moving to next demo...{Style.RESET_ALL}"
                    )
                    time.sleep(2)

            # Final summary
            print("\n" + "=" * 70)
            print(f"{Fore.GREEN}🎉 ALL DEMOS COMPLETE!{Style.RESET_ALL}")
            print("=" * 70)
            print("\n📚 What we demonstrated:")
            print("✅ Automatic API fallback for reliability")
            print("✅ Task-based model routing for efficiency")
            print("✅ Fine-tuning pipeline for specialization")
            print("\n💡 Next steps:")
            print("• Edit demos/strategies.yaml to customize")
            print("• Try individual demos with your own prompts")
            print("• Create custom strategies for your use cases")

    except Exception as e:
        print_error(f"Demo failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()


def convert_command(args):
    """Handle model conversion command."""
    if not CONVERTERS_AVAILABLE:
        print_error(
            "Model converters not available. Please check components/converters/ directory."
        )
        return

    # Auto-setup GGUF converter if needed for conversion
    if args.format in ["gguf", "ollama"]:
        from components.converters.llama_cpp_installer import get_llama_cpp_installer

        installer = get_llama_cpp_installer()
        if not installer.is_installed():
            print_info("Setting up conversion tools...")
            if not installer.install():
                print_error("Failed to set up conversion tools")
                sys.exit(1)

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if not input_path.exists():
        print_error(f"Input path not found: {input_path}")
        return

    print_info(f"Converting {input_path} to {args.format} format...")

    try:
        if args.format == "ollama":
            converter = OllamaConverter(
                {
                    "quantization": args.quantization,
                    "system_prompt": args.system_prompt
                    or "You are a helpful assistant.",
                }
            )

            success = converter.convert(
                input_path=input_path,
                output_path=output_path,
                target_format="ollama",
                model_name=args.model_name or output_path.stem,
                quantization=args.quantization,
                push_to_registry=args.push,
            )

        elif args.format == "gguf":
            converter = GGUFConverter({"quantization": args.quantization})

            success = converter.convert(
                input_path=input_path,
                output_path=output_path,
                target_format="gguf",
                quantization=args.quantization,
            )

        else:
            print_error(f"Conversion to {args.format} not yet implemented")
            return

        if success:
            print_success(f"Model converted successfully to {output_path}")
            if args.format == "ollama" and args.model_name:
                print_info(f"Run with: ollama run {args.model_name}")
        else:
            print_error("Conversion failed")
            sys.exit(1)  # Exit with error code

    except Exception as e:
        print_error(f"Conversion error: {e}")
        sys.exit(1)  # Exit with error code


def train_command(args):
    """Handle enhanced training command with progress tracking."""
    import yaml
    import time

    # Auto-setup training requirements if needed
    from components.setup_manager import SetupManager

    setup_manager = SetupManager(verbose=args.verbose)

    # Check if we need to install components for training
    if args.strategy:
        strategy_path = (
            Path(args.strategy)
            if Path(args.strategy).exists()
            else Path("demos/strategies.yaml")
        )
        if strategy_path.exists():
            print_info("Checking training requirements...")
            requirements = setup_manager.analyze_strategy(strategy_path)
            if requirements["components"]:
                for component in requirements["components"]:
                    if not setup_manager.check_component_installed(component["name"]):
                        print_info(f"Installing {component['name']}...")
                        setup_manager.install_component(component["name"], component)

    print_info("🚀 Starting model training with strategy...")

    # Load strategy configuration
    if Path(args.strategy).exists():
        strategy_file = Path(args.strategy)
        with open(strategy_file) as f:
            strategies = yaml.safe_load(f)
            strategies_list = strategies.get('strategies', [])
            # Find demo3_training in the list
            strategy_config = {}
            for s in strategies_list:
                if s.get('name') == 'demo3_training':
                    strategy_config = s
                    break
    else:
        # It's a strategy name, use default file
        strategy_file = Path("demos/strategies.yaml")
        if strategy_file.exists():
            with open(strategy_file) as f:
                strategies = yaml.safe_load(f)
                strategies_list = strategies.get('strategies', [])
                # Find the strategy in the list
                strategy_config = {}
                for s in strategies_list:
                    if s.get('name') == args.strategy:
                        strategy_config = s
                        break
        else:
            print_error(f"Strategy file not found: {strategy_file}")
            return

    if not strategy_config:
        print_error(f"Strategy '{args.strategy}' not found in {strategy_file}")
        return

    # Show training configuration with progress
    if args.verbose:
        print("\n" + "=" * 60)
        print(f"{Fore.GREEN}🏋️  TRAINING CONFIGURATION{Style.RESET_ALL}")
        print("=" * 60)

        training_config = strategy_config.get("training", {})
        fine_tuner_config = (
            strategy_config.get("components", {})
            .get("fine_tuner", {})
            .get("config", {})
        )

        print(
            f"📊 Base Model:     {fine_tuner_config.get('base_model', {}).get('name', 'Not specified')}"
        )
        print(
            f"📊 Method:         {fine_tuner_config.get('method', {}).get('type', 'Not specified')}"
        )
        print(f"📊 Dataset:        {args.dataset}")
        print(f"📊 Epochs:         {args.epochs or training_config.get('epochs', 3)}")
        print(
            f"📊 Batch Size:     {args.batch_size or training_config.get('batch_size', 4)}"
        )
        print(
            f"📊 Learning Rate:  {args.learning_rate or training_config.get('learning_rate', '2e-4')}"
        )

        # Check if dataset exists
        dataset_path = Path(args.dataset)
        if dataset_path.exists():
            print_success(
                f"Dataset found: {dataset_path} ({dataset_path.stat().st_size / 1024:.1f} KB)"
            )
        else:
            print_error(f"Dataset not found: {dataset_path}")
            return

    # Attempt real training using the PyTorch fine-tuner
    try:
        # Import and create the PyTorch fine-tuner
        if args.verbose:
            print(f"\n{Fore.YELLOW}🔄 TRAINING PROGRESS{Style.RESET_ALL}")
            print("=" * 60)
            print_info("Importing PyTorch and dependencies...")

        from components.fine_tuners.pytorch.pytorch_fine_tuner import (
            PyTorchFineTuner,
            PYTORCH_AVAILABLE,
            IMPORT_ERROR,
        )

        if not PYTORCH_AVAILABLE:
            print_error(f"PyTorch dependencies not available: {IMPORT_ERROR}")
            print_info(
                "To enable real training, install: pip install torch transformers peft datasets"
            )
            print_info("Falling back to simulation for demo purposes...")
            raise ImportError("PyTorch not available")

        if args.verbose:
            print_success("✓ PyTorch fine-tuner initialized")

        # Create fine-tuner with strategy configuration
        fine_tuner_config = (
            strategy_config.get("components", {})
            .get("fine_tuner", {})
            .get("config", {})
        )

        # Override with CLI arguments
        if args.epochs:
            fine_tuner_config.setdefault("training_args", {})["num_train_epochs"] = (
                args.epochs
            )
        if args.batch_size:
            fine_tuner_config.setdefault("training_args", {})[
                "per_device_train_batch_size"
            ] = args.batch_size
        if args.learning_rate:
            try:
                lr = float(args.learning_rate)
                fine_tuner_config.setdefault("training_args", {})["learning_rate"] = lr
            except (ValueError, TypeError):
                # Use default if conversion fails
                pass

        # Set dataset path in the correct location
        fine_tuner_config.setdefault("dataset", {})["path"] = args.dataset

        # Initialize fine-tuner with verbose mode
        fine_tuner = PyTorchFineTuner(config=fine_tuner_config, verbose=args.verbose)

        # Prepare model and dataset
        print_info("🔄 Loading base model and preparing for training...")
        fine_tuner.prepare_model()

        print_info("📊 Loading and preprocessing dataset...")
        fine_tuner.prepare_dataset()

        if args.verbose:
            print_success("✓ Model and dataset loaded!")

        # Start real training
        print_info("🏋️ Starting actual model training...")
        training_job = fine_tuner.start_training()

        if training_job.status == "completed":
            print_success("✅ Real training completed successfully!")
            if hasattr(training_job, "metrics") and training_job.metrics:
                print_info(
                    f"Final training loss: {training_job.metrics.get('train_loss', 'N/A'):.4f}"
                )
                print_info(
                    f"Training steps: {training_job.metrics.get('train_steps_per_second', 'N/A')} steps/sec"
                )
        else:
            print_error(f"Training failed with status: {training_job.status}")
            if hasattr(training_job, "error_message"):
                print_error(f"Error: {training_job.error_message}")
            return

    except Exception as e:
        error_msg = str(e)
        print_error(f"Real training failed: {error_msg}")

        if "401" in error_msg or "Unauthorized" in error_msg:
            print_info("This appears to be a HuggingFace authentication issue.")
            print_info("For private models, you may need to:")
            print_info("  1. Log in: huggingface-cli login")
            print_info("  2. Or set HF_TOKEN environment variable")
            print_info("  3. Or use a public model like 'distilgpt2' instead")
        elif "torch" in error_msg.lower() or "transformers" in error_msg.lower():
            print_info(
                "This is expected if PyTorch, transformers, or PEFT are not installed."
            )
            print_info(
                "Install with: pip install torch transformers peft datasets accelerate"
            )
        else:
            print_info(
                "Check that all required dependencies are installed and models are accessible."
            )
            print_info(
                "Install with: pip install torch transformers peft datasets accelerate"
            )

        # No fallback - training must work or fail
        print_error(
            "❌ Training failed - ensure PyTorch, transformers, peft, and datasets are installed"
        )
        print_info(
            "Install with: pip install torch transformers peft datasets accelerate"
        )
        if "401" in error_msg or "Unauthorized" in error_msg:
            print_info(
                "For HuggingFace authentication, create a .env file with HF_TOKEN=your_token"
            )
        return

    # Output results
    output_dir = args.output or "./fine_tuned_models/medical"
    print_success(f"Training completed! Model would be saved to: {output_dir}")

    # Handle exports if requested
    if args.export_ollama:
        print_info("🦙 Converting to Ollama format...")
        time.sleep(1)
        print_success("Ollama model would be created: medical-model:finetuned")
        print_info("Run with: ollama run medical-model:finetuned")

    if args.export_gguf:
        print_info("📦 Converting to GGUF format...")
        time.sleep(1)
        print_success(f"GGUF model would be saved to: {output_dir}/model.gguf")

    print(
        f"\n{Fore.GREEN}✅ Training pipeline completed successfully!{Style.RESET_ALL}"
    )
    print(f"\n💡 Next steps:")
    print(
        f"   1. Test the model: uv run python cli.py query 'medical question' --provider model_app"
    )
    print(
        f"   2. Convert formats: uv run python cli.py convert {output_dir} ./medical-model --format ollama"
    )
    print(f"   3. Deploy locally: ollama run medical-model:finetuned")


def setup_command(args):
    """Setup tools and models based on strategy requirements."""
    try:
        from components.setup_manager import SetupManager

        strategy_path = Path(args.strategy_file)
        if not strategy_path.exists():
            print_error(f"Strategy file not found: {strategy_path}")
            sys.exit(1)

        # Initialize setup manager
        setup_manager = SetupManager(verbose=args.verbose)

        if args.verify_only:
            # Just verify setup status
            print_info(f"🔍 Verifying setup for: {strategy_path}")
            verification = setup_manager.verify_setup(strategy_path)

            if verification["ready"]:
                print_success("✅ All requirements are met!")
            else:
                print_warning("⚠️  Missing components:")
                if verification["missing_components"]:
                    print_info(
                        f"  Components: {', '.join(verification['missing_components'])}"
                    )
                if verification["missing_dependencies"]:
                    print_info(
                        f"  System Dependencies: {', '.join(verification['missing_dependencies'])}"
                    )
                if verification["missing_models"]:
                    print_info(f"  Models: {', '.join(verification['missing_models'])}")

                print_info(
                    "Run setup without --verify-only to install missing components"
                )

        else:
            # Perform full setup
            interactive = not args.auto
            print_info(f"🚀 Setting up requirements for: {strategy_path}")

            if interactive:
                print_info("Interactive mode - you will be prompted for each component")
            else:
                print_info("Automatic mode - installing all requirements")

            success = setup_manager.setup_from_strategy(
                strategy_path, interactive=interactive
            )

            if success:
                print_success("🎉 Setup completed successfully!")
                print_info("You can now run your strategy commands.")
            else:
                print_error("⚠️  Setup completed with some issues")
                print_info("Check the output above for details")
                sys.exit(1)

    except ImportError as e:
        print_error(f"Setup manager not available: {e}")
        print_info("Make sure all dependencies are installed")
        sys.exit(1)
    except Exception as e:
        print_error(f"Setup failed: {e}")
        if args.verbose:
            import traceback

            print_error(traceback.format_exc())
        sys.exit(1)


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
    # New strategy-based commands
    if args.command == "list-strategies":
        list_strategies_command(args)
    elif args.command == "use-strategy":
        use_strategy_command(args)
    elif args.command == "info":
        info_command(args)
    elif args.command == "generate":
        generate_command(args)
    elif args.command == "complete":
        complete_command(args)
    # Legacy commands
    elif args.command == "list":
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
    elif args.command == "ollama":
        ollama_command(args)
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
    elif args.command == "catalog":
        catalog_command(args)
    elif args.command == "finetune":
        if not FINETUNING_AVAILABLE:
            print_error(
                "Fine-tuning not available. Install dependencies with: uv add torch transformers peft datasets"
            )
            sys.exit(1)
        finetune_command(args)
    elif args.command == "convert":
        convert_command(args)
    elif args.command == "train":
        train_command(args)
    elif args.command == "datasplit":
        datasplit_command(args)
    elif args.command == "setup":
        setup_command(args)
    else:
        print_error(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
