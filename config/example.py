#!/usr/bin/env python3
"""
Example script demonstrating the LlamaFarm configuration loader.
"""

import sys
from pathlib import Path

# Add the config module to the path
sys.path.insert(0, str(Path(__file__).parent))

from config_types import LlamaFarmConfig
from loader import ConfigError, find_config_file, load_config


def main():
    """Demonstrate configuration loading."""
    print("LlamaFarm Configuration Loader Example")
    print("=" * 40)

    # Try to find a configuration file
    config_file = find_config_file()

    if config_file:
        print(f"Found configuration file: {config_file}")

        try:
            # Load and validate the configuration
            config: LlamaFarmConfig = load_config()

            print("\nConfiguration loaded successfully!")
            print(f"Version: {config['version']}")
            print(f"Number of models: {len(config['models'])}")

            # Display model information
            print("\nModels:")
            for i, model in enumerate(config["models"], 1):
                print(f"  {i}. {model['provider']}: {model['model']}")

            # Display RAG configuration
            print("\nRAG Configuration:")
            rag = config["rag"]
            print(f"  Parser: {rag['parser']['type']}")
            print(
                f"  Embedder: {rag['embedder']['type']} ({rag['embedder']['config']['model']})"
            )
            print(f"  Vector Store: {rag['vector_store']['type']}")

            # Display prompts if available
            if "prompts" in config and config["prompts"]:
                print(f"\nPrompts: {len(config['prompts'])} defined")
                for prompt in config["prompts"]:
                    name = prompt.get("name", "Unnamed")
                    print(f"  - {name}")
            else:
                print("\nNo prompts defined")

        except ConfigError as e:
            print(f"\nError loading configuration: {e}")
            return 1

    else:
        print("No configuration file found in current directory")
        print("Looking for: llamafarm.yaml, llamafarm.yml, or llamafarm.toml")

        # Create a sample configuration
        print("\nCreating a sample configuration file...")
        create_sample_config()
        print("Sample configuration created as 'llamafarm.yaml'")
        print("Run this script again to load it!")

    return 0


def create_sample_config():
    """Create a sample configuration file."""
    sample_config = """version: v1

prompts:
  - name: "customer_support"
    prompt: "You are a helpful customer support assistant. Answer questions politely and accurately."
    description: "Default customer support prompt"

rag:
  parser:
    type: CustomerSupportCSVParser
    config:
      content_fields: ["question", "answer"]
      metadata_fields: ["category", "timestamp"]

  embedder:
    type: OllamaEmbedder
    config:
      model: "mxbai-embed-large"
      batch_size: 32

  vector_store:
    type: ChromaStore
    config:
      collection_name: "customer_support"
      persist_directory: "./data/chroma"

models:
  - provider: "local"
    model: "llama3.1:8b"
  - provider: "openai"
    model: "gpt-4"
"""

    with open("llamafarm.yaml", "w") as f:
        f.write(sample_config)


if __name__ == "__main__":
    sys.exit(main())
