#!/usr/bin/env python3
"""
Update all model family YAMLs to include runtime, format, and download_command fields
"""
import yaml
import sys
from pathlib import Path

def update_provider(provider_name, provider_config):
    """Update a single provider config with required fields"""
    # Add runtime (same as provider name)
    if 'runtime' not in provider_config:
        provider_config['runtime'] = provider_name

    # Add format based on provider
    if 'format' not in provider_config:
        if provider_name == 'universal':
            provider_config['format'] = 'transformers'
        elif provider_name == 'ollama':
            provider_config['format'] = 'gguf'
        elif provider_name == 'lemonade':
            # Infer from backend or recipe, default to gguf
            backend = provider_config.get('backend') or provider_config.get('recipe')
            if backend == 'transformers':
                provider_config['format'] = 'transformers'
            elif backend == 'onnx':
                provider_config['format'] = 'onnx'
            else:
                provider_config['format'] = 'gguf'
        elif provider_name == 'openai':
            provider_config['format'] = 'api'

    # Add download_command
    if 'download_command' not in provider_config:
        if provider_name == 'universal':
            provider_config['download_command'] = 'Auto-downloads from HuggingFace on first use'
            if 'notes' not in provider_config:
                provider_config['notes'] = 'Auto-downloads from HuggingFace on first use'
        elif provider_name == 'ollama':
            # Replace pull_command with download_command
            if 'pull_command' in provider_config:
                provider_config['download_command'] = provider_config['pull_command']
                del provider_config['pull_command']
            elif 'model_id' in provider_config:
                provider_config['download_command'] = f"ollama pull {provider_config['model_id']}"
            if 'notes' not in provider_config:
                provider_config['notes'] = 'GGUF quantized for efficient local inference'
        elif provider_name == 'lemonade':
            # Build lemonade download command
            model_id = provider_config.get('model_id', 'ModelName')
            checkpoint = provider_config.get('checkpoint', 'checkpoint')
            recipe = provider_config.get('recipe') or provider_config.get('backend', 'llamacpp')
            provider_config['download_command'] = f"uv run lemonade-server-dev pull {model_id} --checkpoint {checkpoint} --recipe {recipe}"
            if 'notes' not in provider_config:
                if recipe == 'llamacpp':
                    provider_config['notes'] = 'Hardware-optimized (NPU/GPU). GGUF format.'
                elif recipe == 'transformers':
                    provider_config['notes'] = 'Hardware-optimized (NPU/GPU). Transformers format.'
                elif recipe == 'onnx':
                    provider_config['notes'] = 'Hardware-optimized for NPU/GPU. ONNX format.'

    return provider_config

def update_yaml_file(file_path):
    """Update a single YAML file"""
    print(f"Updating {file_path.name}...")

    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)

    if 'variants' not in data:
        print(f"  ⚠️  No variants found, skipping")
        return

    updated_count = 0
    for variant in data['variants']:
        if 'providers' not in variant:
            continue

        for provider_name, provider_config in variant['providers'].items():
            old_config = dict(provider_config)
            updated_config = update_provider(provider_name, provider_config)
            if updated_config != old_config:
                updated_count += 1

    # Write back with proper formatting
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True, width=120)

    print(f"  ✓ Updated {updated_count} provider configs")

def main():
    models_dir = Path(__file__).parent / 'text-generation'

    # Skip qwen3 and deepseek (already updated)
    skip_files = {'qwen3.yaml', 'deepseek.yaml'}

    yaml_files = [f for f in models_dir.glob('*.yaml') if f.name not in skip_files]

    print(f"Updating {len(yaml_files)} model family files...\n")

    for yaml_file in sorted(yaml_files):
        try:
            update_yaml_file(yaml_file)
        except Exception as e:
            print(f"  ❌ Error: {e}")

    print(f"\n✅ Done! Updated {len(yaml_files)} files")

if __name__ == '__main__':
    main()
