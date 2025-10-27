#!/usr/bin/env python3
"""
Model Download Utility for Transformers Runtime

Downloads and caches HuggingFace models for use with the transformers runtime.
Models are cached in ~/.cache/huggingface/hub/ for fast subsequent loads.

Usage:
    python download_model.py MODEL_ID [--text|--image]

Examples:
    python download_model.py google/gemma-3-1b-it --text
    python download_model.py microsoft/phi-2 --text
    python download_model.py stabilityai/stable-diffusion-xl-base-1.0 --image
"""

import sys
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import DiffusionPipeline


def download_text_model(model_id: str, device: str = "auto"):
    """Download a text generation model."""
    print(f"\n{'=' * 60}")
    print(f"Downloading Text Model: {model_id}")
    print(f"{'=' * 60}\n")

    # Check for unsupported quantization formats on macOS/MPS
    if (
        "bnb" in model_id.lower()
        or "4bit" in model_id.lower()
        or "8bit" in model_id.lower()
    ):
        import platform

        if platform.system() == "Darwin":  # macOS
            print(f"⚠️  Warning: This model appears to use bitsandbytes quantization")
            print(f"   bitsandbytes is not supported on macOS/Apple Silicon")
            print(f"\nRecommended alternatives for macOS:")
            print(f"  1. Use standard (non-quantized) models:")
            print(f"     - Qwen/Qwen2.5-0.5B-Instruct")
            print(f"     - Qwen/Qwen2.5-1.5B-Instruct")
            print(f"  2. Use FP8 quantized models (compatible):")
            print(f"     - qwen-community/Qwen3-0.6B-FP8")
            print(f"  3. Use GGUF models with Lemonade runtime (recommended)")
            print(f"\nContinuing anyway, but download may fail...\n")

    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Target device: {device}")
    dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32

    # Download tokenizer
    print("\n1. Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print(f"   ✓ Tokenizer downloaded")
    print(f"   Vocab size: {tokenizer.vocab_size:,}")

    # Download model
    print("\n2. Downloading model...")
    print(f"   Using dtype: {dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if device != "cpu" else None,
        low_cpu_mem_usage=True,
    )

    if device == "cpu":
        model = model.to(device)

    print(f"   ✓ Model downloaded")

    # Model stats
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'=' * 60}")
    print(f"✅ Model Ready!")
    print(f"{'=' * 60}")
    print(f"Model ID: {model_id}")
    print(f"Parameters: {total_params / 1e9:.2f}B")
    print(f"Device: {device}")
    print(f"Cache: ~/.cache/huggingface/hub/")
    print(f"\nTest with:")
    print(f"  curl -X POST http://localhost:11540/v1/chat/completions \\")
    print(f'    -H "Content-Type: application/json" \\')
    print(
        f'    -d \'{{"model": "{model_id}", "messages": [{{"role": "user", "content": "Hello!"}}]}}\''
    )
    print(f"{'=' * 60}\n")


def download_diffusion_model(model_id: str, device: str = "auto"):
    """Download a diffusion model for image generation."""
    print(f"\n{'=' * 60}")
    print(f"Downloading Diffusion Model: {model_id}")
    print(f"{'=' * 60}\n")

    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Target device: {device}")
    dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32

    # Download pipeline
    print("\n1. Downloading diffusion pipeline...")
    print(f"   Using dtype: {dtype}")

    # Try safetensors first, fall back to regular weights
    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            dtype=dtype,
            trust_remote_code=True,
            use_safetensors=True,
        )
    except (OSError, ValueError) as e:
        if "safetensors" in str(e).lower():
            print(f"   Note: Model doesn't have safetensors, using standard weights")
            pipe = DiffusionPipeline.from_pretrained(
                model_id,
                dtype=dtype,
                trust_remote_code=True,
                use_safetensors=False,
            )
        else:
            raise

    pipe = pipe.to(device)

    print(f"   ✓ Pipeline downloaded")

    # Pipeline info
    print(f"\n{'=' * 60}")
    print(f"✅ Model Ready!")
    print(f"{'=' * 60}")
    print(f"Model ID: {model_id}")
    print(f"Pipeline: {pipe.__class__.__name__}")
    print(f"Device: {device}")
    print(f"Cache: ~/.cache/huggingface/hub/")
    print(f"\nTest with:")
    print(f"  curl -X POST http://localhost:11540/v1/images/generations \\")
    print(f'    -H "Content-Type: application/json" \\')
    print(
        f'    -d \'{{"prompt": "mountain sunset", "model": "{model_id}", "size": "512x512"}}\''
    )
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download HuggingFace models for transformers runtime",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download text models
  python download_model.py google/gemma-3-1b-it --text
  python download_model.py microsoft/phi-2 --text
  python download_model.py TinyLlama/TinyLlama-1.1B-Chat-v1.0 --text

  # Download image models
  python download_model.py stabilityai/stable-diffusion-xl-base-1.0 --image
  python download_model.py runwayml/stable-diffusion-v1-5 --image

  # Auto-detect model type (uses --text by default)
  python download_model.py google/gemma-3-1b-it
        """,
    )

    parser.add_argument(
        "model_id", help="HuggingFace model ID (e.g., google/gemma-3-1b-it)"
    )

    model_type = parser.add_mutually_exclusive_group()
    model_type.add_argument(
        "--text",
        action="store_true",
        help="Download as text generation model (default)",
    )
    model_type.add_argument(
        "--image",
        action="store_true",
        help="Download as image generation (diffusion) model",
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Target device (default: auto-detect)",
    )

    args = parser.parse_args()

    # Default to text model if not specified
    is_text = args.text or not args.image

    try:
        if is_text:
            download_text_model(args.model_id, args.device)
        else:
            download_diffusion_model(args.model_id, args.device)
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ Error downloading model: {error_msg}", file=sys.stderr)
        print(f"\nTroubleshooting:")
        print(f"  1. Check model ID is correct: https://huggingface.co/{args.model_id}")
        print(f"  2. For gated models, set HF_TOKEN: export HF_TOKEN=hf_xxxxx")
        print(f"  3. Ensure sufficient disk space (~2-10GB per model)")

        # Specific error handling for bitsandbytes
        if "bitsandbytes" in error_msg.lower():
            import platform

            if platform.system() == "Darwin":
                print(
                    f"\n⚠️  bitsandbytes quantization is NOT supported on macOS/Apple Silicon"
                )
                print(f"\nRecommended models for macOS:")
                print(f"  Standard models:")
                print(
                    f"    uv run python download_model.py Qwen/Qwen2.5-0.5B-Instruct --text"
                )
                print(
                    f"    uv run python download_model.py Qwen/Qwen2.5-1.5B-Instruct --text"
                )
                print(f"  FP8 quantized (compatible):")
                print(
                    f"    uv run python download_model.py qwen-community/Qwen3-0.6B-FP8 --text"
                )
                print(
                    f"  Or use GGUF models with Lemonade runtime (best for Apple Silicon)"
                )

        sys.exit(1)


if __name__ == "__main__":
    main()
