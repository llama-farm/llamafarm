"""Helper utilities for fine-tuning: templates, compatibility, estimation."""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def get_supported_templates() -> list[str]:
    """Get list of supported chat templates.
    
    Returns:
        List of template names
    """
    return [
        "llama-3",
        "llama-2",
        "chatml",
        "gemma",
        "qwen",
        "phi",
        "mistral",
        "zephyr",
        "vicuna",
        "alpaca",
    ]


def check_template_compatibility(model_name: str) -> dict:
    """Auto-detect best chat template for a model.
    
    Args:
        model_name: HuggingFace model name or path
    
    Returns:
        Dict with 'template' and 'confidence' keys
    """
    model_lower = model_name.lower()
    
    # Template detection heuristics
    if "llama-3" in model_lower or "llama3" in model_lower:
        return {"template": "llama-3", "confidence": "high"}
    elif "llama-2" in model_lower or "llama2" in model_lower:
        return {"template": "llama-2", "confidence": "high"}
    elif "qwen" in model_lower:
        return {"template": "qwen", "confidence": "high"}
    elif "gemma" in model_lower:
        return {"template": "gemma", "confidence": "high"}
    elif "phi" in model_lower:
        return {"template": "phi", "confidence": "high"}
    elif "mistral" in model_lower or "mixtral" in model_lower:
        return {"template": "mistral", "confidence": "high"}
    elif "zephyr" in model_lower:
        return {"template": "zephyr", "confidence": "high"}
    elif "vicuna" in model_lower:
        return {"template": "vicuna", "confidence": "medium"}
    elif "chatml" in model_lower or "chat" in model_lower:
        return {"template": "chatml", "confidence": "medium"}
    else:
        return {"template": "chatml", "confidence": "low"}


def estimate_training_time(
    dataset_size: int,
    model_size: str | int,
    epochs: int,
    backend: str = "auto"
) -> dict:
    """Rough estimate of training time.
    
    Args:
        dataset_size: Number of examples
        model_size: Model size in billions (e.g., 7, "7B", "8B") or param count
        epochs: Number of epochs
        backend: "mlx", "cuda", or "auto"
    
    Returns:
        Dict with time estimates in minutes
    """
    # Parse model size
    if isinstance(model_size, str):
        match = re.search(r'(\d+)[Bb]', model_size)
        size_b = int(match.group(1)) if match else 7
    else:
        size_b = model_size
    
    # Very rough heuristics (seconds per example per epoch)
    if backend == "mlx":
        # Apple Silicon (M-series)
        sec_per_example = 0.5 + (size_b / 10) * 0.3
    elif backend == "cuda":
        # NVIDIA GPU (assume modern card)
        sec_per_example = 0.2 + (size_b / 10) * 0.15
    else:
        # Conservative estimate
        sec_per_example = 1.0 + (size_b / 10) * 0.5
    
    total_seconds = dataset_size * epochs * sec_per_example
    minutes = total_seconds / 60
    hours = minutes / 60
    
    return {
        "estimated_minutes": round(minutes, 1),
        "estimated_hours": round(hours, 2),
        "dataset_size": dataset_size,
        "epochs": epochs,
        "model_size": f"{size_b}B",
        "backend": backend,
        "note": "This is a rough estimate. Actual time varies significantly."
    }


def validate_model_for_finetune(model_name: str) -> dict:
    """Check if a model can be fine-tuned.
    
    Args:
        model_name: HuggingFace model name or local path
    
    Returns:
        Dict with validation results
    """
    result = {
        "valid": True,
        "model_name": model_name,
        "is_local": False,
        "is_gguf": False,
        "warnings": [],
        "errors": []
    }
    
    # Check if it's a local path
    model_path = Path(model_name)  # lgtm[py/path-injection] — read-only existence check, no file writes
    if model_path.exists():
        result["is_local"] = True
        
        # Check if it's GGUF in filename
        if model_path.suffix == ".gguf":
            result["is_gguf"] = True
            result["warnings"].append(
                "GGUF models cannot be directly fine-tuned. "
                "Please use the original HuggingFace model instead."
            )
            result["errors"].append("GGUF input not supported")
            result["valid"] = False
    else:
        # Assume it's a HuggingFace model name
        # Check for GGUF in the name
        if "gguf" in model_name.lower():
            result["is_gguf"] = True
            result["warnings"].append(
                f"Model name contains 'GGUF': {model_name}. "
                "Unsloth requires HuggingFace format. "
                "Try removing '-GGUF' suffix (e.g., Qwen/Qwen3-8B instead of Qwen/Qwen3-8B-GGUF)"
            )
            result["errors"].append("GGUF model name detected")
            result["valid"] = False
            # Try to suggest the base model
            base_model = model_name.replace("-GGUF", "").replace("-gguf", "")
            if base_model != model_name:
                result["suggested_model"] = base_model
    
    return result


def resolve_gguf_to_hf(model_name: str) -> str:
    """Try to resolve a GGUF model name to its HF base model.
    
    Args:
        model_name: Model name (possibly containing GGUF)
    
    Returns:
        Resolved model name (best effort)
    """
    # Remove common GGUF suffixes
    resolved = model_name
    resolved = re.sub(r'-GGUF$', '', resolved, flags=re.IGNORECASE)
    resolved = re.sub(r'-gguf$', '', resolved, flags=re.IGNORECASE)
    resolved = re.sub(r'\.gguf$', '', resolved, flags=re.IGNORECASE)
    
    logger.info(f"Resolved GGUF model '{model_name}' -> '{resolved}'")
    return resolved


def get_lora_target_modules(model_name: str, task: str = "sft") -> list[str]:
    """Get recommended LoRA target modules for a model.
    
    Args:
        model_name: Model name (used to detect architecture)
        task: "sft" or "cpt" (CPT includes embeddings)
    
    Returns:
        List of module names to target
    """
    model_lower = model_name.lower()
    
    # Default targets for most transformer models
    base_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # For continued pre-training, also target embeddings and lm_head
    if task == "cpt":
        cpt_targets = base_targets + ["embed_tokens", "lm_head"]
        return cpt_targets
    
    # Model-specific adjustments for SFT
    if "phi" in model_lower:
        # Phi models use different names
        return ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
    elif "gemma" in model_lower or "qwen" in model_lower:
        return base_targets
    else:
        # Default for Llama, Mistral, etc.
        return base_targets
