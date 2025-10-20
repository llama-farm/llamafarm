"""Custom validators for LlamaFarmConfig that extend JSON Schema validation.

These validators handle constraints that cannot be expressed in JSON Schema draft-07,
such as uniqueness of object properties within arrays.
"""

from typing import Any


def validate_llamafarm_config(config_dict: dict[str, Any]) -> None:
    """
    Validate LlamaFarmConfig constraints beyond JSON Schema.

    Raises:
        ValueError: If validation fails with descriptive error message
    """
    # Validate unique prompt names
    if "prompts" in config_dict and config_dict["prompts"]:
        prompt_names = [p.get("name") for p in config_dict["prompts"] if isinstance(p, dict)]
        duplicates = [name for name in prompt_names if prompt_names.count(name) > 1]
        if duplicates:
            raise ValueError(
                f"Duplicate prompt set names found: {', '.join(set(duplicates))}. "
                "Each prompt set must have a unique name."
            )

    # Validate model.prompts reference existing sets
    if "prompts" in config_dict and "runtime" in config_dict:
        prompt_names_set = {p.get("name") for p in config_dict.get("prompts", []) if isinstance(p, dict)}
        runtime = config_dict.get("runtime", {})

        if isinstance(runtime, dict) and "models" in runtime:
            models = runtime.get("models", [])
            if isinstance(models, list):
                for model in models:
                    if isinstance(model, dict) and "prompts" in model:
                        model_prompts = model.get("prompts", [])
                        model_name = model.get("name", "unknown")

                        if isinstance(model_prompts, list):
                            for prompt_ref in model_prompts:
                                if prompt_ref not in prompt_names_set:
                                    available = ', '.join(sorted(prompt_names_set)) if prompt_names_set else "none"
                                    raise ValueError(
                                        f"Model '{model_name}' references non-existent prompt set '{prompt_ref}'. "
                                        f"Available prompt sets: {available}"
                                    )
