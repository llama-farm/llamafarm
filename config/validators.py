"""Custom validators for LlamaFarmConfig that extend JSON Schema validation.

These validators handle constraints that cannot be expressed in JSON Schema draft-07,
such as uniqueness of object properties within arrays.
"""

import re
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
    
    # Validate dataset names
    if "datasets" in config_dict and config_dict["datasets"]:
        datasets = config_dict["datasets"]
        if isinstance(datasets, list):
            dataset_names = []
            for idx, dataset in enumerate(datasets):
                if not isinstance(dataset, dict):
                    continue
                    
                name = dataset.get("name", "")
                if not name:
                    raise ValueError(
                        f"Dataset at index {idx} is missing a name. "
                        "Each dataset must have a name."
                    )
                
                # Validate name length
                if len(name) > 100:
                    raise ValueError(
                        f"Dataset name '{name}' is too long (max 100 characters). "
                        "Please use a shorter name."
                    )
                
                # Validate name characters (only alphanumeric, hyphens, underscores)
                valid_pattern = re.compile(r'^[a-zA-Z0-9_-]+$')
                if not valid_pattern.match(name):
                    raise ValueError(
                        f"Dataset name '{name}' contains invalid characters. "
                        "Dataset names can only contain letters, numbers, underscores (_), and hyphens (-)."
                    )
                
                dataset_names.append(name)
            
            # Check for duplicate dataset names (case-insensitive)
            name_counts: dict[str, list[str]] = {}
            for name in dataset_names:
                lower_name = name.lower()
                if lower_name not in name_counts:
                    name_counts[lower_name] = []
                name_counts[lower_name].append(name)
            
            duplicates = {original[0]: original for original in name_counts.values() if len(original) > 1}
            if duplicates:
                duplicate_list = ', '.join(f"'{name}'" for name in duplicates.keys())
                raise ValueError(
                    f"Duplicate dataset names found: {duplicate_list}. "
                    "Each dataset must have a unique name (case-insensitive)."
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
