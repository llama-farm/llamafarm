"""Prompt resolution service for multi-prompt support.

This service handles resolving which prompts to use for a specific model,
supporting named prompt sets with per-model selection.
"""

import sys
from functools import lru_cache
from pathlib import Path

from core.logging import FastAPIStructLogger

# Add repo root to path for config imports
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from config.datamodel import LlamaFarmConfig, Model, Message  # noqa: E402

logger = FastAPIStructLogger(__name__)


class PromptService:
    """Service for resolving prompts for models."""

    @staticmethod
    def get_prompt_sets(config: LlamaFarmConfig) -> dict[str, list[Message]]:
        """
        Convert config.prompts (list of NamedPromptSet) to a dict.

        Returns:
            dict mapping prompt set name -> list of messages
        """
        if not config.prompts:
            return {}

        return {pset.name: pset.messages for pset in config.prompts}

    @staticmethod
    def resolve_prompts_for_model(
        config: LlamaFarmConfig, model: Model
    ) -> list[Message]:
        """
        Resolve which prompts to use for a specific model.

        Logic:
        1. If model.prompts is set (list of names), merge those sets in order
        2. Otherwise, stack all prompts in definition order

        Note on caching: This method is intentionally NOT cached using @lru_cache because:
        1. Message objects are Pydantic models and not hashable
        2. The operation is lightweight (dict lookup + list concatenation)
        3. Config objects are already cached at the project level
        4. Caching would require serializing/deserializing Message objects, which is slower
           than the actual operation

        For performance optimization, consider caching at the ProjectChatOrchestrator level
        where agent instances are reused across requests.

        Args:
            config: Project configuration
            model: Model configuration

        Returns:
            Merged list of prompt messages
        """
        prompt_sets = PromptService.get_prompt_sets(config)

        if not prompt_sets:
            logger.debug("No prompt sets defined in config")
            return []

        # Model specifies which prompts to use (list of prompt set names)
        if model.prompts and len(model.prompts) > 0:
            logger.info(
                "Resolving model-specific prompts",
                model=model.name,
                prompt_names=model.prompts,
            )
            merged = []
            for pset_name in model.prompts:
                if pset_name in prompt_sets:
                    messages_in_set = prompt_sets[pset_name]
                    logger.info(
                        "Adding prompt set to merged list",
                        model=model.name,
                        prompt_set=pset_name,
                        message_count=len(messages_in_set),
                    )
                    merged.extend(messages_in_set)
                else:
                    available_sets = ', '.join(sorted(prompt_sets.keys()))
                    error_msg = (
                        f"Model '{model.name}' references non-existent prompt set '{pset_name}'. "
                        f"Available prompt sets: {available_sets}"
                    )
                    logger.error(
                        "Prompt set not found",
                        model=model.name,
                        prompt_name=pset_name,
                        available_sets=available_sets,
                    )
                    raise ValueError(error_msg)
            logger.info(
                "Prompt resolution complete",
                model=model.name,
                total_messages=len(merged),
                prompt_sets_used=model.prompts,
            )
            return merged

        # Default: stack all prompts in definition order
        logger.debug(
            "Using all prompts in definition order",
            model=model.name,
            prompt_count=len(config.prompts) if config.prompts else 0,
        )
        merged = []
        if config.prompts:
            for pset in config.prompts:
                merged.extend(pset.messages)
        return merged
