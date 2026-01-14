"""Prompt resolution service for multi-prompt support.

This service handles resolving which prompts to use for a specific model,
supporting named prompt sets with per-model selection and dynamic variable
substitution using {{variable}} syntax.
"""

from typing import Any

from config.datamodel import LlamaFarmConfig, Model, PromptMessage

from core.logging import FastAPIStructLogger
from services.template_service import TemplateService

logger = FastAPIStructLogger(__name__)


class PromptService:
    """Service for resolving prompts for models."""

    @staticmethod
    def get_prompt_sets(config: LlamaFarmConfig) -> dict[str, list[PromptMessage]]:
        """
        Convert config.prompts (list of NamedPromptSet) to a dict.

        Returns:
            dict mapping prompt set name -> list of messages
        """
        if not config.prompts:
            return {}

        return {pset.name: pset.messages for pset in config.prompts}

    @staticmethod
    def _resolve_message_templates(
        messages: list[PromptMessage], variables: dict[str, Any] | None
    ) -> list[PromptMessage]:
        """
        Resolve template variables in prompt message content.

        Args:
            messages: List of prompt messages
            variables: Dict of variable name -> value mappings (or None)

        Returns:
            New list of PromptMessage with resolved content
        """
        # Normalize None to empty dict for resolution
        vars_dict = variables or {}

        resolved = []
        for msg in messages:
            # Resolve template in content if it has template markers
            if msg.content and TemplateService.has_template_markers(msg.content):
                resolved_content = TemplateService.resolve(msg.content, vars_dict)
                # Create new PromptMessage with resolved content
                resolved.append(
                    PromptMessage(
                        role=msg.role,
                        content=resolved_content,
                        tool_call_id=msg.tool_call_id,
                    )
                )
            else:
                resolved.append(msg)

        return resolved

    @staticmethod
    def resolve_prompts_for_model(
        config: LlamaFarmConfig,
        model: Model,
        variables: dict[str, Any] | None = None,
    ) -> list[PromptMessage]:
        """
        Resolve which prompts to use for a specific model.

        Logic:
        1. If model.prompts is set (list of names), merge those sets in order
        2. Otherwise, stack all prompts in definition order
        3. Resolve any {{variable}} templates in the merged prompts

        Note on caching: This method is intentionally NOT cached using @lru_cache because:
        1. PromptMessage objects are Pydantic models and not hashable
        2. The operation is lightweight (dict lookup + list concatenation)
        3. Config objects are already cached at the project level
        4. Caching would require serializing/deserializing PromptMessage objects, which is slower
           than the actual operation

        For performance optimization, consider caching at the ProjectChatOrchestrator level
        where agent instances are reused across requests.

        Args:
            config: Project configuration
            model: Model configuration
            variables: Optional dict of variable values for template substitution.
                       Variables use {{name}} or {{name | default}} syntax in prompts.

        Returns:
            Merged list of prompt messages with templates resolved

        Raises:
            ValueError: If a referenced prompt set doesn't exist
            TemplateError: If a required template variable is missing
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
                    available_sets = ", ".join(sorted(prompt_sets.keys()))
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

            # Resolve templates after merging (always, to handle defaults)
            logger.debug(
                "Resolving template variables in prompts",
                variable_count=len(variables) if variables else 0,
                variable_names=list(variables.keys()) if variables else [],
            )
            merged = PromptService._resolve_message_templates(merged, variables)

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

        # Resolve templates after merging (always, to handle defaults)
        logger.debug(
            "Resolving template variables in prompts",
            variable_count=len(variables) if variables else 0,
            variable_names=list(variables.keys()) if variables else [],
        )
        merged = PromptService._resolve_message_templates(merged, variables)

        return merged
