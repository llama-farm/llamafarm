"""Service for managing prompts within projects."""

import re

from config.datamodel import PromptMessage, PromptSet

from core.logging import FastAPIStructLogger
from services.project_service import ProjectService

logger = FastAPIStructLogger()

# Pattern for valid prompt names (must match schema)
PROMPT_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class PromptNotFoundError(Exception):
    """Raised when a prompt is not found."""

    def __init__(self, name: str):
        self.name = name
        super().__init__(f"Prompt '{name}' not found")


class PromptsService:
    """Service for managing prompts within projects."""

    @classmethod
    def _validate_name(cls, name: str) -> None:
        """Validate prompt name matches schema pattern.

        Raises:
            ValueError: If name doesn't match required pattern
        """
        if not PROMPT_NAME_PATTERN.match(name):
            raise ValueError(
                f"Invalid prompt name '{name}'. "
                "Must start with a lowercase letter and contain only lowercase letters, numbers, and underscores."
            )

    @classmethod
    def list_prompts(cls, namespace: str, project: str) -> list[PromptSet]:
        """List all prompts for a given project."""
        project_config = ProjectService.load_config(namespace, project)
        return project_config.prompts or []

    @classmethod
    def get_prompt(cls, namespace: str, project: str, name: str) -> PromptSet:
        """
        Get a single prompt by name.

        Raises:
            PromptNotFoundError: If prompt with given name is not found
        """
        prompts = cls.list_prompts(namespace, project)
        for prompt in prompts:
            if prompt.name == name:
                return prompt
        raise PromptNotFoundError(name)

    @classmethod
    def create_prompt(
        cls,
        namespace: str,
        project: str,
        name: str,
        messages: list[PromptMessage],
    ) -> PromptSet:
        """
        Create a new prompt in the project.

        Args:
            namespace: Project namespace
            project: Project name
            name: Unique prompt identifier
            messages: List of messages in the prompt set

        Raises:
            ValueError: If prompt with same name already exists or name is invalid
        """
        cls._validate_name(name)

        project_config = ProjectService.load_config(namespace, project)
        existing_prompts = project_config.prompts or []

        # Check if prompt already exists
        for prompt in existing_prompts:
            if prompt.name == name:
                raise ValueError(f"Prompt '{name}' already exists")

        # Create the new prompt
        new_prompt = PromptSet(name=name, messages=messages)

        # Add to config and save
        existing_prompts.append(new_prompt)
        project_config.prompts = existing_prompts
        ProjectService.save_config(namespace, project, project_config)

        logger.info(
            "Created prompt",
            namespace=namespace,
            project=project,
            prompt=name,
            message_count=len(messages),
        )

        return new_prompt

    @classmethod
    def update_prompt(
        cls,
        namespace: str,
        project: str,
        name: str,
        messages: list[PromptMessage],
    ) -> PromptSet:
        """
        Update an existing prompt's messages.

        Args:
            namespace: Project namespace
            project: Project name
            name: Prompt identifier to update
            messages: New list of messages

        Raises:
            PromptNotFoundError: If prompt with given name is not found
        """
        project_config = ProjectService.load_config(namespace, project)
        existing_prompts = project_config.prompts or []

        # Find the prompt to update
        prompt_index = None
        for i, prompt in enumerate(existing_prompts):
            if prompt.name == name:
                prompt_index = i
                break

        if prompt_index is None:
            raise PromptNotFoundError(name)

        # Update the prompt
        updated_prompt = PromptSet(name=name, messages=messages)
        existing_prompts[prompt_index] = updated_prompt

        # Save config
        project_config.prompts = existing_prompts
        ProjectService.save_config(namespace, project, project_config)

        logger.info(
            "Updated prompt",
            namespace=namespace,
            project=project,
            prompt=name,
            message_count=len(messages),
        )

        return updated_prompt

    @classmethod
    def delete_prompt(
        cls,
        namespace: str,
        project: str,
        name: str,
    ) -> PromptSet:
        """
        Delete a prompt from the project.

        Args:
            namespace: Project namespace
            project: Project name
            name: Prompt identifier to delete

        Raises:
            PromptNotFoundError: If prompt with given name is not found
            ValueError: If the prompt is referenced by any model
        """
        project_config = ProjectService.load_config(namespace, project)
        existing_prompts = project_config.prompts or []

        # Find the prompt to delete
        prompt_to_delete = None
        prompt_index = None
        for i, prompt in enumerate(existing_prompts):
            if prompt.name == name:
                prompt_to_delete = prompt
                prompt_index = i
                break

        if prompt_to_delete is None:
            raise PromptNotFoundError(name)

        # Check if any models reference this prompt
        dependent_models = cls.get_dependent_models(namespace, project, name)
        if dependent_models:
            raise ValueError(
                f"Cannot delete prompt '{name}': {len(dependent_models)} model(s) reference it. "
                f"Update or remove these models first: {dependent_models}"
            )

        # Remove from config
        existing_prompts.pop(prompt_index)
        project_config.prompts = existing_prompts
        ProjectService.save_config(namespace, project, project_config)

        logger.info(
            "Deleted prompt",
            namespace=namespace,
            project=project,
            prompt=name,
        )

        return prompt_to_delete

    @classmethod
    def get_dependent_models(
        cls, namespace: str, project: str, prompt_name: str
    ) -> list[str]:
        """
        Get list of model names that reference a prompt.

        Useful for checking before deletion.
        """
        project_config = ProjectService.load_config(namespace, project)
        models = (project_config.runtime.models if project_config.runtime else []) or []

        dependent = []
        for model in models:
            # Check if the model's prompts list includes this prompt
            if model.prompts and prompt_name in model.prompts:
                dependent.append(model.name)

        return dependent
