"""Service for substituting template variables in prompts."""

import re

from core.logging import FastAPIStructLogger

logger = FastAPIStructLogger()

# Pattern to match {{variable_name}} - word characters only (alphanumeric + underscore)
VARIABLE_PATTERN = re.compile(r"\{\{(\w+)\}\}")


class PromptVariableService:
    """Service for substituting template variables in prompt messages."""

    @classmethod
    def substitute(cls, content: str, variables: dict[str, str]) -> str:
        """Substitute {{variable}} patterns in content.

        Args:
            content: String containing {{variable_name}} patterns
            variables: Dict mapping variable names to values

        Returns:
            String with variables substituted. Unmatched variables become empty strings.
        """

        def replacer(match: re.Match) -> str:
            var_name = match.group(1)
            return variables.get(var_name, "")

        return VARIABLE_PATTERN.sub(replacer, content)

    @classmethod
    def substitute_messages(
        cls,
        messages: list[dict],
        model_defaults: dict[str, str] | None,
        request_overrides: dict[str, str] | None,
    ) -> list[dict]:
        """Substitute variables in all message contents.

        Priority: request_overrides > model_defaults > empty string

        Args:
            messages: List of message dicts with 'content' keys
            model_defaults: Variable defaults from model config
            request_overrides: Variable overrides from request

        Returns:
            New list of messages with variables substituted
        """
        # Merge variables: request overrides model defaults
        variables = {**(model_defaults or {}), **(request_overrides or {})}

        if not variables:
            return messages

        result = []
        for msg in messages:
            new_msg = msg.copy()
            if "content" in new_msg and isinstance(new_msg["content"], str):
                new_msg["content"] = cls.substitute(new_msg["content"], variables)
            result.append(new_msg)

        return result

    @classmethod
    def find_variables(cls, content: str) -> list[str]:
        """Find all variable names in content.

        Args:
            content: String to search for {{variable_name}} patterns

        Returns:
            List of variable names found (without braces)
        """
        return VARIABLE_PATTERN.findall(content)

    @classmethod
    def find_variables_in_messages(cls, messages: list[dict]) -> list[str]:
        """Find all unique variable names across all messages.

        Args:
            messages: List of message dicts with 'content' keys

        Returns:
            Sorted list of unique variable names found
        """
        variables = set()
        for msg in messages:
            if "content" in msg and isinstance(msg["content"], str):
                variables.update(cls.find_variables(msg["content"]))
        return sorted(variables)
