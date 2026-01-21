"""Template service for dynamic variable substitution.

This service handles resolving template variables in strings using
Jinja2-style {{variable}} and {{variable | default}} syntax.

Example usage:
    from services.template_service import TemplateService

    # Basic substitution
    result = TemplateService.resolve("Hello {{name}}", {"name": "Alice"})
    # Returns: "Hello Alice"

    # With default value
    result = TemplateService.resolve("Hello {{name | Guest}}", {})
    # Returns: "Hello Guest"

    # Resolve in nested objects
    obj = {"message": "Hello {{name}}", "config": {"value": "{{val | 42}}"}}
    result = TemplateService.resolve_object(obj, {"name": "Alice"})
    # Returns: {"message": "Hello Alice", "config": {"value": "42"}}
"""

import re
from typing import Any

from core.logging import FastAPIStructLogger

logger = FastAPIStructLogger(__name__)


class TemplateError(Exception):
    """Error raised when template resolution fails."""

    def __init__(self, message: str, variable_name: str | None = None):
        super().__init__(message)
        self.variable_name = variable_name


class TemplateService:
    """Service for resolving template variables in strings and objects.

    Supports Jinja2-style syntax:
    - {{variable_name}} - Required variable (error if missing)
    - {{variable_name | default_value}} - Optional variable with default
    - {{ variable_name }} - Whitespace is allowed and trimmed

    Security Note:
        Templates should ONLY be defined in trusted configuration files.
        Variable values are substituted as-is without sanitization.
        This design assumes templates come from config files (not user input)
        and variable values come from trusted API consumers.

        IMPORTANT: Variable values should NEVER come from untrusted end-user input.
        While template markers in values are not recursively expanded (preventing
        injection attacks), values are inserted verbatim which could enable:
        - Prompt injection if values end up in LLM prompts
        - XSS if values end up in HTML output (caller must escape)

    Threat Model:
        - Templates: Trusted (come from config YAML files under developer control)
        - Variable values: Semi-trusted (come from API consumers, not end users)
        - Validation: Max length enforced, complex types rejected

    Limitations:
        - Default values containing '}' before '}}' may not parse correctly
        - Nested template markers (e.g., {{outer_{{inner}}}}) are not supported
        - Only primitive types allowed (str, int, float, bool, None)
    """

    # Maximum allowed length for variable values (prevents DoS)
    MAX_VALUE_LENGTH = 100_000  # 100KB per value

    # Allowed types for variable values (primitives only)
    ALLOWED_TYPES = (str, int, float, bool, type(None))

    # Pattern to match {{variable}} or {{variable | default}}
    # Captures: (variable_name, optional_default_with_pipe)
    _PATTERN = re.compile(
        r"\{\{\s*"  # Opening {{ with optional whitespace
        r"([a-zA-Z_][a-zA-Z0-9_]*)"  # Variable name (captured)
        r"(?:\s*\|\s*([^}]*))?"  # Optional | default (captured, default may be empty)
        r"\s*\}\}"  # Closing }} with optional whitespace
    )

    @classmethod
    def resolve(cls, template: str, variables: dict[str, Any]) -> str:
        """Resolve template variables in a string.

        Args:
            template: String potentially containing {{variable}} placeholders
            variables: Dict of variable name -> value mappings

        Returns:
            String with all variables resolved

        Raises:
            TemplateError: If a required variable is missing (no default provided)

        Examples:
            >>> TemplateService.resolve("Hello {{name}}", {"name": "Alice"})
            "Hello Alice"

            >>> TemplateService.resolve("Hello {{name | Guest}}", {})
            "Hello Guest"
        """
        if not template or "{{" not in template:
            return template

        def replace_match(match: re.Match) -> str:
            var_name = match.group(1)
            default_value = match.group(2)

            if var_name in variables:
                value = variables[var_name]

                # Validate type - only allow primitives
                if not isinstance(value, cls.ALLOWED_TYPES):
                    raise TemplateError(
                        f"Variable '{var_name}' has unsupported type '{type(value).__name__}'. "
                        f"Only primitive types are allowed: str, int, float, bool, None.",
                        variable_name=var_name,
                    )

                # Handle None explicitly - treat as empty string
                if value is None:
                    logger.debug(
                        "Variable value is None, using empty string",
                        variable=var_name,
                    )
                    return ""

                # Convert to string with type logging for non-strings
                if not isinstance(value, str):
                    logger.debug(
                        "Converting non-string variable to string",
                        variable=var_name,
                        original_type=type(value).__name__,
                    )
                    value = str(value)

                # Enforce max length to prevent DoS
                if len(value) > cls.MAX_VALUE_LENGTH:
                    raise TemplateError(
                        f"Variable '{var_name}' value exceeds maximum length "
                        f"({len(value)} > {cls.MAX_VALUE_LENGTH})",
                        variable_name=var_name,
                    )

                return value

            if default_value is not None:
                # Default was provided (even if empty string)
                return default_value.strip()

            # No value and no default - error
            logger.warning(
                "Template variable not found",
                variable=var_name,
                available_variables=list(variables.keys()) if variables else [],
            )
            raise TemplateError(
                f"Template variable '{{{{ {var_name} }}}}' not found in provided variables. "
                f"Available variables: {list(variables.keys()) if variables else '(none)'}. "
                f"Add a default with '{{{{ {var_name} | default_value }}}}'.",
                variable_name=var_name,
            )

        try:
            return cls._PATTERN.sub(replace_match, template)
        except TemplateError:
            raise
        except Exception as e:
            raise TemplateError(f"Failed to resolve template: {e}") from e

    @classmethod
    def resolve_object(cls, obj: Any, variables: dict[str, Any]) -> Any:
        """Recursively resolve template variables in an object.

        Handles nested dicts, lists, and strings. Non-string values
        (int, bool, None, etc.) pass through unchanged.

        Args:
            obj: Object potentially containing template strings
            variables: Dict of variable name -> value mappings

        Returns:
            Object with all template strings resolved

        Raises:
            TemplateError: If a required variable is missing

        Examples:
            >>> obj = {"message": "Hello {{name}}", "count": 42}
            >>> TemplateService.resolve_object(obj, {"name": "Alice"})
            {"message": "Hello Alice", "count": 42}
        """
        if obj is None:
            return None

        if isinstance(obj, str):
            return cls.resolve(obj, variables)

        if isinstance(obj, dict):
            return {key: cls.resolve_object(value, variables) for key, value in obj.items()}

        if isinstance(obj, list):
            return [cls.resolve_object(item, variables) for item in obj]

        # For other types (int, float, bool, etc.), return as-is
        return obj

    @classmethod
    def has_template_markers(cls, text: str) -> bool:
        """Check if a string contains template markers.

        Useful for quick checks before full resolution.

        Args:
            text: String to check

        Returns:
            True if string contains {{...}} markers
        """
        return "{{" in text and "}}" in text

    @classmethod
    def extract_variables(cls, template: str) -> list[tuple[str, str | None]]:
        """Extract variable names and defaults from a template.

        Useful for validation and documentation.

        Args:
            template: Template string to analyze

        Returns:
            List of (variable_name, default_or_none) tuples

        Examples:
            >>> TemplateService.extract_variables("{{name}} is {{age | 0}}")
            [("name", None), ("age", "0")]
        """
        if not template:
            return []

        results = []
        for match in cls._PATTERN.finditer(template):
            var_name = match.group(1)
            default_value = match.group(2)
            if default_value is not None:
                default_value = default_value.strip()
            results.append((var_name, default_value))

        return results
