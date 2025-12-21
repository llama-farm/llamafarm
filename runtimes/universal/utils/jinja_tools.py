"""
Jinja2 template utilities for tool-aware chat template rendering.

This module provides functions to extract chat templates from GGUF files
and render them with tool definitions using Python's Jinja2.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from jinja2 import Environment, TemplateError, Undefined

logger = logging.getLogger(__name__)


class RaiseExceptionUndefined(Undefined):
    """Jinja2 Undefined that raises an exception when used.

    Some chat templates use `raise_exception` to signal errors.
    This class provides that functionality.
    """

    def __str__(self) -> str:
        raise TemplateError(f"Undefined variable: {self._undefined_name}")

    def __iter__(self):
        raise TemplateError(f"Undefined variable: {self._undefined_name}")

    def __bool__(self):
        return False


def _raise_exception(message: str) -> None:
    """Template function to raise an exception."""
    raise TemplateError(message)


def _tojson(value: Any, indent: int | None = None) -> str:
    """Template filter to convert value to JSON string."""
    return json.dumps(value, indent=indent, ensure_ascii=False)


def _namespace(**kwargs) -> dict:
    """Template function to create a namespace (dict) for variable storage."""
    return dict(**kwargs)


def get_chat_template_from_gguf(model_path: str) -> str | None:
    """Extract chat_template from GGUF file metadata.

    Args:
        model_path: Path to the GGUF model file.

    Returns:
        The chat template string, or None if not found.
    """
    try:
        from gguf import GGUFReader

        reader = GGUFReader(model_path)

        # Look for tokenizer.chat_template in metadata
        for field in reader.fields.values():
            if field.name == "tokenizer.chat_template":
                # The value is stored as bytes, decode to string
                if hasattr(field, "parts"):
                    # New GGUF format
                    template_bytes = b"".join(
                        bytes(part) if isinstance(part, (list, tuple)) else part
                        for part in field.parts
                    )
                    return template_bytes.decode("utf-8")
                elif hasattr(field, "data"):
                    # Older format
                    return bytes(field.data).decode("utf-8")

        logger.debug(f"No chat_template found in GGUF metadata for {model_path}")
        return None

    except ImportError:
        logger.warning("gguf package not installed, cannot extract chat template")
        return None
    except Exception as e:
        logger.warning(f"Failed to extract chat template from {model_path}: {e}")
        return None


def supports_native_tools(template: str) -> bool:
    """Check if a chat template has native tool support.

    A template supports tools if it references the 'tools' variable.

    Args:
        template: The Jinja2 chat template string.

    Returns:
        True if the template references tools, False otherwise.
    """
    # Simple heuristic: check if 'tools' appears in the template
    # This catches patterns like {% if tools %}, {{ tools }}, etc.
    return "tools" in template


def create_jinja_environment() -> Environment:
    """Create a Jinja2 environment configured for chat templates.

    Returns:
        Configured Jinja2 Environment.
    """
    env = Environment(
        # Use undefined that returns False for boolean checks
        undefined=RaiseExceptionUndefined,
        # Keep trailing newlines
        keep_trailing_newline=True,
        # Auto-escape disabled (we're not generating HTML)
        autoescape=False,
    )

    # Add template functions used by various chat templates
    env.globals["raise_exception"] = _raise_exception
    env.globals["namespace"] = _namespace

    # Add filters
    env.filters["tojson"] = _tojson

    return env


def render_chat_with_tools(
    template: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    add_generation_prompt: bool = True,
    bos_token: str = "",
    eos_token: str = "",
) -> str:
    """Render a chat template with Jinja2 including tool definitions.

    This function mimics what llama.cpp's Jinja-based template rendering does,
    allowing us to pass tools to models that have tool-aware templates.

    Args:
        template: The Jinja2 chat template string.
        messages: List of chat messages (role, content dicts).
        tools: Optional list of tool definitions (OpenAI format).
        add_generation_prompt: Whether to add the assistant prompt at the end.
        bos_token: Beginning of sequence token.
        eos_token: End of sequence token.

    Returns:
        The rendered prompt string.

    Raises:
        TemplateError: If template rendering fails.
    """
    env = create_jinja_environment()

    try:
        template_obj = env.from_string(template)
    except Exception as e:
        raise TemplateError(f"Failed to parse chat template: {e}") from e

    try:
        rendered = template_obj.render(
            messages=messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            bos_token=bos_token,
            eos_token=eos_token,
        )
        return rendered
    except Exception as e:
        raise TemplateError(f"Failed to render chat template: {e}") from e
