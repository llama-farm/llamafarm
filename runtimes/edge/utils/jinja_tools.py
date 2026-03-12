"""
Jinja2 template utilities for tool-aware chat template rendering.

This module provides functions to extract chat templates from GGUF files
and render them with tool definitions using Python's Jinja2.

Uses the shared GGUF metadata cache to avoid redundant file reads when
extracting chat templates and special tokens.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from jinja2 import TemplateError, Undefined
from jinja2.sandbox import SandboxedEnvironment
from jinja2.utils import Namespace

from utils.gguf_metadata_cache import get_gguf_metadata_cached

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


def get_chat_template_from_gguf(model_path: str) -> str | None:
    """Extract chat_template from GGUF file metadata.

    Uses the shared GGUF metadata cache to avoid redundant file reads.
    The cache is populated once per file and reused by all modules.

    Args:
        model_path: Path to the GGUF model file.

    Returns:
        The chat template string, or None if not found.
    """
    try:
        cached = get_gguf_metadata_cached(model_path)
        return cached.chat_template
    except FileNotFoundError:
        logger.debug(f"GGUF file not found: {model_path}")
        return None
    except Exception as e:
        logger.debug(f"Failed to extract chat template from {model_path}: {e}")
        return None


def get_special_tokens_from_gguf(model_path: str) -> dict[str, str]:
    """Extract BOS and EOS tokens from GGUF file metadata.

    Uses the shared GGUF metadata cache to avoid redundant file reads.
    The cache is populated once per file and reused by all modules.

    Args:
        model_path: Path to the GGUF model file.

    Returns:
        Dictionary with 'bos_token' and 'eos_token' keys.
        Values default to empty strings if not found.
    """
    try:
        cached = get_gguf_metadata_cached(model_path)
        return {
            "bos_token": cached.bos_token,
            "eos_token": cached.eos_token,
        }
    except FileNotFoundError:
        logger.debug(f"GGUF file not found: {model_path}")
        return {"bos_token": "", "eos_token": ""}
    except Exception as e:
        logger.debug(f"Failed to extract special tokens from {model_path}: {e}")
        return {"bos_token": "", "eos_token": ""}


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


def create_jinja_environment() -> SandboxedEnvironment:
    """Create a sandboxed Jinja2 environment configured for chat templates.

    Uses SandboxedEnvironment to prevent arbitrary code execution from
    potentially malicious templates in GGUF files.

    Returns:
        Configured Jinja2 SandboxedEnvironment.
    """
    env = SandboxedEnvironment(
        # Use undefined that returns False for boolean checks
        undefined=RaiseExceptionUndefined,
        # Keep trailing newlines
        keep_trailing_newline=True,
        # Auto-escape disabled (we're not generating HTML)
        autoescape=False,
    )

    # Add template functions used by various chat templates
    env.globals["raise_exception"] = _raise_exception
    # Use Jinja2's built-in Namespace which properly handles attribute assignment
    env.globals["namespace"] = Namespace

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
