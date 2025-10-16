"""Utilities for handling multimodal message content."""

from typing import Any

from api.routers.inference.models import ChatMessage, MessageContentPart
from core.logging import FastAPIStructLogger

logger = FastAPIStructLogger(__name__)


class MultimodalMessageError(Exception):
    """Raised when multimodal message conversion fails."""

    pass


def validate_message_content_part(part: MessageContentPart) -> None:
    """Validate a message content part has required fields.

    Args:
        part: The message content part to validate

    Raises:
        MultimodalMessageError: If the part is missing required fields
    """
    if part.type == "text" and part.text is None:
        raise MultimodalMessageError("Text content part missing 'text' field")

    if part.type in ("image_url", "audio_url", "video_url"):
        media_url_field = getattr(part, part.type, None)
        if media_url_field is None:
            raise MultimodalMessageError(
                f"{part.type} content part missing '{part.type}' field"
            )
        if not hasattr(media_url_field, "url") or not media_url_field.url:
            raise MultimodalMessageError(
                f"{part.type} content part missing 'url' in {part.type} object"
            )


def convert_message_to_openai_format(message: ChatMessage) -> dict[str, Any]:
    """Convert a ChatMessage to OpenAI API format.

    Handles both simple string content and multimodal content parts.
    Validates all content parts before conversion.

    Args:
        message: The ChatMessage to convert

    Returns:
        Dictionary in OpenAI message format

    Raises:
        MultimodalMessageError: If message validation fails
    """
    if isinstance(message.content, str):
        return {"role": message.role, "content": message.content}

    # Multimodal content - validate and convert each part
    content_parts = []
    for part in message.content:
        # Validate the part before conversion
        validate_message_content_part(part)

        if part.type == "text":
            content_parts.append({"type": "text", "text": part.text})
        elif part.type == "image_url" and part.image_url:
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": part.image_url.url,
                        "detail": part.image_url.detail,
                    },
                }
            )
        elif part.type == "audio_url" and part.audio_url:
            content_parts.append(
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": part.audio_url.url,
                    },
                }
            )
        elif part.type == "video_url" and part.video_url:
            content_parts.append(
                {
                    "type": "video_url",
                    "video_url": {
                        "url": part.video_url.url,
                    },
                }
            )

    return {"role": message.role, "content": content_parts}


def convert_messages_to_openai_format(
    messages: list[ChatMessage],
) -> list[dict[str, Any]]:
    """Convert a list of ChatMessages to OpenAI API format.

    Args:
        messages: List of ChatMessage objects to convert

    Returns:
        List of dictionaries in OpenAI message format

    Raises:
        MultimodalMessageError: If any message validation fails
    """
    return [convert_message_to_openai_format(msg) for msg in messages]


def extract_text_from_message(message: ChatMessage) -> str:
    """Extract text content from a ChatMessage.

    For simple string content, returns the string.
    For multimodal content, extracts and joins all text parts.

    Args:
        message: The ChatMessage to extract text from

    Returns:
        Extracted text content, or default message if no text found
    """
    if isinstance(message.content, str):
        return message.content

    if isinstance(message.content, list):
        text_parts = [
            part.text for part in message.content if part.type == "text" and part.text
        ]
        return " ".join(text_parts) if text_parts else "Describe this media."

    return ""


def has_multimodal_content(messages: list[ChatMessage]) -> bool:
    """Check if any message contains multimodal content.

    Args:
        messages: List of messages to check

    Returns:
        True if any message has multimodal content (list of parts)
    """
    return any(isinstance(msg.content, list) for msg in messages)
