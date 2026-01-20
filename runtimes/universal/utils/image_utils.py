"""Shared image loading utilities for vision models.

This module provides a centralized image loading function that handles:
- PIL Image objects (passthrough)
- Raw bytes
- File paths
- Base64-encoded strings

All models that need to load images should use this shared utility
instead of implementing their own _load_image method.
"""

import base64
import contextlib
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image as PILImage


def load_image(image: str | bytes | Any, convert_mode: str = "RGB") -> "PILImage.Image":
    """Load an image from various input formats.

    This is a shared utility function that handles multiple image input formats
    commonly used across vision models. It supports:
    - PIL Image objects (returned as-is after mode conversion)
    - Raw bytes (decoded as image)
    - File paths (loaded from disk)
    - Base64-encoded strings (decoded and loaded)

    The function automatically distinguishes between file paths and base64 strings
    by checking string length (base64 images are typically > 4096 chars).

    Args:
        image: Image input in one of the supported formats:
            - PIL.Image.Image: Returned directly after mode conversion
            - bytes: Raw image bytes
            - str: File path or base64-encoded image data
            - pathlib.Path: File path
        convert_mode: PIL image mode to convert to (default: "RGB").
            Use "RGBA" to preserve transparency.

    Returns:
        PIL Image object in the specified mode.

    Raises:
        ValueError: If the image format is not supported or decoding fails.

    Examples:
        >>> from utils.image_utils import load_image
        >>> img = load_image("/path/to/image.jpg")
        >>> img = load_image(base64_string)
        >>> img = load_image(image_bytes)
        >>> img = load_image(pil_image)
    """
    from PIL import Image

    # Already a PIL Image
    if isinstance(image, Image.Image):
        return image.convert(convert_mode)

    # Raw bytes
    if isinstance(image, bytes):
        return Image.open(BytesIO(image)).convert(convert_mode)

    # String input - could be file path or base64
    if isinstance(image, (str, Path)):
        # Short strings might be file paths, long strings are almost certainly base64
        # File paths are typically < 4096 chars (OS limit), base64 images are much longer
        if isinstance(image, str) and len(image) > 4096:
            # Definitely base64 - skip file path check to avoid "File name too long" error
            return _load_base64_image(image, convert_mode)

        # Could be file path - check if it exists
        path = Path(image)
        try:
            if path.exists():
                return Image.open(path).convert(convert_mode)
        except OSError:
            # Path too long or other OS error - try as base64
            pass

        # Try as base64
        return _load_base64_image(str(image), convert_mode)

    raise ValueError(f"Unsupported image type: {type(image)}")


def _load_base64_image(data: str, convert_mode: str = "RGB") -> "PILImage.Image":
    """Load an image from base64-encoded string.

    Args:
        data: Base64-encoded image data, optionally with data URI prefix
        convert_mode: PIL image mode to convert to

    Returns:
        PIL Image object

    Raises:
        ValueError: If base64 decoding or image loading fails
    """
    from PIL import Image

    # Handle data URI format (e.g., "data:image/png;base64,...")
    if data.startswith("data:"):
        with contextlib.suppress(IndexError):
            data = data.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(data)
        return Image.open(BytesIO(img_bytes)).convert(convert_mode)
    except Exception as e:
        raise ValueError(f"Invalid base64 image data: {str(e)[:100]}") from e
