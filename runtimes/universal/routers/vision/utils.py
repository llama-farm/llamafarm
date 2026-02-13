"""Shared utilities for vision routers."""

import base64

from fastapi import HTTPException


def decode_base64_image(image_str: str) -> bytes:
    """Decode base64 image string to bytes.

    Handles both raw base64 and data URI format (data:image/jpeg;base64,...).

    Raises:
        HTTPException: If the base64 data is malformed.
    """
    if image_str.startswith("data:"):
        if "," not in image_str:
            raise HTTPException(
                status_code=400,
                detail="Malformed data URI: missing comma separator",
            )
        _, base64_data = image_str.split(",", 1)
    else:
        base64_data = image_str

    try:
        return base64.b64decode(base64_data)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid base64 image data",
        )
