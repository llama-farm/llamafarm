"""Shared utilities for vision routers."""

import base64

from fastapi import HTTPException


def decode_base64_image(image_str: str) -> bytes:
    """Decode base64 image string to bytes. Handles data URI format and line-wrapped base64."""
    if image_str.startswith("data:"):
        if "," not in image_str:
            raise HTTPException(status_code=400, detail="Malformed data URI")
        _, base64_data = image_str.split(",", 1)
    else:
        base64_data = image_str
    # Strip whitespace — handles trailing newlines from tools like `jq -Rs` and
    # line-wrapped base64 produced by GNU/BSD `base64` commands.
    base64_data = "".join(base64_data.split())
    try:
        return base64.b64decode(base64_data, validate=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid base64 image data") from e
