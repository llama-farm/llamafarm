"""
File I/O utilities for saving images and metadata.
"""

from pathlib import Path
from PIL import Image
import json
import logging

logger = logging.getLogger(__name__)


def save_image_with_metadata(image: Image.Image, filepath: Path, metadata: dict):
    """
    Save an image with accompanying JSON metadata.

    Args:
        image: PIL Image object
        filepath: Path to save the image
        metadata: Dictionary of metadata to save
    """
    # Save image
    image.save(filepath)
    logger.info(f"Saved image to {filepath}")

    # Save metadata as JSON sidecar
    metadata_path = filepath.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to {metadata_path}")
