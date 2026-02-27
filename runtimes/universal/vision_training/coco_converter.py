"""COCO JSON → YOLO format converter for training datasets.

Converts COCO-format annotations to YOLO txt format:
- COCO: [x, y, width, height] (absolute pixels)
- YOLO: [class_id, x_center, y_center, width, height] (normalized 0-1)

Supports both detection (bounding boxes) and classification tasks.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def convert_coco_to_yolo(
    coco_json_path: str | Path,
    output_dir: str | Path,
    images_dir: str | Path | None = None,
    split_ratio: float = 0.8,
) -> Path:
    """Convert COCO JSON annotations to YOLO dataset format.

    Args:
        coco_json_path: Path to COCO JSON annotation file
        output_dir: Directory to write YOLO dataset
        images_dir: Directory containing images (defaults to same dir as JSON)
        split_ratio: Train/val split ratio (default 0.8 = 80% train)

    Returns:
        Path to generated data.yaml for YOLO training

    Raises:
        FileNotFoundError: If COCO JSON or images not found
        ValueError: If COCO JSON format is invalid
    """
    coco_path = Path(coco_json_path)
    out = Path(output_dir)
    if not coco_path.exists():
        raise FileNotFoundError(f"COCO JSON not found: {coco_path}")

    with open(coco_path) as f:
        coco = json.load(f)

    _validate_coco(coco)

    # Build lookup tables
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    cat_id_to_idx = {cat["id"]: idx for idx, cat in enumerate(coco["categories"])}
    images_by_id: dict[int, dict] = {img["id"]: img for img in coco["images"]}

    # Group annotations by image
    annots_by_image: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        annots_by_image.setdefault(ann["image_id"], []).append(ann)

    # Resolve images directory
    if images_dir is None:
        images_dir = coco_path.parent / "images"
        if not images_dir.exists():
            images_dir = coco_path.parent
    images_dir = Path(images_dir)

    # Create output structure
    train_images = out / "images" / "train"
    train_labels = out / "labels" / "train"
    val_images = out / "images" / "val"
    val_labels = out / "labels" / "val"
    for d in [train_images, train_labels, val_images, val_labels]:
        d.mkdir(parents=True, exist_ok=True)

    # Split images
    image_ids = list(images_by_id.keys())
    split_idx = int(len(image_ids) * split_ratio)
    train_ids = set(image_ids[:split_idx])

    converted = 0
    skipped = 0
    seen_basenames: dict[str, str] = {}  # basename → original raw_filename (collision detection)

    for img_id, img_info in images_by_id.items():
        raw_filename = img_info["file_name"]
        # Sanitize: use basename only to prevent path traversal
        filename = Path(raw_filename).name
        if not filename or ".." in raw_filename or ":" in raw_filename or "\\" in raw_filename:
            logger.warning(f"Skipping image with suspicious filename: {raw_filename!r}")
            skipped += 1
            continue

        # Detect basename collisions from different subdirectories
        if filename in seen_basenames and seen_basenames[filename] != raw_filename:
            logger.warning(
                f"Skipping {raw_filename!r}: basename {filename!r} collides with "
                f"{seen_basenames[filename]!r}"
            )
            skipped += 1
            continue
        seen_basenames[filename] = raw_filename
        img_w = img_info.get("width", 0)
        img_h = img_info.get("height", 0)

        if img_w <= 0 or img_h <= 0:
            logger.warning(f"Skipping {filename}: missing width/height")
            skipped += 1
            continue

        # Determine split
        is_train = img_id in train_ids
        img_dst = train_images if is_train else val_images
        lbl_dst = train_labels if is_train else val_labels

        # Copy image if it exists — try original relative path first, then basename
        src_path = images_dir / raw_filename
        dst_name = filename  # Always use sanitized basename for output
        if src_path.exists() and src_path.resolve().is_relative_to(images_dir.resolve()):
            shutil.copy2(str(src_path), str(img_dst / dst_name))
        else:
            src_path = images_dir / filename
            if src_path.exists():
                shutil.copy2(str(src_path), str(img_dst / dst_name))
            else:
                logger.warning(f"Image not found: {raw_filename}")
                skipped += 1
                continue

        # Convert annotations to YOLO format
        annotations = annots_by_image.get(img_id, [])
        label_name = Path(filename).stem + ".txt"
        lines: list[str] = []

        for ann in annotations:
            if "bbox" not in ann:
                continue

            cat_idx = cat_id_to_idx.get(ann["category_id"])
            if cat_idx is None:
                continue

            # COCO bbox: [x, y, width, height] (absolute, top-left origin)
            bx, by, bw, bh = ann["bbox"]

            # YOLO bbox: [x_center, y_center, width, height] (normalized)
            x_center = (bx + bw / 2) / img_w
            y_center = (by + bh / 2) / img_h
            w_norm = bw / img_w
            h_norm = bh / img_h

            # Clamp to [0, 1]
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w_norm = max(0.0, min(1.0, w_norm))
            h_norm = max(0.0, min(1.0, h_norm))

            lines.append(f"{cat_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        (lbl_dst / label_name).write_text("\n".join(lines) + "\n" if lines else "")
        converted += 1

    # Generate data.yaml for YOLO training
    data_yaml = out / "data.yaml"
    data_config: dict[str, Any] = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {idx: name for idx, name in enumerate(categories.values())},
        "nc": len(categories),
    }
    data_yaml.write_text(yaml.dump(data_config, default_flow_style=False, sort_keys=False))

    logger.info(
        f"COCO→YOLO conversion complete: {converted} images converted, "
        f"{skipped} skipped, {len(categories)} classes → {data_yaml}"
    )
    return data_yaml


def is_coco_format(path: str | Path) -> bool:
    """Check if a file looks like COCO JSON format."""
    p = Path(path)
    if not p.exists() or p.suffix.lower() != ".json":
        return False
    try:
        with open(p) as f:
            data = json.load(f)
        return "images" in data and "annotations" in data and "categories" in data
    except (json.JSONDecodeError, KeyError):
        return False


def _validate_coco(coco: dict) -> None:
    """Validate basic COCO JSON structure."""
    required = ["images", "categories"]
    for key in required:
        if key not in coco:
            raise ValueError(f"Invalid COCO JSON: missing '{key}' field")
    if not isinstance(coco["images"], list):
        raise ValueError("Invalid COCO JSON: 'images' must be a list")
    if not isinstance(coco["categories"], list):
        raise ValueError("Invalid COCO JSON: 'categories' must be a list")
    for cat in coco["categories"]:
        if "id" not in cat or "name" not in cat:
            raise ValueError("Invalid COCO JSON: each category needs 'id' and 'name'")
