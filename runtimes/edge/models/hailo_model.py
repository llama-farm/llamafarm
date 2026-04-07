"""Hailo-10H YOLO detection model.

Uses the hailo_platform Python API to run YOLO inference on the Hailo-10H
AI accelerator via pre-compiled .hef models from the Hailo Model Zoo.

The .hef models include built-in NMS, so the output is already decoded
into bounding boxes, class IDs, and confidence scores.

Requires:
- Hailo-10H PCIe device (/dev/hailo0)
- hailort Python wheel (provides hailo_platform)
- Pre-compiled .hef model files (e.g., yolov11n.hef)
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from .vision_base import DetectionBox, DetectionModel, DetectionResult

logger = logging.getLogger(__name__)

# COCO class names (80 classes) — standard for YOLO models from Hailo Model Zoo
COCO_CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# Map friendly model names to .hef filenames
HAILO_VARIANTS: dict[str, str] = {
    "yolov8n": "yolov8n.hef",
    "yolov8s": "yolov8s.hef",
    "yolov8m": "yolov8m.hef",
    "yolov11n": "yolov11n.hef",
    "yolov11s": "yolov11s.hef",
}

# Default directory for .hef model files
DEFAULT_HEF_DIR = Path("/models")


def _letterbox(
    image: np.ndarray,
    target_size: tuple[int, int],
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize and letterbox an image to the target size.

    Maintains aspect ratio by padding with the specified color.

    Args:
        image: Input RGB image as numpy array (H, W, 3).
        target_size: (height, width) of the model input.
        color: Padding fill color (default: gray).

    Returns:
        Tuple of (letterboxed_image, scale, (pad_x, pad_y)).
    """
    h, w = image.shape[:2]
    th, tw = target_size

    scale = min(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)

    from PIL import Image

    resized = np.array(
        Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR)
    )

    canvas = np.full((th, tw, 3), color, dtype=np.uint8)
    pad_x = (tw - new_w) // 2
    pad_y = (th - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    return canvas, scale, (pad_x, pad_y)


def _parse_nms_output(
    output: np.ndarray,
    scale: float,
    pad: tuple[int, int],
    image_width: int,
    image_height: int,
    confidence_threshold: float,
    class_filter: set[int] | None = None,
    input_size: tuple[int, int] = (640, 640),
) -> list[DetectionBox]:
    """Parse NMS-decoded output from a Hailo .hef YOLO model.

    Hailo Model Zoo YOLO .hef files with built-in NMS produce a flat
    per-class buffer.  For 80 COCO classes with 100 max detections the
    raw shape is ``(40080,)`` = 80 × (1 + 100 × 5).

    Per-class layout (stride = 1 + max_det × 5):
        [count, y1, x1, y2, x2, score, y1, x1, y2, x2, score, …]

    ``count`` is the number of valid detections for that class.
    Each detection is 5 floats: ``[y_min, x_min, y_max, x_max, score]``.
    Coordinates are normalized (0.0–1.0) relative to the letterboxed input.

    Args:
        output: Raw float32 output array from Hailo inference.
        scale: Scale factor from letterboxing.
        pad: (pad_x, pad_y) offset from letterboxing.
        image_width: Original image width (for coordinate rescaling).
        image_height: Original image height (for coordinate rescaling).
        confidence_threshold: Minimum confidence to keep.
        class_filter: Optional set of class IDs to keep.
        input_size: (height, width) of the model input in pixels.

    Returns:
        List of DetectionBox instances in original image coordinates.
    """
    boxes: list[DetectionBox] = []
    flat = output.flatten()
    total = flat.size

    logger.debug(f"Hailo NMS output shape: {output.shape}, flat size: {total}")

    # Determine num_classes and max_det from buffer size.
    # Buffer = num_classes × (1 + max_det × 5).
    # COCO models use 80 classes; try common max_det values.
    num_classes = 0
    max_det = 0
    for nc in (80,):
        if total % nc != 0:
            continue
        stride = total // nc
        # stride = 1 + max_det * 5  →  (stride - 1) must be divisible by 5
        if (stride - 1) % 5 == 0:
            num_classes = nc
            max_det = (stride - 1) // 5
            break

    if num_classes == 0:
        logger.warning(
            f"Cannot parse Hailo NMS output: flat size {total} does not match "
            f"expected num_classes × (1 + max_det × 5) layout."
        )
        return boxes

    stride = 1 + max_det * 5
    logger.debug(
        f"Hailo NMS: {num_classes} classes, {max_det} max detections per class, "
        f"stride {stride}"
    )

    pad_x, pad_y = pad
    input_h, input_w = input_size

    for cls_id in range(num_classes):
        if class_filter is not None and cls_id not in class_filter:
            continue

        class_name = (
            COCO_CLASS_NAMES[cls_id]
            if cls_id < len(COCO_CLASS_NAMES)
            else f"class_{cls_id}"
        )

        offset = cls_id * stride
        n_det = int(flat[offset])
        if n_det <= 0:
            continue
        n_det = min(n_det, max_det)  # safety clamp

        for i in range(n_det):
            base = offset + 1 + i * 5
            y1_norm = float(flat[base])
            x1_norm = float(flat[base + 1])
            y2_norm = float(flat[base + 2])
            x2_norm = float(flat[base + 3])
            score = float(flat[base + 4])

            if score < confidence_threshold:
                continue

            logger.debug(
                f"Hailo det: class={class_name}({cls_id}) score={score:.4f} "
                f"norm=[{y1_norm:.4f}, {x1_norm:.4f}, {y2_norm:.4f}, {x2_norm:.4f}]"
            )

            # Convert normalized coords to pixel space in letterboxed image
            x1_px = x1_norm * input_w
            y1_px = y1_norm * input_h
            x2_px = x2_norm * input_w
            y2_px = y2_norm * input_h

            # Remove letterbox padding and rescale to original image
            x1 = max(0.0, (x1_px - pad_x) / scale)
            y1 = max(0.0, (y1_px - pad_y) / scale)
            x2 = min(float(image_width), (x2_px - pad_x) / scale)
            y2 = min(float(image_height), (y2_px - pad_y) / scale)

            logger.debug(
                f"Hailo mapped: px=({x1_px:.1f},{y1_px:.1f},{x2_px:.1f},{y2_px:.1f}) "
                f"-> orig=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})"
            )

            boxes.append(
                DetectionBox(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    class_name=class_name,
                    class_id=cls_id,
                    confidence=score,
                )
            )

    return boxes


class HailoYOLOModel(DetectionModel):
    """YOLO detection model running on Hailo-10H AI accelerator.

    Uses pre-compiled .hef models from the Hailo Model Zoo. These models
    include built-in NMS so the output is already decoded.

    Requires the hailo_platform package (provided by the hailort wheel).
    """

    def __init__(
        self,
        model_id: str = "yolov11n",
        device: str = "hailo",
        confidence_threshold: float = 0.5,
        hef_dir: str | Path | None = None,
        token: str | None = None,
    ):
        super().__init__(model_id, device="hailo", confidence_threshold=confidence_threshold, token=token)
        self._hef_dir = Path(hef_dir) if hef_dir else DEFAULT_HEF_DIR
        self._vdevice: Any = None
        self._infer_model: Any = None
        self._configured: Any = None
        self._input_shape: tuple[int, int] | None = None  # (height, width)
        self._hef_path: str | None = None

    def _resolve_hef_path(self) -> Path:
        """Resolve the .hef file path from model_id."""
        # Check variant map first
        hef_name = HAILO_VARIANTS.get(self.model_id)
        if hef_name:
            path = self._hef_dir / hef_name
            if path.exists():
                return path

        # Try model_id directly as filename
        if self.model_id.endswith(".hef"):
            path = self._hef_dir / Path(self.model_id).name
        else:
            path = self._hef_dir / f"{self.model_id}.hef"

        if path.exists():
            return path

        # Try VISION_MODELS_DIR fallback
        from utils.safe_home import get_data_dir
        vision_dir = get_data_dir() / "models" / "vision"
        alt_path = vision_dir / path.name
        if alt_path.exists():
            return alt_path

        raise FileNotFoundError(
            f"HEF model not found: tried {path} and {alt_path}. "
            f"Available in {self._hef_dir}: "
            f"{[f.name for f in self._hef_dir.glob('*.hef')] if self._hef_dir.exists() else '(dir missing)'}"
        )

    async def load(self) -> None:
        if self._loaded:
            return

        from hailo_platform import FormatType, VDevice

        logger.info(f"Loading Hailo model {self.model_id}")
        start = time.perf_counter()

        hef_path = self._resolve_hef_path()
        self._hef_path = str(hef_path)
        logger.info(f"HEF file: {hef_path}")

        def _load():
            vdevice = VDevice()
            infer_model = vdevice.create_infer_model(str(hef_path))
            infer_model.output().set_format_type(FormatType.FLOAT32)
            configured = infer_model.configure()

            # Extract input dimensions from the model
            input_vstream = infer_model.input()
            shape = input_vstream.shape  # (H, W, C) or (C, H, W)
            if len(shape) == 3:
                if shape[2] == 3:  # HWC
                    input_shape = (shape[0], shape[1])
                else:  # CHW
                    input_shape = (shape[1], shape[2])
            else:
                input_shape = (640, 640)  # Default YOLO input size
                logger.warning(f"Unexpected input shape {shape}, defaulting to 640x640")

            output_shape = infer_model.output().shape
            logger.info(
                f"Hailo model shapes — input: {shape}, output: {output_shape}"
            )

            return vdevice, infer_model, configured, input_shape

        self._vdevice, self._infer_model, self._configured, self._input_shape = (
            await asyncio.to_thread(_load)
        )

        self.class_names = list(COCO_CLASS_NAMES)
        self._loaded = True
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            f"Hailo model loaded in {elapsed:.0f}ms "
            f"(input: {self._input_shape[1]}x{self._input_shape[0]}, "
            f"{len(self.class_names)} classes)"
        )

    async def unload(self) -> None:
        if self._configured is not None:
            del self._configured
            self._configured = None
        if self._infer_model is not None:
            del self._infer_model
            self._infer_model = None
        if self._vdevice is not None:
            del self._vdevice
            self._vdevice = None
        self._loaded = False
        logger.info(f"Hailo model unloaded: {self.model_id}")

    async def detect(
        self,
        image: bytes | np.ndarray,
        confidence_threshold: float | None = None,
        classes: list[str] | None = None,
    ) -> DetectionResult:
        if not self._loaded or self._configured is None:
            await self.load()

        start = time.perf_counter()
        img_array = self._image_to_numpy(image)
        height, width = img_array.shape[:2]
        conf = confidence_threshold if confidence_threshold is not None else self.confidence_threshold

        # Build class filter
        class_filter: set[int] | None = None
        if classes:
            class_filter = {i for i, n in enumerate(self.class_names) if n in classes}

        # Preprocess: letterbox to model input dimensions
        input_h, input_w = self._input_shape or (640, 640)
        letterboxed, scale, pad = _letterbox(img_array, (input_h, input_w))

        # Ensure uint8 RGB contiguous array
        input_data = np.ascontiguousarray(letterboxed, dtype=np.uint8)

        # Run inference on Hailo
        def _infer():
            bindings = self._configured.create_bindings()
            bindings.input().set_buffer(input_data)
            output_buffer = np.empty(
                self._infer_model.output().shape, dtype=np.float32
            )
            bindings.output().set_buffer(output_buffer)
            self._configured.run([bindings], 5000)
            return output_buffer

        output = await asyncio.to_thread(_infer)
        inference_time = (time.perf_counter() - start) * 1000

        # Parse NMS output into detection boxes
        boxes = _parse_nms_output(
            output, scale, pad, width, height, conf, class_filter,
            input_size=(input_h, input_w),
        )

        return DetectionResult(
            confidence=max((b.confidence for b in boxes), default=0.0),
            inference_time_ms=inference_time,
            model_name=self.model_id,
            boxes=boxes,
            class_names=list({b.class_name for b in boxes}),
            image_width=width,
            image_height=height,
        )

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update({
            "backend": "hailo",
            "variant": self.model_id,
            "hef_path": self._hef_path,
            "input_shape": self._input_shape,
            "num_classes": len(self.class_names),
        })
        return info
