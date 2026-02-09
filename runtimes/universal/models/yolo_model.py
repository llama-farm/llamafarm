"""
YOLO-based object detection model.

Supports:
- YOLOv8 (n, s, m, l, x variants)
- YOLOv11 (n, s, m variants)
- Custom fine-tuned models
- Export to ONNX, CoreML, TensorRT, TFLite
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from .vision_base import (
    DetectionBox,
    DetectionModel,
    DetectionResult,
)

if TYPE_CHECKING:
    from ultralytics import YOLO

logger = logging.getLogger(__name__)


# Supported YOLO variants
YOLO_VARIANTS = {
    # YOLOv8
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
    "yolov8l": "yolov8l.pt",
    "yolov8x": "yolov8x.pt",
    # YOLOv11
    "yolov11n": "yolo11n.pt",
    "yolov11s": "yolo11s.pt",
    "yolov11m": "yolo11m.pt",
}

# Default class names for COCO dataset (80 classes)
COCO_CLASSES = [
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
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


class YOLOModel(DetectionModel):
    """YOLO-based object detection model.

    Supports YOLOv8 and YOLOv11 variants via the Ultralytics library.

    Example:
        ```python
        model = YOLOModel("yolov8n")
        await model.load()

        result = await model.detect(image_bytes)
        for box in result.boxes:
            print(f"{box.class_name}: {box.confidence:.2f}")
        ```
    """

    def __init__(
        self,
        model_id: str = "yolov8n",
        device: str = "auto",
        confidence_threshold: float = 0.5,
        token: str | None = None,
    ):
        """Initialize YOLO model.

        Args:
            model_id: Model variant (yolov8n, yolov8s, etc.) or path to custom model
            device: Device to run on (auto, cpu, cuda, mps)
            confidence_threshold: Default confidence threshold for detections
            token: HuggingFace token (unused for YOLO, kept for API consistency)
        """
        super().__init__(model_id, device, confidence_threshold, token)
        self.yolo: YOLO | None = None
        self._model_path: str | None = None

    async def load(self) -> None:
        """Load the YOLO model.

        Downloads the model if not cached locally.
        """
        if self._loaded:
            logger.debug(f"YOLO model {self.model_id} already loaded")
            return

        from ultralytics import YOLO

        # Resolve device
        self.device = self._resolve_device(self.device)
        logger.info(f"Loading YOLO model {self.model_id} on {self.device}")

        start_time = time.perf_counter()

        # Determine model path
        if self.model_id in YOLO_VARIANTS:
            # Standard variant - Ultralytics will download if needed
            self._model_path = YOLO_VARIANTS[self.model_id]
        elif Path(self.model_id).exists():
            # Custom model path
            self._model_path = self.model_id
        else:
            # Assume it's a variant name without .pt
            self._model_path = f"{self.model_id}.pt"

        # Load model
        self.yolo = YOLO(self._model_path)

        # Move to device
        if self.device != "cpu":
            self.yolo.to(self.device)

        # Get class names
        if hasattr(self.yolo, "names"):
            self.class_names = list(self.yolo.names.values())
        else:
            self.class_names = COCO_CLASSES

        self._loaded = True
        load_time = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"YOLO model {self.model_id} loaded in {load_time:.1f}ms "
            f"({len(self.class_names)} classes)"
        )

    async def unload(self) -> None:
        """Unload the YOLO model."""
        if self.yolo is not None:
            del self.yolo
            self.yolo = None
        self._loaded = False
        await super().unload()

    async def detect(
        self,
        image: bytes | np.ndarray,
        confidence_threshold: float | None = None,
        classes: list[str] | None = None,
    ) -> DetectionResult:
        """Detect objects in an image.

        Args:
            image: Image as bytes (JPEG/PNG) or numpy array
            confidence_threshold: Override default threshold
            classes: Filter to specific class names

        Returns:
            DetectionResult with bounding boxes
        """
        if not self._loaded or self.yolo is None:
            await self.load()

        start_time = time.perf_counter()

        # Convert to numpy if needed
        img_array = self._image_to_numpy(image)
        height, width = img_array.shape[:2]

        # Determine confidence threshold
        conf_thresh = confidence_threshold or self.confidence_threshold

        # Get class indices to filter
        class_indices = None
        if classes:
            class_indices = [
                i for i, name in enumerate(self.class_names)
                if name in classes
            ]
            if not class_indices:
                logger.warning(f"No matching classes found for filter: {classes}")

        # Run inference
        results = self.yolo(
            img_array,
            conf=conf_thresh,
            classes=class_indices,
            verbose=False,
        )

        inference_time = (time.perf_counter() - start_time) * 1000

        # Parse results
        boxes: list[DetectionBox] = []

        if results and len(results) > 0:
            result = results[0]  # First image result

            if result.boxes is not None:
                # Prepare masks if available
                masks_xy = []
                if result.masks is not None:
                    # masks.xy is a list of np.ndarray or a single np.ndarray depending on version/content
                    # In recent ultralytics, result.masks.xy is a list of arrays, one per box
                    try:
                        masks_xy = result.masks.xy
                    except AttributeError:
                        masks_xy = []

                for i, box in enumerate(result.boxes):
                    # Get box coordinates (xyxy format)
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())

                    # Get class name
                    cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
                    
                    # Get mask if available
                    mask = None
                    if i < len(masks_xy):
                        # Convert numpy array to list of lists for JSON serialization
                        mask_np = masks_xy[i]
                        if isinstance(mask_np, np.ndarray):
                            mask = mask_np.tolist()
                        elif isinstance(mask_np, list):
                            mask = mask_np

                    boxes.append(DetectionBox(
                        x1=float(xyxy[0]),
                        y1=float(xyxy[1]),
                        x2=float(xyxy[2]),
                        y2=float(xyxy[3]),
                        class_name=cls_name,
                        class_id=cls_id,
                        confidence=conf,
                        mask=mask,
                    ))

        return DetectionResult(
            confidence=max((b.confidence for b in boxes), default=0.0),
            inference_time_ms=inference_time,
            model_name=self.model_id,
            boxes=boxes,
            class_names=list(set(b.class_name for b in boxes)),
            image_width=width,
            image_height=height,
        )

    async def train(
        self,
        dataset_path: str,
        epochs: int = 10,
        batch_size: int = 16,
        **kwargs,
    ) -> dict:
        """Fine-tune the YOLO model on a custom dataset.

        Args:
            dataset_path: Path to dataset in YOLO format (with data.yaml)
            epochs: Number of training epochs
            batch_size: Training batch size
            **kwargs: Additional training arguments (imgsz, lr0, etc.)

        Returns:
            Training metrics dictionary
        """
        if not self._loaded or self.yolo is None:
            await self.load()

        logger.info(f"Starting YOLO training: {epochs} epochs, batch size {batch_size}")

        # Default training args
        train_args = {
            "data": dataset_path,
            "epochs": epochs,
            "batch": batch_size,
            "device": self.device if self.device != "auto" else None,
            "imgsz": kwargs.get("imgsz", 640),
            "patience": kwargs.get("patience", 50),
            "save": True,
            "plots": kwargs.get("plots", False),
            "verbose": kwargs.get("verbose", True),
        }

        # Add any additional kwargs (filter out non-YOLO args)
        _non_yolo_keys = {"use_ewc", "ewc_lambda", "use_replay", "replay_ratio", "model"}
        for key, value in kwargs.items():
            if key not in train_args and key not in _non_yolo_keys:
                train_args[key] = value

        # Run training
        results = self.yolo.train(**train_args)

        # Extract metrics
        metrics = {}
        if hasattr(results, "results_dict"):
            metrics = results.results_dict
        elif hasattr(results, "maps"):
            metrics["mAP50"] = float(results.maps[0]) if len(results.maps) > 0 else 0
            metrics["mAP50-95"] = float(results.maps.mean()) if len(results.maps) > 0 else 0

        return {
            "metrics": metrics,
            "epochs": epochs,
            "model_path": str(results.save_dir) if hasattr(results, "save_dir") else None,
        }

    async def export(
        self,
        format: Literal["onnx", "coreml", "tensorrt", "tflite", "openvino"],
        output_path: str,
        **kwargs,
    ) -> str:
        """Export model to deployment format.

        Args:
            format: Export format
            output_path: Output directory (filename auto-generated)
            **kwargs: Format-specific options (half, int8, etc.)

        Returns:
            Path to exported model
        """
        if not self._loaded or self.yolo is None:
            await self.load()

        logger.info(f"Exporting YOLO model to {format}")

        # Map format names
        format_map = {
            "onnx": "onnx",
            "coreml": "coreml",
            "tensorrt": "engine",
            "tflite": "tflite",
            "openvino": "openvino",
        }

        ultralytics_format = format_map.get(format, format)

        # Export options
        export_args = {
            "format": ultralytics_format,
            "half": kwargs.get("half", format in ("tensorrt", "coreml")),
            "int8": kwargs.get("int8", False),
            "dynamic": kwargs.get("dynamic", False),
            "simplify": kwargs.get("simplify", True),
        }

        # Run export
        export_path = self.yolo.export(**export_args)

        # Move to output path if specified
        if output_path and Path(output_path).is_dir():
            import shutil
            final_path = Path(output_path) / Path(export_path).name
            shutil.move(export_path, final_path)
            export_path = str(final_path)

        logger.info(f"Model exported to: {export_path}")
        return str(export_path)

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        info = super().get_model_info()
        info.update({
            "variant": self.model_id,
            "num_classes": len(self.class_names),
            "class_names": self.class_names[:10],  # First 10 classes
            "model_path": self._model_path,
        })
        return info
