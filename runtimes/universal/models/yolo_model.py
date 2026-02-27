"""YOLO-based object detection model. Supports YOLOv8/v11 via ultralytics."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from .vision_base import DetectionBox, DetectionModel, DetectionResult

if TYPE_CHECKING:
    from ultralytics import YOLO

logger = logging.getLogger(__name__)

YOLO_VARIANTS = {
    "yolov8n": "yolov8n.pt", "yolov8s": "yolov8s.pt", "yolov8m": "yolov8m.pt",
    "yolov8l": "yolov8l.pt", "yolov8x": "yolov8x.pt",
    "yolov11n": "yolo11n.pt", "yolov11s": "yolo11s.pt", "yolov11m": "yolo11m.pt",
}


class YOLOModel(DetectionModel):
    """YOLO object detection model wrapper."""

    def __init__(self, model_id: str = "yolov8n", device: str = "auto",
                 confidence_threshold: float = 0.5, token: str | None = None):
        super().__init__(model_id, device, confidence_threshold, token)
        self.yolo: YOLO | None = None
        self._model_path: str | None = None

    async def load(self) -> None:
        if self._loaded:
            return
        from ultralytics import YOLO

        self.device = self._resolve_device(self.device)
        logger.info(f"Loading YOLO model {self.model_id} on {self.device}")
        start = time.perf_counter()

        if self.model_id in YOLO_VARIANTS:
            self._model_path = YOLO_VARIANTS[self.model_id]
        elif Path(self.model_id).exists():
            # Validate path — must resolve within home/.llamafarm or cwd
            resolved = Path(self.model_id).resolve()
            allowed_roots = [Path.home() / ".llamafarm", Path.cwd()]
            if not any(str(resolved).startswith(str(r.resolve())) for r in allowed_roots):
                raise ValueError(f"Model path outside allowed directories: {self.model_id}")
            self._model_path = str(resolved)
        else:
            # Basename only for dynamic model IDs (no path components)
            safe_id = Path(self.model_id).name
            if safe_id != self.model_id:
                raise ValueError(f"Invalid model_id: {self.model_id}")
            self._model_path = f"{safe_id}.pt"

        self.yolo = YOLO(self._model_path)
        if self.device != "cpu":
            self.yolo.to(self.device)

        self.class_names = list(self.yolo.names.values()) if hasattr(self.yolo, "names") else []
        self._loaded = True
        logger.info(f"YOLO loaded in {(time.perf_counter() - start) * 1000:.0f}ms ({len(self.class_names)} classes)")

    async def unload(self) -> None:
        if self.yolo is not None:
            del self.yolo
            self.yolo = None
        self._loaded = False
        await super().unload()

    async def detect(self, image: bytes | np.ndarray,
                     confidence_threshold: float | None = None,
                     classes: list[str] | None = None) -> DetectionResult:
        if not self._loaded or self.yolo is None:
            await self.load()

        start = time.perf_counter()
        img_array = self._image_to_numpy(image)
        height, width = img_array.shape[:2]
        conf = confidence_threshold if confidence_threshold is not None else self.confidence_threshold

        class_indices = None
        if classes:
            class_indices = [i for i, n in enumerate(self.class_names) if n in classes]

        results = await asyncio.to_thread(
            self.yolo, img_array, conf=conf, classes=class_indices, verbose=False
        )
        inference_time = (time.perf_counter() - start) * 1000

        boxes: list[DetectionBox] = []
        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0].cpu().numpy())
                boxes.append(DetectionBox(
                    x1=float(xyxy[0]), y1=float(xyxy[1]),
                    x2=float(xyxy[2]), y2=float(xyxy[3]),
                    class_name=self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}",
                    class_id=cls_id,
                    confidence=float(box.conf[0].cpu().numpy()),
                ))

        return DetectionResult(
            confidence=max((b.confidence for b in boxes), default=0.0),
            inference_time_ms=inference_time, model_name=self.model_id,
            boxes=boxes, class_names=list({b.class_name for b in boxes}),
            image_width=width, image_height=height,
        )

    async def train(self, dataset_path: str, epochs: int = 10,
                    batch_size: int = 16, **kwargs) -> dict:
        if not self._loaded or self.yolo is None:
            await self.load()

        logger.info(f"Starting YOLO training: {epochs} epochs, batch {batch_size}")
        train_args = {
            "data": dataset_path, "epochs": epochs, "batch": batch_size,
            "device": self.device if self.device != "auto" else None,
            "imgsz": kwargs.get("imgsz", 640),
            "patience": kwargs.get("patience", 50),
            "save": True, "verbose": kwargs.get("verbose", True),
        }
        results = await asyncio.to_thread(self.yolo.train, **train_args)

        metrics = {}
        if hasattr(results, "results_dict"):
            metrics = results.results_dict
        return {
            "metrics": metrics, "epochs": epochs,
            "model_path": str(results.save_dir) if hasattr(results, "save_dir") else None,
        }

    async def export(self, format: Literal["onnx", "coreml", "tensorrt", "tflite", "openvino"],
                     output_path: str, **kwargs) -> str:
        if not self._loaded or self.yolo is None:
            await self.load()

        logger.info(f"Exporting YOLO model to {format}")
        format_map = {"onnx": "onnx", "coreml": "coreml", "tensorrt": "engine",
                      "tflite": "tflite", "openvino": "openvino"}

        export_path = self.yolo.export(
            format=format_map.get(format, format),
            half=kwargs.get("half", False),
            int8=kwargs.get("int8", False),
            simplify=kwargs.get("simplify", True),
        )

        if output_path and Path(output_path).is_dir():
            import shutil
            final = Path(output_path) / Path(export_path).name
            shutil.move(export_path, final)
            export_path = str(final)

        return str(export_path)

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update({"variant": self.model_id, "num_classes": len(self.class_names),
                     "model_path": self._model_path})
        return info
