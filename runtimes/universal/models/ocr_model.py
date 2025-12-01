"""
OCR model wrapper for text extraction from images.

Supports multiple OCR backends:
- surya: Best accuracy, layout-aware (recommended)
- easyocr: Good multilingual support, widely used
- paddleocr: Fast, excellent for Asian languages
- tesseract: Classic, widely deployed, no GPU needed
"""

import base64
import io
import logging
from dataclasses import dataclass
from typing import Any, Literal, Optional

from PIL import Image

from .base import BaseModel

logger = logging.getLogger(__name__)

OCRBackend = Literal["surya", "easyocr", "paddleocr", "tesseract"]


@dataclass
class BoundingBox:
    """Bounding box for detected text."""

    x1: float
    y1: float
    x2: float
    y2: float
    text: str
    confidence: float


@dataclass
class OCRResult:
    """Result from OCR processing."""

    text: str
    confidence: float
    boxes: list[BoundingBox] | None = None
    language: str | None = None


class OCRModel(BaseModel):
    """Wrapper for OCR models with multiple backend support.

    Backends:
    - surya: Modern, transformer-based OCR with layout detection
    - easyocr: Popular, supports 80+ languages
    - paddleocr: Fast, optimized for production
    - tesseract: Classic OCR engine, CPU-only
    """

    SUPPORTED_BACKENDS = ["surya", "easyocr", "paddleocr", "tesseract"]

    def __init__(
        self,
        model_id: str,
        device: str,
        backend: OCRBackend = "surya",
        languages: list[str] | None = None,
    ):
        """Initialize OCR model.

        Args:
            model_id: Model identifier (used for caching, can be backend name)
            device: Target device (cuda/mps/cpu)
            backend: OCR backend to use
            languages: List of language codes (e.g., ['en', 'fr'])
        """
        super().__init__(model_id, device)
        self.backend = backend
        self.languages = languages or ["en"]
        self.model_type = f"ocr_{backend}"
        self.supports_streaming = False

        # Backend-specific components
        self._reader = None  # EasyOCR reader
        self._ocr = None  # PaddleOCR instance
        self._surya_model = None  # Surya model
        self._surya_processor = None  # Surya processor

    async def load(self) -> None:
        """Load the OCR model based on selected backend."""
        logger.info(f"Loading OCR model: {self.backend} on {self.device}")

        if self.backend == "surya":
            await self._load_surya()
        elif self.backend == "easyocr":
            await self._load_easyocr()
        elif self.backend == "paddleocr":
            await self._load_paddleocr()
        elif self.backend == "tesseract":
            await self._load_tesseract()
        else:
            raise ValueError(f"Unsupported OCR backend: {self.backend}")

        logger.info(f"OCR model loaded: {self.backend}")

    async def _load_surya(self) -> None:
        """Load Surya OCR model."""
        try:
            from surya.model.detection.model import (
                load_model as load_det_model,
                load_processor as load_det_processor,
            )
            from surya.model.recognition.model import load_model as load_rec_model
            from surya.model.recognition.processor import load_processor as load_rec_processor

            # Load detection model
            self._surya_det_model = load_det_model()
            self._surya_det_processor = load_det_processor()

            # Load recognition model
            self._surya_rec_model = load_rec_model()
            self._surya_rec_processor = load_rec_processor()

            # Move to device if not CPU
            if self.device != "cpu":
                self._surya_det_model = self._surya_det_model.to(self.device)
                self._surya_rec_model = self._surya_rec_model.to(self.device)

        except ImportError as e:
            raise ImportError(
                "Surya OCR not installed. Install with: uv pip install surya-ocr"
            ) from e

    async def _load_easyocr(self) -> None:
        """Load EasyOCR reader."""
        try:
            import easyocr

            gpu = self.device in ("cuda", "mps")
            self._reader = easyocr.Reader(self.languages, gpu=gpu)

        except ImportError as e:
            raise ImportError(
                "EasyOCR not installed. Install with: uv pip install easyocr"
            ) from e

    async def _load_paddleocr(self) -> None:
        """Load PaddleOCR instance."""
        try:
            from paddleocr import PaddleOCR

            use_gpu = self.device == "cuda"
            # Map language codes
            lang = self.languages[0] if self.languages else "en"
            self._ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu)

        except ImportError as e:
            raise ImportError(
                "PaddleOCR not installed. Install with: uv pip install paddleocr"
            ) from e

    async def _load_tesseract(self) -> None:
        """Verify Tesseract is available."""
        try:
            import pytesseract

            # Test that tesseract binary is available
            pytesseract.get_tesseract_version()

        except ImportError as e:
            raise ImportError(
                "pytesseract not installed. Install with: uv pip install pytesseract"
            ) from e
        except Exception as e:
            raise RuntimeError(
                "Tesseract binary not found. Install tesseract-ocr system package."
            ) from e

    async def recognize(
        self,
        images: list[str | bytes],
        languages: list[str] | None = None,
        detect_layout: bool = True,
        return_boxes: bool = False,
    ) -> list[OCRResult]:
        """Extract text from images.

        Args:
            images: List of images (base64 strings or raw bytes)
            languages: Override default languages for this request
            detect_layout: Whether to detect document layout
            return_boxes: Whether to return bounding boxes

        Returns:
            List of OCRResult objects
        """
        results = []
        langs = languages or self.languages

        for img_data in images:
            # Decode image
            pil_image = self._decode_image(img_data)

            # Run OCR based on backend
            if self.backend == "surya":
                result = await self._recognize_surya(pil_image, return_boxes)
            elif self.backend == "easyocr":
                result = await self._recognize_easyocr(pil_image, return_boxes)
            elif self.backend == "paddleocr":
                result = await self._recognize_paddleocr(pil_image, return_boxes)
            elif self.backend == "tesseract":
                result = await self._recognize_tesseract(pil_image, langs, return_boxes)
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")

            results.append(result)

        return results

    def _decode_image(self, img_data: str | bytes) -> Image.Image:
        """Decode image from base64 string or bytes."""
        if isinstance(img_data, str):
            # Handle base64 with or without data URI prefix
            if img_data.startswith("data:"):
                # Remove data URI prefix (e.g., "data:image/png;base64,")
                img_data = img_data.split(",", 1)[1]
            img_bytes = base64.b64decode(img_data)
        else:
            img_bytes = img_data

        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    async def _recognize_surya(
        self, image: Image.Image, return_boxes: bool
    ) -> OCRResult:
        """Run Surya OCR."""
        from surya.detection import batch_text_detection
        from surya.recognition import batch_recognition

        # Detect text regions
        det_results = batch_text_detection(
            [image], self._surya_det_model, self._surya_det_processor
        )

        # Recognize text in detected regions
        rec_results = batch_recognition(
            [image],
            [det_results[0]],
            self._surya_rec_model,
            self._surya_rec_processor,
        )

        # Extract results
        text_lines = []
        boxes = []
        confidences = []

        for line in rec_results[0].text_lines:
            text_lines.append(line.text)
            confidences.append(line.confidence)

            if return_boxes and line.bbox:
                boxes.append(
                    BoundingBox(
                        x1=line.bbox[0],
                        y1=line.bbox[1],
                        x2=line.bbox[2],
                        y2=line.bbox[3],
                        text=line.text,
                        confidence=line.confidence,
                    )
                )

        full_text = "\n".join(text_lines)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return OCRResult(
            text=full_text,
            confidence=avg_confidence,
            boxes=boxes if return_boxes else None,
        )

    async def _recognize_easyocr(
        self, image: Image.Image, return_boxes: bool
    ) -> OCRResult:
        """Run EasyOCR."""
        import numpy as np

        # Convert PIL to numpy array
        img_array = np.array(image)

        # Run OCR
        results = self._reader.readtext(img_array)

        text_lines = []
        boxes = []
        confidences = []

        for bbox, text, confidence in results:
            text_lines.append(text)
            confidences.append(confidence)

            if return_boxes:
                # EasyOCR returns 4 corner points, convert to x1,y1,x2,y2
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                boxes.append(
                    BoundingBox(
                        x1=min(x_coords),
                        y1=min(y_coords),
                        x2=max(x_coords),
                        y2=max(y_coords),
                        text=text,
                        confidence=confidence,
                    )
                )

        full_text = " ".join(text_lines)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return OCRResult(
            text=full_text,
            confidence=avg_confidence,
            boxes=boxes if return_boxes else None,
        )

    async def _recognize_paddleocr(
        self, image: Image.Image, return_boxes: bool
    ) -> OCRResult:
        """Run PaddleOCR."""
        import numpy as np

        # Convert PIL to numpy array
        img_array = np.array(image)

        # Run OCR
        results = self._ocr.ocr(img_array, cls=True)

        text_lines = []
        boxes = []
        confidences = []

        if results and results[0]:
            for line in results[0]:
                bbox, (text, confidence) = line
                text_lines.append(text)
                confidences.append(confidence)

                if return_boxes:
                    # PaddleOCR returns 4 corner points
                    x_coords = [p[0] for p in bbox]
                    y_coords = [p[1] for p in bbox]
                    boxes.append(
                        BoundingBox(
                            x1=min(x_coords),
                            y1=min(y_coords),
                            x2=max(x_coords),
                            y2=max(y_coords),
                            text=text,
                            confidence=confidence,
                        )
                    )

        full_text = " ".join(text_lines)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return OCRResult(
            text=full_text,
            confidence=avg_confidence,
            boxes=boxes if return_boxes else None,
        )

    async def _recognize_tesseract(
        self, image: Image.Image, languages: list[str], return_boxes: bool
    ) -> OCRResult:
        """Run Tesseract OCR."""
        import pytesseract

        # Join languages with + for tesseract
        lang_str = "+".join(languages)

        if return_boxes:
            # Get detailed output with bounding boxes
            data = pytesseract.image_to_data(
                image, lang=lang_str, output_type=pytesseract.Output.DICT
            )

            text_lines = []
            boxes = []
            confidences = []

            for i, text in enumerate(data["text"]):
                if text.strip():
                    conf = float(data["conf"][i])
                    if conf > 0:  # Filter out low-confidence detections
                        text_lines.append(text)
                        confidences.append(conf / 100.0)  # Normalize to 0-1

                        boxes.append(
                            BoundingBox(
                                x1=data["left"][i],
                                y1=data["top"][i],
                                x2=data["left"][i] + data["width"][i],
                                y2=data["top"][i] + data["height"][i],
                                text=text,
                                confidence=conf / 100.0,
                            )
                        )

            full_text = " ".join(text_lines)
            avg_confidence = (
                sum(confidences) / len(confidences) if confidences else 0.0
            )

            return OCRResult(
                text=full_text,
                confidence=avg_confidence,
                boxes=boxes,
            )
        else:
            # Simple text extraction
            text = pytesseract.image_to_string(image, lang=lang_str)
            return OCRResult(
                text=text.strip(),
                confidence=0.9,  # Tesseract doesn't provide overall confidence
                boxes=None,
            )

    async def unload(self) -> None:
        """Unload the OCR model and free resources."""
        logger.info(f"Unloading OCR model: {self.backend}")

        # Clear backend-specific components
        self._reader = None
        self._ocr = None
        self._surya_det_model = None
        self._surya_det_processor = None
        self._surya_rec_model = None
        self._surya_rec_processor = None

        # Call parent unload for GPU cleanup
        await super().unload()

        logger.info(f"OCR model unloaded: {self.backend}")

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded OCR model."""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "backend": self.backend,
            "device": self.device,
            "languages": self.languages,
            "supports_streaming": self.supports_streaming,
        }
