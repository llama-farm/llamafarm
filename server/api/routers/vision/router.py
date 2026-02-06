"""
Vision Router - Endpoints for OCR and Document extraction.

Provides access to:
- OCR (text extraction from images/PDFs)
- Document Extraction (structured data from forms/invoices)

All endpoints accept multipart form data with either:
- file: Upload a PDF or image file directly
- images: Base64-encoded image data URIs (comma-separated or JSON array)
"""

import json
import logging
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from server.services.universal_runtime_service import UniversalRuntimeService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vision", tags=["vision"])


# Supported file extensions for upload
SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
}

# Type alias for optional file upload (avoids B008 linting error)
OptionalFileUpload = Annotated[
    UploadFile | None, File(description="PDF or image file to process")
]


async def _get_images_from_input(
    file: UploadFile | None,
    images: str | None,
) -> list[str]:
    """Extract base64 images from either file upload or images parameter.

    Args:
        file: Optional uploaded file (PDF or image)
        images: Optional base64 images (JSON array or comma-separated)

    Returns:
        List of base64 data URIs

    Raises:
        HTTPException: If neither file nor images provided, or file type unsupported
    """
    from pathlib import Path

    from core.image_utils import (
        IMAGE_MIME_TYPES,
        image_bytes_to_base64,
        pdf_bytes_to_base64_images,
    )

    # Case 1: File upload provided
    if file is not None and file.filename:
        filename = file.filename
        ext = Path(filename).suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            )

        content = await file.read()

        if ext == ".pdf":
            return pdf_bytes_to_base64_images(content)
        else:
            mime_type = IMAGE_MIME_TYPES.get(ext, "image/png")
            return [image_bytes_to_base64(content, mime_type)]

    # Case 2: Base64 images provided
    if images:
        # Try to parse as JSON array first
        try:
            parsed = json.loads(images)
            if isinstance(parsed, list):
                # Validate all elements are strings (base64 data URIs)
                if not all(isinstance(item, str) for item in parsed):
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid 'images' format: all array elements must be base64 data URI strings",
                    )
                if not parsed:
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid 'images' format: array cannot be empty",
                    )
                return parsed
        except json.JSONDecodeError:
            pass

        # Fall back to comma-separated (but be careful with base64 which contains commas in data)
        # If it starts with "data:", treat as single image or split on "],["
        if images.startswith("data:"):
            # Single base64 image
            return [images]
        else:
            # Try splitting - but this is tricky with base64
            # Best to require JSON array format for multiple images
            return [images]

    # Neither provided
    raise HTTPException(
        status_code=400,
        detail="Either 'file' (upload) or 'images' (base64) must be provided",
    )


# =============================================================================
# OCR Endpoint
# =============================================================================


@router.post("/ocr")
async def extract_text(
    file: OptionalFileUpload = None,
    images: str | None = Form(
        default=None,
        description='Base64-encoded images as JSON array (e.g., ["data:image/png;base64,..."])',
    ),
    model: str = Form(
        default="surya",
        description="OCR backend: surya, easyocr, paddleocr, tesseract",
    ),
    languages: str = Form(
        default="en",
        description="Comma-separated language codes (e.g., 'en,fr')",
    ),
    return_boxes: bool = Form(
        default=False,
        description="Return bounding boxes for detected text",
    ),
    parse_by_page: bool = Form(
        default=False,
        description="If true, return separate results per page. If false (default), combine all text into single result.",
    ),
) -> dict[str, Any]:
    """OCR endpoint for text extraction from images.

    Accepts either a file upload OR base64-encoded images.

    Supports multiple OCR backends:
    - surya: Best accuracy, transformer-based, layout-aware (recommended)
    - easyocr: Good multilingual support (80+ languages), widely used
    - paddleocr: Fast, optimized for production, excellent for Asian languages
    - tesseract: Classic OCR engine, CPU-only, widely deployed

    Example with file upload (curl):
    ```bash
    curl -X POST http://localhost:14345/v1/vision/ocr \\
      -F "file=@document.pdf" \\
      -F "model=easyocr" \\
      -F "languages=en"
    ```

    Example with base64 images (curl):
    ```bash
    curl -X POST http://localhost:14345/v1/vision/ocr \\
      -F 'images=["data:image/png;base64,iVBORw0KGgo..."]' \\
      -F "model=surya" \\
      -F "languages=en"
    ```

    Response:
    ```json
    {
        "object": "list",
        "data": [
            {
                "index": 0,
                "text": "Extracted text from image...",
                "confidence": 0.95
            }
        ],
        "model": "surya",
        "usage": {"images_processed": 1}
    }
    ```
    """
    # Get images from file or base64 input
    image_list = await _get_images_from_input(file, images)

    # Parse languages
    lang_list = [lang.strip() for lang in languages.split(",") if lang.strip()]

    result = await UniversalRuntimeService.ocr(
        model=model,
        images=image_list,
        languages=lang_list if lang_list else None,
        return_boxes=return_boxes,
    )

    # If not parsing by page and we have multiple results, combine them
    if not parse_by_page and len(result.get("data", [])) > 1:
        data = result["data"]
        combined_text = "\n\n".join(item["text"] for item in data)
        avg_confidence = sum(item["confidence"] for item in data) / len(data)
        original_page_count = len(data)
        result["data"] = [
            {
                "index": 0,
                "text": combined_text,
                "confidence": avg_confidence,
            }
        ]
        result["usage"]["pages_combined"] = original_page_count

    return result


# =============================================================================
# Document Extraction Endpoint
# =============================================================================


@router.post("/documents/extract")
async def extract_from_documents(
    file: OptionalFileUpload = None,
    images: str | None = Form(
        default=None,
        description='Base64-encoded images as JSON array (e.g., ["data:image/png;base64,..."])',
    ),
    model: str = Form(
        ...,
        description="HuggingFace model ID (e.g., naver-clova-ix/donut-base-finetuned-docvqa)",
    ),
    prompts: str = Form(
        default="",
        description="Comma-separated prompts for VQA task",
    ),
    task: str = Form(
        default="extraction",
        description="Task: extraction, vqa, or classification",
    ),
) -> dict[str, Any]:
    """Document understanding endpoint.

    Accepts either a file upload OR base64-encoded images.

    Extract structured information from documents using vision-language models.
    Supports forms, invoices, receipts, and other document types.

    Model types:
    - Donut models: End-to-end, no OCR needed (naver-clova-ix/donut-*)
    - LayoutLM models: Uses OCR + layout features (microsoft/layoutlmv3-*)

    Tasks:
    - extraction: Extract key-value pairs from documents
    - vqa: Answer questions about document content
    - classification: Classify document types

    Example with file upload (curl):
    ```bash
    curl -X POST http://localhost:14345/v1/vision/documents/extract \\
      -F "file=@receipt.pdf" \\
      -F "model=naver-clova-ix/donut-base-finetuned-docvqa" \\
      -F "prompts=What is the total amount?" \\
      -F "task=vqa"
    ```

    Example with base64 images (curl):
    ```bash
    curl -X POST http://localhost:14345/v1/vision/documents/extract \\
      -F 'images=["data:image/png;base64,iVBORw0KGgo..."]' \\
      -F "model=naver-clova-ix/donut-base-finetuned-cord-v2" \\
      -F "task=extraction"
    ```

    Response:
    ```json
    {
        "object": "list",
        "data": [
            {
                "index": 0,
                "text": "...",
                "structured": {...}
            }
        ],
        "model": "naver-clova-ix/donut-base-finetuned-cord-v2",
        "task": "extraction"
    }
    ```
    """
    # Validate task
    valid_tasks = {"extraction", "vqa", "classification"}
    if task not in valid_tasks:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task: {task}. Must be one of: {', '.join(valid_tasks)}",
        )

    # Get images from file or base64 input
    image_list = await _get_images_from_input(file, images)

    # Parse prompts
    prompt_list = (
        [p.strip() for p in prompts.split(",") if p.strip()] if prompts else None
    )

    return await UniversalRuntimeService.extract_documents(
        model=model,
        images=image_list,
        prompts=prompt_list,
        task=task,  # type: ignore
    )


# =============================================================================
# Object Detection Endpoint
# =============================================================================


@router.post("/detect")
async def detect_objects(
    file: OptionalFileUpload = None,
    image: str | None = Form(
        default=None,
        description="Base64-encoded image (data:image/jpeg;base64,...)",
    ),
    model: str = Form(
        default="yolov8n",
        description="YOLO model: yolov8n, yolov8s, yolov8m, yolov11n",
    ),
    confidence_threshold: float = Form(
        default=0.5,
        description="Minimum confidence threshold (0-1)",
    ),
    classes: str = Form(
        default="",
        description="Comma-separated class names to filter (e.g., 'person,car')",
    ),
) -> dict[str, Any]:
    """Detect objects in an image using YOLO.

    Accepts either a file upload OR base64-encoded image.

    Supported models:
    - yolov8n, yolov8s, yolov8m, yolov8l, yolov8x (YOLOv8 variants)
    - yolov11n, yolov11s, yolov11m (YOLOv11 variants)

    Example with file upload:
    ```bash
    curl -X POST http://localhost:14345/v1/vision/detect \\
      -F "file=@photo.jpg" \\
      -F "model=yolov8n" \\
      -F "confidence_threshold=0.5"
    ```

    Example with base64:
    ```bash
    curl -X POST http://localhost:14345/v1/vision/detect \\
      -F "image=data:image/jpeg;base64,..." \\
      -F "model=yolov8n"
    ```

    Response:
    ```json
    {
        "detections": [
            {
                "box": {"x1": 100, "y1": 200, "x2": 300, "y2": 400},
                "class_name": "person",
                "class_id": 0,
                "confidence": 0.92
            }
        ],
        "model": "yolov8n",
        "inference_time_ms": 45.2
    }
    ```
    """
    from services.vision import VisionDetectionService

    # Get image from file or base64
    if file is not None and file.filename:
        import base64
        content = await file.read()
        from pathlib import Path
        ext = Path(file.filename).suffix.lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
        mime = mime_map.get(ext, "image/jpeg")
        image_b64 = f"data:{mime};base64,{base64.b64encode(content).decode()}"
    elif image:
        image_b64 = image
    else:
        raise HTTPException(status_code=400, detail="Either 'file' or 'image' must be provided")

    # Parse classes filter
    class_list = [c.strip() for c in classes.split(",") if c.strip()] if classes else None

    return await VisionDetectionService.detect(
        image=image_b64,
        model=model,
        confidence_threshold=confidence_threshold,
        classes=class_list,
    )


# =============================================================================
# Image Classification Endpoint
# =============================================================================


@router.post("/classify")
async def classify_image(
    file: OptionalFileUpload = None,
    image: str | None = Form(
        default=None,
        description="Base64-encoded image",
    ),
    model: str = Form(
        default="clip-vit-base",
        description="CLIP model: clip-vit-base, clip-vit-large, siglip-base",
    ),
    classes: str = Form(
        ...,
        description="Comma-separated class names for zero-shot classification",
    ),
    top_k: int = Form(
        default=5,
        description="Number of top predictions to return",
    ),
) -> dict[str, Any]:
    """Classify an image using CLIP zero-shot classification.

    Provide a list of class names and the model will predict which
    class best matches the image.

    Example:
    ```bash
    curl -X POST http://localhost:14345/v1/vision/classify \\
      -F "file=@photo.jpg" \\
      -F "classes=cat,dog,bird,fish" \\
      -F "model=clip-vit-base"
    ```

    Response:
    ```json
    {
        "class_name": "cat",
        "class_id": 0,
        "confidence": 0.87,
        "all_scores": {"cat": 0.87, "dog": 0.08, "bird": 0.05}
    }
    ```
    """
    from services.vision import VisionClassificationService

    # Get image
    if file is not None and file.filename:
        import base64
        content = await file.read()
        from pathlib import Path
        ext = Path(file.filename).suffix.lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
        mime = mime_map.get(ext, "image/jpeg")
        image_b64 = f"data:{mime};base64,{base64.b64encode(content).decode()}"
    elif image:
        image_b64 = image
    else:
        raise HTTPException(status_code=400, detail="Either 'file' or 'image' must be provided")

    # Parse classes
    class_list = [c.strip() for c in classes.split(",") if c.strip()]
    if not class_list:
        raise HTTPException(status_code=400, detail="At least one class must be provided")

    return await VisionClassificationService.classify(
        image=image_b64,
        model=model,
        classes=class_list,
        top_k=top_k,
    )


# =============================================================================
# Image Embedding Endpoint
# =============================================================================


@router.post("/embed")
async def embed_images(
    file: OptionalFileUpload = None,
    images: str | None = Form(
        default=None,
        description="JSON array of base64-encoded images",
    ),
    texts: str | None = Form(
        default=None,
        description="Comma-separated text strings to embed",
    ),
    model: str = Form(
        default="clip-vit-base",
        description="CLIP model for embeddings",
    ),
) -> dict[str, Any]:
    """Generate CLIP embeddings for images and/or texts.

    CLIP embeddings exist in a shared vector space, enabling:
    - Image-to-image similarity search
    - Text-to-image search
    - Image RAG

    Example:
    ```bash
    curl -X POST http://localhost:14345/v1/vision/embed \\
      -F "file=@photo.jpg" \\
      -F "texts=a photo of a cat,a photo of a dog"
    ```

    Response:
    ```json
    {
        "embeddings": [[0.123, -0.456, ...]],
        "dimensions": 512,
        "model": "clip-vit-base"
    }
    ```
    """
    from services.vision import VisionEmbeddingService

    image_list = []
    text_list = []

    # Get image from file
    if file is not None and file.filename:
        import base64
        content = await file.read()
        from pathlib import Path
        ext = Path(file.filename).suffix.lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
        mime = mime_map.get(ext, "image/jpeg")
        image_list.append(f"data:{mime};base64,{base64.b64encode(content).decode()}")

    # Parse images JSON array
    if images:
        try:
            parsed = json.loads(images)
            if isinstance(parsed, list):
                image_list.extend(parsed)
        except json.JSONDecodeError:
            image_list.append(images)

    # Parse texts
    if texts:
        text_list = [t.strip() for t in texts.split(",") if t.strip()]

    if not image_list and not text_list:
        raise HTTPException(status_code=400, detail="Either images or texts must be provided")

    return await VisionEmbeddingService.embed(
        model=model,
        images=image_list if image_list else None,
        texts=text_list if text_list else None,
    )
