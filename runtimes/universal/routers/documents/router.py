"""
Document understanding endpoints.

Extract structured information from documents using vision-language models.
"""

from fastapi import APIRouter, HTTPException

from core.logging import UniversalRuntimeLogger
from utils.file_handler import get_file_images

from .service import load_document
from .types import DocumentExtractRequest

router = APIRouter()
logger = UniversalRuntimeLogger("universal-runtime.documents")


@router.post("/v1/documents/extract")
async def extract_from_documents(request: DocumentExtractRequest):
    """
    Document understanding endpoint.

    Extract structured information from documents using vision-language models.
    Supports forms, invoices, receipts, and other document types.

    Model types:
    - Donut models: End-to-end, no OCR needed (naver-clova-ix/donut-*)
    - LayoutLM models: Uses OCR + layout features (microsoft/layoutlmv3-*)

    Tasks:
    - extraction: Extract key-value pairs from documents
    - vqa: Answer questions about document content
    - classification: Classify document types

    You can provide images either as:
    1. Base64-encoded strings in the `images` field
    2. A file ID from a previous upload via `file_id` field

    Example with base64:
    ```json
    {
        "model": "naver-clova-ix/donut-base-finetuned-cord-v2",
        "images": ["base64_encoded_image..."],
        "task": "extraction"
    }
    ```

    Example with file_id (from /v1/files upload):
    ```json
    {
        "model": "naver-clova-ix/donut-base-finetuned-cord-v2",
        "file_id": "file_abc123_def456",
        "task": "extraction"
    }
    ```

    For VQA, include prompts:
    ```json
    {
        "model": "microsoft/layoutlmv3-base-finetuned-docvqa",
        "file_id": "file_abc123_def456",
        "prompts": ["What is the total amount?"],
        "task": "vqa"
    }
    ```
    """
    try:
        # Resolve images from file_id or direct base64
        images = request.images
        if request.file_id:
            images = get_file_images(request.file_id)
            if not images:
                raise HTTPException(
                    status_code=400,
                    detail=f"No images found for file_id: {request.file_id}",
                )
        elif not images:
            raise HTTPException(
                status_code=400,
                detail="Either 'images' or 'file_id' must be provided",
            )

        # Load document model
        model = await load_document(
            model_id=request.model,
            task=request.task,
        )

        # Extract from documents
        results = await model.extract(
            images=images,
            prompts=request.prompts,
        )

        # Format response
        data = []
        for idx, result in enumerate(results):
            item = {
                "index": idx,
                "confidence": result.confidence,
            }

            if result.text:
                item["text"] = result.text

            if result.fields:
                item["fields"] = [
                    {
                        "key": f.key,
                        "value": f.value,
                        "confidence": f.confidence,
                        "bbox": f.bbox,
                    }
                    for f in result.fields
                ]

            if result.answer:
                item["answer"] = result.answer

            if result.classification:
                item["classification"] = result.classification
                item["classification_scores"] = result.classification_scores

            data.append(item)

        return {
            "object": "list",
            "data": data,
            "model": request.model,
            "task": request.task,
            "usage": {
                "documents_processed": len(images),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in extract_from_documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
