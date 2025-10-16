"""
Transformers Runtime Server

A FastAPI server that provides OpenAI-compatible endpoints for any HuggingFace
model without restrictions. Supports both text generation and image generation
(diffusion models).

Key Features:
- Auto-detects hardware (MPS/CUDA/CPU)
- Lazy model loading (load on first request)
- Platform-specific optimizations
- OpenAI API compatibility
- No model restrictions (trust_remote_code=True)
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Literal, List, Union
import torch
import os
from pathlib import Path
import base64
import io
from PIL import Image
import logging
from datetime import datetime
import json

from models.text_model import TextModel
from models.image_model import ImageModel
from utils.device import get_optimal_device, get_device_info
from utils.file_utils import save_image_with_metadata

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Transformers Runtime",
    description="OpenAI-compatible API for HuggingFace models",
    version="1.0.0"
)

# Global model cache
_models = {}
_current_device = None


# ============================================================================
# Helper Functions
# ============================================================================

def encode_image_to_base64(image_source: Union[str, Path, bytes, UploadFile]) -> str:
    """
    Encode an image to base64 string from various input types.

    Args:
        image_source: Can be:
            - File path (str or Path)
            - Raw bytes
            - UploadFile from FastAPI
            - Already base64-encoded string (returns as-is)

    Returns:
        Base64-encoded string of the image

    Raises:
        HTTPException: If file not found or encoding fails
    """
    try:
        # If already base64 string, return as-is
        if isinstance(image_source, str):
            # Check if it's a file path
            if Path(image_source).exists():
                with open(image_source, "rb") as f:
                    image_bytes = f.read()
                return base64.b64encode(image_bytes).decode('utf-8')
            # Assume it's already base64
            return image_source

        # If Path object
        if isinstance(image_source, Path):
            if not image_source.exists():
                raise HTTPException(status_code=404, detail=f"Image file not found: {image_source}")
            with open(image_source, "rb") as f:
                image_bytes = f.read()
            return base64.b64encode(image_bytes).decode('utf-8')

        # If bytes
        if isinstance(image_source, bytes):
            return base64.b64encode(image_source).decode('utf-8')

        # If UploadFile
        if isinstance(image_source, UploadFile):
            image_bytes = image_source.file.read()
            return base64.b64encode(image_bytes).decode('utf-8')

        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image source type: {type(image_source)}"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to encode image: {str(e)}"
        )


def get_device():
    """Get the optimal device for the current platform."""
    global _current_device
    if _current_device is None:
        _current_device = get_optimal_device()
        logger.info(f"Using device: {_current_device}")
    return _current_device


async def load_text_model(model_id: str):
    """Load a text generation model."""
    if model_id not in _models:
        logger.info(f"Loading text model: {model_id}")
        device = get_device()
        model = TextModel(model_id, device)
        await model.load()
        _models[model_id] = model
    return _models[model_id]


async def load_image_model(model_id: str):
    """Load an image generation model."""
    if model_id not in _models:
        logger.info(f"Loading image model: {model_id}")
        device = get_device()
        model = ImageModel(model_id, device)
        await model.load()
        _models[model_id] = model
    return _models[model_id]


# ============================================================================
# Request/Response Models
# ============================================================================

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None


class ImageGenerationRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    n: Optional[int] = Field(default=1, ge=1, le=10)
    size: Optional[Literal[
        "256x256", "512x512", "768x768", "1024x1024",
        "1280x1280", "1792x1024", "1024x1792"
    ]] = "1024x1024"
    quality: Optional[Literal["standard", "hd"]] = "standard"
    style: Optional[Literal["vivid", "natural"]] = "vivid"
    response_format: Optional[Literal["url", "b64_json"]] = "url"
    user: Optional[str] = None

    # Extended diffusion parameters
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = Field(default=None, ge=1, le=150)
    guidance_scale: Optional[float] = Field(default=None, ge=1.0, le=20.0)
    seed: Optional[int] = None
    scheduler: Optional[str] = None


class ImageEditRequest(BaseModel):
    image: str  # Base64 encoded or URL
    prompt: str
    mask: Optional[str] = None  # Base64 encoded or URL
    model: Optional[str] = None
    n: Optional[int] = Field(default=1, ge=1, le=10)
    size: Optional[str] = None
    response_format: Optional[Literal["url", "b64_json"]] = "url"
    user: Optional[str] = None

    # Extended parameters
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = Field(default=None, ge=1, le=150)
    guidance_scale: Optional[float] = Field(default=None, ge=1.0, le=20.0)
    seed: Optional[int] = None


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint with device information."""
    device_info = get_device_info()
    return {
        "status": "healthy",
        "device": device_info,
        "loaded_models": list(_models.keys()),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/v1/models")
async def list_models():
    """List currently loaded models."""
    models_list = []
    for model_id, model in _models.items():
        models_list.append({
            "id": model_id,
            "object": "model",
            "created": int(datetime.now().timestamp()),
            "owned_by": "transformers-runtime",
            "type": model.model_type
        })

    return {
        "object": "list",
        "data": models_list
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.

    Supports any HuggingFace text generation model.
    """
    try:
        model = await load_text_model(request.model)

        # Convert messages to prompt
        messages_dict = [msg.dict() for msg in request.messages]
        prompt = model.format_messages(messages_dict)

        # Generate response
        response_text = await model.generate(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop
        )

        return {
            "id": f"chatcmpl-{os.urandom(16).hex()}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,  # TODO: Implement token counting
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }

    except Exception as e:
        logger.error(f"Error in chat_completions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/images/generations")
async def generate_images(request: ImageGenerationRequest, background_tasks: BackgroundTasks):
    """
    OpenAI-compatible image generation endpoint.

    Supports any HuggingFace diffusion model (Stable Diffusion, FLUX, etc.).
    """
    try:
        # Use default model if not specified
        model_id = request.model or os.getenv("DEFAULT_IMAGE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")

        model = await load_image_model(model_id)

        # Parse size
        width, height = map(int, request.size.split("x"))

        # Generate images
        images = await model.generate(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_images=request.n,
            width=width,
            height=height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            scheduler=request.scheduler
        )

        # Prepare response
        response_images = []
        output_dir = Path(os.getenv("TRANSFORMERS_OUTPUT_DIR", "~/.llamafarm/outputs/images")).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        for idx, img in enumerate(images):
            if request.response_format == "url":
                # Save to disk
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                seed = request.seed if request.seed is not None else idx
                filename = f"{model_id.replace('/', '_')}_{timestamp}_{seed}_{idx}.png"
                filepath = output_dir / filename

                # Save with metadata
                metadata = {
                    "prompt": request.prompt,
                    "negative_prompt": request.negative_prompt,
                    "model": model_id,
                    "size": request.size,
                    "steps": request.num_inference_steps,
                    "guidance_scale": request.guidance_scale,
                    "seed": seed,
                    "created": datetime.utcnow().isoformat()
                }

                save_image_with_metadata(img, filepath, metadata)

                response_images.append({
                    "url": f"file://{filepath}",
                    "revised_prompt": request.prompt  # TODO: Implement prompt revision
                })
            else:
                # Return base64
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()

                response_images.append({
                    "b64_json": img_str,
                    "revised_prompt": request.prompt
                })

        return {
            "created": int(datetime.now().timestamp()),
            "data": response_images
        }

    except Exception as e:
        logger.error(f"Error in generate_images: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/images/edits")
async def edit_images(request: ImageEditRequest):
    """
    OpenAI-compatible image editing endpoint.

    Supports inpainting with Stable Diffusion inpainting models.
    """
    try:
        model_id = request.model or os.getenv("DEFAULT_INPAINT_MODEL", "runwayml/stable-diffusion-inpainting")

        model = await load_image_model(model_id)

        # Decode input image (supports file paths, base64, or data URLs)
        image_b64 = encode_image_to_base64(request.image)

        if image_b64.startswith("data:"):
            img_data = image_b64.split(",")[1]
            img_bytes = base64.b64decode(img_data)
        else:
            img_bytes = base64.b64decode(image_b64)

        input_image = Image.open(io.BytesIO(img_bytes))

        # Decode mask if provided (supports file paths, base64, or data URLs)
        mask_image = None
        if request.mask:
            mask_b64 = encode_image_to_base64(request.mask)

            if mask_b64.startswith("data:"):
                mask_data = mask_b64.split(",")[1]
                mask_bytes = base64.b64decode(mask_data)
            else:
                mask_bytes = base64.b64decode(mask_b64)
            mask_image = Image.open(io.BytesIO(mask_bytes))

        # Generate edited images
        images = await model.edit(
            prompt=request.prompt,
            image=input_image,
            mask=mask_image,
            negative_prompt=request.negative_prompt,
            num_images=request.n,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed
        )

        # Prepare response (similar to generations)
        response_images = []
        output_dir = Path(os.getenv("TRANSFORMERS_OUTPUT_DIR", "~/.llamafarm/outputs/images")).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        for idx, img in enumerate(images):
            if request.response_format == "url":
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"edit_{model_id.replace('/', '_')}_{timestamp}_{idx}.png"
                filepath = output_dir / filename

                metadata = {
                    "operation": "edit",
                    "prompt": request.prompt,
                    "negative_prompt": request.negative_prompt,
                    "model": model_id,
                    "created": datetime.utcnow().isoformat()
                }

                save_image_with_metadata(img, filepath, metadata)

                response_images.append({"url": f"file://{filepath}"})
            else:
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                response_images.append({"b64_json": img_str})

        return {
            "created": int(datetime.now().timestamp()),
            "data": response_images
        }

    except Exception as e:
        logger.error(f"Error in edit_images: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/images/variations")
async def create_variations(request: ImageEditRequest):
    """
    OpenAI-compatible image variations endpoint.

    Creates variations of an input image using image-to-image diffusion.
    """
    try:
        model_id = request.model or os.getenv("DEFAULT_IMAGE_MODEL", "hf-internal-testing/tiny-stable-diffusion-torch")

        model = await load_image_model(model_id)

        # Decode input image (supports file paths, base64, or data URLs)
        image_b64 = encode_image_to_base64(request.image)

        if image_b64.startswith("data:"):
            img_data = image_b64.split(",")[1]
            img_bytes = base64.b64decode(img_data)
        else:
            img_bytes = base64.b64decode(image_b64)

        input_image = Image.open(io.BytesIO(img_bytes))

        # Convert to RGB and resize to target size for img2img
        if input_image.mode != 'RGB':
            input_image = input_image.convert('RGB')

        # Resize to target size if specified
        if request.size:
            target_width, target_height = map(int, request.size.split('x'))
            input_image = input_image.resize((target_width, target_height), Image.Resampling.LANCZOS)

        # Use img2img for variations
        # Strength parameter: how much to transform (0.0 = no change, 1.0 = complete remake)
        # For variations, we want moderate strength to preserve structure
        strength = getattr(request, 'strength', 0.75)  # Default 0.75 for good balance

        images = await model.img2img(
            prompt=getattr(request, 'prompt', "high quality photo"),  # Default prompt if none provided
            image=input_image,
            negative_prompt=request.negative_prompt,
            num_images=request.n,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            strength=strength,
            seed=request.seed
        )

        # Prepare response
        response_images = []
        output_dir = Path(os.getenv("TRANSFORMERS_OUTPUT_DIR", "~/.llamafarm/outputs/images")).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        for idx, img in enumerate(images):
            if request.response_format == "url":
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"variation_{model_id.replace('/', '_')}_{timestamp}_{idx}.png"
                filepath = output_dir / filename

                metadata = {
                    "operation": "img2img",
                    "prompt": getattr(request, 'prompt', "high quality photo"),
                    "negative_prompt": request.negative_prompt,
                    "model": model_id,
                    "strength": strength,
                    "created": datetime.utcnow().isoformat()
                }

                save_image_with_metadata(img, filepath, metadata)

                response_images.append({"url": f"file://{filepath}"})
            else:
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                response_images.append({"b64_json": img_str})

        return {
            "created": int(datetime.now().timestamp()),
            "data": response_images
        }

    except Exception as e:
        logger.error(f"Error in create_variations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("TRANSFORMERS_PORT", "11540"))
    host = os.getenv("TRANSFORMERS_HOST", "127.0.0.1")

    logger.info(f"Starting Transformers Runtime on {host}:{port}")
    logger.info(f"Device: {get_device()}")

    uvicorn.run(app, host=host, port=port)
