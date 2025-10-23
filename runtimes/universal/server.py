"""
Universal Runtime Server

A FastAPI server that provides OpenAI-compatible endpoints for any HuggingFace
model without restrictions. Supports:
- Text generation (Causal LMs: GPT, Llama, Mistral, etc.)
- Text embeddings & classification (Encoders: BERT, sentence-transformers, etc.)
- Image generation (Diffusion: Stable Diffusion, FLUX, etc.)
- Image classification & embeddings (Vision: ViT, CLIP, etc.)
- Speech-to-text (Audio: Whisper, Wav2Vec2, etc.)
- Vision-language (Multimodal: BLIP, LLaVA, Florence, etc.)

Key Features:
- Auto-detects hardware (MPS/CUDA/CPU)
- Lazy model loading (load on first request)
- Platform-specific optimizations
- OpenAI API compatibility
- No model restrictions (trust_remote_code=True)
"""

import asyncio
from fastapi import (
    FastAPI,
    HTTPException,
    BackgroundTasks,
    UploadFile,
    Request,
    File,
    Form,
)
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field
from typing import Optional, Literal, List, Union
import os
from pathlib import Path
import base64
import io
from PIL import Image
import logging
from datetime import datetime
import json

from models import (
    CausalLMModel,
    EncoderModel,
    DiffusionModel,
    VisionModel,
    AudioModel,
    MultimodalModel,
)
from utils.device import get_optimal_device, get_device_info

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Universal Runtime",
    description="OpenAI-compatible API for HuggingFace models (transformers & diffusers)",
    version="2.0.0",
)

# Global model cache
_models = {}
_current_device = None


# ============================================================================
# Helper Functions
# ============================================================================


def get_image_format_from_accept(accept_header: str) -> tuple[str, str]:
    """
    Determine image format and media type from Accept header.

    Returns:
        tuple: (PIL format, media type)
        Default: ("JPEG", "image/jpeg")
    """
    accept_lower = accept_header.lower()

    if "image/png" in accept_lower:
        return ("PNG", "image/png")
    elif "image/webp" in accept_lower:
        return ("WEBP", "image/webp")
    else:
        # Default to JPEG for better compression
        # This includes image/jpeg, image/jpg, image/*, */*
        return ("JPEG", "image/jpeg")


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
                return base64.b64encode(image_bytes).decode("utf-8")
            # Assume it's already base64
            return image_source

        # If Path object
        if isinstance(image_source, Path):
            if not image_source.exists():
                raise HTTPException(
                    status_code=404, detail=f"Image file not found: {image_source}"
                )
            with open(image_source, "rb") as f:
                image_bytes = f.read()
            return base64.b64encode(image_bytes).decode("utf-8")

        # If bytes
        if isinstance(image_source, bytes):
            return base64.b64encode(image_source).decode("utf-8")

        # If UploadFile
        if isinstance(image_source, UploadFile):
            image_bytes = image_source.file.read()
            return base64.b64encode(image_bytes).decode("utf-8")

        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image source type: {type(image_source)}",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to encode image: {str(e)}")


def get_device():
    """Get the optimal device for the current platform."""
    global _current_device
    if _current_device is None:
        _current_device = get_optimal_device()
        logger.info(f"Using device: {_current_device}")
    return _current_device


async def load_causal_lm(model_id: str):
    """Load a causal language model for text generation."""
    cache_key = f"causal_lm:{model_id}"
    if cache_key not in _models:
        logger.info(f"Loading causal LM: {model_id}")
        device = get_device()
        model = CausalLMModel(model_id, device)
        await model.load()
        _models[cache_key] = model
    return _models[cache_key]


async def load_encoder(model_id: str, task: str = "embedding"):
    """Load an encoder model for embeddings or classification."""
    cache_key = f"encoder:{task}:{model_id}"
    if cache_key not in _models:
        logger.info(f"Loading encoder ({task}): {model_id}")
        device = get_device()
        model = EncoderModel(model_id, device, task=task)
        await model.load()
        _models[cache_key] = model
    return _models[cache_key]


async def load_diffusion_model(model_id: str):
    """Load a diffusion model for image generation."""
    cache_key = f"diffusion:{model_id}"
    if cache_key not in _models:
        logger.info(f"Loading diffusion model: {model_id}")
        device = get_device()
        model = DiffusionModel(model_id, device)
        await model.load()
        _models[cache_key] = model
    return _models[cache_key]


async def load_vision_model(model_id: str, task: str = "classification"):
    """Load a vision model for image classification or embeddings."""
    cache_key = f"vision:{task}:{model_id}"
    if cache_key not in _models:
        logger.info(f"Loading vision model ({task}): {model_id}")
        device = get_device()
        model = VisionModel(model_id, device, task=task)
        await model.load()
        _models[cache_key] = model
    return _models[cache_key]


async def load_audio_model(model_id: str, task: str = "transcribe"):
    """Load an audio model for speech-to-text."""
    cache_key = f"audio:{task}:{model_id}"
    if cache_key not in _models:
        logger.info(f"Loading audio model ({task}): {model_id}")
        device = get_device()
        model = AudioModel(model_id, device, task=task)
        await model.load()
        _models[cache_key] = model
    return _models[cache_key]


async def load_multimodal_model(model_id: str, task: str = "image-to-text"):
    """Load a multimodal model for vision-language tasks."""
    cache_key = f"multimodal:{task}:{model_id}"
    if cache_key not in _models:
        logger.info(f"Loading multimodal model ({task}): {model_id}")
        device = get_device()
        model = MultimodalModel(model_id, device, task=task)
        await model.load()
        _models[cache_key] = model
    return _models[cache_key]


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
    size: Optional[
        Literal[
            "256x256",
            "512x512",
            "768x768",
            "1024x1024",
            "1280x1280",
            "1792x1024",
            "1024x1792",
        ]
    ] = "1024x1024"
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
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/v1/models")
async def list_models():
    """List currently loaded models."""
    models_list = []
    for model_id, model in _models.items():
        models_list.append(
            {
                "id": model_id,
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "transformers-runtime",
                "type": model.model_type,
            }
        )

    return {"object": "list", "data": models_list}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.

    Supports any HuggingFace causal language model.
    """
    try:
        model = await load_causal_lm(request.model)

        # Convert messages to prompt
        messages_dict = [msg.model_dump() for msg in request.messages]
        prompt = model.format_messages(messages_dict)

        # Handle streaming if requested
        if request.stream:
            logger.info(f"Streaming chat completions for model: {request.model}")

            # Return SSE stream
            async def generate_sse():
                completion_id = f"chatcmpl-{os.urandom(16).hex()}"
                created_time = int(datetime.now().timestamp())

                # Send initial chunk
                yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': request.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"

                # Stream tokens
                async for token in model.generate_stream(
                    prompt=prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=request.stop,
                ):
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": token},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n".encode()
                    # CRITICAL: This asyncio.sleep(0) forces the event loop to yield,
                    # ensuring the stream flushes immediately for token-by-token delivery.
                    # Without this, tokens would buffer and arrive in large chunks.
                    # See test_streaming_server.py for verification tests.
                    await asyncio.sleep(0)

                # Send final chunk
                final_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(final_chunk)}\n\n".encode()
                await asyncio.sleep(0)
                yield b"data: [DONE]\n\n"

            return StreamingResponse(
                generate_sse(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Non-streaming response
        response_text = await model.generate(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
        )

        return {
            "id": f"chatcmpl-{os.urandom(16).hex()}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,  # TODO: Implement token counting
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    except Exception as e:
        logger.error(f"Error in chat_completions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/images/generations")
async def generate_images(
    req: Request,
    request: ImageGenerationRequest,
    background_tasks: BackgroundTasks,
):
    """
    OpenAI-compatible image generation endpoint.

    Supports any HuggingFace diffusion model (Stable Diffusion, FLUX, etc.).

    Content negotiation via Accept header:
    - Accept: application/json → Returns JSON with base64 (default)
    - Accept: image/png, image/* → Returns raw PNG bytes (single image only)
    """
    try:
        # Use default model if not specified
        model_id = request.model or os.getenv(
            "DEFAULT_IMAGE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0"
        )

        model = await load_diffusion_model(model_id)

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
            scheduler=request.scheduler,
        )

        # Check Accept header for content negotiation
        accept_header = req.headers.get("accept", "application/json")
        image_format, media_type = get_image_format_from_accept(accept_header)

        wants_image = any(
            mime in accept_header.lower()
            for mime in [
                "image/png",
                "image/jpeg",
                "image/jpg",
                "image/webp",
                "image/*",
                "*/*",
            ]
        )

        # If client wants raw image and we have exactly one image, return bytes
        if wants_image and request.n == 1:
            img = images[0]
            buffered = io.BytesIO()

            # Convert RGBA to RGB for JPEG (doesn't support alpha)
            if image_format == "JPEG" and img.mode == "RGBA":
                rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                rgb_img.paste(
                    img, mask=img.split()[3] if len(img.split()) == 4 else None
                )
                img = rgb_img

            # Save with appropriate format and quality
            save_kwargs = {"format": image_format}
            if image_format == "JPEG":
                save_kwargs["quality"] = 95
                save_kwargs["optimize"] = True
            elif image_format == "WEBP":
                save_kwargs["quality"] = 90

            img.save(buffered, **save_kwargs)

            return Response(
                content=buffered.getvalue(),
                media_type=media_type,
                headers={
                    "X-Prompt": request.prompt,
                    "X-Model": model_id,
                },
            )

        # Otherwise return JSON response
        response_images = []
        for idx, img in enumerate(images):
            # Convert image to base64 (always PNG for JSON responses)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            if request.response_format == "url":
                # Return as data URL for in-memory viewing
                data_url = f"data:image/png;base64,{img_str}"
                response_images.append(
                    {
                        "url": data_url,
                        "revised_prompt": request.prompt,
                    }
                )
            else:
                # Return base64
                response_images.append(
                    {"b64_json": img_str, "revised_prompt": request.prompt}
                )

        return {"created": int(datetime.now().timestamp()), "data": response_images}

    except Exception as e:
        logger.error(f"Error in generate_images: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/images/edits")
async def edit_images(req: Request, request: ImageEditRequest):
    """
    OpenAI-compatible image editing endpoint.

    Supports inpainting with Stable Diffusion inpainting models.
    """
    try:
        model_id = request.model or os.getenv(
            "DEFAULT_INPAINT_MODEL", "runwayml/stable-diffusion-inpainting"
        )

        model = await load_diffusion_model(model_id)

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
            seed=request.seed,
        )

        # Check Accept header for content negotiation
        accept_header = req.headers.get("accept", "application/json")
        image_format, media_type = get_image_format_from_accept(accept_header)

        wants_image = any(
            mime in accept_header.lower()
            for mime in [
                "image/png",
                "image/jpeg",
                "image/jpg",
                "image/webp",
                "image/*",
                "*/*",
            ]
        )

        # If client wants raw image and we have exactly one image, return bytes
        if wants_image and request.n == 1:
            img = images[0]
            buffered = io.BytesIO()

            # Convert RGBA to RGB for JPEG (doesn't support alpha)
            if image_format == "JPEG" and img.mode == "RGBA":
                rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                rgb_img.paste(
                    img, mask=img.split()[3] if len(img.split()) == 4 else None
                )
                img = rgb_img

            # Save with appropriate format and quality
            save_kwargs = {"format": image_format}
            if image_format == "JPEG":
                save_kwargs["quality"] = 95
                save_kwargs["optimize"] = True
            elif image_format == "WEBP":
                save_kwargs["quality"] = 90

            img.save(buffered, **save_kwargs)

            return Response(
                content=buffered.getvalue(),
                media_type=media_type,
                headers={
                    "X-Prompt": request.prompt,
                    "X-Model": model_id,
                },
            )

        # Otherwise return JSON response
        response_images = []
        for idx, img in enumerate(images):
            # Convert image to base64 (always PNG for JSON responses)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            if request.response_format == "url":
                # Return as data URL for in-memory viewing
                data_url = f"data:image/png;base64,{img_str}"
                response_images.append({"url": data_url})
            else:
                response_images.append({"b64_json": img_str})

        return {"created": int(datetime.now().timestamp()), "data": response_images}

    except Exception as e:
        logger.error(f"Error in edit_images: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/images/variations")
async def create_variations(req: Request, request: ImageEditRequest):
    """
    OpenAI-compatible image variations endpoint.

    Creates variations of an input image using image-to-image diffusion.
    """
    try:
        model_id = request.model or os.getenv(
            "DEFAULT_IMAGE_MODEL", "hf-internal-testing/tiny-stable-diffusion-torch"
        )

        model = await load_diffusion_model(model_id)

        # Decode input image (supports file paths, base64, or data URLs)
        image_b64 = encode_image_to_base64(request.image)

        if image_b64.startswith("data:"):
            img_data = image_b64.split(",")[1]
            img_bytes = base64.b64decode(img_data)
        else:
            img_bytes = base64.b64decode(image_b64)

        input_image = Image.open(io.BytesIO(img_bytes))

        # Convert to RGB and resize to target size for img2img
        if input_image.mode != "RGB":
            input_image = input_image.convert("RGB")

        # Resize to target size if specified
        if request.size:
            target_width, target_height = map(int, request.size.split("x"))
            input_image = input_image.resize(
                (target_width, target_height), Image.Resampling.LANCZOS
            )

        # Use img2img for variations
        # Strength parameter: how much to transform (0.0 = no change, 1.0 = complete remake)
        # For variations, we want moderate strength to preserve structure
        strength = getattr(request, "strength", 0.75)  # Default 0.75 for good balance

        images = await model.img2img(
            prompt=getattr(
                request, "prompt", "high quality photo"
            ),  # Default prompt if none provided
            image=input_image,
            negative_prompt=request.negative_prompt,
            num_images=request.n,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            strength=strength,
            seed=request.seed,
        )

        # Check Accept header for content negotiation
        accept_header = req.headers.get("accept", "application/json")
        image_format, media_type = get_image_format_from_accept(accept_header)

        wants_image = any(
            mime in accept_header.lower()
            for mime in [
                "image/png",
                "image/jpeg",
                "image/jpg",
                "image/webp",
                "image/*",
                "*/*",
            ]
        )

        # If client wants raw image and we have exactly one image, return bytes
        if wants_image and request.n == 1:
            img = images[0]
            buffered = io.BytesIO()

            # Convert RGBA to RGB for JPEG (doesn't support alpha)
            if image_format == "JPEG" and img.mode == "RGBA":
                rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                rgb_img.paste(
                    img, mask=img.split()[3] if len(img.split()) == 4 else None
                )
                img = rgb_img

            # Save with appropriate format and quality
            save_kwargs = {"format": image_format}
            if image_format == "JPEG":
                save_kwargs["quality"] = 95
                save_kwargs["optimize"] = True
            elif image_format == "WEBP":
                save_kwargs["quality"] = 90

            img.save(buffered, **save_kwargs)

            return Response(
                content=buffered.getvalue(),
                media_type=media_type,
                headers={
                    "X-Model": model_id,
                },
            )

        # Otherwise return JSON response
        response_images = []
        for idx, img in enumerate(images):
            # Convert image to base64 (always PNG for JSON responses)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            if request.response_format == "url":
                # Return as data URL for in-memory viewing
                data_url = f"data:image/png;base64,{img_str}"
                response_images.append({"url": data_url})
            else:
                response_images.append({"b64_json": img_str})

        return {"created": int(datetime.now().timestamp()), "data": response_images}

    except Exception as e:
        logger.error(f"Error in create_variations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Embeddings Endpoint
# ============================================================================


class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    encoding_format: Optional[Literal["float", "base64"]] = "float"


@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    """
    OpenAI-compatible embeddings endpoint.

    Supports any HuggingFace encoder model for text embeddings.
    """
    try:
        model = await load_encoder(request.model, task="embedding")

        # Normalize input to list
        texts = [request.input] if isinstance(request.input, str) else request.input

        # Generate embeddings
        embeddings = await model.embed(texts, normalize=True)

        # Format response
        data = []
        for idx, embedding in enumerate(embeddings):
            if request.encoding_format == "base64":
                import struct

                embedding_bytes = struct.pack(f"{len(embedding)}f", *embedding)
                embedding_data = base64.b64encode(embedding_bytes).decode("utf-8")
            else:
                embedding_data = embedding

            data.append(
                {
                    "object": "embedding",
                    "index": idx,
                    "embedding": embedding_data,
                }
            )

        return {
            "object": "list",
            "data": data,
            "model": request.model,
            "usage": {
                "prompt_tokens": 0,  # TODO: Implement token counting
                "total_tokens": 0,
            },
        }

    except Exception as e:
        logger.error(f"Error in create_embeddings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Audio Endpoints
# ============================================================================


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0),
):
    """
    OpenAI-compatible audio transcription endpoint.

    Supports Whisper and other speech-to-text models.
    Accepts file uploads via multipart/form-data.
    """
    try:
        # Read the uploaded file bytes directly
        audio_bytes = await file.read()

        model_instance = await load_audio_model(model, task="transcribe")

        # Transcribe (AudioModel accepts both bytes and base64 strings)
        result = await model_instance.transcribe(
            audio=audio_bytes,
            language=language,
            prompt=prompt,
            temperature=temperature,
            return_timestamps=True,
        )

        # Format response based on requested format
        if response_format == "text":
            return Response(content=result["text"], media_type="text/plain")
        elif response_format == "verbose_json":
            return {
                "task": "transcribe",
                "language": language or "auto",
                "duration": 0.0,  # TODO: Get actual duration
                "text": result["text"],
                "words": result.get("words", []),
            }
        else:  # json
            return {"text": result["text"]}

    except Exception as e:
        logger.error(f"Error in create_transcription: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/audio/translations")
async def create_translation(
    file: UploadFile = File(...),
    model: str = Form(...),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0),
):
    """
    OpenAI-compatible audio translation endpoint.

    Translates audio to English (Whisper only).
    Accepts file uploads via multipart/form-data.
    """
    try:
        # Read the uploaded file bytes directly
        audio_bytes = await file.read()

        model_instance = await load_audio_model(model, task="translate")

        # Translate (AudioModel accepts both bytes and base64 strings)
        result = await model_instance.translate(
            audio=audio_bytes,
            target_language="en",
            temperature=temperature,
        )

        if response_format == "text":
            return Response(content=result["text"], media_type="text/plain")
        else:
            return {"text": result["text"]}

    except Exception as e:
        logger.error(f"Error in create_translation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Vision Endpoints
# ============================================================================


class VisionClassificationRequest(BaseModel):
    model: str
    images: List[str]  # Base64 encoded images
    top_k: Optional[int] = 5


@app.post("/v1/vision/classify")
async def classify_images(request: VisionClassificationRequest):
    """
    Classify images using vision models (JSON with base64).

    Supports ViT and other image classification models.
    """
    try:
        model = await load_vision_model(request.model, task="classification")

        results = await model.classify(request.images, top_k=request.top_k)

        return {
            "object": "list",
            "data": results,
            "model": request.model,
        }

    except Exception as e:
        logger.error(f"Error in classify_images: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/vision/classify/upload")
async def classify_image_upload(
    file: UploadFile = File(...),
    model: str = Form(...),
    top_k: Optional[int] = Form(5),
):
    """
    Classify an image using vision models (file upload).

    Accepts image uploads via multipart/form-data.
    More efficient than base64 encoding for large images.
    """
    try:
        # Read the uploaded file bytes directly
        image_bytes = await file.read()

        model_instance = await load_vision_model(model, task="classification")

        # Classify (VisionModel accepts bytes directly)
        results = await model_instance.classify([image_bytes], top_k=top_k)

        return {
            "object": "list",
            "data": results,
            "model": model,
        }

    except Exception as e:
        logger.error(f"Error in classify_image_upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class CLIPRequest(BaseModel):
    model: str
    images: List[str]  # Base64 encoded
    candidate_labels: List[str]


@app.post("/v1/vision/clip")
async def clip_classify(request: CLIPRequest):
    """
    Zero-shot image classification using CLIP (JSON with base64).
    """
    try:
        model = await load_vision_model(request.model, task="clip")

        results = await model.clip_classify(request.images, request.candidate_labels)

        return {
            "object": "list",
            "data": results,
            "model": request.model,
        }

    except Exception as e:
        logger.error(f"Error in clip_classify: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/vision/clip/upload")
async def clip_classify_upload(
    file: UploadFile = File(...),
    model: str = Form(...),
    labels: str = Form(...),  # Comma-separated labels
):
    """
    Zero-shot image classification using CLIP (file upload).

    Accepts image uploads via multipart/form-data.
    Labels should be comma-separated (e.g., "cat,dog,bird").
    """
    try:
        # Read the uploaded file bytes directly
        image_bytes = await file.read()

        # Parse comma-separated labels
        candidate_labels = [label.strip() for label in labels.split(",")]

        model_instance = await load_vision_model(model, task="clip")

        # Classify (VisionModel accepts bytes directly)
        results = await model_instance.clip_classify([image_bytes], candidate_labels)

        return {
            "object": "list",
            "data": results,
            "model": model,
        }

    except Exception as e:
        logger.error(f"Error in clip_classify_upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Multimodal Endpoints
# ============================================================================


class ImageCaptionRequest(BaseModel):
    model: str
    image: str  # Base64 encoded
    max_length: Optional[int] = 50


@app.post("/v1/multimodal/caption")
async def caption_image(request: ImageCaptionRequest):
    """
    Generate a caption for an image (JSON with base64).

    Supports BLIP, Florence, and other image-to-text models.
    """
    try:
        model = await load_multimodal_model(request.model, task="image-to-text")

        caption = await model.caption(request.image, max_length=request.max_length)

        return {
            "caption": caption,
            "model": request.model,
        }

    except Exception as e:
        logger.error(f"Error in caption_image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/multimodal/caption/upload")
async def caption_image_upload(
    file: UploadFile = File(...),
    model: str = Form(...),
    max_length: Optional[int] = Form(50),
):
    """
    Generate a caption for an image (file upload).

    Accepts image uploads via multipart/form-data.
    More efficient than base64 encoding.
    """
    try:
        # Read the uploaded file bytes directly
        image_bytes = await file.read()

        model_instance = await load_multimodal_model(model, task="image-to-text")

        # Caption (MultimodalModel accepts bytes directly)
        caption = await model_instance.caption(image_bytes, max_length=max_length)

        return {
            "caption": caption,
            "model": model,
        }

    except Exception as e:
        logger.error(f"Error in caption_image_upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class VQARequest(BaseModel):
    model: str
    image: str  # Base64 encoded
    question: str
    max_length: Optional[int] = 100


@app.post("/v1/multimodal/vqa")
async def visual_question_answering(request: VQARequest):
    """
    Answer questions about images (Visual Question Answering) (JSON with base64).

    Supports BLIP, LLaVA, and other VQA models.
    """
    try:
        model = await load_multimodal_model(request.model, task="vqa")

        answer = await model.answer_question(
            request.image, request.question, max_length=request.max_length
        )

        return {
            "answer": answer,
            "model": request.model,
        }

    except Exception as e:
        logger.error(f"Error in visual_question_answering: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/multimodal/vqa/upload")
async def visual_question_answering_upload(
    file: UploadFile = File(...),
    model: str = Form(...),
    question: str = Form(...),
    max_length: Optional[int] = Form(100),
):
    """
    Answer questions about images (Visual Question Answering) (file upload).

    Accepts image uploads via multipart/form-data.
    More efficient than base64 encoding.
    """
    try:
        # Read the uploaded file bytes directly
        image_bytes = await file.read()

        model_instance = await load_multimodal_model(model, task="vqa")

        # Answer question (MultimodalModel accepts bytes directly)
        answer = await model_instance.answer_question(
            image_bytes, question, max_length=max_length
        )

        return {
            "answer": answer,
            "model": model,
        }

    except Exception as e:
        logger.error(f"Error in visual_question_answering_upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("TRANSFORMERS_PORT", "11540"))
    host = os.getenv("TRANSFORMERS_HOST", "127.0.0.1")

    logger.info(f"Starting Transformers Runtime on {host}:{port}")
    logger.info(f"Device: {get_device()}")

    uvicorn.run(app, host=host, port=port)
