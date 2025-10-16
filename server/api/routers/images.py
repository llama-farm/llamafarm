"""
OpenAI-compatible Images API Router

Provides project-scoped image generation endpoints that route to the transformers runtime.
Follows OpenAI Images API specification:
- POST /v1/projects/{namespace}/{project}/images/generations
- POST /v1/projects/{namespace}/{project}/images/edits
- POST /v1/projects/{namespace}/{project}/images/variations

All endpoints are compatible with OpenAI's Python SDK and API specification.
"""

import sys
from pathlib import Path
from typing import Literal, Optional

import httpx
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from services.project_service import ProjectService

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))
from config.datamodel import LlamaFarmConfig  # noqa: E402


# ============================================================================
# OpenAI Images API Request/Response Models
# ============================================================================

class ImageGenerationRequest(BaseModel):
    """
    OpenAI-compatible image generation request.

    Spec: https://platform.openai.com/docs/api-reference/images/create
    """
    prompt: str = Field(..., description="A text description of the desired image(s)")
    model: Optional[str] = Field(None, description="The model to use for image generation")
    n: Optional[int] = Field(1, ge=1, le=10, description="The number of images to generate")
    quality: Optional[Literal["standard", "hd"]] = Field("standard", description="The quality of the image")
    response_format: Optional[Literal["url", "b64_json"]] = Field("url", description="The format of the generated images")
    size: Optional[Literal["256x256", "512x512", "768x768", "1024x1024", "1280x1280", "1792x1024", "1024x1792"]] = Field(
        "1024x1024",
        description="The size of the generated images"
    )
    style: Optional[Literal["vivid", "natural"]] = Field("vivid", description="The style of the generated images")
    user: Optional[str] = Field(None, description="A unique identifier representing your end-user")

    # Extended diffusion-specific parameters (not in OpenAI spec, but useful)
    negative_prompt: Optional[str] = Field(None, description="Text describing what to avoid in the image")
    num_inference_steps: Optional[int] = Field(None, ge=1, le=150, description="Number of denoising steps")
    guidance_scale: Optional[float] = Field(None, ge=1.0, le=20.0, description="How closely to follow the prompt")
    seed: Optional[int] = Field(None, description="Seed for reproducible generation")
    scheduler: Optional[str] = Field(None, description="Diffusion scheduler (ddim, euler, dpm++, etc.)")


class ImageEditRequest(BaseModel):
    """
    OpenAI-compatible image edit request.

    Spec: https://platform.openai.com/docs/api-reference/images/createEdit
    """
    prompt: str = Field(..., description="A text description of the desired image(s)")
    image: str = Field(..., description="The image to edit (base64 encoded)")
    mask: Optional[str] = Field(None, description="Mask image indicating areas to inpaint (base64 encoded)")
    model: Optional[str] = Field(None, description="The model to use for image editing")
    n: Optional[int] = Field(1, ge=1, le=10, description="The number of images to generate")
    size: Optional[Literal["256x256", "512x512", "1024x1024"]] = Field(
        "1024x1024",
        description="The size of the generated images"
    )
    response_format: Optional[Literal["url", "b64_json"]] = Field("url", description="The format of the generated images")
    user: Optional[str] = Field(None, description="A unique identifier representing your end-user")

    # Extended parameters
    negative_prompt: Optional[str] = Field(None, description="Text describing what to avoid")
    num_inference_steps: Optional[int] = Field(None, ge=1, le=150, description="Number of denoising steps")
    guidance_scale: Optional[float] = Field(None, ge=1.0, le=20.0, description="How closely to follow the prompt")
    seed: Optional[int] = Field(None, description="Seed for reproducible generation")


class ImageVariationRequest(BaseModel):
    """
    OpenAI-compatible image variation request.

    Spec: https://platform.openai.com/docs/api-reference/images/createVariation
    """
    image: str = Field(..., description="The image to use as the basis for variations (base64 encoded)")
    model: Optional[str] = Field(None, description="The model to use for generating variations")
    n: Optional[int] = Field(1, ge=1, le=10, description="The number of images to generate")
    response_format: Optional[Literal["url", "b64_json"]] = Field("url", description="The format of the generated images")
    size: Optional[Literal["256x256", "512x512", "1024x1024"]] = Field(
        "1024x1024",
        description="The size of the generated images"
    )
    user: Optional[str] = Field(None, description="A unique identifier representing your end-user")


class ImageObject(BaseModel):
    """
    OpenAI Image object response.

    Spec: https://platform.openai.com/docs/api-reference/images/object
    """
    b64_json: Optional[str] = Field(None, description="The base64-encoded JSON of the generated image")
    url: Optional[str] = Field(None, description="The URL of the generated image")
    revised_prompt: Optional[str] = Field(None, description="The prompt that was used to generate the image")


class ImagesResponse(BaseModel):
    """
    OpenAI Images API response.

    Spec: https://platform.openai.com/docs/api-reference/images/object
    """
    created: int = Field(..., description="The Unix timestamp (in seconds) when the images were created")
    data: list[ImageObject] = Field(..., description="The list of generated images")


# ============================================================================
# Router Setup
# ============================================================================

router = APIRouter(
    prefix="/projects/{namespace}/{project}/images",
    tags=["images"],
)


# ============================================================================
# Helper Functions
# ============================================================================

def get_model_config(config: LlamaFarmConfig, model_name: Optional[str] = None):
    """
    Get model configuration by name.

    Args:
        config: Project configuration
        model_name: Optional model name to look up

    Returns:
        Tuple of (model_config, base_url)

    Raises:
        HTTPException: If model not found or no image-capable models configured
    """
    if not config.runtime.models:
        raise HTTPException(
            status_code=400,
            detail="No models configured in project. Add models to runtime.models in llamafarm.yaml"
        )

    # If model_name specified, find it
    if model_name:
        model_config = next(
            (m for m in config.runtime.models if m.name == model_name),
            None
        )
        if not model_config:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found in project configuration"
            )
    else:
        # Use default model or first available
        model_config = next(
            (m for m in config.runtime.models if m.name == config.runtime.default_model),
            config.runtime.models[0] if config.runtime.models else None
        )

    if not model_config:
        raise HTTPException(
            status_code=400,
            detail="No model configuration found"
        )

    # Determine base URL based on provider
    from config.datamodel import Provider

    if model_config.provider == Provider.transformers:
        base_url = model_config.base_url or "http://127.0.0.1:11540"
    elif model_config.provider == Provider.openai:
        base_url = model_config.base_url or "https://api.openai.com"
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Provider '{model_config.provider}' does not support image generation"
        )

    return model_config, base_url.rstrip("/")


async def proxy_to_transformers_runtime(
    runtime_url: str,
    endpoint: str,
    request_data: dict,
    timeout: float = 300.0
) -> dict:
    """
    Proxy request to transformers runtime.

    Args:
        runtime_url: Base URL of transformers runtime
        endpoint: API endpoint path (e.g., /v1/images/generations)
        request_data: Request payload
        timeout: Request timeout in seconds

    Returns:
        Response data from transformers runtime

    Raises:
        HTTPException: If request fails
    """
    full_url = f"{runtime_url}{endpoint}"

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(
                full_url,
                json=request_data
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Transformers runtime error: {response.text}"
                )

            return response.json()

        except httpx.TimeoutException:
            raise HTTPException(
                status_code=504,
                detail=f"Request to transformers runtime timed out after {timeout}s"
            )
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail=f"Could not connect to transformers runtime at {runtime_url}. "
                       "Ensure the runtime is running: nx start transformers"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error communicating with transformers runtime: {str(e)}"
            )


# ============================================================================
# API Endpoints
# ============================================================================

@router.post(
    "/generations",
    response_model=ImagesResponse,
    summary="Create image",
    description="Creates an image given a prompt. OpenAI-compatible endpoint."
)
async def create_image(
    namespace: str,
    project: str,
    request: ImageGenerationRequest
) -> ImagesResponse:
    """
    Generate images from text prompts using diffusion models.

    This endpoint is fully compatible with OpenAI's Images API.

    Example:
        ```python
        import openai
        openai.api_base = "http://localhost:8000/v1/projects/default/my-project"

        response = openai.Image.create(
            prompt="a serene mountain landscape at sunset",
            n=1,
            size="1024x1024"
        )
        ```
    """
    # Load project configuration
    try:
        config = ProjectService.load_config(namespace, project)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Project not found: {str(e)}")

    # Get model configuration and runtime URL
    model_config, runtime_url = get_model_config(config, request.model)

    # Prepare request data - replace friendly model name with actual model ID
    request_data = request.dict(exclude_none=True)
    request_data['model'] = model_config.model  # Use actual model ID (e.g., HuggingFace model ID)

    # Apply diffusion config defaults from YAML if not explicitly set in request
    if hasattr(model_config, 'diffusion') and model_config.diffusion:
        diffusion_config = model_config.diffusion

        # Only set if not already provided in request
        if 'num_inference_steps' not in request_data and hasattr(diffusion_config, 'default_steps'):
            request_data['num_inference_steps'] = diffusion_config.default_steps

        if 'guidance_scale' not in request_data and hasattr(diffusion_config, 'default_guidance'):
            request_data['guidance_scale'] = diffusion_config.default_guidance

        if 'scheduler' not in request_data and hasattr(diffusion_config, 'scheduler') and diffusion_config.scheduler:
            # Convert enum to string value if needed
            scheduler_value = diffusion_config.scheduler
            if hasattr(scheduler_value, 'value'):
                request_data['scheduler'] = scheduler_value.value
            else:
                request_data['scheduler'] = str(scheduler_value)

        # Use default_size from config if size not explicitly set or is the default
        if hasattr(diffusion_config, 'default_size') and request.size == "1024x1024":
            # Convert enum to string value if needed
            size_value = diffusion_config.default_size
            if hasattr(size_value, 'value'):
                request_data['size'] = size_value.value
            else:
                request_data['size'] = str(size_value)

    # Proxy to runtime
    response_data = await proxy_to_transformers_runtime(
        runtime_url=runtime_url,
        endpoint="/v1/images/generations",
        request_data=request_data,
        timeout=300.0  # 5 minutes for image generation
    )

    return ImagesResponse(**response_data)


@router.post(
    "/edits",
    response_model=ImagesResponse,
    summary="Create image edit",
    description="Creates an edited or extended image given an original image and a prompt. OpenAI-compatible endpoint."
)
async def create_image_edit(
    namespace: str,
    project: str,
    request: ImageEditRequest
) -> ImagesResponse:
    """
    Edit images using inpainting models.

    This endpoint is fully compatible with OpenAI's Images API.

    Example:
        ```python
        import openai
        openai.api_base = "http://localhost:8000/v1/projects/default/my-project"

        response = openai.Image.create_edit(
            image=open("photo.png", "rb"),
            mask=open("mask.png", "rb"),
            prompt="add a rainbow in the sky",
            n=1,
            size="1024x1024"
        )
        ```
    """
    # Load project configuration
    try:
        config = ProjectService.load_config(namespace, project)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Project not found: {str(e)}")

    # Get model configuration and runtime URL
    model_config, runtime_url = get_model_config(config, request.model)

    # Prepare request data - replace friendly model name with actual model ID
    request_data = request.dict(exclude_none=True)
    request_data['model'] = model_config.model

    # Apply diffusion config defaults from YAML if not explicitly set in request
    if hasattr(model_config, 'diffusion') and model_config.diffusion:
        diffusion_config = model_config.diffusion

        if 'num_inference_steps' not in request_data and hasattr(diffusion_config, 'default_steps'):
            request_data['num_inference_steps'] = diffusion_config.default_steps

        if 'guidance_scale' not in request_data and hasattr(diffusion_config, 'default_guidance'):
            request_data['guidance_scale'] = diffusion_config.default_guidance

    # Proxy to transformers runtime
    response_data = await proxy_to_transformers_runtime(
        runtime_url=runtime_url,
        endpoint="/v1/images/edits",
        request_data=request_data,
        timeout=300.0
    )

    return ImagesResponse(**response_data)


@router.post(
    "/variations",
    response_model=ImagesResponse,
    summary="Create image variation",
    description="Creates a variation of a given image. OpenAI-compatible endpoint."
)
async def create_image_variation(
    namespace: str,
    project: str,
    request: ImageVariationRequest
) -> ImagesResponse:
    """
    Create variations of an input image.

    This endpoint is fully compatible with OpenAI's Images API.

    Example:
        ```python
        import openai
        openai.api_base = "http://localhost:8000/v1/projects/default/my-project"

        response = openai.Image.create_variation(
            image=open("artwork.png", "rb"),
            n=3,
            size="1024x1024"
        )
        ```
    """
    # Load project configuration
    try:
        config = ProjectService.load_config(namespace, project)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Project not found: {str(e)}")

    # Get model configuration and runtime URL
    model_config, runtime_url = get_model_config(config, request.model)

    # Prepare request data - replace friendly model name with actual model ID
    request_data = request.dict(exclude_none=True)
    request_data['model'] = model_config.model

    # Proxy to transformers runtime
    response_data = await proxy_to_transformers_runtime(
        runtime_url=runtime_url,
        endpoint="/v1/images/variations",
        request_data=request_data,
        timeout=300.0
    )

    return ImagesResponse(**response_data)
