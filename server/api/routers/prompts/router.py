"""Prompts CRUD router for managing project prompts."""

from config.datamodel import PromptMessage, PromptSet
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.logging import FastAPIStructLogger
from services.prompts_service import PromptNotFoundError, PromptsService

logger = FastAPIStructLogger()

router = APIRouter(
    prefix="/projects/{namespace}/{project}/prompts",
    tags=["prompts"],
)


# ============================================================================
# Request/Response Models
# ============================================================================


class PromptMessageRequest(BaseModel):
    """Request model for a prompt message."""

    role: str = Field(
        ..., description='Message role (e.g., "system", "user", "assistant", "tool")'
    )
    content: str = Field(..., description="Message content")
    tool_call_id: str | None = Field(None, description="Tool call ID (optional)")


class PromptResponse(BaseModel):
    """Response model for a single prompt."""

    name: str = Field(..., description="Unique prompt identifier")
    messages: list[PromptMessageRequest] = Field(
        ..., description="List of messages in this prompt set"
    )


class ListPromptsResponse(BaseModel):
    """Response model for listing prompts."""

    total: int = Field(..., description="Total number of prompts")
    prompts: list[PromptResponse] = Field(..., description="List of prompts")


class CreatePromptRequest(BaseModel):
    """Request model for creating a new prompt."""

    name: str = Field(
        ...,
        description="Unique prompt identifier (lowercase, alphanumeric with underscores)",
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    messages: list[PromptMessageRequest] = Field(
        ..., description="List of messages in this prompt set", min_length=1
    )


class UpdatePromptRequest(BaseModel):
    """Request model for updating a prompt."""

    messages: list[PromptMessageRequest] = Field(
        ..., description="New list of messages", min_length=1
    )


class DeletePromptResponse(BaseModel):
    """Response model for prompt deletion."""

    message: str = Field(..., description="Deletion confirmation message")
    prompt: PromptResponse = Field(..., description="The deleted prompt")


# ============================================================================
# Helper Functions
# ============================================================================


def _prompt_set_to_response(prompt: PromptSet) -> PromptResponse:
    """Convert a PromptSet model to PromptResponse."""
    return PromptResponse(
        name=prompt.name,
        messages=[
            PromptMessageRequest(
                role=msg.role,
                content=msg.content,
                tool_call_id=msg.tool_call_id,
            )
            for msg in prompt.messages
        ],
    )


def _request_messages_to_model(
    messages: list[PromptMessageRequest],
) -> list[PromptMessage]:
    """Convert request messages to PromptMessage models."""
    return [
        PromptMessage(
            role=msg.role,
            content=msg.content,
            tool_call_id=msg.tool_call_id,
        )
        for msg in messages
    ]


# ============================================================================
# Endpoints
# ============================================================================


@router.get(
    "/",
    operation_id="prompts_list",
    tags=["mcp"],
    summary="List all prompts in a project",
    response_model=ListPromptsResponse,
)
@router.get("", include_in_schema=False)
async def list_prompts(namespace: str, project: str) -> ListPromptsResponse:
    """
    List all prompts configured in the project.

    Prompts are named collections of messages that can be referenced by models
    to provide system instructions, examples, or other context.
    """
    logger.bind(namespace=namespace, project=project)

    prompts = PromptsService.list_prompts(namespace, project)

    return ListPromptsResponse(
        total=len(prompts),
        prompts=[_prompt_set_to_response(p) for p in prompts],
    )


@router.get(
    "/{prompt_name}",
    operation_id="prompt_get",
    tags=["mcp"],
    summary="Get a single prompt by name",
    response_model=PromptResponse,
)
async def get_prompt(namespace: str, project: str, prompt_name: str) -> PromptResponse:
    """
    Get detailed information about a specific prompt.

    Returns the prompt name and all its messages.
    """
    logger.bind(namespace=namespace, project=project, prompt=prompt_name)

    try:
        prompt = PromptsService.get_prompt(namespace, project, prompt_name)
    except PromptNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Prompt '{prompt_name}' not found"
        ) from None

    return _prompt_set_to_response(prompt)


@router.post(
    "/",
    operation_id="prompt_create",
    tags=["mcp"],
    summary="Create a new prompt",
    response_model=PromptResponse,
    status_code=201,
)
async def create_prompt(
    namespace: str,
    project: str,
    request: CreatePromptRequest,
) -> PromptResponse:
    """
    Create a new prompt in the project configuration.

    The prompt will be added to the project's llamafarm.yaml config file.
    Prompt names must be unique and follow the naming pattern:
    lowercase letters, numbers, and underscores, starting with a letter.

    Example:
    ```json
    {
      "name": "code_assistant",
      "messages": [
        {"role": "system", "content": "You are an expert programmer."},
        {"role": "user", "content": "Help me write clean, maintainable code."}
      ]
    }
    ```
    """
    logger.bind(namespace=namespace, project=project, prompt=request.name)

    try:
        messages = _request_messages_to_model(request.messages)
        prompt = PromptsService.create_prompt(
            namespace=namespace,
            project=project,
            name=request.name,
            messages=messages,
        )
    except ValueError as e:
        error_msg = str(e)
        # Return 409 Conflict for duplicate prompt names
        if "already exists" in error_msg:
            raise HTTPException(status_code=409, detail=error_msg) from e
        raise HTTPException(status_code=400, detail=error_msg) from e

    return _prompt_set_to_response(prompt)


@router.put(
    "/{prompt_name}",
    operation_id="prompt_update",
    tags=["mcp"],
    summary="Update an existing prompt",
    response_model=PromptResponse,
)
async def update_prompt(
    namespace: str,
    project: str,
    prompt_name: str,
    request: UpdatePromptRequest,
) -> PromptResponse:
    """
    Update an existing prompt's messages.

    The prompt name cannot be changed. To rename a prompt,
    delete the old one and create a new one with the desired name.
    """
    logger.bind(namespace=namespace, project=project, prompt=prompt_name)

    try:
        messages = _request_messages_to_model(request.messages)
        prompt = PromptsService.update_prompt(
            namespace=namespace,
            project=project,
            name=prompt_name,
            messages=messages,
        )
    except PromptNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Prompt '{prompt_name}' not found"
        ) from None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return _prompt_set_to_response(prompt)


@router.delete(
    "/{prompt_name}",
    operation_id="prompt_delete",
    tags=["mcp"],
    summary="Delete a prompt",
    response_model=DeletePromptResponse,
    responses={
        200: {"model": DeletePromptResponse},
        404: {"description": "Prompt not found"},
        409: {"description": "Prompt is referenced by models"},
    },
)
async def delete_prompt(
    namespace: str,
    project: str,
    prompt_name: str,
) -> DeletePromptResponse:
    """
    Delete a prompt from the project.

    This will remove the prompt from the project's llamafarm.yaml config file.

    **Important**: You cannot delete a prompt that is referenced by any model.
    The endpoint will return a 409 Conflict error listing the dependent models if any exist.
    Update the models to remove the prompt reference first.
    """
    logger.bind(namespace=namespace, project=project, prompt=prompt_name)

    try:
        deleted_prompt = PromptsService.delete_prompt(
            namespace=namespace,
            project=project,
            name=prompt_name,
        )
    except PromptNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Prompt '{prompt_name}' not found"
        ) from None
    except ValueError as e:
        error_msg = str(e)
        # Check if this is a dependent models error (409 Conflict)
        if "model(s) reference it" in error_msg:
            raise HTTPException(status_code=409, detail=error_msg) from e
        raise HTTPException(status_code=400, detail=error_msg) from e

    return DeletePromptResponse(
        message=f"Prompt '{prompt_name}' deleted successfully",
        prompt=_prompt_set_to_response(deleted_prompt),
    )
