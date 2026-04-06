"""Pydantic models for fine-tuning API."""

from pydantic import BaseModel, Field, field_validator


class SFTRequest(BaseModel):
    """Supervised Fine-Tuning request."""
    
    model: str = Field(..., description="HF model name or local path")
    dataset: list[dict] = Field(..., description="Training data (inline)")
    dataset_format: str | None = Field(None, description="auto, sharegpt, alpaca, chat, raw")
    chat_template: str | None = Field(None, description="auto, llama-3, chatml, etc.")
    train_on_responses_only: bool = Field(True, description="Mask user turns in loss")
    output_dir: str | None = Field(None, description="Custom output directory")
    output_gguf: bool = Field(True, description="Export to GGUF format")
    quantization: str = Field("q8_0", description="GGUF quantization method")
    
    # LoRA params
    lora_rank: int = Field(16, ge=1, le=256, description="LoRA rank")
    lora_alpha: int = Field(16, ge=1, description="LoRA alpha")
    target_modules: list[str] | None = Field(None, description="LoRA target modules (auto if None)")
    
    # Training params
    epochs: int = Field(3, ge=1, le=100, description="Number of epochs")
    batch_size: int = Field(2, ge=1, le=128, description="Batch size")
    learning_rate: float = Field(2e-4, gt=0, description="Learning rate")
    max_seq_length: int = Field(2048, ge=128, le=32768, description="Max sequence length")
    max_steps: int | None = Field(None, description="Override epochs with step count")
    warmup_steps: int = Field(10, ge=0, description="Warmup steps")
    gradient_accumulation_steps: int = Field(2, ge=1, description="Gradient accumulation")
    
    @field_validator("dataset")
    @classmethod
    def validate_dataset_size(cls, v):
        if len(v) > 100_000:
            raise ValueError("Dataset too large (max 100K examples)")
        if len(v) == 0:
            raise ValueError("Dataset cannot be empty")
        return v


class CPTRequest(BaseModel):
    """Continued Pre-Training request."""
    
    model: str = Field(..., description="HF model name or local path")
    dataset: list[dict] = Field(..., description="Training data [{'text': '...'}, ...]")
    output_dir: str | None = Field(None, description="Custom output directory")
    output_gguf: bool = Field(True, description="Export to GGUF format")
    quantization: str = Field("q8_0", description="GGUF quantization method")
    
    # LoRA params
    lora_rank: int = Field(16, ge=1, le=256, description="LoRA rank")
    lora_alpha: int = Field(16, ge=1, description="LoRA alpha")
    
    # Training params
    epochs: int = Field(1, ge=1, le=100, description="Number of epochs")
    batch_size: int = Field(2, ge=1, le=128, description="Batch size")
    learning_rate: float = Field(5e-5, gt=0, description="Learning rate")
    embedding_learning_rate: float = Field(5e-6, gt=0, description="Embedding layer LR")
    max_seq_length: int = Field(2048, ge=128, le=32768, description="Max sequence length")
    max_steps: int | None = Field(None, description="Override epochs with step count")
    
    @field_validator("dataset")
    @classmethod
    def validate_dataset_size(cls, v):
        if len(v) > 100_000:
            raise ValueError("Dataset too large (max 100K examples)")
        if len(v) == 0:
            raise ValueError("Dataset cannot be empty")
        return v


class ValidateRequest(BaseModel):
    """Dataset validation request."""
    
    dataset: list[dict] = Field(..., description="Dataset to validate")
    dataset_format: str | None = Field(None, description="Expected format")
    chat_template: str | None = Field(None, description="Chat template")


class JobStatus(BaseModel):
    """Training job status."""
    
    job_id: str
    status: str  # queued, running, completed, failed, cancelled
    type: str  # sft, cpt
    model: str
    progress: float = Field(0.0, ge=0.0, le=1.0)
    metrics: dict | None = None
    output_dir: str | None = None
    error: str | None = None
    created_at: float
    started_at: float | None = None
    completed_at: float | None = None
