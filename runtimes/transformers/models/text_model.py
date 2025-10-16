"""
Text generation model wrapper.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Optional
import logging

from .base import BaseModel

logger = logging.getLogger(__name__)


class TextModel(BaseModel):
    """Wrapper for HuggingFace text generation models."""

    def __init__(self, model_id: str, device: str):
        super().__init__(model_id, device)
        self.model_type = "text"

    async def load(self):
        """Load the text generation model."""
        logger.info(f"Loading text model: {self.model_id}")

        torch_dtype = self.get_torch_dtype()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map="auto" if self.device == "cuda" else None
        )

        if self.device != "cuda":
            self.model = self.model.to(self.device)

        logger.info(f"Text model loaded on {self.device}")

    def format_messages(self, messages: List[dict]) -> str:
        """Format chat messages into a prompt."""
        # Try to use tokenizer's chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                pass

        # Fallback to simple concatenation
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt_parts.append(f"{role.capitalize()}: {content}")

        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)

    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None
    ) -> str:
        """Generate text completion."""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        max_new_tokens = max_tokens or 512

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode only the new tokens
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return generated_text.strip()
