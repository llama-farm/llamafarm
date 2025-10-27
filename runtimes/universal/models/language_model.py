"""
Language model wrapper for text generation or embedding.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
from typing import List, Optional, AsyncGenerator
import logging
from threading import Thread

from .base import BaseModel

logger = logging.getLogger(__name__)


class LanguageModel(BaseModel):
    """Wrapper for HuggingFace language models (GPT-style text generation)."""

    def __init__(self, model_id: str, device: str, token: Optional[str] = None):
        super().__init__(model_id, device, token=token)
        self.model_type = "language"
        self.supports_streaming = True

    async def load(self):
        """Load the causal language model."""
        logger.info(f"Loading causal LM: {self.model_id}")

        dtype = self.get_dtype()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            token=self.token,
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=dtype,
            trust_remote_code=True,
            device_map="auto" if self.device == "cuda" else None,
            token=self.token,
        )

        if self.device != "cuda":
            self.model = self.model.to(self.device)

        logger.info(f"Causal LM loaded on {self.device}")

    def format_messages(self, messages: List[dict]) -> str:
        """Format chat messages into a prompt."""
        # Try to use tokenizer's chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
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
        stop: Optional[List[str]] = None,
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
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        return generated_text.strip()

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
    ) -> AsyncGenerator[str, None]:
        """Generate text completion with streaming (yields tokens as they're generated)."""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        max_new_tokens = max_tokens or 512

        # Create a streamer that will yield tokens as they're generated
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        # Generation kwargs
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
        }

        # Run generation in a separate thread so we can stream the results
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Yield tokens as they become available
        for text in streamer:
            # Check for stop sequences
            if stop:
                for stop_seq in stop:
                    if stop_seq in text:
                        # Yield up to the stop sequence
                        idx = text.index(stop_seq)
                        if idx > 0:
                            yield text[:idx]
                        thread.join()
                        return
            yield text

        # Wait for generation to complete
        thread.join()
