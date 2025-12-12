"""
Document understanding model wrapper for document extraction and VQA.

Supports models like:
- LayoutLMv3: Layout-aware document understanding with text + layout + image
- Donut: End-to-end document understanding without OCR
- DocVQA models: Document Visual Question Answering
"""

import base64
import io
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from PIL import Image

from .base import BaseModel

logger = logging.getLogger(__name__)

DocumentTask = Literal["extraction", "vqa", "classification"]


@dataclass
class ExtractedField:
    """A field extracted from a document."""

    key: str
    value: str
    confidence: float
    bbox: tuple[float, float, float, float] | None = None  # x1, y1, x2, y2


@dataclass
class DocumentResult:
    """Result from document processing."""

    text: str = ""
    fields: list[ExtractedField] = field(default_factory=list)
    classification: str | None = None
    classification_scores: dict[str, float] | None = None
    answer: str | None = None  # For VQA
    confidence: float = 0.0


class DocumentModel(BaseModel):
    """Wrapper for document understanding models.

    Supports:
    - Document extraction: Extract structured data from forms, invoices, receipts
    - Document VQA: Answer questions about document content
    - Document classification: Classify document types

    Model types:
    - LayoutLMv3: Uses layout + text + image features, requires OCR
    - Donut: End-to-end, no OCR needed, generates text directly
    """

    # Model type detection patterns
    DONUT_PATTERNS = ["donut", "naver-clova-ix"]
    LAYOUTLM_PATTERNS = ["layoutlm", "layoutlmv2", "layoutlmv3"]

    def __init__(
        self,
        model_id: str,
        device: str,
        task: DocumentTask = "extraction",
        token: str | None = None,
    ):
        """Initialize document model.

        Args:
            model_id: HuggingFace model ID
            device: Target device (cuda/mps/cpu)
            task: Model task - "extraction", "vqa", or "classification"
            token: HuggingFace authentication token
        """
        super().__init__(model_id, device, token=token)
        self.task = task
        self.model_type = f"document_{task}"
        self.supports_streaming = False
        self._model_family: str = "unknown"

    def _detect_model_family(self) -> str:
        """Detect the model family from model_id."""
        model_lower = self.model_id.lower()

        for pattern in self.DONUT_PATTERNS:
            if pattern in model_lower:
                return "donut"

        for pattern in self.LAYOUTLM_PATTERNS:
            if pattern in model_lower:
                return "layoutlm"

        # Default to donut-style (generates text)
        return "donut"

    async def load(self) -> None:
        """Load the document understanding model."""
        logger.info(f"Loading document model ({self.task}): {self.model_id}")

        self._model_family = self._detect_model_family()
        logger.info(f"Detected model family: {self._model_family}")

        dtype = self.get_dtype()

        if self._model_family == "donut":
            await self._load_donut(dtype)
        elif self._model_family == "layoutlm":
            await self._load_layoutlm(dtype)
        else:
            raise ValueError(f"Unsupported model family: {self._model_family}")

        logger.info(f"Document model loaded on {self.device}")

    async def _load_donut(self, dtype: torch.dtype) -> None:
        """Load a Donut-style model."""
        from transformers import DonutProcessor, VisionEncoderDecoderModel

        self.processor = DonutProcessor.from_pretrained(
            self.model_id, trust_remote_code=True, token=self.token
        )

        model_kwargs = {
            "trust_remote_code": True,
            "token": self.token,
        }
        if self.device != "cpu":
            model_kwargs["torch_dtype"] = dtype

        self.model = VisionEncoderDecoderModel.from_pretrained(
            self.model_id, **model_kwargs
        )

        if self.model is not None:
            self.model = self.model.to(self.device)
            self.model.eval()

    async def _load_layoutlm(self, dtype: torch.dtype) -> None:
        """Load a LayoutLM-style model."""
        from transformers import (
            AutoModelForDocumentQuestionAnswering,
            AutoModelForSequenceClassification,
            AutoModelForTokenClassification,
            AutoProcessor,
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True, token=self.token, apply_ocr=True
        )

        model_kwargs = {
            "trust_remote_code": True,
            "token": self.token,
        }
        if self.device != "cpu":
            model_kwargs["torch_dtype"] = dtype

        # Load appropriate model class based on task
        if self.task == "vqa":
            self.model = AutoModelForDocumentQuestionAnswering.from_pretrained(
                self.model_id, **model_kwargs
            )
        elif self.task == "classification":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id, **model_kwargs
            )
        else:  # extraction
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_id, **model_kwargs
            )

        if self.model is not None:
            self.model = self.model.to(self.device)
            self.model.eval()

    def _decode_image(self, img_data: str | bytes) -> Image.Image:
        """Decode image from base64 string or bytes."""
        if isinstance(img_data, str):
            if img_data.startswith("data:"):
                img_data = img_data.split(",", 1)[1]
            img_bytes = base64.b64decode(img_data)
        else:
            img_bytes = img_data

        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    async def extract(
        self,
        images: list[str | bytes],
        prompts: list[str] | None = None,
    ) -> list[DocumentResult]:
        """Extract information from document images.

        For Donut models: Uses prompts to guide extraction
        For LayoutLM models: Uses token classification for field extraction

        Args:
            images: List of document images (base64 or bytes)
            prompts: Optional prompts for extraction (Donut) or questions (VQA)

        Returns:
            List of DocumentResult objects
        """
        results = []

        for idx, img_data in enumerate(images):
            image = self._decode_image(img_data)
            prompt = prompts[idx] if prompts and idx < len(prompts) else None

            if self._model_family == "donut":
                result = await self._extract_donut(image, prompt)
            elif self._model_family == "layoutlm":
                result = await self._extract_layoutlm(image, prompt)
            else:
                raise ValueError(f"Unsupported model family: {self._model_family}")

            results.append(result)

        return results

    # Prompt format registry - extensible mapping of model patterns to prompt formatters
    # Each entry: pattern -> (format_func, default_prompt)
    # format_func takes (prompt: str | None) -> str
    PROMPT_FORMATS: dict[str, tuple[callable, str]] = {
        "docvqa": (
            lambda p: f"<s_docvqa><s_question>{p}</s_question><s_answer>"
            if p
            else "<s_docvqa><s_question>What is this document about?</s_question><s_answer>",
            "What is this document about?",
        ),
        "cord": (
            lambda p: p if p else "<s_cord-v2>",
            "<s_cord-v2>",
        ),
        "rvlcdip": (
            lambda p: p if p else "<s_rvlcdip>",
            "<s_rvlcdip>",
        ),
        "zhtrainticket": (
            lambda p: p if p else "<s_zhtrainticket>",
            "<s_zhtrainticket>",
        ),
        "synthdog": (
            lambda p: p if p else "<s_synthdog>",
            "<s_synthdog>",
        ),
    }

    def _format_prompt(self, prompt: str | None) -> str:
        """Format prompt based on model type using the prompt format registry.

        Args:
            prompt: User-provided prompt or None

        Returns:
            Formatted prompt string for the model
        """
        model_lower = self.model_id.lower()

        # Find matching format
        for pattern, (formatter, _) in self.PROMPT_FORMATS.items():
            if pattern in model_lower:
                return formatter(prompt)

        # Default - use prompt as-is or generic start token
        return prompt if prompt else "<s>"

    async def _extract_donut(
        self, image: Image.Image, prompt: str | None
    ) -> DocumentResult:
        """Extract using Donut model."""
        task_prompt = self._format_prompt(prompt)
        logger.debug(f"Using task prompt: {task_prompt[:100]}...")

        # Prepare pixel values and move to device with correct dtype
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        pixel_values = self.to_device(pixel_values)

        # Prepare decoder input
        decoder_input_ids = self.processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        decoder_input_ids = decoder_input_ids.to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=self.model.decoder.config.max_position_embeddings,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Decode output
        generated_text = self.processor.batch_decode(outputs.sequences)[0]

        # Clean up special tokens and parse JSON-like output
        generated_text = generated_text.replace(self.processor.tokenizer.eos_token, "")
        generated_text = generated_text.replace(self.processor.tokenizer.pad_token, "")

        # Try to parse extracted fields
        fields = self._parse_donut_output(generated_text)

        return DocumentResult(
            text=generated_text,
            fields=fields,
            confidence=0.9,  # Donut doesn't provide per-field confidence
        )

    def _parse_donut_output(self, text: str) -> list[ExtractedField]:
        """Parse Donut's structured output into fields."""
        import re

        fields = []

        # Parse key-value pairs from Donut's XML-like output
        # Pattern: <s_key>value</s_key> or <key>value</key>
        pattern = r"<s?_?(\w+)>(.*?)</s?_?\1>"
        matches = re.findall(pattern, text, re.DOTALL)

        for key, value in matches:
            value = value.strip()
            if value:
                fields.append(
                    ExtractedField(
                        key=key,
                        value=value,
                        confidence=0.9,
                        bbox=None,
                    )
                )

        return fields

    async def _extract_layoutlm(
        self, image: Image.Image, question: str | None
    ) -> DocumentResult:
        """Extract using LayoutLM model."""
        if self.task == "vqa":
            if not question:
                raise ValueError(
                    "VQA task requires a question/prompt. "
                    "Please provide a question in the 'prompts' field."
                )
            return await self._layoutlm_vqa(image, question)
        elif self.task == "classification":
            return await self._layoutlm_classify(image)
        else:
            return await self._layoutlm_extract(image)

    async def _layoutlm_vqa(self, image: Image.Image, question: str) -> DocumentResult:
        """Answer question about document."""
        # Process image with OCR
        encoding = self.processor(
            image, question, return_tensors="pt", truncation=True, max_length=512
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**encoding)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            # Get best answer span
            start_idx = torch.argmax(start_logits).item()
            end_idx = torch.argmax(end_logits).item()

            # Handle invalid span (end before start) - model is uncertain
            # Swap indices to get a valid span, or return empty answer
            if end_idx < start_idx:
                # Try swapping - sometimes the model predicts them reversed
                start_idx, end_idx = end_idx, start_idx
                logger.debug(
                    f"VQA: Swapped start/end indices (model uncertainty): {start_idx}-{end_idx}"
                )

            # Decode answer
            answer_ids = encoding["input_ids"][0][start_idx : end_idx + 1]
            answer = self.processor.tokenizer.decode(
                answer_ids, skip_special_tokens=True
            )

            # If answer is empty after swap, indicate no answer found
            if not answer.strip():
                answer = "[No answer found]"
                confidence = 0.0
            else:
                # Calculate confidence
                start_prob = torch.softmax(start_logits, dim=-1)[0][start_idx].item()
                end_prob = torch.softmax(end_logits, dim=-1)[0][end_idx].item()
                confidence = (start_prob + end_prob) / 2

        return DocumentResult(
            answer=answer,
            confidence=confidence,
        )

    async def _layoutlm_classify(self, image: Image.Image) -> DocumentResult:
        """Classify document type."""
        encoding = self.processor(image, return_tensors="pt", truncation=True)
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**encoding)
            predictions = torch.softmax(outputs.logits, dim=-1)[0]

            # Get top prediction
            max_idx = torch.argmax(predictions).item()
            id2label = getattr(self.model.config, "id2label", {})

            classification = id2label.get(max_idx, str(max_idx))
            scores = {
                id2label.get(i, str(i)): predictions[i].item()
                for i in range(len(predictions))
            }

        return DocumentResult(
            classification=classification,
            classification_scores=scores,
            confidence=predictions[max_idx].item(),
        )

    async def _layoutlm_extract(self, image: Image.Image) -> DocumentResult:
        """Extract fields using token classification."""
        encoding = self.processor(image, return_tensors="pt", truncation=True)

        # Store original words and boxes for reconstruction
        words = encoding.get("words", [[]])[0] if "words" in encoding else []
        boxes = encoding.get("bbox", [[]])[0] if "bbox" in encoding else []

        encoding = {
            k: v.to(self.device)
            for k, v in encoding.items()
            if k not in ["words", "overflow_to_sample_mapping"]
        }

        with torch.no_grad():
            outputs = self.model(**encoding)
            predictions = torch.argmax(outputs.logits, dim=-1)[0]
            scores = torch.softmax(outputs.logits, dim=-1)[0]

        # Get label mapping
        id2label = getattr(self.model.config, "id2label", {})

        # Extract fields using BIO format
        fields = []
        current_field = None

        for idx, (pred_id, word) in enumerate(zip(predictions, words, strict=False)):
            if idx >= len(predictions):
                break

            label = id2label.get(pred_id.item(), "O")
            score = scores[idx][pred_id].item()

            bbox = boxes[idx] if idx < len(boxes) else None

            if label.startswith("B-"):
                if current_field:
                    fields.append(current_field)
                field_type = label[2:]
                current_field = ExtractedField(
                    key=field_type,
                    value=word,
                    confidence=score,
                    bbox=tuple(bbox.tolist()) if bbox is not None else None,
                )
            elif label.startswith("I-") and current_field:
                current_field.value += " " + word
                current_field.confidence = (current_field.confidence + score) / 2
            elif label != "O":
                if current_field:
                    fields.append(current_field)
                    current_field = None
                fields.append(
                    ExtractedField(
                        key=label,
                        value=word,
                        confidence=score,
                        bbox=tuple(bbox.tolist()) if bbox is not None else None,
                    )
                )
            else:
                if current_field:
                    fields.append(current_field)
                    current_field = None

        if current_field:
            fields.append(current_field)

        return DocumentResult(
            fields=fields,
            confidence=sum(f.confidence for f in fields) / len(fields) if fields else 0,
        )

    async def answer_question(
        self, images: list[str | bytes], questions: list[str]
    ) -> list[DocumentResult]:
        """Answer questions about documents (VQA task).

        Args:
            images: List of document images
            questions: List of questions (one per image)

        Returns:
            List of DocumentResult with answers
        """
        if self.task != "vqa":
            raise ValueError(f"Model task is '{self.task}', not 'vqa'")

        return await self.extract(images, prompts=questions)

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        info = super().get_model_info()
        info.update(
            {
                "task": self.task,
                "model_family": self._model_family,
            }
        )
        return info
