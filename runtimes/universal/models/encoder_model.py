"""
Encoder model wrapper for embeddings, classification, reranking, and NER.

Supports modern encoder architectures including:
- ModernBERT (8,192 token context, RoPE, Flash Attention)
- Classic BERT variants (512 token context)
- sentence-transformers models
- Cross-encoder rerankers.
"""

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from .base import BaseModel

logger = logging.getLogger(__name__)

EncoderTask = str  # "embedding", "classification", "reranking", "ner"


@dataclass
class NEREntity:
    """A named entity extracted from text."""

    text: str
    label: str
    start: int
    end: int
    score: float


class EncoderModel(BaseModel):
    """Wrapper for HuggingFace encoder models (BERT-style).

    Supports:
    - Embeddings: Generate dense vector representations
    - Classification: Sentiment, spam, intent routing, etc.
    - Reranking: Cross-encoder document reranking
    - NER: Named entity recognition

    Modern features:
    - Configurable max_length (up to 8,192 for ModernBERT)
    - Flash Attention 2 support for faster inference
    - Automatic model capability detection
    """

    # Known models with extended context
    EXTENDED_CONTEXT_MODELS = {
        "answerdotai/ModernBERT": 8192,
        "nomic-ai/nomic-embed-text": 8192,
        "jinaai/jina-embeddings-v2": 8192,
    }

    def __init__(
        self,
        model_id: str,
        device: str,
        task: str = "embedding",
        token: str | None = None,
        max_length: int | None = None,
        use_flash_attention: bool = True,
    ):
        """
        Initialize encoder model.

        Args:
            model_id: HuggingFace model ID
            device: Target device (cuda/mps/cpu)
            task: Model task - "embedding", "classification", "reranking", or "ner"
            token: HuggingFace authentication token for gated models
            max_length: Maximum sequence length (auto-detected if None)
            use_flash_attention: Whether to use Flash Attention 2 if available
        """
        super().__init__(model_id, device, token=token)
        self.task = task
        self.model_type = f"encoder_{task}"
        self.supports_streaming = False
        self._max_length = max_length
        self._use_flash_attention = use_flash_attention
        self._flash_attention_enabled: bool = False  # Track actual activation
        self._detected_max_length: int = 512  # Will be updated on load

    @property
    def max_length(self) -> int:
        """Get the effective max sequence length."""
        return self._max_length or self._detected_max_length

    async def load(self) -> None:
        """Load the encoder model with optimal settings."""
        logger.info(f"Loading encoder model ({self.task}): {self.model_id}")

        # Load config first to detect capabilities
        config = AutoConfig.from_pretrained(
            self.model_id, trust_remote_code=True, token=self.token
        )

        # Detect max sequence length
        self._detected_max_length = self._detect_max_length(config)
        logger.info(f"Detected max sequence length: {self._detected_max_length}")

        dtype = self.get_dtype()

        # Prepare model kwargs
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "token": self.token,
        }

        # Only set torch_dtype for non-CPU devices
        if self.device != "cpu":
            model_kwargs["torch_dtype"] = dtype

        # Enable Flash Attention 2 if available and requested
        if self._use_flash_attention and self._supports_flash_attention(config):
            model_kwargs["attn_implementation"] = "flash_attention_2"
            self._flash_attention_enabled = True
            logger.info("Flash Attention 2 enabled")

        # Load tokenizer
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True, token=self.token
        )

        # Load model based on task
        if self.task == "classification":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id, **model_kwargs
            )
        elif self.task == "reranking":
            # Rerankers are typically sequence classification models
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id, **model_kwargs
            )
        elif self.task == "ner":
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_id, **model_kwargs
            )
        else:  # embedding
            self.model = AutoModel.from_pretrained(self.model_id, **model_kwargs)

        if self.model is not None:
            self.model = self.model.to(self.device)
            self.model.eval()

        logger.info(
            f"Encoder model loaded on {self.device} "
            f"(max_length={self.max_length}, task={self.task})"
        )

    def _detect_max_length(self, config: AutoConfig) -> int:
        """Detect the maximum sequence length from model config.

        Args:
            config: Model configuration

        Returns:
            Maximum sequence length
        """
        # Check for known extended context models
        for prefix, length in self.EXTENDED_CONTEXT_MODELS.items():
            if self.model_id.startswith(prefix):
                return length

        # Try to get from config
        if hasattr(config, "max_position_embeddings"):
            return config.max_position_embeddings
        if hasattr(config, "max_seq_length"):
            return config.max_seq_length
        if hasattr(config, "n_positions"):
            return config.n_positions

        # Default for classic BERT
        return 512

    def _supports_flash_attention(self, config: AutoConfig) -> bool:
        """Check if model supports Flash Attention 2.

        Flash Attention requires:
        - CUDA device
        - Compatible architecture (BERT, RoBERTa, ModernBERT, etc.)
        - torch >= 2.0
        """
        if self.device != "cuda":
            return False

        # Check torch version
        torch_version = tuple(map(int, torch.__version__.split(".")[:2]))
        if torch_version < (2, 0):
            return False

        # Check if flash_attn is available
        try:
            import flash_attn  # noqa: F401

            return True
        except ImportError:
            logger.debug("flash_attn not installed, falling back to standard attention")
            return False

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling - take attention mask into account for correct averaging."""
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    async def embed(
        self, texts: list[str], normalize: bool = True
    ) -> list[list[float]]:
        """
        Generate embeddings for input texts.

        Args:
            texts: List of input texts
            normalize: Whether to L2 normalize embeddings

        Returns:
            List of embedding vectors
        """
        if self.task != "embedding":
            raise ValueError(f"Model task is '{self.task}', not 'embedding'")

        assert self.model is not None, "Model not loaded"
        assert self.tokenizer is not None, "Tokenizer not loaded"

        # Tokenize with configured max_length
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**encoded)

        # Mean pooling
        embeddings = self._mean_pooling(model_output, encoded["attention_mask"])

        # Normalize
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().tolist()

    async def classify(self, texts: list[str]) -> list[dict[str, Any]]:
        """
        Classify input texts.

        Args:
            texts: List of input texts

        Returns:
            List of classification results with labels and scores
        """
        if self.task != "classification":
            raise ValueError(f"Model task is '{self.task}', not 'classification'")

        assert self.model is not None, "Model not loaded"
        assert self.tokenizer is not None, "Tokenizer not loaded"

        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        # Classify
        with torch.no_grad():
            outputs = self.model(**encoded)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Format results
        results = []
        for pred in predictions:
            scores = pred.cpu().tolist()
            # Get top prediction
            max_idx = pred.argmax().item()

            # Get label mapping from config
            id2label = getattr(self.model.config, "id2label", {})

            result = {
                "label": id2label.get(max_idx, str(max_idx)),
                "score": scores[max_idx],
                "all_scores": {
                    id2label.get(i, str(i)): score for i, score in enumerate(scores)
                },
            }
            results.append(result)

        return results

    async def extract_entities(self, texts: list[str]) -> list[list[NEREntity]]:
        """
        Extract named entities from texts.

        Args:
            texts: List of input texts

        Returns:
            List of lists of NEREntity objects
        """
        if self.task != "ner":
            raise ValueError(f"Model task is '{self.task}', not 'ner'")

        assert self.model is not None, "Model not loaded"
        assert self.tokenizer is not None, "Tokenizer not loaded"

        results = []

        for text in texts:
            # Tokenize
            encoded = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                return_offsets_mapping=True,
            )

            # Get offset mapping and remove from encoded (model doesn't need it)
            offset_mapping = encoded.pop("offset_mapping")[0].tolist()
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**encoded)
                predictions = torch.argmax(outputs.logits, dim=-1)[0]
                scores = torch.softmax(outputs.logits, dim=-1)[0]

            # Get label mapping
            id2label = getattr(self.model.config, "id2label", {})

            # Extract entities using BIO/IOB format
            entities = []
            current_entity = None

            for idx, (pred_id, offset) in enumerate(
                zip(predictions, offset_mapping, strict=False)
            ):
                pred_id = pred_id.item()
                label = id2label.get(pred_id, f"LABEL_{pred_id}")
                score = scores[idx][pred_id].item()

                # Skip special tokens (offset is [0, 0])
                if offset[0] == 0 and offset[1] == 0 and idx > 0:
                    continue

                # Parse BIO tag
                if label.startswith("B-"):
                    # Start of new entity
                    if current_entity:
                        entities.append(current_entity)
                    entity_type = label[2:]
                    current_entity = NEREntity(
                        text=text[offset[0] : offset[1]],
                        label=entity_type,
                        start=offset[0],
                        end=offset[1],
                        score=score,
                    )
                elif label.startswith("I-") and current_entity:
                    # Continuation of entity
                    entity_type = label[2:]
                    if entity_type == current_entity.label:
                        current_entity.text = text[current_entity.start : offset[1]]
                        current_entity.end = offset[1]
                        current_entity.score = (current_entity.score + score) / 2
                    else:
                        # Different entity type, save current and start new
                        entities.append(current_entity)
                        current_entity = NEREntity(
                            text=text[offset[0] : offset[1]],
                            label=entity_type,
                            start=offset[0],
                            end=offset[1],
                            score=score,
                        )
                elif label != "O":
                    # Single token entity (no B-/I- prefix)
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None
                    entities.append(
                        NEREntity(
                            text=text[offset[0] : offset[1]],
                            label=label,
                            start=offset[0],
                            end=offset[1],
                            score=score,
                        )
                    )
                else:
                    # O tag - outside any entity
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None

            # Don't forget the last entity
            if current_entity:
                entities.append(current_entity)

            results.append(entities)

        return results

    async def rerank(
        self, query: str, documents: list[str], top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Rerank documents using cross-encoder scoring.

        This performs proper cross-encoder reranking where query and document
        are jointly encoded together (not independently like bi-encoders).

        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Optional number of top results to return

        Returns:
            List of dicts with 'index', 'document', and 'relevance_score'
        """
        if self.task != "reranking":
            raise ValueError(f"Model task is '{self.task}', not 'reranking'")

        assert self.model is not None, "Model not loaded"
        assert self.tokenizer is not None, "Tokenizer not loaded"

        scores = []

        # Score each query-document pair
        for doc_idx, doc in enumerate(documents):
            # Tokenize query and document together (key for cross-encoding!)
            encoded = self.tokenizer(
                query,
                doc,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # Get relevance score
            with torch.no_grad():
                outputs = self.model(**encoded)
                logits = outputs.logits

                # Extract score (handle both single and multi-class outputs)
                if logits.shape[-1] == 1:
                    score = logits[0][0].item()
                else:
                    # Multi-class output, take the positive class
                    score = logits[0][-1].item()

            scores.append({"index": doc_idx, "document": doc, "relevance_score": score})

        # Sort by relevance score (descending)
        scores.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Apply top_k if specified
        if top_k is not None:
            scores = scores[:top_k]

        return scores

    async def generate(self, *args, **kwargs):
        """Not applicable for encoder models."""
        raise NotImplementedError("Encoder models do not support text generation")

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        info = super().get_model_info()
        info.update(
            {
                "task": self.task,
                "max_length": self.max_length,
                "flash_attention": self._flash_attention_enabled,
            }
        )
        return info
