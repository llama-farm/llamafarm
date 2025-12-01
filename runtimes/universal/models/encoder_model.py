"""
Encoder model wrapper for embeddings and classification.
"""

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from .base import BaseModel

logger = logging.getLogger(__name__)


class EncoderModel(BaseModel):
    """Wrapper for HuggingFace encoder models (BERT-style embeddings & classification)."""

    def __init__(
        self,
        model_id: str,
        device: str,
        task: str = "embedding",
        token: Optional[str] = None,
    ):
        """
        Initialize encoder model.

        Args:
            model_id: HuggingFace model ID
            device: Target device (cuda/mps/cpu)
            task: Model task - "embedding", "classification", or "reranking"
            token: HuggingFace authentication token for gated models
        """
        super().__init__(model_id, device, token=token)
        self.task = task
        self.model_type = f"encoder_{task}"
        self.supports_streaming = False

    async def load(self) -> None:
        """Load the encoder model."""
        logger.info(f"Loading encoder model ({self.task}): {self.model_id}")

        dtype = self.get_dtype()

        # Load tokenizer
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True, token=self.token
        )

        # Load model based on task
        if self.task == "classification":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id,
                dtype=dtype,
                trust_remote_code=True,
                token=self.token,
            )
        elif self.task == "reranking":
            # Rerankers are typically sequence classification models with 1 output class
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id,
                dtype=dtype,
                trust_remote_code=True,
                token=self.token,
            )
        else:  # embedding
            self.model = AutoModel.from_pretrained(
                self.model_id,
                dtype=dtype,
                trust_remote_code=True,
                token=self.token,
            )

        if self.model is not None:
            self.model = self.model.to(self.device)  # type: ignore[arg-type]
            self.model.eval()

        logger.info(f"Encoder model loaded on {self.device}")

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
        self, texts: List[str], normalize: bool = True
    ) -> List[List[float]]:
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

        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
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

    async def classify(self, texts: List[str]) -> List[Dict[str, Any]]:
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
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        # Classify
        with torch.no_grad():
            outputs = self.model(**encoded)  # type: ignore[misc]
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Format results
        results = []
        for pred in predictions:
            scores = pred.cpu().tolist()
            # Get top prediction
            max_idx = pred.argmax().item()

            result = {
                "label": self.model.config.id2label.get(max_idx, str(max_idx)),  # type: ignore[union-attr]
                "score": scores[max_idx],
                "all_scores": {
                    self.model.config.id2label.get(i, str(i)): score  # type: ignore[union-attr]
                    for i, score in enumerate(scores)
                },
            }
            results.append(result)

        return results

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
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
                max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # Get relevance score
            with torch.no_grad():
                outputs = self.model(**encoded)
                # For cross-encoder rerankers, logits represent relevance
                # Most models output [batch_size, num_labels] where num_labels=1
                logits = outputs.logits

                # Extract score (handle both single and multi-class outputs)
                if logits.shape[-1] == 1:
                    # Single output (common for rerankers)
                    score = logits[0][0].item()
                else:
                    # Multi-class output, take the positive class (usually index 1 or last)
                    score = logits[0][-1].item()

            scores.append({
                'index': doc_idx,
                'document': doc,
                'relevance_score': score
            })

        # Sort by relevance score (descending)
        scores.sort(key=lambda x: x['relevance_score'], reverse=True)

        # Apply top_k if specified
        if top_k is not None:
            scores = scores[:top_k]

        return scores

    async def generate(self, *args, **kwargs):
        """Not applicable for encoder models."""
        raise NotImplementedError("Encoder models do not support text generation")
