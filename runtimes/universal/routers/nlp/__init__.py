"""NLP router for embeddings, reranking, classification, and NER."""

from .router import router, set_encoder_loader

__all__ = ["router", "set_encoder_loader"]
