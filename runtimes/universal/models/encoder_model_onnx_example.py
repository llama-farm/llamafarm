"""
Example: ONNX-enabled Encoder Model with Backend Switching

This demonstrates how to modify EncoderModel to support both PyTorch and ONNX backends.
"""

import os
import logging
from typing import List, Dict, Any
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# ONNX dependencies (install with: uv pip install optimum[onnxruntime])
try:
    from optimum.onnxruntime import ORTModelForFeatureExtraction

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning(
        "ONNX support not available. Install with: pip install optimum[onnxruntime]"
    )

logger = logging.getLogger(__name__)


class EncoderModelWithONNX:
    """
    Encoder model that supports both PyTorch and ONNX backends.

    Backend selection:
    1. Environment variable: RUNTIME_BACKEND=pytorch|onnx
    2. Constructor parameter: EncoderModelWithONNX(..., backend="onnx")
    3. Automatic fallback to PyTorch if ONNX unavailable
    """

    def __init__(
        self, model_id: str, device: str, task: str = "embedding", backend: str = None
    ):
        self.model_id = model_id
        self.device = device
        self.task = task

        # Determine backend
        self.backend = backend or os.getenv("RUNTIME_BACKEND", "pytorch")

        # Fallback to PyTorch if ONNX not available
        if self.backend == "onnx" and not ONNX_AVAILABLE:
            logger.warning("ONNX requested but not available, falling back to PyTorch")
            self.backend = "pytorch"

        self.model_type = f"encoder_{task}"
        self.supports_streaming = False

        self.model = None
        self.tokenizer = None

        logger.info(f"Encoder initialized with {self.backend.upper()} backend")

    async def load(self):
        """Load model using appropriate backend."""
        if self.backend == "onnx":
            await self._load_onnx()
        else:
            await self._load_pytorch()

    async def _load_pytorch(self):
        """Load model using PyTorch (development/fallback)."""
        logger.info(f"Loading PyTorch encoder: {self.model_id}")

        dtype = torch.float16 if self.device != "cpu" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )

        self.model = AutoModel.from_pretrained(
            self.model_id,
            dtype=dtype,
            trust_remote_code=True,
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info(f"PyTorch encoder loaded on {self.device}")

    async def _load_onnx(self):
        """Load model using ONNX Runtime (production)."""
        logger.info(f"Loading ONNX encoder: {self.model_id}")

        # Determine ONNX provider
        if self.device == "cuda":
            provider = "CUDAExecutionProvider"
        elif self.device == "mps":
            # ONNX Runtime doesn't support MPS yet, fallback to CPU
            logger.warning("ONNX Runtime doesn't support MPS, using CPU")
            provider = "CPUExecutionProvider"
        else:
            provider = "CPUExecutionProvider"

        # Load tokenizer (same as PyTorch)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )

        # Load ONNX model (auto-converts if needed)
        self.model = ORTModelForFeatureExtraction.from_pretrained(
            self.model_id,
            export=True,  # Auto-export to ONNX if not already done
            provider=provider,
        )

        logger.info(f"ONNX encoder loaded with {provider}")

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling - works for both PyTorch and NumPy."""
        if self.backend == "pytorch":
            token_embeddings = model_output[0]
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )
        else:
            # ONNX returns NumPy arrays
            import numpy as np

            token_embeddings = model_output.last_hidden_state
            input_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(
                np.float32
            )
            input_mask_expanded = np.broadcast_to(
                input_mask_expanded, token_embeddings.shape
            )

            sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
            sum_mask = np.clip(
                np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None
            )
            return sum_embeddings / sum_mask

    async def embed(
        self, texts: List[str], normalize: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for input texts.
        Works with both PyTorch and ONNX backends.
        """
        if self.task != "embedding":
            raise ValueError(f"Model task is '{self.task}', not 'embedding'")

        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt" if self.backend == "pytorch" else "np",
        )

        if self.backend == "pytorch":
            # PyTorch inference
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                model_output = self.model(**encoded)

            # Mean pooling
            embeddings = self._mean_pooling(model_output, encoded["attention_mask"])

            # Normalize
            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)

            return embeddings.cpu().tolist()

        else:
            # ONNX inference
            import numpy as np

            # Run inference
            model_output = self.model(**encoded)

            # Mean pooling
            embeddings = self._mean_pooling(model_output, encoded["attention_mask"])

            # Normalize
            if normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / norms

            return embeddings.tolist()

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "device": self.device,
            "backend": self.backend,
            "supports_streaming": self.supports_streaming,
        }


# Example usage
if __name__ == "__main__":
    import asyncio

    async def test_both_backends():
        """Test the model with both backends."""
        model_id = "sentence-transformers/all-MiniLM-L6-v2"
        texts = ["Hello world", "How are you?"]

        print("=" * 60)
        print("Testing PyTorch Backend")
        print("=" * 60)

        pytorch_model = EncoderModelWithONNX(model_id, "cpu", backend="pytorch")
        await pytorch_model.load()
        pytorch_embeddings = await pytorch_model.embed(texts)
        print(
            f"PyTorch Embeddings shape: {len(pytorch_embeddings)}x{len(pytorch_embeddings[0])}"
        )
        print(f"First embedding (first 5 dims): {pytorch_embeddings[0][:5]}")

        if ONNX_AVAILABLE:
            print("\n" + "=" * 60)
            print("Testing ONNX Backend")
            print("=" * 60)

            onnx_model = EncoderModelWithONNX(model_id, "cpu", backend="onnx")
            await onnx_model.load()
            onnx_embeddings = await onnx_model.embed(texts)
            print(
                f"ONNX Embeddings shape: {len(onnx_embeddings)}x{len(onnx_embeddings[0])}"
            )
            print(f"First embedding (first 5 dims): {onnx_embeddings[0][:5]}")

            # Compare results
            import numpy as np

            diff = np.abs(np.array(pytorch_embeddings) - np.array(onnx_embeddings))
            print(f"\nMax difference between backends: {diff.max():.6f}")
            print(f"Mean difference: {diff.mean():.6f}")
        else:
            print("\n⚠️  ONNX not available, skipping ONNX test")

    asyncio.run(test_both_backends())


# Performance comparison helper
class PerformanceComparison:
    """Helper to benchmark PyTorch vs ONNX performance."""

    @staticmethod
    async def benchmark(
        model_id: str, texts: List[str], iterations: int = 100, device: str = "cpu"
    ):
        """Compare performance of PyTorch vs ONNX backends."""
        import time
        import numpy as np

        results = {}

        for backend in ["pytorch", "onnx"]:
            if backend == "onnx" and not ONNX_AVAILABLE:
                continue

            print(f"\nBenchmarking {backend.upper()}...")
            model = EncoderModelWithONNX(model_id, device, backend=backend)
            await model.load()

            # Warmup
            for _ in range(5):
                await model.embed(texts)

            # Benchmark
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                await model.embed(texts)
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            results[backend] = {
                "mean": np.mean(times) * 1000,  # ms
                "std": np.std(times) * 1000,
                "min": np.min(times) * 1000,
                "max": np.max(times) * 1000,
            }

            print(f"  Mean: {results[backend]['mean']:.2f}ms")
            print(f"  Std:  {results[backend]['std']:.2f}ms")

        if "pytorch" in results and "onnx" in results:
            speedup = results["pytorch"]["mean"] / results["onnx"]["mean"]
            print(f"\n🚀 ONNX is {speedup:.2f}x faster than PyTorch")

        return results
