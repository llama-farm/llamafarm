"""Tests for GGUF encoder model support in Universal Runtime."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from models.gguf_encoder_model import GGUFEncoderModel
from utils.model_format import (
    clear_format_cache,
    detect_model_format,
    get_gguf_file_path,
)


class TestGGUFEncoderModel:
    """Tests for GGUFEncoderModel class."""

    def test_model_initialization(self):
        """Test GGUF encoder model initialization."""
        model = GGUFEncoderModel("test/embed-model", "cpu")
        assert model.model_id == "test/embed-model"
        assert model.device == "cpu"
        assert model.model_type == "encoder_embedding"
        assert model.supports_streaming is False
        assert model.llama is None  # Not loaded yet

    @pytest.mark.asyncio
    async def test_load_model_cpu(self, tmp_path):
        """Test loading GGUF embedding model for CPU."""
        # Create mock GGUF file
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        gguf_file = model_dir / "model.gguf"
        gguf_file.write_text("mock gguf embedding content")

        model = GGUFEncoderModel("test/embed-model", "cpu")

        # Mock the Llama class
        mock_llama = MagicMock()

        with patch(
            "models.gguf_encoder_model.get_gguf_file_path", return_value=str(gguf_file)
        ):
            with patch(
                "models.gguf_encoder_model.Llama", return_value=mock_llama
            ) as mock_llama_cls:
                await model.load()
                assert model.llama is not None
                # Verify embedding=True was passed
                call_kwargs = mock_llama_cls.call_args[1]
                assert call_kwargs["embedding"] is True
                assert call_kwargs["n_gpu_layers"] == 0  # CPU mode

    @pytest.mark.asyncio
    async def test_load_model_gpu(self, tmp_path):
        """Test loading GGUF embedding model for GPU/MPS."""
        # Create mock GGUF file
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        gguf_file = model_dir / "model.gguf"
        gguf_file.write_text("mock gguf embedding content")

        model = GGUFEncoderModel("test/embed-model", "mps")

        # Mock the Llama class
        mock_llama = MagicMock()

        with patch(
            "models.gguf_encoder_model.get_gguf_file_path", return_value=str(gguf_file)
        ):
            with patch(
                "models.gguf_encoder_model.Llama", return_value=mock_llama
            ) as mock_llama_cls:
                await model.load()
                assert model.llama is not None
                # Verify that n_gpu_layers=-1 was passed for GPU
                call_kwargs = mock_llama_cls.call_args[1]
                assert call_kwargs["n_gpu_layers"] == -1
                assert call_kwargs["embedding"] is True

    @pytest.mark.asyncio
    async def test_embed_not_loaded(self):
        """Test embed raises error if model not loaded."""
        model = GGUFEncoderModel("test/embed-model", "cpu")

        with pytest.raises(AssertionError, match="Model not loaded"):
            await model.embed(["Hello"])

    @pytest.mark.asyncio
    async def test_embed_single_text(self, tmp_path):
        """Test embedding a single text."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        gguf_file = model_dir / "model.gguf"
        gguf_file.write_text("mock gguf embedding content")

        model = GGUFEncoderModel("test/embed-model", "cpu")

        # Mock llama instance that returns embedding result
        mock_llama = MagicMock()
        # Mock the create_embedding response format
        mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_llama.create_embedding.return_value = {
            "data": [{"embedding": mock_embedding}]
        }

        with patch(
            "models.gguf_encoder_model.get_gguf_file_path", return_value=str(gguf_file)
        ):
            with patch("models.gguf_encoder_model.Llama", return_value=mock_llama):
                await model.load()
                embeddings = await model.embed(["Hello world"], normalize=False)

                assert isinstance(embeddings, list)
                assert len(embeddings) == 1
                assert isinstance(embeddings[0], list)
                assert len(embeddings[0]) == 5

    @pytest.mark.asyncio
    async def test_embed_batch(self, tmp_path):
        """Test embedding multiple texts."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        gguf_file = model_dir / "model.gguf"
        gguf_file.write_text("mock gguf embedding content")

        model = GGUFEncoderModel("test/embed-model", "cpu")

        # Mock llama instance
        mock_llama = MagicMock()

        def mock_create_embedding(input):
            """Mock different embeddings for different inputs."""
            embeddings_map = {
                "Hello": [0.1, 0.2, 0.3],
                "World": [0.4, 0.5, 0.6],
                "Test": [0.7, 0.8, 0.9],
            }
            return {"data": [{"embedding": embeddings_map.get(input, [0.0, 0.0, 0.0])}]}

        mock_llama.create_embedding.side_effect = mock_create_embedding

        with patch(
            "models.gguf_encoder_model.get_gguf_file_path", return_value=str(gguf_file)
        ):
            with patch("models.gguf_encoder_model.Llama", return_value=mock_llama):
                await model.load()
                embeddings = await model.embed(
                    ["Hello", "World", "Test"], normalize=False
                )

                assert isinstance(embeddings, list)
                assert len(embeddings) == 3
                # All embeddings should have same dimension
                dims = [len(emb) for emb in embeddings]
                assert dims == [3, 3, 3]

    @pytest.mark.asyncio
    async def test_embed_normalization(self, tmp_path):
        """Test that normalization works correctly."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        gguf_file = model_dir / "model.gguf"
        gguf_file.write_text("mock gguf embedding content")

        model = GGUFEncoderModel("test/embed-model", "cpu")

        # Mock llama instance with unnormalized embedding
        mock_llama = MagicMock()
        mock_embedding = [3.0, 4.0]  # Length = 5.0
        mock_llama.create_embedding.return_value = {
            "data": [{"embedding": mock_embedding}]
        }

        with patch(
            "models.gguf_encoder_model.get_gguf_file_path", return_value=str(gguf_file)
        ):
            with patch("models.gguf_encoder_model.Llama", return_value=mock_llama):
                await model.load()

                # Get normalized embedding
                embeddings_normalized = await model.embed(["Test"], normalize=True)

                # Check that it's normalized (L2 norm should be 1.0)
                emb_array = np.array(embeddings_normalized[0])
                norm = np.linalg.norm(emb_array)
                assert np.isclose(norm, 1.0, atol=1e-6)

                # Expected normalized values: [3/5, 4/5] = [0.6, 0.8]
                assert np.isclose(embeddings_normalized[0][0], 0.6, atol=1e-6)
                assert np.isclose(embeddings_normalized[0][1], 0.8, atol=1e-6)

    @pytest.mark.asyncio
    async def test_embed_no_normalization(self, tmp_path):
        """Test that unnormalized embeddings are preserved."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        gguf_file = model_dir / "model.gguf"
        gguf_file.write_text("mock gguf embedding content")

        model = GGUFEncoderModel("test/embed-model", "cpu")

        # Mock llama instance with unnormalized embedding
        mock_llama = MagicMock()
        mock_embedding = [3.0, 4.0]  # Length = 5.0
        mock_llama.create_embedding.return_value = {
            "data": [{"embedding": mock_embedding}]
        }

        with patch(
            "models.gguf_encoder_model.get_gguf_file_path", return_value=str(gguf_file)
        ):
            with patch("models.gguf_encoder_model.Llama", return_value=mock_llama):
                await model.load()

                # Get unnormalized embedding
                embeddings_unnormalized = await model.embed(["Test"], normalize=False)

                # Should preserve original values
                assert embeddings_unnormalized[0][0] == 3.0
                assert embeddings_unnormalized[0][1] == 4.0

    @pytest.mark.asyncio
    async def test_generate_not_supported(self):
        """Test that generate raises NotImplementedError."""
        model = GGUFEncoderModel("test/embed-model", "cpu")

        with pytest.raises(NotImplementedError, match="do not support text generation"):
            await model.generate("Hello")

    def test_model_info(self):
        """Test get_model_info returns correct information."""
        model = GGUFEncoderModel("test/embed-model", "mps")
        info = model.get_model_info()

        assert info["model_id"] == "test/embed-model"
        assert info["model_type"] == "encoder_embedding"
        assert info["device"] == "mps"
        assert info["supports_streaming"] is False


@pytest.mark.integration
class TestGGUFEncoderIntegration:
    """Integration tests for GGUF encoder support (requires actual model download)."""

    @pytest.mark.skip(
        reason="Requires downloading actual GGUF embedding model - run manually"
    )
    @pytest.mark.asyncio
    async def test_real_gguf_embedding_model_load(self):
        """Test loading a real GGUF embedding model from HuggingFace."""
        # This test downloads a small GGUF embedding model - skip by default
        # Example models: nomic-ai/nomic-embed-text-v1.5-GGUF, mixedbread-ai/mxbai-embed-xsmall-v1
        model_id = "nomic-ai/nomic-embed-text-v1.5-GGUF"
        model = GGUFEncoderModel(model_id, "cpu")
        await model.load()
        assert model.llama is not None

    @pytest.mark.skip(
        reason="Requires downloading actual GGUF embedding model - run manually"
    )
    @pytest.mark.asyncio
    async def test_real_gguf_embedding_generation(self):
        """Test embedding generation with a real GGUF model."""
        model_id = "nomic-ai/nomic-embed-text-v1.5-GGUF"
        model = GGUFEncoderModel(model_id, "cpu")
        await model.load()

        texts = ["Hello world", "How are you?"]
        embeddings = await model.embed(texts, normalize=True)

        assert isinstance(embeddings, list)
        assert len(embeddings) == 2
        assert all(isinstance(emb, list) for emb in embeddings)

        # Check that embeddings are normalized
        for emb in embeddings:
            norm = np.linalg.norm(np.array(emb))
            assert np.isclose(norm, 1.0, atol=1e-5)

    @pytest.mark.skip(
        reason="Requires downloading actual GGUF embedding model - run manually"
    )
    @pytest.mark.asyncio
    async def test_real_gguf_embedding_similarity(self):
        """Test that similar texts have similar embeddings."""
        model_id = "nomic-ai/nomic-embed-text-v1.5-GGUF"
        model = GGUFEncoderModel(model_id, "cpu")
        await model.load()

        texts = [
            "The cat sits on the mat",
            "A feline rests on a rug",
            "Python is a programming language",
        ]
        embeddings = await model.embed(texts, normalize=True)

        # Compute cosine similarities
        emb_array = np.array(embeddings)
        similarities = np.dot(emb_array, emb_array.T)

        # Similar sentences should have higher similarity
        sim_cat_feline = similarities[0, 1]
        sim_cat_python = similarities[0, 2]

        assert sim_cat_feline > sim_cat_python
