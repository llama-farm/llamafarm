"""Tests for the unified ModelManager service."""

import asyncio
import time

import pytest

from services.model_manager import ModelManager, ModelType


class MockModel:
    """Mock model for testing."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.unloaded = False

    async def unload(self):
        self.unloaded = True


class TestModelManagerInit:
    """Test ModelManager initialization."""

    def test_initializes_with_default_ttl(self):
        """ModelManager initializes with default TTL."""
        manager = ModelManager()
        assert manager.ttl == 300

    def test_initializes_with_custom_ttl(self):
        """ModelManager initializes with custom TTL."""
        manager = ModelManager(ttl=600)
        assert manager.ttl == 600

    def test_creates_caches_for_all_model_types(self):
        """ModelManager creates a cache for each ModelType."""
        manager = ModelManager(ttl=300)
        for model_type in ModelType:
            assert model_type in manager._caches
            assert manager._caches[model_type].ttl == 300


class TestCacheKeyGeneration:
    """Test cache key generation methods."""

    def test_language_cache_key_basic(self):
        """Language cache key with basic params."""
        key = ModelManager.make_language_cache_key("meta-llama/Llama-2-7b")
        assert "language:meta-llama/Llama-2-7b" in key
        assert "ctxauto" in key
        assert "quantdefault" in key

    def test_language_cache_key_with_params(self):
        """Language cache key with all params."""
        key = ModelManager.make_language_cache_key(
            model_id="meta-llama/Llama-2-7b",
            n_ctx=4096,
            n_batch=512,
            n_gpu_layers=-1,
            preferred_quantization="Q4_K_M",
        )
        assert "ctx4096" in key
        assert "batch512" in key
        assert "gpu-1" in key
        assert "quantQ4_K_M" in key

    def test_encoder_cache_key(self):
        """Encoder cache key generation."""
        key = ModelManager.make_encoder_cache_key(
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            task="embedding",
            model_format="transformers",
        )
        assert "encoder:embedding:transformers" in key
        assert "sentence-transformers/all-MiniLM-L6-v2" in key

    def test_vision_cache_key(self):
        """Vision cache key generation."""
        key = ModelManager.make_vision_cache_key("openai/clip-vit-base-patch32")
        assert key == "vision:openai/clip-vit-base-patch32"

    def test_few_shot_cache_key(self):
        """Few-shot classifier cache key generation."""
        key = ModelManager.make_few_shot_cache_key(
            classifier_id="my-classifier",
            model_name="openai/clip-vit-base-patch32",
        )
        assert key == "few_shot:my-classifier:openai/clip-vit-base-patch32"

    def test_anomaly_cache_key(self):
        """Anomaly model cache key generation."""
        key = ModelManager.make_anomaly_cache_key(
            model_id="sensor-detector",
            backend="isolation_forest",
            normalization="standardization",
            scaler_type="robust",
        )
        assert key == "anomaly:sensor-detector:isolation_forest:standardization:robust"


class TestCacheOperations:
    """Test basic cache operations."""

    def test_contains_returns_false_for_missing(self):
        """contains() returns False for missing keys."""
        manager = ModelManager(ttl=300)
        assert not manager.contains(ModelType.VISION, "nonexistent")

    def test_set_and_get(self):
        """set() and get() work correctly."""
        manager = ModelManager(ttl=300)
        model = MockModel("test-model")

        manager.set(ModelType.VISION, "test-key", model)
        assert manager.contains(ModelType.VISION, "test-key")

        retrieved = manager.get(ModelType.VISION, "test-key")
        assert retrieved is model

    def test_get_returns_none_for_missing(self):
        """get() returns None for missing keys."""
        manager = ModelManager(ttl=300)
        assert manager.get(ModelType.VISION, "nonexistent") is None

    def test_remove(self):
        """remove() removes model from cache."""
        manager = ModelManager(ttl=300)
        model = MockModel("test-model")

        manager.set(ModelType.VISION, "test-key", model)
        removed = manager.remove(ModelType.VISION, "test-key")

        assert removed is model
        assert not manager.contains(ModelType.VISION, "test-key")

    def test_remove_nonexistent_returns_none(self):
        """remove() returns None for nonexistent keys."""
        manager = ModelManager(ttl=300)
        assert manager.remove(ModelType.VISION, "nonexistent") is None


class TestGetOrLoad:
    """Test the get_or_load method."""

    @pytest.mark.asyncio
    async def test_loads_model_when_not_cached(self):
        """get_or_load() loads model when not in cache."""
        manager = ModelManager(ttl=300)
        model = MockModel("test-model")

        async def loader():
            return model

        result = await manager.get_or_load(
            ModelType.VISION,
            "test-key",
            loader,
        )

        assert result is model
        assert manager.contains(ModelType.VISION, "test-key")

    @pytest.mark.asyncio
    async def test_returns_cached_model(self):
        """get_or_load() returns cached model without calling loader."""
        manager = ModelManager(ttl=300)
        model = MockModel("test-model")
        manager.set(ModelType.VISION, "test-key", model)

        loader_called = False

        async def loader():
            nonlocal loader_called
            loader_called = True
            return MockModel("new-model")

        result = await manager.get_or_load(
            ModelType.VISION,
            "test-key",
            loader,
        )

        assert result is model
        assert not loader_called

    @pytest.mark.asyncio
    async def test_concurrent_loads_only_load_once(self):
        """Concurrent get_or_load() calls only load once."""
        manager = ModelManager(ttl=300)
        load_count = 0

        async def loader():
            nonlocal load_count
            load_count += 1
            await asyncio.sleep(0.1)  # Simulate loading time
            return MockModel(f"model-{load_count}")

        # Start multiple concurrent loads
        results = await asyncio.gather(
            manager.get_or_load(ModelType.VISION, "test-key", loader),
            manager.get_or_load(ModelType.VISION, "test-key", loader),
            manager.get_or_load(ModelType.VISION, "test-key", loader),
        )

        # Should only have loaded once
        assert load_count == 1
        # All results should be the same model
        assert all(r is results[0] for r in results)


class TestExpiration:
    """Test TTL expiration functionality."""

    def test_pop_all_expired_with_no_expired(self):
        """pop_all_expired() returns empty list when nothing expired."""
        manager = ModelManager(ttl=300)
        model = MockModel("test-model")
        manager.set(ModelType.VISION, "test-key", model)

        expired = manager.pop_all_expired()
        assert expired == []
        assert manager.contains(ModelType.VISION, "test-key")

    def test_pop_all_expired_removes_expired_models(self):
        """pop_all_expired() removes expired models."""
        # Use very short TTL for testing
        manager = ModelManager(ttl=0.1)
        model = MockModel("test-model")
        manager.set(ModelType.VISION, "test-key", model)

        # Wait for expiration
        time.sleep(0.2)

        expired = manager.pop_all_expired()
        assert len(expired) == 1
        assert expired[0][0] == "test-key"
        assert expired[0][1] is model
        assert not manager.contains(ModelType.VISION, "test-key")

    def test_get_refreshes_ttl(self):
        """get() refreshes TTL to prevent expiration."""
        manager = ModelManager(ttl=0.2)
        model = MockModel("test-model")
        manager.set(ModelType.VISION, "test-key", model)

        # Access before expiration to refresh TTL
        time.sleep(0.1)
        manager.get(ModelType.VISION, "test-key")

        # Wait a bit more (but not past refreshed TTL)
        time.sleep(0.1)

        # Should not be expired yet
        expired = manager.pop_all_expired()
        assert expired == []
        assert manager.contains(ModelType.VISION, "test-key")


class TestUtilityMethods:
    """Test utility methods."""

    def test_get_all_models(self):
        """get_all_models() returns all models from all caches."""
        manager = ModelManager(ttl=300)
        model1 = MockModel("model1")
        model2 = MockModel("model2")

        manager.set(ModelType.VISION, "key1", model1)
        manager.set(ModelType.ANOMALY, "key2", model2)

        all_models = manager.get_all_models()
        assert len(all_models) == 2
        keys = [k for k, _ in all_models]
        assert "key1" in keys
        assert "key2" in keys

    def test_clear_all(self):
        """clear_all() clears all caches."""
        manager = ModelManager(ttl=300)
        manager.set(ModelType.VISION, "key1", MockModel("model1"))
        manager.set(ModelType.ANOMALY, "key2", MockModel("model2"))
        manager.set_encoder("enc1", {"test": "encoder"})
        manager.set_drift_detector("drift1", {"test": "detector"})

        manager.clear_all()

        assert manager.get(ModelType.VISION, "key1") is None
        assert manager.get(ModelType.ANOMALY, "key2") is None
        assert manager.get_encoder("enc1") is None
        assert manager.get_drift_detector("drift1") is None

    def test_get_loaded_model_ids(self):
        """get_loaded_model_ids() returns all cache keys."""
        manager = ModelManager(ttl=300)
        manager.set(ModelType.VISION, "vision-key", MockModel("model1"))
        manager.set(ModelType.ANOMALY, "anomaly-key", MockModel("model2"))

        ids = manager.get_loaded_model_ids()
        assert "vision-key" in ids
        assert "anomaly-key" in ids

    def test_get_stats(self):
        """get_stats() returns cache statistics."""
        manager = ModelManager(ttl=300)
        manager.set(ModelType.VISION, "key1", MockModel("model1"))
        manager.set(ModelType.VISION, "key2", MockModel("model2"))
        manager.set(ModelType.ANOMALY, "key3", MockModel("model3"))

        stats = manager.get_stats()
        assert stats["ttl_seconds"] == 300
        assert stats["total_models"] == 3
        assert stats["by_type"]["vision"] == 2
        assert stats["by_type"]["anomaly"] == 1


class TestEncoderManagement:
    """Test feature encoder management."""

    def test_set_and_get_encoder(self):
        """set_encoder() and get_encoder() work correctly."""
        manager = ModelManager(ttl=300)
        encoder = {"schema": {"field1": "numeric"}}

        manager.set_encoder("cache-key", encoder)
        retrieved = manager.get_encoder("cache-key")

        assert retrieved is encoder

    def test_remove_encoder(self):
        """remove_encoder() removes encoder."""
        manager = ModelManager(ttl=300)
        encoder = {"schema": {"field1": "numeric"}}

        manager.set_encoder("cache-key", encoder)
        removed = manager.remove_encoder("cache-key")

        assert removed is encoder
        assert manager.get_encoder("cache-key") is None


class TestDriftDetectorManagement:
    """Test drift detector management."""

    def test_set_and_get_drift_detector(self):
        """set_drift_detector() and get_drift_detector() work correctly."""
        manager = ModelManager(ttl=300)
        detector = {"algorithm": "adwin"}

        manager.set_drift_detector("detector-1", detector)
        retrieved = manager.get_drift_detector("detector-1")

        assert retrieved is detector

    def test_list_drift_detectors(self):
        """list_drift_detectors() returns all IDs."""
        manager = ModelManager(ttl=300)
        manager.set_drift_detector("detector-1", {})
        manager.set_drift_detector("detector-2", {})

        ids = manager.list_drift_detectors()
        assert "detector-1" in ids
        assert "detector-2" in ids

    def test_remove_drift_detector(self):
        """remove_drift_detector() removes detector."""
        manager = ModelManager(ttl=300)
        detector = {"algorithm": "adwin"}

        manager.set_drift_detector("detector-1", detector)
        removed = manager.remove_drift_detector("detector-1")

        assert removed is detector
        assert manager.get_drift_detector("detector-1") is None
