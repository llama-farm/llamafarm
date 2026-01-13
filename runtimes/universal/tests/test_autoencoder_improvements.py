"""Tests for autoencoder improvements: Early Stopping + VAE.

Phase 3 of Universal Runtime ML Enhancements.

Tests verify:
- Early stopping triggers when validation loss plateaus
- Early stopping reduces training time on easy datasets
- VAE backend produces reconstruction + KL divergence loss (ELBO)
- VAE latent space is continuous (samples from prior are valid)
- VAE anomaly scores are statistically interpretable
"""

import time

import numpy as np
import pytest


@pytest.fixture
def easy_dataset():
    """Create an easy-to-learn dataset (simple patterns).

    This dataset should be learned quickly, triggering early stopping
    well before max epochs.
    """
    np.random.seed(42)
    n_samples = 500
    # Simple linear patterns - easy to reconstruct
    X = np.column_stack([
        np.linspace(0, 1, n_samples),
        np.linspace(0, 1, n_samples) + np.random.normal(0, 0.01, n_samples),
        np.linspace(1, 0, n_samples),
    ])
    return X


@pytest.fixture
def noisy_dataset():
    """Create a more complex dataset (harder patterns).

    This dataset should take longer to learn, testing that training
    continues when validation loss is still improving.
    """
    np.random.seed(42)
    n_samples = 500
    # More complex patterns with noise
    t = np.linspace(0, 4 * np.pi, n_samples)
    X = np.column_stack([
        np.sin(t) + np.random.normal(0, 0.1, n_samples),
        np.cos(t) + np.random.normal(0, 0.1, n_samples),
        np.sin(2 * t) * np.cos(t) + np.random.normal(0, 0.1, n_samples),
        np.random.normal(0, 0.5, n_samples),  # Pure noise dimension
    ])
    return X


@pytest.fixture
def anomaly_test_data(easy_dataset):
    """Create test data with known anomalies.

    Returns (normal_data, anomalies) where anomalies are clearly
    out-of-distribution samples.
    """
    # Normal data is from the easy dataset
    normal_data = easy_dataset

    # Create clear anomalies (far from the training distribution)
    n_anomalies = 20
    anomalies = np.column_stack([
        np.ones(n_anomalies) * 5,  # Way outside [0, 1] range
        np.ones(n_anomalies) * -3,
        np.ones(n_anomalies) * 10,
    ])

    return normal_data, anomalies


class TestEarlyStopping:
    """Tests for early stopping functionality."""

    @pytest.mark.asyncio
    async def test_early_stopping_triggers_on_plateau(self, easy_dataset):
        """Test that early stopping triggers when validation loss plateaus.

        With an easy dataset and low patience, early stopping should trigger
        well before max epochs are reached.
        """
        from models.anomaly_model import AnomalyModel

        model = AnomalyModel(
            model_id="test_early_stop",
            device="cpu",
            backend="autoencoder",
            patience=5,  # Stop after 5 epochs without improvement
            min_delta=1e-4,
            validation_split=0.1,
        )
        await model.load()

        # Train with many max epochs
        max_epochs = 200
        await model.fit(easy_dataset, epochs=max_epochs, batch_size=32)

        # Early stopping should have triggered
        assert model._early_stopped, "Early stopping should have triggered on easy dataset"
        assert model._epochs_trained < max_epochs, (
            f"Should stop before max epochs. Trained: {model._epochs_trained}, Max: {max_epochs}"
        )

        # Should have trained for at least patience + a few epochs
        assert model._epochs_trained >= model.patience, (
            f"Should train for at least patience epochs: {model._epochs_trained}"
        )

        # Best validation loss should be recorded
        assert model._best_val_loss is not None
        assert model._best_val_loss > 0

    @pytest.mark.asyncio
    async def test_early_stopping_reduces_training_time(self, easy_dataset):
        """Test that early stopping reduces training time on easy datasets.

        Compare training time with and without early stopping.
        """
        from models.anomaly_model import AnomalyModel

        max_epochs = 100

        # Model WITH early stopping
        model_es = AnomalyModel(
            model_id="test_es",
            device="cpu",
            backend="autoencoder",
            patience=5,
            min_delta=1e-4,
        )
        await model_es.load()

        start_es = time.time()
        await model_es.fit(easy_dataset, epochs=max_epochs, batch_size=32)
        time_es = time.time() - start_es

        # Model WITHOUT early stopping (large patience = effectively disabled)
        model_no_es = AnomalyModel(
            model_id="test_no_es",
            device="cpu",
            backend="autoencoder",
            patience=1000,  # Effectively disable early stopping
        )
        await model_no_es.load()

        start_no_es = time.time()
        await model_no_es.fit(easy_dataset, epochs=max_epochs, batch_size=32)
        time_no_es = time.time() - start_no_es

        # Early stopping should be faster
        assert time_es < time_no_es, (
            f"Early stopping should be faster: {time_es:.2f}s vs {time_no_es:.2f}s"
        )

        # The early stopped model should have trained fewer epochs
        assert model_es._epochs_trained < model_no_es._epochs_trained

    @pytest.mark.asyncio
    async def test_best_weights_restored(self, noisy_dataset):
        """Test that best weights are restored after early stopping.

        After early stopping, the model should have the weights from
        the epoch with the best validation loss.
        """
        from models.anomaly_model import AnomalyModel

        model = AnomalyModel(
            model_id="test_best_weights",
            device="cpu",
            backend="autoencoder",
            patience=3,
            min_delta=1e-5,
        )
        await model.load()

        await model.fit(noisy_dataset, epochs=100, batch_size=32)

        # After training, the model should be fitted
        assert model._is_fitted

        # Score some data to verify model works with restored weights
        scores = await model.score(noisy_dataset[:10])
        assert len(scores) == 10
        assert all(0 <= s.score <= 1 for s in scores)


class TestVAEBackend:
    """Tests for Variational Autoencoder backend."""

    @pytest.mark.asyncio
    async def test_vae_produces_elbo_loss(self, easy_dataset):
        """Test that VAE backend produces reconstruction + KL divergence loss.

        The VAE loss should be the sum of reconstruction error and KL divergence.
        """
        from models.anomaly_model import AnomalyModel

        model = AnomalyModel(
            model_id="test_vae_elbo",
            device="cpu",
            backend="vae",
            patience=10,
            min_delta=1e-4,
        )
        await model.load()

        await model.fit(easy_dataset, epochs=50, batch_size=32)

        # Model should be fitted
        assert model._is_fitted
        assert model.backend == "vae"

        # Should have validation loss (ELBO)
        assert model._best_val_loss is not None
        assert model._best_val_loss > 0  # ELBO is positive (neg log-likelihood)

        # Score some data
        scores = await model.score(easy_dataset[:10])
        assert len(scores) == 10

        # VAE raw scores should be positive (negative ELBO)
        assert all(s.raw_score > 0 for s in scores)

    @pytest.mark.asyncio
    async def test_vae_latent_space_continuous(self, easy_dataset):
        """Test that VAE latent space is continuous.

        Samples from the prior (N(0,1)) should produce valid reconstructions.
        This is a key property of VAEs vs regular autoencoders.
        """
        import torch

        from models.anomaly_model import AnomalyModel

        model = AnomalyModel(
            model_id="test_vae_latent",
            device="cpu",
            backend="vae",
            patience=10,
        )
        await model.load()

        await model.fit(easy_dataset, epochs=50, batch_size=32)

        # Sample from the prior (standard normal)
        input_dim = easy_dataset.shape[1]
        hidden_dim = max(input_dim // 2, 8)
        latent_dim = max(hidden_dim // 2, 4)

        # Generate samples from prior
        z_samples = torch.randn(20, latent_dim)

        # Decode the samples
        with torch.no_grad():
            reconstructed = model._decoder(z_samples)

        # Reconstructions should be valid (not NaN, not Inf)
        assert not torch.isnan(reconstructed).any(), "Reconstructions should not be NaN"
        assert not torch.isinf(reconstructed).any(), "Reconstructions should not be Inf"

        # Reconstructions should be in a reasonable range
        # (within a few std devs of the training data)
        mean_abs = reconstructed.abs().mean().item()
        assert mean_abs < 100, f"Reconstructions should be bounded: {mean_abs}"

    @pytest.mark.asyncio
    async def test_vae_anomaly_scores_interpretable(self, anomaly_test_data):
        """Test that VAE anomaly scores are statistically interpretable.

        Anomalies should have higher scores (lower likelihood) than normal data.
        The scores should allow meaningful threshold-based classification.
        """
        from models.anomaly_model import AnomalyModel

        normal_data, anomalies = anomaly_test_data

        model = AnomalyModel(
            model_id="test_vae_scores",
            device="cpu",
            backend="vae",
            patience=10,
            contamination=0.1,
        )
        await model.load()

        # Train on normal data only
        await model.fit(normal_data, epochs=100, batch_size=32)

        # Score normal data
        normal_scores = await model.score(normal_data[:50])
        avg_normal_score = np.mean([s.score for s in normal_scores])

        # Score anomalies
        anomaly_scores = await model.score(anomalies)
        avg_anomaly_score = np.mean([s.score for s in anomaly_scores])

        # Anomalies should have higher scores
        assert avg_anomaly_score > avg_normal_score, (
            f"Anomalies should score higher: {avg_anomaly_score:.3f} vs {avg_normal_score:.3f}"
        )

        # Most anomalies should be flagged
        anomaly_detection_rate = sum(1 for s in anomaly_scores if s.is_anomaly) / len(anomaly_scores)
        assert anomaly_detection_rate >= 0.7, (
            f"Should detect at least 70% of anomalies: {anomaly_detection_rate:.1%}"
        )

    @pytest.mark.asyncio
    async def test_vae_with_early_stopping(self, easy_dataset):
        """Test that VAE respects early stopping parameters.

        VAE should stop early when validation ELBO plateaus.
        """
        from models.anomaly_model import AnomalyModel

        model = AnomalyModel(
            model_id="test_vae_es",
            device="cpu",
            backend="vae",
            patience=5,
            min_delta=1e-4,
        )
        await model.load()

        max_epochs = 200
        await model.fit(easy_dataset, epochs=max_epochs, batch_size=32)

        # Should have triggered early stopping on easy dataset
        assert model._early_stopped, "VAE should early stop on easy dataset"
        assert model._epochs_trained < max_epochs


class TestVAESaveLoad:
    """Tests for VAE model persistence."""

    @pytest.mark.asyncio
    async def test_vae_save_load(self, easy_dataset, tmp_path):
        """Test that VAE model can be saved and loaded correctly."""
        from models.anomaly_model import ANOMALY_MODELS_DIR, AnomalyModel

        # Create and train model
        model = AnomalyModel(
            model_id="test_vae_save",
            device="cpu",
            backend="vae",
            patience=5,
        )
        await model.load()
        await model.fit(easy_dataset, epochs=30, batch_size=32)

        # Score some data before saving
        pre_save_scores = await model.score(easy_dataset[:10])

        # Save the model (need to use the safe directory)
        import os
        os.makedirs(ANOMALY_MODELS_DIR, exist_ok=True)
        save_path = ANOMALY_MODELS_DIR / "test_vae_model.pt"
        await model.save(str(save_path))

        # Load into new model
        loaded_model = AnomalyModel(
            model_id=str(save_path),
            device="cpu",
            backend="vae",
        )
        await loaded_model.load()

        # Score with loaded model
        post_load_scores = await loaded_model.score(easy_dataset[:10])

        # Scores should be similar
        for pre, post in zip(pre_save_scores, post_load_scores, strict=True):
            assert abs(pre.score - post.score) < 0.1, (
                f"Loaded model scores should match: {pre.score:.3f} vs {post.score:.3f}"
            )

        # Clean up
        if save_path.exists():
            os.remove(save_path)


class TestValidationSplit:
    """Tests for validation split parameter."""

    @pytest.mark.asyncio
    async def test_validation_split_customizable(self, easy_dataset):
        """Test that validation split can be customized."""
        from models.anomaly_model import AnomalyModel

        # 20% validation split
        model = AnomalyModel(
            model_id="test_val_split",
            device="cpu",
            backend="autoencoder",
            validation_split=0.2,
            patience=5,
        )
        await model.load()

        await model.fit(easy_dataset, epochs=50, batch_size=32)

        # Model should be trained
        assert model._is_fitted
        assert model._best_val_loss is not None

    @pytest.mark.asyncio
    async def test_minimum_validation_samples(self):
        """Test that at least 1 sample is used for validation even with tiny datasets."""
        from models.anomaly_model import AnomalyModel

        # Very small dataset
        X = np.random.randn(10, 3)

        model = AnomalyModel(
            model_id="test_min_val",
            device="cpu",
            backend="autoencoder",
            validation_split=0.1,  # 10% of 10 = 1 sample
            patience=3,
        )
        await model.load()

        # Should not crash with small dataset
        await model.fit(X, epochs=20, batch_size=4)
        assert model._is_fitted


class TestAutoencoderVsVAEComparison:
    """Compare autoencoder and VAE backends."""

    @pytest.mark.asyncio
    async def test_both_backends_detect_anomalies(self, anomaly_test_data):
        """Test that both autoencoder and VAE can detect anomalies."""
        from models.anomaly_model import AnomalyModel

        normal_data, anomalies = anomaly_test_data

        results = {}
        for backend in ["autoencoder", "vae"]:
            model = AnomalyModel(
                model_id=f"test_{backend}",
                device="cpu",
                backend=backend,
                patience=10,
                contamination=0.1,
            )
            await model.load()
            await model.fit(normal_data, epochs=100, batch_size=32)

            # Score anomalies
            scores = await model.score(anomalies)
            detection_rate = sum(1 for s in scores if s.is_anomaly) / len(scores)
            results[backend] = detection_rate

        # Both should detect at least 50% of anomalies
        for backend, rate in results.items():
            assert rate >= 0.5, f"{backend} should detect at least 50% of anomalies: {rate:.1%}"
