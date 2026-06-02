"""
Unit tests for MiniMax runtime provider.

Tests the MiniMax provider implementation including:
- Default base URL and API key handling
- Temperature clamping
- Client creation
- Health check
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from config.datamodel import Model, Provider

from services.runtime_service.providers.minimax_provider import (
    MINIMAX_API_KEY_ENV,
    MINIMAX_BASE_URL,
    MiniMaxProvider,
)


@pytest.fixture
def model_config():
    """Create test model config for MiniMax."""
    return Model(
        name="minimax-test",
        provider=Provider.minimax,
        model="MiniMax-M2.7",
        api_key="test-minimax-key",
    )


@pytest.fixture
def model_config_no_key():
    """Create test model config without API key."""
    return Model(
        name="minimax-test",
        provider=Provider.minimax,
        model="MiniMax-M2.7",
    )


@pytest.fixture
def model_config_custom_url():
    """Create test model config with custom base URL."""
    return Model(
        name="minimax-test",
        provider=Provider.minimax,
        model="MiniMax-M2.7",
        base_url="https://custom.api.example.com/v1",
        api_key="test-key",
    )


@pytest.fixture
def provider(model_config):
    """Create MiniMax provider instance."""
    return MiniMaxProvider(model_config=model_config)


class TestMiniMaxProvider:
    """Test suite for MiniMaxProvider."""

    def test_default_base_url(self, provider):
        """Test that default base URL is set correctly."""
        assert provider._base_url == MINIMAX_BASE_URL

    def test_custom_base_url(self, model_config_custom_url):
        """Test that custom base URL is used when specified."""
        p = MiniMaxProvider(model_config=model_config_custom_url)
        assert p._base_url == "https://custom.api.example.com/v1"

    def test_api_key_from_config(self, provider):
        """Test that API key from config is used."""
        assert provider._api_key == "test-minimax-key"

    def test_api_key_from_env(self, model_config_no_key):
        """Test that API key falls back to environment variable."""
        with patch.dict(os.environ, {MINIMAX_API_KEY_ENV: "env-key"}):
            p = MiniMaxProvider(model_config=model_config_no_key)
            assert p._api_key == "env-key"

    def test_api_key_empty_when_not_set(self, model_config_no_key):
        """Test that API key is empty when not set anywhere."""
        env = os.environ.copy()
        env.pop(MINIMAX_API_KEY_ENV, None)
        with patch.dict(os.environ, env, clear=True):
            p = MiniMaxProvider(model_config=model_config_no_key)
            assert p._api_key == ""

    def test_get_client_returns_openai_client(self, provider):
        """Test that get_client returns an LFAgentClientOpenAI instance."""
        from agents.base.clients.openai import LFAgentClientOpenAI

        client = provider.get_client()
        assert isinstance(client, LFAgentClientOpenAI)

    def test_get_client_sets_base_url(self, model_config):
        """Test that the client gets the MiniMax base URL."""
        p = MiniMaxProvider(model_config=model_config)
        client = p.get_client()
        assert client._model_config.base_url == MINIMAX_BASE_URL

    def test_get_client_preserves_custom_url(self, model_config_custom_url):
        """Test that custom base URL is preserved in client."""
        p = MiniMaxProvider(model_config=model_config_custom_url)
        client = p.get_client()
        assert client._model_config.base_url == "https://custom.api.example.com/v1"

    def test_temperature_clamping_zero(self, model_config):
        """Test that temperature=0 is clamped to 0.01."""
        model_config.model_api_parameters = {"temperature": 0}
        cfg_copy = model_config.model_copy()
        MiniMaxProvider._clamp_temperature(cfg_copy)
        assert cfg_copy.model_api_parameters["temperature"] == 0.01

    def test_temperature_clamping_high(self, model_config):
        """Test that temperature>1.0 is clamped to 1.0."""
        model_config.model_api_parameters = {"temperature": 1.5}
        cfg_copy = model_config.model_copy()
        MiniMaxProvider._clamp_temperature(cfg_copy)
        assert cfg_copy.model_api_parameters["temperature"] == 1.0

    def test_temperature_valid_range(self, model_config):
        """Test that valid temperature is not modified."""
        model_config.model_api_parameters = {"temperature": 0.7}
        cfg_copy = model_config.model_copy()
        MiniMaxProvider._clamp_temperature(cfg_copy)
        assert cfg_copy.model_api_parameters["temperature"] == 0.7

    def test_temperature_clamping_no_params(self, model_config):
        """Test that clamping is a no-op when no parameters set."""
        MiniMaxProvider._clamp_temperature(model_config)  # Should not raise

    def test_temperature_clamping_does_not_mutate_original(self, model_config):
        """Test that clamping on a copy does not affect the original config."""
        model_config.model_api_parameters = {"temperature": 0}
        cfg_copy = model_config.model_copy()
        MiniMaxProvider._clamp_temperature(cfg_copy)
        assert model_config.model_api_parameters["temperature"] == 0
        assert cfg_copy.model_api_parameters["temperature"] == 0.01

    @patch("services.runtime_service.providers.minimax_provider.requests.get")
    def test_health_check_healthy(self, mock_get, provider):
        """Test health check when API is healthy."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        result = provider.check_health()
        assert result.name == "minimax"
        assert result.status == "healthy"

    @patch("services.runtime_service.providers.minimax_provider.requests.get")
    def test_health_check_unauthorized(self, mock_get, provider):
        """Test health check when API key is invalid."""
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_get.return_value = mock_resp

        result = provider.check_health()
        assert result.name == "minimax"
        assert result.status == "reachable"

    @patch("services.runtime_service.providers.minimax_provider.requests.get")
    def test_health_check_timeout(self, mock_get, provider):
        """Test health check when API times out."""
        import requests

        mock_get.side_effect = requests.exceptions.Timeout()

        result = provider.check_health()
        assert result.name == "minimax"
        assert result.status == "unhealthy"
        assert "Timeout" in result.message

    @patch("services.runtime_service.providers.minimax_provider.requests.get")
    def test_health_check_error(self, mock_get, provider):
        """Test health check when connection fails."""
        mock_get.side_effect = ConnectionError("Connection refused")

        result = provider.check_health()
        assert result.name == "minimax"
        assert result.status == "unhealthy"


class TestRuntimeServiceMiniMaxRouting:
    """Test that RuntimeService correctly routes to MiniMax provider."""

    def test_get_provider_minimax(self):
        """Test that minimax provider enum routes to MiniMaxProvider."""
        from services.runtime_service.runtime_service import RuntimeService

        config = Model(
            name="minimax-model",
            provider=Provider.minimax,
            model="MiniMax-M2.7",
            api_key="test-key",
        )
        provider = RuntimeService.get_provider(config)
        assert isinstance(provider, MiniMaxProvider)

    def test_minimax_provider_is_separate_from_openai(self):
        """Test that MiniMax provider is not confused with OpenAI provider."""
        from services.runtime_service.providers.openai_provider import (
            OpenAIProvider,
        )
        from services.runtime_service.runtime_service import RuntimeService

        minimax_config = Model(
            name="minimax-model",
            provider=Provider.minimax,
            model="MiniMax-M2.7",
            api_key="test-key",
        )
        openai_config = Model(
            name="openai-model",
            provider=Provider.openai,
            model="gpt-4",
            api_key="test-key",
        )

        minimax_provider = RuntimeService.get_provider(minimax_config)
        openai_provider = RuntimeService.get_provider(openai_config)

        assert isinstance(minimax_provider, MiniMaxProvider)
        assert isinstance(openai_provider, OpenAIProvider)
        assert type(minimax_provider) is not type(openai_provider)


class TestMiniMaxIntegration:
    """Integration-style tests for MiniMax provider (no real API calls)."""

    def test_full_client_creation_flow(self):
        """Test creating a MiniMax client through the full service path."""
        from services.runtime_service.runtime_service import RuntimeService

        config = Model(
            name="m2.7-test",
            provider=Provider.minimax,
            model="MiniMax-M2.7",
            api_key="test-api-key",
        )
        provider = RuntimeService.get_provider(config)
        client = provider.get_client()

        assert client._model_config.base_url == MINIMAX_BASE_URL
        assert client._model_config.api_key == "test-api-key"
        assert client._model_config.model == "MiniMax-M2.7"

    def test_highspeed_model(self):
        """Test MiniMax-M2.7-highspeed model configuration."""
        from services.runtime_service.runtime_service import RuntimeService

        config = Model(
            name="m2.7-hs",
            provider=Provider.minimax,
            model="MiniMax-M2.7-highspeed",
            api_key="test-api-key",
        )
        provider = RuntimeService.get_provider(config)
        client = provider.get_client()

        assert client._model_config.model == "MiniMax-M2.7-highspeed"
        assert client._model_config.base_url == MINIMAX_BASE_URL

    def test_env_key_flow(self):
        """Test MiniMax with API key from environment."""
        from services.runtime_service.runtime_service import RuntimeService

        with patch.dict(os.environ, {MINIMAX_API_KEY_ENV: "env-test-key"}):
            config = Model(
                name="minimax-env",
                provider=Provider.minimax,
                model="MiniMax-M2.7",
            )
            provider = RuntimeService.get_provider(config)
            client = provider.get_client()

            assert client._model_config.api_key == "env-test-key"
