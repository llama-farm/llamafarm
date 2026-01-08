"""
Integration tests for Router as Model Provider.

Tests Phase 6: Server Integration - Router as Model Provider
Tests the semantic router integration with the LlamaFarm server.
"""

import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from config.datamodel import (
    LlamaFarmConfig,
    Model,
    Provider,
    Route,
    Runtime,
    Version,
)

from agents.chat_orchestrator import ChatOrchestratorAgent
from services.runtime_service.providers.router_provider import (
    ModelResolver,
    RouterClient,
    RouterProvider,
)
from services.runtime_service.runtime_service import RuntimeService


def make_completion(content: str, model: str = "test-model"):
    """Create a mock chat completion response."""
    import time

    message = SimpleNamespace(content=content, tool_calls=[])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    return SimpleNamespace(
        id="chatcmpl-123",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[choice],
        usage=usage,
    )


def make_chunk(content: str | None, finish_reason: str | None = None):
    """Create a mock streaming chunk."""
    import time

    delta = SimpleNamespace(content=content, tool_calls=[])
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(
        id="chunk-123",
        created=int(time.time()),
        model="test-model",
        object="chat.completion.chunk",
        system_fingerprint=None,
        service_tier=None,
        usage=None,
        choices=[choice],
    )


@pytest.fixture
def router_config():
    """Create a config with a router model."""
    return LlamaFarmConfig(
        version=Version.v1,
        name="test-project",
        namespace="test",
        runtime=Runtime(
            default_model="my_router",
            models=[
                Model(
                    name="my_router",
                    provider=Provider.router,
                    embedder_model="sentence-transformers/all-MiniLM-L6-v2",
                    default_model="general_assistant",
                    similarity_threshold=0.7,
                    routes=[
                        Route(
                            name="billing",
                            target_model="billing_specialist",
                            utterances=[
                                "what is my bill",
                                "how much do I owe",
                                "payment question",
                            ],
                        ),
                        Route(
                            name="support",
                            target_model="tech_support",
                            utterances=[
                                "help with login",
                                "password reset",
                                "app not working",
                            ],
                        ),
                    ],
                ),
                Model(
                    name="billing_specialist",
                    provider=Provider.openai,
                    model="gpt-4o-mini",
                    api_key="test-key",
                ),
                Model(
                    name="tech_support",
                    provider=Provider.openai,
                    model="gpt-4o-mini",
                    api_key="test-key",
                ),
                Model(
                    name="general_assistant",
                    provider=Provider.openai,
                    model="gpt-4o-mini",
                    api_key="test-key",
                ),
            ],
        ),
    )


class TestRouterProviderBasics:
    """Test basic RouterProvider functionality."""

    def test_router_provider_instantiation(self, router_config):
        """Test that RouterProvider can be instantiated."""
        model_config = router_config.runtime.models[0]
        all_models = router_config.runtime.models

        provider = RouterProvider(
            model_config=model_config,
            model_resolver=ModelResolver(all_models),
        )

        assert provider is not None
        assert provider.name == "router"

    def test_router_provider_get_client_returns_router_client(self, router_config):
        """Test that get_client returns RouterClient."""
        model_config = router_config.runtime.models[0]
        all_models = router_config.runtime.models

        provider = RouterProvider(
            model_config=model_config,
            model_resolver=ModelResolver(all_models),
        )

        client = provider.get_client()
        assert isinstance(client, RouterClient)

    def test_runtime_service_returns_router_provider(self, router_config):
        """Test that RuntimeService correctly returns RouterProvider for router models."""
        model_config = router_config.runtime.models[0]
        all_models = router_config.runtime.models

        provider = RuntimeService.get_provider(model_config, all_models=all_models)

        assert isinstance(provider, RouterProvider)


class TestModelResolver:
    """Test ModelResolver functionality."""

    def test_model_resolver_finds_model(self, router_config):
        """Test that ModelResolver can find models by name."""
        all_models = router_config.runtime.models
        resolver = ModelResolver(all_models)

        config = resolver.get_model_config("billing_specialist")
        assert config is not None
        assert config.name == "billing_specialist"
        assert config.provider == Provider.openai

    def test_model_resolver_returns_none_for_unknown(self, router_config):
        """Test that ModelResolver returns None for unknown models."""
        all_models = router_config.runtime.models
        resolver = ModelResolver(all_models)

        config = resolver.get_model_config("nonexistent_model")
        assert config is None

    def test_model_resolver_get_client_raises_for_unknown(self, router_config):
        """Test that get_client raises ValueError for unknown models."""
        all_models = router_config.runtime.models
        resolver = ModelResolver(all_models)

        with pytest.raises(ValueError, match="not found"):
            resolver.get_client("nonexistent_model")


class TestRouterClient:
    """Test RouterClient functionality."""

    @pytest.fixture
    def router_client(self, router_config):
        """Create a RouterClient with mocked dependencies."""
        model_config = router_config.runtime.models[0]
        all_models = router_config.runtime.models
        return RouterClient(
            model_config=model_config,
            model_resolver=ModelResolver(all_models),
        )

    def test_extract_query_from_messages(self, router_client):
        """Test query extraction from message list."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is my bill?"},
        ]

        query = router_client._extract_query(messages)
        assert query == "What is my bill?"

    def test_extract_query_gets_latest_user_message(self, router_client):
        """Test that query extraction gets the latest user message."""
        messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
        ]

        query = router_client._extract_query(messages)
        assert query == "Second question"

    def test_extract_query_empty_when_no_user_message(self, router_client):
        """Test that query extraction returns empty for no user message."""
        messages = [
            {"role": "system", "content": "You are helpful."},
        ]

        query = router_client._extract_query(messages)
        assert query == ""

    @pytest.mark.asyncio
    async def test_chat_routes_to_correct_target(self, router_config):
        """Test that chat correctly routes to target model."""
        model_config = router_config.runtime.models[0]
        all_models = router_config.runtime.models

        # Create client
        client = RouterClient(
            model_config=model_config,
            model_resolver=ModelResolver(all_models),
        )

        # Mock the router endpoint and target client
        with patch.object(client, "_ensure_router_loaded", new_callable=AsyncMock):
            with patch.object(client, "_get_route_decision", new_callable=AsyncMock) as mock_route:
                mock_route.return_value = {
                    "target_model": "billing_specialist",
                    "route_name": "billing",
                    "similarity_score": 0.95,
                }

                with patch.object(client, "_get_target_client") as mock_get_client:
                    mock_target_client = MagicMock()
                    mock_target_client.chat = AsyncMock(
                        return_value=make_completion("Your bill is $50.")
                    )
                    mock_get_client.return_value = mock_target_client

                    messages = [{"role": "user", "content": "What is my bill?"}]
                    response = await client.chat(messages=messages)

                    # Verify routing was called
                    mock_route.assert_called_once_with("What is my bill?")

                    # Verify target client was called
                    mock_target_client.chat.assert_called_once()
                    assert response.choices[0].message.content == "Your bill is $50."

    @pytest.mark.asyncio
    async def test_stream_chat_routes_correctly(self, router_config):
        """Test that stream_chat correctly routes to target model."""
        model_config = router_config.runtime.models[0]
        all_models = router_config.runtime.models

        client = RouterClient(
            model_config=model_config,
            model_resolver=ModelResolver(all_models),
        )

        async def mock_stream():
            yield make_chunk("Hello")
            yield make_chunk(" there", finish_reason="stop")

        with patch.object(client, "_ensure_router_loaded", new_callable=AsyncMock):
            with patch.object(client, "_get_route_decision", new_callable=AsyncMock) as mock_route:
                mock_route.return_value = {
                    "target_model": "tech_support",
                    "route_name": "support",
                    "similarity_score": 0.88,
                }

                with patch.object(client, "_get_target_client") as mock_get_client:
                    mock_target_client = MagicMock()
                    mock_target_client.stream_chat = MagicMock(return_value=mock_stream())
                    mock_get_client.return_value = mock_target_client

                    messages = [{"role": "user", "content": "Help with login"}]
                    chunks = []
                    async for chunk in client.stream_chat(messages=messages):
                        chunks.append(chunk)

                    assert len(chunks) == 2
                    assert chunks[0].choices[0].delta.content == "Hello"
                    assert chunks[1].choices[0].delta.content == " there"

    @pytest.mark.asyncio
    async def test_chat_raises_for_empty_messages(self, router_client):
        """Test that chat raises error when no user message found."""
        messages = [{"role": "system", "content": "You are helpful."}]

        with patch.object(router_client, "_ensure_router_loaded", new_callable=AsyncMock):
            with pytest.raises(ValueError, match="No user message"):
                await router_client.chat(messages=messages)


class TestRouterIntegrationWithChatOrchestrator:
    """Test router integration with ChatOrchestratorAgent."""

    @pytest.mark.asyncio
    async def test_orchestrator_creates_with_router_model(self, router_config):
        """Test that ChatOrchestratorAgent can be created with router model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock the client to avoid actual network calls
            # Need to create a proper spec for LFAgentClient
            from agents.base.clients.client import LFAgentClient

            with patch(
                "agents.chat_orchestrator.RuntimeService.get_provider"
            ) as mock_get_provider:
                mock_provider = MagicMock()
                # Create mock client with proper spec
                mock_client = MagicMock(spec=LFAgentClient)
                mock_client.model_name = "my_router"
                mock_provider.get_client.return_value = mock_client
                mock_get_provider.return_value = mock_provider

                agent = ChatOrchestratorAgent(
                    project_config=router_config,
                    project_dir=tmpdir,
                    model_name="my_router",
                )

                assert agent is not None
                assert agent.model_name == "my_router"

                # Verify get_provider was called with all_models
                mock_get_provider.assert_called()
                call_kwargs = mock_get_provider.call_args.kwargs
                assert "all_models" in call_kwargs
                assert len(call_kwargs["all_models"]) == 4  # router + 3 targets


class TestRouterHealthCheck:
    """Test router health check functionality."""

    def test_health_check_when_runtime_healthy(self, router_config):
        """Test health check when universal runtime is healthy."""
        model_config = router_config.runtime.models[0]
        provider = RouterProvider(model_config=model_config)

        with patch("requests.get") as mock_get:
            # Mock health endpoint
            mock_health_response = MagicMock()
            mock_health_response.status_code = 200
            mock_health_response.json.return_value = {"status": "healthy"}

            # Mock models endpoint
            mock_models_response = MagicMock()
            mock_models_response.status_code = 200
            mock_models_response.json.return_value = {
                "data": [{"name": "my_router"}]
            }

            mock_get.side_effect = [mock_health_response, mock_models_response]

            result = provider.check_health()

            assert result.status == "healthy"
            assert "my_router" in result.message
            assert result.details["router_loaded"] is True

    def test_health_check_when_router_not_loaded(self, router_config):
        """Test health check when router is not yet loaded."""
        model_config = router_config.runtime.models[0]
        provider = RouterProvider(model_config=model_config)

        with patch("requests.get") as mock_get:
            mock_health_response = MagicMock()
            mock_health_response.status_code = 200

            mock_models_response = MagicMock()
            mock_models_response.status_code = 200
            mock_models_response.json.return_value = {"data": []}

            mock_get.side_effect = [mock_health_response, mock_models_response]

            result = provider.check_health()

            assert result.status == "healthy"
            assert "not loaded" in result.message
            assert result.details["router_loaded"] is False

    def test_health_check_when_runtime_unavailable(self, router_config):
        """Test health check when universal runtime is unavailable."""
        import requests

        model_config = router_config.runtime.models[0]
        provider = RouterProvider(model_config=model_config)

        with patch("requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.Timeout()

            result = provider.check_health()

            assert result.status == "unhealthy"
            assert "Timeout" in result.message


class TestRouterWithMultipleProviders:
    """Test router routing to different provider types."""

    @pytest.fixture
    def multi_provider_config(self):
        """Create config with router targeting different providers."""
        return LlamaFarmConfig(
            version=Version.v1,
            name="test-project",
            namespace="test",
            runtime=Runtime(
                default_model="multi_router",
                models=[
                    Model(
                        name="multi_router",
                        provider=Provider.router,
                        embedder_model="sentence-transformers/all-MiniLM-L6-v2",
                        default_model="openai_model",
                        similarity_threshold=0.7,
                        routes=[
                            Route(
                                name="fast",
                                target_model="ollama_model",
                                utterances=["quick question", "simple task"],
                            ),
                            Route(
                                name="complex",
                                target_model="universal_model",
                                utterances=["complex analysis", "detailed explanation"],
                            ),
                        ],
                    ),
                    Model(
                        name="ollama_model",
                        provider=Provider.ollama,
                        model="llama3.2:latest",
                        base_url="http://localhost:11434/v1",
                        api_key="ollama",
                    ),
                    Model(
                        name="universal_model",
                        provider=Provider.universal,
                        model="unsloth/Qwen3-1.7B-GGUF:Q4_K_M",
                    ),
                    Model(
                        name="openai_model",
                        provider=Provider.openai,
                        model="gpt-4o-mini",
                        api_key="test-key",
                    ),
                ],
            ),
        )

    def test_model_resolver_handles_different_providers(self, multi_provider_config):
        """Test that ModelResolver can resolve models with different providers."""
        all_models = multi_provider_config.runtime.models
        resolver = ModelResolver(all_models)

        # Check each provider type
        ollama_config = resolver.get_model_config("ollama_model")
        assert ollama_config.provider == Provider.ollama

        universal_config = resolver.get_model_config("universal_model")
        assert universal_config.provider == Provider.universal

        openai_config = resolver.get_model_config("openai_model")
        assert openai_config.provider == Provider.openai
