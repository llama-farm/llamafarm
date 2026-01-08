"""Router provider implementation.

The router provider enables semantic routing of LLM requests based on
query content. It uses embeddings to match queries to predefined routes
and forwards requests to the appropriate target model.

Flow:
1. Chat request comes in with router model name
2. RouterClient calls universal runtime's /v1/router/route endpoint
3. Router returns target model name based on semantic similarity
4. RouterClient resolves target model config and forwards request
5. Response is returned with optional routing metadata
"""

import time
from collections.abc import AsyncGenerator

import httpx
import requests
from config.datamodel import Model

from agents.base.clients.client import (
    LFAgentClient,
    LFChatCompletion,
    LFChatCompletionChunk,
    LFChatCompletionChunkWithRouting,
)
from agents.base.history import LFChatCompletionMessageParam
from agents.base.types import ToolDefinition
from core.logging import FastAPIStructLogger
from core.settings import settings

from .base import RuntimeProvider
from .health import HealthCheckResult

logger = FastAPIStructLogger(__name__)


class RouterClient(LFAgentClient):
    """Client that routes requests through the semantic router.

    This client:
    1. Extracts the query from messages
    2. Calls the universal runtime router endpoint
    3. Resolves the target model configuration
    4. Forwards the request to the target model's client
    """

    def __init__(
        self,
        *,
        model_config: Model,
        model_resolver: "ModelResolver | None" = None,
    ):
        super().__init__(model_config=model_config)
        self._model_resolver = model_resolver
        self._universal_base_url = (
            model_config.base_url
            or f"http://{settings.universal_host}:{settings.universal_port}"
        )
        self._router_name = model_config.name

    def _extract_query(self, messages: list[LFChatCompletionMessageParam]) -> str:
        """Extract the latest user message as the routing query."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                # Handle content that might be a list (multimodal)
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            return item.get("text", "")
        return ""

    async def _get_route_decision(self, query: str) -> dict:
        """Call the universal runtime to get routing decision."""
        route_url = f"{self._universal_base_url}/v1/router/route"

        # Log the query being sent for routing (truncated for readability)
        query_preview = query[:100] + "..." if len(query) > 100 else query
        logger.info(f"Router '{self._router_name}' routing query: '{query_preview}'")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                route_url,
                json={
                    "model": self._router_name,
                    "query": query,
                },
            )
            response.raise_for_status()
            result = response.json()
            logger.info(
                f"Router decision: route={result.get('route_name')}, "
                f"target={result.get('target_model')}, "
                f"score={result.get('similarity_score', 0):.3f}"
            )
            return result

    async def _ensure_router_loaded(self) -> None:
        """Ensure the router is loaded in the universal runtime.

        If the router isn't loaded, we need to train it from the config.
        """
        # First, check if router exists
        try:
            models_url = f"{self._universal_base_url}/v1/router/models"
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(models_url)
                if response.status_code == 200:
                    data = response.json()
                    router_names = [r.get("name") for r in data.get("data", [])]
                    if self._router_name in router_names:
                        # Router exists, try to load it
                        load_url = f"{self._universal_base_url}/v1/router/load"
                        await client.post(
                            load_url,
                            json={"model": self._router_name},
                        )
                        return
        except Exception as e:
            logger.debug(f"Router check failed: {e}")

        # Router doesn't exist, train it from config
        await self._train_router_from_config()

    async def _train_router_from_config(self) -> None:
        """Train the router using the model configuration."""
        if not self._model_config.routes:
            raise ValueError(
                f"Router '{self._router_name}' has no routes configured"
            )

        train_url = f"{self._universal_base_url}/v1/router/train"

        # Convert routes to training format
        routes_data = []
        for route in self._model_config.routes:
            route_data = {
                "name": route.name,
                "target_model": route.target_model,
                "utterances": route.utterances or [],
            }
            routes_data.append(route_data)

        train_request = {
            "model": self._router_name,
            "embedder_model": self._model_config.embedder_model
            or "sentence-transformers/all-MiniLM-L6-v2",
            "default_model": self._model_config.default_model or "default",
            "similarity_threshold": self._model_config.similarity_threshold or 0.7,
            "routes": routes_data,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(train_url, json=train_request)
            response.raise_for_status()
            logger.info(f"Router '{self._router_name}' trained successfully")

    def _get_target_client(self, target_model: str) -> LFAgentClient:
        """Get the client for the target model."""
        if not self._model_resolver:
            raise RuntimeError(
                "ModelResolver not provided - cannot resolve target model"
            )

        return self._model_resolver.get_client(target_model)

    async def chat(
        self,
        *,
        messages: list[LFChatCompletionMessageParam],
        tools: list[ToolDefinition] | None = None,
        extra_body: dict | None = None,
    ) -> LFChatCompletion:
        """Route and execute chat request."""
        # Ensure router is loaded
        await self._ensure_router_loaded()

        # Extract query and get routing decision
        query = self._extract_query(messages)
        if not query:
            raise ValueError("No user message found in messages")

        route_decision = await self._get_route_decision(query)
        target_model = route_decision.get("target_model")

        if not target_model:
            raise ValueError(f"Router returned no target model: {route_decision}")

        logger.info(
            f"Router '{self._router_name}' routing to '{target_model}' "
            f"(route: {route_decision.get('route_name')}, "
            f"score: {route_decision.get('similarity_score', 0):.3f})"
        )

        # Get target client and forward request
        target_client = self._get_target_client(target_model)
        response = await target_client.chat(
            messages=messages,
            tools=tools,
            extra_body=extra_body,
        )

        # Add routing metadata to response if possible
        # Note: ChatCompletion is immutable, so we can't modify it directly
        # The routing info is logged and could be added via headers in HTTP layer

        return response

    async def stream_chat(
        self,
        *,
        messages: list[LFChatCompletionMessageParam],
        tools: list[ToolDefinition] | None = None,
        extra_body: dict | None = None,
    ) -> AsyncGenerator[LFChatCompletionChunk]:
        """Route and stream chat request."""
        # Ensure router is loaded
        await self._ensure_router_loaded()

        # Extract query and get routing decision
        query = self._extract_query(messages)
        if not query:
            raise ValueError("No user message found in messages")

        route_decision = await self._get_route_decision(query)
        target_model = route_decision.get("target_model")

        if not target_model:
            raise ValueError(f"Router returned no target model: {route_decision}")

        logger.info(
            f"Router '{self._router_name}' streaming to '{target_model}' "
            f"(route: {route_decision.get('route_name')}, "
            f"score: {route_decision.get('similarity_score', 0):.3f})"
        )

        # Get target client and forward request
        target_client = self._get_target_client(target_model)
        first_chunk = True
        routing_info = {
            "router_name": self._router_name,
            "target_model": target_model,
            "route_name": route_decision.get("route_name"),
            "similarity_score": route_decision.get("similarity_score", 0.0),
            "matched_utterance": route_decision.get("matched_utterance"),
        }

        async for chunk in target_client.stream_chat(
            messages=messages,
            tools=tools,
            extra_body=extra_body,
        ):
            # Wrap first chunk with routing metadata
            if first_chunk:
                first_chunk = False
                yield LFChatCompletionChunkWithRouting(chunk, routing_info)
            else:
                yield chunk


class ModelResolver:
    """Resolves model names to their configurations and clients.

    This is a callback interface that RouterProvider uses to look up
    target model configurations from the project's llamafarm.yaml.
    """

    def __init__(self, models: list[Model]):
        """Initialize with list of model configurations."""
        self._models: dict[str, Model] = {m.name: m for m in models}
        self._all_models = models  # Keep reference for nested routing

    def get_model_config(self, model_name: str) -> Model | None:
        """Get model configuration by name."""
        return self._models.get(model_name)

    def get_client(self, model_name: str) -> LFAgentClient:
        """Get client for a model by name."""
        model_config = self._models.get(model_name)
        if not model_config:
            raise ValueError(f"Model '{model_name}' not found in configuration")

        # Import here to avoid circular imports
        from services.runtime_service import RuntimeService

        # Pass all_models for nested router resolution
        provider = RuntimeService.get_provider(model_config, all_models=self._all_models)
        return provider.get_client()


class RouterProvider(RuntimeProvider):
    """Provider for semantic router models.

    The router provider:
    1. Connects to the universal runtime's router endpoints
    2. Uses semantic embeddings to route queries to target models
    3. Forwards requests to the appropriate target model

    Configuration in llamafarm.yaml:
    ```yaml
    runtime:
      models:
        - name: my_router
          provider: router
          embedder_model: sentence-transformers/all-MiniLM-L6-v2
          default_model: general_assistant
          similarity_threshold: 0.7
          routes:
            - name: billing
              target_model: billing_specialist
              utterances:
                - "what is my bill"
                - "payment question"
            - name: support
              target_model: tech_support
              utterances:
                - "help with login"
                - "password reset"
    ```
    """

    name = "router"

    def __init__(
        self,
        *,
        model_config: Model,
        model_resolver: ModelResolver | None = None,
    ):
        super().__init__(model_config=model_config)
        self._model_resolver = model_resolver

    @property
    def _base_url(self) -> str:
        """Get base URL for Universal Runtime (hosts router endpoints)."""
        return (
            self._model_config.base_url
            or f"http://{settings.universal_host}:{settings.universal_port}"
        )

    def get_client(self) -> LFAgentClient:
        """Get router client that handles routing logic."""
        return RouterClient(
            model_config=self._model_config,
            model_resolver=self._model_resolver,
        )

    def check_health(self) -> HealthCheckResult:
        """Check health of router and universal runtime."""
        start = int(time.time() * 1000)
        base = self._base_url.rstrip("/")
        health_url = f"{base}/health"
        router_name = self._model_config.name

        try:
            # Check universal runtime health
            health_resp = requests.get(health_url, timeout=2.0)
            latency = int(time.time() * 1000) - start

            if health_resp.status_code != 200:
                return HealthCheckResult(
                    name=self.name,
                    status="unhealthy",
                    message=f"Universal runtime returned HTTP {health_resp.status_code}",
                    latency_ms=latency,
                    details={"host": base},
                )

            # Check if router is available
            models_url = f"{base}/v1/router/models"
            models_resp = requests.get(models_url, timeout=2.0)

            router_loaded = False
            if models_resp.status_code == 200:
                data = models_resp.json()
                router_names = [r.get("name") for r in data.get("data", [])]
                router_loaded = router_name in router_names

            message = (
                f"Router '{router_name}' "
                f"{'loaded' if router_loaded else 'not loaded (will train on first request)'}"
            )

            return HealthCheckResult(
                name=self.name,
                status="healthy",
                message=message,
                latency_ms=latency,
                details={
                    "host": base,
                    "router_name": router_name,
                    "router_loaded": router_loaded,
                    "routes": len(self._model_config.routes or []),
                },
            )

        except requests.exceptions.Timeout:
            return HealthCheckResult(
                name=self.name,
                status="unhealthy",
                message=f"Timeout connecting to {base}",
                latency_ms=int(time.time() * 1000) - start,
                details={"host": base},
            )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status="unhealthy",
                message=f"Error: {str(e)}",
                latency_ms=int(time.time() * 1000) - start,
                details={"host": base, "error": str(e)},
            )
