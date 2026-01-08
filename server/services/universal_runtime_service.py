"""
UniversalRuntimeService - HTTP client for proxying requests to Universal Runtime.

Provides a one-to-one mapping to Universal Runtime's specialized ML endpoints.
"""

import logging
from typing import Any

import httpx
from fastapi import HTTPException

from core.settings import settings

logger = logging.getLogger(__name__)


class UniversalRuntimeService:
    """Service for proxying requests to the Universal Runtime."""

    @staticmethod
    def get_base_url() -> str:
        """Get the base URL for the Universal Runtime."""
        return f"http://{settings.universal_host}:{settings.universal_port}"

    @classmethod
    async def _make_request(
        cls,
        method: str,
        path: str,
        json: dict | None = None,
        timeout: float = 300.0,
    ) -> dict[str, Any]:
        """Make an HTTP request to the Universal Runtime.

        Args:
            method: HTTP method (GET, POST, DELETE)
            path: API path (e.g., /v1/ocr)
            json: JSON body for POST requests
            timeout: Request timeout in seconds (default 5 minutes for ML operations)

        Returns:
            Response JSON as dict

        Raises:
            HTTPException: If the request fails
        """
        url = f"{cls.get_base_url()}{path}"
        logger.debug(f"Making {method} request to {url}")

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                if method == "GET":
                    response = await client.get(url)
                elif method == "POST":
                    response = await client.post(url, json=json)
                elif method == "DELETE":
                    response = await client.delete(url)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                if response.status_code >= 400:
                    # Forward error from Universal Runtime
                    try:
                        error_detail = response.json().get("detail", response.text)
                    except Exception:
                        error_detail = response.text
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=error_detail,
                    )

                return response.json()

        except httpx.ConnectError as e:
            logger.error(f"Failed to connect to Universal Runtime at {url}: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Universal Runtime not available at {cls.get_base_url()}. "
                "Start it with: nx start universal",
            ) from e
        except httpx.TimeoutException as e:
            logger.error(f"Request to Universal Runtime timed out: {e}")
            raise HTTPException(
                status_code=504,
                detail="Universal Runtime request timed out",
            ) from e
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error calling Universal Runtime: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error calling Universal Runtime: {str(e)}",
            ) from e

    # =========================================================================
    # OCR
    # =========================================================================

    @classmethod
    async def ocr(
        cls,
        model: str = "surya",
        images: list[str] | None = None,
        languages: list[str] | None = None,
        return_boxes: bool = False,
    ) -> dict[str, Any]:
        """Extract text from images using OCR.

        Args:
            model: OCR backend (surya, easyocr, paddleocr, tesseract)
            images: Base64-encoded images
            languages: Language codes
            return_boxes: Whether to return bounding boxes
        """
        payload = {
            "model": model,
            "return_boxes": return_boxes,
        }
        if images:
            payload["images"] = images
        if languages:
            payload["languages"] = languages

        return await cls._make_request("POST", "/v1/ocr", json=payload)

    # =========================================================================
    # Document Extraction
    # =========================================================================

    @classmethod
    async def extract_documents(
        cls,
        model: str,
        images: list[str] | None = None,
        prompts: list[str] | None = None,
        task: str = "extraction",
    ) -> dict[str, Any]:
        """Extract structured information from documents.

        Args:
            model: HuggingFace model ID
            images: Base64-encoded images
            prompts: Prompts for VQA task
            task: Task type (extraction, vqa, classification)
        """
        payload = {
            "model": model,
            "task": task,
        }
        if images:
            payload["images"] = images
        if prompts:
            payload["prompts"] = prompts

        return await cls._make_request("POST", "/v1/documents/extract", json=payload)

    # =========================================================================
    # SetFit Classifier
    # =========================================================================

    @classmethod
    async def classifier_fit(
        cls,
        model: str,
        training_data: list[dict],
        base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        num_iterations: int = 20,
        batch_size: int = 16,
    ) -> dict[str, Any]:
        """Train a text classifier.

        Args:
            model: Model identifier
            training_data: List of {text, label} dicts
            base_model: Base sentence transformer
            num_iterations: Training iterations
            batch_size: Training batch size
        """
        payload = {
            "model": model,
            "base_model": base_model,
            "training_data": training_data,
            "num_iterations": num_iterations,
            "batch_size": batch_size,
        }
        return await cls._make_request("POST", "/v1/classifier/fit", json=payload)

    @classmethod
    async def classifier_predict(
        cls,
        model: str,
        texts: list[str],
    ) -> dict[str, Any]:
        """Classify texts using a trained classifier.

        Args:
            model: Model identifier
            texts: Texts to classify
        """
        payload = {
            "model": model,
            "texts": texts,
        }
        return await cls._make_request("POST", "/v1/classifier/predict", json=payload)

    @classmethod
    async def classifier_save(cls, model: str) -> dict[str, Any]:
        """Save a trained classifier."""
        return await cls._make_request(
            "POST", "/v1/classifier/save", json={"model": model}
        )

    @classmethod
    async def classifier_load(cls, model: str) -> dict[str, Any]:
        """Load a pre-trained classifier."""
        return await cls._make_request(
            "POST", "/v1/classifier/load", json={"model": model}
        )

    @classmethod
    async def classifier_list_models(cls) -> dict[str, Any]:
        """List all saved classifiers."""
        return await cls._make_request("GET", "/v1/classifier/models")

    @classmethod
    async def classifier_delete_model(cls, model_name: str) -> dict[str, Any]:
        """Delete a saved classifier."""
        return await cls._make_request("DELETE", f"/v1/classifier/models/{model_name}")

    # =========================================================================
    # Anomaly Detection
    # =========================================================================

    @classmethod
    async def anomaly_fit(
        cls,
        model: str,
        data: list[list[float]] | list[dict],
        backend: str = "isolation_forest",
        schema: dict[str, str] | None = None,
        contamination: float = 0.1,
        normalization: str = "standardization",
        epochs: int = 100,
        batch_size: int = 32,
    ) -> dict[str, Any]:
        """Train an anomaly detector.

        Args:
            model: Model identifier
            data: Training data (numeric arrays or dicts)
            backend: Algorithm backend
            schema: Feature encoding schema (for dict data)
            contamination: Expected proportion of anomalies
            normalization: Score normalization method (standardization, zscore, raw)
            epochs: Training epochs (autoencoder only)
            batch_size: Batch size (autoencoder only)
        """
        payload = {
            "model": model,
            "backend": backend,
            "data": data,
            "contamination": contamination,
            "normalization": normalization,
            "epochs": epochs,
            "batch_size": batch_size,
        }
        if schema:
            payload["schema"] = schema

        return await cls._make_request("POST", "/v1/anomaly/fit", json=payload)

    @classmethod
    async def anomaly_score(
        cls,
        model: str,
        data: list[list[float]] | list[dict],
        backend: str = "isolation_forest",
        schema: dict[str, str] | None = None,
        normalization: str = "standardization",
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Score data points for anomalies.

        Args:
            model: Model identifier
            data: Data to score
            backend: Algorithm backend
            schema: Feature encoding schema (for dict data)
            normalization: Score normalization method (standardization, zscore, raw)
            threshold: Anomaly threshold
        """
        payload = {
            "model": model,
            "backend": backend,
            "data": data,
            "normalization": normalization,
        }
        if schema:
            payload["schema"] = schema
        if threshold is not None:
            payload["threshold"] = threshold

        return await cls._make_request("POST", "/v1/anomaly/score", json=payload)

    @classmethod
    async def anomaly_detect(
        cls,
        model: str,
        data: list[list[float]] | list[dict],
        backend: str = "isolation_forest",
        schema: dict[str, str] | None = None,
        normalization: str = "standardization",
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Detect anomalies (returns only anomalous points).

        Args:
            model: Model identifier
            data: Data to check
            backend: Algorithm backend
            schema: Feature encoding schema (for dict data)
            normalization: Score normalization method (standardization, zscore, raw)
            threshold: Anomaly threshold
        """
        payload = {
            "model": model,
            "backend": backend,
            "data": data,
            "normalization": normalization,
        }
        if schema:
            payload["schema"] = schema
        if threshold is not None:
            payload["threshold"] = threshold

        return await cls._make_request("POST", "/v1/anomaly/detect", json=payload)

    @classmethod
    async def anomaly_save(
        cls,
        model: str,
        backend: str = "isolation_forest",
        normalization: str = "standardization",
    ) -> dict[str, Any]:
        """Save a trained anomaly model."""
        return await cls._make_request(
            "POST",
            "/v1/anomaly/save",
            json={"model": model, "backend": backend, "normalization": normalization},
        )

    @classmethod
    async def anomaly_load(
        cls,
        model: str,
        backend: str = "isolation_forest",
    ) -> dict[str, Any]:
        """Load a pre-trained anomaly model."""
        return await cls._make_request(
            "POST",
            "/v1/anomaly/load",
            json={"model": model, "backend": backend},
        )

    @classmethod
    async def anomaly_list_models(cls) -> dict[str, Any]:
        """List all saved anomaly models."""
        return await cls._make_request("GET", "/v1/anomaly/models")

    @classmethod
    async def anomaly_delete_model(cls, filename: str) -> dict[str, Any]:
        """Delete a saved anomaly model."""
        return await cls._make_request("DELETE", f"/v1/anomaly/models/{filename}")

    # =========================================================================
    # Router
    # =========================================================================

    @classmethod
    async def router_train(
        cls,
        model: str,
        routes: list[dict],
        embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        default_model: str = "default",
        similarity_threshold: float = 0.7,
        storage_path: str | None = None,
    ) -> dict[str, Any]:
        """Train a semantic router.

        Args:
            model: Router model name
            routes: List of routes with name, target_model, and utterances
            embedder_model: HuggingFace model for embeddings
            default_model: Fallback model for unmatched queries
            similarity_threshold: Minimum similarity to route (0-1)
            storage_path: Optional custom storage path for project-specific storage
        """
        payload = {
            "model": model,
            "embedder_model": embedder_model,
            "default_model": default_model,
            "similarity_threshold": similarity_threshold,
            "routes": routes,
        }
        if storage_path:
            payload["storage_path"] = storage_path

        return await cls._make_request(
            "POST",
            "/v1/router/train",
            json=payload,
        )

    @classmethod
    async def router_route(
        cls,
        model: str,
        query: str,
        storage_path: str | None = None,
    ) -> dict[str, Any]:
        """Route a query using a trained router.

        Args:
            model: Router model name
            query: Query text to route
            storage_path: Optional storage path for project-specific router lookup
        """
        payload = {"model": model, "query": query}
        if storage_path:
            payload["storage_path"] = storage_path

        return await cls._make_request(
            "POST",
            "/v1/router/route",
            json=payload,
        )

    @classmethod
    async def router_load(
        cls,
        model: str,
        storage_path: str | None = None,
    ) -> dict[str, Any]:
        """Load a saved router into memory.

        Args:
            model: Router model name
            storage_path: Optional storage path for project-specific router
        """
        payload = {"model": model}
        if storage_path:
            payload["storage_path"] = storage_path

        return await cls._make_request(
            "POST",
            "/v1/router/load",
            json=payload,
        )

    @classmethod
    async def router_list_models(
        cls,
        storage_path: str | None = None,
    ) -> dict[str, Any]:
        """List all saved router models.

        Args:
            storage_path: Optional storage path for project-specific routers
        """
        if storage_path:
            return await cls._make_request(
                "GET",
                "/v1/router/models",
                json={"storage_path": storage_path},
            )
        return await cls._make_request("GET", "/v1/router/models")

    @classmethod
    async def router_delete_model(
        cls,
        model_name: str,
        storage_path: str | None = None,
    ) -> dict[str, Any]:
        """Delete a saved router model.

        Args:
            model_name: Router model name to delete
            storage_path: Optional storage path for project-specific router
        """
        if storage_path:
            # Use POST with body for storage_path
            return await cls._make_request(
                "POST",
                f"/v1/router/models/{model_name}/delete",
                json={"storage_path": storage_path},
            )
        return await cls._make_request("DELETE", f"/v1/router/models/{model_name}")

    @classmethod
    async def router_generate_data(
        cls,
        route_description: str | None = None,
        count: int = 20,
        complexity: str = "mixed",
        style: str | None = None,
        model: str = "unsloth/Qwen3-1.7B-GGUF:Q4_K_M",
        api_key: str | None = None,
        base_url: str | None = None,
        routes: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Generate synthetic training data for router routes.

        Args:
            route_description: Description for single route generation
            count: Number of utterances to generate
            complexity: "simple", "complex", or "mixed"
            style: Optional custom style instructions
            model: LLM model for generation
            api_key: API key for the LLM
            base_url: Custom API base URL
            routes: List of routes for batch generation
        """
        payload: dict[str, Any] = {
            "count": count,
            "model": model,
            "complexity": complexity,
        }
        if route_description:
            payload["route_description"] = route_description
        if style:
            payload["style"] = style
        if api_key:
            payload["api_key"] = api_key
        if base_url:
            payload["base_url"] = base_url
        if routes:
            payload["routes"] = routes

        return await cls._make_request(
            "POST",
            "/v1/router/generate-data",
            json=payload,
        )

    # =========================================================================
    # Health Check
    # =========================================================================

    @classmethod
    async def health_check(cls) -> dict[str, Any]:
        """Check Universal Runtime health."""
        return await cls._make_request("GET", "/health", timeout=10.0)
