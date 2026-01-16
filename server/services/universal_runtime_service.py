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
        data: list[list[float]] | list[dict] | None = None,
        backend: str = "isolation_forest",
        schema: dict[str, str] | None = None,
        contamination: float = 0.1,
        normalization: str = "standardization",
        scaler_type: str = "robust",
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.1,
        patience: int = 10,
        min_delta: float = 1e-4,
        training_file: str | None = None,
    ) -> dict[str, Any]:
        """Train an anomaly detector.

        Args:
            model: Model identifier
            data: Training data (numeric arrays or dicts)
            backend: Algorithm backend
            schema: Feature encoding schema (for dict data)
            contamination: Expected proportion of anomalies
            normalization: Score normalization method (standardization, zscore, raw)
            scaler_type: Input data scaler (robust or standard)
            epochs: Training epochs (autoencoder only)
            batch_size: Batch size (autoencoder only)
            validation_split: Fraction of data for validation (autoencoder/vae)
            patience: Epochs without improvement before stopping
            min_delta: Minimum change in validation loss for improvement
            training_file: File reference ID from upload-training-data endpoint
        """
        payload = {
            "model": model,
            "backend": backend,
            "contamination": contamination,
            "normalization": normalization,
            "scaler_type": scaler_type,
            "epochs": epochs,
            "batch_size": batch_size,
            "validation_split": validation_split,
            "patience": patience,
            "min_delta": min_delta,
        }
        if data is not None:
            payload["data"] = data
        if schema:
            payload["schema"] = schema
        if training_file:
            payload["training_file"] = training_file

        return await cls._make_request("POST", "/v1/anomaly/fit", json=payload)

    @classmethod
    async def anomaly_score(
        cls,
        model: str,
        data: list[list[float]] | list[dict],
        backend: str = "isolation_forest",
        schema: dict[str, str] | None = None,
        normalization: str = "standardization",
        scaler_type: str = "robust",
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Score data points for anomalies.

        Args:
            model: Model identifier
            data: Data to score
            backend: Algorithm backend
            schema: Feature encoding schema (for dict data)
            normalization: Score normalization method (standardization, zscore, raw)
            scaler_type: Input data scaler (robust or standard)
            threshold: Anomaly threshold
        """
        payload = {
            "model": model,
            "backend": backend,
            "data": data,
            "normalization": normalization,
            "scaler_type": scaler_type,
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
        scaler_type: str = "robust",
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Detect anomalies (returns only anomalous points).

        Args:
            model: Model identifier
            data: Data to check
            backend: Algorithm backend
            schema: Feature encoding schema (for dict data)
            normalization: Score normalization method (standardization, zscore, raw)
            scaler_type: Input data scaler (robust or standard)
            threshold: Anomaly threshold
        """
        payload = {
            "model": model,
            "backend": backend,
            "data": data,
            "normalization": normalization,
            "scaler_type": scaler_type,
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
    # Speech-to-Text
    # =========================================================================

    @classmethod
    async def transcribe_audio(
        cls,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        model: str = "distil-large-v3",
        language: str | None = None,
        prompt: str | None = None,
        response_format: str = "json",
        temperature: float = 0.0,
        timestamp_granularities: str | None = None,
        timeout: float = 600.0,
    ) -> dict[str, Any]:
        """Transcribe audio to text.

        Args:
            audio_bytes: Raw audio file bytes
            filename: Original filename (for format detection)
            model: Whisper model size (tiny, base, small, medium, large-v3, distil-large-v3)
            language: ISO language code (auto-detected if None)
            prompt: Optional conditioning text
            response_format: Output format (json, text, srt, vtt, verbose_json)
            temperature: Sampling temperature
            timestamp_granularities: Comma-separated list (word, segment)
            timeout: Request timeout (default 10 minutes for long audio)

        Returns:
            Transcription result
        """
        url = f"{cls.get_base_url()}/v1/audio/transcriptions"
        logger.debug(f"Transcribing audio via {url}")

        try:
            # Build form data
            files = {"file": (filename, audio_bytes)}
            data = {
                "model": model,
                "response_format": response_format,
                "temperature": str(temperature),
            }
            if language:
                data["language"] = language
            if prompt:
                data["prompt"] = prompt
            if timestamp_granularities:
                data["timestamp_granularities"] = timestamp_granularities

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, files=files, data=data)

                if response.status_code >= 400:
                    try:
                        error_detail = response.json().get("detail", response.text)
                    except Exception:
                        error_detail = response.text
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=error_detail,
                    )

                # Handle text response formats
                if response_format in ("text", "srt", "vtt"):
                    return {"text": response.text}

                return response.json()

        except httpx.ConnectError as e:
            logger.error(f"Failed to connect to Universal Runtime at {url}: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Universal Runtime not available at {cls.get_base_url()}. "
                "Start it with: nx start universal",
            ) from e
        except httpx.TimeoutException as e:
            logger.error(f"Transcription request timed out: {e}")
            raise HTTPException(
                status_code=504,
                detail="Transcription request timed out. Audio may be too long.",
            ) from e
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error transcribing audio: {str(e)}",
            ) from e

    @classmethod
    async def translate_audio(
        cls,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        model: str = "distil-large-v3",
        prompt: str | None = None,
        response_format: str = "json",
        temperature: float = 0.0,
        timeout: float = 600.0,
    ) -> dict[str, Any]:
        """Translate audio to English text.

        Args:
            audio_bytes: Raw audio file bytes
            filename: Original filename
            model: Whisper model size
            prompt: Optional conditioning text
            response_format: Output format (json, text)
            temperature: Sampling temperature
            timeout: Request timeout

        Returns:
            Translation result
        """
        url = f"{cls.get_base_url()}/v1/audio/translations"
        logger.debug(f"Translating audio via {url}")

        try:
            files = {"file": (filename, audio_bytes)}
            data = {
                "model": model,
                "response_format": response_format,
                "temperature": str(temperature),
            }
            if prompt:
                data["prompt"] = prompt

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, files=files, data=data)

                if response.status_code >= 400:
                    try:
                        error_detail = response.json().get("detail", response.text)
                    except Exception:
                        error_detail = response.text
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=error_detail,
                    )

                if response_format == "text":
                    return {"text": response.text}

                return response.json()

        except httpx.ConnectError as e:
            logger.error(f"Failed to connect to Universal Runtime at {url}: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Universal Runtime not available at {cls.get_base_url()}. "
                "Start it with: nx start universal",
            ) from e
        except httpx.TimeoutException as e:
            logger.error(f"Translation request timed out: {e}")
            raise HTTPException(
                status_code=504,
                detail="Translation request timed out. Audio may be too long.",
            ) from e
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error translating audio: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error translating audio: {str(e)}",
            ) from e

    # =========================================================================
    # Health Check
    # =========================================================================

    @classmethod
    async def health_check(cls) -> dict[str, Any]:
        """Check Universal Runtime health."""
        return await cls._make_request("GET", "/health", timeout=10.0)

    # =========================================================================
    # Embeddings
    # =========================================================================

    @classmethod
    async def embeddings(
        cls,
        model: str,
        input: str | list[str],
        encoding_format: str = "float",
    ) -> dict[str, Any]:
        """Generate embeddings for text.

        Args:
            model: Encoder model name (e.g., sentence-transformers/all-MiniLM-L6-v2)
            input: Text or list of texts to embed
            encoding_format: Output format (float or base64)
        """
        return await cls._make_request(
            "POST",
            "/v1/embeddings",
            json={
                "model": model,
                "input": input,
                "encoding_format": encoding_format,
            },
        )

    # =========================================================================
    # NLP: Language Detection
    # =========================================================================

    @classmethod
    async def detect_language(
        cls,
        text: str,
        model: str = "papluca/xlm-roberta-base-language-detection",
    ) -> dict[str, Any]:
        """Detect the language of text.

        Args:
            text: Text to analyze
            model: Language detection model
        """
        return await cls._make_request(
            "POST",
            "/v1/text/language",
            json={"text": text, "model": model},
        )

    @classmethod
    async def detect_language_batch(
        cls,
        texts: list[str],
        model: str = "papluca/xlm-roberta-base-language-detection",
    ) -> dict[str, Any]:
        """Detect languages for multiple texts.

        Args:
            texts: List of texts to analyze
            model: Language detection model
        """
        return await cls._make_request(
            "POST",
            "/v1/text/language/batch",
            json={"texts": texts, "model": model},
        )

    # =========================================================================
    # NLP: Keyword Extraction
    # =========================================================================

    @classmethod
    async def extract_keywords(
        cls,
        text: str,
        top_k: int = 10,
        diversity: float = 0.5,
        ngram_range: list[int] | None = None,
    ) -> dict[str, Any]:
        """Extract keywords from text.

        Args:
            text: Text to extract keywords from
            top_k: Number of keywords to return
            diversity: Diversity parameter (0-1)
            ngram_range: Min and max n-gram size [min, max]
        """
        payload = {
            "text": text,
            "top_k": top_k,
            "diversity": diversity,
        }
        if ngram_range:
            payload["ngram_range"] = ngram_range

        return await cls._make_request("POST", "/v1/text/keywords", json=payload)

    @classmethod
    async def extract_keywords_batch(
        cls,
        texts: list[str],
        top_k: int = 10,
        diversity: float = 0.5,
        ngram_range: list[int] | None = None,
    ) -> dict[str, Any]:
        """Extract keywords from multiple texts.

        Args:
            texts: List of texts
            top_k: Number of keywords per text
            diversity: Diversity parameter (0-1)
            ngram_range: Min and max n-gram size [min, max]
        """
        payload = {
            "texts": texts,
            "top_k": top_k,
            "diversity": diversity,
        }
        if ngram_range:
            payload["ngram_range"] = ngram_range

        return await cls._make_request("POST", "/v1/text/keywords/batch", json=payload)

    # =========================================================================
    # NLP: PII Detection and Redaction
    # =========================================================================

    @classmethod
    async def detect_pii(
        cls,
        text: str,
        entity_types: list[str] | None = None,
        threshold: float = 0.5,
        use_regex: bool = True,
    ) -> dict[str, Any]:
        """Detect PII in text.

        Args:
            text: Text to analyze
            entity_types: Custom entity types (default: standard PII)
            threshold: Detection confidence threshold
            use_regex: Also use regex patterns
        """
        payload = {
            "text": text,
            "threshold": threshold,
            "use_regex": use_regex,
        }
        if entity_types:
            payload["entity_types"] = entity_types

        return await cls._make_request("POST", "/v1/text/pii-detect", json=payload)

    @classmethod
    async def redact_pii(
        cls,
        text: str,
        entity_types: list[str] | None = None,
        replacement: str = "[REDACTED]",
        replacement_map: dict[str, str] | None = None,
        threshold: float = 0.5,
        use_regex: bool = True,
    ) -> dict[str, Any]:
        """Redact PII from text.

        Args:
            text: Text to redact
            entity_types: Entity types to redact
            replacement: Default replacement text
            replacement_map: Per-type replacements
            threshold: Detection confidence threshold
            use_regex: Also use regex patterns
        """
        payload = {
            "text": text,
            "replacement": replacement,
            "threshold": threshold,
            "use_regex": use_regex,
        }
        if entity_types:
            payload["entity_types"] = entity_types
        if replacement_map:
            payload["replacement_map"] = replacement_map

        return await cls._make_request("POST", "/v1/text/pii-redact", json=payload)

    # =========================================================================
    # Time Series
    # =========================================================================

    @classmethod
    async def forecast_timeseries(
        cls,
        data: list[float],
        horizon: int = 10,
        model: str = "amazon/chronos-t5-small",
    ) -> dict[str, Any]:
        """Forecast future values in a time series.

        Args:
            data: Historical time series data
            horizon: Number of future points to forecast
            model: Chronos model name
        """
        return await cls._make_request(
            "POST",
            "/v1/timeseries/forecast",
            json={"data": data, "horizon": horizon, "model": model},
        )

    @classmethod
    async def detect_changepoints(
        cls,
        data: list[float],
        algorithm: str = "binseg",
        n_changepoints: int | None = None,
        penalty: float | None = None,
    ) -> dict[str, Any]:
        """Detect change points in time series data.

        Args:
            data: Time series data
            algorithm: Detection algorithm (binseg, pelt, window, dynp)
            n_changepoints: Expected number of change points
            penalty: Penalty value for PELT algorithm
        """
        payload = {"data": data, "algorithm": algorithm}
        if n_changepoints is not None:
            payload["n_changepoints"] = n_changepoints
        if penalty is not None:
            payload["penalty"] = penalty

        return await cls._make_request(
            "POST", "/v1/timeseries/changepoints", json=payload
        )

    # =========================================================================
    # Vision: Object Detection
    # =========================================================================

    @classmethod
    async def detect_objects(
        cls,
        image: str,
        threshold: float = 0.5,
        labels: list[str] | None = None,
        model: str = "hustvl/yolos-tiny",
    ) -> dict[str, Any]:
        """Detect objects in an image.

        Args:
            image: Base64-encoded image or file path
            threshold: Confidence threshold (0-1)
            labels: Filter to specific object labels
            model: Object detection model
        """
        payload = {
            "image": image,
            "threshold": threshold,
            "model": model,
        }
        if labels:
            payload["labels"] = labels

        return await cls._make_request("POST", "/v1/vision/detect-objects", json=payload)

    @classmethod
    async def detect_objects_batch(
        cls,
        images: list[str],
        threshold: float = 0.5,
        labels: list[str] | None = None,
        model: str = "hustvl/yolos-tiny",
    ) -> dict[str, Any]:
        """Detect objects in multiple images.

        Args:
            images: List of base64-encoded images
            threshold: Confidence threshold (0-1)
            labels: Filter to specific object labels
            model: Object detection model
        """
        payload = {
            "images": images,
            "threshold": threshold,
            "model": model,
        }
        if labels:
            payload["labels"] = labels

        return await cls._make_request(
            "POST", "/v1/vision/detect-objects/batch", json=payload
        )

    # =========================================================================
    # Vision: Background Removal
    # =========================================================================

    @classmethod
    async def remove_background(
        cls,
        image: str,
        model: str = "briaai/RMBG-1.4",
    ) -> dict[str, Any]:
        """Remove background from an image.

        Args:
            image: Base64-encoded image
            model: Background removal model
        """
        return await cls._make_request(
            "POST",
            "/v1/vision/remove-background",
            json={"image": image, "model": model},
        )

    # =========================================================================
    # Analysis: Table QA
    # =========================================================================

    @classmethod
    async def table_qa(
        cls,
        table: list[dict[str, Any]],
        query: str,
        model: str = "google/tapas-base-finetuned-wtq",
    ) -> dict[str, Any]:
        """Answer questions about tabular data.

        Args:
            table: Table as list of row dicts
            query: Question to answer
            model: Table QA model
        """
        return await cls._make_request(
            "POST",
            "/v1/analysis/table-qa",
            json={"table": table, "query": query, "model": model},
        )

    # =========================================================================
    # Analysis: Dataset Audit
    # =========================================================================

    @classmethod
    async def audit_dataset(
        cls,
        labels: list[int],
        pred_probs: list[list[float]],
        label_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Audit dataset for label quality issues.

        Args:
            labels: Ground truth labels
            pred_probs: Predicted probabilities per class
            label_names: Optional class names
        """
        payload = {
            "labels": labels,
            "pred_probs": pred_probs,
        }
        if label_names:
            payload["label_names"] = label_names

        return await cls._make_request("POST", "/v1/dataset/audit", json=payload)

    # =========================================================================
    # Analysis: Drift Detection
    # =========================================================================

    @classmethod
    async def detect_drift(
        cls,
        reference_data: list[float],
        current_data: list[float],
        algorithm: str = "adwin",
    ) -> dict[str, Any]:
        """Detect drift between reference and current data.

        Args:
            reference_data: Reference/baseline data
            current_data: Current data to check for drift
            algorithm: Drift detection algorithm
        """
        return await cls._make_request(
            "POST",
            "/v1/streaming/drift/detect",
            json={
                "reference_data": reference_data,
                "current_data": current_data,
                "algorithm": algorithm,
            },
        )

    # =========================================================================
    # Anomaly Explanation
    # =========================================================================

    @classmethod
    async def explain_anomaly(
        cls,
        model_id: str,
        data: list[list[float]],
        feature_names: list[str] | None = None,
        backend: str = "isolation_forest",
        background_samples: int = 100,
        nsamples: int = 100,
    ) -> dict[str, Any]:
        """Explain why data points are flagged as anomalies using SHAP.

        Args:
            model_id: Anomaly model identifier
            data: Data points to explain
            feature_names: Names for features
            backend: Anomaly detection backend
            background_samples: Number of background samples for SHAP
            nsamples: Number of samples for SHAP estimation
        """
        payload = {
            "model_id": model_id,
            "data": data,
            "backend": backend,
            "background_samples": background_samples,
            "nsamples": nsamples,
        }
        if feature_names:
            payload["feature_names"] = feature_names

        return await cls._make_request("POST", "/v1/anomaly/explain", json=payload)

    # =========================================================================
    # Vision - Open Vocabulary Detection
    # =========================================================================

    @classmethod
    async def detect_open_vocab(
        cls,
        image: str,
        labels: list[str],
        threshold: float = 0.1,
        model: str = "google/owlvit-base-patch32",
    ) -> dict[str, Any]:
        """Detect objects by text description (open-vocabulary)."""
        return await cls._make_request(
            "POST",
            "/v1/vision/detect-open",
            {
                "image": image,
                "labels": labels,
                "threshold": threshold,
                "model": model,
            },
        )

    @classmethod
    async def detect_open_vocab_batch(
        cls,
        images: list[str],
        labels: list[str],
        threshold: float = 0.1,
        model: str = "google/owlvit-base-patch32",
    ) -> dict[str, Any]:
        """Detect objects in multiple images by text description."""
        return await cls._make_request(
            "POST",
            "/v1/vision/detect-open/batch",
            {
                "images": images,
                "labels": labels,
                "threshold": threshold,
                "model": model,
            },
        )

    @classmethod
    async def detect_open_vocab_by_image(
        cls,
        query_image: str,
        reference_images: list[str],
        threshold: float = 0.1,
        model: str = "google/owlvit-base-patch32",
    ) -> dict[str, Any]:
        """Detect objects using reference images (one-shot detection)."""
        return await cls._make_request(
            "POST",
            "/v1/vision/detect-open/by-image",
            {
                "query_image": query_image,
                "reference_images": reference_images,
                "threshold": threshold,
                "model": model,
            },
        )

    # =========================================================================
    # Vision - Zero-Shot Classification
    # =========================================================================

    @classmethod
    async def classify_zero_shot(
        cls,
        image: str,
        labels: list[str],
        model: str = "openai/clip-vit-base-patch32",
    ) -> dict[str, Any]:
        """Classify an image without training (zero-shot)."""
        return await cls._make_request(
            "POST",
            "/v1/vision/classify-zero-shot",
            {
                "image": image,
                "labels": labels,
                "model": model,
            },
        )

    @classmethod
    async def classify_zero_shot_batch(
        cls,
        images: list[str],
        labels: list[str],
        model: str = "openai/clip-vit-base-patch32",
    ) -> dict[str, Any]:
        """Classify multiple images without training (zero-shot)."""
        return await cls._make_request(
            "POST",
            "/v1/vision/classify-zero-shot/batch",
            {
                "images": images,
                "labels": labels,
                "model": model,
            },
        )

    # =========================================================================
    # Vision - Few-Shot Classification
    # =========================================================================

    @classmethod
    async def fit_few_shot_classifier(
        cls,
        classifier_id: str,
        training_data: list[dict[str, str]],
        model: str = "openai/clip-vit-base-patch32",
        num_iterations: int = 20,
    ) -> dict[str, Any]:
        """Train a few-shot image classifier."""
        return await cls._make_request(
            "POST",
            "/v1/vision/classify/fit",
            {
                "classifier_id": classifier_id,
                "training_data": training_data,
                "model": model,
                "num_iterations": num_iterations,
            },
        )

    @classmethod
    async def predict_few_shot(
        cls,
        classifier_id: str,
        image: str,
    ) -> dict[str, Any]:
        """Predict using a fitted few-shot classifier."""
        return await cls._make_request(
            "POST",
            "/v1/vision/classify/predict",
            {
                "classifier_id": classifier_id,
                "image": image,
            },
        )

    @classmethod
    async def predict_few_shot_batch(
        cls,
        classifier_id: str,
        images: list[str],
    ) -> dict[str, Any]:
        """Predict on multiple images using a fitted few-shot classifier."""
        return await cls._make_request(
            "POST",
            "/v1/vision/classify/predict/batch",
            {
                "classifier_id": classifier_id,
                "images": images,
            },
        )

    @classmethod
    async def refine_few_shot_classifier(
        cls,
        classifier_id: str,
        training_data: list[dict[str, str]],
        num_iterations: int = 10,
    ) -> dict[str, Any]:
        """Refine a few-shot classifier with additional data."""
        return await cls._make_request(
            "POST",
            "/v1/vision/classify/refine",
            {
                "classifier_id": classifier_id,
                "training_data": training_data,
                "num_iterations": num_iterations,
            },
        )

    @classmethod
    async def get_few_shot_classifier_info(cls, classifier_id: str) -> dict[str, Any]:
        """Get information about a fitted few-shot classifier."""
        return await cls._make_request(
            "GET",
            f"/v1/vision/classify/info/{classifier_id}",
        )

    # =========================================================================
    # Vision - Segmentation
    # =========================================================================

    @classmethod
    async def segment_image(
        cls,
        image: str,
        model: str = "facebook/sam-vit-base",
        points: list[list[float]] | None = None,
        boxes: list[list[float]] | None = None,
    ) -> dict[str, Any]:
        """Segment an image using SAM."""
        return await cls._make_request(
            "POST",
            "/v1/vision/segment",
            {
                "image": image,
                "model": model,
                "points": points,
                "boxes": boxes,
            },
        )

    @classmethod
    async def segment_batch(
        cls,
        images: list[str],
        model: str = "facebook/sam-vit-base",
        points: list[list[list[float]]] | None = None,
        boxes: list[list[list[float]]] | None = None,
    ) -> dict[str, Any]:
        """Segment multiple images using SAM."""
        return await cls._make_request(
            "POST",
            "/v1/vision/segment/batch",
            {
                "images": images,
                "model": model,
                "points": points,
                "boxes": boxes,
            },
        )

    # =========================================================================
    # Batch Endpoints
    # =========================================================================

    @classmethod
    async def forecast_timeseries_batch(
        cls,
        data: list[list[float]],
        horizon: int = 12,
        model: str = "amazon/chronos-t5-tiny",
    ) -> dict[str, Any]:
        """Forecast multiple time series in batch."""
        return await cls._make_request(
            "POST",
            "/v1/timeseries/forecast/batch",
            {
                "data": data,
                "horizon": horizon,
                "model": model,
            },
        )

    @classmethod
    async def detect_changepoints_batch(
        cls,
        data: list[list[float]],
        algorithm: str = "binseg",
        n_changepoints: int | None = None,
        penalty: float | None = None,
    ) -> dict[str, Any]:
        """Detect change points in multiple time series."""
        return await cls._make_request(
            "POST",
            "/v1/timeseries/changepoints/batch",
            {
                "data": data,
                "algorithm": algorithm,
                "n_changepoints": n_changepoints,
                "penalty": penalty,
            },
        )

    @classmethod
    async def table_qa_batch(
        cls,
        tables: list[list[dict[str, Any]]],
        queries: list[str],
        model: str = "google/tapas-base-finetuned-wtq",
    ) -> dict[str, Any]:
        """Answer questions about multiple tables."""
        return await cls._make_request(
            "POST",
            "/v1/analysis/table-qa/batch",
            {
                "tables": tables,
                "queries": queries,
                "model": model,
            },
        )
