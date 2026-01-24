"""
UniversalRuntimeService - HTTP client for proxying requests to Universal Runtime.

Provides a one-to-one mapping to Universal Runtime's specialized ML endpoints.
Uses a persistent connection pool to minimize TCP/TLS handshake overhead.
"""

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx
import websockets
from fastapi import HTTPException

from core.settings import settings

logger = logging.getLogger(__name__)


# Global persistent HTTP client for connection pooling
_http_client: httpx.AsyncClient | None = None
_client_lock = asyncio.Lock()


async def get_runtime_client() -> httpx.AsyncClient:
    """Get or create the persistent HTTP client for runtime communication.

    Uses connection pooling with HTTP/1.1 keep-alive to avoid
    TCP handshake overhead on each request (~50-100ms savings per request).

    Returns:
        Shared httpx.AsyncClient instance with connection pooling enabled.
    """
    global _http_client

    if _http_client is not None and not _http_client.is_closed:
        return _http_client

    async with _client_lock:
        # Double-check after acquiring lock
        if _http_client is not None and not _http_client.is_closed:
            return _http_client

        base_url = f"http://{settings.universal_host}:{settings.universal_port}"
        logger.info(f"Creating persistent HTTP client for Universal Runtime at {base_url}")

        # Configure connection pool limits
        # max_connections: Total connections across all hosts
        # max_keepalive_connections: Connections to keep alive in pool
        # keepalive_expiry: How long to keep idle connections (seconds)
        limits = httpx.Limits(
            max_connections=20,
            max_keepalive_connections=10,
            keepalive_expiry=30.0,
        )

        _http_client = httpx.AsyncClient(
            base_url=base_url,
            limits=limits,
            timeout=httpx.Timeout(
                connect=10.0,  # Connection timeout
                read=300.0,    # Read timeout (5 min for ML ops)
                write=60.0,    # Write timeout
                pool=10.0,     # Pool checkout timeout
            ),
            # Enable HTTP/2 for multiplexing (requires httpx[http2])
            http2=True,
        )

        logger.info("Persistent HTTP client created with connection pooling")
        return _http_client


async def close_runtime_client() -> None:
    """Close the persistent HTTP client.

    Should be called during application shutdown to cleanly close connections.
    """
    global _http_client

    async with _client_lock:
        if _http_client is not None:
            logger.info("Closing persistent HTTP client")
            await _http_client.aclose()
            _http_client = None
            logger.info("Persistent HTTP client closed")


class UniversalRuntimeService:
    """Service for proxying requests to the Universal Runtime.

    Uses a persistent connection pool for efficient communication.
    Connection pooling reduces per-request latency by ~50-100ms by
    reusing TCP connections instead of establishing new ones.
    """

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

        Uses persistent connection pool for efficient communication.

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
        logger.debug(f"Making {method} request to {path}")

        try:
            client = await get_runtime_client()

            # Create request-specific timeout if different from default
            request_timeout = httpx.Timeout(
                connect=10.0,
                read=timeout,
                write=60.0,
                pool=10.0,
            )

            if method == "GET":
                response = await client.get(path, timeout=request_timeout)
            elif method == "POST":
                response = await client.post(path, json=json, timeout=request_timeout)
            elif method == "DELETE":
                response = await client.delete(path, timeout=request_timeout)
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
            logger.error(f"Failed to connect to Universal Runtime at {path}: {e}")
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
    # NLP (Embeddings, Rerank, Classify, NER)
    # =========================================================================

    @classmethod
    async def embeddings(
        cls,
        input: str | list[str],
        model: str,
        encoding_format: str = "float",
        dimensions: int | None = None,
    ) -> dict[str, Any]:
        """Generate embeddings for text input.

        Args:
            input: Text or list of texts to embed
            model: HuggingFace model ID or GGUF path
            encoding_format: Output format (float, base64)
            dimensions: Truncate embeddings to this size
        """
        payload: dict[str, Any] = {
            "input": input,
            "model": model,
            "encoding_format": encoding_format,
        }
        if dimensions is not None:
            payload["dimensions"] = dimensions

        return await cls._make_request("POST", "/v1/embeddings", json=payload)

    @classmethod
    async def rerank(
        cls,
        query: str,
        documents: list[str],
        model: str,
        top_n: int | None = None,
        return_documents: bool = True,
    ) -> dict[str, Any]:
        """Rerank documents by relevance to a query.

        Args:
            query: The query to rank against
            documents: List of documents to rerank
            model: Reranker model ID
            top_n: Return only top N results
            return_documents: Include document text in response
        """
        payload: dict[str, Any] = {
            "query": query,
            "documents": documents,
            "model": model,
            "return_documents": return_documents,
        }
        if top_n is not None:
            payload["top_n"] = top_n

        return await cls._make_request("POST", "/v1/rerank", json=payload)

    @classmethod
    async def classify(
        cls,
        input: str | list[str],
        model: str,
        labels: list[str] | None = None,
    ) -> dict[str, Any]:
        """Classify text using zero-shot or trained classifier.

        Args:
            input: Text or list of texts to classify
            model: Classification model ID
            labels: Labels for zero-shot classification
        """
        payload: dict[str, Any] = {
            "input": input,
            "model": model,
        }
        if labels:
            payload["labels"] = labels

        return await cls._make_request("POST", "/v1/classify", json=payload)

    @classmethod
    async def ner(
        cls,
        input: str | list[str],
        model: str,
    ) -> dict[str, Any]:
        """Extract named entities from text.

        Args:
            input: Text or list of texts
            model: NER model ID
        """
        payload = {
            "input": input,
            "model": model,
        }
        return await cls._make_request("POST", "/v1/ner", json=payload)

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
    # Speech-to-Text
    # =========================================================================

    @classmethod
    async def transcribe_audio(
        cls,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        model: str = "distil-large-v3-turbo",
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
        # === TIMING INSTRUMENTATION ===
        t_start = time.perf_counter()

        audio_kb = len(audio_bytes) / 1024
        logger.debug(f"Transcribing audio ({audio_kb:.1f}KB)")

        try:
            client = await get_runtime_client()

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

            request_timeout = httpx.Timeout(
                connect=10.0,
                read=timeout,
                write=120.0,  # Longer write timeout for large audio files
                pool=10.0,
            )

            response = await client.post(
                "/v1/audio/transcriptions",
                files=files,
                data=data,
                timeout=request_timeout,
            )

            t_response = time.perf_counter()
            logger.info(f"⏱️ STT HTTP: Response in {(t_response - t_start)*1000:.1f}ms for {audio_kb:.1f}KB")

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
            logger.error(f"Failed to connect to Universal Runtime: {e}")
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
    async def transcribe_audio_stream(
        cls,
        audio_bytes: bytes,
        model: str = "distil-large-v3-turbo",
        language: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream transcription segments as they're processed.

        Uses WebSocket to get segments as they're transcribed, enabling
        parallel processing (e.g., start LLM on first segment).

        Args:
            audio_bytes: Raw PCM audio (16kHz, 16-bit, mono)
            model: Whisper model
            language: ISO language code (auto-detected if None)

        Yields:
            Transcription segments with text, timestamps, etc.
        """
        # === TIMING INSTRUMENTATION ===
        t_start = time.perf_counter()
        t_connected = None
        t_audio_sent = None
        t_first_segment = None
        first_segment_logged = False

        ws_url = f"ws://{settings.universal_host}:{settings.universal_port}/v1/audio/transcriptions/stream"
        params = [
            f"model={model}",
            "chunk_interval=0.5",  # Faster chunking for lower latency
        ]
        if language:
            params.append(f"language={language}")
        ws_url += "?" + "&".join(params)

        try:
            async with websockets.connect(ws_url) as ws:
                t_connected = time.perf_counter()
                logger.info(f"⏱️ STT WS: Connected in {(t_connected - t_start)*1000:.1f}ms")

                # Send all audio at once (it's already collected by VAD)
                await ws.send(audio_bytes)
                # Signal end of audio
                await ws.send("END")
                t_audio_sent = time.perf_counter()
                audio_kb = len(audio_bytes) / 1024
                logger.info(f"⏱️ STT WS: Audio sent ({audio_kb:.1f}KB) in {(t_audio_sent - t_connected)*1000:.1f}ms")

                # Receive segments as they're transcribed
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
                        data = json.loads(msg)

                        msg_type = data.get("type", "")
                        if msg_type == "segment":
                            if not first_segment_logged:
                                t_first_segment = time.perf_counter()
                                first_segment_logged = True
                                logger.info(f"⏱️ STT WS: First segment in {(t_first_segment - t_start)*1000:.1f}ms total, {(t_first_segment - t_audio_sent)*1000:.1f}ms processing")
                            yield data
                        elif msg_type == "done":
                            t_done = time.perf_counter()
                            logger.info(f"⏱️ STT WS: Complete in {(t_done - t_start)*1000:.1f}ms total")
                            break
                        elif msg_type == "error":
                            logger.error(f"STT stream error: {data.get('message')}")
                            break
                        elif msg_type == "warning":
                            logger.warning(f"STT stream warning: {data.get('message')}")
                        # Ignore other message types

                    except TimeoutError:
                        logger.warning("STT stream timeout")
                        break

        except websockets.exceptions.ConnectionClosed:
            # This can happen when consumer breaks early from the generator (e.g., to start LLM early)
            logger.debug("STT WebSocket closed (may be expected during early exit)")
            raise
        except Exception as e:
            logger.error(f"STT streaming error: {e}")
            raise

    @classmethod
    async def translate_audio(
        cls,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        model: str = "distil-large-v3-turbo",
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
        logger.debug("Translating audio")

        try:
            client = await get_runtime_client()

            files = {"file": (filename, audio_bytes)}
            data = {
                "model": model,
                "response_format": response_format,
                "temperature": str(temperature),
            }
            if prompt:
                data["prompt"] = prompt

            request_timeout = httpx.Timeout(
                connect=10.0,
                read=timeout,
                write=120.0,
                pool=10.0,
            )

            response = await client.post(
                "/v1/audio/translations",
                files=files,
                data=data,
                timeout=request_timeout,
            )

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
            logger.error(f"Failed to connect to Universal Runtime: {e}")
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
    # Text-to-Speech
    # =========================================================================

    @classmethod
    async def synthesize_speech(
        cls,
        text: str,
        model: str = "kokoro",
        voice: str = "af_heart",
        response_format: str = "mp3",
        speed: float = 1.0,
        timeout: float = 60.0,
    ) -> bytes:
        """Generate speech from text.

        Args:
            text: Text to synthesize (max 4096 characters)
            model: TTS model identifier
            voice: Voice ID to use
            response_format: Audio format (mp3, opus, wav, pcm, flac, aac)
            speed: Speech speed multiplier (0.25 to 4.0)
            timeout: Request timeout in seconds

        Returns:
            Audio bytes in the requested format
        """
        logger.debug("Synthesizing speech")

        payload = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": response_format,
            "speed": speed,
        }

        try:
            client = await get_runtime_client()

            request_timeout = httpx.Timeout(
                connect=10.0,
                read=timeout,
                write=30.0,
                pool=10.0,
            )

            response = await client.post(
                "/v1/audio/speech",
                json=payload,
                timeout=request_timeout,
            )

            if response.status_code >= 400:
                try:
                    error_detail = response.json().get("detail", response.text)
                except Exception:
                    error_detail = response.text
                raise HTTPException(
                    status_code=response.status_code,
                    detail=error_detail,
                )

            return response.content

        except httpx.ConnectError as e:
            logger.error(f"Failed to connect to Universal Runtime: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Universal Runtime not available at {cls.get_base_url()}. "
                "Start it with: nx start universal",
            ) from e
        except httpx.TimeoutException as e:
            logger.error(f"TTS request timed out: {e}")
            raise HTTPException(
                status_code=504,
                detail="TTS request timed out. Text may be too long.",
            ) from e
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error synthesizing speech: {str(e)}",
            ) from e

    @classmethod
    async def list_tts_voices(cls, model: str | None = None) -> dict[str, Any]:
        """List available TTS voices.

        Args:
            model: Filter by model ID (optional)

        Returns:
            List of available voices
        """
        path = "/v1/audio/voices"
        if model:
            path += f"?model={model}"
        return await cls._make_request("GET", path, timeout=10.0)

    # =========================================================================
    # Health Check
    # =========================================================================

    @classmethod
    async def health_check(cls) -> dict[str, Any]:
        """Check Universal Runtime health."""
        return await cls._make_request("GET", "/health", timeout=10.0)
