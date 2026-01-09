"""
Router model for semantic routing of queries to target models.

Uses sentence-transformer embeddings for sub-millisecond routing decisions
based on semantic similarity to route-specific utterances.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

from .base import BaseModel

logger = logging.getLogger(__name__)

# Default directory for saving router models
# Uses ~/.llamafarm/models/router/ (or LF_DATA_DIR/models/router/)
# to match anomaly/classifier storage
_LF_DATA_DIR = Path(os.environ.get("LF_DATA_DIR", Path.home() / ".llamafarm"))
ROUTER_MODELS_DIR = _LF_DATA_DIR / "models" / "router"


@dataclass
class RouteDecision:
    """Result of a routing decision."""

    target_model: str
    route_name: str | None
    similarity_score: float
    matched_utterance: str | None = None
    # Complexity classifier fields (Phase 8)
    complexity_label: str | None = None
    complexity_score: float | None = None


@dataclass
class Route:
    """A single route configuration."""

    name: str
    target_model: str
    utterances: list[str] = field(default_factory=list)


class RouterModel(BaseModel):
    """Semantic router model for query-based routing.

    Routes incoming queries to target models based on semantic similarity
    to pre-defined route utterances. Uses cosine similarity with embeddings
    for sub-millisecond routing decisions.

    Example config:
        {
            "name": "my_router",
            "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
            "default_model": "general_llm",
            "similarity_threshold": 0.7,
            "routes": [
                {
                    "name": "billing",
                    "target_model": "billing_model",
                    "utterances": ["what is my bill", "payment options"]
                },
                {
                    "name": "support",
                    "target_model": "support_model",
                    "utterances": ["help with login", "password reset"]
                }
            ]
        }
    """

    def __init__(
        self,
        model_id: str,
        device: str,
        config: dict[str, Any] | None = None,
        token: str | None = None,
    ):
        """Initialize the router model.

        Args:
            model_id: Unique identifier for this router (or path to saved router)
            device: Target device (cuda/mps/cpu)
            config: Router configuration with routes and settings
            token: HuggingFace authentication token for gated models
        """
        super().__init__(model_id, device, token=token)
        self.model_type = "router"
        self.supports_streaming = False

        # Router-specific attributes
        self._embedder: Any | None = None
        self._embedder_tokenizer: PreTrainedTokenizerBase | None = None
        self._is_loaded = False

        # Complexity classifier (Phase 8)
        self._complexity_classifier: Any | None = None

        # Parse config
        if config:
            self.embedder_model = config.get(
                "embedder_model", "sentence-transformers/all-MiniLM-L6-v2"
            )
            self.default_model = config.get("default_model", "default")
            self.similarity_threshold = config.get("similarity_threshold", 0.7)
            self.routes: list[Route] = []
            for route_config in config.get("routes", []):
                self.routes.append(
                    Route(
                        name=route_config["name"],
                        target_model=route_config["target_model"],
                        utterances=route_config.get("utterances", []),
                    )
                )
            # Complexity classifier config (Phase 8)
            self.complexity_classifier_name = config.get("complexity_classifier")
            self.complexity_threshold = config.get("complexity_threshold", 0.5)
            self.complex_model = config.get("complex_model")
        else:
            # Will be loaded from saved model
            self.embedder_model = "sentence-transformers/all-MiniLM-L6-v2"
            self.default_model = "default"
            self.similarity_threshold = 0.7
            self.routes = []
            self.complexity_classifier_name = None
            self.complexity_threshold = 0.5
            self.complex_model = None

        # Pre-computed embeddings for each route
        self._route_embeddings: dict[str, np.ndarray] = {}

    def _validate_path_within_root(self, path: Path, root: Path) -> bool:
        """Validate that a path is safely within a root directory.

        Prevents path traversal attacks by ensuring the resolved path
        is a subdirectory of the allowed root.
        """
        try:
            resolved_path = path.resolve()
            resolved_root = root.resolve()
            # Check if path is within root (handles symlinks, .., etc.)
            resolved_path.relative_to(resolved_root)
            return True
        except ValueError:
            return False

    def _is_allowed_storage_path(self, path: Path) -> bool:
        """Check if a path is within any allowed storage directory.

        Allowed directories:
        1. Global router models dir (~/.llamafarm/models/router/)
        2. Project-specific dirs (~/.llamafarm/projects/.../lf_data/routers/)

        For testing, set LF_ALLOW_ANY_STORAGE_PATH=1 to bypass validation.
        """
        # Allow bypassing validation for tests
        if os.environ.get("LF_ALLOW_ANY_STORAGE_PATH") == "1":
            return True

        # Check global models directory
        if self._validate_path_within_root(path, ROUTER_MODELS_DIR):
            return True

        # Check project-specific directories
        projects_root = _LF_DATA_DIR / "projects"
        if self._validate_path_within_root(path, projects_root):
            # Additional check: ensure it's in a lf_data/routers subdirectory
            try:
                resolved_path = path.resolve()
                resolved_projects = projects_root.resolve()
                relative_path = resolved_path.relative_to(resolved_projects)
                path_parts = relative_path.parts
                # Expected: {namespace}/{project}/lf_data/routers/...
                if len(path_parts) >= 4 and path_parts[2] == "lf_data" and path_parts[3] == "routers":
                    return True
            except ValueError:
                pass

        return False

    async def load(self) -> None:
        """Load the embedder model and compute route embeddings.

        If model_id is a path to a saved router, loads from disk instead.
        Only loads from paths within allowed directories for security.
        """
        # Ensure ROUTER_MODELS_DIR exists for validation
        ROUTER_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Check if model_id looks like a path and is within allowed directory
        model_path = Path(self.model_id)
        if model_path.is_absolute():
            # For absolute paths, validate they are within allowed roots
            if self._is_allowed_storage_path(model_path):
                resolved_path = model_path.resolve()
                if resolved_path.exists() and (resolved_path / "config.json").exists():
                    await self._load_from_disk(resolved_path)
                    return
            else:
                logger.warning(
                    f"Refusing to load router from path outside allowed directory: "
                    f"{model_path}"
                )

        # Check in ROUTER_MODELS_DIR using sanitized model_id (no path traversal)
        # Sanitize: remove any path components, just use the base name
        safe_model_id = Path(self.model_id).name if "/" in self.model_id else self.model_id
        saved_path = ROUTER_MODELS_DIR / safe_model_id
        if self._validate_path_within_root(saved_path, ROUTER_MODELS_DIR):
            resolved_saved = saved_path.resolve()
            if resolved_saved.exists() and (resolved_saved / "config.json").exists():
                await self._load_from_disk(resolved_saved)
                return

        logger.info(f"Loading router model: {self.model_id}")
        logger.info(f"Using embedder: {self.embedder_model}")

        # Load the embedder model
        dtype = self.get_dtype()

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "token": self.token,
        }
        if self.device != "cpu":
            model_kwargs["torch_dtype"] = dtype

        self._embedder_tokenizer = AutoTokenizer.from_pretrained(
            self.embedder_model, trust_remote_code=True, token=self.token
        )

        self._embedder = AutoModel.from_pretrained(self.embedder_model, **model_kwargs)
        self._embedder = self._embedder.to(self.device)
        self._embedder.eval()

        # Compute embeddings for all route utterances
        await self._compute_route_embeddings()

        # Load complexity classifier if configured (Phase 8)
        if self.complexity_classifier_name:
            await self._load_complexity_classifier()

        self._is_loaded = True
        logger.info(
            f"Router loaded with {len(self.routes)} routes, "
            f"threshold={self.similarity_threshold}"
        )

    async def _load_from_disk(self, path: Path) -> None:
        """Load router from saved files on disk.

        Args:
            path: Path to saved router directory
        """
        logger.info(f"Loading router from disk: {path}")

        # Load config
        with open(path / "config.json") as f:
            config = json.load(f)

        self.embedder_model = config["embedder_model"]
        self.default_model = config["default_model"]
        self.similarity_threshold = config["similarity_threshold"]
        self.routes = [
            Route(
                name=r["name"],
                target_model=r["target_model"],
                utterances=r.get("utterances", []),
            )
            for r in config["routes"]
        ]
        # Complexity classifier config (Phase 8)
        self.complexity_classifier_name = config.get("complexity_classifier")
        self.complexity_threshold = config.get("complexity_threshold", 0.5)
        self.complex_model = config.get("complex_model")

        # Load embedder
        dtype = self.get_dtype()
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "token": self.token,
        }
        if self.device != "cpu":
            model_kwargs["torch_dtype"] = dtype

        self._embedder_tokenizer = AutoTokenizer.from_pretrained(
            self.embedder_model, trust_remote_code=True, token=self.token
        )
        self._embedder = AutoModel.from_pretrained(self.embedder_model, **model_kwargs)
        self._embedder = self._embedder.to(self.device)
        self._embedder.eval()

        # Load pre-computed embeddings
        embeddings_path = path / "embeddings.npz"
        if embeddings_path.exists():
            data = np.load(embeddings_path)
            self._route_embeddings = {key: data[key] for key in data.files}
            logger.info(f"Loaded embeddings for {len(self._route_embeddings)} routes")
        else:
            # Recompute if not found
            await self._compute_route_embeddings()

        # Load complexity classifier if configured (Phase 8)
        if self.complexity_classifier_name:
            await self._load_complexity_classifier()

        self._is_loaded = True
        logger.info(f"Router loaded from disk: {path}")

    async def _compute_route_embeddings(self) -> None:
        """Compute and cache embeddings for all route utterances."""
        for route in self.routes:
            if not route.utterances:
                logger.warning(f"Route '{route.name}' has no utterances")
                continue

            # Encode all utterances for this route
            embeddings = []
            for utterance in route.utterances:
                embedding = self._encode_text(utterance)
                embeddings.append(embedding)

            # Store as numpy array for efficient similarity computation
            self._route_embeddings[route.name] = np.stack(embeddings)
            logger.debug(
                f"Computed {len(embeddings)} embeddings for route '{route.name}'"
            )

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text to embedding vector.

        Args:
            text: Input text to encode

        Returns:
            Normalized embedding vector as numpy array
        """
        assert self._embedder is not None, "Embedder not loaded"
        assert self._embedder_tokenizer is not None, "Tokenizer not loaded"

        # Tokenize
        encoded = self._embedder_tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        # Generate embedding
        with torch.no_grad():
            outputs = self._embedder(**encoded)

        # Mean pooling
        token_embeddings = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

        # Normalize
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding[0].cpu().numpy()

    def _get_classifier_path(self) -> Path:
        """Get path to classifier models directory."""
        return _LF_DATA_DIR / "models" / "classifier"

    async def _load_complexity_classifier(self) -> None:
        """Load the complexity classifier if configured (Phase 8).

        The classifier is expected to be a SetFit model that predicts
        'simple' or 'complex' labels.
        """
        if not self.complexity_classifier_name:
            return

        classifier_path = self._get_classifier_path() / self.complexity_classifier_name

        if not classifier_path.exists():
            logger.warning(
                f"Complexity classifier not found: {self.complexity_classifier_name}. "
                f"Complexity routing will be disabled."
            )
            return

        try:
            # Try to load SetFit model
            from setfit import SetFitModel

            self._complexity_classifier = SetFitModel.from_pretrained(
                str(classifier_path)
            )
            self._complexity_classifier = self._complexity_classifier.to(self.device)
            logger.info(
                f"Loaded complexity classifier: {self.complexity_classifier_name}"
            )
        except ImportError:
            logger.warning(
                "setfit not installed. Install with: uv pip install setfit. "
                "Complexity routing will be disabled."
            )
            self._complexity_classifier = None
        except Exception as e:
            logger.warning(
                f"Failed to load complexity classifier: {e}. "
                f"Complexity routing will be disabled."
            )
            self._complexity_classifier = None

    async def _predict_complexity(self, query: str) -> tuple[str | None, float | None]:
        """Predict the complexity of a query (Phase 8).

        Args:
            query: The input query

        Returns:
            Tuple of (label, score) or (None, None) if classifier not available
        """
        if self._complexity_classifier is None:
            return None, None

        try:
            # SetFit predict returns labels
            predictions = self._complexity_classifier.predict([query])
            label = str(predictions[0])

            # Try to get prediction probabilities if available
            try:
                probs = self._complexity_classifier.predict_proba([query])
                # Get the probability for the predicted class
                score = float(max(probs[0]))
            except Exception:
                # If predict_proba not available, use 1.0 as score
                score = 1.0

            return label, score
        except Exception as e:
            logger.warning(f"Complexity prediction failed: {e}")
            return None, None

    async def route(self, query: str) -> RouteDecision:
        """Route a query to the appropriate target model.

        Args:
            query: The input query to route

        Returns:
            RouteDecision with target model and routing metadata
        """
        if not self._is_loaded:
            raise RuntimeError("Router model not loaded")

        # Handle empty query
        if not query or not query.strip():
            return RouteDecision(
                target_model=self.default_model,
                route_name=None,
                similarity_score=0.0,
                matched_utterance=None,
            )

        # Encode query
        query_embedding = self._encode_text(query)

        # Find best matching route
        best_route: Route | None = None
        best_score = -1.0
        best_utterance: str | None = None

        for route in self.routes:
            if route.name not in self._route_embeddings:
                continue

            route_embeddings = self._route_embeddings[route.name]

            # Compute cosine similarity with all utterances in this route
            similarities = np.dot(route_embeddings, query_embedding)

            # Get best match for this route
            max_idx = np.argmax(similarities)
            max_score = similarities[max_idx]

            if max_score > best_score:
                best_score = max_score
                best_route = route
                best_utterance = route.utterances[max_idx]

        # Check threshold and return decision
        if best_route and best_score >= self.similarity_threshold:
            return RouteDecision(
                target_model=best_route.target_model,
                route_name=best_route.name,
                similarity_score=float(best_score),
                matched_utterance=best_utterance,
            )

        # No topic match - check complexity classifier (Phase 8)
        complexity_label, complexity_score = await self._predict_complexity(query)

        # If complex and we have a complex_model configured, route to it
        if (
            complexity_label == "complex"
            and self.complex_model
            and complexity_score is not None
            and complexity_score >= self.complexity_threshold
        ):
            return RouteDecision(
                target_model=self.complex_model,
                route_name=None,
                similarity_score=float(best_score) if best_score >= 0 else 0.0,
                matched_utterance=None,
                complexity_label=complexity_label,
                complexity_score=complexity_score,
            )

        # Default fallback
        return RouteDecision(
            target_model=self.default_model,
            route_name=None,
            similarity_score=float(best_score) if best_score >= 0 else 0.0,
            matched_utterance=None,
            complexity_label=complexity_label,
            complexity_score=complexity_score,
        )

    async def save(self, path: str | Path) -> Path:
        """Save the router to disk.

        Args:
            path: Full path to save directory, or name to save under ROUTER_MODELS_DIR

        Returns:
            Path to saved router directory
        """
        save_path = Path(path)

        # If path doesn't look like a full path, treat as name and use default dir
        if not save_path.is_absolute():
            # Sanitize name to prevent path traversal
            safe_name = "".join(c for c in str(path) if c.isalnum() or c in "-_")
            if not safe_name:
                raise ValueError("Invalid router name")
            save_path = ROUTER_MODELS_DIR / safe_name

        save_path.mkdir(parents=True, exist_ok=True)

        # Save config
        config = {
            "embedder_model": self.embedder_model,
            "default_model": self.default_model,
            "similarity_threshold": self.similarity_threshold,
            "routes": [
                {
                    "name": r.name,
                    "target_model": r.target_model,
                    "utterances": r.utterances,
                }
                for r in self.routes
            ],
            # Complexity classifier config (Phase 8)
            "complexity_classifier": self.complexity_classifier_name,
            "complexity_threshold": self.complexity_threshold,
            "complex_model": self.complex_model,
        }

        with open(save_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save embeddings
        if self._route_embeddings:
            np.savez(save_path / "embeddings.npz", **self._route_embeddings)

        logger.info(f"Router saved to: {save_path}")
        return save_path

    async def unload(self) -> None:
        """Unload the router model and free resources."""
        logger.info(f"Unloading router: {self.model_id}")

        if self._embedder is not None and hasattr(self._embedder, "to"):
            try:
                self._embedder = self._embedder.to("cpu")
            except Exception as e:
                logger.warning(f"Could not move embedder to CPU: {e}")

        self._embedder = None
        self._embedder_tokenizer = None
        self._route_embeddings = {}
        self._complexity_classifier = None
        self._is_loaded = False

        # Clear CUDA/MPS cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

        logger.info(f"Router unloaded: {self.model_id}")

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded router."""
        info = super().get_model_info()
        info.update(
            {
                "embedder_model": self.embedder_model,
                "default_model": self.default_model,
                "similarity_threshold": self.similarity_threshold,
                "num_routes": len(self.routes),
                "routes": [r.name for r in self.routes],
                "is_loaded": self._is_loaded,
                # Complexity classifier info (Phase 8)
                "complexity_classifier": self.complexity_classifier_name,
                "complexity_threshold": self.complexity_threshold,
                "complex_model": self.complex_model,
                "complexity_classifier_loaded": self._complexity_classifier is not None,
            }
        )
        return info
