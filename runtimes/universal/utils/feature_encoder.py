"""
Feature encoder for mixed data types (numeric + categorical).

Provides automatic encoding of categorical features for ML models that only accept
numeric input. Uses a registry pattern for extensibility.

Supported encoding types:
- numeric: Pass through as-is (int/float)
- hash: MD5 hash to integer (good for high-cardinality strings like user_agent)
- label: Label encoding (category → integer mapping, learned from training data)
- onehot: One-hot encoding (expands to multiple columns)
- binary: Binary encoding (yes/no, true/false → 0/1)
- frequency: Encode as frequency count from training data

Usage:
    encoder = FeatureEncoder()

    # Define schema
    schema = {
        "response_time_ms": "numeric",
        "user_agent": "hash",
        "endpoint": "label",
        "is_authenticated": "binary"
    }

    # Fit on training data (learns label mappings)
    encoder.fit(training_data, schema)

    # Transform data
    encoded = encoder.transform(new_data)

    # Save/load for production
    encoder.save("encoder.joblib")
    encoder = FeatureEncoder.load("encoder.joblib")
"""

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)

# Type alias for encoding types
EncodingType = Literal["numeric", "hash", "label", "onehot", "binary", "frequency"]


# =============================================================================
# Encoder Registry - Extensible pattern for adding new encoders
# =============================================================================


class BaseFeatureEncoderStrategy(ABC):
    """Base class for feature encoding strategies."""

    @abstractmethod
    def fit(self, values: list[Any]) -> None:
        """Learn encoding from training data."""
        pass

    @abstractmethod
    def transform(self, value: Any) -> float | list[float]:
        """Transform a single value to numeric."""
        pass

    @abstractmethod
    def get_output_dim(self) -> int:
        """Return number of output dimensions (1 for most, >1 for onehot)."""
        pass

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Get serializable state for saving."""
        pass

    @classmethod
    @abstractmethod
    def from_state(cls, state: dict[str, Any]) -> "BaseFeatureEncoderStrategy":
        """Restore from saved state."""
        pass


class NumericEncoder(BaseFeatureEncoderStrategy):
    """Pass-through encoder for numeric values."""

    def fit(self, values: list[Any]) -> None:
        pass  # No fitting needed

    def transform(self, value: Any) -> float:
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def get_output_dim(self) -> int:
        return 1

    def get_state(self) -> dict[str, Any]:
        return {"type": "numeric"}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "NumericEncoder":
        return cls()


class HashEncoder(BaseFeatureEncoderStrategy):
    """Hash strings to integers using MD5.

    Good for high-cardinality categorical features like user agents, IPs, etc.
    Produces consistent output without needing to store a vocabulary.
    """

    def __init__(self, max_value: int = 1_000_000):
        self.max_value = max_value

    def fit(self, values: list[Any]) -> None:
        pass  # No fitting needed - hash is deterministic

    def transform(self, value: Any) -> float:
        if value is None:
            return 0.0
        str_value = str(value)
        hash_bytes = hashlib.md5(str_value.encode()).hexdigest()[:8]
        return float(int(hash_bytes, 16) % self.max_value)

    def get_output_dim(self) -> int:
        return 1

    def get_state(self) -> dict[str, Any]:
        return {"type": "hash", "max_value": self.max_value}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "HashEncoder":
        return cls(max_value=state.get("max_value", 1_000_000))


class LabelEncoder(BaseFeatureEncoderStrategy):
    """Label encoding - maps categories to integers.

    Learns a vocabulary from training data. Unknown values at inference
    time are mapped to a special "unknown" category.
    """

    def __init__(self):
        self.label_map: dict[str, int] = {}
        self.unknown_value: int = 0

    def fit(self, values: list[Any]) -> None:
        unique_values = sorted(set(str(v) for v in values if v is not None))
        # Reserve 0 for unknown
        self.label_map = {v: i + 1 for i, v in enumerate(unique_values)}
        self.unknown_value = 0
        logger.debug(f"LabelEncoder fit with {len(self.label_map)} categories")

    def transform(self, value: Any) -> float:
        if value is None:
            return float(self.unknown_value)
        str_value = str(value)
        return float(self.label_map.get(str_value, self.unknown_value))

    def get_output_dim(self) -> int:
        return 1

    def get_state(self) -> dict[str, Any]:
        return {
            "type": "label",
            "label_map": self.label_map,
            "unknown_value": self.unknown_value,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "LabelEncoder":
        encoder = cls()
        encoder.label_map = state.get("label_map", {})
        encoder.unknown_value = state.get("unknown_value", 0)
        return encoder


class OneHotEncoder(BaseFeatureEncoderStrategy):
    """One-hot encoding - expands to multiple binary columns.

    Best for low-cardinality categorical features (< 20 categories).
    Unknown values get all zeros.
    """

    def __init__(self, max_categories: int = 100):
        self.categories: list[str] = []
        self.max_categories = max_categories

    def fit(self, values: list[Any]) -> None:
        unique_values = sorted(set(str(v) for v in values if v is not None))
        if len(unique_values) > self.max_categories:
            logger.warning(
                f"OneHotEncoder: {len(unique_values)} categories exceeds max "
                f"{self.max_categories}, truncating"
            )
            unique_values = unique_values[: self.max_categories]
        self.categories = unique_values
        logger.debug(f"OneHotEncoder fit with {len(self.categories)} categories")

    def transform(self, value: Any) -> list[float]:
        # If no categories were learned, return a single zero placeholder
        # to keep schema in sync with get_output_dim() and get_feature_names()
        if not self.categories:
            return [0.0]
        result = [0.0] * len(self.categories)
        if value is not None:
            str_value = str(value)
            if str_value in self.categories:
                result[self.categories.index(str_value)] = 1.0
        return result

    def get_output_dim(self) -> int:
        return max(len(self.categories), 1)

    def get_state(self) -> dict[str, Any]:
        return {
            "type": "onehot",
            "categories": self.categories,
            "max_categories": self.max_categories,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "OneHotEncoder":
        encoder = cls(max_categories=state.get("max_categories", 100))
        encoder.categories = state.get("categories", [])
        return encoder


class BinaryEncoder(BaseFeatureEncoderStrategy):
    """Binary encoding for boolean-like values.

    Handles various truthy/falsy representations:
    - true/false, True/False, TRUE/FALSE
    - yes/no, Yes/No, YES/NO
    - 1/0, "1"/"0"
    - on/off
    """

    TRUTHY = {"true", "yes", "1", "on", "t", "y"}
    FALSY = {"false", "no", "0", "off", "f", "n", ""}

    def fit(self, values: list[Any]) -> None:
        pass  # No fitting needed

    def transform(self, value: Any) -> float:
        if value is None:
            return 0.0
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return 1.0 if value else 0.0
        str_value = str(value).lower().strip()
        if str_value in self.TRUTHY:
            return 1.0
        return 0.0

    def get_output_dim(self) -> int:
        return 1

    def get_state(self) -> dict[str, Any]:
        return {"type": "binary"}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "BinaryEncoder":
        return cls()


class FrequencyEncoder(BaseFeatureEncoderStrategy):
    """Frequency encoding - encode as occurrence count from training data.

    Maps each category to its frequency (count) in the training data.
    Good for capturing "rarity" of categorical values.
    """

    def __init__(self, normalize: bool = True):
        self.frequency_map: dict[str, float] = {}
        self.normalize = normalize
        self.default_value: float = 0.0

    def fit(self, values: list[Any]) -> None:
        counts: dict[str, int] = {}
        for v in values:
            if v is not None:
                str_value = str(v)
                counts[str_value] = counts.get(str_value, 0) + 1

        if self.normalize and counts:
            total = sum(counts.values())
            self.frequency_map = {k: v / total for k, v in counts.items()}
        else:
            self.frequency_map = {k: float(v) for k, v in counts.items()}

        # Default for unseen values is minimum frequency or 0
        if self.frequency_map:
            self.default_value = min(self.frequency_map.values()) / 2
        logger.debug(f"FrequencyEncoder fit with {len(self.frequency_map)} categories")

    def transform(self, value: Any) -> float:
        if value is None:
            return self.default_value
        str_value = str(value)
        return self.frequency_map.get(str_value, self.default_value)

    def get_output_dim(self) -> int:
        return 1

    def get_state(self) -> dict[str, Any]:
        return {
            "type": "frequency",
            "frequency_map": self.frequency_map,
            "normalize": self.normalize,
            "default_value": self.default_value,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "FrequencyEncoder":
        encoder = cls(normalize=state.get("normalize", True))
        encoder.frequency_map = state.get("frequency_map", {})
        encoder.default_value = state.get("default_value", 0.0)
        return encoder


# =============================================================================
# Encoder Registry
# =============================================================================

# Registry mapping encoding type names to encoder classes
ENCODER_REGISTRY: dict[str, type] = {
    "numeric": NumericEncoder,
    "hash": HashEncoder,
    "label": LabelEncoder,
    "onehot": OneHotEncoder,
    "binary": BinaryEncoder,
    "frequency": FrequencyEncoder,
}


def register_encoder(name: str, encoder_class: type) -> None:
    """Register a custom encoder type.

    Args:
        name: Name to use in schema (e.g., "my_encoder")
        encoder_class: Class that extends BaseFeatureEncoderStrategy

    Example:
        class MyEncoder(BaseFeatureEncoderStrategy):
            ...

        register_encoder("my_encoder", MyEncoder)
    """
    if not issubclass(encoder_class, BaseFeatureEncoderStrategy):
        raise ValueError("Encoder class must extend BaseFeatureEncoderStrategy")
    ENCODER_REGISTRY[name] = encoder_class
    logger.info(f"Registered custom encoder: {name}")


def get_encoder(encoding_type: str) -> BaseFeatureEncoderStrategy:
    """Get an encoder instance by type name."""
    if encoding_type not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoding type: {encoding_type}. "
            f"Available: {list(ENCODER_REGISTRY.keys())}"
        )
    return ENCODER_REGISTRY[encoding_type]()


# =============================================================================
# Main FeatureEncoder class
# =============================================================================


@dataclass
class FeatureSchema:
    """Schema defining how to encode each feature."""

    features: dict[str, EncodingType] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> "FeatureSchema":
        return cls(features=d)

    @classmethod
    def infer(cls, data: list[dict[str, Any]]) -> "FeatureSchema":
        """Infer schema from data by examining types.

        Heuristics:
        - int/float → numeric
        - bool → binary
        - str with < 20 unique values → label
        - str with >= 20 unique values → hash
        """
        if not data:
            return cls()

        # Collect all keys and sample values
        all_keys = set()
        value_samples: dict[str, list[Any]] = {}

        for row in data:
            for k, v in row.items():
                all_keys.add(k)
                if k not in value_samples:
                    value_samples[k] = []
                value_samples[k].append(v)

        features = {}
        for key in all_keys:
            samples = value_samples.get(key, [])
            features[key] = cls._infer_type(samples)

        return cls(features=features)

    @staticmethod
    def _infer_type(samples: list[Any]) -> EncodingType:
        """Infer encoding type from sample values."""
        non_null = [s for s in samples if s is not None]
        if not non_null:
            return "numeric"

        sample = non_null[0]

        # Check for numeric
        if isinstance(sample, (int, float)) and not isinstance(sample, bool):
            return "numeric"

        # Check for boolean
        if isinstance(sample, bool):
            return "binary"

        # String - check cardinality
        unique_values = set(str(v) for v in non_null)
        if len(unique_values) <= 20:
            return "label"
        else:
            return "hash"


class FeatureEncoder:
    """Encode mixed-type features to numeric arrays.

    Provides automatic encoding of categorical features for ML models.
    Supports saving/loading for production use.

    Example:
        encoder = FeatureEncoder()

        # Option 1: Explicit schema
        encoder.fit(data, schema={"response_time": "numeric", "user_agent": "hash"})

        # Option 2: Auto-infer schema
        encoder.fit(data)  # Schema inferred from data types

        # Transform
        encoded = encoder.transform(new_data)

        # Save/load
        encoder.save("encoder.joblib")
        encoder = FeatureEncoder.load("encoder.joblib")
    """

    def __init__(self):
        self.schema: FeatureSchema | None = None
        self.encoders: dict[str, BaseFeatureEncoderStrategy] = {}
        self.feature_order: list[str] = []
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def output_dim(self) -> int:
        """Total output dimensions after encoding."""
        if not self._is_fitted:
            return 0
        return sum(enc.get_output_dim() for enc in self.encoders.values())

    def fit(
        self,
        data: list[dict[str, Any]],
        schema: dict[str, str] | FeatureSchema | None = None,
    ) -> "FeatureEncoder":
        """Fit encoder on training data.

        Args:
            data: List of dicts with feature values
            schema: Optional schema mapping feature names to encoding types.
                    If None, schema is inferred from data.

        Returns:
            self (for chaining)
        """
        if not data:
            raise ValueError("Cannot fit on empty data")

        # Get or infer schema
        if schema is None:
            self.schema = FeatureSchema.infer(data)
            logger.info(f"Inferred schema: {self.schema.features}")
        elif isinstance(schema, dict):
            self.schema = FeatureSchema.from_dict(schema)
        else:
            self.schema = schema

        # Create and fit encoders for each feature
        self.encoders = {}
        self.feature_order = sorted(self.schema.features.keys())

        for feature_name in self.feature_order:
            encoding_type = self.schema.features[feature_name]
            encoder = get_encoder(encoding_type)

            # Extract values for this feature
            values = [row.get(feature_name) for row in data]
            encoder.fit(values)

            self.encoders[feature_name] = encoder

        self._is_fitted = True
        logger.info(
            f"FeatureEncoder fitted: {len(self.encoders)} features, "
            f"{self.output_dim} output dimensions"
        )
        return self

    def transform(self, data: list[dict[str, Any]] | dict[str, Any]) -> np.ndarray:
        """Transform data to numeric array.

        Args:
            data: Single dict or list of dicts with feature values

        Returns:
            2D numpy array of shape (n_samples, output_dim)
        """
        if not self._is_fitted:
            raise RuntimeError("Encoder not fitted. Call fit() first.")

        # Handle single dict
        if isinstance(data, dict):
            data = [data]

        result = []
        for row in data:
            row_encoded = []
            for feature_name in self.feature_order:
                encoder = self.encoders[feature_name]
                value = row.get(feature_name)
                encoded_value = encoder.transform(value)

                # Handle multi-dimensional outputs (onehot)
                if isinstance(encoded_value, list):
                    row_encoded.extend(encoded_value)
                else:
                    row_encoded.append(encoded_value)

            result.append(row_encoded)

        return np.array(result, dtype=np.float32)

    def fit_transform(
        self,
        data: list[dict[str, Any]],
        schema: dict[str, str] | FeatureSchema | None = None,
    ) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(data, schema).transform(data)

    def get_feature_names(self) -> list[str]:
        """Get output feature names after encoding.

        Useful for debugging and understanding output structure.
        """
        if not self._is_fitted:
            return []

        names = []
        for feature_name in self.feature_order:
            encoder = self.encoders[feature_name]
            dim = encoder.get_output_dim()

            if dim == 1:
                names.append(feature_name)
            else:
                # Multi-dimensional (onehot)
                for i in range(dim):
                    if isinstance(encoder, OneHotEncoder) and i < len(
                        encoder.categories
                    ):
                        names.append(f"{feature_name}_{encoder.categories[i]}")
                    else:
                        names.append(f"{feature_name}_{i}")

        return names

    def save(self, path: str | Path) -> None:
        """Save encoder to disk using JSON (safe serialization).

        Security Note: Uses JSON instead of pickle/joblib to prevent
        arbitrary code execution during deserialization.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted encoder")

        state = {
            "version": 1,  # For future compatibility
            "schema": self.schema.features if self.schema else {},
            "feature_order": self.feature_order,
            "encoders": {
                name: encoder.get_state() for name, encoder in self.encoders.items()
            },
        }

        path = Path(path)
        # Use JSON for safe serialization - all encoder state is JSON-compatible
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"FeatureEncoder saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "FeatureEncoder":
        """Load encoder from disk using JSON (safe deserialization).

        Security Notes:
        - Uses JSON instead of pickle/joblib to prevent arbitrary code execution
        - Path validation prevents path traversal attacks
        - This method should only be called with paths derived from validated sources
        """
        path = Path(path)

        # Security: Reject obvious path traversal patterns
        path_str = str(path)
        if ".." in path_str:
            raise ValueError(
                f"Security error: Path '{path}' contains '..' which is not allowed"
            )

        # Use JSON for safe deserialization
        with open(path) as f:
            state = json.load(f)

        encoder = cls()
        encoder.schema = FeatureSchema.from_dict(state.get("schema", {}))
        encoder.feature_order = state.get("feature_order", [])

        # Restore encoders from state
        encoder.encoders = {}
        for name, enc_state in state.get("encoders", {}).items():
            enc_type = enc_state.get("type", "numeric")
            enc_class = ENCODER_REGISTRY.get(enc_type, NumericEncoder)
            encoder.encoders[name] = enc_class.from_state(enc_state)

        encoder._is_fitted = True
        logger.info(f"FeatureEncoder loaded from {path}")
        return encoder

    def get_schema_info(self) -> dict[str, Any]:
        """Get information about the encoder schema."""
        if not self._is_fitted:
            return {"is_fitted": False}

        return {
            "is_fitted": True,
            "features": self.schema.features if self.schema else {},
            "output_dim": self.output_dim,
            "feature_names": self.get_feature_names(),
        }
