"""PII Detection and Redaction Model using GLiNER.

GLiNER (Generalist Model for Named Entity Recognition) can detect
custom entity types without training, making it perfect for PII detection.

Supports:
- Standard PII types (SSN, phone, email, credit card, address)
- Custom entity types defined at inference time
- Redaction with configurable replacement patterns

Usage:
    model = PIIModel(model_id="pii")
    await model.load()
    result = await model.detect("Contact me at john@email.com or 555-1234")
    # Returns detected entities with spans
    redacted = await model.redact("My SSN is 123-45-6789", replacement="[REDACTED]")
    # Returns "My SSN is [REDACTED]"
"""

import asyncio
import logging
import re
from typing import Any

from models.base import BaseModel

logger = logging.getLogger(__name__)

# Default GLiNER model - good balance of speed and accuracy
DEFAULT_GLINER_MODEL = "urchade/gliner_small-v2.1"

# Standard PII entity types
DEFAULT_PII_TYPES = [
    "person",
    "email",
    "phone number",
    "social security number",
    "credit card number",
    "address",
    "date of birth",
    "passport number",
    "driver license",
    "bank account",
    "ip address",
]

# Regex patterns for common PII (used as fallback or supplement)
PII_PATTERNS = {
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "phone": r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
    "ssn": r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
    "credit_card": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
    "ip_address": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
}


class PIIModel(BaseModel):
    """PII detection and redaction model using GLiNER.

    GLiNER is a zero-shot NER model that can detect custom entity types
    without additional training. This makes it flexible for PII detection
    with custom requirements.

    Attributes:
        model_id: Unique identifier for this model instance
        device: Device to run inference on (cpu, cuda, mps)
        hf_model_name: HuggingFace GLiNER model name
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        hf_model_name: str = DEFAULT_GLINER_MODEL,
        token: str | None = None,
    ):
        """Initialize PII model.

        Args:
            model_id: Unique identifier for this model instance
            device: Device to run on (cpu, cuda, mps)
            hf_model_name: HuggingFace GLiNER model to load
            token: Optional HuggingFace token
        """
        super().__init__(model_id=model_id, device=device, token=token)
        self.hf_model_name = hf_model_name
        self.model_type = "pii_detection"
        self.supports_streaming = False
        self._is_loaded = False
        self.gliner_model = None

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self._is_loaded and self.gliner_model is not None

    async def load(self) -> None:
        """Load the GLiNER model."""
        if self.is_loaded:
            logger.info(f"Model {self.model_id} already loaded")
            return

        logger.info(f"Loading GLiNER PII model: {self.hf_model_name}")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_sync)

        self._is_loaded = True
        logger.info(f"GLiNER PII model loaded: {self.model_id}")

    def _load_sync(self) -> None:
        """Synchronous model loading (run in executor)."""
        from gliner import GLiNER

        self.gliner_model = GLiNER.from_pretrained(
            self.hf_model_name,
            # GLiNER handles device internally
        )
        # Move to device if supported
        if self.device in ["cuda", "mps"]:
            try:
                self.gliner_model.to(self.device)
            except Exception as e:
                logger.warning(f"Could not move GLiNER to {self.device}: {e}")

    async def unload(self) -> None:
        """Unload the model and free resources."""
        if self.gliner_model is not None:
            del self.gliner_model
            self.gliner_model = None
        await super().unload()
        self._is_loaded = False

    async def detect(
        self,
        text: str,
        entity_types: list[str] | None = None,
        threshold: float = 0.5,
        use_regex: bool = True,
    ) -> list[dict[str, Any]]:
        """Detect PII entities in text.

        Args:
            text: Text to analyze for PII
            entity_types: List of entity types to detect (default: standard PII types)
            threshold: Minimum confidence threshold (0-1)
            use_regex: Whether to also use regex patterns for common PII

        Returns:
            List of detected entities with:
                - text: The matched text
                - label: Entity type label
                - start: Start character position
                - end: End character position
                - score: Confidence score (regex matches get score=1.0)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if not text or not text.strip():
            return []

        if entity_types is None:
            entity_types = DEFAULT_PII_TYPES

        loop = asyncio.get_running_loop()
        entities = await loop.run_in_executor(
            None,
            self._detect_sync,
            text,
            entity_types,
            threshold,
        )

        # Optionally add regex-based detection
        if use_regex:
            regex_entities = self._detect_with_regex(text)
            entities = self._merge_entities(entities, regex_entities)

        return entities

    def _detect_sync(
        self,
        text: str,
        entity_types: list[str],
        threshold: float,
    ) -> list[dict[str, Any]]:
        """Synchronous PII detection."""
        # GLiNER prediction
        raw_entities = self.gliner_model.predict_entities(
            text,
            entity_types,
            threshold=threshold,
        )

        # Convert to standard format
        entities = []
        for ent in raw_entities:
            entities.append({
                "text": ent["text"],
                "label": ent["label"],
                "start": ent["start"],
                "end": ent["end"],
                "score": float(ent["score"]),
            })

        return entities

    def _detect_with_regex(self, text: str) -> list[dict[str, Any]]:
        """Detect PII using regex patterns."""
        entities = []

        for label, pattern in PII_PATTERNS.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    "text": match.group(),
                    "label": label,
                    "start": match.start(),
                    "end": match.end(),
                    "score": 1.0,  # Regex matches are certain
                })

        return entities

    def _merge_entities(
        self,
        gliner_entities: list[dict],
        regex_entities: list[dict],
    ) -> list[dict[str, Any]]:
        """Merge GLiNER and regex entities, removing overlaps.

        When entities overlap, prioritizes by score (higher wins).
        This handles all overlap cases: GLiNER-GLiNER, regex-regex,
        and GLiNER-regex overlaps.
        """
        # Combine all entities and sort by start position, then by score (descending)
        all_entities = gliner_entities + regex_entities
        if not all_entities:
            return []

        all_entities.sort(key=lambda x: (x["start"], -x.get("score", 0.0)))

        # Greedily select non-overlapping entities
        merged = []
        current_end = -1

        for entity in all_entities:
            # If this entity starts after the current covered region, add it
            if entity["start"] >= current_end:
                merged.append(entity)
                current_end = entity["end"]
            # If it overlaps, only keep it if it has a higher score and different span
            elif entity.get("score", 0) > merged[-1].get("score", 0):
                # Check if it's a better match for a similar span
                if entity["start"] == merged[-1]["start"]:
                    merged[-1] = entity
                    current_end = entity["end"]

        return merged

    async def redact(
        self,
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
            entity_types: Entity types to redact (default: standard PII types)
            replacement: Default replacement string
            replacement_map: Map of entity type to replacement (e.g., {"email": "[EMAIL]"})
            threshold: Detection confidence threshold
            use_regex: Whether to use regex patterns

        Returns:
            Dictionary with:
                - redacted_text: Text with PII replaced
                - entities: List of detected/redacted entities
                - entity_count: Number of entities redacted
        """
        # Detect entities
        entities = await self.detect(
            text,
            entity_types=entity_types,
            threshold=threshold,
            use_regex=use_regex,
        )

        if not entities:
            return {
                "redacted_text": text,
                "entities": [],
                "entity_count": 0,
            }

        # Build redacted text (process from end to preserve positions)
        redacted_text = text
        entities_sorted = sorted(entities, key=lambda x: x["start"], reverse=True)

        for entity in entities_sorted:
            # Get replacement for this entity type
            if replacement_map and entity["label"] in replacement_map:
                repl = replacement_map[entity["label"]]
            else:
                repl = replacement

            # Replace
            redacted_text = (
                redacted_text[:entity["start"]] +
                repl +
                redacted_text[entity["end"]:]
            )

        return {
            "redacted_text": redacted_text,
            "entities": sorted(entities, key=lambda x: x["start"]),
            "entity_count": len(entities),
        }

    async def detect_batch(
        self,
        texts: list[str],
        entity_types: list[str] | None = None,
        threshold: float = 0.5,
        use_regex: bool = True,
    ) -> list[list[dict[str, Any]]]:
        """Detect PII in multiple texts.

        Args:
            texts: List of texts to analyze
            entity_types: Entity types to detect
            threshold: Detection threshold
            use_regex: Whether to use regex patterns

        Returns:
            List of entity lists (one per input text)
        """
        results = []
        for text in texts:
            entities = await self.detect(
                text,
                entity_types=entity_types,
                threshold=threshold,
                use_regex=use_regex,
            )
            results.append(entities)
        return results

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        info = super().get_model_info()
        info.update({
            "hf_model_name": self.hf_model_name,
            "is_loaded": self.is_loaded,
            "default_pii_types": DEFAULT_PII_TYPES,
            "regex_patterns": list(PII_PATTERNS.keys()),
        })
        return info
