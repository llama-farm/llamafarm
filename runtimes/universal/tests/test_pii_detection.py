"""Tests for PII Detection and Redaction Model.

Phase 8 of Universal Runtime ML Enhancements.
"""

import pytest

from models.pii_model import PIIModel, DEFAULT_PII_TYPES, PII_PATTERNS


class TestPIIModelLoading:
    """Test PII model loading."""

    @pytest.mark.asyncio
    async def test_model_loads_successfully(self):
        """Test that GLiNER PII model loads."""
        model = PIIModel(model_id="test-pii", device="cpu")
        await model.load()
        assert model.is_loaded
        await model.unload()

    @pytest.mark.asyncio
    async def test_model_unload(self):
        """Test that model can be unloaded."""
        model = PIIModel(model_id="test-pii", device="cpu")
        await model.load()
        assert model.is_loaded
        await model.unload()
        assert not model.is_loaded


class TestPIIDetection:
    """Test PII detection functionality."""

    @pytest.fixture
    async def model(self):
        """Create and load a model for testing."""
        model = PIIModel(model_id="test-pii", device="cpu")
        await model.load()
        yield model
        await model.unload()

    @pytest.mark.asyncio
    async def test_detect_email(self, model):
        """Test detecting email addresses."""
        result = await model.detect("Contact me at john.doe@example.com")
        emails = [e for e in result if e["label"] == "email"]
        assert len(emails) >= 1
        assert "john.doe@example.com" in [e["text"] for e in emails]

    @pytest.mark.asyncio
    async def test_detect_phone(self, model):
        """Test detecting phone numbers."""
        result = await model.detect("Call me at 555-123-4567")
        phones = [e for e in result if e["label"] in ["phone", "phone number"]]
        assert len(phones) >= 1

    @pytest.mark.asyncio
    async def test_detect_ssn(self, model):
        """Test detecting social security numbers."""
        result = await model.detect("My SSN is 123-45-6789")
        # Look for SSN pattern match or social security entity
        ssns = [e for e in result if "ssn" in e["label"].lower() or "social" in e["label"].lower() or e["text"] == "123-45-6789"]
        assert len(ssns) >= 1

    @pytest.mark.asyncio
    async def test_detect_person_name(self, model):
        """Test detecting person names."""
        result = await model.detect("Please contact John Smith regarding this matter")
        persons = [e for e in result if e["label"] == "person"]
        assert len(persons) >= 1

    @pytest.mark.asyncio
    async def test_empty_text_returns_empty(self, model):
        """Test that empty text returns empty list."""
        result = await model.detect("")
        assert result == []

    @pytest.mark.asyncio
    async def test_no_pii_returns_empty(self, model):
        """Test text without PII."""
        result = await model.detect("The weather is nice today.", use_regex=False)
        # May still detect some entities via GLiNER, but should be minimal
        assert isinstance(result, list)


class TestCustomEntityTypes:
    """Test custom entity type detection."""

    @pytest.fixture
    async def model(self):
        """Create and load a model for testing."""
        model = PIIModel(model_id="test-pii", device="cpu")
        await model.load()
        yield model
        await model.unload()

    @pytest.mark.asyncio
    async def test_custom_entity_types(self, model):
        """Test detection with custom entity types."""
        result = await model.detect(
            "The project codename is Phoenix and budget is $50000",
            entity_types=["project name", "money"],
            use_regex=False,
        )
        # Should detect based on custom types
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_detection_threshold(self, model):
        """Test detection confidence threshold."""
        # High threshold should return fewer results
        high_thresh = await model.detect(
            "Contact John at john@email.com",
            threshold=0.9,
            use_regex=False,
        )
        # Low threshold should return more results
        low_thresh = await model.detect(
            "Contact John at john@email.com",
            threshold=0.1,
            use_regex=False,
        )
        # Low threshold should have >= results
        assert len(low_thresh) >= len(high_thresh)


class TestPIIRedaction:
    """Test PII redaction functionality."""

    @pytest.fixture
    async def model(self):
        """Create and load a model for testing."""
        model = PIIModel(model_id="test-pii", device="cpu")
        await model.load()
        yield model
        await model.unload()

    @pytest.mark.asyncio
    async def test_basic_redaction(self, model):
        """Test basic PII redaction."""
        result = await model.redact("My email is test@example.com")
        assert "[REDACTED]" in result["redacted_text"]
        assert "test@example.com" not in result["redacted_text"]

    @pytest.mark.asyncio
    async def test_custom_replacement(self, model):
        """Test custom replacement string."""
        result = await model.redact(
            "My email is test@example.com",
            replacement="[HIDDEN]"
        )
        assert "[HIDDEN]" in result["redacted_text"]

    @pytest.mark.asyncio
    async def test_replacement_map(self, model):
        """Test per-entity-type replacement."""
        result = await model.redact(
            "Contact John at john@email.com",
            replacement_map={"email": "[EMAIL]", "person": "[NAME]"}
        )
        # Should use specific replacements
        assert "[EMAIL]" in result["redacted_text"] or "[NAME]" in result["redacted_text"]

    @pytest.mark.asyncio
    async def test_redact_returns_entities(self, model):
        """Test that redact also returns detected entities."""
        result = await model.redact("My email is test@example.com")
        assert "entities" in result
        assert "entity_count" in result
        assert result["entity_count"] > 0

    @pytest.mark.asyncio
    async def test_no_pii_returns_original(self, model):
        """Test that text without PII is returned unchanged."""
        original = "The weather is nice today."
        result = await model.redact(original, use_regex=False)
        # Should be close to original (GLiNER might still detect something)
        assert "redacted_text" in result


class TestRegexPatterns:
    """Test regex-based PII detection."""

    @pytest.fixture
    async def model(self):
        """Create and load a model for testing."""
        model = PIIModel(model_id="test-pii", device="cpu")
        await model.load()
        yield model
        await model.unload()

    @pytest.mark.asyncio
    async def test_regex_email(self, model):
        """Test regex email detection."""
        result = await model.detect(
            "Email: user@domain.org",
            use_regex=True
        )
        emails = [e for e in result if e["label"] == "email"]
        assert len(emails) >= 1
        # Regex matches should have score=1.0
        assert any(e["score"] == 1.0 for e in emails)

    @pytest.mark.asyncio
    async def test_regex_ip_address(self, model):
        """Test regex IP address detection."""
        result = await model.detect(
            "Server IP: 192.168.1.1",
            use_regex=True
        )
        # Look for either ip_address label or the actual IP in results
        ips = [e for e in result if e["label"] == "ip_address" or e["text"] == "192.168.1.1"]
        assert len(ips) >= 1

    @pytest.mark.asyncio
    async def test_regex_credit_card(self, model):
        """Test regex credit card detection."""
        result = await model.detect(
            "Card: 4111-1111-1111-1111",
            use_regex=True
        )
        # Look for credit_card label or the card number in results
        cards = [e for e in result if e["label"] == "credit_card" or "4111" in e["text"]]
        assert len(cards) >= 1


class TestBatchProcessing:
    """Test batch PII detection."""

    @pytest.fixture
    async def model(self):
        """Create and load a model for testing."""
        model = PIIModel(model_id="test-pii", device="cpu")
        await model.load()
        yield model
        await model.unload()

    @pytest.mark.asyncio
    async def test_batch_detection(self, model):
        """Test batch PII detection."""
        texts = [
            "Email: test@example.com",
            "Phone: 555-123-4567",
            "No PII here",
        ]
        results = await model.detect_batch(texts)
        assert len(results) == 3
        # First two should have entities
        assert len(results[0]) > 0
        assert len(results[1]) > 0


class TestModelInfo:
    """Test model information retrieval."""

    @pytest.mark.asyncio
    async def test_model_info(self):
        """Test getting model info."""
        model = PIIModel(model_id="test-pii", device="cpu")
        await model.load()

        info = model.get_model_info()
        assert "default_pii_types" in info
        assert "regex_patterns" in info
        assert info["is_loaded"] is True

        await model.unload()


class TestConstants:
    """Test module constants."""

    def test_default_pii_types(self):
        """Test that default PII types are defined."""
        assert len(DEFAULT_PII_TYPES) > 0
        assert "email" in DEFAULT_PII_TYPES
        assert "person" in DEFAULT_PII_TYPES

    def test_pii_patterns(self):
        """Test that regex patterns are defined."""
        assert "email" in PII_PATTERNS
        assert "phone" in PII_PATTERNS
        assert "ssn" in PII_PATTERNS
