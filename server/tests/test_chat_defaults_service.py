"""Tests for ChatDefaultsService."""

from config.datamodel import ChatDefaults

from services.chat_defaults_service import ChatDefaultsService, ResolvedChatParams


class TestChatDefaultsService:
    """Tests for ChatDefaultsService."""

    def test_resolve_no_defaults_no_request(self):
        """Test with no defaults and no request params."""
        result = ChatDefaultsService.resolve(None, {})

        assert result.temperature is None
        assert result.rag_enabled is None
        assert result.database is None

    def test_resolve_model_defaults_only(self):
        """Test with model defaults and no request params."""
        model_defaults = ChatDefaults(
            temperature=0.7,
            rag_enabled=True,
            database="main_db",
            rag_top_k=5,
        )

        result = ChatDefaultsService.resolve(model_defaults, {})

        assert result.temperature == 0.7
        assert result.rag_enabled is True
        assert result.database == "main_db"
        assert result.rag_top_k == 5
        assert result.top_p is None  # Not set in defaults

    def test_resolve_request_overrides_defaults(self):
        """Test that request params override model defaults."""
        model_defaults = ChatDefaults(
            temperature=0.7,
            rag_enabled=True,
            database="main_db",
        )
        request_params = {
            "temperature": 0.9,
            "rag_enabled": False,
        }

        result = ChatDefaultsService.resolve(model_defaults, request_params)

        assert result.temperature == 0.9  # Request wins
        assert result.rag_enabled is False  # Request wins
        assert result.database == "main_db"  # Model default (not in request)

    def test_resolve_request_only(self):
        """Test with request params only, no model defaults."""
        request_params = {
            "temperature": 0.5,
            "think": True,
            "thinking_budget": 2000,
        }

        result = ChatDefaultsService.resolve(None, request_params)

        assert result.temperature == 0.5
        assert result.think is True
        assert result.thinking_budget == 2000
        assert result.rag_enabled is None

    def test_resolve_all_params(self):
        """Test with all parameters set."""
        model_defaults = ChatDefaults(
            temperature=0.7,
            top_p=0.9,
            max_tokens=1000,
            rag_enabled=True,
            database="db1",
            rag_retrieval_strategy="reranked",
            rag_top_k=10,
            rag_score_threshold=0.7,
            rag_queries=["query1"],
            think=False,
            thinking_budget=500,
            n_ctx=4096,
        )

        result = ChatDefaultsService.resolve(model_defaults, {})

        assert result.temperature == 0.7
        assert result.top_p == 0.9
        assert result.max_tokens == 1000
        assert result.rag_enabled is True
        assert result.database == "db1"
        assert result.rag_retrieval_strategy == "reranked"
        assert result.rag_top_k == 10
        assert result.rag_score_threshold == 0.7
        assert result.rag_queries == ["query1"]
        assert result.think is False
        assert result.thinking_budget == 500
        assert result.n_ctx == 4096

    def test_resolve_partial_override(self):
        """Test partial override of some params."""
        model_defaults = ChatDefaults(
            temperature=0.7,
            rag_enabled=True,
            database="default_db",
            rag_top_k=10,
        )
        request_params = {
            "database": "custom_db",
            "rag_top_k": 5,
            # temperature and rag_enabled not provided in request
        }

        result = ChatDefaultsService.resolve(model_defaults, request_params)

        assert result.temperature == 0.7  # From defaults
        assert result.rag_enabled is True  # From defaults
        assert result.database == "custom_db"  # From request
        assert result.rag_top_k == 5  # From request

    def test_resolve_none_in_request_uses_default(self):
        """Test that None in request params uses model default."""
        model_defaults = ChatDefaults(
            temperature=0.7,
        )
        request_params = {
            "temperature": None,  # Explicitly None
        }

        result = ChatDefaultsService.resolve(model_defaults, request_params)

        assert result.temperature == 0.7  # Default, because request was None

    def test_to_request_dict_filters_none(self):
        """Test that to_request_dict only includes non-None values."""
        params = ResolvedChatParams(
            temperature=0.7,
            top_p=None,
            max_tokens=1000,
            rag_enabled=True,
            database=None,
            rag_retrieval_strategy=None,
            rag_top_k=5,
            rag_score_threshold=None,
            rag_queries=None,
            think=None,
            thinking_budget=None,
            n_ctx=None,
        )

        result = ChatDefaultsService.to_request_dict(params)

        assert result == {
            "temperature": 0.7,
            "max_tokens": 1000,
            "rag_enabled": True,
            "rag_top_k": 5,
        }

    def test_to_request_dict_empty_when_all_none(self):
        """Test to_request_dict returns empty dict when all None."""
        params = ResolvedChatParams(
            temperature=None,
            top_p=None,
            max_tokens=None,
            rag_enabled=None,
            database=None,
            rag_retrieval_strategy=None,
            rag_top_k=None,
            rag_score_threshold=None,
            rag_queries=None,
            think=None,
            thinking_budget=None,
            n_ctx=None,
        )

        result = ChatDefaultsService.to_request_dict(params)

        assert result == {}

    def test_to_request_dict_includes_false_values(self):
        """Test that False values are included (not filtered as None)."""
        params = ResolvedChatParams(
            temperature=0.0,  # Falsy but valid
            top_p=None,
            max_tokens=None,
            rag_enabled=False,  # Falsy but valid
            database=None,
            rag_retrieval_strategy=None,
            rag_top_k=0,  # Falsy but valid
            rag_score_threshold=None,
            rag_queries=None,
            think=False,  # Falsy but valid
            thinking_budget=None,
            n_ctx=None,
        )

        result = ChatDefaultsService.to_request_dict(params)

        assert result == {
            "temperature": 0.0,
            "rag_enabled": False,
            "rag_top_k": 0,
            "think": False,
        }
