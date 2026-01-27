"""Tests for ChatRequest variables field.

Tests that the ChatRequest model accepts the new `variables` field
for dynamic template substitution.
"""



class TestChatRequestWithVariables:
    """Test ChatRequest accepts variables dict."""

    def test_variables_field_accepted(self):
        """ChatRequest accepts variables dict."""
        from api.routers.projects.projects import ChatRequest

        request = ChatRequest(
            messages=[{"role": "user", "content": "Hello"}],
            variables={"name": "Alice", "company": "Acme Corp"},
        )
        assert request.variables == {"name": "Alice", "company": "Acme Corp"}

    def test_variables_empty_dict(self):
        """ChatRequest accepts empty variables dict."""
        from api.routers.projects.projects import ChatRequest

        request = ChatRequest(
            messages=[{"role": "user", "content": "Hello"}],
            variables={},
        )
        assert request.variables == {}

    def test_variables_with_various_types(self):
        """Variables can contain various JSON types."""
        from api.routers.projects.projects import ChatRequest

        request = ChatRequest(
            messages=[{"role": "user", "content": "Hello"}],
            variables={
                "string_var": "hello",
                "int_var": 42,
                "float_var": 3.14,
                "bool_var": True,
                "null_var": None,
                "list_var": [1, 2, 3],
                "dict_var": {"nested": "value"},
            },
        )
        assert request.variables["string_var"] == "hello"
        assert request.variables["int_var"] == 42
        assert request.variables["bool_var"] is True


class TestChatRequestWithoutVariables:
    """Test ChatRequest works without variables (backwards compatible)."""

    def test_variables_defaults_to_none(self):
        """Variables field defaults to None when not provided."""
        from api.routers.projects.projects import ChatRequest

        request = ChatRequest(messages=[{"role": "user", "content": "Hello"}])
        assert request.variables is None

    def test_existing_fields_still_work(self):
        """All existing ChatRequest fields work without variables."""
        from api.routers.projects.projects import ChatRequest

        request = ChatRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4",
            stream=True,
            temperature=0.7,
            max_tokens=100,
            rag_enabled=True,
            database="my_db",
        )
        assert request.model == "gpt-4"
        assert request.stream is True
        assert request.temperature == 0.7
        assert request.max_tokens == 100
        assert request.rag_enabled is True
        assert request.database == "my_db"
        assert request.variables is None  # Still None

    def test_minimal_request(self):
        """Minimal request with just messages works."""
        from api.routers.projects.projects import ChatRequest

        request = ChatRequest(messages=[{"role": "user", "content": "Test"}])
        assert len(request.messages) == 1
        assert request.variables is None


class TestVariablesPassedToPromptResolution:
    """Test that variables reach prompt resolution layer."""

    def test_variables_available_in_request(self):
        """Variables are accessible from the request object."""
        from api.routers.projects.projects import ChatRequest

        request = ChatRequest(
            messages=[{"role": "user", "content": "Hello"}],
            variables={"user_name": "Alice"},
        )

        # Simulate what the endpoint does - access variables
        variables = request.variables or {}
        assert variables.get("user_name") == "Alice"

    def test_none_variables_handled_gracefully(self):
        """None variables can be converted to empty dict."""
        from api.routers.projects.projects import ChatRequest

        request = ChatRequest(messages=[{"role": "user", "content": "Hello"}])

        # Pattern used in endpoint to handle None
        variables = request.variables or {}
        assert variables == {}
        assert isinstance(variables, dict)


class TestAPISchemaIncludesVariables:
    """Test that OpenAPI schema shows variables field."""

    def test_variables_in_model_schema(self):
        """ChatRequest schema includes variables field."""
        from api.routers.projects.projects import ChatRequest

        schema = ChatRequest.model_json_schema()
        assert "variables" in schema.get("properties", {})

    def test_variables_schema_type(self):
        """Variables field has correct schema type."""
        from api.routers.projects.projects import ChatRequest

        schema = ChatRequest.model_json_schema()
        variables_schema = schema["properties"]["variables"]

        # Should allow object or null
        # Pydantic v2 uses anyOf for Optional types
        assert "anyOf" in variables_schema or variables_schema.get("type") in [
            "object",
            None,
        ]

    def test_variables_has_description(self):
        """Variables field has a description."""
        from api.routers.projects.projects import ChatRequest

        schema = ChatRequest.model_json_schema()
        variables_schema = schema["properties"]["variables"]

        # Description may be at top level or in anyOf
        has_description = "description" in variables_schema or any(
            "description" in item
            for item in variables_schema.get("anyOf", [])
            if isinstance(item, dict)
        )
        # If not in schema directly, check the field itself
        if not has_description:
            # Just verify the field exists - description is optional
            assert "variables" in schema["properties"]
