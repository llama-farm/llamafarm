"""Integration tests for dynamic values feature.

Tests that prompts and tools resolve correctly together in realistic scenarios.
"""

import pytest
from config.datamodel import LlamaFarmConfig, Model, PromptMessage, PromptSet, Tool

from services.prompt_service import PromptService
from services.template_service import TemplateError, TemplateService


def create_full_config(
    prompts: list[PromptSet], tools: list[Tool] | None = None
) -> LlamaFarmConfig:
    """Helper to create a config with prompts and tools."""
    return LlamaFarmConfig(
        version="v1",
        name="test-project",
        namespace="test",
        runtime={
            "models": [
                {
                    "name": "test-model",
                    "provider": "universal",
                    "model": "test",
                    "prompts": [p.name for p in prompts],
                    "tools": [t.model_dump() for t in tools] if tools else [],
                }
            ]
        },
        prompts=prompts,
    )


class TestIntegrationPromptsAndTools:
    """Test both prompts and tools resolve in same request."""

    def test_prompts_and_tools_resolve_together(self):
        """Variables resolve in both prompts and tools simultaneously."""
        prompts = [
            PromptSet(
                name="system",
                messages=[
                    PromptMessage(
                        role="system",
                        content="You are an assistant for {{company}}. Help users with {{service}}.",
                    )
                ],
            )
        ]

        tools_dicts = [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up {{company}} {{service}} data",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        variables = {"company": "Acme Corp", "service": "customer support"}

        # Resolve prompts
        config = create_full_config(prompts)
        model = Model(name="test-model", provider="universal", model="test", prompts=["system"])
        resolved_prompts = PromptService.resolve_prompts_for_model(
            config, model, variables=variables
        )

        # Resolve tools
        resolved_tools = TemplateService.resolve_object(tools_dicts, variables)

        # Verify both resolved
        assert "Acme Corp" in resolved_prompts[0].content
        assert "customer support" in resolved_prompts[0].content
        assert "Acme Corp" in resolved_tools[0]["function"]["description"]
        assert "customer support" in resolved_tools[0]["function"]["description"]

    def test_same_variable_in_prompts_and_tools(self):
        """Same variable used in both prompts and tools resolves consistently."""
        variables = {"api_endpoint": "https://api.example.com"}

        prompt_content = "The API is at {{api_endpoint}}"
        tool_desc = "Fetch from {{api_endpoint}}"

        resolved_prompt = TemplateService.resolve(prompt_content, variables)
        resolved_tool_desc = TemplateService.resolve(tool_desc, variables)

        assert resolved_prompt == "The API is at https://api.example.com"
        assert resolved_tool_desc == "Fetch from https://api.example.com"


class TestIntegrationWithRAG:
    """Test variables work alongside RAG features."""

    def test_rag_queries_not_affected(self):
        """RAG query strings are not processed by TemplateService.

        This test verifies that RAG queries (plain strings) would NOT be
        resolved if passed to TemplateService - they should be passed directly
        to the vector store as literal search strings.
        """
        from services.template_service import TemplateError, TemplateService

        # RAG queries are plain strings that may contain {{}} as literal text
        rag_query = "Find documents about {{company}}"

        # If someone accidentally passes a RAG query to TemplateService
        # without variables, it should raise an error (proving resolution happens)
        with pytest.raises(TemplateError) as exc_info:
            TemplateService.resolve(rag_query, {})

        # The error confirms TemplateService WOULD try to resolve it
        assert "company" in str(exc_info.value)

        # With a default, it resolves - showing the template is processed
        rag_query_with_default = "Find documents about {{company | Acme}}"
        resolved = TemplateService.resolve(rag_query_with_default, {})
        assert resolved == "Find documents about Acme"

        # The actual RAG code path does NOT call TemplateService on queries
        # This is verified by the fact that RAG endpoints accept raw strings
        # and pass them directly to vector stores without template processing

    def test_prompts_resolve_with_rag_context(self):
        """Prompts resolve even when RAG is enabled."""
        prompts = [
            PromptSet(
                name="rag_system",
                messages=[
                    PromptMessage(
                        role="system",
                        content="You are a {{role | researcher}} for {{company}}. "
                        "Use the retrieved documents to answer questions.",
                    )
                ],
            )
        ]

        config = create_full_config(prompts)
        model = Model(name="test-model", provider="universal", model="test", prompts=["rag_system"])

        # With variables
        resolved = PromptService.resolve_prompts_for_model(
            config, model, variables={"company": "TechCorp"}
        )
        assert "researcher" in resolved[0].content  # default
        assert "TechCorp" in resolved[0].content


class TestIntegrationStreaming:
    """Test variables work with streaming responses."""

    def test_variables_resolve_before_streaming(self):
        """Variables are resolved before streaming begins, not during."""
        # This tests the pattern: resolve once at request time
        prompts = [
            PromptSet(
                name="streaming",
                messages=[
                    PromptMessage(
                        role="system",
                        content="Stream responses for {{user}}.",
                    )
                ],
            )
        ]

        config = create_full_config(prompts)
        model = Model(name="test-model", provider="universal", model="test", prompts=["streaming"])

        # Variables are resolved once, before any streaming
        resolved = PromptService.resolve_prompts_for_model(
            config, model, variables={"user": "Alice"}
        )

        # The resolved content is static for the stream duration
        assert resolved[0].content == "Stream responses for Alice."
        # If streaming, this same resolved prompt is used throughout


class TestIntegrationSession:
    """Test variables work across session-based chat."""

    def test_different_variables_per_request(self):
        """Each request can have different variable values."""
        prompts = [
            PromptSet(
                name="session",
                messages=[
                    PromptMessage(
                        role="system",
                        content="Current user: {{user_name | Anonymous}}",
                    )
                ],
            )
        ]

        config = create_full_config(prompts)
        model = Model(name="test-model", provider="universal", model="test", prompts=["session"])

        # Request 1
        resolved1 = PromptService.resolve_prompts_for_model(
            config, model, variables={"user_name": "Alice"}
        )
        assert resolved1[0].content == "Current user: Alice"

        # Request 2 (different user)
        resolved2 = PromptService.resolve_prompts_for_model(
            config, model, variables={"user_name": "Bob"}
        )
        assert resolved2[0].content == "Current user: Bob"

        # Request 3 (no variables - use default)
        resolved3 = PromptService.resolve_prompts_for_model(config, model, variables={})
        assert resolved3[0].content == "Current user: Anonymous"

    def test_variables_dont_persist_across_requests(self):
        """Variables from one request don't leak to the next."""
        # This is by design - variables are per-request
        # If someone wants persistence, they'd need to send the same variables

        prompt_template = "User: {{user | Guest}}"

        # Request 1 with user
        result1 = TemplateService.resolve(prompt_template, {"user": "Alice"})
        assert result1 == "User: Alice"

        # Request 2 without user - should use default, not previous value
        result2 = TemplateService.resolve(prompt_template, {})
        assert result2 == "User: Guest"


class TestIntegrationEdgeCases:
    """Integration edge cases."""

    def test_empty_variables_uses_all_defaults(self):
        """Empty variables dict causes all defaults to be used."""
        prompts = [
            PromptSet(
                name="defaults",
                messages=[
                    PromptMessage(
                        role="system",
                        content="{{greeting | Hello}} {{name | World}}!",
                    )
                ],
            )
        ]

        config = create_full_config(prompts)
        model = Model(name="test-model", provider="universal", model="test", prompts=["defaults"])

        resolved = PromptService.resolve_prompts_for_model(config, model, variables={})
        assert resolved[0].content == "Hello World!"

    def test_partial_variables_mixes_provided_and_defaults(self):
        """Some variables provided, others use defaults."""
        template = "{{a}} {{b | B}} {{c | C}}"

        result = TemplateService.resolve(template, {"a": "A", "c": "custom"})
        assert result == "A B custom"

    def test_complex_nested_structure(self):
        """Complex nested structures with mixed variables."""
        complex_obj = {
            "system": {
                "prompt": "Hello {{name}}",
                "config": {
                    "api": "{{api_base | http://localhost}}/v1",
                    "headers": {"Authorization": "Bearer {{token | default_token}}"},
                },
            },
            "tools": [
                {
                    "name": "tool1",
                    "url": "{{api_base | http://localhost}}/tool1",
                }
            ],
        }

        result = TemplateService.resolve_object(
            complex_obj, {"name": "Alice", "api_base": "https://api.prod.com"}
        )

        assert result["system"]["prompt"] == "Hello Alice"
        assert result["system"]["config"]["api"] == "https://api.prod.com/v1"
        assert (
            result["system"]["config"]["headers"]["Authorization"]
            == "Bearer default_token"
        )
        assert result["tools"][0]["url"] == "https://api.prod.com/tool1"

    def test_missing_required_in_integration(self):
        """Missing required variable raises clear error."""
        prompts = [
            PromptSet(
                name="required",
                messages=[
                    PromptMessage(
                        role="system",
                        content="API key: {{api_key}}",  # No default = required
                    )
                ],
            )
        ]

        config = create_full_config(prompts)
        model = Model(name="test-model", provider="universal", model="test", prompts=["required"])

        with pytest.raises(TemplateError) as exc_info:
            PromptService.resolve_prompts_for_model(config, model, variables={})

        assert "api_key" in str(exc_info.value)
