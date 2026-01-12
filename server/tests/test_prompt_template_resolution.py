"""Tests for prompt template resolution.

Tests that PromptService resolves template variables in prompt messages.
"""

import pytest
from config.datamodel import LlamaFarmConfig, Model, PromptMessage, PromptSet

from services.prompt_service import PromptService


def create_config_with_prompts(prompts: list[PromptSet]) -> LlamaFarmConfig:
    """Helper to create a config with prompts."""
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
                }
            ]
        },
        prompts=prompts,
    )


def create_model(prompts: list[str] | None = None) -> Model:
    """Helper to create a model config."""
    return Model(
        name="test-model",
        provider="universal",
        model="test",
        prompts=prompts or [],
    )


class TestPromptWithVariableResolved:
    """Test prompt content with {{variable}} gets resolved."""

    def test_simple_variable_in_prompt(self):
        """Prompt content with {{user_name}} gets resolved."""
        prompts = [
            PromptSet(
                name="greeting",
                messages=[
                    PromptMessage(role="system", content="Hello {{user_name}}, how can I help?")
                ],
            )
        ]
        config = create_config_with_prompts(prompts)
        model = create_model(["greeting"])

        result = PromptService.resolve_prompts_for_model(
            config, model, variables={"user_name": "Alice"}
        )

        assert len(result) == 1
        assert result[0].content == "Hello Alice, how can I help?"

    def test_multiple_variables_in_prompt(self):
        """Multiple variables in same prompt resolve."""
        prompts = [
            PromptSet(
                name="intro",
                messages=[
                    PromptMessage(
                        role="system",
                        content="Welcome {{user_name}} from {{company}}!",
                    )
                ],
            )
        ]
        config = create_config_with_prompts(prompts)
        model = create_model(["intro"])

        result = PromptService.resolve_prompts_for_model(
            config, model, variables={"user_name": "Bob", "company": "Acme Corp"}
        )

        assert result[0].content == "Welcome Bob from Acme Corp!"

    def test_variable_in_multiline_prompt(self):
        """Variables in multiline prompts resolve."""
        prompts = [
            PromptSet(
                name="system",
                messages=[
                    PromptMessage(
                        role="system",
                        content="""You are an assistant for {{company}}.
Your role is {{role}}.
Please help the user with their questions.""",
                    )
                ],
            )
        ]
        config = create_config_with_prompts(prompts)
        model = create_model(["system"])

        result = PromptService.resolve_prompts_for_model(
            config, model, variables={"company": "TechCorp", "role": "support agent"}
        )

        assert "TechCorp" in result[0].content
        assert "support agent" in result[0].content


class TestPromptWithDefaultUsed:
    """Test {{variable | default}} uses default when not provided."""

    def test_default_used_when_not_provided(self):
        """Default value used when variable not in variables dict."""
        prompts = [
            PromptSet(
                name="greeting",
                messages=[
                    PromptMessage(role="system", content="Hello {{name | Guest}}!")
                ],
            )
        ]
        config = create_config_with_prompts(prompts)
        model = create_model(["greeting"])

        result = PromptService.resolve_prompts_for_model(config, model, variables={})

        assert result[0].content == "Hello Guest!"

    def test_provided_value_overrides_default(self):
        """Provided value used instead of default."""
        prompts = [
            PromptSet(
                name="greeting",
                messages=[
                    PromptMessage(role="system", content="Hello {{name | Guest}}!")
                ],
            )
        ]
        config = create_config_with_prompts(prompts)
        model = create_model(["greeting"])

        result = PromptService.resolve_prompts_for_model(
            config, model, variables={"name": "Alice"}
        )

        assert result[0].content == "Hello Alice!"

    def test_mixed_with_and_without_defaults(self):
        """Mix of variables with and without defaults."""
        prompts = [
            PromptSet(
                name="mixed",
                messages=[
                    PromptMessage(
                        role="system",
                        content="User: {{user_name}}, Role: {{role | member}}, Tier: {{tier | basic}}",
                    )
                ],
            )
        ]
        config = create_config_with_prompts(prompts)
        model = create_model(["mixed"])

        result = PromptService.resolve_prompts_for_model(
            config, model, variables={"user_name": "Alice", "tier": "premium"}
        )

        assert result[0].content == "User: Alice, Role: member, Tier: premium"


class TestMultiplePromptsResolved:
    """Test variables resolved across multiple prompt sets."""

    def test_variables_in_multiple_prompt_sets(self):
        """Variables resolved in all prompt sets."""
        prompts = [
            PromptSet(
                name="system",
                messages=[
                    PromptMessage(role="system", content="You work for {{company}}.")
                ],
            ),
            PromptSet(
                name="context",
                messages=[
                    PromptMessage(role="system", content="The user is {{user_name}}.")
                ],
            ),
        ]
        config = create_config_with_prompts(prompts)
        model = create_model(["system", "context"])

        result = PromptService.resolve_prompts_for_model(
            config, model, variables={"company": "Acme", "user_name": "Alice"}
        )

        assert len(result) == 2
        assert result[0].content == "You work for Acme."
        assert result[1].content == "The user is Alice."

    def test_same_variable_in_multiple_prompts(self):
        """Same variable used across multiple prompts."""
        prompts = [
            PromptSet(
                name="first",
                messages=[
                    PromptMessage(role="system", content="Hello {{name}}!")
                ],
            ),
            PromptSet(
                name="second",
                messages=[
                    PromptMessage(role="system", content="{{name}} is logged in.")
                ],
            ),
        ]
        config = create_config_with_prompts(prompts)
        model = create_model(["first", "second"])

        result = PromptService.resolve_prompts_for_model(
            config, model, variables={"name": "Bob"}
        )

        assert result[0].content == "Hello Bob!"
        assert result[1].content == "Bob is logged in."


class TestStaticPromptUnchanged:
    """Test prompts without variables work unchanged."""

    def test_static_prompt_passes_through(self):
        """Prompt without {{}} markers passes through unchanged."""
        prompts = [
            PromptSet(
                name="static",
                messages=[
                    PromptMessage(
                        role="system", content="You are a helpful assistant."
                    )
                ],
            )
        ]
        config = create_config_with_prompts(prompts)
        model = create_model(["static"])

        result = PromptService.resolve_prompts_for_model(
            config, model, variables={"unused": "value"}
        )

        assert result[0].content == "You are a helpful assistant."

    def test_static_prompt_without_variables_param(self):
        """Static prompt works when variables not provided."""
        prompts = [
            PromptSet(
                name="static",
                messages=[
                    PromptMessage(role="system", content="Hello world")
                ],
            )
        ]
        config = create_config_with_prompts(prompts)
        model = create_model(["static"])

        # Call without variables parameter (uses default None)
        result = PromptService.resolve_prompts_for_model(config, model)

        assert result[0].content == "Hello world"

    def test_single_braces_not_interpreted(self):
        """Single braces {like this} pass through unchanged."""
        prompts = [
            PromptSet(
                name="json",
                messages=[
                    PromptMessage(
                        role="system",
                        content='Output JSON: {"key": "value"}',
                    )
                ],
            )
        ]
        config = create_config_with_prompts(prompts)
        model = create_model(["json"])

        result = PromptService.resolve_prompts_for_model(config, model, variables={})

        assert result[0].content == 'Output JSON: {"key": "value"}'


class TestPromptResolutionWithEmptyVariables:
    """Test empty {} variables dict works correctly."""

    def test_empty_dict_with_all_defaults(self):
        """Empty dict works when all variables have defaults."""
        prompts = [
            PromptSet(
                name="defaults",
                messages=[
                    PromptMessage(
                        role="system",
                        content="Role: {{role | user}}, Level: {{level | 1}}",
                    )
                ],
            )
        ]
        config = create_config_with_prompts(prompts)
        model = create_model(["defaults"])

        result = PromptService.resolve_prompts_for_model(config, model, variables={})

        assert result[0].content == "Role: user, Level: 1"

    def test_none_variables_with_defaults(self):
        """None variables treated as empty dict."""
        prompts = [
            PromptSet(
                name="defaults",
                messages=[
                    PromptMessage(role="system", content="Hello {{name | World}}!")
                ],
            )
        ]
        config = create_config_with_prompts(prompts)
        model = create_model(["defaults"])

        result = PromptService.resolve_prompts_for_model(config, model, variables=None)

        assert result[0].content == "Hello World!"

    def test_empty_dict_fails_for_required_var(self):
        """Empty dict fails when required variable missing."""
        from services.template_service import TemplateError

        prompts = [
            PromptSet(
                name="required",
                messages=[
                    PromptMessage(role="system", content="Hello {{name}}!")
                ],
            )
        ]
        config = create_config_with_prompts(prompts)
        model = create_model(["required"])

        with pytest.raises(TemplateError) as exc_info:
            PromptService.resolve_prompts_for_model(config, model, variables={})

        assert "name" in str(exc_info.value)


class TestPromptResolutionPreservesStructure:
    """Test that resolution preserves message structure."""

    def test_role_preserved(self):
        """Message role is preserved after resolution."""
        prompts = [
            PromptSet(
                name="test",
                messages=[
                    PromptMessage(role="system", content="{{msg}}"),
                    PromptMessage(role="user", content="{{user_msg}}"),
                    PromptMessage(role="assistant", content="{{assist_msg}}"),
                ],
            )
        ]
        config = create_config_with_prompts(prompts)
        model = create_model(["test"])

        result = PromptService.resolve_prompts_for_model(
            config,
            model,
            variables={"msg": "system", "user_msg": "user", "assist_msg": "assistant"},
        )

        assert result[0].role == "system"
        assert result[1].role == "user"
        assert result[2].role == "assistant"

    def test_tool_call_id_preserved(self):
        """Tool call ID is preserved after resolution."""
        prompts = [
            PromptSet(
                name="tool",
                messages=[
                    PromptMessage(
                        role="tool",
                        content="Result: {{result}}",
                        tool_call_id="call_123",
                    )
                ],
            )
        ]
        config = create_config_with_prompts(prompts)
        model = create_model(["tool"])

        result = PromptService.resolve_prompts_for_model(
            config, model, variables={"result": "success"}
        )

        assert result[0].tool_call_id == "call_123"
        assert result[0].content == "Result: success"
