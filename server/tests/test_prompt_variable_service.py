"""Tests for PromptVariableService."""

from services.prompt_variable_service import PromptVariableService


class TestPromptVariableService:
    """Tests for PromptVariableService."""

    def test_substitute_single_variable(self):
        """Test substituting a single variable."""
        content = "Hello {{name}}, welcome!"
        variables = {"name": "Alice"}

        result = PromptVariableService.substitute(content, variables)

        assert result == "Hello Alice, welcome!"

    def test_substitute_multiple_variables(self):
        """Test substituting multiple variables."""
        content = "You are {{role}}, an expert in {{domain}}."
        variables = {"role": "DataBot", "domain": "analytics"}

        result = PromptVariableService.substitute(content, variables)

        assert result == "You are DataBot, an expert in analytics."

    def test_substitute_missing_variable_becomes_empty(self):
        """Test that missing variables become empty strings."""
        content = "Hello {{name}}, your role is {{role}}."
        variables = {"name": "Bob"}

        result = PromptVariableService.substitute(content, variables)

        assert result == "Hello Bob, your role is ."

    def test_substitute_no_variables(self):
        """Test content without variables passes through unchanged."""
        content = "Hello world!"
        variables = {"name": "Alice"}

        result = PromptVariableService.substitute(content, variables)

        assert result == "Hello world!"

    def test_substitute_empty_variables_dict(self):
        """Test with empty variables dict - all variables become empty."""
        content = "Hello {{name}}!"
        variables = {}

        result = PromptVariableService.substitute(content, variables)

        assert result == "Hello !"

    def test_substitute_special_chars_in_value(self):
        """Test that special characters in values are preserved."""
        content = "The command is: {{cmd}}"
        variables = {"cmd": "rm -rf /tmp/*"}

        result = PromptVariableService.substitute(content, variables)

        assert result == "The command is: rm -rf /tmp/*"

    def test_substitute_multiline_content(self):
        """Test substitution in multiline content."""
        content = """You are {{name}}.
Your expertise: {{expertise}}
Style: {{style}}"""
        variables = {
            "name": "Assistant",
            "expertise": "coding",
            "style": "helpful",
        }

        result = PromptVariableService.substitute(content, variables)

        expected = """You are Assistant.
Your expertise: coding
Style: helpful"""
        assert result == expected

    def test_substitute_messages_basic(self):
        """Test substituting variables in a list of messages."""
        messages = [
            {"role": "system", "content": "You are {{persona}}."},
            {"role": "user", "content": "Hello!"},
        ]
        model_defaults = {"persona": "a helpful assistant"}

        result = PromptVariableService.substitute_messages(
            messages, model_defaults, None
        )

        assert result[0]["content"] == "You are a helpful assistant."
        assert result[1]["content"] == "Hello!"

    def test_substitute_messages_request_overrides_model(self):
        """Test that request variables override model defaults."""
        messages = [
            {"role": "system", "content": "You are {{persona}} with {{style}} style."},
        ]
        model_defaults = {"persona": "Alice", "style": "formal"}
        request_overrides = {"style": "casual"}

        result = PromptVariableService.substitute_messages(
            messages, model_defaults, request_overrides
        )

        assert result[0]["content"] == "You are Alice with casual style."

    def test_substitute_messages_preserves_other_fields(self):
        """Test that non-content fields are preserved."""
        messages = [
            {"role": "system", "content": "Hello {{name}}", "name": "system_msg"},
        ]
        variables = {"name": "World"}

        result = PromptVariableService.substitute_messages(messages, variables, None)

        assert result[0]["role"] == "system"
        assert result[0]["name"] == "system_msg"
        assert result[0]["content"] == "Hello World"

    def test_substitute_messages_no_variables(self):
        """Test with no variables provided."""
        messages = [
            {"role": "user", "content": "Hello world"},
        ]

        result = PromptVariableService.substitute_messages(messages, None, None)

        assert result == messages

    def test_substitute_messages_non_string_content(self):
        """Test that non-string content is not modified."""
        messages = [
            {"role": "user", "content": ["list", "content"]},
        ]

        result = PromptVariableService.substitute_messages(
            messages, {"name": "test"}, None
        )

        assert result[0]["content"] == ["list", "content"]

    def test_find_variables_basic(self):
        """Test finding variables in content."""
        content = "Hello {{name}}, you are {{role}}."

        result = PromptVariableService.find_variables(content)

        assert set(result) == {"name", "role"}

    def test_find_variables_none(self):
        """Test finding no variables."""
        content = "Hello world!"

        result = PromptVariableService.find_variables(content)

        assert result == []

    def test_find_variables_duplicate(self):
        """Test that duplicate variables are all found."""
        content = "{{name}} and {{name}} again"

        result = PromptVariableService.find_variables(content)

        assert result == ["name", "name"]

    def test_find_variables_in_messages(self):
        """Test finding variables across multiple messages."""
        messages = [
            {"role": "system", "content": "You are {{persona}}."},
            {"role": "user", "content": "Set {{style}} mode."},
        ]

        result = PromptVariableService.find_variables_in_messages(messages)

        assert result == ["persona", "style"]

    def test_find_variables_in_messages_deduplicates(self):
        """Test that find_variables_in_messages returns unique names."""
        messages = [
            {"role": "system", "content": "{{name}} is {{name}}."},
            {"role": "user", "content": "Hello {{name}}!"},
        ]

        result = PromptVariableService.find_variables_in_messages(messages)

        assert result == ["name"]

    def test_variable_pattern_alphanumeric_underscore(self):
        """Test that variable names can contain alphanumeric and underscore."""
        content = "{{var_1}} and {{VAR_2}} and {{var123}}"
        variables = {"var_1": "a", "VAR_2": "b", "var123": "c"}

        result = PromptVariableService.substitute(content, variables)

        assert result == "a and b and c"
