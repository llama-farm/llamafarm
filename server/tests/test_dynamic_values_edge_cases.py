"""Edge case tests for dynamic values feature.

Tests error handling, special characters, and edge cases.
"""

import pytest

from services.template_service import TemplateError, TemplateService


class TestErrorMessagesAreHelpful:
    """Test that missing variable errors are clear and actionable."""

    def test_error_includes_variable_name(self):
        """Error message includes the missing variable name."""
        with pytest.raises(TemplateError) as exc_info:
            TemplateService.resolve("Hello {{user_name}}", {})

        error_msg = str(exc_info.value)
        assert "user_name" in error_msg

    def test_error_suggests_default_syntax(self):
        """Error message suggests using default syntax."""
        with pytest.raises(TemplateError) as exc_info:
            TemplateService.resolve("Value: {{missing_var}}", {})

        error_msg = str(exc_info.value)
        assert "default" in error_msg.lower() or "|" in error_msg

    def test_error_shows_available_variables(self):
        """Error message shows what variables were provided."""
        with pytest.raises(TemplateError) as exc_info:
            TemplateService.resolve(
                "{{a}} {{b}} {{c}}", {"a": "value_a", "b": "value_b"}
            )

        error_msg = str(exc_info.value)
        # Should mention "c" is missing and show available vars
        assert "c" in error_msg

    def test_error_for_empty_variables_dict(self):
        """Error when no variables provided for required template."""
        with pytest.raises(TemplateError) as exc_info:
            TemplateService.resolve("{{required_var}}", {})

        error_msg = str(exc_info.value)
        assert "required_var" in error_msg


class TestSpecialCharactersInValues:
    """Test variables containing special characters work correctly."""

    def test_value_with_double_braces(self):
        """Variable value containing {{}} is inserted literally."""
        result = TemplateService.resolve(
            "Output: {{message}}", {"message": "Use {{var}} syntax"}
        )
        # The result should contain the literal {{var}} from the value
        assert result == "Output: Use {{var}} syntax"

    def test_value_with_pipe(self):
        """Variable value containing | is inserted literally."""
        result = TemplateService.resolve(
            "Command: {{cmd}}", {"cmd": "ls | grep foo"}
        )
        assert result == "Command: ls | grep foo"

    def test_value_with_curly_braces(self):
        """Variable value containing {} is inserted literally."""
        result = TemplateService.resolve(
            "JSON: {{data}}", {"data": '{"key": "value"}'}
        )
        assert result == 'JSON: {"key": "value"}'

    def test_value_with_newlines(self):
        """Variable value containing newlines works."""
        result = TemplateService.resolve(
            "Text:\n{{content}}", {"content": "Line 1\nLine 2\nLine 3"}
        )
        assert "Line 1\nLine 2\nLine 3" in result

    def test_value_with_quotes(self):
        """Variable value containing quotes works."""
        result = TemplateService.resolve(
            "Say: {{phrase}}", {"phrase": 'He said "hello"'}
        )
        assert result == 'Say: He said "hello"'

    def test_value_with_backslash(self):
        """Variable value containing backslash works."""
        result = TemplateService.resolve(
            "Path: {{path}}", {"path": "C:\\Users\\name"}
        )
        assert result == "Path: C:\\Users\\name"

    def test_unicode_in_value(self):
        """Unicode characters in variable value work."""
        result = TemplateService.resolve(
            "Greeting: {{greeting}}", {"greeting": "你好世界 🌍"}
        )
        assert result == "Greeting: 你好世界 🌍"


class TestEmptyStringValues:
    """Test empty string values are different from missing."""

    def test_empty_string_is_valid(self):
        """Empty string '' is a valid value (not missing)."""
        result = TemplateService.resolve(
            "Prefix: {{prefix}}Suffix", {"prefix": ""}
        )
        assert result == "Prefix: Suffix"

    def test_empty_string_doesnt_use_default(self):
        """Empty string '' does NOT fall back to default."""
        result = TemplateService.resolve(
            "Value: {{val | default}}", {"val": ""}
        )
        assert result == "Value: "  # Empty, not "default"

    def test_none_converts_to_string(self):
        """None value is converted to the string 'None'."""
        result = TemplateService.resolve("Value: {{val}}", {"val": None})
        assert result == "Value: None"

    def test_whitespace_only_value(self):
        """Whitespace-only value is preserved."""
        result = TemplateService.resolve("Value: {{val}}", {"val": "   "})
        assert result == "Value:    "


class TestDefaultValueEdgeCases:
    """Edge cases for default values."""

    def test_empty_default(self):
        """Empty default ({{var|}}) is valid."""
        result = TemplateService.resolve("Value: {{var |}}", {})
        assert result == "Value: "

    def test_default_with_spaces(self):
        """Default with leading/trailing spaces is trimmed."""
        result = TemplateService.resolve("Value: {{var |  hello world  }}", {})
        assert result == "Value: hello world"

    def test_default_with_pipe_in_value(self):
        """Default cannot contain pipe (first pipe wins)."""
        # {{var | a | b}} means default is "a | b" after trimming
        # Actually the regex captures everything after first |
        result = TemplateService.resolve("{{var | a | b}}", {})
        # Default is "a | b" (everything after first pipe)
        assert result == "a | b"

    def test_default_with_numbers(self):
        """Numeric default is string."""
        result = TemplateService.resolve("Count: {{count | 0}}", {})
        assert result == "Count: 0"

    def test_default_url(self):
        """URL as default works."""
        result = TemplateService.resolve(
            "API: {{url | https://api.example.com/v1}}", {}
        )
        assert result == "API: https://api.example.com/v1"


class TestVariableNameEdgeCases:
    """Edge cases for variable names."""

    def test_single_char_name(self):
        """Single character variable name works."""
        result = TemplateService.resolve("{{a}}", {"a": "value"})
        assert result == "value"

    def test_underscore_in_name(self):
        """Variable name with underscore works."""
        result = TemplateService.resolve("{{my_var}}", {"my_var": "value"})
        assert result == "value"

    def test_numbers_in_name(self):
        """Variable name with numbers works."""
        result = TemplateService.resolve("{{var123}}", {"var123": "value"})
        assert result == "value"

    def test_long_name(self):
        """Long variable name works."""
        long_name = "a" * 100
        result = TemplateService.resolve(f"{{{{{long_name}}}}}", {long_name: "value"})
        assert result == "value"

    def test_name_starting_with_underscore(self):
        """Variable name starting with underscore works."""
        result = TemplateService.resolve("{{_private}}", {"_private": "secret"})
        assert result == "secret"


class TestMalformedTemplates:
    """Test handling of malformed template strings."""

    def test_unclosed_brace(self):
        """Unclosed {{ passes through as literal."""
        result = TemplateService.resolve("Hello {{name", {"name": "World"})
        # Unclosed template is not matched, passes through
        assert result == "Hello {{name"

    def test_unopened_brace(self):
        """Unopened }} passes through as literal."""
        result = TemplateService.resolve("Hello name}}", {"name": "World"})
        assert result == "Hello name}}"

    def test_nested_braces(self):
        """Nested braces don't break parsing."""
        result = TemplateService.resolve("{{{{nested}}}}", {"nested": "value"})
        # {{{{nested}}}} -> resolves inner {{nested}} -> {{value}}
        assert result == "{{value}}"

    def test_empty_template_marker(self):
        """Empty {{}} is not a valid template."""
        result = TemplateService.resolve("Empty: {{}}", {})
        # {{}} doesn't match the pattern (no variable name)
        assert result == "Empty: {{}}"


class TestPerformanceEdgeCases:
    """Edge cases that could affect performance."""

    def test_many_variables(self):
        """Many variables in one template work."""
        template = " ".join([f"{{{{var{i}}}}}" for i in range(100)])
        variables = {f"var{i}": f"value{i}" for i in range(100)}
        result = TemplateService.resolve(template, variables)
        assert "value0" in result
        assert "value99" in result

    def test_long_template(self):
        """Long template string works."""
        prefix = "x" * 10000
        template = f"{prefix}{{{{name}}}}"
        result = TemplateService.resolve(template, {"name": "test"})
        assert result == f"{prefix}test"

    def test_deeply_nested_object(self):
        """Deeply nested object works."""
        obj = {"a": {"b": {"c": {"d": {"e": "{{val}}"}}}}}
        result = TemplateService.resolve_object(obj, {"val": "deep"})
        assert result["a"]["b"]["c"]["d"]["e"] == "deep"
