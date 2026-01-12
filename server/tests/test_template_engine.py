"""Tests for the template engine service.

Tests the TemplateService class that handles dynamic variable substitution
using {{variable}} and {{variable | default}} syntax.
"""

import pytest

from services.template_service import TemplateError, TemplateService


class TestTemplateSubstitutionBasic:
    """Test basic {{var}} substitution."""

    def test_simple_variable(self):
        """Basic {{var}} substitution works."""
        result = TemplateService.resolve("Hello {{name}}", {"name": "Alice"})
        assert result == "Hello Alice"

    def test_multiple_same_variable(self):
        """Same variable used multiple times."""
        result = TemplateService.resolve(
            "{{name}} says hello to {{name}}", {"name": "Bob"}
        )
        assert result == "Bob says hello to Bob"

    def test_variable_at_start(self):
        """Variable at start of string."""
        result = TemplateService.resolve("{{greeting}} world", {"greeting": "Hello"})
        assert result == "Hello world"

    def test_variable_at_end(self):
        """Variable at end of string."""
        result = TemplateService.resolve("Hello {{name}}", {"name": "world"})
        assert result == "Hello world"

    def test_only_variable(self):
        """String is only a variable."""
        result = TemplateService.resolve("{{message}}", {"message": "Hello world"})
        assert result == "Hello world"


class TestTemplateSubstitutionWithDefault:
    """Test {{var | default_value}} syntax."""

    def test_default_used_when_missing(self):
        """Default value used when variable not provided."""
        result = TemplateService.resolve("Hello {{name | Guest}}", {})
        assert result == "Hello Guest"

    def test_default_not_used_when_provided(self):
        """Provided value used instead of default."""
        result = TemplateService.resolve("Hello {{name | Guest}}", {"name": "Alice"})
        assert result == "Hello Alice"

    def test_default_with_spaces(self):
        """Default value can contain spaces."""
        result = TemplateService.resolve("{{msg | Hello World}}", {})
        assert result == "Hello World"

    def test_default_empty_string(self):
        """Empty default is valid."""
        result = TemplateService.resolve("Value: {{val |}}", {})
        assert result == "Value: "

    def test_default_with_special_chars(self):
        """Default can contain special characters."""
        result = TemplateService.resolve("URL: {{url | http://localhost:3000}}", {})
        assert result == "URL: http://localhost:3000"


class TestTemplateNoSubstitution:
    """Test that static strings pass through unchanged (backwards compatibility)."""

    def test_no_template_markers(self):
        """String without {{}} passes through unchanged."""
        original = "This is a plain string with no variables."
        result = TemplateService.resolve(original, {"unused": "value"})
        assert result == original

    def test_single_braces(self):
        """Single braces are not template markers."""
        original = "JSON: {key: value}"
        result = TemplateService.resolve(original, {})
        assert result == original

    def test_triple_braces(self):
        """Triple braces are not template markers."""
        original = "{{{not_a_var}}}"
        result = TemplateService.resolve(original, {"not_a_var": "test"})
        # Should still resolve the inner {{}} but leave outer braces
        assert result == "{test}"

    def test_empty_string(self):
        """Empty string passes through."""
        result = TemplateService.resolve("", {"var": "value"})
        assert result == ""


class TestTemplateMissingVarNoDefault:
    """Test that missing variables without defaults raise clear errors."""

    def test_missing_var_raises_error(self):
        """Missing variable without default raises TemplateError."""
        with pytest.raises(TemplateError) as exc_info:
            TemplateService.resolve("Hello {{name}}", {})
        assert "name" in str(exc_info.value)
        assert "missing" in str(exc_info.value).lower() or "not found" in str(
            exc_info.value
        ).lower()

    def test_error_includes_variable_name(self):
        """Error message includes the missing variable name."""
        with pytest.raises(TemplateError) as exc_info:
            TemplateService.resolve("{{user_name}} logged in", {})
        assert "user_name" in str(exc_info.value)

    def test_partial_missing_raises(self):
        """Error raised even if some variables are provided."""
        with pytest.raises(TemplateError) as exc_info:
            TemplateService.resolve(
                "{{first}} and {{second}}", {"first": "one"}
            )
        assert "second" in str(exc_info.value)


class TestTemplateMultipleVars:
    """Test multiple variables in one string."""

    def test_two_variables(self):
        """Two different variables in one string."""
        result = TemplateService.resolve(
            "{{greeting}}, {{name}}!", {"greeting": "Hello", "name": "Alice"}
        )
        assert result == "Hello, Alice!"

    def test_many_variables(self):
        """Many variables in one string."""
        template = "{{a}} {{b}} {{c}} {{d}} {{e}}"
        variables = {"a": "1", "b": "2", "c": "3", "d": "4", "e": "5"}
        result = TemplateService.resolve(template, variables)
        assert result == "1 2 3 4 5"

    def test_mixed_with_defaults(self):
        """Mix of variables with and without defaults."""
        result = TemplateService.resolve(
            "{{name}} ({{role | user}})", {"name": "Alice"}
        )
        assert result == "Alice (user)"

    def test_adjacent_variables(self):
        """Variables directly adjacent to each other."""
        result = TemplateService.resolve(
            "{{first}}{{second}}", {"first": "Hello", "second": "World"}
        )
        assert result == "HelloWorld"


class TestTemplateNestedInObject:
    """Test variables in nested dict structures."""

    def test_simple_dict(self):
        """Variables in simple dict values."""
        obj = {"message": "Hello {{name}}"}
        result = TemplateService.resolve_object(obj, {"name": "Alice"})
        assert result == {"message": "Hello Alice"}

    def test_nested_dict(self):
        """Variables in nested dict values."""
        obj = {"outer": {"inner": "Value is {{value}}"}}
        result = TemplateService.resolve_object(obj, {"value": "42"})
        assert result == {"outer": {"inner": "Value is 42"}}

    def test_list_in_dict(self):
        """Variables in list items within dict."""
        obj = {"items": ["{{first}}", "{{second}}"]}
        result = TemplateService.resolve_object(obj, {"first": "A", "second": "B"})
        assert result == {"items": ["A", "B"]}

    def test_mixed_types(self):
        """Non-string values pass through unchanged."""
        obj = {"count": 42, "enabled": True, "name": "{{name}}"}
        result = TemplateService.resolve_object(obj, {"name": "Test"})
        assert result == {"count": 42, "enabled": True, "name": "Test"}

    def test_deeply_nested(self):
        """Variables in deeply nested structures."""
        obj = {"a": {"b": {"c": {"d": "{{deep}}"}}}}
        result = TemplateService.resolve_object(obj, {"deep": "found"})
        assert result == {"a": {"b": {"c": {"d": "found"}}}}

    def test_list_of_dicts(self):
        """Variables in list of dicts."""
        obj = [{"name": "{{name1}}"}, {"name": "{{name2}}"}]
        result = TemplateService.resolve_object(obj, {"name1": "Alice", "name2": "Bob"})
        assert result == [{"name": "Alice"}, {"name": "Bob"}]


class TestTemplateWhitespaceHandling:
    """Test that whitespace variants work correctly."""

    def test_no_whitespace(self):
        """{{var}} works."""
        result = TemplateService.resolve("{{name}}", {"name": "test"})
        assert result == "test"

    def test_whitespace_after_open(self):
        """{{ var}} works."""
        result = TemplateService.resolve("{{ name}}", {"name": "test"})
        assert result == "test"

    def test_whitespace_before_close(self):
        """{{var }} works."""
        result = TemplateService.resolve("{{name }}", {"name": "test"})
        assert result == "test"

    def test_whitespace_both_sides(self):
        """{{ var }} works."""
        result = TemplateService.resolve("{{ name }}", {"name": "test"})
        assert result == "test"

    def test_lots_of_whitespace(self):
        """{{   var   }} works."""
        result = TemplateService.resolve("{{   name   }}", {"name": "test"})
        assert result == "test"

    def test_whitespace_around_pipe(self):
        """{{ var | default }} works."""
        result = TemplateService.resolve("{{ name | Guest }}", {})
        assert result == "Guest"

    def test_no_whitespace_around_pipe(self):
        """{{var|default}} works."""
        result = TemplateService.resolve("{{name|Guest}}", {})
        assert result == "Guest"

    def test_tabs_and_newlines(self):
        """Tabs and newlines in template markers are handled."""
        result = TemplateService.resolve("{{\tname\n}}", {"name": "test"})
        assert result == "test"


class TestTemplateEdgeCases:
    """Additional edge case tests."""

    def test_none_value(self):
        """None value becomes empty string (not 'None' literal)."""
        result = TemplateService.resolve("Value: {{val}}", {"val": None})
        assert result == "Value: "

    def test_numeric_value(self):
        """Numeric value is converted to string."""
        result = TemplateService.resolve("Count: {{count}}", {"count": 42})
        assert result == "Count: 42"

    def test_boolean_value(self):
        """Boolean value is converted to string."""
        result = TemplateService.resolve("Enabled: {{enabled}}", {"enabled": True})
        assert result == "Enabled: True"

    def test_multiline_template(self):
        """Template spanning multiple lines."""
        template = """Hello {{name}},
Welcome to {{service}}.
Your role is {{role | user}}."""
        result = TemplateService.resolve(
            template, {"name": "Alice", "service": "LlamaFarm"}
        )
        assert result == """Hello Alice,
Welcome to LlamaFarm.
Your role is user."""

    def test_unicode_variable_value(self):
        """Unicode characters in variable value."""
        result = TemplateService.resolve("Hello {{name}}", {"name": "世界"})
        assert result == "Hello 世界"

    def test_empty_variables_dict(self):
        """Empty variables dict with defaults."""
        result = TemplateService.resolve("{{name | Guest}}", {})
        assert result == "Guest"

    def test_variable_name_with_underscore(self):
        """Variable names can contain underscores."""
        result = TemplateService.resolve("{{user_name}}", {"user_name": "alice"})
        assert result == "alice"

    def test_variable_name_with_numbers(self):
        """Variable names can contain numbers."""
        result = TemplateService.resolve("{{var1}}", {"var1": "value"})
        assert result == "value"
