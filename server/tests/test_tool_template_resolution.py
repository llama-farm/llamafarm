"""Tests for tool template resolution.

Tests that TemplateService resolves variables in tool definitions.
"""

import pytest

from services.template_service import TemplateError, TemplateService


class TestToolDescriptionResolved:
    """Test tool description with variable is resolved."""

    def test_simple_description_variable(self):
        """Tool description with {{variable}} gets resolved."""
        tool = {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the {{service_name}} database",
                "parameters": {"type": "object", "properties": {}},
            },
        }

        result = TemplateService.resolve_object(tool, {"service_name": "ProductCatalog"})

        assert result["function"]["description"] == "Search the ProductCatalog database"

    def test_description_with_default(self):
        """Tool description uses default when variable not provided."""
        tool = {
            "type": "function",
            "function": {
                "name": "api_call",
                "description": "Call the {{api_name | External API}} endpoint",
                "parameters": {"type": "object", "properties": {}},
            },
        }

        result = TemplateService.resolve_object(tool, {})

        assert result["function"]["description"] == "Call the External API endpoint"


class TestToolParameterDefaultResolved:
    """Test tool parameter default with variable is resolved."""

    def test_parameter_description_variable(self):
        """Parameter description with variable gets resolved."""
        tool = {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for {{database_name}}",
                        }
                    },
                },
            },
        }

        result = TemplateService.resolve_object(tool, {"database_name": "users"})

        assert (
            result["function"]["parameters"]["properties"]["query"]["description"]
            == "Search query for users"
        )

    def test_parameter_default_value_variable(self):
        """Parameter default value with variable gets resolved."""
        tool = {
            "type": "function",
            "function": {
                "name": "fetch",
                "description": "Fetch data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "default": "{{api_base}}/data",
                        }
                    },
                },
            },
        }

        result = TemplateService.resolve_object(
            tool, {"api_base": "https://api.example.com"}
        )

        assert (
            result["function"]["parameters"]["properties"]["url"]["default"]
            == "https://api.example.com/data"
        )


class TestConfigToolsResolved:
    """Test tools from config get variables resolved."""

    def test_list_of_tools_resolved(self):
        """Multiple tools in a list all get resolved."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "tool1",
                    "description": "Tool for {{company}}",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "tool2",
                    "description": "Another tool for {{company}}",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

        result = TemplateService.resolve_object(tools, {"company": "Acme"})

        assert result[0]["function"]["description"] == "Tool for Acme"
        assert result[1]["function"]["description"] == "Another tool for Acme"

    def test_deeply_nested_parameters_resolved(self):
        """Variables in deeply nested parameter schemas get resolved."""
        tool = {
            "type": "function",
            "function": {
                "name": "complex",
                "description": "Complex tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "config": {
                            "type": "object",
                            "properties": {
                                "endpoint": {
                                    "type": "string",
                                    "description": "Endpoint at {{base_url}}",
                                    "default": "{{base_url}}/api",
                                }
                            },
                        }
                    },
                },
            },
        }

        result = TemplateService.resolve_object(
            tool, {"base_url": "https://example.com"}
        )

        nested = result["function"]["parameters"]["properties"]["config"]["properties"][
            "endpoint"
        ]
        assert nested["description"] == "Endpoint at https://example.com"
        assert nested["default"] == "https://example.com/api"


class TestRequestToolsResolved:
    """Test tools from request body get variables resolved."""

    def test_openai_format_tool_resolved(self):
        """OpenAI-format tool from request gets resolved."""
        request_tool = {
            "type": "function",
            "function": {
                "name": "custom_api",
                "description": "Call {{api_name}} with {{method | GET}} request",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path relative to {{base_url}}",
                        }
                    },
                    "required": ["path"],
                },
            },
        }

        result = TemplateService.resolve_object(
            request_tool, {"api_name": "UserService", "base_url": "https://api.local"}
        )

        assert (
            result["function"]["description"]
            == "Call UserService with GET request"
        )
        assert (
            result["function"]["parameters"]["properties"]["path"]["description"]
            == "Path relative to https://api.local"
        )


class TestStaticToolsUnchanged:
    """Test tools without variables work unchanged."""

    def test_static_tool_passes_through(self):
        """Tool without {{}} markers passes through unchanged."""
        tool = {
            "type": "function",
            "function": {
                "name": "static_tool",
                "description": "A static tool description",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string", "description": "Input value"}
                    },
                },
            },
        }

        result = TemplateService.resolve_object(tool, {"unused": "value"})

        assert result == tool

    def test_tool_with_braces_in_json_schema(self):
        """Tool with JSON schema braces (not templates) works."""
        tool = {
            "type": "function",
            "function": {
                "name": "json_tool",
                "description": "Returns JSON like {key: value}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "description": "Object with {nested: structure}",
                        }
                    },
                },
            },
        }

        result = TemplateService.resolve_object(tool, {})

        assert result["function"]["description"] == "Returns JSON like {key: value}"


class TestToolResolutionEdgeCases:
    """Edge cases for tool resolution."""

    def test_missing_required_variable_raises(self):
        """Missing required variable in tool raises error."""
        tool = {
            "type": "function",
            "function": {
                "name": "api",
                "description": "Call {{api_name}}",
                "parameters": {"type": "object", "properties": {}},
            },
        }

        with pytest.raises(TemplateError) as exc_info:
            TemplateService.resolve_object(tool, {})

        assert "api_name" in str(exc_info.value)

    def test_non_string_values_preserved(self):
        """Non-string values in tool schema are preserved."""
        tool = {
            "type": "function",
            "function": {
                "name": "typed",
                "description": "Tool for {{service}}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer", "minimum": 0, "maximum": 100},
                        "enabled": {"type": "boolean", "default": True},
                    },
                    "required": ["count"],
                },
            },
        }

        result = TemplateService.resolve_object(tool, {"service": "test"})

        props = result["function"]["parameters"]["properties"]
        assert props["count"]["minimum"] == 0
        assert props["count"]["maximum"] == 100
        assert props["enabled"]["default"] is True
        assert result["function"]["parameters"]["required"] == ["count"]
