"""
Tests for Jinja2 template utilities.
"""

import pytest
from jinja2 import TemplateError

from utils.jinja_tools import (
    create_jinja_environment,
    render_chat_with_tools,
    supports_native_tools,
)


class TestCreateJinjaEnvironment:
    """Tests for create_jinja_environment function."""

    def test_environment_has_namespace_function(self):
        """Test that environment includes namespace function for template use."""
        env = create_jinja_environment()
        assert "namespace" in env.globals

    def test_environment_has_raise_exception_function(self):
        """Test that environment includes raise_exception function."""
        env = create_jinja_environment()
        assert "raise_exception" in env.globals

    def test_environment_has_tojson_filter(self):
        """Test that environment includes tojson filter."""
        env = create_jinja_environment()
        assert "tojson" in env.filters

    def test_namespace_supports_attribute_assignment(self):
        """Test that namespace() supports attribute assignment in templates.

        This is critical for templates like Qwen3 that use:
        {% set ns = namespace(found=false) %}
        {% set ns.found = true %}
        """
        env = create_jinja_environment()
        template = env.from_string(
            """
{% set ns = namespace(found=false) %}
{% for item in items %}
{% if item == 'target' %}{% set ns.found = true %}{% endif %}
{% endfor %}
Found: {{ ns.found }}
            """.strip()
        )

        # Test with target present
        result = template.render(items=["a", "target", "b"])
        assert "Found: True" in result

        # Test with target absent
        result = template.render(items=["a", "b", "c"])
        assert "Found: False" in result

    def test_namespace_multiple_attributes(self):
        """Test namespace with multiple attributes."""
        env = create_jinja_environment()
        template = env.from_string(
            """
{% set ns = namespace(count=0, last_item='') %}
{% for item in items %}
{% set ns.count = ns.count + 1 %}
{% set ns.last_item = item %}
{% endfor %}
Count: {{ ns.count }}, Last: {{ ns.last_item }}
            """.strip()
        )

        result = template.render(items=["apple", "banana", "cherry"])
        assert "Count: 3" in result
        assert "Last: cherry" in result

    def test_tojson_filter_basic(self):
        """Test tojson filter converts to JSON string."""
        env = create_jinja_environment()
        template = env.from_string("{{ data | tojson }}")

        result = template.render(data={"key": "value"})
        assert result == '{"key": "value"}'

    def test_tojson_filter_with_unicode(self):
        """Test tojson filter handles unicode correctly."""
        env = create_jinja_environment()
        template = env.from_string("{{ data | tojson }}")

        result = template.render(data={"emoji": "😀"})
        # ensure_ascii=False should preserve unicode
        assert "😀" in result

    def test_raise_exception_function(self):
        """Test that raise_exception raises TemplateError."""
        env = create_jinja_environment()
        template = env.from_string("{{ raise_exception('Test error') }}")

        with pytest.raises(TemplateError, match="Test error"):
            template.render()


class TestSupportsNativeTools:
    """Tests for supports_native_tools function."""

    def test_detects_tools_in_template(self):
        """Test detection of tools reference in template."""
        template = """
        {% if tools %}
        {{ tools | tojson }}
        {% endif %}
        """
        assert supports_native_tools(template) is True

    def test_no_tools_reference(self):
        """Test template without tools reference."""
        template = """
        {{ messages[0].content }}
        """
        assert supports_native_tools(template) is False

    def test_tools_in_loop(self):
        """Test template with tools in for loop."""
        template = """
        {% for tool in tools %}
        {{ tool.function.name }}
        {% endfor %}
        """
        assert supports_native_tools(template) is True


class TestRenderChatWithTools:
    """Tests for render_chat_with_tools function."""

    def test_render_simple_messages(self):
        """Test rendering simple messages without tools."""
        template = """
{% for message in messages %}
{{ message.role }}: {{ message.content }}
{% endfor %}
{% if add_generation_prompt %}assistant: {% endif %}
        """.strip()

        messages = [
            {"role": "user", "content": "Hello"},
        ]

        result = render_chat_with_tools(
            template=template,
            messages=messages,
            tools=None,
            add_generation_prompt=True,
        )

        assert "user: Hello" in result
        assert "assistant:" in result

    def test_render_with_tools(self):
        """Test rendering with tools present."""
        template = """
{% if tools %}
Available tools:
{% for tool in tools %}
- {{ tool.function.name }}: {{ tool.function.description }}
{% endfor %}
{% endif %}
{% for message in messages %}
{{ message.role }}: {{ message.content }}
{% endfor %}
        """.strip()

        messages = [{"role": "user", "content": "What's the weather?"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        result = render_chat_with_tools(
            template=template,
            messages=messages,
            tools=tools,
        )

        assert "get_weather" in result
        assert "Get current weather" in result
        assert "What's the weather?" in result

    def test_render_with_special_tokens(self):
        """Test rendering with BOS/EOS tokens."""
        template = "{{ bos_token }}Content{{ eos_token }}"

        result = render_chat_with_tools(
            template=template,
            messages=[],
            bos_token="<s>",
            eos_token="</s>",
        )

        assert result == "<s>Content</s>"

    def test_render_invalid_template_raises_error(self):
        """Test that invalid template raises TemplateError."""
        invalid_template = "{% if unclosed"

        with pytest.raises(TemplateError, match="Failed to parse"):
            render_chat_with_tools(
                template=invalid_template,
                messages=[],
            )

    def test_render_with_namespace_style_template(self):
        """Test rendering with a template that uses namespace (like Qwen3).

        This tests the exact pattern that Qwen3 and similar models use.
        """
        template = """
{%- set ns = namespace(found=false) -%}
{%- for message in messages -%}
{%- if message.role == 'system' -%}
{%- set ns.found = true -%}
System: {{ message.content }}
{%- endif -%}
{%- endfor -%}
{%- if not ns.found -%}
(No system message)
{%- endif -%}
{%- for message in messages -%}
{%- if message.role == 'user' -%}
User: {{ message.content }}
{%- endif -%}
{%- endfor -%}
        """.strip()

        # Test with system message
        messages_with_system = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = render_chat_with_tools(
            template=template, messages=messages_with_system
        )
        assert "System: You are helpful." in result
        assert "User: Hello" in result
        assert "(No system message)" not in result

        # Test without system message
        messages_without_system = [
            {"role": "user", "content": "Hello"},
        ]
        result = render_chat_with_tools(
            template=template, messages=messages_without_system
        )
        assert "(No system message)" in result
        assert "User: Hello" in result

    def test_render_complex_tool_template(self):
        """Test a more complex template similar to real chat templates."""
        template = """
{%- if tools %}
<tools>
{%- for tool in tools %}
{{ tool | tojson }}
{%- endfor %}
</tools>
{%- endif %}
{%- for message in messages %}
<|{{ message.role }}|>
{{ message.content }}
{%- endfor %}
{%- if add_generation_prompt %}
<|assistant|>
{%- endif %}
        """.strip()

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Calculate 2+2"},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Perform math",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                    },
                },
            }
        ]

        result = render_chat_with_tools(
            template=template,
            messages=messages,
            tools=tools,
            add_generation_prompt=True,
        )

        assert "<tools>" in result
        assert "calculator" in result
        assert "<|system|>" in result
        assert "<|user|>" in result
        assert "<|assistant|>" in result
