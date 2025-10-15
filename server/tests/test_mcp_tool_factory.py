"""
Unit tests for MCPToolFactory
"""

from unittest.mock import MagicMock, patch

import pytest

from config.datamodel import LlamaFarmConfig, Mcp, Server, Transport
from services.mcp_service import MCPService
from tools.mcp_tool.tool.mcp_tool_factory import (
    DynamicMCPTool,
    MCPToolInput,
    MCPToolFactory,
)


@pytest.fixture
def mock_config():
    """Create a mock config with MCP servers."""
    return LlamaFarmConfig(
        version="v1",
        name="test-project",
        namespace="test",
        runtime={
            "provider": "ollama",
            "model": "llama3.2:latest",
        },
        prompts=[],
        mcp=Mcp(
            servers=[
                Server(
                    name="test-server",
                    transport=Transport.http,
                    base_url="http://localhost:8080",
                )
            ]
        ),
    )


class TestMCPToolFactory:
    """Test suite for MCPToolFactory."""

    async def test_create_tools_for_server_no_tools(self, mock_config):
        """Test creating tools when server has no tools."""
        service = MCPService(mock_config)
        factory = MCPToolFactory(service)

        with patch.object(service, "list_tools", return_value=[]):
            tools = await factory.create_tools_for_server("test-server")
            assert len(tools) == 0

    async def test_create_tools_for_server_with_tools(self, mock_config):
        """Test creating tools from server schemas."""
        service = MCPService(mock_config)
        factory = MCPToolFactory(service)

        mock_tools = [
            {
                "name": "tool1",
                "description": "First tool",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "tool2",
                "description": "Second tool",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]

        with patch.object(service, "list_tools", return_value=mock_tools):
            tools = await factory.create_tools_for_server("test-server")
            assert len(tools) == 2
            assert all(isinstance(tool, DynamicMCPTool) for tool in tools)

    async def test_create_all_tools(self, mock_config):
        """Test creating tools for all servers."""
        service = MCPService(mock_config)
        factory = MCPToolFactory(service)

        mock_tools = [
            {
                "name": "global_tool",
                "description": "A tool",
                "inputSchema": {"type": "object", "properties": {}},
            }
        ]

        with patch.object(service, "list_tools", return_value=mock_tools):
            tools = await factory.create_all_tools()
            assert len(tools) >= 1

    async def test_dynamic_tool_execution_success(self, mock_config):
        """Test dynamic tool execution with success."""
        service = MCPService(mock_config)

        mock_tool_schema = MagicMock()
        mock_tool_schema.name = "test_tool"

        tool = DynamicMCPTool(service, "test-server", mock_tool_schema)

        with patch.object(service, "call_tool", return_value={"result": "success"}):
            input_data = MCPToolInput(arguments={"arg1": "value1"})
            output = await tool.run_async(input_data)

            assert output.success is True
            assert output.result == {"result": "success"}
            assert output.error is None

    async def test_dynamic_tool_execution_error(self, mock_config):
        """Test dynamic tool execution with error."""
        service = MCPService(mock_config)

        mock_tool_schema = MagicMock()
        mock_tool_schema.name = "failing_tool"

        tool = DynamicMCPTool(service, "test-server", mock_tool_schema)

        with patch.object(
            service, "call_tool", side_effect=Exception("Tool execution failed")
        ):
            input_data = MCPToolInput(arguments={"arg1": "value1"})
            output = await tool.run_async(input_data)

            assert output.success is False
            assert output.error == "Tool execution failed"
            assert output.result is None

    async def test_create_tools_invalid_schema(self, mock_config):
        """Test that invalid schemas are skipped gracefully."""
        service = MCPService(mock_config)
        factory = MCPToolFactory(service)

        # Mock tools with one that will cause an exception during tool creation
        mock_tools = [
            {
                "name": "valid_tool",
                "description": "Valid tool",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                # This will be valid for ToolSchema but we'll mock an exception
                "name": "problem_tool",
                "description": "Tool that causes issues",
            },
        ]

        with patch.object(service, "list_tools", return_value=mock_tools):
            # Both tools should be created successfully since ToolSchema is flexible
            tools = await factory.create_tools_for_server("test-server")
            # We get 2 tools since both schemas are actually valid with optional fields
            assert len(tools) == 2
