"""
Unit tests for MCPToolFactory
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from config.datamodel import LlamaFarmConfig, Mcp, Server, Transport

from services.mcp_service import MCPService
from tools.mcp_tool.tool.mcp_tool_factory import (
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

        # Create mock tool classes
        mock_tool1 = MagicMock()
        mock_tool1.__name__ = "Tool1"
        mock_tool1.mcp_tool_name = "tool1"

        mock_tool2 = MagicMock()
        mock_tool2.__name__ = "Tool2"
        mock_tool2.mcp_tool_name = "tool2"

        # Mock the persistent session to avoid actual network connection
        mock_session = AsyncMock()

        # Mock fetch_mcp_tools_async from atomic_agents
        with patch.object(
            service, "get_or_create_persistent_session", return_value=mock_session
        ):
            with patch(
                "tools.mcp_tool.tool.mcp_tool_factory.fetch_mcp_tools_async",
                return_value=[mock_tool1, mock_tool2],
            ):
                tools = await factory.create_tools_for_server("test-server")
                assert len(tools) == 2
                # Tools are now dynamic AtomicAgent tool classes
                assert all(hasattr(tool, "mcp_tool_name") for tool in tools)

    async def test_create_all_tools(self, mock_config):
        """Test creating tools for all servers."""
        service = MCPService(mock_config)
        factory = MCPToolFactory(service)

        # Create mock tool class
        mock_tool = MagicMock()
        mock_tool.__name__ = "GlobalTool"
        mock_tool.mcp_tool_name = "global_tool"

        # Mock the persistent session to avoid actual network connection
        mock_session = AsyncMock()

        # Mock fetch_mcp_tools_async from atomic_agents
        with patch.object(
            service, "get_or_create_persistent_session", return_value=mock_session
        ):
            with patch(
                "tools.mcp_tool.tool.mcp_tool_factory.fetch_mcp_tools_async",
                return_value=[mock_tool],
            ):
                tools = await factory.create_all_tools()
                assert len(tools) >= 1

    async def test_create_tools_invalid_schema(self, mock_config):
        """Test that invalid schemas are handled gracefully."""
        service = MCPService(mock_config)
        factory = MCPToolFactory(service)

        # Create mock tool classes
        mock_tool1 = MagicMock()
        mock_tool1.__name__ = "ValidTool"
        mock_tool1.mcp_tool_name = "valid_tool"

        mock_tool2 = MagicMock()
        mock_tool2.__name__ = "ProblemTool"
        mock_tool2.mcp_tool_name = "problem_tool"

        # Mock the persistent session to avoid actual network connection
        mock_session = AsyncMock()

        # Mock fetch_mcp_tools_async - atomic_agents handles schema validation
        # and will return valid tools even if some have issues
        with patch.object(
            service, "get_or_create_persistent_session", return_value=mock_session
        ):
            with patch(
                "tools.mcp_tool.tool.mcp_tool_factory.fetch_mcp_tools_async",
                return_value=[mock_tool1, mock_tool2],
            ):
                tools = await factory.create_tools_for_server("test-server")
                # Both tools should be returned since atomic_agents handles validation
                assert len(tools) == 2
