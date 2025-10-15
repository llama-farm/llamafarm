"""
Unit tests for MCPService (using official Python MCP SDK)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.datamodel import LlamaFarmConfig, Mcp, Server, Transport
from services.mcp_service import MCPService, ToolSchema


@pytest.fixture
def mock_stdio_config():
    """Create a mock config with a stdio MCP server."""
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
                    name="stdio-server",
                    transport=Transport.stdio,
                    command="python",
                    args=["-m", "mcp_server"],
                    env={"MCP_ENV": "test"},
                )
            ]
        ),
    )


@pytest.fixture
def mock_http_config():
    """Create a mock config with an HTTP MCP server."""
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
                    name="http-server",
                    transport=Transport.http,
                    base_url="http://localhost:8080",
                    headers={"Authorization": "Bearer token123"},
                )
            ]
        ),
    )


class TestMCPService:
    """Test suite for MCPService."""

    def test_init_with_no_mcp_config(self):
        """Test initialization with no MCP config."""
        config = LlamaFarmConfig(
            version="v1",
            name="test-project",
            namespace="test",
            runtime={
                "provider": "ollama",
                "model": "llama3.2:latest",
            },
            prompts=[],
        )
        service = MCPService(config)
        assert service.list_servers() == []

    def test_init_with_servers(self, mock_stdio_config):
        """Test initialization with MCP servers."""
        service = MCPService(mock_stdio_config)
        assert "stdio-server" in service.list_servers()

    def test_list_servers(self, mock_stdio_config, mock_http_config):
        """Test listing configured servers."""
        config = LlamaFarmConfig(
            version="v1",
            name="test-project",
            namespace="test",
            runtime={
                "provider": "ollama",
                "model": "llama3.2:latest",
            },
            prompts=[],
            mcp=Mcp(
                servers=mock_stdio_config.mcp.servers + mock_http_config.mcp.servers
            ),
        )
        service = MCPService(config)
        servers = service.list_servers()
        assert "stdio-server" in servers
        assert "http-server" in servers

    @patch("services.mcp_service.stdio_client")
    @patch("services.mcp_service.ClientSession")
    async def test_list_stdio_tools_success(
        self, mock_session_class, mock_stdio_client, mock_stdio_config
    ):
        """Test listing tools from STDIO server using MCP SDK."""
        # Create mock tool objects
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.inputSchema = {
            "type": "object",
            "properties": {"arg1": {"type": "string"}},
        }

        # Create mock response
        mock_response = MagicMock()
        mock_response.tools = [mock_tool]

        # Setup mock session
        mock_session = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=mock_response)
        mock_session.initialize = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        # Setup mock stdio client
        mock_streams = AsyncMock()
        mock_streams.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_streams.__aexit__ = AsyncMock(return_value=None)
        mock_stdio_client.return_value = mock_streams

        service = MCPService(mock_stdio_config)
        tools = await service.list_tools("stdio-server")

        assert len(tools) == 1
        assert tools[0]["name"] == "test_tool"

    @patch("services.mcp_service.stdio_client")
    @patch("services.mcp_service.ClientSession")
    async def test_call_stdio_tool_success(
        self, mock_session_class, mock_stdio_client, mock_stdio_config
    ):
        """Test calling a tool on STDIO server using MCP SDK."""
        # Create mock result
        mock_content = MagicMock()
        mock_content.text = "success output"
        mock_result = MagicMock()
        mock_result.content = [mock_content]

        # Setup mock session
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        mock_session.initialize = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        # Setup mock stdio client
        mock_streams = AsyncMock()
        mock_streams.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_streams.__aexit__ = AsyncMock(return_value=None)
        mock_stdio_client.return_value = mock_streams

        service = MCPService(mock_stdio_config)
        result = await service.call_tool(
            "stdio-server", "test_tool", {"arg1": "value1"}
        )

        assert "result" in result
        assert result["result"] == "success output"

    @patch("services.mcp_service.sse_client")
    @patch("services.mcp_service.ClientSession")
    async def test_list_http_tools_success(
        self, mock_session_class, mock_sse_client, mock_http_config
    ):
        """Test listing tools from HTTP server using MCP SDK."""
        # Create mock tool objects
        mock_tool = MagicMock()
        mock_tool.name = "http_tool"
        mock_tool.description = "An HTTP test tool"
        mock_tool.inputSchema = {
            "type": "object",
            "properties": {"param1": {"type": "string"}},
        }

        # Create mock response
        mock_response = MagicMock()
        mock_response.tools = [mock_tool]

        # Setup mock session
        mock_session = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=mock_response)
        mock_session.initialize = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        # Setup mock SSE client
        mock_streams = AsyncMock()
        mock_streams.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_streams.__aexit__ = AsyncMock(return_value=None)
        mock_sse_client.return_value = mock_streams

        service = MCPService(mock_http_config)
        tools = await service.list_tools("http-server")

        assert len(tools) == 1
        assert tools[0]["name"] == "http_tool"

    @patch("services.mcp_service.sse_client")
    @patch("services.mcp_service.ClientSession")
    async def test_call_http_tool_success(
        self, mock_session_class, mock_sse_client, mock_http_config
    ):
        """Test calling a tool on HTTP server using MCP SDK."""
        # Create mock result
        mock_content = MagicMock()
        mock_content.text = "http success"
        mock_result = MagicMock()
        mock_result.content = [mock_content]

        # Setup mock session
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        mock_session.initialize = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        # Setup mock SSE client
        mock_streams = AsyncMock()
        mock_streams.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_streams.__aexit__ = AsyncMock(return_value=None)
        mock_sse_client.return_value = mock_streams

        service = MCPService(mock_http_config)
        result = await service.call_tool(
            "http-server", "http_tool", {"param1": "value"}
        )

        assert "result" in result
        assert result["result"] == "http success"

    async def test_list_tools_invalid_server(self, mock_stdio_config):
        """Test listing tools for non-existent server."""
        service = MCPService(mock_stdio_config)
        tools = await service.list_tools("invalid-server")
        assert len(tools) == 0

    async def test_call_tool_invalid_server(self, mock_stdio_config):
        """Test calling tool on non-existent server."""
        service = MCPService(mock_stdio_config)
        result = await service.call_tool("invalid-server", "test_tool", {})
        assert "error" in result
        assert "not found" in result["error"].lower()

    @patch("services.mcp_service.stdio_client")
    @patch("services.mcp_service.ClientSession")
    async def test_tool_caching(
        self, mock_session_class, mock_stdio_client, mock_stdio_config
    ):
        """Test that tool list is cached properly."""
        # Create mock tool objects
        mock_tool = MagicMock()
        mock_tool.name = "cached_tool"
        mock_tool.description = "A cached tool"
        mock_tool.inputSchema = {"type": "object", "properties": {}}

        # Create mock response
        mock_response = MagicMock()
        mock_response.tools = [mock_tool]

        # Setup mock session
        mock_session = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=mock_response)
        mock_session.initialize = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        # Setup mock stdio client
        mock_streams = AsyncMock()
        mock_streams.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_streams.__aexit__ = AsyncMock(return_value=None)
        mock_stdio_client.return_value = mock_streams

        service = MCPService(mock_stdio_config)

        # First call should hit the server
        tools1 = await service.list_tools("stdio-server")
        assert len(tools1) == 1
        assert mock_session.list_tools.await_count == 1

        # Second call should use cache
        tools2 = await service.list_tools("stdio-server")
        assert len(tools2) == 1
        assert mock_session.list_tools.await_count == 1  # Still 1, not 2

        # Results should be the same
        assert tools1 == tools2
