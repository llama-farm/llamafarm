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

    async def test_list_tools_invalid_server(self, mock_stdio_config):
        """Test listing tools for non-existent server."""
        service = MCPService(mock_stdio_config)
        tools = await service.list_tools("invalid-server")
        assert len(tools) == 0

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


class TestMCPServicePersistentSessions:
    """Test suite for MCP persistent session management."""

    @pytest.mark.asyncio
    @patch("services.mcp_service.stdio_client")
    @patch("services.mcp_service.ClientSession")
    async def test_create_persistent_session_stdio(
        self, mock_session_class, mock_stdio_client, mock_stdio_config
    ):
        """Test creating a persistent session for STDIO server."""
        # Setup mock session
        mock_session = AsyncMock()
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

        # Create persistent session
        session = await service.get_or_create_persistent_session("stdio-server")

        assert session is not None
        assert "stdio-server" in service._persistent_sessions
        mock_session.initialize.assert_called_once()

    @pytest.mark.asyncio
    @patch("services.mcp_service.sse_client")
    @patch("services.mcp_service.ClientSession")
    async def test_create_persistent_session_http(
        self, mock_session_class, mock_sse_client, mock_http_config
    ):
        """Test creating a persistent session for HTTP server."""
        # Setup mock session
        mock_session = AsyncMock()
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

        # Create persistent session
        session = await service.get_or_create_persistent_session("http-server")

        assert session is not None
        assert "http-server" in service._persistent_sessions
        mock_session.initialize.assert_called_once()

    @pytest.mark.asyncio
    @patch("services.mcp_service.stdio_client")
    @patch("services.mcp_service.ClientSession")
    async def test_get_existing_persistent_session(
        self, mock_session_class, mock_stdio_client, mock_stdio_config
    ):
        """Test getting an existing persistent session."""
        # Setup mock session
        mock_session = AsyncMock()
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

        # Create session first time
        session1 = await service.get_or_create_persistent_session("stdio-server")

        # Get same session second time
        session2 = await service.get_or_create_persistent_session("stdio-server")

        # Should return the same session instance
        assert session1 is session2
        # Initialize should only be called once
        assert mock_session.initialize.call_count == 1

    @pytest.mark.asyncio
    async def test_persistent_session_invalid_server(self, mock_stdio_config):
        """Test creating persistent session for non-existent server."""
        service = MCPService(mock_stdio_config)

        with pytest.raises(ValueError, match="not found"):
            await service.get_or_create_persistent_session("invalid-server")

    @pytest.mark.asyncio
    @patch("services.mcp_service.stdio_client")
    @patch("services.mcp_service.ClientSession")
    async def test_close_persistent_session(
        self, mock_session_class, mock_stdio_client, mock_stdio_config
    ):
        """Test closing a persistent session."""
        # Setup mock session
        mock_session = AsyncMock()
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

        # Create session
        await service.get_or_create_persistent_session("stdio-server")
        assert "stdio-server" in service._persistent_sessions

        # Close session
        await service.close_persistent_session("stdio-server")

        # Session should be removed
        assert "stdio-server" not in service._persistent_sessions
        assert "stdio-server" not in service._session_tasks
        assert "stdio-server" not in service._shutdown_events

    @pytest.mark.asyncio
    async def test_close_nonexistent_session(self, mock_stdio_config):
        """Test closing a session that doesn't exist."""
        service = MCPService(mock_stdio_config)

        # Should not raise error
        await service.close_persistent_session("nonexistent-server")

    @pytest.mark.asyncio
    @patch("services.mcp_service.stdio_client")
    @patch("services.mcp_service.ClientSession")
    async def test_close_all_persistent_sessions(
        self, mock_session_class, mock_stdio_client, mock_stdio_config
    ):
        """Test closing all persistent sessions."""
        # Setup mock session
        mock_session = AsyncMock()
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

        # Create multiple sessions
        await service.get_or_create_persistent_session("stdio-server")
        assert len(service._persistent_sessions) == 1

        # Close all sessions
        await service.close_all_persistent_sessions()

        # All sessions should be closed
        assert len(service._persistent_sessions) == 0
        assert len(service._session_tasks) == 0
        assert len(service._shutdown_events) == 0

    @pytest.mark.asyncio
    @patch("services.mcp_service.stdio_client")
    @patch("services.mcp_service.ClientSession")
    async def test_session_initialization_error(
        self, mock_session_class, mock_stdio_client, mock_stdio_config
    ):
        """Test handling of session initialization errors."""
        # Setup mock to raise error during initialization
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock(side_effect=Exception("Connection failed"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        # Setup mock stdio client
        mock_streams = AsyncMock()
        mock_streams.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_streams.__aexit__ = AsyncMock(return_value=None)
        mock_stdio_client.return_value = mock_streams

        service = MCPService(mock_stdio_config)

        # Should raise the initialization error
        with pytest.raises(Exception, match="Connection failed"):
            await service.get_or_create_persistent_session("stdio-server")

    @pytest.mark.asyncio
    async def test_stdio_server_without_command(self):
        """Test STDIO server without command configured."""
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
                servers=[
                    Server(
                        name="invalid-stdio",
                        transport=Transport.stdio,
                        # No command specified
                    )
                ]
            ),
        )

        service = MCPService(config)

        with pytest.raises(ValueError, match="has no command"):
            await service.get_or_create_persistent_session("invalid-stdio")

    @pytest.mark.asyncio
    async def test_http_server_without_base_url(self):
        """Test HTTP server without base_url configured."""
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
                servers=[
                    Server(
                        name="invalid-http",
                        transport=Transport.http,
                        # No base_url specified
                    )
                ]
            ),
        )

        service = MCPService(config)

        with pytest.raises(ValueError, match="has no base_url"):
            await service.get_or_create_persistent_session("invalid-http")
