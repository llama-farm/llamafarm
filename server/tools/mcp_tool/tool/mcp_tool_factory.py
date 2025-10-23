"""
MCP Tool Factory

Creates dynamic AtomicAgents tools from MCP server tool schemas
using atomic-agents' native MCP support.
"""

from atomic_agents import BaseTool  # type: ignore
from atomic_agents.connectors.mcp import (  # type: ignore
    MCPTransportType,
    fetch_mcp_tools_async,
)
from config.datamodel import Server, Transport

from core.logging import FastAPIStructLogger  # type: ignore
from services.mcp_service import MCPService  # type: ignore

logger = FastAPIStructLogger(__name__)


class MCPToolFactory:
    """Factory for creating dynamic MCP tools with atomic-agents."""

    def __init__(self, mcp_service: MCPService):
        self._mcp_service = mcp_service

    def _get_transport_type(self, server: Server) -> MCPTransportType:
        """Convert LlamaFarm transport to atomic-agents MCPTransportType.

        For fastapi-mcp servers, use HTTP_STREAM transport.
        """
        if server.transport == Transport.http:
            # Use HTTP_STREAM for fastapi-mcp compatibility
            return MCPTransportType.HTTP_STREAM
        elif server.transport == Transport.stdio:
            return MCPTransportType.STDIO
        elif server.transport == Transport.sse:
            return MCPTransportType.SSE
        else:
            raise ValueError(f"Unsupported transport type: {server.transport}")

    def _get_mcp_endpoint(self, server: Server) -> str:
        """Get MCP endpoint string based on transport type.

        Returns the base_url for passing to atomic-agents MCP tools.
        The transport type determines what atomic-agents appends:
        - HTTP_STREAM: appends '/mcp/'
        - SSE: appends '/sse'

        For fastapi-mcp servers at /mcp endpoint:
        - Server is at: http://host/mcp
        - We return: http://host/mcp (atomic-agents won't append anything)
        """
        if server.transport == Transport.http:
            if not server.base_url:
                raise ValueError(f"HTTP server '{server.name}' has no base_url")

            # Return base_url exactly as configured
            # Don't strip anything - let user specify exact endpoint
            return server.base_url.rstrip("/")

        elif server.transport == Transport.stdio:
            # For STDIO, create command string
            if not server.command:
                raise ValueError(f"STDIO server '{server.name}' has no command")
            command_parts = [server.command] + (server.args or [])
            return " ".join(command_parts)

        elif server.transport == Transport.sse:
            if not server.base_url:
                raise ValueError(f"SSE server '{server.name}' has no base_url")
            return server.base_url.rstrip("/")

        else:
            raise ValueError(f"Unsupported transport type: {server.transport}")

    async def create_tools_for_server(self, server_name: str) -> list[type[BaseTool]]:
        """Create AtomicAgents tools for MCP server.

        Uses atomic-agents' native fetch_mcp_tools_async to create
        properly structured tools that work with the orchestrator pattern.

        The tools will use a persistent session from MCPService that remains
        open for the lifetime of the service, avoiding connection errors.
        """
        try:
            # Get server configuration
            servers = {
                s.name: s
                for s in (self._mcp_service._config.mcp.servers or [])
                if self._mcp_service._config.mcp
            }
            if server_name not in servers:
                logger.warning("Server not found", server_name=server_name)
                return []

            server_config = servers[server_name]
            transport_type = self._get_transport_type(server_config)
            mcp_endpoint = self._get_mcp_endpoint(server_config)

            logger.info(
                "Creating MCP tools using atomic-agents with persistent session",
                server_name=server_name,
                transport=transport_type.value,
                endpoint=mcp_endpoint,
            )

            # Get or create persistent session for this server
            # Session remains open for the lifetime of the service
            persistent_session = (
                await self._mcp_service.get_or_create_persistent_session(server_name)
            )

            # Fetch tools using the persistent session
            # The tools will hold a reference to this session for future calls
            tools = await fetch_mcp_tools_async(
                mcp_endpoint=mcp_endpoint,
                transport_type=transport_type,
                client_session=persistent_session,
            )

            tool_names = [getattr(t, "mcp_tool_name", t.__name__) for t in tools]
            logger.info(
                "Created MCP tools with persistent session",
                server_name=server_name,
                tool_count=len(tools),
                tool_names=tool_names,
            )
            return tools

        except Exception as e:
            logger.error(
                "Error creating tools for server",
                server_name=server_name,
                error=str(e),
                exc_info=True,
            )
            return []

    async def create_all_tools(self) -> list[type[BaseTool]]:
        """Create AtomicAgents tools for all configured MCP servers."""
        all_tools = []

        for server_name in self._mcp_service.list_servers():
            server_tools = await self.create_tools_for_server(server_name)
            all_tools.extend(server_tools)

        logger.info(
            "Created all MCP tools",
            total_tools=len(all_tools),
            servers=self._mcp_service.list_servers(),
        )
        return all_tools
