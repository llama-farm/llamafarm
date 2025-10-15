from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

from config.datamodel import LlamaFarmConfig, Server, Transport
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from pydantic import BaseModel

from core.logging import FastAPIStructLogger

logger = FastAPIStructLogger(__name__)


class ToolSchema(BaseModel):
    """Schema for MCP tool definition."""

    name: str
    description: str | None = None
    inputSchema: dict[str, Any] | None = None


class MCPService:
    """Manage MCP client sessions and tool calls based on project config.

    Uses the official Python MCP SDK for communication with MCP servers.
    """

    def __init__(self, config: LlamaFarmConfig) -> None:
        self._config = config
        self._servers = (
            {s.name: s for s in (config.mcp.servers or [])} if config.mcp else {}
        )
        self._tool_cache: dict[str, list[ToolSchema]] = {}
        self._cache_ttl = 300  # 5 minutes
        self._last_cache_update: dict[str, float] = {}
        logger.info("MCPService initialized", server_count=len(self._servers))

    def list_servers(self) -> list[str]:
        """List all configured MCP server names."""
        return list(self._servers.keys())

    async def list_tools(self, server_name: str) -> list[dict[str, Any]]:
        """List tools available from the specified MCP server."""
        if server_name not in self._servers:
            logger.warning("MCP server not found", server_name=server_name)
            return []

        server_config = self._servers[server_name]

        # Check cache first
        if self._is_cache_valid(server_name):
            cached_tools = self._tool_cache.get(server_name, [])
            return [tool.model_dump() for tool in cached_tools]

        try:
            # Call async method directly (no thread needed in async context)
            tools = await self._list_tools_async(server_config)

            # Cache the results
            self._tool_cache[server_name] = tools
            self._last_cache_update[server_name] = time.time()

            return [tool.model_dump() for tool in tools]
        except Exception as e:
            logger.exception(
                "Error listing tools", server_name=server_name, error=str(e)
            )
            return []

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call a tool on the specified MCP server."""
        if server_name not in self._servers:
            return {"error": f"Server '{server_name}' not found"}

        server_config = self._servers[server_name]

        try:
            # Call async method directly (no thread needed in async context)
            result = await self._call_tool_async(
                server_config, tool_name, arguments or {}
            )
            return result
        except Exception as e:
            logger.exception(
                "Error calling tool",
                server_name=server_name,
                tool_name=tool_name,
                error=str(e),
            )
            return {"error": str(e)}

    def _is_cache_valid(self, server_name: str) -> bool:
        """Check if cached tools are still valid."""
        if server_name not in self._last_cache_update:
            return False
        return time.time() - self._last_cache_update[server_name] < self._cache_ttl

    @asynccontextmanager
    async def _get_client_session(self, server_config: Server):
        """Get an MCP client session based on transport type."""
        if server_config.transport == Transport.stdio:
            if not server_config.command:
                raise ValueError(
                    f"STDIO server '{server_config.name}' has no command configured"
                )

            server_params = StdioServerParameters(
                command=server_config.command,
                args=server_config.args or [],
                env=server_config.env,
            )

            async with (
                stdio_client(server_params) as (read_stream, write_stream),
                ClientSession(read_stream, write_stream) as session,
            ):
                await session.initialize()
                yield session

        elif server_config.transport == Transport.http:
            if not server_config.base_url:
                raise ValueError(
                    f"HTTP server '{server_config.name}' has no base_url configured"
                )

            # Use SSE client for HTTP transport
            async with (
                sse_client(server_config.base_url) as (
                    read_stream,
                    write_stream,
                ),
                ClientSession(read_stream, write_stream) as session,
            ):
                await session.initialize()
                yield session
        else:
            raise ValueError(f"Unsupported transport: {server_config.transport}")

    async def _list_tools_async(self, server_config: Server) -> list[ToolSchema]:
        """List tools from MCP server using official SDK."""
        logger.info(
            "Listing MCP tools",
            server_name=server_config.name,
            transport=server_config.transport.value,
        )

        try:
            async with self._get_client_session(server_config) as session:
                response = await session.list_tools()

                tools = []
                for tool in response.tools:
                    tools.append(
                        ToolSchema(
                            name=tool.name,
                            description=tool.description,
                            inputSchema=(
                                tool.inputSchema
                                if hasattr(tool, "inputSchema")
                                else None
                            ),
                        )
                    )

                logger.info(
                    "Retrieved MCP tools",
                    server_name=server_config.name,
                    tool_count=len(tools),
                )
                return tools
        except Exception as e:
            logger.error(
                "Error in _list_tools_async",
                server_name=server_config.name,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    async def _call_tool_async(
        self, server_config: Server, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Call a tool on MCP server using official SDK."""
        logger.info(
            "Calling MCP tool",
            server_name=server_config.name,
            tool_name=tool_name,
            arguments=arguments,
        )

        try:
            async with self._get_client_session(server_config) as session:
                result = await session.call_tool(tool_name, arguments=arguments)

                # Extract content from the result
                if hasattr(result, "content"):
                    content_list = result.content
                    if content_list:
                        # Return the first content item's text or data
                        first_content = content_list[0]
                        if hasattr(first_content, "text"):
                            return {"result": first_content.text}
                        elif hasattr(first_content, "data"):
                            return {"result": first_content.data}
                        else:
                            return {"result": str(first_content)}

                # Fallback: return the whole result as dict
                return {"result": str(result)}
        except Exception as e:
            logger.error(
                "Error in _call_tool_async",
                server_name=server_config.name,
                tool_name=tool_name,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise
