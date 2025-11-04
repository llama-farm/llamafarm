from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager, suppress
from typing import Any

from config.datamodel import LlamaFarmConfig, Server, Transport
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from pydantic import BaseModel

from core.logging import FastAPIStructLogger
from services.model_service import ModelService

logger = FastAPIStructLogger(__name__)


class ToolSchema(BaseModel):
    """Schema for MCP tool definition."""

    name: str
    description: str | None = None
    inputSchema: dict[str, Any] | None = None


class MCPService:
    """Manage MCP client sessions and tool calls based on project config.

    Uses the official Python MCP SDK for communication with MCP servers.
    Maintains persistent sessions for each server to avoid connection overhead.
    """

    def __init__(self, config: LlamaFarmConfig, model_name: str | None = None) -> None:
        self._config = config
        model_service = ModelService()
        self._model_config = model_service.get_model(config, model_name)
        self._servers = self._resolve_servers()
        self._tool_cache: dict[str, list[ToolSchema]] = {}
        self._cache_ttl = 300  # 5 minutes
        self._last_cache_update: dict[str, float] = {}

        # Persistent session management
        # Note: Sessions are kept alive for the lifetime of the service
        # to avoid ClosedResourceError when tools are invoked
        self._persistent_sessions: dict[str, ClientSession] = {}
        self._session_tasks: dict[str, asyncio.Task] = {}  # Background tasks
        self._shutdown_events: dict[str, asyncio.Event] = {}  # For graceful shutdown
        self._cleanup_lock = asyncio.Lock()

        logger.info("MCPService initialized", server_count=len(self._servers))

    def _resolve_servers(self) -> list[Server]:
        """Resolve the list of MCP servers to use for the model."""
        servers_to_use: list[Server] = []
        if self._model_config.mcp_servers is not None:
            servers_to_use.extend(
                [
                    s
                    for s in self._config.mcp.servers
                    if s.name in self._model_config.mcp_servers
                ]
            )
        else:
            servers_to_use.extend(
                self._config.mcp.servers if self._config.mcp is not None else []
            )
        return servers_to_use

    def list_servers(self) -> list[str]:
        """List all configured MCP server names."""
        return [s.name for s in self._servers]

    def get_server(self, server_name: str) -> Server | None:
        """Get the MCP server configuration by name."""
        return next((s for s in self._servers if s.name == server_name), None)

    async def list_tools(self, server_name: str) -> list[dict[str, Any]]:
        """List tools available from the specified MCP server."""
        server_config = self.get_server(server_name)
        if server_config is None:
            logger.warning("MCP server not found", server_name=server_name)
            return []

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

    def _is_cache_valid(self, server_name: str) -> bool:
        """Check if cached tools are still valid."""
        if server_name not in self._last_cache_update:
            return False
        return time.time() - self._last_cache_update[server_name] < self._cache_ttl

    async def get_or_create_persistent_session(self, server_name: str) -> ClientSession:
        """Get or create a persistent session for the specified server.

        This session will remain open for the lifetime of the service,
        allowing MCP tools to reuse the same connection.

        The session is kept alive by a background task that maintains
        the context managers, avoiding cancel scope issues.

        Args:
            server_name: Name of the MCP server

        Returns:
            ClientSession that will persist across tool calls

        Raises:
            ValueError: If server not found or misconfigured
        """
        server_config = self.get_server(server_name)
        if server_config is None:
            raise ValueError(f"Server '{server_name}' not found")

        # Return existing session if available
        if server_name in self._persistent_sessions:
            return self._persistent_sessions[server_name]

        logger.info(
            "Creating persistent MCP session in background task",
            server_name=server_name,
            transport=server_config.transport.value,
        )

        # Create an event to signal when session is ready
        session_ready = asyncio.Event()
        shutdown_event = asyncio.Event()
        session_container = {}
        error_container = {}

        # Store shutdown event for this server
        self._shutdown_events[server_name] = shutdown_event

        async def maintain_session():
            """Background task that keeps the session context alive."""
            try:
                # Use longer timeout (1 hour) for persistent sessions
                async with self._create_session_context(
                    server_config, sse_read_timeout=3600.0
                ) as session:
                    session_container["session"] = session
                    session_ready.set()

                    # Wait for shutdown signal - keeps context alive
                    await shutdown_event.wait()

            except asyncio.CancelledError:
                logger.info(
                    "MCP session task cancelled (shutting down)",
                    server_name=server_name,
                )
                raise
            except Exception as e:
                error_container["error"] = e
                session_ready.set()

        # Start background task
        task = asyncio.create_task(maintain_session())
        self._session_tasks[server_name] = task

        # Wait for session to be ready
        await session_ready.wait()

        # Check for errors
        if "error" in error_container:
            error = error_container["error"]
            logger.error(
                "Failed to create persistent MCP session",
                server_name=server_name,
                error=str(error),
                error_type=type(error).__name__,
            )
            raise error

        # Get the session
        session = session_container["session"]
        self._persistent_sessions[server_name] = session

        logger.info(
            "Persistent MCP session created in background task",
            server_name=server_name,
        )

        return session

    async def close_persistent_session(self, server_name: str) -> None:
        """Close a persistent session for the specified server.

        Args:
            server_name: Name of the MCP server
        """
        async with self._cleanup_lock:
            if server_name in self._shutdown_events:
                logger.info("Closing MCP session", server_name=server_name)
                # Signal the background task to shutdown
                self._shutdown_events[server_name].set()

                # Wait for task to complete (with timeout)
                if server_name in self._session_tasks:
                    task = self._session_tasks[server_name]
                    try:
                        await asyncio.wait_for(task, timeout=5.0)
                    except TimeoutError:
                        logger.warning(
                            "MCP session task did not complete, cancelling",
                            server_name=server_name,
                        )
                        task.cancel()
                        with suppress(asyncio.CancelledError):
                            await task

                # Cleanup
                self._persistent_sessions.pop(server_name, None)
                self._session_tasks.pop(server_name, None)
                self._shutdown_events.pop(server_name, None)

                logger.info("MCP session closed", server_name=server_name)

    async def close_all_persistent_sessions(self) -> None:
        """Close all persistent sessions gracefully."""
        logger.info(
            "Closing all persistent MCP sessions", count=len(self._persistent_sessions)
        )

        # Get all server names to avoid dict modification during iteration
        server_names = list(self._persistent_sessions.keys())

        # Close all sessions
        for server_name in server_names:
            try:
                await self.close_persistent_session(server_name)
            except Exception as e:
                logger.error(
                    "Error closing MCP session",
                    server_name=server_name,
                    error=str(e),
                    exc_info=True,
                )

        logger.info("All MCP sessions closed")

    @asynccontextmanager
    async def _create_session_context(
        self, server_config: Server, *, sse_read_timeout: float = 300.0
    ):
        """Create an MCP client session context based on transport type.

        Args:
            server_config: Server configuration
            sse_read_timeout: Timeout for SSE/HTTP read operations in seconds.
                             Use 300 (5 min) for one-off operations,
                             3600 (1 hour) for persistent sessions.

        Yields:
            Initialized ClientSession
        """
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

            async with (
                streamablehttp_client(
                    server_config.base_url,
                    timeout=30.0,
                    sse_read_timeout=sse_read_timeout,
                ) as (
                    read_stream,
                    write_stream,
                    _get_session_id,  # noqa: F841
                ),
                ClientSession(read_stream, write_stream) as session,
            ):
                await session.initialize()
                yield session

        elif server_config.transport == Transport.sse:
            if not server_config.base_url:
                raise ValueError(
                    f"SSE server '{server_config.name}' has no base_url configured"
                )

            async with (
                sse_client(
                    server_config.base_url,
                    timeout=30.0,
                    sse_read_timeout=sse_read_timeout,
                ) as (read_stream, write_stream),
                ClientSession(read_stream, write_stream) as session,
            ):
                await session.initialize()
                yield session

        else:
            raise ValueError(f"Unsupported transport: {server_config.transport}")

    @asynccontextmanager
    async def _get_client_session(self, server_config: Server):
        """Get an MCP client session for one-off operations.

        Uses shorter timeout (5 minutes) suitable for quick operations.
        """
        async with self._create_session_context(
            server_config, sse_read_timeout=300.0
        ) as session:
            yield session

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
