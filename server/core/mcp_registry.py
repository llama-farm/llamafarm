"""MCP Service Registry for managing service lifecycle and cleanup."""

from core.logging import FastAPIStructLogger

logger = FastAPIStructLogger(__name__)

# Global registry for MCP services to clean up on shutdown
_mcp_services_registry: set = set()


def register_mcp_service(mcp_service):
    """Register an MCP service for cleanup on shutdown.

    Args:
        mcp_service: An MCPService instance to register for cleanup
    """
    _mcp_services_registry.add(mcp_service)
    logger.debug("Registered MCP service", service=mcp_service)


async def cleanup_all_mcp_services():
    """Clean up all registered MCP services by closing their sessions."""
    logger.info("Closing MCP connections...")
    for mcp_service in _mcp_services_registry:
        try:
            await mcp_service.close_all_persistent_sessions()
        except Exception as e:
            logger.error("Error closing MCP service", error=str(e), exc_info=True)
    logger.info("All MCP connections closed")
