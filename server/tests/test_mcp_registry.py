"""
Unit tests for MCP Registry.

Tests the MCP service registry for managing service lifecycle and cleanup.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.mcp_registry import (
    _mcp_services_registry,
    cleanup_all_mcp_services,
    register_mcp_service,
)


class TestMCPRegistry:
    """Test suite for MCP service registry."""

    def setup_method(self):
        """Clear registry before each test."""
        _mcp_services_registry.clear()

    def test_register_mcp_service(self):
        """Test registering an MCP service."""
        mock_service = MagicMock()

        register_mcp_service(mock_service)

        assert mock_service in _mcp_services_registry

    def test_register_multiple_services(self):
        """Test registering multiple MCP services."""
        mock_service1 = MagicMock()
        mock_service2 = MagicMock()

        register_mcp_service(mock_service1)
        register_mcp_service(mock_service2)

        assert len(_mcp_services_registry) == 2
        assert mock_service1 in _mcp_services_registry
        assert mock_service2 in _mcp_services_registry

    def test_register_same_service_twice(self):
        """Test that registering the same service twice doesn't duplicate."""
        mock_service = MagicMock()

        register_mcp_service(mock_service)
        register_mcp_service(mock_service)

        # Sets don't allow duplicates
        assert len(_mcp_services_registry) == 1

    @pytest.mark.asyncio
    async def test_cleanup_all_mcp_services(self):
        """Test cleaning up all registered services."""
        # Create mock services
        mock_service1 = MagicMock()
        mock_service1.close_all_persistent_sessions = AsyncMock()

        mock_service2 = MagicMock()
        mock_service2.close_all_persistent_sessions = AsyncMock()

        # Register services
        register_mcp_service(mock_service1)
        register_mcp_service(mock_service2)

        # Cleanup all
        await cleanup_all_mcp_services()

        # Both services should have their cleanup called
        mock_service1.close_all_persistent_sessions.assert_called_once()
        mock_service2.close_all_persistent_sessions.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_with_error(self):
        """Test that cleanup continues even if one service fails."""
        # Create mock services
        mock_service1 = MagicMock()
        mock_service1.close_all_persistent_sessions = AsyncMock(
            side_effect=Exception("Cleanup failed")
        )

        mock_service2 = MagicMock()
        mock_service2.close_all_persistent_sessions = AsyncMock()

        # Register services
        register_mcp_service(mock_service1)
        register_mcp_service(mock_service2)

        # Cleanup should not raise, but continue with other services
        await cleanup_all_mcp_services()

        # Both should have been attempted
        mock_service1.close_all_persistent_sessions.assert_called_once()
        mock_service2.close_all_persistent_sessions.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_empty_registry(self):
        """Test cleanup with no registered services."""
        # Should not raise error
        await cleanup_all_mcp_services()
