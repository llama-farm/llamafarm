"""
Channel Registry for managing multiple messaging channels.
"""

import asyncio
import logging
from typing import Callable, Dict, List, Optional, Type, Awaitable

from .base import Channel, ChannelConfig, ChannelMessage

logger = logging.getLogger(__name__)


class ChannelRegistry:
    """
    Registry for managing multiple channels.
    
    Features:
    - Register/unregister channels
    - Route messages to appropriate channels
    - Global message handling
    - Channel lifecycle management
    """
    
    def __init__(self):
        self._channels: Dict[str, Channel] = {}
        self._channel_types: Dict[str, Type[Channel]] = {}
        self._global_handlers: List[Callable[[ChannelMessage, Channel], Awaitable[None]]] = []
        self._running = False
    
    def register_channel_type(
        self,
        channel_type: str,
        channel_class: Type[Channel]
    ) -> None:
        """Register a channel type for factory creation."""
        self._channel_types[channel_type] = channel_class
        logger.info(f"Registered channel type: {channel_type}")
    
    def create_channel(self, config: ChannelConfig) -> Channel:
        """Create a channel from config."""
        channel_class = self._channel_types.get(config.channel_type)
        if not channel_class:
            raise ValueError(f"Unknown channel type: {config.channel_type}")
        
        channel = channel_class(config)
        return channel
    
    def add_channel(self, channel: Channel) -> None:
        """Add a channel to the registry."""
        if channel.channel_id in self._channels:
            raise ValueError(f"Channel already registered: {channel.channel_id}")
        
        self._channels[channel.channel_id] = channel
        
        # Add global handler wrapper
        async def handler(message: ChannelMessage) -> None:
            await self._dispatch_global(message, channel)
        
        channel.add_handler(handler)
        logger.info(f"Added channel: {channel.channel_id} ({channel.channel_type})")
    
    def remove_channel(self, channel_id: str) -> bool:
        """Remove a channel from the registry."""
        if channel_id in self._channels:
            del self._channels[channel_id]
            logger.info(f"Removed channel: {channel_id}")
            return True
        return False
    
    def get_channel(self, channel_id: str) -> Optional[Channel]:
        """Get a channel by ID."""
        return self._channels.get(channel_id)
    
    def get_channels_by_type(self, channel_type: str) -> List[Channel]:
        """Get all channels of a specific type."""
        return [
            c for c in self._channels.values()
            if c.channel_type == channel_type
        ]
    
    def list_channels(self) -> List[Channel]:
        """List all registered channels."""
        return list(self._channels.values())
    
    def add_global_handler(
        self,
        handler: Callable[[ChannelMessage, Channel], Awaitable[None]]
    ) -> None:
        """Add a handler that receives all messages from all channels."""
        self._global_handlers.append(handler)
    
    def remove_global_handler(
        self,
        handler: Callable[[ChannelMessage, Channel], Awaitable[None]]
    ) -> None:
        """Remove a global handler."""
        if handler in self._global_handlers:
            self._global_handlers.remove(handler)
    
    async def _dispatch_global(
        self,
        message: ChannelMessage,
        channel: Channel
    ) -> None:
        """Dispatch message to global handlers."""
        for handler in self._global_handlers:
            try:
                await handler(message, channel)
            except Exception as e:
                logger.error(f"Global handler error: {e}")
    
    async def send_to_channel(
        self,
        channel_id: str,
        target: str,
        content: str,
        **kwargs
    ) -> Optional[ChannelMessage]:
        """Send a message through a specific channel."""
        channel = self._channels.get(channel_id)
        if not channel:
            logger.warning(f"Channel not found: {channel_id}")
            return None
        
        if not channel.is_connected:
            logger.warning(f"Channel not connected: {channel_id}")
            return None
        
        return await channel.send(target, content, **kwargs)
    
    async def broadcast(
        self,
        content: str,
        channel_type: Optional[str] = None,
        targets: Optional[Dict[str, str]] = None
    ) -> Dict[str, Optional[ChannelMessage]]:
        """
        Broadcast a message to multiple channels.
        
        Args:
            content: Message content
            channel_type: Only send to channels of this type
            targets: Map of channel_id -> target, or uses default_target
        
        Returns:
            Dict of channel_id -> sent message (or None if failed)
        """
        results = {}
        channels = self._channels.values()
        
        if channel_type:
            channels = [c for c in channels if c.channel_type == channel_type]
        
        for channel in channels:
            if not channel.is_connected:
                continue
            
            target = None
            if targets and channel.channel_id in targets:
                target = targets[channel.channel_id]
            elif channel.config.default_target:
                target = channel.config.default_target
            else:
                continue  # No target specified
            
            try:
                result = await channel.send(target, content)
                results[channel.channel_id] = result
            except Exception as e:
                logger.error(f"Broadcast to {channel.channel_id} failed: {e}")
                results[channel.channel_id] = None
        
        return results
    
    async def connect_all(self) -> Dict[str, bool]:
        """Connect all channels."""
        results = {}
        for channel_id, channel in self._channels.items():
            try:
                results[channel_id] = await channel.connect()
            except Exception as e:
                logger.error(f"Failed to connect {channel_id}: {e}")
                results[channel_id] = False
        return results
    
    async def disconnect_all(self) -> None:
        """Disconnect all channels."""
        for channel in self._channels.values():
            try:
                await channel.disconnect()
            except Exception as e:
                logger.error(f"Failed to disconnect {channel.channel_id}: {e}")
    
    async def start(self) -> None:
        """Start all channels and begin receiving."""
        if self._running:
            return
        
        self._running = True
        await self.connect_all()
        
        # Start receive loops for all channels
        tasks = []
        for channel in self._channels.values():
            task = asyncio.create_task(self._receive_loop(channel))
            tasks.append(task)
        
        logger.info(f"Started {len(tasks)} channel receive loops")
    
    async def stop(self) -> None:
        """Stop all channels."""
        self._running = False
        await self.disconnect_all()
        logger.info("Stopped all channels")
    
    async def _receive_loop(self, channel: Channel) -> None:
        """Receive loop for a single channel."""
        while self._running:
            try:
                async for message in channel.receive():
                    await channel.dispatch_message(message)
            except Exception as e:
                logger.error(f"Receive error for {channel.channel_id}: {e}")
                await asyncio.sleep(5)  # Back off on error


# Global registry instance
_registry: Optional[ChannelRegistry] = None


def get_channel_registry() -> ChannelRegistry:
    """Get or create the global channel registry."""
    global _registry
    if _registry is None:
        _registry = ChannelRegistry()
    return _registry
