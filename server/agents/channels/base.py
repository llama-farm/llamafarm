"""
Base Channel abstraction for messaging surfaces.

Channels provide a unified interface for:
- Receiving messages from external platforms
- Sending responses back
- Handling platform-specific features (reactions, threads, etc.)
"""

import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Awaitable

logger = logging.getLogger(__name__)


class MessageType(Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"
    LOCATION = "location"
    REACTION = "reaction"
    REPLY = "reply"
    SYSTEM = "system"


@dataclass
class ChannelUser:
    """A user on a channel."""
    id: str
    name: Optional[str] = None
    username: Optional[str] = None
    avatar_url: Optional[str] = None
    is_bot: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChannelMessage:
    """
    A message from/to a channel.
    
    Unified representation that works across platforms.
    """
    id: str
    channel_id: str
    content: str
    message_type: MessageType = MessageType.TEXT
    timestamp: float = field(default_factory=time.time)
    
    # Sender info
    sender: Optional[ChannelUser] = None
    
    # Threading
    thread_id: Optional[str] = None
    reply_to: Optional[str] = None
    
    # Attachments
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    
    # Platform-specific
    raw_message: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "channel_id": self.channel_id,
            "content": self.content,
            "message_type": self.message_type.value,
            "timestamp": self.timestamp,
            "sender": {
                "id": self.sender.id,
                "name": self.sender.name,
                "username": self.sender.username
            } if self.sender else None,
            "thread_id": self.thread_id,
            "reply_to": self.reply_to,
            "attachments": self.attachments,
            "metadata": self.metadata
        }


@dataclass
class ChannelConfig:
    """Configuration for a channel."""
    channel_type: str  # telegram, slack, discord, etc.
    channel_id: str
    name: Optional[str] = None
    
    # Authentication
    token: Optional[str] = None
    api_key: Optional[str] = None
    webhook_url: Optional[str] = None
    
    # Features
    supports_threads: bool = False
    supports_reactions: bool = False
    supports_edits: bool = False
    supports_voice: bool = False
    
    # Targeting
    default_target: Optional[str] = None  # Default chat/channel to send to
    
    # Rate limiting
    rate_limit_per_sec: float = 1.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class Channel(ABC):
    """
    Abstract base class for messaging channels.
    
    Implement this to add support for new platforms.
    """
    
    def __init__(self, config: ChannelConfig):
        self.config = config
        self._message_handlers: List[Callable[[ChannelMessage], Awaitable[None]]] = []
        self._connected = False
        self._last_message_time = 0.0
    
    @property
    def channel_id(self) -> str:
        return self.config.channel_id
    
    @property
    def channel_type(self) -> str:
        return self.config.channel_type
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    # --- Lifecycle ---
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the channel.
        
        Returns True if connection successful.
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the channel."""
        pass
    
    async def reconnect(self) -> bool:
        """Reconnect to the channel."""
        await self.disconnect()
        return await self.connect()
    
    # --- Sending ---
    
    @abstractmethod
    async def send(
        self,
        target: str,
        content: str,
        message_type: MessageType = MessageType.TEXT,
        reply_to: Optional[str] = None,
        thread_id: Optional[str] = None,
        attachments: Optional[List[Dict]] = None
    ) -> Optional[ChannelMessage]:
        """
        Send a message to the channel.
        
        Args:
            target: Target chat/channel/user ID
            content: Message content
            message_type: Type of message
            reply_to: Message ID to reply to
            thread_id: Thread to send in
            attachments: File attachments
            
        Returns:
            The sent message, or None if failed
        """
        pass
    
    async def reply(
        self,
        message: ChannelMessage,
        content: str,
        message_type: MessageType = MessageType.TEXT
    ) -> Optional[ChannelMessage]:
        """Reply to a message."""
        return await self.send(
            target=message.channel_id,
            content=content,
            message_type=message_type,
            reply_to=message.id,
            thread_id=message.thread_id
        )
    
    async def react(
        self,
        message: ChannelMessage,
        emoji: str
    ) -> bool:
        """
        React to a message with an emoji.
        
        Override in subclasses that support reactions.
        """
        if not self.config.supports_reactions:
            logger.warning(f"Channel {self.channel_id} doesn't support reactions")
            return False
        return False
    
    async def edit(
        self,
        message_id: str,
        new_content: str,
        target: Optional[str] = None
    ) -> bool:
        """
        Edit a sent message.
        
        Override in subclasses that support edits.
        """
        if not self.config.supports_edits:
            logger.warning(f"Channel {self.channel_id} doesn't support edits")
            return False
        return False
    
    async def delete(
        self,
        message_id: str,
        target: Optional[str] = None
    ) -> bool:
        """
        Delete a message.
        
        Override in subclasses that support deletion.
        """
        return False
    
    # --- Receiving ---
    
    @abstractmethod
    async def receive(self) -> AsyncIterator[ChannelMessage]:
        """
        Receive messages from the channel.
        
        Yields ChannelMessage objects as they arrive.
        Override this to implement platform-specific receiving.
        """
        yield  # Make this a generator
        return
    
    def add_handler(
        self,
        handler: Callable[[ChannelMessage], Awaitable[None]]
    ) -> None:
        """Add a message handler."""
        self._message_handlers.append(handler)
    
    def remove_handler(
        self,
        handler: Callable[[ChannelMessage], Awaitable[None]]
    ) -> None:
        """Remove a message handler."""
        if handler in self._message_handlers:
            self._message_handlers.remove(handler)
    
    async def dispatch_message(self, message: ChannelMessage) -> None:
        """Dispatch a message to all handlers."""
        for handler in self._message_handlers:
            try:
                await handler(message)
            except Exception as e:
                logger.error(f"Handler error for message {message.id}: {e}")
    
    # --- Utilities ---
    
    def _rate_limit_check(self) -> bool:
        """Check if we're within rate limits."""
        now = time.time()
        min_interval = 1.0 / self.config.rate_limit_per_sec
        if now - self._last_message_time < min_interval:
            return False
        self._last_message_time = now
        return True
    
    async def _rate_limit_wait(self) -> None:
        """Wait to respect rate limits."""
        import asyncio
        now = time.time()
        min_interval = 1.0 / self.config.rate_limit_per_sec
        elapsed = now - self._last_message_time
        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)
        self._last_message_time = time.time()
    
    def create_message_id(self) -> str:
        """Generate a message ID."""
        return f"msg-{uuid.uuid4().hex[:12]}"
