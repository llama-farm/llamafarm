"""
Channels Layer for OpenClaw Lite.

Provides messaging surface abstraction for:
- Telegram
- Slack
- Discord
- WhatsApp
- Webhooks
- WebSocket
"""

from .base import Channel, ChannelMessage, ChannelConfig
from .registry import ChannelRegistry, get_channel_registry
from .webhook import WebhookChannel
from .websocket import WebSocketChannel

__all__ = [
    "Channel",
    "ChannelMessage",
    "ChannelConfig",
    "ChannelRegistry",
    "get_channel_registry",
    "WebhookChannel",
    "WebSocketChannel",
]
