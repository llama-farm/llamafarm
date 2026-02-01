"""
Webhook Channel for receiving/sending messages via HTTP.

Provides:
- Inbound webhook endpoint for receiving messages
- Outbound HTTP POST for sending messages
"""

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional

import aiohttp

from .base import Channel, ChannelConfig, ChannelMessage, ChannelUser, MessageType

logger = logging.getLogger(__name__)


class WebhookChannel(Channel):
    """
    Webhook-based messaging channel.
    
    Receives messages via HTTP POST to an endpoint,
    sends responses via HTTP POST to configured URLs.
    """
    
    def __init__(self, config: ChannelConfig):
        super().__init__(config)
        self._message_queue: asyncio.Queue[ChannelMessage] = asyncio.Queue()
        self._http_session: Optional[aiohttp.ClientSession] = None
    
    async def connect(self) -> bool:
        """Initialize HTTP session."""
        try:
            self._http_session = aiohttp.ClientSession()
            self._connected = True
            logger.info(f"Webhook channel {self.channel_id} connected")
            return True
        except Exception as e:
            logger.error(f"Webhook connect failed: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Close HTTP session."""
        self._connected = False
        if self._http_session:
            await self._http_session.close()
            self._http_session = None
        logger.info(f"Webhook channel {self.channel_id} disconnected")
    
    async def send(
        self,
        target: str,
        content: str,
        message_type: MessageType = MessageType.TEXT,
        reply_to: Optional[str] = None,
        thread_id: Optional[str] = None,
        attachments: Optional[List[Dict]] = None
    ) -> Optional[ChannelMessage]:
        """Send a message via HTTP POST."""
        if not self._http_session:
            logger.error("HTTP session not initialized")
            return None
        
        await self._rate_limit_wait()
        
        # Use target as URL if it looks like a URL, otherwise use webhook_url
        url = target
        if not target.startswith("http"):
            url = self.config.webhook_url or f"https://{target}"
        
        payload = {
            "channel_id": self.channel_id,
            "content": content,
            "message_type": message_type.value,
            "timestamp": time.time(),
            "reply_to": reply_to,
            "thread_id": thread_id,
            "attachments": attachments or []
        }
        
        # Add auth headers if configured
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        try:
            async with self._http_session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status >= 400:
                    logger.warning(f"Webhook send failed: {resp.status}")
                    return None
                
                # Try to parse response for message ID
                try:
                    data = await resp.json()
                    message_id = data.get("id") or data.get("message_id") or self.create_message_id()
                except:
                    message_id = self.create_message_id()
                
                return ChannelMessage(
                    id=message_id,
                    channel_id=self.channel_id,
                    content=content,
                    message_type=message_type,
                    timestamp=time.time()
                )
                
        except asyncio.TimeoutError:
            logger.error(f"Webhook timeout: {url}")
            return None
        except Exception as e:
            logger.error(f"Webhook send error: {e}")
            return None
    
    async def receive(self) -> AsyncIterator[ChannelMessage]:
        """Yield messages from the queue."""
        while self._connected:
            try:
                # Wait for message with timeout so we can check connected status
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0
                )
                yield message
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Receive error: {e}")
                continue
    
    def enqueue_message(self, message: ChannelMessage) -> None:
        """
        Add a message to the receive queue.
        
        Called by the webhook endpoint handler.
        """
        self._message_queue.put_nowait(message)
    
    def parse_webhook_payload(
        self,
        data: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None
    ) -> Optional[ChannelMessage]:
        """
        Parse an incoming webhook payload into a ChannelMessage.
        
        Override this to handle platform-specific webhook formats.
        
        Generic format expected:
        {
            "id": "msg-id",
            "content": "message text",
            "sender": {"id": "user-id", "name": "User Name"},
            "timestamp": 1234567890,
            "thread_id": "optional-thread",
            "reply_to": "optional-reply-to"
        }
        """
        try:
            sender_data = data.get("sender", {})
            sender = ChannelUser(
                id=sender_data.get("id", "unknown"),
                name=sender_data.get("name"),
                username=sender_data.get("username")
            ) if sender_data else None
            
            message_type = MessageType.TEXT
            if data.get("message_type"):
                try:
                    message_type = MessageType(data["message_type"])
                except ValueError:
                    pass
            
            return ChannelMessage(
                id=data.get("id") or self.create_message_id(),
                channel_id=self.channel_id,
                content=data.get("content", ""),
                message_type=message_type,
                timestamp=data.get("timestamp", time.time()),
                sender=sender,
                thread_id=data.get("thread_id"),
                reply_to=data.get("reply_to"),
                attachments=data.get("attachments", []),
                raw_message=data,
                metadata=data.get("metadata", {})
            )
            
        except Exception as e:
            logger.error(f"Failed to parse webhook payload: {e}")
            return None


def create_webhook_handler(channel: WebhookChannel):
    """
    Create a FastAPI route handler for webhook ingestion.
    
    Usage:
        from fastapi import FastAPI, Request
        app = FastAPI()
        
        channel = WebhookChannel(config)
        
        @app.post("/webhook/{channel_id}")
        async def webhook(request: Request, channel_id: str):
            return await create_webhook_handler(channel)(request)
    """
    async def handler(request):
        from fastapi import Request
        from fastapi.responses import JSONResponse
        
        try:
            data = await request.json()
        except:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        
        # Verify token if configured
        if channel.config.token:
            auth_header = request.headers.get("Authorization", "")
            expected = f"Bearer {channel.config.token}"
            if auth_header != expected:
                return JSONResponse({"error": "Unauthorized"}, status_code=401)
        
        # Parse and enqueue
        message = channel.parse_webhook_payload(
            data,
            dict(request.headers)
        )
        
        if message:
            channel.enqueue_message(message)
            return JSONResponse({"status": "ok", "message_id": message.id})
        else:
            return JSONResponse({"error": "Failed to parse message"}, status_code=400)
    
    return handler
