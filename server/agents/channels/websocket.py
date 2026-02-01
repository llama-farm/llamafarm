"""
WebSocket Channel for real-time bidirectional messaging.
"""

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Set

import aiohttp

from .base import Channel, ChannelConfig, ChannelMessage, ChannelUser, MessageType

logger = logging.getLogger(__name__)


class WebSocketChannel(Channel):
    """
    WebSocket-based messaging channel.
    
    Supports:
    - Real-time bidirectional messaging
    - Connection management with reconnection
    - Multiple client connections (server mode)
    """
    
    def __init__(self, config: ChannelConfig):
        super().__init__(config)
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._message_queue: asyncio.Queue[ChannelMessage] = asyncio.Queue()
        self._receive_task: Optional[asyncio.Task] = None
        
        # Server mode: track connected clients
        self._clients: Set[Any] = set()  # WebSocket connections
    
    async def connect(self) -> bool:
        """Connect to WebSocket server."""
        ws_url = self.config.metadata.get("ws_url")
        if not ws_url:
            # Server mode - we accept connections, not connect
            self._connected = True
            logger.info(f"WebSocket channel {self.channel_id} ready (server mode)")
            return True
        
        # Client mode - connect to URL
        try:
            self._http_session = aiohttp.ClientSession()
            
            headers = {}
            if self.config.token:
                headers["Authorization"] = f"Bearer {self.config.token}"
            
            self._ws = await self._http_session.ws_connect(
                ws_url,
                headers=headers,
                heartbeat=30
            )
            
            self._connected = True
            
            # Start receive loop
            self._receive_task = asyncio.create_task(self._ws_receive_loop())
            
            logger.info(f"WebSocket channel {self.channel_id} connected to {ws_url}")
            return True
            
        except Exception as e:
            logger.error(f"WebSocket connect failed: {e}")
            self._connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from WebSocket server."""
        self._connected = False
        
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        
        if self._ws:
            await self._ws.close()
            self._ws = None
        
        if self._http_session:
            await self._http_session.close()
            self._http_session = None
        
        # Close all client connections (server mode)
        for client in list(self._clients):
            try:
                await client.close()
            except:
                pass
        self._clients.clear()
        
        logger.info(f"WebSocket channel {self.channel_id} disconnected")
    
    async def send(
        self,
        target: str,
        content: str,
        message_type: MessageType = MessageType.TEXT,
        reply_to: Optional[str] = None,
        thread_id: Optional[str] = None,
        attachments: Optional[List[Dict]] = None
    ) -> Optional[ChannelMessage]:
        """Send a message via WebSocket."""
        await self._rate_limit_wait()
        
        message_id = self.create_message_id()
        payload = {
            "type": "message",
            "id": message_id,
            "target": target,
            "content": content,
            "message_type": message_type.value,
            "timestamp": time.time(),
            "reply_to": reply_to,
            "thread_id": thread_id,
            "attachments": attachments or []
        }
        
        # Client mode: send to server
        if self._ws:
            try:
                await self._ws.send_json(payload)
                return ChannelMessage(
                    id=message_id,
                    channel_id=self.channel_id,
                    content=content,
                    message_type=message_type,
                    timestamp=time.time()
                )
            except Exception as e:
                logger.error(f"WebSocket send error: {e}")
                return None
        
        # Server mode: broadcast to matching clients
        sent = False
        for client in list(self._clients):
            try:
                # If target is "all" or matches client ID
                client_id = getattr(client, "client_id", None)
                if target == "all" or target == client_id:
                    await client.send_json(payload)
                    sent = True
            except Exception as e:
                logger.warning(f"Failed to send to client: {e}")
                self._clients.discard(client)
        
        if sent:
            return ChannelMessage(
                id=message_id,
                channel_id=self.channel_id,
                content=content,
                message_type=message_type,
                timestamp=time.time()
            )
        return None
    
    async def receive(self) -> AsyncIterator[ChannelMessage]:
        """Yield messages from the queue."""
        while self._connected:
            try:
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
    
    async def _ws_receive_loop(self) -> None:
        """Receive loop for client mode."""
        if not self._ws:
            return
        
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        message = self._parse_ws_message(data)
                        if message:
                            self._message_queue.put_nowait(message)
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON received")
                
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {self._ws.exception()}")
                    break
                
                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                    logger.info("WebSocket closed by server")
                    break
                    
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"WebSocket receive loop error: {e}")
        
        # Attempt reconnection
        if self._connected:
            logger.info("Attempting WebSocket reconnection...")
            await asyncio.sleep(5)
            await self.reconnect()
    
    def _parse_ws_message(self, data: Dict[str, Any]) -> Optional[ChannelMessage]:
        """Parse a WebSocket message into ChannelMessage."""
        try:
            msg_type = data.get("type", "message")
            if msg_type != "message":
                return None  # Ignore non-message types (ping, etc.)
            
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
            logger.error(f"Failed to parse WebSocket message: {e}")
            return None
    
    # --- Server Mode Methods ---
    
    def add_client(self, ws: Any, client_id: Optional[str] = None) -> None:
        """
        Add a client connection (server mode).
        
        Called when a new WebSocket client connects.
        """
        if client_id:
            ws.client_id = client_id
        self._clients.add(ws)
        logger.info(f"Client connected: {client_id or 'anonymous'}")
    
    def remove_client(self, ws: Any) -> None:
        """Remove a client connection."""
        self._clients.discard(ws)
        client_id = getattr(ws, "client_id", "anonymous")
        logger.info(f"Client disconnected: {client_id}")
    
    async def handle_client_message(
        self,
        ws: Any,
        data: Dict[str, Any]
    ) -> None:
        """
        Handle a message from a connected client (server mode).
        
        Called by the WebSocket endpoint handler.
        """
        message = self._parse_ws_message(data)
        if message:
            # Add sender info from client
            client_id = getattr(ws, "client_id", "anonymous")
            if not message.sender:
                message.sender = ChannelUser(id=client_id)
            
            self._message_queue.put_nowait(message)
    
    @property
    def client_count(self) -> int:
        """Number of connected clients (server mode)."""
        return len(self._clients)


def create_websocket_handler(channel: WebSocketChannel):
    """
    Create a FastAPI WebSocket endpoint handler.
    
    Usage:
        from fastapi import FastAPI, WebSocket
        app = FastAPI()
        
        channel = WebSocketChannel(config)
        
        @app.websocket("/ws/{channel_id}")
        async def websocket(websocket: WebSocket, channel_id: str):
            await create_websocket_handler(channel)(websocket)
    """
    async def handler(websocket):
        from fastapi import WebSocket
        
        # Accept connection
        await websocket.accept()
        
        # Get client ID from query params or headers
        client_id = websocket.query_params.get("client_id")
        if not client_id:
            client_id = websocket.headers.get("X-Client-ID")
        
        channel.add_client(websocket, client_id)
        
        try:
            while True:
                data = await websocket.receive_json()
                await channel.handle_client_message(websocket, data)
        except Exception as e:
            logger.debug(f"WebSocket connection closed: {e}")
        finally:
            channel.remove_client(websocket)
    
    return handler
