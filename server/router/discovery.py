"""
Peer discovery for mesh networking.

Discovers other LlamaFarm nodes on the local network using:
- mDNS/Bonjour for service discovery
- UDP broadcast for simple discovery
- Manual peer configuration
"""

import asyncio
import json
import socket
from typing import Callable, Awaitable, Set, Optional
import logging

logger = logging.getLogger(__name__)


# Discovery configuration
DISCOVERY_PORT = 47471  # UDP port for mesh discovery
DISCOVERY_INTERVAL_SEC = 30
MDNS_SERVICE_TYPE = "_llamafarm-mesh._udp.local."


class PeerDiscovery:
    """
    Discover LlamaFarm mesh peers on the local network.
    
    Uses UDP broadcast for simple discovery. Can be extended with
    mDNS/Bonjour for more robust service discovery.
    
    Usage:
        discovery = PeerDiscovery(
            node_id="my-node",
            port=47471,
            on_peer_found=handle_peer
        )
        await discovery.start()
    """
    
    def __init__(
        self,
        node_id: str,
        port: int = DISCOVERY_PORT,
        on_peer_found: Optional[Callable[[str, str, int], Awaitable[None]]] = None
    ):
        self.node_id = node_id
        self.port = port
        self.on_peer_found = on_peer_found
        
        self._running = False
        self._socket: Optional[socket.socket] = None
        self._tasks: list[asyncio.Task] = []
        self._known_peers: Set[str] = set()
    
    async def start(self) -> None:
        """Start peer discovery."""
        if self._running:
            return
        
        self._running = True
        
        # Create UDP socket
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._socket.setblocking(False)
        
        # Bind to discovery port
        try:
            self._socket.bind(('', self.port))
            logger.info(f"Listening for mesh peers on port {self.port}")
        except OSError as e:
            logger.error(f"Failed to bind discovery port {self.port}: {e}")
            return
        
        # Start broadcast and listen tasks
        self._tasks.append(asyncio.create_task(self._broadcast_loop()))
        self._tasks.append(asyncio.create_task(self._listen_loop()))
        
        logger.info("Peer discovery started")
    
    async def stop(self) -> None:
        """Stop peer discovery."""
        self._running = False
        
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        if self._socket:
            self._socket.close()
        
        logger.info("Peer discovery stopped")
    
    async def _broadcast_loop(self) -> None:
        """Periodically broadcast our presence."""
        while self._running:
            try:
                await self._broadcast_presence()
            except Exception as e:
                logger.error(f"Broadcast failed: {e}")
            
            await asyncio.sleep(DISCOVERY_INTERVAL_SEC)
    
    async def _broadcast_presence(self) -> None:
        """Broadcast our node info."""
        message = {
            "type": "mesh_discovery",
            "node_id": self.node_id,
            "port": self.port
        }
        
        data = json.dumps(message).encode()
        
        # Broadcast to subnet
        try:
            self._socket.sendto(data, ('<broadcast>', self.port))
            logger.debug(f"Broadcast discovery message")
        except Exception as e:
            logger.error(f"Failed to broadcast: {e}")
    
    async def _listen_loop(self) -> None:
        """Listen for discovery messages."""
        loop = asyncio.get_event_loop()
        
        while self._running:
            try:
                # Non-blocking receive
                data, addr = await loop.sock_recvfrom(self._socket, 4096)
                await self._handle_discovery_message(data, addr)
            except Exception as e:
                if self._running:
                    logger.error(f"Listen error: {e}")
                await asyncio.sleep(0.1)
    
    async def _handle_discovery_message(self, data: bytes, addr: tuple) -> None:
        """Handle incoming discovery message."""
        try:
            message = json.loads(data.decode())
            
            if message.get("type") != "mesh_discovery":
                return
            
            peer_id = message.get("node_id")
            peer_host = addr[0]
            peer_port = message.get("port", self.port)
            
            # Don't discover ourselves
            if peer_id == self.node_id:
                return
            
            # Check if this is a new peer
            peer_key = f"{peer_id}@{peer_host}:{peer_port}"
            if peer_key not in self._known_peers:
                self._known_peers.add(peer_key)
                logger.info(f"Discovered peer: {peer_id} at {peer_host}:{peer_port}")
                
                if self.on_peer_found:
                    await self.on_peer_found(peer_id, peer_host, peer_port)
        
        except (json.JSONDecodeError, KeyError) as e:
            logger.debug(f"Invalid discovery message from {addr}: {e}")
    
    def get_known_peers(self) -> Set[str]:
        """Get set of known peer IDs."""
        return {p.split('@')[0] for p in self._known_peers}


class StaticPeerRegistry:
    """
    Registry for manually configured peers.
    
    Used when auto-discovery is not available or for connecting
    to peers on different networks.
    """
    
    def __init__(self):
        self._peers: dict[str, tuple[str, int]] = {}
    
    def add_peer(self, node_id: str, host: str, port: int) -> None:
        """Add a peer manually."""
        self._peers[node_id] = (host, port)
        logger.info(f"Registered static peer: {node_id} at {host}:{port}")
    
    def remove_peer(self, node_id: str) -> None:
        """Remove a peer."""
        if node_id in self._peers:
            del self._peers[node_id]
            logger.info(f"Removed static peer: {node_id}")
    
    def get_peer(self, node_id: str) -> Optional[tuple[str, int]]:
        """Get peer address."""
        return self._peers.get(node_id)
    
    def list_peers(self) -> dict[str, tuple[str, int]]:
        """List all registered peers."""
        return self._peers.copy()
