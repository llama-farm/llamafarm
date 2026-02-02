"""
Network layer for Atmosphere gossip.

Real TCP/UDP communication between mesh nodes.
Enables communication over LAN and internet.
"""

import asyncio
import json
import logging
import socket
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Set, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

# Protocol constants
DEFAULT_PORT = 11450
RECONNECT_DELAY_SEC = 5
MAX_RECONNECT_DELAY_SEC = 300
PEER_TIMEOUT_SEC = 120
HEARTBEAT_INTERVAL_SEC = 30


class MessageType(str, Enum):
    """Gossip message types."""
    HANDSHAKE = "handshake"
    HANDSHAKE_ACK = "handshake_ack"
    HEARTBEAT = "heartbeat"
    CAPABILITY_ANNOUNCE = "capability_announce"
    ROUTE_UPDATE = "route_update"
    SESSION_UPDATE = "session_update"
    REVOCATION = "revocation"
    INTENT = "intent"
    INTENT_RESPONSE = "intent_response"


@dataclass
class Peer:
    """A connected peer node."""
    node_id: str
    address: str  # host:port
    reader: Optional[asyncio.StreamReader] = None
    writer: Optional[asyncio.StreamWriter] = None
    connected: bool = False
    last_seen: float = field(default_factory=time.time)
    capabilities: List[str] = field(default_factory=list)
    reconnect_delay: float = RECONNECT_DELAY_SEC
    
    def touch(self):
        self.last_seen = time.time()
    
    @property
    def is_stale(self) -> bool:
        return time.time() - self.last_seen > PEER_TIMEOUT_SEC


@dataclass
class GossipMessage:
    """A message in the gossip protocol."""
    type: MessageType
    sender_id: str
    payload: Dict
    nonce: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: float = field(default_factory=time.time)
    ttl: int = 10
    
    def to_json(self) -> str:
        return json.dumps({
            "type": self.type.value if isinstance(self.type, MessageType) else self.type,
            "sender_id": self.sender_id,
            "payload": self.payload,
            "nonce": self.nonce,
            "timestamp": self.timestamp,
            "ttl": self.ttl
        })
    
    @classmethod
    def from_json(cls, data: str) -> "GossipMessage":
        obj = json.loads(data)
        return cls(
            type=MessageType(obj["type"]),
            sender_id=obj["sender_id"],
            payload=obj["payload"],
            nonce=obj["nonce"],
            timestamp=obj["timestamp"],
            ttl=obj.get("ttl", 10)
        )


class GossipNetwork:
    """
    Real network gossip over TCP.
    
    Manages peer connections and message propagation.
    """
    
    def __init__(
        self,
        node_id: str,
        port: int = DEFAULT_PORT,
        bind_host: str = "0.0.0.0"
    ):
        self.node_id = node_id
        self.port = port
        self.bind_host = bind_host
        
        # Peer management
        self.peers: Dict[str, Peer] = {}
        self.seed_peers: List[str] = []  # host:port format
        
        # Message handling
        self._handlers: Dict[MessageType, List[Callable]] = {}
        self._seen_nonces: Set[str] = set()
        self._nonce_expiry: Dict[str, float] = {}
        
        # Server
        self._server: Optional[asyncio.Server] = None
        self._running = False
        
        # Background tasks
        self._tasks: List[asyncio.Task] = []
    
    async def start(self, seed_peers: List[str] = None):
        """
        Start the gossip network.
        
        Args:
            seed_peers: List of seed peer addresses (host:port)
        """
        if seed_peers:
            self.seed_peers = seed_peers
        
        # Start TCP server
        self._server = await asyncio.start_server(
            self._handle_connection,
            self.bind_host,
            self.port
        )
        self._running = True
        
        addrs = [str(s.getsockname()) for s in self._server.sockets]
        logger.info(f"Gossip server listening on {addrs}")
        
        # Start background tasks
        self._tasks.append(asyncio.create_task(self._connect_seeds()))
        self._tasks.append(asyncio.create_task(self._heartbeat_loop()))
        self._tasks.append(asyncio.create_task(self._cleanup_loop()))
        
        # Start mDNS discovery (LAN)
        self._tasks.append(asyncio.create_task(self._mdns_discovery()))
    
    async def stop(self):
        """Stop the gossip network."""
        self._running = False
        
        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Close peer connections
        for peer in self.peers.values():
            await self._close_peer(peer)
        
        # Close server
        if self._server:
            self._server.close()
            await self._server.wait_closed()
    
    # === Connection Management ===
    
    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle incoming peer connection."""
        addr = writer.get_extra_info('peername')
        logger.info(f"Incoming connection from {addr}")
        
        try:
            # Wait for handshake
            line = await asyncio.wait_for(reader.readline(), timeout=10.0)
            if not line:
                return
            
            msg = GossipMessage.from_json(line.decode())
            
            if msg.type != MessageType.HANDSHAKE:
                logger.warning(f"Expected handshake, got {msg.type}")
                return
            
            peer_id = msg.sender_id
            peer_addr = f"{addr[0]}:{msg.payload.get('port', DEFAULT_PORT)}"
            
            # Create or update peer
            if peer_id in self.peers:
                peer = self.peers[peer_id]
                peer.reader = reader
                peer.writer = writer
                peer.connected = True
            else:
                peer = Peer(
                    node_id=peer_id,
                    address=peer_addr,
                    reader=reader,
                    writer=writer,
                    connected=True,
                    capabilities=msg.payload.get("capabilities", [])
                )
                self.peers[peer_id] = peer
            
            # Send handshake ack
            ack = GossipMessage(
                type=MessageType.HANDSHAKE_ACK,
                sender_id=self.node_id,
                payload={
                    "port": self.port,
                    "capabilities": []  # Will be filled by caller
                }
            )
            writer.write((ack.to_json() + "\n").encode())
            await writer.drain()
            
            logger.info(f"Peer connected: {peer_id} from {peer_addr}")
            
            # Handle messages
            await self._read_messages(peer, reader)
            
        except asyncio.TimeoutError:
            logger.warning(f"Connection timeout from {addr}")
        except Exception as e:
            logger.error(f"Connection error from {addr}: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
    
    async def _connect_peer(self, address: str) -> Optional[Peer]:
        """Connect to a peer by address."""
        try:
            host, port_str = address.split(":")
            port = int(port_str)
            
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=10.0
            )
            
            # Send handshake
            handshake = GossipMessage(
                type=MessageType.HANDSHAKE,
                sender_id=self.node_id,
                payload={
                    "port": self.port,
                    "capabilities": []
                }
            )
            writer.write((handshake.to_json() + "\n").encode())
            await writer.drain()
            
            # Wait for ack
            line = await asyncio.wait_for(reader.readline(), timeout=10.0)
            msg = GossipMessage.from_json(line.decode())
            
            if msg.type != MessageType.HANDSHAKE_ACK:
                logger.warning(f"Expected handshake_ack, got {msg.type}")
                writer.close()
                await writer.wait_closed()
                return None
            
            peer_id = msg.sender_id
            
            peer = Peer(
                node_id=peer_id,
                address=address,
                reader=reader,
                writer=writer,
                connected=True,
                capabilities=msg.payload.get("capabilities", [])
            )
            self.peers[peer_id] = peer
            
            logger.info(f"Connected to peer: {peer_id} at {address}")
            
            # Start reading messages in background
            asyncio.create_task(self._read_messages(peer, reader))
            
            return peer
            
        except asyncio.TimeoutError:
            logger.warning(f"Connection timeout to {address}")
        except ConnectionRefusedError:
            logger.debug(f"Connection refused by {address}")
        except Exception as e:
            logger.warning(f"Failed to connect to {address}: {e}")
        
        return None
    
    async def _read_messages(self, peer: Peer, reader: asyncio.StreamReader):
        """Read messages from peer."""
        try:
            while self._running and peer.connected:
                try:
                    line = await asyncio.wait_for(reader.readline(), timeout=PEER_TIMEOUT_SEC)
                except asyncio.TimeoutError:
                    logger.warning(f"Read timeout from {peer.node_id}, disconnecting")
                    break
                
                if not line:
                    logger.info(f"Connection closed by {peer.node_id}")
                    break
                
                try:
                    msg = GossipMessage.from_json(line.decode())
                    peer.touch()
                    await self._handle_message(peer, msg)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON from {peer.node_id}: {e}")
                    # Don't disconnect on parse errors, just skip
                except Exception as e:
                    logger.error(f"Error handling message from {peer.node_id}: {e}", exc_info=True)
        
        except asyncio.CancelledError:
            logger.debug(f"Message reader cancelled for {peer.node_id}")
        except ConnectionResetError:
            logger.info(f"Connection reset by {peer.node_id}")
        except Exception as e:
            logger.warning(f"Connection lost to {peer.node_id}: {e}")
        finally:
            peer.connected = False
            logger.info(f"Peer disconnected: {peer.node_id}")
    
    async def _close_peer(self, peer: Peer):
        """Close peer connection."""
        peer.connected = False
        if peer.writer:
            try:
                peer.writer.close()
                await peer.writer.wait_closed()
            except Exception:
                pass
    
    # === Message Handling ===
    
    def on(self, msg_type: MessageType, handler: Callable):
        """Register a message handler."""
        if msg_type not in self._handlers:
            self._handlers[msg_type] = []
        self._handlers[msg_type].append(handler)
    
    async def _handle_message(self, peer: Peer, msg: GossipMessage):
        """Handle incoming message."""
        # Dedup by nonce
        if msg.nonce in self._seen_nonces:
            return
        self._seen_nonces.add(msg.nonce)
        self._nonce_expiry[msg.nonce] = time.time() + 300  # 5 min
        
        # Call registered handlers
        handlers = self._handlers.get(msg.type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(peer, msg)
                else:
                    handler(peer, msg)
            except Exception as e:
                logger.error(f"Handler error for {msg.type}: {e}")
        
        # Propagate if TTL > 0
        if msg.ttl > 1:
            msg.ttl -= 1
            await self._propagate(msg, exclude={peer.node_id})
    
    async def broadcast(self, msg_type: MessageType, payload: Dict, ttl: int = 10):
        """Broadcast message to all peers."""
        msg = GossipMessage(
            type=msg_type,
            sender_id=self.node_id,
            payload=payload,
            ttl=ttl
        )
        self._seen_nonces.add(msg.nonce)
        await self._propagate(msg)
    
    async def send(self, peer_id: str, msg_type: MessageType, payload: Dict):
        """Send message to specific peer."""
        peer = self.peers.get(peer_id)
        if not peer or not peer.connected or not peer.writer:
            logger.debug(f"Cannot send to {peer_id}: peer not connected")
            return False
        
        msg = GossipMessage(
            type=msg_type,
            sender_id=self.node_id,
            payload=payload,
            ttl=1
        )
        
        try:
            data = (msg.to_json() + "\n").encode()
            peer.writer.write(data)
            await asyncio.wait_for(peer.writer.drain(), timeout=5.0)
            return True
        except asyncio.TimeoutError:
            logger.warning(f"Timeout sending to {peer_id}")
            peer.connected = False
            return False
        except ConnectionResetError:
            logger.info(f"Connection reset while sending to {peer_id}")
            peer.connected = False
            return False
        except Exception as e:
            logger.warning(f"Failed to send to {peer_id}: {e}")
            peer.connected = False
            return False
    
    async def _propagate(self, msg: GossipMessage, exclude: Set[str] = None):
        """Propagate message to peers."""
        exclude = exclude or set()
        exclude.add(self.node_id)
        
        data = (msg.to_json() + "\n").encode()
        failed_peers = []
        
        for peer_id, peer in list(self.peers.items()):
            if peer_id in exclude or not peer.connected or not peer.writer:
                continue
            
            try:
                peer.writer.write(data)
                await asyncio.wait_for(peer.writer.drain(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(f"Timeout propagating to {peer_id}")
                peer.connected = False
                failed_peers.append(peer_id)
            except ConnectionResetError:
                logger.debug(f"Connection reset while propagating to {peer_id}")
                peer.connected = False
                failed_peers.append(peer_id)
            except Exception as e:
                logger.warning(f"Failed to propagate to {peer_id}: {e}")
                peer.connected = False
                failed_peers.append(peer_id)
        
        if failed_peers:
            logger.debug(f"Message propagation failed to {len(failed_peers)} peer(s)")
    
    # === Background Tasks ===
    
    async def _connect_seeds(self):
        """Connect to seed peers with exponential backoff retry."""
        retry_delays = {}  # Track per-peer retry delays
        
        while self._running:
            for address in self.seed_peers:
                # Check if already connected
                existing = any(
                    p.address == address and p.connected
                    for p in self.peers.values()
                )
                if not existing:
                    # Initialize delay for new peer
                    if address not in retry_delays:
                        retry_delays[address] = RECONNECT_DELAY_SEC
                    
                    peer = await self._connect_peer(address)
                    
                    if peer:
                        # Connection successful - reset delay
                        retry_delays[address] = RECONNECT_DELAY_SEC
                    else:
                        # Connection failed - exponential backoff
                        retry_delays[address] = min(
                            retry_delays[address] * 2,
                            MAX_RECONNECT_DELAY_SEC
                        )
                        logger.debug(
                            f"Next retry for {address} in {retry_delays[address]}s"
                        )
            
            await asyncio.sleep(30)
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats."""
        while self._running:
            await asyncio.sleep(HEARTBEAT_INTERVAL_SEC)
            
            await self.broadcast(
                MessageType.HEARTBEAT,
                {"time": time.time(), "peers": len(self.peers)}
            )
    
    async def _cleanup_loop(self):
        """Clean up stale peers and nonces."""
        while self._running:
            await asyncio.sleep(60)
            
            now = time.time()
            
            # Prune stale peers
            stale = [
                peer_id for peer_id, peer in self.peers.items()
                if peer.is_stale and not peer.connected
            ]
            for peer_id in stale:
                del self.peers[peer_id]
                logger.info(f"Pruned stale peer: {peer_id}")
            
            # Prune old nonces
            expired = [
                nonce for nonce, expiry in self._nonce_expiry.items()
                if now > expiry
            ]
            for nonce in expired:
                self._seen_nonces.discard(nonce)
                del self._nonce_expiry[nonce]
    
    async def _mdns_discovery(self):
        """Discover peers on local network via mDNS."""
        # Placeholder - would use zeroconf or similar
        # For now, LAN discovery happens via seed peers
        pass
    
    # === Status ===
    
    @property
    def connected_peers(self) -> List[str]:
        return [p.node_id for p in self.peers.values() if p.connected]
    
    def get_status(self) -> Dict:
        return {
            "node_id": self.node_id,
            "port": self.port,
            "running": self._running,
            "peers": {
                peer_id: {
                    "address": peer.address,
                    "connected": peer.connected,
                    "last_seen": peer.last_seen,
                    "capabilities": peer.capabilities
                }
                for peer_id, peer in self.peers.items()
            }
        }
