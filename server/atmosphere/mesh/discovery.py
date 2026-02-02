"""
mDNS/DNS-SD discovery for Atmosphere mesh.

Enables automatic peer discovery on local networks using
multicast DNS (like Apple's Bonjour / Matter).
"""

import asyncio
import json
import logging
import socket
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

try:
    from zeroconf import ServiceBrowser, ServiceInfo, Zeroconf, ServiceStateChange
    from zeroconf.asyncio import AsyncZeroconf, AsyncServiceBrowser
    ZEROCONF_AVAILABLE = True
except ImportError:
    ZEROCONF_AVAILABLE = False

logger = logging.getLogger(__name__)

# Service type for Atmosphere mesh nodes
SERVICE_TYPE = "_atmosphere._tcp.local."
SERVICE_NAME_PREFIX = "atmosphere-"


@dataclass
class DiscoveredPeer:
    """A discovered peer node."""
    node_id: str
    name: str
    host: str
    port: int
    mesh_id: Optional[str] = None
    capabilities: List[str] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []
    
    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


class MeshDiscovery:
    """
    mDNS-based mesh discovery.
    
    Advertises this node and discovers peers on the local network.
    
    Usage:
        discovery = MeshDiscovery(
            node_id="abc123",
            port=11434,
            mesh_id="mymesh"
        )
        
        # Set callback for new peers
        discovery.on_peer_found = lambda peer: print(f"Found: {peer.name}")
        
        await discovery.start()
        # ... later ...
        await discovery.stop()
    """
    
    def __init__(
        self,
        node_id: str,
        port: int,
        name: Optional[str] = None,
        mesh_id: Optional[str] = None,
        capabilities: Optional[List[str]] = None
    ):
        self.node_id = node_id
        self.port = port
        self.name = name or f"atmosphere-{node_id[:8]}"
        self.mesh_id = mesh_id
        self.capabilities = capabilities or []
        
        self._zeroconf: Optional["AsyncZeroconf"] = None
        self._browser: Optional["AsyncServiceBrowser"] = None
        self._service_info: Optional["ServiceInfo"] = None
        self._peers: Dict[str, DiscoveredPeer] = {}
        
        # Callbacks
        self.on_peer_found: Optional[Callable[[DiscoveredPeer], None]] = None
        self.on_peer_lost: Optional[Callable[[str], None]] = None
    
    @property
    def available(self) -> bool:
        """Check if mDNS discovery is available."""
        return ZEROCONF_AVAILABLE
    
    async def start(self) -> bool:
        """
        Start advertising and discovering peers.
        
        Returns:
            True if started successfully, False if mDNS not available
        """
        if not ZEROCONF_AVAILABLE:
            logger.warning("zeroconf not installed, mDNS discovery disabled")
            return False
        
        try:
            self._zeroconf = AsyncZeroconf()
            
            # Build service info
            properties = {
                b"node_id": self.node_id.encode(),
                b"mesh_id": (self.mesh_id or "").encode(),
                b"capabilities": ",".join(self.capabilities).encode(),
            }
            
            hostname = socket.gethostname()
            
            self._service_info = ServiceInfo(
                SERVICE_TYPE,
                f"{self.name}.{SERVICE_TYPE}",
                port=self.port,
                properties=properties,
                server=f"{hostname}.local.",
            )
            
            # Register our service
            await self._zeroconf.async_register_service(self._service_info)
            logger.info(f"Advertising as {self.name} on port {self.port}")
            
            # Start browsing for peers
            self._browser = AsyncServiceBrowser(
                self._zeroconf.zeroconf,
                SERVICE_TYPE,
                handlers=[self._on_service_state_change]
            )
            
            logger.info("mDNS discovery started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start mDNS discovery: {e}")
            return False
    
    async def stop(self) -> None:
        """Stop advertising and discovering."""
        if self._browser:
            self._browser.cancel()
            self._browser = None
        
        if self._zeroconf:
            if self._service_info:
                await self._zeroconf.async_unregister_service(self._service_info)
            await self._zeroconf.async_close()
            self._zeroconf = None
        
        logger.info("mDNS discovery stopped")
    
    def _on_service_state_change(
        self,
        zeroconf: "Zeroconf",
        service_type: str,
        name: str,
        state_change: "ServiceStateChange"
    ) -> None:
        """Handle service state changes."""
        if state_change == ServiceStateChange.Added:
            asyncio.create_task(self._handle_service_added(zeroconf, service_type, name))
        elif state_change == ServiceStateChange.Removed:
            self._handle_service_removed(name)
    
    async def _handle_service_added(
        self,
        zeroconf: "Zeroconf",
        service_type: str,
        name: str
    ) -> None:
        """Handle a newly discovered service."""
        info = zeroconf.get_service_info(service_type, name, timeout=3000)
        if not info:
            return
        
        # Parse properties
        node_id = info.properties.get(b"node_id", b"").decode()
        mesh_id = info.properties.get(b"mesh_id", b"").decode() or None
        caps_str = info.properties.get(b"capabilities", b"").decode()
        capabilities = caps_str.split(",") if caps_str else []
        
        # Skip ourselves
        if node_id == self.node_id:
            return
        
        # Get address
        if info.addresses:
            host = socket.inet_ntoa(info.addresses[0])
        else:
            host = info.server.rstrip(".")
        
        peer = DiscoveredPeer(
            node_id=node_id,
            name=name.replace(f".{SERVICE_TYPE}", ""),
            host=host,
            port=info.port,
            mesh_id=mesh_id,
            capabilities=capabilities
        )
        
        self._peers[node_id] = peer
        logger.info(f"Discovered peer: {peer.name} at {peer.address}")
        
        if self.on_peer_found:
            self.on_peer_found(peer)
    
    def _handle_service_removed(self, name: str) -> None:
        """Handle a removed service."""
        # Find peer by service name
        for node_id, peer in list(self._peers.items()):
            if f"{peer.name}.{SERVICE_TYPE}" == name:
                del self._peers[node_id]
                logger.info(f"Peer left: {peer.name}")
                
                if self.on_peer_lost:
                    self.on_peer_lost(node_id)
                break
    
    @property
    def peers(self) -> List[DiscoveredPeer]:
        """Get list of discovered peers."""
        return list(self._peers.values())
    
    def get_peer(self, node_id: str) -> Optional[DiscoveredPeer]:
        """Get a specific peer by node ID."""
        return self._peers.get(node_id)
    
    def get_mesh_peers(self, mesh_id: str) -> List[DiscoveredPeer]:
        """Get peers in a specific mesh."""
        return [p for p in self._peers.values() if p.mesh_id == mesh_id]


class ManualDiscovery:
    """
    Manual peer discovery for when mDNS isn't available.
    
    Stores a list of known peer addresses.
    """
    
    def __init__(self):
        self._peers: Dict[str, DiscoveredPeer] = {}
    
    def add_peer(self, host: str, port: int = 11434) -> None:
        """Add a peer by address."""
        peer_id = f"{host}:{port}"
        self._peers[peer_id] = DiscoveredPeer(
            node_id=peer_id,
            name=host,
            host=host,
            port=port
        )
    
    def remove_peer(self, host: str, port: int = 11434) -> None:
        """Remove a peer."""
        peer_id = f"{host}:{port}"
        self._peers.pop(peer_id, None)
    
    @property
    def peers(self) -> List[DiscoveredPeer]:
        """Get list of peers."""
        return list(self._peers.values())
