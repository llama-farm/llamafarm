"""
Router service initialization and lifecycle management.

Initializes and manages the semantic router components:
- Embedding engine
- Capability matcher
- Gradient table
- Gossip protocol
- Peer discovery
"""

import asyncio
import uuid
import logging
from typing import List, Optional

from .embeddings import EmbeddingEngine, EmbeddingConfig
from .matcher import CapabilityMatcher, Capability
from .gradient import GradientTable
from .gossip import GossipProtocol, CapabilityInfo
from .discovery import PeerDiscovery
from .api import init_router_state

logger = logging.getLogger(__name__)


class RouterService:
    """
    Manages the semantic router lifecycle.
    
    Usage:
        service = RouterService()
        await service.initialize()
        
        # Register capabilities
        await service.register_capability(
            label="vision",
            description="Analyze images and detect objects"
        )
        
        # Shutdown
        await service.shutdown()
    """
    
    def __init__(
        self,
        node_id: Optional[str] = None,
        enable_discovery: bool = True,
        enable_gossip: bool = True
    ):
        """
        Initialize router service.
        
        Args:
            node_id: Unique node identifier (auto-generated if None)
            enable_discovery: Enable automatic peer discovery
            enable_gossip: Enable gossip protocol
        """
        self.node_id = node_id or f"llamafarm-{uuid.uuid4().hex[:8]}"
        self.enable_discovery = enable_discovery
        self.enable_gossip = enable_gossip
        
        # Components
        self.embedding_engine: Optional[EmbeddingEngine] = None
        self.gradient_table: Optional[GradientTable] = None
        self.matcher: Optional[CapabilityMatcher] = None
        self.gossip: Optional[GossipProtocol] = None
        self.discovery: Optional[PeerDiscovery] = None
        
        # State
        self.local_capabilities: List[Capability] = []
        self._initialized = False
        self._maintenance_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> None:
        """Initialize all router components."""
        if self._initialized:
            logger.warning("Router service already initialized")
            return
        
        logger.info(f"Initializing router service for node: {self.node_id}")
        
        # 1. Initialize embedding engine
        self.embedding_engine = EmbeddingEngine(EmbeddingConfig())
        await self.embedding_engine.initialize()
        logger.info(f"Embedding engine initialized (backend: {self.embedding_engine.backend})")
        
        # 2. Initialize gradient table
        self.gradient_table = GradientTable(
            node_id=self.node_id,
            max_size=1000,
            expire_sec=300
        )
        logger.info("Gradient table initialized")
        
        # 3. Initialize capability matcher
        self.matcher = CapabilityMatcher(
            match_threshold=0.75,
            min_route_threshold=0.50,
            hop_penalty=0.95
        )
        logger.info("Capability matcher initialized")
        
        # 4. Initialize gossip protocol
        if self.enable_gossip:
            self.gossip = GossipProtocol(
                node_id=self.node_id,
                gradient_table=self.gradient_table,
                local_capabilities=self._build_capability_infos(),
                announce_interval=30
            )
            
            # Set broadcast callback (will be connected to actual transport later)
            # For now, just a placeholder
            self.gossip.set_broadcast_callback(self._broadcast_to_peers)
            
            await self.gossip.start()
            logger.info("Gossip protocol started")
        
        # 5. Initialize peer discovery
        if self.enable_discovery:
            self.discovery = PeerDiscovery(
                node_id=self.node_id,
                port=47471,
                on_peer_found=self._handle_peer_discovered
            )
            await self.discovery.start()
            logger.info("Peer discovery started")
        
        # 6. Start maintenance task
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())
        
        # 7. Initialize API state
        init_router_state(
            node_id=self.node_id,
            embedding_engine=self.embedding_engine,
            gradient_table=self.gradient_table,
            matcher=self.matcher,
            gossip=self.gossip,
            local_capabilities=self.local_capabilities
        )
        
        self._initialized = True
        logger.info("Router service initialization complete")
    
    async def shutdown(self) -> None:
        """Shutdown the router service."""
        logger.info("Shutting down router service")
        
        # Stop maintenance task
        if self._maintenance_task:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
        
        # Stop gossip protocol
        if self.gossip:
            await self.gossip.stop()
        
        # Stop peer discovery
        if self.discovery:
            await self.discovery.stop()
        
        # Close embedding engine
        if self.embedding_engine:
            await self.embedding_engine.close()
        
        self._initialized = False
        logger.info("Router service shutdown complete")
    
    async def register_capability(
        self,
        label: str,
        description: str,
        handler: str = "default",
        models: Optional[List[str]] = None
    ) -> Capability:
        """
        Register a new local capability.
        
        Args:
            label: Capability label (e.g., "vision", "llm")
            description: Detailed description for semantic matching
            handler: Name of handler function
            models: List of required model names
        
        Returns:
            The registered Capability
        """
        if not self._initialized or not self.embedding_engine:
            raise RuntimeError("Router service not initialized")
        
        # Generate embedding
        cap_text = f"{label}: {description}"
        cap_vec = await self.embedding_engine.embed(cap_text)
        
        # Create capability
        capability = Capability(
            id=f"{self.node_id}:{label}:{uuid.uuid4().hex[:8]}",
            label=label,
            description=description,
            vector=cap_vec,
            handler=handler,
            models=models or []
        )
        
        self.local_capabilities.append(capability)
        
        # Update gossip protocol
        if self.gossip:
            self.gossip.update_local_capabilities(self._build_capability_infos())
        
        logger.info(f"Registered capability: {label}")
        return capability
    
    def _build_capability_infos(self) -> List[CapabilityInfo]:
        """Convert local capabilities to CapabilityInfo for gossip."""
        infos = []
        for cap in self.local_capabilities:
            infos.append(CapabilityInfo(
                id=cap.id,
                label=cap.label,
                description=cap.description,
                vector=cap.vector.tolist(),
                local=True,
                hops=0,
                models=cap.models
            ))
        return infos
    
    async def _broadcast_to_peers(self, from_node: str, data: bytes) -> None:
        """
        Broadcast callback for gossip protocol.
        
        TODO: Implement actual network transport (UDP, WebSocket, etc.)
        For now, this is a placeholder.
        """
        # This would normally send data to all known peers
        # Will be implemented when we add actual networking
        logger.debug(f"Broadcast from {from_node}: {len(data)} bytes")
    
    async def _handle_peer_discovered(
        self,
        peer_id: str,
        peer_host: str,
        peer_port: int
    ) -> None:
        """Handle discovery of a new peer."""
        logger.info(f"New peer discovered: {peer_id} at {peer_host}:{peer_port}")
        
        # TODO: Establish connection to peer
        # TODO: Exchange capabilities
    
    async def _maintenance_loop(self) -> None:
        """Periodic maintenance tasks."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Prune expired gradient entries
                if self.gradient_table:
                    removed = self.gradient_table.prune_expired()
                    if removed > 0:
                        logger.debug(f"Pruned {removed} expired gradient entries")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Maintenance task error: {e}")
    
    def get_stats(self) -> dict:
        """Get router service statistics."""
        stats = {
            "node_id": self.node_id,
            "initialized": self._initialized,
            "local_capabilities": len(self.local_capabilities),
        }
        
        if self.gradient_table:
            stats["gradient_table"] = self.gradient_table.stats()
        
        if self.gossip:
            stats["gossip"] = self.gossip.stats()
        
        if self.discovery:
            stats["known_peers"] = len(self.discovery.get_known_peers())
        
        return stats


# Global service instance
_router_service: Optional[RouterService] = None


async def get_router_service() -> RouterService:
    """Get the global router service instance."""
    global _router_service
    
    if _router_service is None:
        _router_service = RouterService()
        await _router_service.initialize()
    
    return _router_service


async def shutdown_router_service() -> None:
    """Shutdown the global router service."""
    global _router_service
    
    if _router_service:
        await _router_service.shutdown()
        _router_service = None
