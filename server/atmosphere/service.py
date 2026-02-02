"""
Atmosphere Service - The main mesh coordinator.

Ties together:
- Rownd Local auth (identity, tokens)
- Network gossip (peer communication)
- LlamaFarm adapter (AI capabilities)
- Semantic routing (intent matching)
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .auth.rownd_local import RowndLocalService
from .mesh.network import GossipNetwork, MessageType, GossipMessage
from .mesh.gossip import CapabilityInfo, Announcement
from .router.gradient import GradientTable
from .discovery.llamafarm import LlamaFarmAdapter, discover_llamafarm

logger = logging.getLogger(__name__)


@dataclass
class AtmosphereConfig:
    """Configuration for Atmosphere service."""
    data_dir: Path
    mesh_name: str = ""
    mesh_id: str = ""
    node_id: str = ""
    gossip_port: int = 11450
    api_port: int = 11451
    seeds: List[str] = field(default_factory=list)
    llamafarm_url: str = "http://localhost:14345"


class AtmosphereService:
    """
    Main Atmosphere service.
    
    Coordinates mesh operations:
    - Authentication (Rownd Local)
    - Gossip (peer discovery, capability sharing)
    - Routing (semantic intent matching)
    - Integration (LlamaFarm, Ollama)
    """
    
    def __init__(
        self,
        data_dir: Path,
        gossip_port: int = 11450,
        api_port: int = 11451
    ):
        self.data_dir = Path(data_dir)
        self.config_file = self.data_dir / "config.json"
        
        # Load config
        if self.config_file.exists():
            with open(self.config_file) as f:
                cfg = json.load(f)
            self.config = AtmosphereConfig(
                data_dir=self.data_dir,
                mesh_name=cfg.get("mesh_name", ""),
                mesh_id=cfg.get("mesh_id", ""),
                node_id=cfg.get("node_id", ""),
                gossip_port=gossip_port,
                api_port=api_port,
                seeds=cfg.get("seeds", [])
            )
        else:
            self.config = AtmosphereConfig(
                data_dir=self.data_dir,
                gossip_port=gossip_port,
                api_port=api_port
            )
        
        # Components (initialized in start())
        self._auth: Optional[RowndLocalService] = None
        self._network: Optional[GossipNetwork] = None
        self._llamafarm: Optional[LlamaFarmAdapter] = None
        self._gradient_table: Optional[GradientTable] = None
        
        # State
        self._capabilities: List[CapabilityInfo] = []
        self._running = False
        self._api_server = None
    
    async def start(self):
        """Start all components."""
        logger.info(f"Starting Atmosphere service...")
        
        # Initialize auth
        auth_dir = self.data_dir / "auth"
        auth_dir.mkdir(exist_ok=True)
        self._auth = RowndLocalService(str(auth_dir))
        logger.info(f"Auth initialized: {self.config.node_id[:12]}...")
        
        # Initialize gradient table
        self._gradient_table = GradientTable(self.config.node_id)
        
        # Discover LlamaFarm
        self._llamafarm = await discover_llamafarm()
        if self._llamafarm:
            logger.info(f"LlamaFarm discovered with {len(self._llamafarm.capabilities)} capabilities")
            self._register_llamafarm_capabilities()
        else:
            logger.warning("LlamaFarm not found - running without AI capabilities")
        
        # Start network gossip
        self._network = GossipNetwork(
            node_id=self.config.node_id,
            port=self.config.gossip_port
        )
        
        # Register message handlers
        self._network.on(MessageType.CAPABILITY_ANNOUNCE, self._handle_capability_announce)
        self._network.on(MessageType.ROUTE_UPDATE, self._handle_route_update)
        self._network.on(MessageType.REVOCATION, self._handle_revocation)
        self._network.on(MessageType.INTENT, self._handle_intent)
        
        await self._network.start(seed_peers=self.config.seeds)
        logger.info(f"Gossip network started on port {self.config.gossip_port}")
        
        # Announce our capabilities
        await self._announce_capabilities()
        
        # Start API server
        await self._start_api()
        
        self._running = True
        logger.info("Atmosphere service started")
    
    async def stop(self):
        """Stop all components."""
        self._running = False
        
        if self._network:
            await self._network.stop()
        
        if self._llamafarm:
            await self._llamafarm.disconnect()
        
        if self._api_server:
            self._api_server.close()
            await self._api_server.wait_closed()
        
        logger.info("Atmosphere service stopped")
    
    async def run_forever(self):
        """Run until interrupted."""
        await self.start()
        
        try:
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()
    
    # === Capability Management ===
    
    def _register_llamafarm_capabilities(self):
        """Register LlamaFarm capabilities for gossip."""
        if not self._llamafarm:
            return
        
        for cap in self._llamafarm.capabilities:
            info = CapabilityInfo(
                id=f"llamafarm:{cap.name}",
                label=cap.name,
                description=cap.description,
                vector=[],  # Will be filled by embedding
                local=True,
                hops=0,
                models=cap.models
            )
            self._capabilities.append(info)
    
    async def _announce_capabilities(self):
        """Announce our capabilities to the mesh."""
        if not self._network:
            return
        
        # Embed capability descriptions
        if self._llamafarm:
            for cap in self._capabilities:
                try:
                    embedding = await self._llamafarm.embed_single(cap.description)
                    cap.vector = embedding
                except Exception as e:
                    logger.warning(f"Failed to embed capability {cap.id}: {e}")
        
        # Broadcast announcement
        await self._network.broadcast(
            MessageType.CAPABILITY_ANNOUNCE,
            {
                "node_id": self.config.node_id,
                "capabilities": [cap.to_dict() for cap in self._capabilities],
                "timestamp": time.time()
            }
        )
    
    # === Message Handlers ===
    
    async def _handle_capability_announce(self, peer, msg: GossipMessage):
        """Handle incoming capability announcement."""
        payload = msg.payload
        node_id = payload.get("node_id")
        
        if node_id == self.config.node_id:
            return  # Ignore own announcements
        
        for cap_data in payload.get("capabilities", []):
            cap = CapabilityInfo.from_dict(cap_data)
            cap.local = False
            cap.hops = msg.ttl
            cap.via = peer.node_id
            
            # Update gradient table
            if cap.vector:
                import numpy as np
                self._gradient_table.update(
                    capability_id=cap.id,
                    capability_vector=np.array(cap.vector),
                    next_hop=peer.node_id,
                    hops=cap.hops,
                    latency_ms=50  # Estimate
                )
        
        logger.debug(f"Received {len(payload.get('capabilities', []))} capabilities from {node_id}")
    
    async def _handle_route_update(self, peer, msg: GossipMessage):
        """Handle route update from peer."""
        # Update gradient table based on peer's routes
        pass
    
    async def _handle_revocation(self, peer, msg: GossipMessage):
        """Handle token revocation gossip."""
        device_id = msg.payload.get("device_id")
        reason = msg.payload.get("reason")
        
        if device_id and self._auth:
            self._auth.handle_remote_revocation(device_id, reason)
            logger.info(f"Received revocation for {device_id}: {reason}")
    
    async def _handle_intent(self, peer, msg: GossipMessage):
        """Handle incoming intent request."""
        intent_text = msg.payload.get("intent")
        session_id = msg.payload.get("session_id")
        
        # Route the intent
        result = await self.route_intent(intent_text)
        
        # Send response back
        await self._network.send(
            peer.node_id,
            MessageType.INTENT_RESPONSE,
            {
                "session_id": session_id,
                "result": result
            }
        )
    
    # === Intent Routing ===
    
    async def route_intent(self, intent_text: str) -> Dict[str, Any]:
        """
        Route an intent to the best capable node.
        
        This is THE core function of Atmosphere.
        """
        start_time = time.time()
        
        if not self._llamafarm:
            return {
                "error": "No AI backend available",
                "routed_to": None,
                "result": None
            }
        
        # Embed the intent
        intent_vector = await self._llamafarm.embed_single(intent_text)
        
        # Find best route
        import numpy as np
        best_route = self._gradient_table.find_best_route(np.array(intent_vector))
        
        if best_route and best_route.next_hop != "local":
            # Route to remote node
            # TODO: Implement remote execution
            return {
                "routed_to": best_route.next_hop,
                "score": best_route.score,
                "result": f"Would route to {best_route.next_hop} (not implemented)",
                "latency_ms": (time.time() - start_time) * 1000
            }
        
        # Execute locally
        try:
            result = await self._llamafarm.generate(intent_text)
            return {
                "routed_to": "local",
                "result": result,
                "latency_ms": (time.time() - start_time) * 1000
            }
        except Exception as e:
            return {
                "routed_to": "local",
                "error": str(e),
                "latency_ms": (time.time() - start_time) * 1000
            }
    
    # === API Server ===
    
    async def _start_api(self):
        """Start the HTTP API server."""
        from aiohttp import web
        
        app = web.Application()
        app.add_routes([
            web.get("/status", self._api_status),
            web.get("/nodes", self._api_nodes),
            web.post("/intent", self._api_intent),
        ])
        
        runner = web.AppRunner(app)
        await runner.setup()
        
        self._api_server = web.TCPSite(runner, "localhost", self.config.api_port)
        await self._api_server.start()
        
        logger.info(f"API server started on port {self.config.api_port}")
    
    async def _api_status(self, request):
        """GET /status - Return mesh status."""
        from aiohttp import web
        
        status = {
            "mesh_name": self.config.mesh_name,
            "mesh_id": self.config.mesh_id,
            "node_id": self.config.node_id,
            "role": "founder" if self._auth else "member",
            "connected": self._network is not None and len(self._network.connected_peers) > 0,
            "peer_count": len(self._network.connected_peers) if self._network else 0,
            "capabilities": [cap.label for cap in self._capabilities],
            "llamafarm_connected": self._llamafarm is not None and self._llamafarm.is_connected
        }
        
        return web.json_response(status)
    
    async def _api_nodes(self, request):
        """GET /nodes - Return connected nodes."""
        from aiohttp import web
        
        if not self._network:
            return web.json_response({"nodes": []})
        
        nodes = []
        for peer_id, peer in self._network.peers.items():
            nodes.append({
                "id": peer_id,
                "address": peer.address,
                "connected": peer.connected,
                "last_seen": peer.last_seen,
                "capabilities": peer.capabilities
            })
        
        return web.json_response({"nodes": nodes})
    
    async def _api_intent(self, request):
        """POST /intent - Route an intent."""
        from aiohttp import web
        
        try:
            data = await request.json()
            intent_text = data.get("intent")
            
            if not intent_text:
                return web.json_response({"error": "No intent provided"}, status=400)
            
            result = await self.route_intent(intent_text)
            return web.json_response(result)
            
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    # === Public Interface ===
    
    def get_status(self) -> Dict:
        """Get current service status."""
        return {
            "running": self._running,
            "config": {
                "mesh_name": self.config.mesh_name,
                "mesh_id": self.config.mesh_id,
                "node_id": self.config.node_id
            },
            "network": self._network.get_status() if self._network else None,
            "capabilities": len(self._capabilities),
            "llamafarm": self._llamafarm.is_connected if self._llamafarm else False
        }
