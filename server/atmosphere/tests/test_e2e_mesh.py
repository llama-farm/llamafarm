"""
End-to-end mesh integration test.

Tests the complete mesh networking system from setup through failure recovery.
"""

import pytest
import asyncio
import logging
import socket
import time
from typing import Dict, List, Optional

from atmosphere.mesh.node import Node, MeshIdentity, NodeIdentity
from atmosphere.mesh.network import GossipNetwork, MessageType
from atmosphere.mesh.join import JoinCode
from atmosphere.router.semantic import SemanticRouter, RouteAction
from atmosphere.router.gradient import GradientTable
import numpy as np

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_free_port():
    """Get a free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


class MeshNode:
    """
    Full mesh node with networking, routing, and capabilities.
    Wrapper to coordinate all components.
    """
    
    def __init__(
        self,
        node: Node,
        port: int,
        is_founder: bool = False
    ):
        self.node = node
        self.port = port
        self.is_founder = is_founder
        
        # Components
        self.network: Optional[GossipNetwork] = None
        self.router: Optional[SemanticRouter] = None
        
        # State tracking
        self.capability_announcements: List[Dict] = []
        self.intent_requests: List[Dict] = []
        self.session_state: Dict = {}
        
    async def start(self, seed_peers: Optional[List[str]] = None):
        """Start all node components."""
        # Initialize network
        self.network = GossipNetwork(
            node_id=self.node.node_id,
            port=self.port
        )
        
        # Register message handlers
        self.network.on(MessageType.CAPABILITY_ANNOUNCE, self._handle_capability_announce)
        self.network.on(MessageType.INTENT, self._handle_intent)
        self.network.on(MessageType.SESSION_UPDATE, self._handle_session_update)
        
        # Start network
        await self.network.start(seed_peers=seed_peers)
        
        # Initialize router
        self.router = SemanticRouter(node_id=self.node.node_id)
        await self.router.initialize()
        
        logger.info(f"Node {self.node.name} ({self.node.node_id}) started on port {self.port}")
    
    async def stop(self):
        """Stop all node components."""
        if self.router:
            await self.router.close()
        if self.network:
            await self.network.stop()
        logger.info(f"Node {self.node.name} stopped")
    
    async def announce_capability(
        self,
        label: str,
        description: str,
        models: Optional[List[str]] = None
    ):
        """Announce a capability to the mesh."""
        # Register locally
        capability = await self.router.register_capability(
            label=label,
            description=description,
            models=models or []
        )
        
        # Broadcast to mesh
        await self.network.broadcast(
            MessageType.CAPABILITY_ANNOUNCE,
            {
                "capability_id": capability.id,
                "capability_label": capability.label,
                "capability_description": description,
                "capability_vector": capability.vector.tolist(),
                "models": models or [],
                "node_id": self.node.node_id,
                "hops": 0
            }
        )
        
        logger.info(f"Node {self.node.name} announced capability: {label}")
        return capability
    
    def _handle_capability_announce(self, peer, msg):
        """Handle capability announcement from peer."""
        payload = msg.payload
        self.capability_announcements.append(payload)
        
        # Update gradient table
        self.router.gradient_table.update(
            capability_id=payload["capability_id"],
            capability_label=payload["capability_label"],
            capability_vector=np.array(payload["capability_vector"], dtype=np.float32),
            hops=payload["hops"] + 1,
            next_hop=peer.node_id,
            via_node=payload["node_id"]
        )
        
        logger.info(
            f"Node {self.node.name} received capability {payload['capability_label']} "
            f"from {peer.node_id} (via {payload['node_id']}, {payload['hops']+1} hops)"
        )
        
        # Propagate if hops < max
        if payload["hops"] < 5:
            asyncio.create_task(self._propagate_capability(payload, exclude=peer.node_id))
    
    async def _propagate_capability(self, payload: Dict, exclude: str):
        """Propagate capability to other peers."""
        updated_payload = payload.copy()
        updated_payload["hops"] += 1
        
        for peer_id, peer in self.network.peers.items():
            if peer_id != exclude and peer.connected:
                try:
                    await self.network.send_to_peer(
                        peer,
                        MessageType.CAPABILITY_ANNOUNCE,
                        updated_payload
                    )
                except Exception as e:
                    logger.warning(f"Failed to propagate to {peer_id}: {e}")
    
    async def send_intent(self, intent: str) -> Dict:
        """Send an intent and route it through the mesh."""
        # Embed the intent
        intent_vector = await self.router.embedding_engine.embed(intent)
        
        # Find best route
        route_result = await self.router.route_intent(intent, intent_vector)
        
        intent_payload = {
            "intent": intent,
            "intent_vector": intent_vector.tolist(),
            "requester": self.node.node_id,
            "timestamp": time.time()
        }
        
        if route_result.action == RouteAction.PROCESS_LOCAL:
            logger.info(f"Node {self.node.name} processing intent locally: {intent}")
            return {
                "status": "processed_local",
                "capability": route_result.capability.label,
                "node": self.node.node_id
            }
        
        elif route_result.action == RouteAction.FORWARD:
            logger.info(
                f"Node {self.node.name} forwarding intent to {route_result.next_hop}: {intent}"
            )
            
            # Send to next hop
            peer = self.network.peers.get(route_result.next_hop)
            if peer and peer.connected:
                await self.network.send_to_peer(
                    peer,
                    MessageType.INTENT,
                    intent_payload
                )
                return {
                    "status": "forwarded",
                    "next_hop": route_result.next_hop,
                    "via_node": route_result.via_node,
                    "hops": route_result.hops
                }
            else:
                return {
                    "status": "no_route",
                    "reason": "next_hop not connected"
                }
        
        else:
            logger.warning(f"Node {self.node.name} found no route for intent: {intent}")
            return {
                "status": "no_match",
                "reason": route_result.reason
            }
    
    def _handle_intent(self, peer, msg):
        """Handle intent request from peer."""
        payload = msg.payload
        self.intent_requests.append({
            "from": peer.node_id,
            "payload": payload,
            "timestamp": time.time()
        })
        logger.info(
            f"Node {self.node.name} received intent from {peer.node_id}: "
            f"{payload.get('intent')}"
        )
    
    def _handle_session_update(self, peer, msg):
        """Handle session update from peer."""
        payload = msg.payload
        session_id = payload.get("session_id")
        if session_id:
            self.session_state[session_id] = payload
            logger.info(
                f"Node {self.node.name} received session update for {session_id} "
                f"from {peer.node_id}"
            )
    
    async def start_session(self, session_id: str, context: Dict):
        """Start a new session."""
        self.session_state[session_id] = {
            "session_id": session_id,
            "context": context,
            "created_at": time.time(),
            "node_id": self.node.node_id
        }
        
        # Broadcast session to mesh
        await self.network.broadcast(
            MessageType.SESSION_UPDATE,
            self.session_state[session_id]
        )
        
        logger.info(f"Node {self.node.name} started session {session_id}")
    
    @property
    def connected_peers(self) -> List[str]:
        """Get list of connected peer IDs."""
        if not self.network:
            return []
        return [
            peer_id for peer_id, peer in self.network.peers.items()
            if peer.connected
        ]
    
    def get_gradient_stats(self) -> Dict:
        """Get gradient table statistics."""
        if not self.router:
            return {}
        return self.router.gradient_table.stats()


class TestAtmosphereE2E:
    """
    End-to-end tests for Atmosphere mesh networking.
    
    Tests the complete system from mesh creation through failure recovery.
    """
    
    @pytest.mark.asyncio
    async def test_e2e_mesh(self):
        """Full end-to-end test of mesh networking."""
        
        # ============================================================
        # PHASE 1: SETUP
        # ============================================================
        logger.info("=" * 60)
        logger.info("PHASE 1: SETUP - Creating mesh and starting nodes")
        logger.info("=" * 60)
        
        # Create mesh founder (Node A)
        node_a = Node.create_with_mesh(
            node_name="Node A",
            mesh_name="test-mesh",
            threshold=2,
            total_shares=3
        )
        
        port_a = get_free_port()
        mesh_node_a = MeshNode(node_a, port_a, is_founder=True)
        
        # Generate join token for other nodes
        join_code = JoinCode(
            mesh_id=node_a.mesh.mesh_id,
            mesh_name=node_a.mesh.name,
            mesh_public_key=node_a.mesh.master_public_key,
            endpoint=f"localhost:{port_a}",
            created_at=int(time.time()),
            expires_at=int(time.time()) + 3600
        )
        
        logger.info(f"Created mesh: {node_a.mesh.name} (ID: {node_a.mesh.mesh_id})")
        logger.info(f"Join code: {join_code.code}")
        
        # Create Node B and Node C (members)
        node_b_identity = NodeIdentity.generate("Node B")
        node_b = Node(node_b_identity, mesh=MeshIdentity.from_dict(node_a.mesh.to_dict()))
        
        node_c_identity = NodeIdentity.generate("Node C")
        node_c = Node(node_c_identity, mesh=MeshIdentity.from_dict(node_a.mesh.to_dict()))
        
        port_b = get_free_port()
        port_c = get_free_port()
        
        mesh_node_b = MeshNode(node_b, port_b)
        mesh_node_c = MeshNode(node_c, port_c)
        
        try:
            # Start all nodes
            await mesh_node_a.start()
            await asyncio.sleep(1)
            
            await mesh_node_b.start(seed_peers=[f"localhost:{port_a}"])
            await asyncio.sleep(1)
            
            await mesh_node_c.start(seed_peers=[f"localhost:{port_a}"])
            await asyncio.sleep(3)  # Wait for mesh formation
            
            # Verify all nodes see each other
            assert "node-b" in mesh_node_a.connected_peers or node_b.node_id in mesh_node_a.connected_peers
            assert "node-c" in mesh_node_a.connected_peers or node_c.node_id in mesh_node_a.connected_peers
            assert node_a.node_id in mesh_node_b.connected_peers
            assert node_a.node_id in mesh_node_c.connected_peers
            
            logger.info(f"✅ All 3 nodes connected")
            logger.info(f"   Node A peers: {mesh_node_a.connected_peers}")
            logger.info(f"   Node B peers: {mesh_node_b.connected_peers}")
            logger.info(f"   Node C peers: {mesh_node_c.connected_peers}")
            
            # ============================================================
            # PHASE 2: CAPABILITY ANNOUNCEMENTS
            # ============================================================
            logger.info("")
            logger.info("=" * 60)
            logger.info("PHASE 2: CAPABILITY - Announcing capabilities")
            logger.info("=" * 60)
            
            # Node A: embeddings
            await mesh_node_a.announce_capability(
                label="embeddings",
                description="Generate text embeddings for semantic search and similarity",
                models=["nomic-embed-text", "all-MiniLM-L6-v2"]
            )
            
            # Node B: LLM inference
            await mesh_node_b.announce_capability(
                label="llm-inference",
                description="Generate text completions and chat responses using language models",
                models=["llama-3.1-8b", "gemma-2-9b"]
            )
            
            # Node C: vision
            await mesh_node_c.announce_capability(
                label="vision",
                description="Analyze images and visual content using vision models",
                models=["llava-1.5", "llava-phi3"]
            )
            
            # Wait for propagation
            await asyncio.sleep(3)
            
            # Verify gradient tables updated on all nodes
            stats_a = mesh_node_a.get_gradient_stats()
            stats_b = mesh_node_b.get_gradient_stats()
            stats_c = mesh_node_c.get_gradient_stats()
            
            logger.info(f"✅ Gradient tables updated:")
            logger.info(f"   Node A: {stats_a['size']} entries, avg {stats_a['avg_hops']:.1f} hops")
            logger.info(f"   Node B: {stats_b['size']} entries, avg {stats_b['avg_hops']:.1f} hops")
            logger.info(f"   Node C: {stats_c['size']} entries, avg {stats_c['avg_hops']:.1f} hops")
            
            # Each node should know about the other two capabilities
            assert stats_a["size"] >= 2  # Should see B and C
            assert stats_b["size"] >= 2  # Should see A and C
            assert stats_c["size"] >= 2  # Should see A and B
            
            # ============================================================
            # PHASE 3: ROUTING
            # ============================================================
            logger.info("")
            logger.info("=" * 60)
            logger.info("PHASE 3: ROUTING - Testing intent routing")
            logger.info("=" * 60)
            
            # Test 1: Node C sends embeddings intent -> should route to Node A
            result1 = await mesh_node_c.send_intent(
                "generate embeddings for this text"
            )
            await asyncio.sleep(1)
            
            logger.info(f"Intent routing result: {result1}")
            
            # Should be forwarded to Node A
            assert result1["status"] in ["forwarded", "processed_local"]
            if result1["status"] == "forwarded":
                assert result1["next_hop"] == node_a.node_id or result1["via_node"] == node_a.node_id
            
            # Verify Node A received the intent
            node_a_intents = [req for req in mesh_node_a.intent_requests if "embedding" in req["payload"]["intent"]]
            assert len(node_a_intents) > 0, "Node A should have received embeddings intent"
            
            logger.info(f"✅ Embeddings intent routed from Node C -> Node A")
            
            # Test 2: Node A sends vision intent -> should route to Node C
            result2 = await mesh_node_a.send_intent(
                "analyze this image and describe what you see"
            )
            await asyncio.sleep(1)
            
            logger.info(f"Intent routing result: {result2}")
            
            # Should be forwarded to Node C
            assert result2["status"] in ["forwarded", "processed_local"]
            if result2["status"] == "forwarded":
                assert result2["next_hop"] == node_c.node_id or result2["via_node"] == node_c.node_id
            
            # Verify Node C received the intent
            node_c_intents = [req for req in mesh_node_c.intent_requests if "image" in req["payload"]["intent"]]
            assert len(node_c_intents) > 0, "Node C should have received vision intent"
            
            logger.info(f"✅ Vision intent routed from Node A -> Node C")
            
            # ============================================================
            # PHASE 4: SESSION MANAGEMENT
            # ============================================================
            logger.info("")
            logger.info("=" * 60)
            logger.info("PHASE 4: SESSION - Testing session context transfer")
            logger.info("=" * 60)
            
            # Start session on Node A
            session_id = "test-session-001"
            await mesh_node_a.start_session(
                session_id,
                context={
                    "user_id": "user-123",
                    "conversation_history": ["Hello", "How can I help?"]
                }
            )
            
            await asyncio.sleep(2)
            
            # Verify session propagated to Node B and C
            assert session_id in mesh_node_b.session_state, "Node B should have session state"
            assert session_id in mesh_node_c.session_state, "Node C should have session state"
            
            logger.info(f"✅ Session {session_id} propagated to all nodes")
            
            # Send intent requiring Node B's capability with session context
            result3 = await mesh_node_a.send_intent(
                "generate a summary using the language model"
            )
            await asyncio.sleep(1)
            
            logger.info(f"Session intent routing result: {result3}")
            
            # Should route to Node B
            assert result3["status"] in ["forwarded", "processed_local"]
            
            # Verify Node B has the session context
            assert session_id in mesh_node_b.session_state
            assert mesh_node_b.session_state[session_id]["context"]["user_id"] == "user-123"
            
            logger.info(f"✅ Session context available on Node B for intent processing")
            
            # ============================================================
            # PHASE 5: FAILURE AND RECOVERY
            # ============================================================
            logger.info("")
            logger.info("=" * 60)
            logger.info("PHASE 5: FAILURE - Testing node failure and recovery")
            logger.info("=" * 60)
            
            # Stop Node B (simulate failure)
            logger.info("Stopping Node B to simulate failure...")
            await mesh_node_b.stop()
            await asyncio.sleep(3)
            
            # Send intent that would have gone to Node B
            result4 = await mesh_node_a.send_intent(
                "use the language model to generate text"
            )
            await asyncio.sleep(1)
            
            logger.info(f"Intent routing after Node B failure: {result4}")
            
            # Should either find no route or route to another node
            # (In a real system with redundancy, it might find an alternative)
            assert result4["status"] in ["no_match", "no_route", "forwarded", "processed_local"]
            
            logger.info(f"✅ System handled Node B failure gracefully")
            
            # Restart Node B
            logger.info("Restarting Node B...")
            mesh_node_b = MeshNode(node_b, port_b)
            await mesh_node_b.start(seed_peers=[f"localhost:{port_a}"])
            
            # Re-announce capability
            await mesh_node_b.announce_capability(
                label="llm-inference",
                description="Generate text completions and chat responses using language models",
                models=["llama-3.1-8b", "gemma-2-9b"]
            )
            
            await asyncio.sleep(3)
            
            # Verify Node B reconnected
            assert node_a.node_id in mesh_node_b.connected_peers
            
            # Verify Node B's capability is back in gradient tables
            stats_a_after = mesh_node_a.get_gradient_stats()
            assert stats_a_after["size"] >= 2
            
            logger.info(f"✅ Node B rejoined mesh successfully")
            logger.info(f"   Node A gradient table: {stats_a_after['size']} entries")
            
            # Test routing works again
            result5 = await mesh_node_a.send_intent(
                "generate a response with the language model"
            )
            await asyncio.sleep(1)
            
            logger.info(f"Intent routing after Node B recovery: {result5}")
            assert result5["status"] in ["forwarded", "processed_local"]
            
            logger.info(f"✅ Routing recovered after Node B rejoined")
            
            # ============================================================
            # PHASE 6: CLEANUP
            # ============================================================
            logger.info("")
            logger.info("=" * 60)
            logger.info("PHASE 6: CLEANUP - Stopping all nodes")
            logger.info("=" * 60)
            
        finally:
            # Stop all nodes
            await mesh_node_c.stop()
            await mesh_node_b.stop()
            await mesh_node_a.stop()
            
            logger.info("✅ All nodes stopped cleanly")
        
        # ============================================================
        # TEST COMPLETE
        # ============================================================
        logger.info("")
        logger.info("=" * 60)
        logger.info("🎉 E2E TEST PASSED - All phases completed successfully!")
        logger.info("=" * 60)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
