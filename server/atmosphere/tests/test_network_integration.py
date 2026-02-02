"""
Network integration tests.

Tests real TCP gossip between nodes.
"""

import pytest
import asyncio
import logging
import socket
from atmosphere.mesh.network import (
    GossipNetwork,
    MessageType,
    GossipMessage,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_free_port():
    """Get a free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


class TestNetworkGossip:
    """Test network gossip integration."""
    
    @pytest.mark.asyncio
    async def test_server_startup(self):
        """Test that GossipNetwork can start a server."""
        port = get_free_port()
        network = GossipNetwork(node_id="node-a", port=port)
        
        try:
            await network.start()
            assert network._running
            assert network._server is not None
            
            # Verify server is listening
            status = network.get_status()
            assert status["running"]
            assert status["node_id"] == "node-a"
            assert status["port"] == port
            
        finally:
            await network.stop()
    
    @pytest.mark.asyncio
    async def test_two_node_handshake(self):
        """Test handshake between two nodes."""
        port_a = get_free_port()
        port_b = get_free_port()
        node_a = GossipNetwork(node_id="node-a", port=port_a)
        node_b = GossipNetwork(node_id="node-b", port=port_b)
        
        try:
            # Start both nodes
            await node_a.start()
            
            # Node B starts with Node A as seed peer
            await node_b.start(seed_peers=[f"localhost:{port_a}"])
            
            # Wait for connection
            await asyncio.sleep(2)
            
            # Verify connection
            assert "node-b" in node_a.connected_peers
            assert "node-a" in node_b.connected_peers
            
            logger.info(f"Node A peers: {node_a.connected_peers}")
            logger.info(f"Node B peers: {node_b.connected_peers}")
            
        finally:
            await node_b.stop()
            await node_a.stop()
    
    @pytest.mark.asyncio
    async def test_message_propagation(self):
        """Test message propagation between nodes."""
        port_a = get_free_port()
        port_b = get_free_port()
        node_a = GossipNetwork(node_id="node-a", port=port_a)
        node_b = GossipNetwork(node_id="node-b", port=port_b)
        
        # Track received messages
        messages_a = []
        messages_b = []
        
        def handler_a(peer, msg):
            messages_a.append((peer.node_id, msg.type, msg.payload))
            logger.info(f"Node A received: {msg.type} from {peer.node_id}")
        
        def handler_b(peer, msg):
            messages_b.append((peer.node_id, msg.type, msg.payload))
            logger.info(f"Node B received: {msg.type} from {peer.node_id}")
        
        try:
            # Register handlers
            node_a.on(MessageType.CAPABILITY_ANNOUNCE, handler_a)
            node_b.on(MessageType.CAPABILITY_ANNOUNCE, handler_b)
            
            # Start both nodes
            await node_a.start()
            await node_b.start(seed_peers=[f"localhost:{port_a}"])
            
            # Wait for connection
            await asyncio.sleep(2)
            
            # Node A broadcasts capability
            await node_a.broadcast(
                MessageType.CAPABILITY_ANNOUNCE,
                {
                    "capability": "text-generation",
                    "model": "llama-3.1-8b",
                    "endpoint": "http://node-a:8080"
                }
            )
            
            # Wait for propagation
            await asyncio.sleep(1)
            
            # Node B should have received it
            assert len(messages_b) > 0
            peer_id, msg_type, payload = messages_b[0]
            assert peer_id == "node-a"
            assert msg_type == MessageType.CAPABILITY_ANNOUNCE
            assert payload["capability"] == "text-generation"
            assert payload["model"] == "llama-3.1-8b"
            
            logger.info(f"✅ Node B received capability from Node A")
            
            # Now Node B broadcasts
            messages_a.clear()
            await node_b.broadcast(
                MessageType.CAPABILITY_ANNOUNCE,
                {
                    "capability": "embeddings",
                    "model": "nomic-embed-text",
                    "endpoint": "http://node-b:8080"
                }
            )
            
            # Wait for propagation
            await asyncio.sleep(1)
            
            # Node A should have received it
            assert len(messages_a) > 0
            peer_id, msg_type, payload = messages_a[0]
            assert peer_id == "node-b"
            assert msg_type == MessageType.CAPABILITY_ANNOUNCE
            assert payload["capability"] == "embeddings"
            assert payload["model"] == "nomic-embed-text"
            
            logger.info(f"✅ Node A received capability from Node B")
            
        finally:
            await node_b.stop()
            await node_a.stop()
    
    @pytest.mark.asyncio
    async def test_bidirectional_capabilities(self):
        """Test that both nodes see each other's capabilities."""
        port_a = get_free_port()
        port_b = get_free_port()
        node_a = GossipNetwork(node_id="node-a", port=port_a)
        node_b = GossipNetwork(node_id="node-b", port=port_b)
        
        # Track capabilities seen by each node
        capabilities_seen_by_a = {}
        capabilities_seen_by_b = {}
        
        def track_cap_a(peer, msg):
            cap_id = msg.payload.get("capability_id", msg.payload.get("capability"))
            capabilities_seen_by_a[cap_id] = {
                "peer": peer.node_id,
                "payload": msg.payload
            }
            logger.info(f"Node A sees capability: {cap_id} from {peer.node_id}")
        
        def track_cap_b(peer, msg):
            cap_id = msg.payload.get("capability_id", msg.payload.get("capability"))
            capabilities_seen_by_b[cap_id] = {
                "peer": peer.node_id,
                "payload": msg.payload
            }
            logger.info(f"Node B sees capability: {cap_id} from {peer.node_id}")
        
        try:
            # Register handlers
            node_a.on(MessageType.CAPABILITY_ANNOUNCE, track_cap_a)
            node_b.on(MessageType.CAPABILITY_ANNOUNCE, track_cap_b)
            
            # Start both nodes
            await node_a.start()
            await node_b.start(seed_peers=[f"localhost:{port_a}"])
            
            # Wait for connection
            await asyncio.sleep(2)
            
            # Both announce capabilities
            await node_a.broadcast(
                MessageType.CAPABILITY_ANNOUNCE,
                {
                    "capability_id": "node-a-llm",
                    "capability": "text-generation",
                    "model": "llama-3.1-70b",
                }
            )
            
            await node_b.broadcast(
                MessageType.CAPABILITY_ANNOUNCE,
                {
                    "capability_id": "node-b-embed",
                    "capability": "embeddings",
                    "model": "nomic-embed-text",
                }
            )
            
            # Wait for propagation
            await asyncio.sleep(2)
            
            # Verify Node A sees Node B's capability
            assert "node-b-embed" in capabilities_seen_by_a
            assert capabilities_seen_by_a["node-b-embed"]["peer"] == "node-b"
            
            # Verify Node B sees Node A's capability
            assert "node-a-llm" in capabilities_seen_by_b
            assert capabilities_seen_by_b["node-a-llm"]["peer"] == "node-a"
            
            logger.info("✅ Both nodes see each other's capabilities")
            logger.info(f"Node A sees: {list(capabilities_seen_by_a.keys())}")
            logger.info(f"Node B sees: {list(capabilities_seen_by_b.keys())}")
            
        finally:
            await node_b.stop()
            await node_a.stop()
    
    @pytest.mark.asyncio
    async def test_reconnection_on_disconnect(self):
        """Test that peers reconnect after disconnect."""
        port_a = get_free_port()
        port_b = get_free_port()
        node_a = GossipNetwork(node_id="node-a", port=port_a)
        node_b = GossipNetwork(node_id="node-b", port=port_b)
        
        try:
            # Start Node A
            await node_a.start()
            
            # Start Node B with Node A as seed
            await node_b.start(seed_peers=[f"localhost:{port_a}"])
            
            # Wait for initial connection
            await asyncio.sleep(2)
            assert "node-b" in node_a.connected_peers
            assert "node-a" in node_b.connected_peers
            
            logger.info("✅ Initial connection established")
            
            # Simulate disconnect by stopping Node A
            await node_a.stop()
            logger.info("Node A stopped")
            
            # Wait for disconnect detection
            await asyncio.sleep(3)
            
            # Restart Node A
            node_a = GossipNetwork(node_id="node-a", port=port_a)
            await node_a.start()
            logger.info("Node A restarted")
            
            # Wait for reconnection (Node B should retry connection)
            # The _connect_seeds task runs every 30 seconds, so this might take time
            # For testing, we'll wait up to 35 seconds
            max_wait = 35
            for i in range(max_wait):
                await asyncio.sleep(1)
                if "node-a" in node_b.connected_peers and "node-b" in node_a.connected_peers:
                    logger.info(f"✅ Reconnection successful after {i+1} seconds")
                    break
            else:
                pytest.fail("Reconnection did not happen within timeout")
            
        finally:
            await node_b.stop()
            await node_a.stop()
    
    @pytest.mark.asyncio
    async def test_heartbeat_propagation(self):
        """Test that heartbeats propagate between nodes."""
        port_a = get_free_port()
        port_b = get_free_port()
        node_a = GossipNetwork(node_id="node-a", port=port_a)
        node_b = GossipNetwork(node_id="node-b", port=port_b)
        
        heartbeats_received = []
        
        def track_heartbeat(peer, msg):
            heartbeats_received.append({
                "from": peer.node_id,
                "time": msg.payload.get("time"),
                "peers": msg.payload.get("peers")
            })
            logger.info(f"Heartbeat from {peer.node_id}: {msg.payload}")
        
        try:
            # Register handler
            node_b.on(MessageType.HEARTBEAT, track_heartbeat)
            
            # Start both nodes
            await node_a.start()
            await node_b.start(seed_peers=[f"localhost:{port_a}"])
            
            # Wait for connection and heartbeats
            # Heartbeat interval is 30 seconds, so we need to wait
            # Let's manually trigger one for testing
            await asyncio.sleep(2)
            
            # Manually send heartbeat from Node A
            await node_a.broadcast(
                MessageType.HEARTBEAT,
                {"time": 12345, "peers": len(node_a.peers)}
            )
            
            await asyncio.sleep(1)
            
            # Node B should have received it
            assert len(heartbeats_received) > 0
            hb = heartbeats_received[0]
            assert hb["from"] == "node-a"
            assert hb["time"] == 12345
            
            logger.info("✅ Heartbeat propagation works")
            
        finally:
            await node_b.stop()
            await node_a.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
