"""
Test TCP gossip between Atmosphere network nodes.
"""

import asyncio
import pytest
from atmosphere.mesh.network import GossipNetwork, MessageType


@pytest.mark.asyncio
async def test_tcp_gossip_handshake_and_broadcast():
    """
    Test TCP gossip between two nodes:
    - Node A starts on port 11450
    - Node B starts on port 11451 with Node A as seed
    - Verify handshake completes
    - Node A broadcasts a capability
    - Node B receives the capability
    - Both nodes see each other as peers
    """
    
    # Track received messages
    node_b_received_capability = asyncio.Event()
    received_capability_data = {}
    
    def on_capability_received(peer, msg):
        """Handler for Node B to receive capability announcements."""
        received_capability_data['peer_id'] = peer.node_id
        received_capability_data['payload'] = msg.payload
        node_b_received_capability.set()
    
    # Create Node A on port 11450
    node_a = GossipNetwork(
        node_id="node-a",
        port=11450,
        bind_host="127.0.0.1"
    )
    
    # Create Node B on port 11451
    node_b = GossipNetwork(
        node_id="node-b",
        port=11451,
        bind_host="127.0.0.1"
    )
    
    # Register capability handler on Node B
    node_b.on(MessageType.CAPABILITY_ANNOUNCE, on_capability_received)
    
    try:
        # Start Node A (no seeds)
        await node_a.start()
        print(f"✓ Node A started on port 11450")
        
        # Start Node B with Node A as seed
        await node_b.start(seed_peers=["127.0.0.1:11450"])
        print(f"✓ Node B started on port 11451 with Node A as seed")
        
        # Wait for handshake to complete (give nodes time to connect)
        await asyncio.sleep(2)
        
        # Verify handshake: both nodes should see each other as peers
        assert "node-b" in node_a.connected_peers, \
            f"Node A should see Node B as peer. Connected peers: {node_a.connected_peers}"
        print(f"✓ Node A sees Node B as peer")
        
        assert "node-a" in node_b.connected_peers, \
            f"Node B should see Node A as peer. Connected peers: {node_b.connected_peers}"
        print(f"✓ Node B sees Node A as peer")
        
        # Node A broadcasts a capability announcement
        capability_payload = {
            "capability": "model-inference",
            "model": "llama-3.1-70b",
            "free_memory_gb": 40
        }
        
        await node_a.broadcast(
            MessageType.CAPABILITY_ANNOUNCE,
            capability_payload
        )
        print(f"✓ Node A broadcast capability announcement")
        
        # Wait for Node B to receive the capability
        await asyncio.wait_for(node_b_received_capability.wait(), timeout=5.0)
        print(f"✓ Node B received capability announcement")
        
        # Verify the received data
        assert received_capability_data['peer_id'] == "node-a", \
            f"Expected peer_id 'node-a', got {received_capability_data['peer_id']}"
        
        assert received_capability_data['payload'] == capability_payload, \
            f"Payload mismatch. Expected {capability_payload}, got {received_capability_data['payload']}"
        
        print(f"✓ Capability data verified: {received_capability_data['payload']}")
        
        # Final verification: check peer counts
        assert len(node_a.connected_peers) == 1, \
            f"Node A should have 1 connected peer, has {len(node_a.connected_peers)}"
        assert len(node_b.connected_peers) == 1, \
            f"Node B should have 1 connected peer, has {len(node_b.connected_peers)}"
        
        print(f"✓ Both nodes have exactly 1 peer each")
        
        # Get detailed status
        status_a = node_a.get_status()
        status_b = node_b.get_status()
        
        print(f"\n📊 Node A Status:")
        print(f"   - Node ID: {status_a['node_id']}")
        print(f"   - Port: {status_a['port']}")
        print(f"   - Running: {status_a['running']}")
        print(f"   - Peers: {list(status_a['peers'].keys())}")
        
        print(f"\n📊 Node B Status:")
        print(f"   - Node ID: {status_b['node_id']}")
        print(f"   - Port: {status_b['port']}")
        print(f"   - Running: {status_b['running']}")
        print(f"   - Peers: {list(status_b['peers'].keys())}")
        
        print(f"\n✅ All gossip tests passed!")
        
    finally:
        # Clean up: stop both nodes
        await node_a.stop()
        await node_b.stop()
        print(f"\n🛑 Nodes stopped cleanly")


if __name__ == "__main__":
    # Allow running directly with python
    asyncio.run(test_tcp_gossip_handshake_and_broadcast())
