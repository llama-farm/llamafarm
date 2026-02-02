"""
Basic Atmosphere tests.

Validates core functionality works.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path


class TestAuthIntegration:
    """Test Rownd Local auth integration."""
    
    def test_import_rownd_local(self):
        """Can import Rownd Local components."""
        from atmosphere.auth.rownd_local import (
            RowndLocalService,
            MeshIdentity,
            DeviceIdentity,
            TokenIssuer,
            TokenVerifier,
        )
        assert RowndLocalService is not None
    
    def test_create_keypair(self):
        """Can create Ed25519 keypairs."""
        from atmosphere.auth.identity import KeyPair
        
        kp = KeyPair.generate()
        # Check that key exists (different attr names in different versions)
        assert hasattr(kp, 'public_bytes') or hasattr(kp, 'public_key_bytes')
    
    def test_sign_verify(self):
        """Can sign and verify messages."""
        from atmosphere.auth.identity import KeyPair, verify_signature_b64
        
        kp = KeyPair.generate()
        message = b"test message"
        signature = kp.sign(message)
        
        # Use verify_signature_b64 for verification
        import base64
        sig_b64 = base64.b64encode(signature).decode()
        # public_bytes is a method in some versions, property in others
        pub_bytes = kp.public_bytes() if callable(kp.public_bytes) else kp.public_bytes
        pub_b64 = base64.b64encode(pub_bytes).decode()
        assert verify_signature_b64(pub_b64, message, sig_b64)


class TestMeshComponents:
    """Test mesh components."""
    
    def test_import_mesh(self):
        """Can import mesh components."""
        from atmosphere import Node, GossipProtocol, GradientTable
        assert Node is not None
        assert GradientTable is not None
    
    def test_gradient_table(self):
        """Gradient table basic operations."""
        from atmosphere.router.gradient import GradientTable
        import numpy as np
        
        table = GradientTable(node_id="test-node")
        
        # Add a route
        cap_vector = np.random.randn(384).astype(np.float32)
        cap_vector = cap_vector / np.linalg.norm(cap_vector)
        
        table.update(
            capability_id="test-cap",
            capability_label="test",
            capability_vector=cap_vector,
            next_hop="peer-1",
            via_node="peer-1",
            hops=1,
            estimated_latency_ms=50
        )
        
        # Find route
        route = table.find_best_route(cap_vector)
        assert route is not None
        assert route.next_hop == "peer-1"


class TestDiscovery:
    """Test discovery adapters."""
    
    def test_import_adapters(self):
        """Can import discovery adapters."""
        from atmosphere.discovery import (
            LlamaFarmAdapter,
            OllamaBackend,
        )
        assert LlamaFarmAdapter is not None
        assert OllamaBackend is not None
    
    def test_llamafarm_adapter_init(self):
        """Can initialize LlamaFarm adapter."""
        from atmosphere.discovery import LlamaFarmAdapter
        
        adapter = LlamaFarmAdapter("http://localhost:14345")
        assert adapter.url == "http://localhost:14345"
        assert not adapter.is_connected


class TestService:
    """Test main service."""
    
    def test_import_service(self):
        """Can import service."""
        from atmosphere.service import AtmosphereService
        assert AtmosphereService is not None
    
    def test_service_init(self):
        """Can initialize service with temp directory."""
        from atmosphere.service import AtmosphereService
        
        with tempfile.TemporaryDirectory() as tmpdir:
            service = AtmosphereService(
                data_dir=Path(tmpdir),
                gossip_port=11450,
                api_port=11451
            )
            assert service.config.gossip_port == 11450


class TestNetwork:
    """Test network gossip."""
    
    def test_import_network(self):
        """Can import network components."""
        from atmosphere.mesh.network import (
            GossipNetwork,
            MessageType,
            GossipMessage,
        )
        assert GossipNetwork is not None
    
    def test_gossip_message(self):
        """Can create gossip messages."""
        from atmosphere.mesh.network import GossipMessage, MessageType
        
        msg = GossipMessage(
            type=MessageType.HEARTBEAT,
            sender_id="test-node",
            payload={"time": 12345}
        )
        
        # Serialize and deserialize
        json_str = msg.to_json()
        msg2 = GossipMessage.from_json(json_str)
        
        assert msg2.type == MessageType.HEARTBEAT
        assert msg2.sender_id == "test-node"
        assert msg2.payload["time"] == 12345


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
