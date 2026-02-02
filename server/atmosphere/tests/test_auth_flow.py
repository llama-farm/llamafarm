"""
Test the complete authentication flow including token export/import.

This tests:
- Mesh creation
- Invite token generation
- Joining with invite tokens
- Token verification offline
- Token revocation
"""

import json
import os
import tempfile
import time
from pathlib import Path

import pytest

from atmosphere.auth.rownd_local.service import RowndLocalService
from atmosphere.auth.rownd_local.tokens import MeshToken


class TestAuthFlow:
    """Test complete authentication flows."""
    
    def test_create_mesh_and_invite_token(self):
        """Test creating a mesh and generating an invite token."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a founder node
            founder = RowndLocalService(data_dir=Path(tmpdir) / "founder")
            
            # Create mesh
            mesh_info = founder.create_mesh(
                name="test-mesh",
                threshold=1,
                shares=1,
                device_name="founder-node"
            )
            
            assert mesh_info["name"] == "test-mesh"
            assert mesh_info["is_founder"] is True
            assert founder.mesh is not None
            assert founder.device is not None
            
            # Create an invite token
            invite_token = founder.create_invite_token(
                recipient_name="invited-device",
                capabilities=["compute.inference", "storage.read"],
                validity_hours=24
            )
            
            assert invite_token is not None
            assert isinstance(invite_token, str)
            assert len(invite_token) > 0
            
            # Verify we can parse the token
            token = MeshToken.from_compact(invite_token)
            assert token.mesh_id == mesh_info["mesh_id"]
            assert token.device_name == "invited-device"
            assert token.capabilities == ["compute.inference", "storage.read"]
            assert not token.is_expired
    
    def test_join_with_invite_token(self):
        """Test joining a mesh using a pre-issued invite token."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: Create founder and mesh
            founder = RowndLocalService(data_dir=Path(tmpdir) / "founder")
            mesh_info = founder.create_mesh(
                name="test-mesh",
                threshold=1,
                shares=1,
                device_name="founder-node"
            )
            
            # Step 2: Create invite token
            invite_token = founder.create_invite_token(
                recipient_name="new-member",
                capabilities=["compute.inference"],
                validity_hours=24
            )
            
            # Step 3: New device joins using the invite token
            new_device = RowndLocalService(data_dir=Path(tmpdir) / "new-device")
            
            join_result = new_device.join_with_token(
                token=invite_token,
                seed_peers=["founder-node:8080"]
            )
            
            assert join_result["success"] is True
            assert join_result["mesh_name"] == "test-mesh"
            assert join_result["mesh_id"] == mesh_info["mesh_id"]
            assert "compute.inference" in join_result["capabilities"]
            assert new_device.token is not None
            assert not new_device.token.is_expired
    
    def test_verify_token_offline(self):
        """Test that tokens can be verified offline without network access."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create founder and mesh
            founder = RowndLocalService(data_dir=Path(tmpdir) / "founder")
            mesh_info = founder.create_mesh(
                name="test-mesh",
                threshold=1,
                shares=1,
                device_name="founder-node"
            )
            
            # Create invite token
            invite_token = founder.create_invite_token(
                recipient_name="device-a",
                capabilities=["compute.inference", "storage.read"],
                validity_hours=24
            )
            
            # Parse token
            token = MeshToken.from_compact(invite_token)
            
            # Verify token using founder's verifier
            is_valid, reason = founder.verify_token(token)
            assert is_valid is True
            assert reason == "Token is valid"
            
            # Verify using compact format
            is_valid_compact, reason_compact = founder.verify_compact_token(invite_token)
            assert is_valid_compact is True
            assert reason_compact == "Token is valid"
    
    def test_list_issued_tokens(self):
        """Test listing all issued tokens."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create founder and mesh
            founder = RowndLocalService(data_dir=Path(tmpdir) / "founder")
            founder.create_mesh(
                name="test-mesh",
                threshold=1,
                shares=1,
                device_name="founder-node"
            )
            
            # Issue multiple invite tokens
            invite1 = founder.create_invite_token(
                recipient_name="device-1",
                capabilities=["compute.inference"],
                validity_hours=24
            )
            
            invite2 = founder.create_invite_token(
                recipient_name="device-2",
                capabilities=["storage.read"],
                validity_hours=48
            )
            
            # List issued tokens
            issued = founder.list_issued_tokens()
            
            assert len(issued) == 2
            
            # Check first token
            token1 = next(t for t in issued if t["device_name"] == "device-1")
            assert token1["capabilities"] == ["compute.inference"]
            assert token1["is_invite"] is True
            assert token1["expired"] is False
            
            # Check second token
            token2 = next(t for t in issued if t["device_name"] == "device-2")
            assert token2["capabilities"] == ["storage.read"]
            assert token2["is_invite"] is True
            assert token2["expired"] is False
    
    def test_token_expiration(self):
        """Test that expired tokens are properly rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create founder and mesh
            founder = RowndLocalService(data_dir=Path(tmpdir) / "founder")
            founder.create_mesh(
                name="test-mesh",
                threshold=1,
                shares=1,
                device_name="founder-node"
            )
            
            # Create a token with very short validity (negative to make it already expired)
            # We'll manually create an expired token by manipulating the time
            invite_token = founder.create_invite_token(
                recipient_name="device-expired",
                capabilities=["compute.inference"],
                validity_hours=1
            )
            
            token = MeshToken.from_compact(invite_token)
            
            # Manually expire the token by setting expires_at to past
            token.expires_at = int(time.time()) - 3600  # 1 hour ago
            
            # Verify token is rejected
            is_valid, reason = founder.verify_token(token)
            assert is_valid is False
            assert "expired" in reason.lower()
    
    def test_revocation_propagation(self):
        """Test that device revocation is properly handled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create founder and mesh
            founder = RowndLocalService(data_dir=Path(tmpdir) / "founder")
            founder.create_mesh(
                name="test-mesh",
                threshold=1,
                shares=1,
                device_name="founder-node"
            )
            
            # Issue a token
            invite_token = founder.create_invite_token(
                recipient_name="device-to-revoke",
                capabilities=["compute.inference"],
                validity_hours=24
            )
            
            token = MeshToken.from_compact(invite_token)
            
            # Verify token is initially valid
            is_valid, reason = founder.verify_token(token)
            assert is_valid is True
            
            # Revoke the device
            founder.revoke_device(token.device_id, "Security breach")
            
            # Verify token is now rejected
            is_valid, reason = founder.verify_token(token)
            assert is_valid is False
            assert "revoked" in reason.lower()
    
    def test_remote_revocation_handling(self):
        """Test handling revocation notices from remote nodes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two nodes
            node1 = RowndLocalService(data_dir=Path(tmpdir) / "node1")
            node1.create_mesh(
                name="test-mesh",
                threshold=1,
                shares=1,
                device_name="node1"
            )
            
            node2 = RowndLocalService(data_dir=Path(tmpdir) / "node2")
            
            # Node1 issues token
            invite_token = node1.create_invite_token(
                recipient_name="device-x",
                capabilities=["compute.inference"],
                validity_hours=24
            )
            
            token = MeshToken.from_compact(invite_token)
            
            # Node2 handles remote revocation from node1
            node2.handle_remote_revocation(
                device_id=token.device_id,
                reason="Detected compromise"
            )
            
            # Check that node2 has the revocation
            revocations = node2.get_revocation_list()
            assert len(revocations) == 1
            assert revocations[0]["device_id"] == token.device_id
            assert "Detected compromise" in revocations[0]["reason"]
    
    def test_full_mesh_join_flow(self):
        """Test the complete flow from mesh creation to device joining."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # === PHASE 1: Founder creates mesh ===
            founder = RowndLocalService(data_dir=Path(tmpdir) / "founder")
            mesh_info = founder.create_mesh(
                name="production-mesh",
                threshold=1,
                shares=1,
                device_name="founder-node"
            )
            
            founder_status = founder.get_status()
            assert founder_status["initialized"] is True
            assert founder_status["can_issue"] is True
            
            # === PHASE 2: Generate invite tokens for multiple devices ===
            token_alpha = founder.create_invite_token(
                recipient_name="alpha-device",
                capabilities=["compute.inference", "storage.read"],
                validity_hours=168  # 7 days
            )
            
            token_beta = founder.create_invite_token(
                recipient_name="beta-device",
                capabilities=["compute.inference"],
                validity_hours=168
            )
            
            # === PHASE 3: Devices join using their invite tokens ===
            alpha = RowndLocalService(data_dir=Path(tmpdir) / "alpha")
            alpha_join = alpha.join_with_token(
                token=token_alpha,
                seed_peers=["founder.local:8080"]
            )
            
            assert alpha_join["success"] is True
            assert alpha_join["mesh_name"] == "production-mesh"
            assert "compute.inference" in alpha_join["capabilities"]
            assert "storage.read" in alpha_join["capabilities"]
            
            beta = RowndLocalService(data_dir=Path(tmpdir) / "beta")
            beta_join = beta.join_with_token(
                token=token_beta,
                seed_peers=["founder.local:8080"]
            )
            
            assert beta_join["success"] is True
            assert beta_join["mesh_name"] == "production-mesh"
            assert "compute.inference" in beta_join["capabilities"]
            
            # === PHASE 4: Verify tokens offline ===
            alpha_token_obj = MeshToken.from_compact(token_alpha)
            is_valid_alpha, _ = founder.verify_token(alpha_token_obj)
            assert is_valid_alpha is True
            
            beta_token_obj = MeshToken.from_compact(token_beta)
            is_valid_beta, _ = founder.verify_token(beta_token_obj)
            assert is_valid_beta is True
            
            # === PHASE 5: List all issued tokens ===
            issued = founder.list_issued_tokens()
            assert len(issued) == 2
            
            device_names = {t["device_name"] for t in issued}
            assert "alpha-device" in device_names
            assert "beta-device" in device_names
            
            # === PHASE 6: Revoke one device ===
            founder.revoke_device(
                alpha_token_obj.device_id,
                "Device compromised"
            )
            
            # Verify alpha's token is now invalid
            is_valid_alpha_after, reason = founder.verify_token(alpha_token_obj)
            assert is_valid_alpha_after is False
            assert "revoked" in reason.lower()
            
            # Verify beta's token still works
            is_valid_beta_after, _ = founder.verify_token(beta_token_obj)
            assert is_valid_beta_after is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
