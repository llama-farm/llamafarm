"""
Rownd Local Service: The main authentication service for the mesh.

This runs on each node and handles:
- Device identity management
- Token issuance and verification
- Join request processing
- Revocation propagation
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from .mesh import MeshIdentity
from .device import DeviceIdentity, detect_capabilities, detect_tier
from .tokens import MeshToken, TokenIssuer, TokenVerifier

logger = logging.getLogger(__name__)


@dataclass
class RevocationEntry:
    """A revoked device or token."""
    device_id: str
    revoked_at: int
    reason: str
    revoked_by: str


class RowndLocalService:
    """
    The main Rownd Local authentication service.
    
    Runs on each mesh node to handle identity and authorization.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the service.
        
        Args:
            data_dir: Directory for storing identity data
        """
        if data_dir is None:
            data_dir = Path.home() / ".rownd-local"
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.mesh: Optional[MeshIdentity] = None
        self.device: Optional[DeviceIdentity] = None
        self.token: Optional[MeshToken] = None
        
        self._issuer: Optional[TokenIssuer] = None
        self._verifier: Optional[TokenVerifier] = None
        
        self._revocations: Dict[str, RevocationEntry] = {}
        self._pending_requests: Dict[str, dict] = {}
        
        self._load_state()
    
    def _load_state(self) -> None:
        """Load persisted state from disk."""
        mesh_path = self.data_dir / "mesh.json"
        device_path = self.data_dir / "device.json"
        token_path = self.data_dir / "token.json"
        
        if mesh_path.exists():
            self.mesh = MeshIdentity.load(mesh_path)
            logger.info(f"Loaded mesh: {self.mesh.name} ({self.mesh.mesh_id})")
        
        if device_path.exists():
            self.device = DeviceIdentity.load(device_path)
            logger.info(f"Loaded device: {self.device.name} ({self.device.device_id})")
        
        if token_path.exists():
            with open(token_path) as f:
                self.token = MeshToken.from_dict(json.load(f))
            logger.info(f"Loaded token (expires in {self.token.time_remaining}s)")
        
        self._init_components()
    
    def _init_components(self) -> None:
        """Initialize issuer and verifier if possible."""
        if self.mesh and self.device:
            if self.mesh.can_issue_certificates():
                try:
                    self._issuer = TokenIssuer(self.mesh, self.device)
                    logger.info("Token issuer initialized (this node can issue certificates)")
                except Exception as e:
                    logger.warning(f"Could not initialize issuer: {e}")
            
            self._verifier = TokenVerifier(
                self.mesh.master_public_key,
                [f.__dict__ for f in self.mesh.founding_members]
            )
    
    def _save_state(self) -> None:
        """Persist state to disk."""
        if self.mesh:
            self.mesh.save(self.data_dir / "mesh.json")
        
        if self.device:
            self.device.save(self.data_dir / "device.json")
        
        if self.token:
            with open(self.data_dir / "token.json", 'w') as f:
                json.dump(self.token.to_dict(), f, indent=2)
    
    # ==================== Mesh Management ====================
    
    def create_mesh(
        self,
        name: str,
        threshold: int = 2,
        shares: int = 3,
        device_name: Optional[str] = None
    ) -> dict:
        """
        Create a new mesh and become a founding member.
        
        Args:
            name: Human-readable mesh name
            threshold: Minimum founders needed to issue certificates
            shares: Total number of founding positions
            device_name: Name for this device
        
        Returns:
            Mesh info including shares to distribute
        """
        if self.mesh:
            raise RuntimeError("Already part of a mesh")
        
        # Detect capabilities
        capabilities = detect_capabilities()
        tier = detect_tier()
        
        # Create mesh (it generates the node keypair)
        self.mesh = MeshIdentity.create(
            name=name,
            threshold=threshold,
            total_shares=shares,
            founding_capabilities=capabilities
        )
        
        # Create device identity using the mesh's node keypair
        # This ensures the device ID matches the founding member ID
        founder = self.mesh.founding_members[0]
        from .crypto import get_hardware_fingerprint
        
        self.device = DeviceIdentity(
            device_id=founder.node_id,
            public_key=founder.public_key,
            hardware_hash=get_hardware_fingerprint(),
            name=device_name or f"founder-{os.uname().nodename}",
            capabilities=capabilities,
            tier=tier,
            created_at=int(time.time())
        )
        # Use the mesh's node keypair
        self.device._key_pair = self.mesh._local_key_pair
        
        # Initialize components
        self._init_components()
        self._save_state()
        
        logger.info(f"Created mesh '{name}' with {shares} founding positions")
        
        return {
            "mesh_id": self.mesh.mesh_id,
            "name": self.mesh.name,
            "device_id": self.device.device_id,
            "is_founder": True,
            "pending_shares": len(self.mesh.get_pending_shares()),
            "threshold": threshold
        }
    
    def get_founder_invite(self, share_index: int) -> dict:
        """Get an invite for another founding member."""
        if not self.mesh:
            raise RuntimeError("No mesh configured")
        
        share_data = self.mesh.export_share_for_founder(share_index)
        return {
            "type": "founder_invite",
            "share_data": share_data,
            "instructions": "Import this on the new founding device using 'rownd-local join-as-founder'"
        }
    
    # ==================== Device Management ====================
    
    def create_join_request(self, mesh_id: str, device_name: Optional[str] = None) -> dict:
        """
        Create a request to join a mesh.
        
        Args:
            mesh_id: The mesh to join
            device_name: Name for this device
        
        Returns:
            Join request to present to a founder
        """
        if self.device is None:
            self.device = DeviceIdentity.create(
                name=device_name or f"device-{os.uname().nodename}",
                capabilities=detect_capabilities(),
                tier=detect_tier()
            )
            self._save_state()
        
        return self.device.create_join_request(mesh_id)
    
    def process_join_request(
        self,
        request: dict,
        approve: bool = True,
        granted_capabilities: Optional[List[str]] = None
    ) -> Optional[MeshToken]:
        """
        Process a join request from a new device.
        
        Args:
            request: The join request from the device
            approve: Whether to approve the request
            granted_capabilities: Capabilities to grant (defaults to requested)
        
        Returns:
            MeshToken if approved, None otherwise
        """
        if not self._issuer:
            raise RuntimeError("This node cannot issue certificates")
        
        if not approve:
            logger.info(f"Rejected join request from {request['device']['device_id']}")
            return None
        
        token = self._issuer.issue_token(
            request,
            granted_capabilities=granted_capabilities
        )
        
        logger.info(f"Issued token to {token.device_id} ({token.device_name})")
        return token
    
    def accept_token(self, token: MeshToken) -> bool:
        """
        Accept a token issued to this device.
        
        Args:
            token: The token issued by a founder
        
        Returns:
            True if token is valid and accepted
        """
        # Verify the token
        if self._verifier:
            is_valid, reason = self._verifier.verify(token)
            if not is_valid:
                logger.error(f"Token rejected: {reason}")
                return False
        
        # Check it's for this device
        if token.device_id != self.device.device_id:
            logger.error("Token is for a different device")
            return False
        
        self.token = token
        self._save_state()
        
        logger.info(f"Accepted token for mesh {token.mesh_name}")
        return True
    
    # ==================== Verification ====================
    
    def verify_token(self, token: MeshToken) -> tuple:
        """
        Verify a token from another device.
        
        Returns:
            (is_valid, reason) tuple
        """
        if not self._verifier:
            return False, "No verifier configured"
        
        # Check revocation list
        if token.device_id in self._revocations:
            rev = self._revocations[token.device_id]
            return False, f"Device revoked: {rev.reason}"
        
        return self._verifier.verify(token)
    
    def verify_compact_token(self, compact: str) -> tuple:
        """Verify a compact (base64) token string."""
        try:
            token = MeshToken.from_compact(compact)
            return self.verify_token(token)
        except Exception as e:
            return False, f"Invalid token format: {e}"
    
    # ==================== Revocation ====================
    
    def revoke_device(self, device_id: str, reason: str) -> RevocationEntry:
        """
        Revoke a device's access.
        
        This is propagated via gossip to other nodes.
        """
        if not self.device:
            raise RuntimeError("No device identity")
        
        entry = RevocationEntry(
            device_id=device_id,
            revoked_at=int(time.time()),
            reason=reason,
            revoked_by=self.device.device_id
        )
        
        self._revocations[device_id] = entry
        logger.info(f"Revoked device {device_id}: {reason}")
        
        return entry
    
    def get_revocation_list(self) -> List[dict]:
        """Get current revocation list for gossip."""
        return [
            {
                "device_id": r.device_id,
                "revoked_at": r.revoked_at,
                "reason": r.reason,
                "revoked_by": r.revoked_by
            }
            for r in self._revocations.values()
        ]
    
    def merge_revocation_list(self, revocations: List[dict]) -> int:
        """Merge revocations from gossip. Returns count of new entries."""
        new_count = 0
        for r in revocations:
            if r["device_id"] not in self._revocations:
                self._revocations[r["device_id"]] = RevocationEntry(
                    device_id=r["device_id"],
                    revoked_at=r["revoked_at"],
                    reason=r["reason"],
                    revoked_by=r["revoked_by"]
                )
                new_count += 1
        return new_count
    
    # ==================== Status ====================
    
    def get_status(self) -> dict:
        """Get current service status."""
        return {
            "initialized": self.mesh is not None,
            "mesh": {
                "id": self.mesh.mesh_id if self.mesh else None,
                "name": self.mesh.name if self.mesh else None,
                "threshold": self.mesh.threshold if self.mesh else None,
                "founders": len(self.mesh.founding_members) if self.mesh else 0
            } if self.mesh else None,
            "device": {
                "id": self.device.device_id if self.device else None,
                "name": self.device.name if self.device else None,
                "capabilities": self.device.capabilities if self.device else [],
                "tier": self.device.tier if self.device else None
            } if self.device else None,
            "token": {
                "valid": self.token is not None and not self.token.is_expired,
                "expires_in": self.token.time_remaining if self.token else 0
            } if self.token else None,
            "can_issue": self._issuer is not None,
            "revocations": len(self._revocations)
        }
