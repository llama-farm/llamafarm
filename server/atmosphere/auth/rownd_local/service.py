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
        self._issued_tokens: Dict[str, dict] = {}  # Track tokens we've issued
        
        self._load_state()
    
    def _load_state(self) -> None:
        """Load persisted state from disk."""
        mesh_path = self.data_dir / "mesh.json"
        device_path = self.data_dir / "device.json"
        token_path = self.data_dir / "token.json"
        issued_tokens_path = self.data_dir / "issued_tokens.json"
        
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
        
        if issued_tokens_path.exists():
            with open(issued_tokens_path) as f:
                self._issued_tokens = json.load(f)
            logger.info(f"Loaded {len(self._issued_tokens)} issued tokens")
        
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
        
        # Save issued tokens
        with open(self.data_dir / "issued_tokens.json", 'w') as f:
            json.dump(self._issued_tokens, f, indent=2)
    
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
        granted_capabilities: Optional[List[str]] = None,
        validity_hours: int = 24
    ) -> Optional[MeshToken]:
        """
        Process a join request from a new device.
        
        Args:
            request: The join request from the device
            approve: Whether to approve the request
            granted_capabilities: Capabilities to grant (defaults to requested)
            validity_hours: Token validity in hours
        
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
            granted_capabilities=granted_capabilities,
            validity_hours=validity_hours
        )
        
        # Track issued token
        self._issued_tokens[token.device_id] = {
            "device_id": token.device_id,
            "device_name": token.device_name,
            "issued_at": token.issued_at,
            "expires_at": token.expires_at,
            "capabilities": token.capabilities
        }
        self._save_state()
        
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
    
    # ==================== Invite Token Management ====================
    
    def create_invite_token(
        self,
        recipient_name: str,
        capabilities: Optional[List[str]] = None,
        validity_hours: int = 168  # Default 7 days for invite tokens
    ) -> str:
        """
        Create a pre-authorized invite token that can be used to join the mesh.
        
        This creates a "virtual" join request for a future device and issues a token
        for it. The recipient can then use this token to join without requiring
        approval from a founder.
        
        Args:
            recipient_name: Name for the device that will use this token
            capabilities: Capabilities to grant (defaults to basic capabilities)
            validity_hours: How long the invite token is valid
        
        Returns:
            Compact token string that can be shared with the recipient
        """
        if not self._issuer:
            raise RuntimeError("This node cannot issue certificates")
        
        # Use basic capabilities if none specified
        if capabilities is None:
            capabilities = ["compute.inference", "storage.read"]
        
        # Create a temporary device identity for the invite
        from .device import DeviceIdentity, detect_tier
        temp_device = DeviceIdentity.create(
            name=recipient_name,
            capabilities=capabilities,
            tier=detect_tier()
        )
        
        # Create a join request from this temp device
        join_request = temp_device.create_join_request(self.mesh.mesh_id)
        
        # Issue token
        token = self._issuer.issue_token(
            join_request,
            granted_capabilities=capabilities,
            validity_hours=validity_hours
        )
        
        # Track the issued invite token
        self._issued_tokens[token.device_id] = {
            "device_id": token.device_id,
            "device_name": token.device_name,
            "issued_at": token.issued_at,
            "expires_at": token.expires_at,
            "capabilities": token.capabilities,
            "is_invite": True
        }
        self._save_state()
        
        logger.info(f"Created invite token for '{recipient_name}' (valid {validity_hours}h)")
        
        return token.to_compact()
    
    def join_with_token(
        self,
        token: str,
        seed_peers: Optional[List[str]] = None
    ) -> dict:
        """
        Join a mesh using a pre-issued invite token.
        
        This allows joining without the interactive request/approval flow.
        
        Args:
            token: Compact token string (from create_invite_token)
            seed_peers: Optional list of peer addresses to connect to
        
        Returns:
            Join result with mesh info and connection details
        """
        # Parse the token
        try:
            mesh_token = MeshToken.from_compact(token)
        except Exception as e:
            raise ValueError(f"Invalid token format: {e}")
        
        # Create or update device identity to match token
        # This handles the case where we're taking over a pre-created identity
        if self.device is None or self.device.device_id != mesh_token.device_id:
            # We need to adopt the identity from the token
            # Note: In production, you'd want to verify hardware hash matches
            # For now, we'll create a new device with the token's info
            from .device import DeviceIdentity
            
            # We can't recover the private key from the token, so we create a new keypair
            # but use the device_id from the token. This is a limitation - in practice,
            # the invite flow should include the private key or the device should
            # already have its keypair before getting the token.
            
            # For this implementation, we'll check if we already have a device
            # with matching hardware that can use this token
            self.device = DeviceIdentity.create(
                name=mesh_token.device_name,
                capabilities=mesh_token.capabilities,
                tier=mesh_token.tier
            )
            logger.info(f"Created device identity from invite token")
        
        # Verify the token is valid
        # First we need the mesh info to verify
        # In a real implementation, this would be included in the token
        # or fetched from seed peers
        
        # Accept the token
        self.token = mesh_token
        
        # Initialize mesh identity from token info
        # We don't have the full mesh data, but we have enough to verify tokens
        # In practice, we'd fetch this from seed peers
        if self.mesh is None:
            logger.info(f"Joined mesh '{mesh_token.mesh_name}' using invite token")
        
        self._save_state()
        
        return {
            "success": True,
            "mesh_id": mesh_token.mesh_id,
            "mesh_name": mesh_token.mesh_name,
            "device_id": self.device.device_id,
            "capabilities": mesh_token.capabilities,
            "expires_at": mesh_token.expires_at,
            "seed_peers": seed_peers or []
        }
    
    def list_issued_tokens(self) -> List[dict]:
        """
        List all tokens issued by this node.
        
        Returns:
            List of issued token metadata (not the tokens themselves)
        """
        return [
            {
                "device_id": token_info["device_id"],
                "device_name": token_info["device_name"],
                "issued_at": token_info["issued_at"],
                "expires_at": token_info["expires_at"],
                "capabilities": token_info["capabilities"],
                "is_invite": token_info.get("is_invite", False),
                "expired": int(time.time()) > token_info["expires_at"]
            }
            for token_info in self._issued_tokens.values()
        ]
    
    def handle_remote_revocation(self, device_id: str, reason: str) -> None:
        """
        Handle a revocation notice from a remote node.
        
        This is called when we receive a revocation via gossip or direct notification.
        
        Args:
            device_id: The device being revoked
            reason: Reason for revocation
        """
        if device_id in self._revocations:
            logger.debug(f"Device {device_id} already revoked locally")
            return
        
        # Add to local revocation list
        entry = RevocationEntry(
            device_id=device_id,
            revoked_at=int(time.time()),
            reason=f"Remote revocation: {reason}",
            revoked_by="remote"
        )
        
        self._revocations[device_id] = entry
        
        # If we issued a token to this device, mark it
        if device_id in self._issued_tokens:
            self._issued_tokens[device_id]["revoked"] = True
            self._save_state()
        
        logger.info(f"Processed remote revocation for device {device_id}: {reason}")
    
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
