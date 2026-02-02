"""
Mesh Tokens: Self-contained, offline-verifiable membership certificates.

These tokens prove a device is a member of a mesh without requiring
network access to verify.
"""

import base64
import hashlib
import json
import time
from dataclasses import dataclass
from typing import List, Optional

from .crypto import KeyPair, verify_signature_b64

# Production-hardening constants
CLOCK_SKEW_TOLERANCE = 300  # 5 minutes tolerance for clock differences
EXPIRATION_GRACE_PERIOD = 60  # 1 minute grace period for in-flight requests


@dataclass
class MeshToken:
    """
    A mesh membership token (certificate).
    
    This is a self-contained proof that a device is a member of a mesh.
    It can be verified offline using only the mesh's public key.
    """
    version: int
    mesh_id: str
    mesh_name: str
    
    # Device info
    device_id: str
    device_public_key: str
    device_name: str
    hardware_hash: str
    
    # Authorization
    capabilities: List[str]
    tier: str
    
    # Validity
    issued_at: int
    expires_at: int
    
    # Issuer info (the founding member who issued this)
    issuer_node_id: str
    issuer_public_key: str
    issuer_signature: str
    
    # Mesh signature (proves issuer is authorized by mesh)
    mesh_public_key: str
    
    @property
    def is_expired(self) -> bool:
        """
        Check if token has expired.
        
        Includes grace period for in-flight requests and clock skew tolerance.
        """
        now = time.time()
        # Add grace period + clock skew tolerance to expiration
        effective_expiration = self.expires_at + EXPIRATION_GRACE_PERIOD + CLOCK_SKEW_TOLERANCE
        return now > effective_expiration
    
    @property
    def time_remaining(self) -> int:
        """
        Seconds until expiration (including grace period).
        
        This shows actual usable time remaining before token becomes invalid.
        """
        now = int(time.time())
        effective_expiration = self.expires_at + EXPIRATION_GRACE_PERIOD + CLOCK_SKEW_TOLERANCE
        return max(0, effective_expiration - now)
    
    def to_dict(self) -> dict:
        """Serialize token to dict."""
        return {
            "version": self.version,
            "mesh_id": self.mesh_id,
            "mesh_name": self.mesh_name,
            "device": {
                "device_id": self.device_id,
                "public_key": self.device_public_key,
                "name": self.device_name,
                "hardware_hash": self.hardware_hash
            },
            "authorization": {
                "capabilities": self.capabilities,
                "tier": self.tier
            },
            "validity": {
                "issued_at": self.issued_at,
                "expires_at": self.expires_at
            },
            "issuer": {
                "node_id": self.issuer_node_id,
                "public_key": self.issuer_public_key,
                "signature": self.issuer_signature
            },
            "mesh_public_key": self.mesh_public_key
        }
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def to_compact(self) -> str:
        """Serialize to compact base64 string for transmission."""
        data = json.dumps(self.to_dict(), separators=(',', ':'))
        return base64.urlsafe_b64encode(data.encode()).decode()
    
    @classmethod
    def from_dict(cls, data: dict) -> "MeshToken":
        """Load token from dict."""
        return cls(
            version=data["version"],
            mesh_id=data["mesh_id"],
            mesh_name=data["mesh_name"],
            device_id=data["device"]["device_id"],
            device_public_key=data["device"]["public_key"],
            device_name=data["device"]["name"],
            hardware_hash=data["device"]["hardware_hash"],
            capabilities=data["authorization"]["capabilities"],
            tier=data["authorization"]["tier"],
            issued_at=data["validity"]["issued_at"],
            expires_at=data["validity"]["expires_at"],
            issuer_node_id=data["issuer"]["node_id"],
            issuer_public_key=data["issuer"]["public_key"],
            issuer_signature=data["issuer"]["signature"],
            mesh_public_key=data["mesh_public_key"]
        )
    
    @classmethod
    def from_compact(cls, compact: str) -> "MeshToken":
        """Load token from compact base64 string."""
        data = json.loads(base64.urlsafe_b64decode(compact))
        return cls.from_dict(data)


class TokenIssuer:
    """Issues mesh tokens to approved devices."""
    
    def __init__(self, mesh_identity, device_identity):
        """
        Create a token issuer.
        
        Args:
            mesh_identity: The MeshIdentity for this mesh
            device_identity: This node's DeviceIdentity (must be a founder)
        """
        self.mesh = mesh_identity
        self.device = device_identity
        
        if not mesh_identity.can_issue_certificates():
            raise ValueError("This node is not authorized to issue certificates")
    
    def issue_token(
        self,
        join_request: dict,
        validity_hours: int = 24,
        granted_capabilities: Optional[List[str]] = None
    ) -> MeshToken:
        """
        Issue a membership token to a device.
        
        Args:
            join_request: The device's join request
            validity_hours: How long the token is valid
            granted_capabilities: Capabilities to grant (defaults to requested)
        
        Returns:
            MeshToken that the device can use to prove membership
        """
        device_data = join_request["device"]
        
        # Use requested capabilities or grant subset
        capabilities = granted_capabilities or device_data["capabilities"]
        
        now = int(time.time())
        expires = now + (validity_hours * 3600)
        
        # Build token data (without signature)
        token_data = {
            "version": 1,
            "mesh_id": self.mesh.mesh_id,
            "mesh_name": self.mesh.name,
            "device": {
                "device_id": device_data["device_id"],
                "public_key": device_data["public_key"],
                "name": device_data["name"],
                "hardware_hash": device_data["hardware_hash"]
            },
            "authorization": {
                "capabilities": capabilities,
                "tier": device_data["tier"]
            },
            "validity": {
                "issued_at": now,
                "expires_at": expires
            },
            "issuer": {
                "node_id": self.device.device_id,
                "public_key": self.device.public_key
            },
            "mesh_public_key": self.mesh.master_public_key
        }
        
        # Sign the token
        message = json.dumps(token_data, sort_keys=True).encode()
        signature = self.device.sign(message)
        
        return MeshToken(
            version=1,
            mesh_id=self.mesh.mesh_id,
            mesh_name=self.mesh.name,
            device_id=device_data["device_id"],
            device_public_key=device_data["public_key"],
            device_name=device_data["name"],
            hardware_hash=device_data["hardware_hash"],
            capabilities=capabilities,
            tier=device_data["tier"],
            issued_at=now,
            expires_at=expires,
            issuer_node_id=self.device.device_id,
            issuer_public_key=self.device.public_key,
            issuer_signature=signature,
            mesh_public_key=self.mesh.master_public_key
        )


class TokenVerifier:
    """Verifies mesh tokens offline."""
    
    def __init__(self, mesh_public_key: str, founding_members: List[dict]):
        """
        Create a token verifier.
        
        Args:
            mesh_public_key: The mesh's master public key
            founding_members: List of founding member info (for issuer verification)
        """
        self.mesh_public_key = mesh_public_key
        self.founding_members = {
            f["node_id"]: f for f in founding_members
        }
    
    def verify(self, token: MeshToken) -> tuple:
        """
        Verify a mesh token.
        
        Returns:
            (is_valid, reason) tuple
        """
        # Check version
        if token.version != 1:
            return False, f"Unknown token version: {token.version}"
        
        # Check expiration
        if token.is_expired:
            return False, "Token has expired"
        
        # Check mesh public key matches
        if token.mesh_public_key != self.mesh_public_key:
            return False, "Token is for a different mesh"
        
        # Check issuer is a known founding member
        if token.issuer_node_id not in self.founding_members:
            return False, f"Unknown issuer: {token.issuer_node_id}"
        
        founder = self.founding_members[token.issuer_node_id]
        if founder["public_key"] != token.issuer_public_key:
            return False, "Issuer public key mismatch"
        
        # Verify issuer signature
        token_data = {
            "version": token.version,
            "mesh_id": token.mesh_id,
            "mesh_name": token.mesh_name,
            "device": {
                "device_id": token.device_id,
                "public_key": token.device_public_key,
                "name": token.device_name,
                "hardware_hash": token.hardware_hash
            },
            "authorization": {
                "capabilities": token.capabilities,
                "tier": token.tier
            },
            "validity": {
                "issued_at": token.issued_at,
                "expires_at": token.expires_at
            },
            "issuer": {
                "node_id": token.issuer_node_id,
                "public_key": token.issuer_public_key
            },
            "mesh_public_key": token.mesh_public_key
        }
        
        message = json.dumps(token_data, sort_keys=True).encode()
        
        if not verify_signature_b64(
            token.issuer_public_key,
            message,
            token.issuer_signature
        ):
            return False, "Invalid issuer signature"
        
        return True, "Token is valid"
    
    def quick_verify(self, token: MeshToken) -> bool:
        """Quick verification (returns bool only)."""
        is_valid, _ = self.verify(token)
        return is_valid
