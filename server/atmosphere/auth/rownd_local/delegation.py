"""
Agentic Delegation: Allow agents to grant permissions to sub-agents.

Key Use Case (Atmosphere):
  Human (founder) 
    └── issues token to Agent A (caps: ["vision", "llm", "delegate"])
        └── Agent A delegates to Sub-Agent B (caps: ["vision"])
            └── Sub-Agent B operates with restricted capabilities

Properties:
- Fully offline-verifiable (no network calls)
- Capabilities can only narrow, not expand
- Expiration must be <= parent's expiration
- Parent can revoke delegated tokens
- Chain verification: SubToken → AgentToken → MeshToken
"""

import json
import time
from dataclasses import dataclass, field
from typing import List, Optional

from .crypto import verify_signature_b64
from .tokens import MeshToken


@dataclass
class DelegatedToken:
    """
    A token issued by an agent (not a mesh founder).
    
    Contains the full chain back to the original mesh token for verification.
    """
    version: int = 1
    
    # This token's info
    device_id: str = ""
    device_public_key: str = ""
    device_name: str = ""
    capabilities: List[str] = field(default_factory=list)
    tier: str = "iot"
    
    issued_at: int = 0
    expires_at: int = 0
    
    # Parent agent who issued this
    parent_device_id: str = ""
    parent_public_key: str = ""
    parent_signature: str = ""  # Parent signs this token
    
    # The mesh this belongs to
    mesh_id: str = ""
    mesh_name: str = ""
    
    # Chain of parent tokens (for full verification)
    # chain[0] = immediate parent, chain[-1] = mesh founder token
    delegation_chain: List[dict] = field(default_factory=list)
    
    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at
    
    @property
    def time_remaining(self) -> int:
        """Seconds until expiration."""
        return max(0, self.expires_at - int(time.time()))
    
    @property
    def chain_depth(self) -> int:
        """How many delegation hops? 1 = direct from mesh token."""
        return len(self.delegation_chain)
    
    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "device": {
                "device_id": self.device_id,
                "public_key": self.device_public_key,
                "name": self.device_name
            },
            "authorization": {
                "capabilities": self.capabilities,
                "tier": self.tier
            },
            "validity": {
                "issued_at": self.issued_at,
                "expires_at": self.expires_at
            },
            "delegation": {
                "parent_device_id": self.parent_device_id,
                "parent_public_key": self.parent_public_key,
                "parent_signature": self.parent_signature
            },
            "mesh": {
                "mesh_id": self.mesh_id,
                "mesh_name": self.mesh_name
            },
            "delegation_chain": self.delegation_chain
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "DelegatedToken":
        return cls(
            version=data["version"],
            device_id=data["device"]["device_id"],
            device_public_key=data["device"]["public_key"],
            device_name=data["device"]["name"],
            capabilities=data["authorization"]["capabilities"],
            tier=data["authorization"]["tier"],
            issued_at=data["validity"]["issued_at"],
            expires_at=data["validity"]["expires_at"],
            parent_device_id=data["delegation"]["parent_device_id"],
            parent_public_key=data["delegation"]["parent_public_key"],
            parent_signature=data["delegation"]["parent_signature"],
            mesh_id=data["mesh"]["mesh_id"],
            mesh_name=data["mesh"]["mesh_name"],
            delegation_chain=data["delegation_chain"]
        )


class DelegationIssuer:
    """
    Allows an agent with a mesh token OR delegated token to delegate permissions to sub-agents.
    
    Supports multi-hop delegation: MeshToken → DelegatedToken → DelegatedToken → ...
    """
    
    def __init__(self, parent_token, parent_key_pair):
        """
        Create a delegation issuer.
        
        Args:
            parent_token: The agent's MeshToken or DelegatedToken (must have "delegate" capability)
            parent_key_pair: The agent's KeyPair for signing
        """
        self.parent_token = parent_token
        self.parent_key_pair = parent_key_pair
        
        # Verify parent has delegation capability
        if "delegate" not in parent_token.capabilities:
            raise ValueError(
                "Parent token does not have 'delegate' capability. "
                "Cannot issue delegated tokens."
            )
    
    def delegate(
        self,
        sub_agent_name: str,
        sub_agent_public_key: str,
        capabilities: List[str],
        validity_hours: int = 1,  # Shorter-lived by default
        tier: Optional[str] = None
    ) -> DelegatedToken:
        """
        Delegate a subset of this agent's capabilities to a sub-agent.
        
        Args:
            sub_agent_name: Name of the sub-agent
            sub_agent_public_key: Sub-agent's public key
            capabilities: Capabilities to grant (must be subset of parent's)
            validity_hours: Token lifetime (must be <= parent's remaining time)
            tier: Device tier (defaults to parent's tier)
        
        Returns:
            DelegatedToken for the sub-agent
        
        Raises:
            ValueError: If capabilities expand beyond parent's or expiration too long
        """
        # Verify capabilities are a subset
        parent_caps = set(self.parent_token.capabilities)
        requested_caps = set(capabilities)
        
        if not requested_caps.issubset(parent_caps):
            invalid = requested_caps - parent_caps
            raise ValueError(
                f"Cannot grant capabilities parent doesn't have: {invalid}"
            )
        
        # Verify expiration doesn't exceed parent's
        now = int(time.time())
        parent_remaining = self.parent_token.time_remaining
        
        if validity_hours * 3600 > parent_remaining:
            raise ValueError(
                f"Cannot issue token that outlives parent. "
                f"Parent expires in {parent_remaining}s, requested {validity_hours}h"
            )
        
        expires_at = now + (validity_hours * 3600)
        
        # Build sub-agent token
        token_data = {
            "version": 1,
            "device": {
                "device_id": f"agent-{sub_agent_name}",  # Synthetic ID for agents
                "public_key": sub_agent_public_key,
                "name": sub_agent_name
            },
            "authorization": {
                "capabilities": capabilities,
                "tier": tier or self.parent_token.tier
            },
            "validity": {
                "issued_at": now,
                "expires_at": expires_at
            },
            "delegation": {
                "parent_device_id": self.parent_token.device_id,
                "parent_public_key": self.parent_token.device_public_key
            },
            "mesh": {
                "mesh_id": self.parent_token.mesh_id,
                "mesh_name": self.parent_token.mesh_name
            }
        }
        
        # Sign with parent's key
        message = json.dumps(token_data, sort_keys=True).encode()
        signature = self.parent_key_pair.sign_b64(message)
        
        # Build delegation chain
        # If parent is a DelegatedToken, extend its chain
        # If parent is a MeshToken, start the chain
        if isinstance(self.parent_token, DelegatedToken):
            # Multi-hop: include parent's entire chain
            delegation_chain = [self.parent_token.to_dict()] + self.parent_token.delegation_chain
        else:
            # First hop: start with mesh token
            delegation_chain = [self.parent_token.to_dict()]
        
        return DelegatedToken(
            version=1,
            device_id=f"agent-{sub_agent_name}",
            device_public_key=sub_agent_public_key,
            device_name=sub_agent_name,
            capabilities=capabilities,
            tier=tier or self.parent_token.tier,
            issued_at=now,
            expires_at=expires_at,
            parent_device_id=self.parent_token.device_id,
            parent_public_key=self.parent_token.device_public_key,
            parent_signature=signature,
            mesh_id=self.parent_token.mesh_id,
            mesh_name=self.parent_token.mesh_name,
            delegation_chain=delegation_chain
        )


class DelegationVerifier:
    """
    Verifies delegated tokens by walking the chain back to the mesh.
    """
    
    def __init__(self, mesh_public_key: str, founding_members: List[dict]):
        """
        Create a delegation verifier.
        
        Args:
            mesh_public_key: The mesh's master public key
            founding_members: List of founding member info
        """
        self.mesh_public_key = mesh_public_key
        self.founding_members = {
            f["node_id"]: f for f in founding_members
        }
    
    def verify(self, token: DelegatedToken) -> tuple:
        """
        Verify a delegated token by checking the full chain.
        
        Returns:
            (is_valid, reason) tuple
        """
        # Check version
        if token.version != 1:
            return False, f"Unknown token version: {token.version}"
        
        # Check expiration
        if token.is_expired:
            return False, "Token has expired"
        
        # Must have at least one parent in the chain
        if not token.delegation_chain:
            return False, "No delegation chain provided"
        
        # Verify the parent signature on this token
        token_data = {
            "version": token.version,
            "device": {
                "device_id": token.device_id,
                "public_key": token.device_public_key,
                "name": token.device_name
            },
            "authorization": {
                "capabilities": token.capabilities,
                "tier": token.tier
            },
            "validity": {
                "issued_at": token.issued_at,
                "expires_at": token.expires_at
            },
            "delegation": {
                "parent_device_id": token.parent_device_id,
                "parent_public_key": token.parent_public_key
            },
            "mesh": {
                "mesh_id": token.mesh_id,
                "mesh_name": token.mesh_name
            }
        }
        
        message = json.dumps(token_data, sort_keys=True).encode()
        
        if not verify_signature_b64(
            token.parent_public_key,
            message,
            token.parent_signature
        ):
            return False, "Invalid parent signature"
        
        # Verify the delegation chain
        # Start with the immediate parent
        parent = token.delegation_chain[0]
        
        # Verify parent's expiration >= this token's expiration
        if parent["validity"]["expires_at"] < token.expires_at:
            return False, "Token expires after parent token"
        
        # Verify capabilities are a subset of parent's
        parent_caps = set(parent["authorization"]["capabilities"])
        token_caps = set(token.capabilities)
        
        if not token_caps.issubset(parent_caps):
            invalid = token_caps - parent_caps
            return False, f"Token has capabilities parent doesn't have: {invalid}"
        
        # Verify parent has "delegate" capability
        if "delegate" not in parent_caps:
            return False, "Parent token cannot delegate (missing 'delegate' capability)"
        
        # Now verify the parent token (could be MeshToken or another DelegatedToken)
        # Check if parent is a DelegatedToken (has "delegation" field) or MeshToken
        if "delegation" in parent:
            # Parent is a DelegatedToken - recursively verify it
            parent_delegated = DelegatedToken.from_dict(parent)
            parent_valid, parent_reason = self.verify(parent_delegated)
            if not parent_valid:
                return False, f"Parent delegation invalid: {parent_reason}"
            return True, "Delegated token is valid"
        
        # Parent is a MeshToken - verify the mesh signature
        parent_token = MeshToken.from_dict(parent)
        
        # Check parent's mesh public key
        if parent_token.mesh_public_key != self.mesh_public_key:
            return False, "Chain is for a different mesh"
        
        # Check parent's issuer is a known founder
        if parent_token.issuer_node_id not in self.founding_members:
            return False, f"Unknown issuer in chain: {parent_token.issuer_node_id}"
        
        founder = self.founding_members[parent_token.issuer_node_id]
        if founder["public_key"] != parent_token.issuer_public_key:
            return False, "Issuer public key mismatch in chain"
        
        # Verify parent's issuer signature
        parent_data = {
            "version": parent_token.version,
            "mesh_id": parent_token.mesh_id,
            "mesh_name": parent_token.mesh_name,
            "device": {
                "device_id": parent_token.device_id,
                "public_key": parent_token.device_public_key,
                "name": parent_token.device_name,
                "hardware_hash": parent_token.hardware_hash
            },
            "authorization": {
                "capabilities": parent_token.capabilities,
                "tier": parent_token.tier
            },
            "validity": {
                "issued_at": parent_token.issued_at,
                "expires_at": parent_token.expires_at
            },
            "issuer": {
                "node_id": parent_token.issuer_node_id,
                "public_key": parent_token.issuer_public_key
            },
            "mesh_public_key": parent_token.mesh_public_key
        }
        
        parent_message = json.dumps(parent_data, sort_keys=True).encode()
        
        if not verify_signature_b64(
            parent_token.issuer_public_key,
            parent_message,
            parent_token.issuer_signature
        ):
            return False, "Invalid issuer signature in chain"
        
        return True, "Delegated token is valid"
    
    def quick_verify(self, token: DelegatedToken) -> bool:
        """Quick verification (returns bool only)."""
        is_valid, _ = self.verify(token)
        return is_valid


def can_delegate(token: MeshToken) -> bool:
    """
    Check if a mesh token has delegation capability.
    
    Args:
        token: A MeshToken
    
    Returns:
        True if token can delegate permissions to sub-agents
    """
    return "delegate" in token.capabilities
