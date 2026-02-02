"""
Federation: Hierarchical mesh trust for multi-organization deployments.

Example hierarchy:
  DoD (root mesh)
    └── Air Force (inherits from DoD)
        └── Squadron 42 (inherits from Air Force)
            └── Field Unit Alpha (can disconnect and operate independently)

Key properties:
- Child meshes inherit trust from parent
- Children can operate fully disconnected
- Trust flows down, not up (parent doesn't automatically trust child members)
- Each level can add restrictions, not expand permissions
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from .crypto import KeyPair, verify_signature_b64
from .mesh import MeshIdentity


@dataclass
class FederationLink:
    """
    A cryptographic link between a child mesh and its parent.
    
    This proves that the parent mesh authorized the child to exist.
    """
    child_mesh_id: str
    child_mesh_name: str
    child_public_key: str
    
    parent_mesh_id: str
    parent_mesh_name: str
    parent_public_key: str
    
    # What the child is allowed to do
    allowed_capabilities: List[str]  # Empty = all parent caps inherited
    max_tier: str  # Highest tier child can grant ("compute", "edge", "iot")
    can_create_children: bool  # Can this child create grandchildren?
    
    # Validity
    created_at: int
    expires_at: int  # 0 = never expires
    
    # Parent's signature over this link
    parent_signature: str
    
    def to_dict(self) -> dict:
        return {
            "child": {
                "mesh_id": self.child_mesh_id,
                "mesh_name": self.child_mesh_name,
                "public_key": self.child_public_key
            },
            "parent": {
                "mesh_id": self.parent_mesh_id,
                "mesh_name": self.parent_mesh_name,
                "public_key": self.parent_public_key
            },
            "permissions": {
                "allowed_capabilities": self.allowed_capabilities,
                "max_tier": self.max_tier,
                "can_create_children": self.can_create_children
            },
            "validity": {
                "created_at": self.created_at,
                "expires_at": self.expires_at
            },
            "parent_signature": self.parent_signature
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "FederationLink":
        return cls(
            child_mesh_id=data["child"]["mesh_id"],
            child_mesh_name=data["child"]["mesh_name"],
            child_public_key=data["child"]["public_key"],
            parent_mesh_id=data["parent"]["mesh_id"],
            parent_mesh_name=data["parent"]["mesh_name"],
            parent_public_key=data["parent"]["public_key"],
            allowed_capabilities=data["permissions"]["allowed_capabilities"],
            max_tier=data["permissions"]["max_tier"],
            can_create_children=data["permissions"]["can_create_children"],
            created_at=data["validity"]["created_at"],
            expires_at=data["validity"]["expires_at"],
            parent_signature=data["parent_signature"]
        )
    
    @property
    def is_expired(self) -> bool:
        if self.expires_at == 0:
            return False
        return time.time() > self.expires_at


class FederatedMesh:
    """
    A mesh that participates in a federation hierarchy.
    
    Can be:
    - A root mesh (no parent)
    - A child mesh (has parent, may have children)
    - A leaf mesh (has parent, no children)
    """
    
    def __init__(self, mesh: MeshIdentity):
        self.mesh = mesh
        self.parent_link: Optional[FederationLink] = None
        self.child_links: Dict[str, FederationLink] = {}
        
        # Cache of known meshes in our federation tree
        self._known_meshes: Dict[str, dict] = {}
    
    @property
    def is_root(self) -> bool:
        """Is this the root of the federation tree?"""
        return self.parent_link is None
    
    @property
    def federation_path(self) -> List[str]:
        """
        Path from root to this mesh.
        
        Example: ["dod", "air-force", "squadron-42"]
        """
        if self.parent_link is None:
            return [self.mesh.mesh_id]
        # In practice, we'd walk up the tree
        return [self.parent_link.parent_mesh_id, self.mesh.mesh_id]
    
    def create_child_mesh(
        self,
        child_name: str,
        allowed_capabilities: Optional[List[str]] = None,
        max_tier: str = "compute",
        can_create_children: bool = True,
        expires_in_days: int = 0  # 0 = never
    ) -> tuple:
        """
        Create a new child mesh that inherits from this one.
        
        Returns:
            (child_mesh, federation_link)
        """
        if not self.mesh.can_issue_certificates():
            raise RuntimeError("This mesh cannot create children (not a founder)")
        
        # Create the child mesh
        child_mesh = MeshIdentity.create(
            name=child_name,
            threshold=1,  # Child decides its own governance
            total_shares=1
        )
        
        # Determine timestamps (must be same for signing and link)
        created_at = int(time.time())
        expires_at = 0
        if expires_in_days > 0:
            expires_at = created_at + (expires_in_days * 86400)
        
        # Normalize capabilities
        caps = allowed_capabilities or []
        
        # Use the founding node's keypair for signing
        # (parent_public_key is the signer's public key, not mesh master)
        signer_public_key = self.mesh._local_key_pair.public_key_b64()
        
        # Build the link data (without signature)
        link_data = {
            "child": {
                "mesh_id": child_mesh.mesh_id,
                "mesh_name": child_mesh.name,
                "public_key": child_mesh.master_public_key
            },
            "parent": {
                "mesh_id": self.mesh.mesh_id,
                "mesh_name": self.mesh.name,
                "public_key": signer_public_key
            },
            "permissions": {
                "allowed_capabilities": caps,
                "max_tier": max_tier,
                "can_create_children": can_create_children
            },
            "validity": {
                "created_at": created_at,
                "expires_at": expires_at
            }
        }
        
        # Sign with parent founder's key
        message = json.dumps(link_data, sort_keys=True).encode()
        signature = self.mesh._local_key_pair.sign_b64(message)
        
        link = FederationLink(
            child_mesh_id=child_mesh.mesh_id,
            child_mesh_name=child_mesh.name,
            child_public_key=child_mesh.master_public_key,
            parent_mesh_id=self.mesh.mesh_id,
            parent_mesh_name=self.mesh.name,
            parent_public_key=signer_public_key,
            allowed_capabilities=caps,
            max_tier=max_tier,
            can_create_children=can_create_children,
            created_at=created_at,
            expires_at=expires_at,
            parent_signature=signature
        )
        
        # Track the child
        self.child_links[child_mesh.mesh_id] = link
        
        return child_mesh, link
    
    def accept_parent_link(self, link: FederationLink) -> bool:
        """
        Accept a federation link from a parent mesh.
        
        This makes us a child of that parent.
        """
        # Verify the link is for us
        if link.child_mesh_id != self.mesh.mesh_id:
            return False
        
        # Verify parent's signature
        link_data = {
            "child": {
                "mesh_id": link.child_mesh_id,
                "mesh_name": link.child_mesh_name,
                "public_key": link.child_public_key
            },
            "parent": {
                "mesh_id": link.parent_mesh_id,
                "mesh_name": link.parent_mesh_name,
                "public_key": link.parent_public_key
            },
            "permissions": {
                "allowed_capabilities": link.allowed_capabilities,
                "max_tier": link.max_tier,
                "can_create_children": link.can_create_children
            },
            "validity": {
                "created_at": link.created_at,
                "expires_at": link.expires_at
            }
        }
        
        message = json.dumps(link_data, sort_keys=True).encode()
        
        if not verify_signature_b64(
            link.parent_public_key,
            message,
            link.parent_signature
        ):
            return False
        
        self.parent_link = link
        return True
    
    def verify_federation_chain(self, chain: List[FederationLink]) -> tuple:
        """
        Verify a chain of federation links from a remote mesh back to a known root.
        
        Args:
            chain: List of FederationLink from target mesh up to known root
        
        Returns:
            (is_valid, effective_permissions)
        """
        if not chain:
            return False, None
        
        # Start with full permissions
        effective_caps = None  # None = all allowed
        effective_tier = "compute"
        
        # Walk the chain from root to leaf
        for i, link in enumerate(chain):
            # Verify each link's signature
            link_data = {
                "child": {
                    "mesh_id": link.child_mesh_id,
                    "mesh_name": link.child_mesh_name,
                    "public_key": link.child_public_key
                },
                "parent": {
                    "mesh_id": link.parent_mesh_id,
                    "mesh_name": link.parent_mesh_name,
                    "public_key": link.parent_public_key
                },
                "permissions": {
                    "allowed_capabilities": link.allowed_capabilities,
                    "max_tier": link.max_tier,
                    "can_create_children": link.can_create_children
                },
                "validity": {
                    "created_at": link.created_at,
                    "expires_at": link.expires_at
                }
            }
            
            message = json.dumps(link_data, sort_keys=True).encode()
            
            if not verify_signature_b64(
                link.parent_public_key,
                message,
                link.parent_signature
            ):
                return False, None
            
            # Check expiration
            if link.is_expired:
                return False, None
            
            # Intersect capabilities (can only narrow, not expand)
            if link.allowed_capabilities:
                if effective_caps is None:
                    effective_caps = set(link.allowed_capabilities)
                else:
                    effective_caps &= set(link.allowed_capabilities)
            
            # Tier can only go down
            tier_order = ["iot", "edge", "compute"]
            link_tier_idx = tier_order.index(link.max_tier)
            current_tier_idx = tier_order.index(effective_tier)
            effective_tier = tier_order[min(link_tier_idx, current_tier_idx)]
            
            # Verify chain continuity
            if i > 0:
                prev_link = chain[i - 1]
                if prev_link.child_mesh_id != link.parent_mesh_id:
                    return False, None
        
        return True, {
            "capabilities": list(effective_caps) if effective_caps else None,
            "max_tier": effective_tier
        }
    
    def can_operate_disconnected(self) -> bool:
        """
        Can this mesh operate without connection to parent?
        
        Yes - the federation link is cached locally and tokens
        can be verified with local crypto. The only limitation
        is that new cross-mesh trust relationships can't be
        established while disconnected.
        """
        return True  # Always yes - that's the point!
    
    def get_disconnected_capabilities(self) -> dict:
        """
        What can we do while disconnected from parent?
        """
        return {
            "issue_local_tokens": True,
            "verify_local_tokens": True,
            "verify_parent_tokens": True,  # If we have parent's public key
            "issue_cross_mesh_tokens": False,  # Need parent online
            "create_child_mesh": self.parent_link.can_create_children if self.parent_link else True,
            "revoke_local_devices": True,
            "propagate_revocations": False  # Need network
        }


def create_federation_root(name: str, threshold: int = 2, shares: int = 3) -> FederatedMesh:
    """
    Create a root federation mesh (e.g., DoD, Corporate HQ).
    
    This is the ultimate trust anchor for the federation tree.
    """
    mesh = MeshIdentity.create(
        name=name,
        threshold=threshold,
        total_shares=shares
    )
    return FederatedMesh(mesh)
