"""
Mesh Identity: The distributed trust root for the Atmosphere.

A mesh is defined by its master public key. The master private key
is split across founding nodes using Shamir Secret Sharing.
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .crypto import (
    KeyPair,
    split_secret,
    combine_shares,
    get_hardware_fingerprint,
)


@dataclass
class FoundingMember:
    """A founding member of the mesh who holds a key share."""
    node_id: str
    public_key: str
    share_index: int
    capabilities: List[str]
    hardware_hash: str
    joined_at: int


@dataclass
class MeshIdentity:
    """
    The identity of a mesh network.
    
    Holds the master public key and metadata about the mesh.
    The master private key is never stored whole - only shares.
    """
    mesh_id: str  # Hash of master public key
    name: str
    master_public_key: str
    threshold: int  # k shares needed
    total_shares: int  # n total shares
    founding_members: List[FoundingMember] = field(default_factory=list)
    created_at: int = field(default_factory=lambda: int(time.time()))
    version: int = 1
    
    # Local state (not serialized to mesh)
    _local_share: Optional[Tuple[int, bytes]] = field(default=None, repr=False)
    _local_key_pair: Optional[KeyPair] = field(default=None, repr=False)
    
    @classmethod
    def create(
        cls,
        name: str,
        threshold: int = 2,
        total_shares: int = 3,
        founding_capabilities: Optional[List[str]] = None
    ) -> "MeshIdentity":
        """
        Create a new mesh with threshold signing.
        
        This generates the master keypair and splits the private key
        into shares. The creator becomes the first founding member.
        
        Args:
            name: Human-readable mesh name
            threshold: Minimum shares needed to issue certificates
            total_shares: Total number of shares to create
            founding_capabilities: Capabilities of this founding node
        
        Returns:
            MeshIdentity with this node as first founder
        """
        if threshold > total_shares:
            raise ValueError("Threshold cannot exceed total shares")
        
        # Generate master keypair
        master_keypair = KeyPair.generate()
        master_public_key = master_keypair.public_key_b64()
        mesh_id = hashlib.sha256(master_keypair.public_bytes()).hexdigest()[:16]
        
        # Split the private key
        shares = split_secret(
            master_keypair.private_bytes(),
            threshold,
            total_shares
        )
        
        # Generate node keypair (separate from master)
        node_keypair = KeyPair.generate()
        
        # Create mesh identity
        mesh = cls(
            mesh_id=mesh_id,
            name=name,
            master_public_key=master_public_key,
            threshold=threshold,
            total_shares=total_shares,
        )
        
        # Add self as first founding member with first share
        founder = FoundingMember(
            node_id=node_keypair.key_id(),
            public_key=node_keypair.public_key_b64(),
            share_index=shares[0][0],
            capabilities=founding_capabilities or ["mesh-admin"],
            hardware_hash=get_hardware_fingerprint(),
            joined_at=int(time.time())
        )
        mesh.founding_members.append(founder)
        
        # Store local state
        mesh._local_share = shares[0]
        mesh._local_key_pair = node_keypair
        
        # Return remaining shares for distribution
        mesh._pending_shares = shares[1:]
        
        return mesh
    
    def get_pending_shares(self) -> List[Tuple[int, bytes]]:
        """Get shares that need to be distributed to other founders."""
        return getattr(self, '_pending_shares', [])
    
    def export_share_for_founder(self, share_index: int) -> dict:
        """Export a share for a new founding member."""
        pending = self.get_pending_shares()
        for idx, share_bytes in pending:
            if idx == share_index:
                return {
                    "mesh_id": self.mesh_id,
                    "mesh_name": self.name,
                    "master_public_key": self.master_public_key,
                    "share_index": idx,
                    "share_data": share_bytes.hex(),
                    "threshold": self.threshold,
                    "total_shares": self.total_shares
                }
        raise ValueError(f"Share {share_index} not found or already distributed")
    
    def import_share(self, share_data: dict, node_keypair: KeyPair) -> None:
        """Import a share to become a founding member."""
        share_index = share_data["share_index"]
        share_bytes = bytes.fromhex(share_data["share_data"])
        
        self._local_share = (share_index, share_bytes)
        self._local_key_pair = node_keypair
        
        # Add self as founding member
        founder = FoundingMember(
            node_id=node_keypair.key_id(),
            public_key=node_keypair.public_key_b64(),
            share_index=share_index,
            capabilities=["mesh-admin"],
            hardware_hash=get_hardware_fingerprint(),
            joined_at=int(time.time())
        )
        self.founding_members.append(founder)
    
    def can_issue_certificates(self) -> bool:
        """Check if this node can issue certificates (is a founder with share)."""
        return self._local_share is not None and self._local_key_pair is not None
    
    def to_dict(self) -> dict:
        """Serialize mesh identity (without private data)."""
        return {
            "version": self.version,
            "mesh_id": self.mesh_id,
            "name": self.name,
            "master_public_key": self.master_public_key,
            "threshold": self.threshold,
            "total_shares": self.total_shares,
            "founding_members": [
                {
                    "node_id": f.node_id,
                    "public_key": f.public_key,
                    "share_index": f.share_index,
                    "capabilities": f.capabilities,
                    "hardware_hash": f.hardware_hash,
                    "joined_at": f.joined_at
                }
                for f in self.founding_members
            ],
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "MeshIdentity":
        """Load mesh identity from dict."""
        mesh = cls(
            mesh_id=data["mesh_id"],
            name=data["name"],
            master_public_key=data["master_public_key"],
            threshold=data["threshold"],
            total_shares=data["total_shares"],
            created_at=data.get("created_at", int(time.time())),
            version=data.get("version", 1)
        )
        
        for f_data in data.get("founding_members", []):
            founder = FoundingMember(
                node_id=f_data["node_id"],
                public_key=f_data["public_key"],
                share_index=f_data["share_index"],
                capabilities=f_data["capabilities"],
                hardware_hash=f_data["hardware_hash"],
                joined_at=f_data["joined_at"]
            )
            mesh.founding_members.append(founder)
        
        return mesh
    
    def save(self, path: Path) -> None:
        """Save mesh identity to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        # Save local secrets separately (encrypted in production)
        if self._local_share:
            secrets_path = path.with_suffix('.secrets')
            secrets_data = {
                "share_index": self._local_share[0],
                "share_data": self._local_share[1].hex(),
                "node_private_key": self._local_key_pair.private_bytes().hex() if self._local_key_pair else None
            }
            with open(secrets_path, 'w') as f:
                json.dump(secrets_data, f)
            # In production: encrypt with hardware key
    
    @classmethod
    def load(cls, path: Path) -> "MeshIdentity":
        """Load mesh identity from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        mesh = cls.from_dict(data)
        
        # Try to load local secrets
        secrets_path = path.with_suffix('.secrets')
        if secrets_path.exists():
            with open(secrets_path, 'r') as f:
                secrets_data = json.load(f)
            
            mesh._local_share = (
                secrets_data["share_index"],
                bytes.fromhex(secrets_data["share_data"])
            )
            
            if secrets_data.get("node_private_key"):
                mesh._local_key_pair = KeyPair.from_private_bytes(
                    bytes.fromhex(secrets_data["node_private_key"])
                )
        
        return mesh
