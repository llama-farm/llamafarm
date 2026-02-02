"""
Node identity and mesh membership.

A node is a single device in the Atmosphere mesh.
A mesh is a collection of nodes that trust each other.
"""

import hashlib
import json
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..auth.identity import KeyPair, get_hardware_fingerprint


@dataclass
class FoundingMember:
    """A founding member of the mesh who holds a key share."""
    node_id: str
    public_key: str
    share_index: int
    capabilities: List[str]
    hardware_hash: str
    joined_at: int


# Large prime for Shamir Secret Sharing
PRIME = 2**255 - 19


def _mod_inverse(a: int, p: int) -> int:
    """Modular multiplicative inverse."""
    def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y
    
    _, x, _ = extended_gcd(a % p, p)
    return (x % p + p) % p


def split_secret(secret: bytes, threshold: int, num_shares: int) -> List[Tuple[int, bytes]]:
    """Split a secret using Shamir's Secret Sharing."""
    if threshold > num_shares:
        raise ValueError("Threshold cannot exceed number of shares")
    
    secret_int = int.from_bytes(secret.ljust(32, b'\x00')[:32], 'big')
    
    coefficients = [secret_int]
    for _ in range(threshold - 1):
        coefficients.append(secrets.randbelow(PRIME))
    
    shares = []
    for x in range(1, num_shares + 1):
        y = 0
        for i, coeff in enumerate(coefficients):
            y = (y + coeff * pow(x, i, PRIME)) % PRIME
        share_bytes = y.to_bytes(32, 'big')
        shares.append((x, share_bytes))
    
    return shares


def combine_shares(shares: List[Tuple[int, bytes]]) -> bytes:
    """Reconstruct a secret from shares using Lagrange interpolation."""
    if len(shares) < 2:
        raise ValueError("Need at least 2 shares")
    
    points = [(x, int.from_bytes(share, 'big')) for x, share in shares]
    
    secret = 0
    for i, (xi, yi) in enumerate(points):
        numerator = 1
        denominator = 1
        
        for j, (xj, _) in enumerate(points):
            if i != j:
                numerator = (numerator * (-xj)) % PRIME
                denominator = (denominator * (xi - xj)) % PRIME
        
        lagrange = (yi * numerator * _mod_inverse(denominator, PRIME)) % PRIME
        secret = (secret + lagrange) % PRIME
    
    return secret.to_bytes(32, 'big')


@dataclass
class NodeIdentity:
    """Identity of a single node in the mesh."""
    keypair: KeyPair
    name: str
    hardware_hash: str
    created_at: int
    
    @property
    def node_id(self) -> str:
        return self.keypair.key_id()
    
    @property
    def public_key(self) -> str:
        return self.keypair.public_key_b64()
    
    def sign(self, message: bytes) -> str:
        return self.keypair.sign_b64(message)
    
    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "public_key": self.public_key,
            "name": self.name,
            "hardware_hash": self.hardware_hash,
            "created_at": self.created_at
        }
    
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "private_key": self.keypair.private_bytes().hex(),
            "name": self.name,
            "hardware_hash": self.hardware_hash,
            "created_at": self.created_at
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        path.chmod(0o600)
    
    @classmethod
    def load(cls, path: Path) -> "NodeIdentity":
        with open(path, 'r') as f:
            data = json.load(f)
        keypair = KeyPair.from_private_bytes(bytes.fromhex(data["private_key"]))
        return cls(
            keypair=keypair,
            name=data["name"],
            hardware_hash=data["hardware_hash"],
            created_at=data["created_at"]
        )
    
    @classmethod
    def generate(cls, name: str) -> "NodeIdentity":
        return cls(
            keypair=KeyPair.generate(),
            name=name,
            hardware_hash=get_hardware_fingerprint(),
            created_at=int(time.time())
        )


@dataclass
class MeshIdentity:
    """
    The identity of a mesh network.
    
    The master key is split using Shamir Secret Sharing.
    """
    mesh_id: str
    name: str
    master_public_key: str
    threshold: int
    total_shares: int
    founding_members: List[FoundingMember] = field(default_factory=list)
    created_at: int = field(default_factory=lambda: int(time.time()))
    version: int = 1
    
    # Local state (not serialized)
    _local_share: Optional[Tuple[int, bytes]] = field(default=None, repr=False)
    _local_key_pair: Optional[KeyPair] = field(default=None, repr=False)
    _pending_shares: List[Tuple[int, bytes]] = field(default_factory=list, repr=False)
    
    @classmethod
    def create(
        cls,
        name: str,
        threshold: int = 2,
        total_shares: int = 3,
        founding_capabilities: Optional[List[str]] = None
    ) -> "MeshIdentity":
        """Create a new mesh with threshold signing."""
        if threshold > total_shares:
            raise ValueError("Threshold cannot exceed total shares")
        
        master_keypair = KeyPair.generate()
        master_public_key = master_keypair.public_key_b64()
        mesh_id = hashlib.sha256(master_keypair.public_bytes()).hexdigest()[:16]
        
        shares = split_secret(
            master_keypair.private_bytes(),
            threshold,
            total_shares
        )
        
        node_keypair = KeyPair.generate()
        
        mesh = cls(
            mesh_id=mesh_id,
            name=name,
            master_public_key=master_public_key,
            threshold=threshold,
            total_shares=total_shares,
        )
        
        founder = FoundingMember(
            node_id=node_keypair.key_id(),
            public_key=node_keypair.public_key_b64(),
            share_index=shares[0][0],
            capabilities=founding_capabilities or ["mesh-admin"],
            hardware_hash=get_hardware_fingerprint(),
            joined_at=int(time.time())
        )
        mesh.founding_members.append(founder)
        
        mesh._local_share = shares[0]
        mesh._local_key_pair = node_keypair
        mesh._pending_shares = shares[1:]
        
        return mesh
    
    def can_issue_certificates(self) -> bool:
        return self._local_share is not None and self._local_key_pair is not None
    
    def get_pending_shares(self) -> List[Tuple[int, bytes]]:
        return self._pending_shares
    
    def to_dict(self) -> dict:
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
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        if self._local_share:
            secrets_path = path.with_suffix('.secrets')
            secrets_data = {
                "share_index": self._local_share[0],
                "share_data": self._local_share[1].hex(),
                "node_private_key": self._local_key_pair.private_bytes().hex() if self._local_key_pair else None
            }
            with open(secrets_path, 'w') as f:
                json.dump(secrets_data, f)
            secrets_path.chmod(0o600)
    
    @classmethod
    def load(cls, path: Path) -> "MeshIdentity":
        with open(path, 'r') as f:
            data = json.load(f)
        
        mesh = cls.from_dict(data)
        
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


class Node:
    """
    A node in the Atmosphere mesh.
    
    Combines node identity, mesh membership, and runtime state.
    """
    
    def __init__(
        self,
        identity: NodeIdentity,
        mesh: Optional[MeshIdentity] = None
    ):
        self.identity = identity
        self.mesh = mesh
        self.capabilities: List[str] = []
        self.peers: Dict[str, dict] = {}
        self._running = False
    
    @property
    def node_id(self) -> str:
        return self.identity.node_id
    
    @property
    def name(self) -> str:
        return self.identity.name
    
    @property
    def is_mesh_member(self) -> bool:
        return self.mesh is not None
    
    @property
    def is_founder(self) -> bool:
        return self.mesh is not None and self.mesh.can_issue_certificates()
    
    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "capabilities": self.capabilities,
            "mesh_id": self.mesh.mesh_id if self.mesh else None,
            "is_founder": self.is_founder,
            "peer_count": len(self.peers)
        }
    
    @classmethod
    def create(cls, name: str) -> "Node":
        """Create a new standalone node."""
        identity = NodeIdentity.generate(name)
        return cls(identity=identity)
    
    @classmethod
    def create_with_mesh(
        cls,
        node_name: str,
        mesh_name: str,
        threshold: int = 2,
        total_shares: int = 3
    ) -> "Node":
        """Create a new node that's also a mesh founder."""
        mesh = MeshIdentity.create(
            name=mesh_name,
            threshold=threshold,
            total_shares=total_shares
        )
        
        # Use the mesh's generated node identity
        identity = NodeIdentity(
            keypair=mesh._local_key_pair,
            name=node_name,
            hardware_hash=get_hardware_fingerprint(),
            created_at=int(time.time())
        )
        
        return cls(identity=identity, mesh=mesh)
