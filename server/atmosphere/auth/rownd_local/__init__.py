"""
Rownd Local: Zero-Trust Mesh Authentication

A local-first authentication system for distributed AI meshes.

Key Features:
- Offline-capable: All verification uses local Ed25519 signatures
- Threshold signing: No single point of failure (k-of-n founders)
- Hardware-bound: Keys tied to physical devices
- Federated: Meshes can inherit trust from parent meshes
- Agentic: Agents can delegate capabilities to sub-agents
- Gossip revocation: Revocations spread without central server
"""

__version__ = "0.1.0"

from .crypto import KeyPair, split_secret, combine_shares, verify_signature_b64
from .mesh import MeshIdentity, FoundingMember
from .device import DeviceIdentity, detect_capabilities, detect_tier
from .tokens import MeshToken, TokenIssuer, TokenVerifier
from .service import RowndLocalService
from .federation import FederatedMesh, FederationLink, create_federation_root
from .delegation import DelegatedToken, DelegationIssuer, DelegationVerifier, can_delegate
from .renewal import TokenRenewer, RenewalPolicy, RenewalRecord, RenewalStrategy
from .revocation import RevocationLog, RevocationEntry, GossipProtocol, RevocationType

__all__ = [
    # Crypto primitives
    "KeyPair",
    "split_secret",
    "combine_shares",
    "verify_signature_b64",
    
    # Mesh identity
    "MeshIdentity",
    "FoundingMember",
    
    # Device identity
    "DeviceIdentity",
    "detect_capabilities",
    "detect_tier",
    
    # Tokens
    "MeshToken",
    "TokenIssuer",
    "TokenVerifier",
    
    # Main service
    "RowndLocalService",
    
    # Federation (hierarchical trust)
    "FederatedMesh",
    "FederationLink",
    "create_federation_root",
    
    # Agentic delegation
    "DelegatedToken",
    "DelegationIssuer",
    "DelegationVerifier",
    "can_delegate",
    
    # Token renewal
    "TokenRenewer",
    "RenewalPolicy",
    "RenewalRecord",
    "RenewalStrategy",
    
    # Revocation
    "RevocationLog",
    "RevocationEntry",
    "GossipProtocol",
    "RevocationType",
]
