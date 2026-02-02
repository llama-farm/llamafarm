"""
Gossip-Based Revocation: Propagate revocations without central server.

The challenge: How do you revoke credentials when there's no central authority?
The solution: Epidemic (gossip) protocol with cryptographic proofs.

Key properties:
- Revocations are signed (can't fake revocation)
- Revocations spread via gossip (no central point)
- Revocations are cumulative (can only add, not remove)
- Revocations have sequence numbers (detect missing entries)
- Eventually consistent (all nodes converge)
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum

from .crypto import KeyPair, verify_signature_b64


class RevocationType(Enum):
    """Types of revocations."""
    DEVICE = "device"           # Revoke a device's membership
    DELEGATION = "delegation"   # Revoke a delegation token
    FOUNDER = "founder"         # Revoke a founding member (requires threshold)
    CAPABILITY = "capability"   # Revoke specific capability from device


@dataclass
class RevocationEntry:
    """
    A single revocation in the revocation log.
    """
    revocation_id: str
    revocation_type: str
    
    # What's being revoked
    target_id: str              # Device ID, delegation ID, or founder ID
    target_type: str            # "device", "delegation", "founder"
    
    # Why (for audit)
    reason: str
    
    # Who issued the revocation
    issuer_id: str
    issuer_public_key: str
    
    # Timing
    revoked_at: int
    sequence_number: int        # Monotonic sequence for ordering
    
    # Signature
    signature: str
    
    def to_dict(self) -> dict:
        return {
            "revocation_id": self.revocation_id,
            "revocation_type": self.revocation_type,
            "target": {
                "id": self.target_id,
                "type": self.target_type
            },
            "reason": self.reason,
            "issuer": {
                "id": self.issuer_id,
                "public_key": self.issuer_public_key
            },
            "revoked_at": self.revoked_at,
            "sequence_number": self.sequence_number,
            "signature": self.signature
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "RevocationEntry":
        return cls(
            revocation_id=data["revocation_id"],
            revocation_type=data["revocation_type"],
            target_id=data["target"]["id"],
            target_type=data["target"]["type"],
            reason=data["reason"],
            issuer_id=data["issuer"]["id"],
            issuer_public_key=data["issuer"]["public_key"],
            revoked_at=data["revoked_at"],
            sequence_number=data["sequence_number"],
            signature=data["signature"]
        )


class RevocationLog:
    """
    Local revocation log that syncs via gossip.
    """
    
    def __init__(self, mesh_id: str):
        self.mesh_id = mesh_id
        self.entries: Dict[str, RevocationEntry] = {}
        self.sequence_counter = 0
        
        # Index by target for fast lookup
        self._by_target: Dict[str, Set[str]] = {}  # target_id -> revocation_ids
        
        # Track which sequences we've seen (for gap detection)
        self._seen_sequences: Set[int] = set()
        
        # Known issuers (founders who can revoke)
        self._authorized_issuers: Dict[str, str] = {}  # id -> public_key
    
    def add_authorized_issuer(self, issuer_id: str, public_key: str) -> None:
        """Add a founding member who can issue revocations."""
        self._authorized_issuers[issuer_id] = public_key
    
    def create_revocation(
        self,
        target_id: str,
        target_type: str,
        reason: str,
        issuer_keypair: KeyPair
    ) -> RevocationEntry:
        """
        Create a new revocation entry.
        
        Args:
            target_id: ID of what's being revoked
            target_type: Type of target (device, delegation, founder)
            reason: Human-readable reason
            issuer_keypair: Keypair of the issuer (must be authorized)
        
        Returns:
            Signed RevocationEntry
        """
        self.sequence_counter += 1
        now = int(time.time())
        
        revocation_id = hashlib.sha256(
            f"{self.mesh_id}:{target_id}:{now}:{self.sequence_counter}".encode()
        ).hexdigest()[:16]
        
        # Build entry data (without signature)
        entry_data = {
            "revocation_id": revocation_id,
            "revocation_type": RevocationType.DEVICE.value,
            "target": {
                "id": target_id,
                "type": target_type
            },
            "reason": reason,
            "issuer": {
                "id": issuer_keypair.key_id(),
                "public_key": issuer_keypair.public_key_b64()
            },
            "revoked_at": now,
            "sequence_number": self.sequence_counter,
            "mesh_id": self.mesh_id
        }
        
        # Sign
        message = json.dumps(entry_data, sort_keys=True).encode()
        signature = issuer_keypair.sign_b64(message)
        
        entry = RevocationEntry(
            revocation_id=revocation_id,
            revocation_type=RevocationType.DEVICE.value,
            target_id=target_id,
            target_type=target_type,
            reason=reason,
            issuer_id=issuer_keypair.key_id(),
            issuer_public_key=issuer_keypair.public_key_b64(),
            revoked_at=now,
            sequence_number=self.sequence_counter,
            signature=signature
        )
        
        self._add_entry(entry)
        return entry
    
    def _add_entry(self, entry: RevocationEntry) -> None:
        """Add an entry to the log."""
        self.entries[entry.revocation_id] = entry
        
        # Index by target
        if entry.target_id not in self._by_target:
            self._by_target[entry.target_id] = set()
        self._by_target[entry.target_id].add(entry.revocation_id)
        
        self._seen_sequences.add(entry.sequence_number)
    
    def is_revoked(self, target_id: str) -> bool:
        """Check if a target has been revoked."""
        return target_id in self._by_target and len(self._by_target[target_id]) > 0
    
    def get_revocations_for(self, target_id: str) -> List[RevocationEntry]:
        """Get all revocations for a target."""
        if target_id not in self._by_target:
            return []
        
        return [
            self.entries[rev_id]
            for rev_id in self._by_target[target_id]
        ]
    
    def verify_entry(self, entry: RevocationEntry) -> Tuple[bool, str]:
        """
        Verify a revocation entry is valid.
        
        Returns:
            (is_valid, reason)
        """
        # Check issuer is authorized
        if entry.issuer_id not in self._authorized_issuers:
            return False, f"Issuer {entry.issuer_id} not authorized"
        
        if self._authorized_issuers[entry.issuer_id] != entry.issuer_public_key:
            return False, "Issuer public key mismatch"
        
        # Verify signature
        entry_data = {
            "revocation_id": entry.revocation_id,
            "revocation_type": entry.revocation_type,
            "target": {
                "id": entry.target_id,
                "type": entry.target_type
            },
            "reason": entry.reason,
            "issuer": {
                "id": entry.issuer_id,
                "public_key": entry.issuer_public_key
            },
            "revoked_at": entry.revoked_at,
            "sequence_number": entry.sequence_number,
            "mesh_id": self.mesh_id
        }
        
        message = json.dumps(entry_data, sort_keys=True).encode()
        
        if not verify_signature_b64(
            entry.issuer_public_key,
            message,
            entry.signature
        ):
            return False, "Invalid signature"
        
        return True, "Valid"
    
    # ==================== Gossip Protocol ====================
    
    def get_gossip_digest(self) -> dict:
        """
        Get a digest of our revocation log for gossip.
        
        This is sent to peers to determine what to sync.
        """
        return {
            "mesh_id": self.mesh_id,
            "entry_count": len(self.entries),
            "max_sequence": max(self._seen_sequences) if self._seen_sequences else 0,
            "entry_hashes": sorted(self.entries.keys())[:100]  # First 100 for comparison
        }
    
    def get_entries_since(self, sequence: int) -> List[RevocationEntry]:
        """Get all entries with sequence > given value."""
        return [
            entry for entry in self.entries.values()
            if entry.sequence_number > sequence
        ]
    
    def get_missing_sequences(self) -> List[int]:
        """Detect gaps in our sequence numbers."""
        if not self._seen_sequences:
            return []
        
        max_seq = max(self._seen_sequences)
        expected = set(range(1, max_seq + 1))
        return sorted(expected - self._seen_sequences)
    
    def merge_entries(self, entries: List[RevocationEntry]) -> dict:
        """
        Merge entries received from gossip.
        
        Returns:
            {
                "accepted": count of new entries added,
                "rejected": count of invalid entries,
                "already_had": count of duplicates
            }
        """
        result = {
            "accepted": 0,
            "rejected": 0,
            "already_had": 0
        }
        
        for entry in entries:
            # Skip duplicates
            if entry.revocation_id in self.entries:
                result["already_had"] += 1
                continue
            
            # Verify entry
            is_valid, reason = self.verify_entry(entry)
            if not is_valid:
                result["rejected"] += 1
                continue
            
            # Add it
            self._add_entry(entry)
            result["accepted"] += 1
            
            # Update sequence counter if needed
            if entry.sequence_number > self.sequence_counter:
                self.sequence_counter = entry.sequence_number
        
        return result
    
    def get_all_entries(self) -> List[dict]:
        """Get all entries for persistence or full sync."""
        return [entry.to_dict() for entry in self.entries.values()]
    
    def load_entries(self, entries_data: List[dict]) -> None:
        """Load entries from persistence."""
        for data in entries_data:
            entry = RevocationEntry.from_dict(data)
            self._add_entry(entry)
            if entry.sequence_number > self.sequence_counter:
                self.sequence_counter = entry.sequence_number


class GossipProtocol:
    """
    Implements the gossip protocol for revocation propagation.
    
    This is a pull-based anti-entropy protocol:
    1. Periodically exchange digests with random peers
    2. Request missing entries
    3. Send entries we have that peer is missing
    """
    
    def __init__(self, local_log: RevocationLog, node_id: str):
        self.log = local_log
        self.node_id = node_id
        
        # Track peer state
        self.peer_digests: Dict[str, dict] = {}  # peer_id -> last digest
        self.last_sync: Dict[str, int] = {}      # peer_id -> timestamp
    
    def create_sync_request(self, peer_digest: dict) -> dict:
        """
        Create a sync request based on peer's digest.
        
        Returns entries we have that peer might be missing,
        and requests for entries peer has that we're missing.
        """
        our_digest = self.log.get_gossip_digest()
        
        # What peer might be missing
        our_hashes = set(our_digest["entry_hashes"])
        their_hashes = set(peer_digest.get("entry_hashes", []))
        they_might_need = our_hashes - their_hashes
        
        # What we might be missing
        we_might_need = their_hashes - our_hashes
        
        # Get entries they might need
        entries_to_send = [
            self.log.entries[rev_id].to_dict()
            for rev_id in they_might_need
            if rev_id in self.log.entries
        ]
        
        return {
            "type": "sync_request",
            "from_node": self.node_id,
            "our_digest": our_digest,
            "entries": entries_to_send,
            "requesting": list(we_might_need)[:50]  # Limit request size
        }
    
    def handle_sync_request(self, request: dict) -> dict:
        """
        Handle incoming sync request.
        
        Merge their entries and send back requested entries.
        """
        # Merge entries they sent
        their_entries = [
            RevocationEntry.from_dict(d) for d in request.get("entries", [])
        ]
        merge_result = self.log.merge_entries(their_entries)
        
        # Prepare entries they requested
        requested_ids = request.get("requesting", [])
        response_entries = [
            self.log.entries[rev_id].to_dict()
            for rev_id in requested_ids
            if rev_id in self.log.entries
        ]
        
        # Update peer tracking
        peer_id = request.get("from_node")
        if peer_id:
            self.peer_digests[peer_id] = request.get("our_digest", {})
            self.last_sync[peer_id] = int(time.time())
        
        return {
            "type": "sync_response",
            "from_node": self.node_id,
            "entries": response_entries,
            "merge_result": merge_result
        }
    
    def handle_sync_response(self, response: dict) -> dict:
        """Handle sync response, merge any new entries."""
        their_entries = [
            RevocationEntry.from_dict(d) for d in response.get("entries", [])
        ]
        return self.log.merge_entries(their_entries)
    
    def get_peers_needing_sync(self, max_age_seconds: int = 300) -> List[str]:
        """Get peers we haven't synced with recently."""
        now = int(time.time())
        stale_peers = []
        
        for peer_id, last_time in self.last_sync.items():
            if now - last_time > max_age_seconds:
                stale_peers.append(peer_id)
        
        return stale_peers
