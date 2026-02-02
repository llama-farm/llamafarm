"""
Token Renewal: Offline token refresh mechanisms.

The problem: Tokens expire (for security), but network might be down.
The solution: Multiple renewal strategies depending on trust level.

Strategies:
1. SELF_RENEWAL - Device can extend its own token (limited times)
2. PEER_RENEWAL - Any mesh member can vouch for another
3. PROOF_OF_WORK - Device proves it's still active via computation
4. GRACE_PERIOD - Token remains valid for short period after expiry
"""

import hashlib
import json
import time
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

from .crypto import KeyPair, verify_signature_b64


class RenewalStrategy(Enum):
    """Available token renewal strategies."""
    SELF_RENEWAL = "self_renewal"      # Device signs its own extension
    PEER_RENEWAL = "peer_renewal"      # Another mesh member vouches
    PROOF_OF_WORK = "proof_of_work"    # Computational proof of activity
    GRACE_PERIOD = "grace_period"      # Built-in grace after expiry


@dataclass
class RenewalPolicy:
    """
    Configures how tokens can be renewed.
    
    This is set at the mesh level and embedded in tokens.
    """
    # Self-renewal limits
    max_self_renewals: int = 3          # How many times device can self-renew
    self_renewal_hours: int = 24        # How long each self-renewal lasts
    
    # Peer renewal
    peer_renewal_enabled: bool = True   # Can peers vouch for each other?
    peer_renewal_hours: int = 48        # How long peer renewal lasts
    min_peer_vouches: int = 1           # Minimum peers needed to vouch
    
    # Proof of work
    pow_enabled: bool = True            # Can device extend via PoW?
    pow_difficulty: int = 20            # Leading zero bits required
    pow_renewal_hours: int = 12         # How long PoW renewal lasts
    
    # Grace period
    grace_period_minutes: int = 60      # Token valid this long after expiry
    
    def to_dict(self) -> dict:
        return {
            "self_renewal": {
                "max_renewals": self.max_self_renewals,
                "duration_hours": self.self_renewal_hours
            },
            "peer_renewal": {
                "enabled": self.peer_renewal_enabled,
                "duration_hours": self.peer_renewal_hours,
                "min_vouches": self.min_peer_vouches
            },
            "proof_of_work": {
                "enabled": self.pow_enabled,
                "difficulty": self.pow_difficulty,
                "duration_hours": self.pow_renewal_hours
            },
            "grace_period_minutes": self.grace_period_minutes
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "RenewalPolicy":
        return cls(
            max_self_renewals=data["self_renewal"]["max_renewals"],
            self_renewal_hours=data["self_renewal"]["duration_hours"],
            peer_renewal_enabled=data["peer_renewal"]["enabled"],
            peer_renewal_hours=data["peer_renewal"]["duration_hours"],
            min_peer_vouches=data["peer_renewal"]["min_vouches"],
            pow_enabled=data["proof_of_work"]["enabled"],
            pow_difficulty=data["proof_of_work"]["difficulty"],
            pow_renewal_hours=data["proof_of_work"]["duration_hours"],
            grace_period_minutes=data["grace_period_minutes"]
        )


@dataclass
class RenewalRecord:
    """
    Record of a token renewal.
    
    Appended to token to track renewal history.
    """
    renewal_id: str
    strategy: str
    renewed_at: int
    new_expires_at: int
    renewal_count: int  # How many times this token has been renewed
    
    # Strategy-specific data
    voucher_ids: Optional[List[str]] = None  # For peer renewal
    pow_nonce: Optional[int] = None          # For proof of work
    
    # Signature from renewing party
    signature: str = ""
    signer_id: str = ""
    signer_public_key: str = ""
    
    def to_dict(self) -> dict:
        return {
            "renewal_id": self.renewal_id,
            "strategy": self.strategy,
            "renewed_at": self.renewed_at,
            "new_expires_at": self.new_expires_at,
            "renewal_count": self.renewal_count,
            "voucher_ids": self.voucher_ids,
            "pow_nonce": self.pow_nonce,
            "signer": {
                "id": self.signer_id,
                "public_key": self.signer_public_key,
                "signature": self.signature
            }
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "RenewalRecord":
        return cls(
            renewal_id=data["renewal_id"],
            strategy=data["strategy"],
            renewed_at=data["renewed_at"],
            new_expires_at=data["new_expires_at"],
            renewal_count=data["renewal_count"],
            voucher_ids=data.get("voucher_ids"),
            pow_nonce=data.get("pow_nonce"),
            signature=data["signer"]["signature"],
            signer_id=data["signer"]["id"],
            signer_public_key=data["signer"]["public_key"]
        )


class TokenRenewer:
    """
    Handles token renewal operations.
    """
    
    def __init__(self, policy: RenewalPolicy):
        self.policy = policy
    
    def self_renew(
        self,
        token_data: dict,
        device_keypair: KeyPair,
        current_renewal_count: int = 0
    ) -> Optional[RenewalRecord]:
        """
        Device renews its own token.
        
        Args:
            token_data: The original token data
            device_keypair: The device's keypair
            current_renewal_count: How many times already renewed
        
        Returns:
            RenewalRecord if successful, None if limit reached
        """
        # Check renewal limit
        if current_renewal_count >= self.policy.max_self_renewals:
            return None
        
        now = int(time.time())
        new_expires = now + (self.policy.self_renewal_hours * 3600)
        
        renewal_id = hashlib.sha256(
            f"{token_data.get('device', {}).get('device_id', 'unknown')}:{now}:self".encode()
        ).hexdigest()[:16]
        
        # Build renewal record
        record_data = {
            "renewal_id": renewal_id,
            "strategy": RenewalStrategy.SELF_RENEWAL.value,
            "renewed_at": now,
            "new_expires_at": new_expires,
            "renewal_count": current_renewal_count + 1,
            "token_id": token_data.get("device", {}).get("device_id"),
            "mesh_id": token_data.get("mesh_id")
        }
        
        # Sign the renewal
        message = json.dumps(record_data, sort_keys=True).encode()
        signature = device_keypair.sign_b64(message)
        
        return RenewalRecord(
            renewal_id=renewal_id,
            strategy=RenewalStrategy.SELF_RENEWAL.value,
            renewed_at=now,
            new_expires_at=new_expires,
            renewal_count=current_renewal_count + 1,
            signature=signature,
            signer_id=device_keypair.key_id(),
            signer_public_key=device_keypair.public_key_b64()
        )
    
    def peer_renew(
        self,
        token_data: dict,
        voucher_keypairs: List[KeyPair]
    ) -> Optional[RenewalRecord]:
        """
        Peers vouch for a device's token renewal.
        
        Args:
            token_data: The token being renewed
            voucher_keypairs: Keypairs of vouching peers
        
        Returns:
            RenewalRecord if enough vouches, None otherwise
        """
        if not self.policy.peer_renewal_enabled:
            return None
        
        if len(voucher_keypairs) < self.policy.min_peer_vouches:
            return None
        
        now = int(time.time())
        new_expires = now + (self.policy.peer_renewal_hours * 3600)
        
        voucher_ids = [kp.key_id() for kp in voucher_keypairs]
        
        renewal_id = hashlib.sha256(
            f"{token_data.get('device', {}).get('device_id', 'unknown')}:{now}:peer".encode()
        ).hexdigest()[:16]
        
        # Build renewal record
        record_data = {
            "renewal_id": renewal_id,
            "strategy": RenewalStrategy.PEER_RENEWAL.value,
            "renewed_at": now,
            "new_expires_at": new_expires,
            "voucher_ids": voucher_ids,
            "token_id": token_data.get("device", {}).get("device_id"),
            "mesh_id": token_data.get("mesh_id")
        }
        
        # Each voucher signs (in production, combine into aggregate signature)
        # For now, use first voucher as primary signer
        message = json.dumps(record_data, sort_keys=True).encode()
        signature = voucher_keypairs[0].sign_b64(message)
        
        return RenewalRecord(
            renewal_id=renewal_id,
            strategy=RenewalStrategy.PEER_RENEWAL.value,
            renewed_at=now,
            new_expires_at=new_expires,
            renewal_count=0,  # Peer renewal resets self-renewal count
            voucher_ids=voucher_ids,
            signature=signature,
            signer_id=voucher_keypairs[0].key_id(),
            signer_public_key=voucher_keypairs[0].public_key_b64()
        )
    
    def proof_of_work_renew(
        self,
        token_data: dict,
        device_keypair: KeyPair
    ) -> Optional[RenewalRecord]:
        """
        Device proves activity via proof-of-work to renew token.
        
        This is a fallback when self-renewals are exhausted and
        no peers are available.
        
        Args:
            token_data: The token being renewed
            device_keypair: Device's keypair
        
        Returns:
            RenewalRecord with valid PoW, or None if PoW disabled
        """
        if not self.policy.pow_enabled:
            return None
        
        now = int(time.time())
        device_id = token_data.get("device", {}).get("device_id", "unknown")
        
        # Find a nonce that produces required leading zeros
        challenge = f"{device_id}:{now}:pow"
        nonce = self._find_pow_nonce(challenge, self.policy.pow_difficulty)
        
        new_expires = now + (self.policy.pow_renewal_hours * 3600)
        
        renewal_id = hashlib.sha256(
            f"{device_id}:{now}:pow:{nonce}".encode()
        ).hexdigest()[:16]
        
        record_data = {
            "renewal_id": renewal_id,
            "strategy": RenewalStrategy.PROOF_OF_WORK.value,
            "renewed_at": now,
            "new_expires_at": new_expires,
            "pow_nonce": nonce,
            "pow_difficulty": self.policy.pow_difficulty,
            "token_id": device_id,
            "mesh_id": token_data.get("mesh_id")
        }
        
        message = json.dumps(record_data, sort_keys=True).encode()
        signature = device_keypair.sign_b64(message)
        
        return RenewalRecord(
            renewal_id=renewal_id,
            strategy=RenewalStrategy.PROOF_OF_WORK.value,
            renewed_at=now,
            new_expires_at=new_expires,
            renewal_count=0,
            pow_nonce=nonce,
            signature=signature,
            signer_id=device_keypair.key_id(),
            signer_public_key=device_keypair.public_key_b64()
        )
    
    def _find_pow_nonce(self, challenge: str, difficulty: int) -> int:
        """Find nonce that produces hash with required leading zero bits."""
        target = 2 ** (256 - difficulty)
        nonce = 0
        
        while True:
            candidate = f"{challenge}:{nonce}"
            hash_bytes = hashlib.sha256(candidate.encode()).digest()
            hash_int = int.from_bytes(hash_bytes, 'big')
            
            if hash_int < target:
                return nonce
            
            nonce += 1
            
            # Safety limit
            if nonce > 10_000_000:
                raise RuntimeError("PoW computation limit exceeded")
    
    def is_in_grace_period(self, expires_at: int) -> bool:
        """Check if a token is in its grace period."""
        now = int(time.time())
        grace_end = expires_at + (self.policy.grace_period_minutes * 60)
        return expires_at < now <= grace_end
    
    def verify_renewal(
        self,
        renewal: RenewalRecord,
        original_token_data: dict
    ) -> tuple:
        """
        Verify a renewal record is valid.
        
        Returns:
            (is_valid, reason)
        """
        # Verify signature
        if renewal.strategy == RenewalStrategy.SELF_RENEWAL.value:
            # Signer must be the device itself
            device_key = original_token_data.get("device", {}).get("public_key")
            if renewal.signer_public_key != device_key:
                return False, "Self-renewal signer mismatch"
            
            # Check renewal count within limits
            if renewal.renewal_count > self.policy.max_self_renewals:
                return False, "Self-renewal limit exceeded"
        
        elif renewal.strategy == RenewalStrategy.PEER_RENEWAL.value:
            if not self.policy.peer_renewal_enabled:
                return False, "Peer renewal not enabled"
            
            if len(renewal.voucher_ids or []) < self.policy.min_peer_vouches:
                return False, "Insufficient peer vouches"
        
        elif renewal.strategy == RenewalStrategy.PROOF_OF_WORK.value:
            if not self.policy.pow_enabled:
                return False, "PoW renewal not enabled"
            
            # Verify the PoW
            device_id = original_token_data.get("device", {}).get("device_id", "unknown")
            challenge = f"{device_id}:{renewal.renewed_at}:pow"
            
            if not self._verify_pow(challenge, renewal.pow_nonce, self.policy.pow_difficulty):
                return False, "Invalid proof of work"
        
        # Verify signature
        record_data = {
            "renewal_id": renewal.renewal_id,
            "strategy": renewal.strategy,
            "renewed_at": renewal.renewed_at,
            "new_expires_at": renewal.new_expires_at,
            "renewal_count": renewal.renewal_count if renewal.strategy == RenewalStrategy.SELF_RENEWAL.value else 0,
            "token_id": original_token_data.get("device", {}).get("device_id"),
            "mesh_id": original_token_data.get("mesh_id")
        }
        
        if renewal.voucher_ids:
            record_data["voucher_ids"] = renewal.voucher_ids
        
        if renewal.pow_nonce is not None:
            record_data["pow_nonce"] = renewal.pow_nonce
            record_data["pow_difficulty"] = self.policy.pow_difficulty
        
        message = json.dumps(record_data, sort_keys=True).encode()
        
        if not verify_signature_b64(
            renewal.signer_public_key,
            message,
            renewal.signature
        ):
            return False, "Invalid renewal signature"
        
        return True, "Renewal valid"
    
    def _verify_pow(self, challenge: str, nonce: int, difficulty: int) -> bool:
        """Verify a proof of work."""
        target = 2 ** (256 - difficulty)
        candidate = f"{challenge}:{nonce}"
        hash_bytes = hashlib.sha256(candidate.encode()).digest()
        hash_int = int.from_bytes(hash_bytes, 'big')
        return hash_int < target
