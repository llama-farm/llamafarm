"""
Core cryptographic primitives for Rownd Local.

Uses Ed25519 for all signing operations (fast, secure, small keys).
"""

import hashlib
import secrets
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.exceptions import InvalidSignature
import base64
import json


@dataclass
class KeyPair:
    """Ed25519 key pair for signing."""
    private_key: Ed25519PrivateKey
    public_key: Ed25519PublicKey
    
    @classmethod
    def generate(cls) -> "KeyPair":
        """Generate a new random key pair."""
        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        return cls(private_key=private_key, public_key=public_key)
    
    @classmethod
    def from_private_bytes(cls, private_bytes: bytes) -> "KeyPair":
        """Load key pair from private key bytes."""
        private_key = Ed25519PrivateKey.from_private_bytes(private_bytes)
        public_key = private_key.public_key()
        return cls(private_key=private_key, public_key=public_key)
    
    def private_bytes(self) -> bytes:
        """Export private key as raw bytes (32 bytes)."""
        return self.private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption()
        )
    
    def public_bytes(self) -> bytes:
        """Export public key as raw bytes (32 bytes)."""
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
    
    def public_key_b64(self) -> str:
        """Export public key as base64 string."""
        return base64.b64encode(self.public_bytes()).decode('ascii')
    
    def key_id(self) -> str:
        """Get the key ID (hash of public key)."""
        return hashlib.sha256(self.public_bytes()).hexdigest()[:16]
    
    def sign(self, message: bytes) -> bytes:
        """Sign a message."""
        return self.private_key.sign(message)
    
    def sign_b64(self, message: bytes) -> str:
        """Sign a message and return base64 signature."""
        return base64.b64encode(self.sign(message)).decode('ascii')


def verify_signature(
    public_key_bytes: bytes,
    message: bytes,
    signature: bytes
) -> bool:
    """Verify an Ed25519 signature."""
    try:
        public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
        public_key.verify(signature, message)
        return True
    except InvalidSignature:
        return False
    except Exception:
        return False


def verify_signature_b64(
    public_key_b64: str,
    message: bytes,
    signature_b64: str
) -> bool:
    """Verify a base64-encoded signature."""
    try:
        public_key_bytes = base64.b64decode(public_key_b64)
        signature = base64.b64decode(signature_b64)
        return verify_signature(public_key_bytes, message, signature)
    except Exception:
        return False


# ==================== Shamir Secret Sharing ====================
# For threshold signing (k-of-n founders can issue certificates)

def _mod_inverse(a: int, p: int) -> int:
    """Modular multiplicative inverse using extended Euclidean algorithm."""
    def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y
    
    _, x, _ = extended_gcd(a % p, p)
    return (x % p + p) % p


# Large prime for finite field operations
PRIME = 2**255 - 19  # Same as Ed25519's field


def split_secret(secret: bytes, threshold: int, num_shares: int) -> List[Tuple[int, bytes]]:
    """
    Split a secret into shares using Shamir's Secret Sharing.
    
    Args:
        secret: The secret to split (up to 32 bytes)
        threshold: Minimum shares needed to reconstruct
        num_shares: Total number of shares to create
    
    Returns:
        List of (index, share) tuples
    """
    if threshold > num_shares:
        raise ValueError("Threshold cannot exceed number of shares")
    
    # Convert secret to integer
    secret_int = int.from_bytes(secret.ljust(32, b'\x00')[:32], 'big')
    
    # Generate random coefficients for polynomial
    coefficients = [secret_int]
    for _ in range(threshold - 1):
        coefficients.append(secrets.randbelow(PRIME))
    
    # Evaluate polynomial at x = 1, 2, ..., num_shares
    shares = []
    for x in range(1, num_shares + 1):
        y = 0
        for i, coeff in enumerate(coefficients):
            y = (y + coeff * pow(x, i, PRIME)) % PRIME
        share_bytes = y.to_bytes(32, 'big')
        shares.append((x, share_bytes))
    
    return shares


def combine_shares(shares: List[Tuple[int, bytes]]) -> bytes:
    """
    Reconstruct a secret from shares using Lagrange interpolation.
    
    Args:
        shares: List of (index, share) tuples
    
    Returns:
        The reconstructed secret
    """
    if len(shares) < 2:
        raise ValueError("Need at least 2 shares")
    
    # Convert shares to integers
    points = [(x, int.from_bytes(share, 'big')) for x, share in shares]
    
    # Lagrange interpolation at x = 0
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


# ==================== Hardware Identity ====================

def get_hardware_fingerprint() -> str:
    """
    Get a hardware fingerprint for this device.
    
    In production, this would use:
    - TPM 2.0 attestation
    - Secure Enclave (Apple)
    - Android Keystore
    - PUF (embedded devices)
    
    For demo, we use a stable machine ID.
    """
    import platform
    import uuid
    
    # Combine multiple system identifiers
    components = [
        platform.node(),  # hostname
        platform.machine(),  # architecture
        platform.processor(),  # CPU info
    ]
    
    # Try to get a more stable ID
    try:
        # On macOS, use hardware UUID
        import subprocess
        result = subprocess.run(
            ['ioreg', '-rd1', '-c', 'IOPlatformExpertDevice'],
            capture_output=True,
            text=True
        )
        if 'IOPlatformUUID' in result.stdout:
            for line in result.stdout.split('\n'):
                if 'IOPlatformUUID' in line:
                    uuid_str = line.split('"')[-2]
                    components.append(uuid_str)
                    break
    except:
        pass
    
    # Hash all components
    combined = '|'.join(components)
    return hashlib.sha256(combined.encode()).hexdigest()


def bind_key_to_hardware(key_pair: KeyPair) -> str:
    """
    Create a hardware-bound key attestation.
    
    Returns a signature that proves this key was generated on this hardware.
    """
    hw_fingerprint = get_hardware_fingerprint()
    attestation_data = {
        "public_key": key_pair.public_key_b64(),
        "hardware_hash": hw_fingerprint,
        "timestamp": int(time.time()),
        "type": "hardware_binding"
    }
    message = json.dumps(attestation_data, sort_keys=True).encode()
    signature = key_pair.sign_b64(message)
    
    return json.dumps({
        **attestation_data,
        "signature": signature
    })
