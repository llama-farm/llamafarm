"""
Identity compatibility shim.

Re-exports from rownd_local.crypto for backwards compatibility
with the Atmosphere mesh code.
"""

from .rownd_local.crypto import (
    KeyPair,
    split_secret,
    combine_shares,
    verify_signature_b64,
)

import hashlib
import platform
import uuid


def get_hardware_fingerprint() -> str:
    """
    Generate a hardware fingerprint for this device.
    
    Used to bind tokens to specific hardware.
    """
    # Gather hardware identifiers
    parts = [
        platform.node(),           # hostname
        platform.machine(),        # architecture
        platform.system(),         # OS
        str(uuid.getnode()),       # MAC address
    ]
    
    # Hash them together
    combined = "|".join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:32]


__all__ = [
    "KeyPair",
    "split_secret",
    "combine_shares",
    "verify_signature_b64",
    "get_hardware_fingerprint",
]
