"""
Device Identity: Hardware-bound cryptographic identity for mesh devices.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .crypto import KeyPair, get_hardware_fingerprint, bind_key_to_hardware


@dataclass
class DeviceIdentity:
    """
    A device's identity within the mesh.
    
    Each device has:
    - A unique Ed25519 keypair
    - Hardware binding (prevents credential theft)
    - Declared capabilities
    - Optional human-readable name
    """
    device_id: str  # Hash of public key
    public_key: str
    hardware_hash: str
    name: str
    capabilities: List[str]
    tier: str  # "compute", "edge", "iot"
    created_at: int
    
    # Local state
    _key_pair: Optional[KeyPair] = field(default=None, repr=False)
    
    @classmethod
    def create(
        cls,
        name: str,
        capabilities: Optional[List[str]] = None,
        tier: str = "edge"
    ) -> "DeviceIdentity":
        """
        Create a new device identity.
        
        Generates a new keypair and binds it to this hardware.
        """
        # Generate keypair
        key_pair = KeyPair.generate()
        
        # Get hardware fingerprint
        hw_hash = get_hardware_fingerprint()
        
        device = cls(
            device_id=key_pair.key_id(),
            public_key=key_pair.public_key_b64(),
            hardware_hash=hw_hash,
            name=name,
            capabilities=capabilities or [],
            tier=tier,
            created_at=int(time.time())
        )
        device._key_pair = key_pair
        
        return device
    
    def sign(self, message: bytes) -> str:
        """Sign a message with this device's key."""
        if not self._key_pair:
            raise RuntimeError("No private key available for signing")
        return self._key_pair.sign_b64(message)
    
    def create_join_request(self, mesh_id: str) -> dict:
        """
        Create a request to join a mesh.
        
        This is presented to an existing mesh member for approval.
        """
        request_data = {
            "type": "join_request",
            "version": 1,
            "mesh_id": mesh_id,
            "device": {
                "device_id": self.device_id,
                "public_key": self.public_key,
                "hardware_hash": self.hardware_hash,
                "name": self.name,
                "capabilities": self.capabilities,
                "tier": self.tier
            },
            "attestation": bind_key_to_hardware(self._key_pair),
            "requested_at": int(time.time())
        }
        
        # Sign the request
        message = json.dumps(request_data, sort_keys=True).encode()
        signature = self.sign(message)
        
        return {
            **request_data,
            "signature": signature
        }
    
    def verify_hardware_binding(self) -> bool:
        """Verify this identity is running on the correct hardware."""
        current_hw = get_hardware_fingerprint()
        return current_hw == self.hardware_hash
    
    def to_dict(self) -> dict:
        """Serialize device identity (without private key)."""
        return {
            "device_id": self.device_id,
            "public_key": self.public_key,
            "hardware_hash": self.hardware_hash,
            "name": self.name,
            "capabilities": self.capabilities,
            "tier": self.tier,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "DeviceIdentity":
        """Load device identity from dict."""
        return cls(
            device_id=data["device_id"],
            public_key=data["public_key"],
            hardware_hash=data["hardware_hash"],
            name=data["name"],
            capabilities=data["capabilities"],
            tier=data["tier"],
            created_at=data["created_at"]
        )
    
    def save(self, path: Path) -> None:
        """Save device identity to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        # Save private key separately
        if self._key_pair:
            key_path = path.with_suffix('.key')
            with open(key_path, 'w') as f:
                json.dump({
                    "private_key": self._key_pair.private_bytes().hex()
                }, f)
            # In production: encrypt with hardware key
    
    @classmethod
    def load(cls, path: Path) -> "DeviceIdentity":
        """Load device identity from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        device = cls.from_dict(data)
        
        # Try to load private key
        key_path = path.with_suffix('.key')
        if key_path.exists():
            with open(key_path, 'r') as f:
                key_data = json.load(f)
            device._key_pair = KeyPair.from_private_bytes(
                bytes.fromhex(key_data["private_key"])
            )
        
        return device


def detect_capabilities() -> List[str]:
    """
    Auto-detect device capabilities based on hardware.
    
    In production, this would probe for:
    - GPU (nvidia-smi, Metal)
    - Sensors (I2C, GPIO)
    - Camera (v4l2, AVFoundation)
    - etc.
    """
    import platform
    import os
    import shutil
    
    caps = []
    
    # Check for GPU
    if shutil.which('nvidia-smi'):
        caps.extend(['gpu-inference', 'vision', 'training'])
    
    # Check for Apple Silicon
    if platform.machine() == 'arm64' and platform.system() == 'Darwin':
        caps.extend(['apple-silicon', 'llm-inference'])
    
    # Check RAM
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        if ram_gb >= 64:
            caps.append('llm-70b')
        elif ram_gb >= 32:
            caps.append('llm-13b')
        elif ram_gb >= 16:
            caps.append('llm-7b')
    except:
        pass
    
    # Basic capabilities everyone has
    caps.extend(['text-generation', 'embeddings'])
    
    return list(set(caps))


def detect_tier() -> str:
    """Auto-detect device tier based on hardware."""
    import platform
    
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()
    except:
        ram_gb = 8
        cpu_count = 4
    
    # Compute tier: lots of RAM and CPUs
    if ram_gb >= 64 and cpu_count >= 8:
        return "compute"
    
    # Edge tier: decent resources
    if ram_gb >= 4:
        return "edge"
    
    # IoT tier: minimal resources
    return "iot"
