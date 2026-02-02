"""
Mesh joining logic.

Handles the process of joining an existing mesh network.
"""

import base64
import hashlib
import json
import secrets
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import aiohttp

from .node import NodeIdentity, MeshIdentity
from ..auth.tokens import MeshToken


@dataclass
class JoinCode:
    """
    A join code for mesh membership.
    
    Format: MESH-XXXX-XXXX-XXXX (12 chars, case-insensitive)
    
    The code encodes:
    - Mesh ID (first 8 chars)
    - Verification hash (last 4 chars)
    """
    mesh_id: str
    mesh_name: str
    mesh_public_key: str
    endpoint: str  # Address of a founding node
    created_at: int
    expires_at: int
    
    @property
    def code(self) -> str:
        """Generate the short join code."""
        # Combine mesh ID with verification
        data = f"{self.mesh_id}:{self.mesh_public_key[:16]}"
        hash_bytes = hashlib.sha256(data.encode()).digest()
        
        # Base32 encode for human readability (no ambiguous chars)
        import base64
        b32 = base64.b32encode(hash_bytes[:9]).decode().replace('=', '')
        
        # Format as XXXX-XXXX-XXXX
        return f"{b32[:4]}-{b32[4:8]}-{b32[8:12]}"
    
    def to_dict(self) -> dict:
        return {
            "mesh_id": self.mesh_id,
            "mesh_name": self.mesh_name,
            "mesh_public_key": self.mesh_public_key,
            "endpoint": self.endpoint,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "code": self.code
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "JoinCode":
        return cls(
            mesh_id=data["mesh_id"],
            mesh_name=data["mesh_name"],
            mesh_public_key=data["mesh_public_key"],
            endpoint=data["endpoint"],
            created_at=data["created_at"],
            expires_at=data["expires_at"]
        )
    
    def to_compact(self) -> str:
        """Encode as compact string for sharing."""
        data = json.dumps(self.to_dict(), separators=(',', ':'))
        return base64.urlsafe_b64encode(data.encode()).decode()
    
    @classmethod
    def from_compact(cls, compact: str) -> "JoinCode":
        """Decode from compact string."""
        data = json.loads(base64.urlsafe_b64decode(compact))
        return cls.from_dict(data)
    
    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at


class MeshJoin:
    """
    Handles the mesh joining process.
    
    Steps:
    1. Node generates identity
    2. Node submits join request with its public key
    3. Founder verifies and issues membership token
    4. Node stores token and joins mesh
    """
    
    def __init__(self, identity: NodeIdentity):
        self.identity = identity
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30.0)
            )
        return self._session
    
    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
    
    def build_join_request(self, capabilities: list[str] = None) -> dict:
        """Build a join request payload."""
        return {
            "type": "join_request",
            "device": {
                "device_id": self.identity.node_id,
                "public_key": self.identity.public_key,
                "name": self.identity.name,
                "hardware_hash": self.identity.hardware_hash,
                "capabilities": capabilities or ["llm", "embeddings"],
                "tier": "compute"
            },
            "timestamp": int(time.time()),
            "signature": self.identity.sign(
                json.dumps({
                    "device_id": self.identity.node_id,
                    "timestamp": int(time.time())
                }, sort_keys=True).encode()
            )
        }
    
    async def join_by_code(self, code: str) -> Tuple[bool, str, Optional[MeshToken]]:
        """
        Join a mesh using a join code.
        
        Returns:
            (success, message, token)
        """
        # Try to decode as compact first
        try:
            if '-' in code:
                # Short code format - need to look up
                return False, "Short code lookup not implemented. Use full code or IP address.", None
            else:
                join_info = JoinCode.from_compact(code)
        except Exception:
            return False, f"Invalid join code format", None
        
        if join_info.is_expired:
            return False, "Join code has expired", None
        
        return await self.join_by_endpoint(
            join_info.endpoint,
            join_info.mesh_id
        )
    
    async def join_by_endpoint(
        self,
        endpoint: str,
        mesh_id: Optional[str] = None
    ) -> Tuple[bool, str, Optional[MeshToken]]:
        """
        Join a mesh by contacting a founding node directly.
        
        Args:
            endpoint: Address of founding node (host:port or http://host:port)
            mesh_id: Optional mesh ID to verify
            
        Returns:
            (success, message, token)
        """
        if not endpoint.startswith("http"):
            endpoint = f"http://{endpoint}"
        
        session = await self._get_session()
        
        try:
            # Build and send join request
            request = self.build_join_request()
            
            async with session.post(
                f"{endpoint}/v1/mesh/join",
                json=request
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    return False, f"Join request rejected: {error}", None
                
                data = await resp.json()
                
                # Verify mesh ID if provided
                if mesh_id and data.get("mesh_id") != mesh_id:
                    return False, "Mesh ID mismatch", None
                
                # Parse token
                token_data = data.get("token")
                if not token_data:
                    return False, "No token in response", None
                
                token = MeshToken.from_dict(token_data)
                
                return True, f"Joined mesh: {token.mesh_name}", token
                
        except aiohttp.ClientError as e:
            return False, f"Connection failed: {e}", None
        except Exception as e:
            return False, f"Join failed: {e}", None
    
    async def join_by_discovery(self, mesh_id: str) -> Tuple[bool, str, Optional[MeshToken]]:
        """
        Join a mesh by finding a founder via mDNS.
        
        Args:
            mesh_id: Mesh ID to join
            
        Returns:
            (success, message, token)
        """
        from .discovery import MeshDiscovery
        
        # Create temporary discovery
        discovery = MeshDiscovery(
            node_id=self.identity.node_id,
            port=0,  # Don't advertise
            mesh_id=None
        )
        
        if not await discovery.start():
            return False, "mDNS discovery not available", None
        
        try:
            # Wait for peers
            import asyncio
            await asyncio.sleep(3.0)
            
            # Find peers in target mesh
            mesh_peers = discovery.get_mesh_peers(mesh_id)
            
            if not mesh_peers:
                return False, f"No peers found for mesh {mesh_id}", None
            
            # Try each peer
            for peer in mesh_peers:
                success, message, token = await self.join_by_endpoint(
                    f"{peer.host}:{peer.port}",
                    mesh_id
                )
                if success:
                    return success, message, token
            
            return False, "Could not join via any discovered peer", None
            
        finally:
            await discovery.stop()


def generate_join_code(
    mesh: MeshIdentity,
    endpoint: str,
    validity_hours: int = 24
) -> JoinCode:
    """Generate a join code for a mesh."""
    now = int(time.time())
    
    return JoinCode(
        mesh_id=mesh.mesh_id,
        mesh_name=mesh.name,
        mesh_public_key=mesh.master_public_key,
        endpoint=endpoint,
        created_at=now,
        expires_at=now + (validity_hours * 3600)
    )
