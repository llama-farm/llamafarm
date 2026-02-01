"""
FastAPI router for semantic routing endpoints.

Provides HTTP API for:
- Capability announcements
- Intent routing
- Gradient table inspection
- Mesh status and health
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, status
import numpy as np

from .embeddings import EmbeddingEngine
from .matcher import CapabilityMatcher, Capability
from .gradient import GradientTable
from .gossip import GossipProtocol, CapabilityInfo

router = APIRouter(prefix="/router", tags=["router"])


# Request/Response Models
class IntentRequest(BaseModel):
    """Request to route an intent."""
    text: str = Field(..., description="Intent text to route")
    min_score: float = Field(0.5, ge=0.0, le=1.0, description="Minimum match score")


class CapabilityResponse(BaseModel):
    """Capability information in responses."""
    id: str
    label: str
    description: str
    score: float
    hops: int = 0
    via_node: Optional[str] = None
    local: bool = True


class RouteResponse(BaseModel):
    """Response from intent routing."""
    action: str
    capability: Optional[CapabilityResponse] = None
    score: float = 0.0
    next_hop: Optional[str] = None
    reason: Optional[str] = None


class GradientEntryResponse(BaseModel):
    """Gradient table entry for inspection."""
    capability_id: str
    capability_label: str
    hops: int
    next_hop: str
    via_node: str
    estimated_latency_ms: float
    confidence: float


class MeshStatusResponse(BaseModel):
    """Status of the mesh network."""
    node_id: str
    known_nodes: int
    gradient_table_size: int
    local_capabilities: int
    announcements_sent: int
    announcements_received: int
    avg_hops: float
    avg_latency_ms: float


class RegisterCapabilityRequest(BaseModel):
    """Request to register a new local capability."""
    label: str = Field(..., description="Capability label")
    description: str = Field(..., description="Capability description")
    handler: str = Field("default", description="Handler function name")
    models: List[str] = Field(default_factory=list, description="Required models")


# Global state (initialized on startup via dependency injection)
_embedding_engine: Optional[EmbeddingEngine] = None
_gradient_table: Optional[GradientTable] = None
_matcher: Optional[CapabilityMatcher] = None
_gossip: Optional[GossipProtocol] = None
_local_capabilities: List[Capability] = []
_node_id: str = "llamafarm-default"


def init_router_state(
    node_id: str,
    embedding_engine: EmbeddingEngine,
    gradient_table: GradientTable,
    matcher: CapabilityMatcher,
    gossip: GossipProtocol,
    local_capabilities: List[Capability]
):
    """Initialize router state (called from FastAPI startup)."""
    global _embedding_engine, _gradient_table, _matcher, _gossip, _local_capabilities, _node_id
    _node_id = node_id
    _embedding_engine = embedding_engine
    _gradient_table = gradient_table
    _matcher = matcher
    _gossip = gossip
    _local_capabilities = local_capabilities


@router.get("/health", response_model=dict)
async def health():
    """Health check for router subsystem."""
    if _embedding_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Router not initialized"
        )
    
    backend = _embedding_engine.backend if _embedding_engine._initialized else "not initialized"
    
    return {
        "status": "healthy",
        "embedding_backend": backend,
        "gradient_table_size": len(_gradient_table) if _gradient_table else 0,
        "local_capabilities": len(_local_capabilities)
    }


@router.post("/route", response_model=RouteResponse)
async def route_intent(request: IntentRequest):
    """
    Route an intent to the best matching capability.
    
    Returns routing decision with next hop or local handler.
    """
    if not _embedding_engine or not _matcher:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Router not initialized"
        )
    
    # Generate intent embedding
    intent_vec = await _embedding_engine.embed(request.text)
    
    # Get gradient entries
    gradient_entries = _gradient_table.to_routing_tuples() if _gradient_table else []
    
    # Match intent
    result = _matcher.match_with_routing(
        intent_vec,
        _local_capabilities,
        gradient_entries
    )
    
    response = RouteResponse(
        action=result.action.value,
        score=result.score
    )
    
    if result.matched:
        if result.capability:
            # Local match
            response.capability = CapabilityResponse(
                id=result.capability.id,
                label=result.capability.label,
                description=result.capability.description,
                score=result.score,
                hops=0,
                local=True
            )
        else:
            # Remote match
            response.next_hop = result.next_hop
            response.capability = CapabilityResponse(
                id="remote",
                label="remote_capability",
                description="Remote capability",
                score=result.score,
                hops=result.hops,
                via_node=result.via_node,
                local=False
            )
    else:
        response.reason = "No capability match above threshold"
    
    return response


@router.get("/capabilities", response_model=List[CapabilityResponse])
async def list_capabilities():
    """List all known capabilities (local + remote)."""
    capabilities = []
    
    # Add local capabilities
    for cap in _local_capabilities:
        capabilities.append(CapabilityResponse(
            id=cap.id,
            label=cap.label,
            description=cap.description,
            score=1.0,
            hops=0,
            local=True
        ))
    
    # Add remote capabilities from gradient table
    if _gradient_table:
        for entry in _gradient_table:
            capabilities.append(CapabilityResponse(
                id=entry.capability_id,
                label=entry.capability_label,
                description="",
                score=entry.confidence,
                hops=entry.hops,
                via_node=entry.via_node,
                local=False
            ))
    
    return capabilities


@router.post("/capabilities/register", response_model=CapabilityResponse)
async def register_capability(request: RegisterCapabilityRequest):
    """Register a new local capability."""
    if not _embedding_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Router not initialized"
        )
    
    # Generate embedding for capability
    cap_text = f"{request.label}: {request.description}"
    cap_vec = await _embedding_engine.embed(cap_text)
    
    # Create capability
    capability = Capability(
        id=f"{_node_id}:{request.label}",
        label=request.label,
        description=request.description,
        vector=cap_vec,
        handler=request.handler,
        models=request.models
    )
    
    _local_capabilities.append(capability)
    
    return CapabilityResponse(
        id=capability.id,
        label=capability.label,
        description=capability.description,
        score=1.0,
        hops=0,
        local=True
    )


@router.get("/gradient", response_model=List[GradientEntryResponse])
async def get_gradient_table():
    """Inspect the gradient table."""
    if not _gradient_table:
        return []
    
    entries = []
    for entry in _gradient_table:
        entries.append(GradientEntryResponse(
            capability_id=entry.capability_id,
            capability_label=entry.capability_label,
            hops=entry.hops,
            next_hop=entry.next_hop,
            via_node=entry.via_node,
            estimated_latency_ms=entry.estimated_latency_ms,
            confidence=entry.confidence
        ))
    
    return entries


@router.get("/mesh/status", response_model=MeshStatusResponse)
async def get_mesh_status():
    """Get mesh network status."""
    if not _gradient_table or not _gossip:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Router not initialized"
        )
    
    table_stats = _gradient_table.stats()
    gossip_stats = _gossip.stats()
    
    return MeshStatusResponse(
        node_id=_node_id,
        known_nodes=gossip_stats["known_nodes"],
        gradient_table_size=gossip_stats["gradient_table_size"],
        local_capabilities=len(_local_capabilities),
        announcements_sent=gossip_stats["announcements_sent"],
        announcements_received=gossip_stats["announcements_received"],
        avg_hops=table_stats["avg_hops"],
        avg_latency_ms=table_stats["avg_latency_ms"]
    )


@router.post("/mesh/announce")
async def trigger_announcement():
    """Manually trigger a capability announcement."""
    if not _gossip:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Router not initialized"
        )
    
    await _gossip.announce()
    
    return {"status": "announced"}
