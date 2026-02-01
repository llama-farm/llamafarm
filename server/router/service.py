"""
Router Service - Integrates semantic routing with config-defined nodes and agents.

Provides:
- Config-driven initialization of router, nodes, and agents
- Automatic node discovery via mDNS/gossip
- Session-based agent management
- Cross-node routing with credential handling
"""

import asyncio
import logging
import os
import socket
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .embeddings import EmbeddingConfig, EmbeddingEngine, EmbeddingBackend
from .matcher import CapabilityMatcher, Capability, MatchResult
from .gradient import GradientTable
from .gossip import GossipProtocol, CapabilityInfo
from .learning import RouteLearner, get_route_learner

logger = logging.getLogger(__name__)


@dataclass
class NodePeerConfig:
    """Configuration for a peer node."""
    name: str
    url: str
    auth_type: str = "none"
    auth_token: Optional[str] = None
    auth_username: Optional[str] = None
    auth_password: Optional[str] = None
    auth_key_path: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    priority: int = 0
    hardware: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str
    description: str = ""
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    prompts: List[str] = field(default_factory=list)
    tools: List[Dict] = field(default_factory=list)
    mcp_servers: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    can_spawn: bool = False
    spawn_allowed_agents: List[str] = field(default_factory=list)
    max_iterations: int = 100
    memory_enabled: bool = True
    memory_max_short_term: int = 100
    memory_max_long_term: int = 1000
    memory_storage_path: Optional[str] = None


@dataclass
class RouterServiceConfig:
    """Full router service configuration.
    
    Default embedding: sentence-transformers/all-MiniLM-L6-v2 (384 dims)
    Matches LlamaFarm's default RAG config.
    """
    # Node identity
    node_id: str = ""
    node_display_name: str = ""
    
    # Embedding config - matches LlamaFarm default.yaml
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    embedding_backend: str = "universal"
    universal_host: str = "localhost"
    universal_port: int = 14345  # LlamaFarm server port
    
    # Routing config
    min_score: float = 0.5
    local_bias: float = 0.1
    max_hops: int = 3
    learning_enabled: bool = True
    degradation_threshold: float = 0.7
    
    # Discovery config
    mdns_enabled: bool = True
    gossip_interval_sec: int = 30
    stale_threshold_sec: int = 120
    
    # Peers and capabilities
    peers: List[NodePeerConfig] = field(default_factory=list)
    local_capabilities: List[Dict[str, Any]] = field(default_factory=list)
    
    # Agents
    agents: List[AgentConfig] = field(default_factory=list)
    default_agent: Optional[str] = None
    session_storage_path: Optional[str] = None


class RouterService:
    """
    Main router service that integrates all routing components.
    
    Initializes from LlamaFarm config and provides:
    - Semantic intent routing
    - Node discovery and peer management
    - Agent session management
    - Quality tracking and learning
    """
    
    def __init__(self, config: RouterServiceConfig):
        self.config = config
        
        # Generate node ID if not provided
        if not config.node_id:
            hostname = socket.gethostname().lower().replace('.', '-')
            self.node_id = f"lf-{hostname}-{uuid.uuid4().hex[:6]}"
        else:
            self.node_id = config.node_id
        
        self.display_name = config.node_display_name or self.node_id
        
        # Core components (initialized in start())
        self._embedding_engine: Optional[EmbeddingEngine] = None
        self._matcher: Optional[CapabilityMatcher] = None
        self._gradient_table: Optional[GradientTable] = None
        self._gossip: Optional[GossipProtocol] = None
        self._learner: Optional[RouteLearner] = None
        
        # Local capabilities
        self._capabilities: List[Capability] = []
        
        # Agent registry
        self._agents: Dict[str, AgentConfig] = {}
        
        # Running state
        self._started = False
        self._background_tasks: List[asyncio.Task] = []
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RouterService":
        """
        Create RouterService from LlamaFarm config dict.
        
        Extracts relevant sections (router, nodes, agents) and builds
        the service configuration.
        """
        router_cfg = config.get("router", {})
        nodes_cfg = config.get("nodes", {})
        agents_cfg = config.get("agents", {})
        
        # Build embedding config
        embedding_cfg = router_cfg.get("embedding", {})
        embedding_backend = embedding_cfg.get("backend", "universal")
        
        # Build peer list
        peers = []
        for peer_dict in nodes_cfg.get("peers", []):
            auth = peer_dict.get("auth", {})
            peers.append(NodePeerConfig(
                name=peer_dict.get("name", "unknown"),
                url=peer_dict.get("url", ""),
                auth_type=auth.get("type", "none"),
                auth_token=auth.get("token"),
                auth_username=auth.get("username"),
                auth_password=auth.get("password"),
                auth_key_path=auth.get("key_path"),
                capabilities=peer_dict.get("capabilities", []),
                priority=peer_dict.get("priority", 0),
                hardware=peer_dict.get("hardware", {})
            ))
        
        # Build local capabilities
        self_cfg = nodes_cfg.get("self", {})
        local_caps = self_cfg.get("capabilities", [])
        
        # Build agent list
        agents = []
        for agent_dict in agents_cfg.get("definitions", []):
            memory_cfg = agent_dict.get("memory", {})
            agents.append(AgentConfig(
                name=agent_dict.get("name", ""),
                description=agent_dict.get("description", ""),
                model=agent_dict.get("model"),
                system_prompt=agent_dict.get("system_prompt"),
                prompts=agent_dict.get("prompts", []),
                tools=agent_dict.get("tools", []),
                mcp_servers=agent_dict.get("mcp_servers", []),
                capabilities=agent_dict.get("capabilities", []),
                can_spawn=agent_dict.get("can_spawn", False),
                spawn_allowed_agents=agent_dict.get("spawn_allowed_agents", []),
                max_iterations=agent_dict.get("max_iterations", 100),
                memory_enabled=memory_cfg.get("enabled", True),
                memory_max_short_term=memory_cfg.get("max_short_term", 100),
                memory_max_long_term=memory_cfg.get("max_long_term", 1000),
                memory_storage_path=memory_cfg.get("storage_path")
            ))
        
        # Build discovery config
        discovery_cfg = nodes_cfg.get("discovery", {})
        
        # Build matching config
        matching_cfg = router_cfg.get("matching", {})
        learning_cfg = router_cfg.get("learning", {})
        
        service_config = RouterServiceConfig(
            node_id=self_cfg.get("node_id", ""),
            node_display_name=self_cfg.get("display_name", ""),
            embedding_model=embedding_cfg.get("model", "nomic-embed-text"),
            embedding_dimension=embedding_cfg.get("dimension", 768),
            embedding_backend=embedding_backend,
            min_score=matching_cfg.get("min_score", 0.5),
            local_bias=matching_cfg.get("local_bias", 0.1),
            max_hops=matching_cfg.get("max_hops", 3),
            learning_enabled=learning_cfg.get("enabled", True),
            degradation_threshold=learning_cfg.get("degradation_threshold", 0.7),
            mdns_enabled=discovery_cfg.get("mdns", True),
            gossip_interval_sec=discovery_cfg.get("gossip_interval_sec", 30),
            stale_threshold_sec=discovery_cfg.get("stale_threshold_sec", 120),
            peers=peers,
            local_capabilities=local_caps,
            agents=agents,
            default_agent=agents_cfg.get("default_agent"),
            session_storage_path=agents_cfg.get("session_storage")
        )
        
        return cls(service_config)
    
    async def start(self) -> None:
        """Start the router service and all background tasks."""
        if self._started:
            return
        
        logger.info(f"Starting RouterService for node {self.node_id}")
        
        # Initialize embedding engine
        backend = EmbeddingBackend.UNIVERSAL
        if self.config.embedding_backend == "ollama":
            backend = EmbeddingBackend.OLLAMA
        elif self.config.embedding_backend == "llamafarm":
            backend = EmbeddingBackend.LLAMAFARM
        
        embed_config = EmbeddingConfig(
            backend=backend,
            model=self.config.embedding_model,
            dimension=self.config.embedding_dimension,
            universal_host=self.config.universal_host,
            universal_port=self.config.universal_port
        )
        
        self._embedding_engine = EmbeddingEngine(embed_config)
        await self._embedding_engine.initialize()
        logger.info(f"Embedding engine initialized (backend: {self._embedding_engine.backend})")
        
        # Initialize matcher
        self._matcher = CapabilityMatcher(match_threshold=self.config.min_score)
        
        # Initialize gradient table
        self._gradient_table = GradientTable(
            node_id=self.config.node_id,
            max_size=1000,
            expire_sec=self.config.stale_threshold_sec
        )
        
        # Build local capabilities
        await self._build_local_capabilities()
        
        # Initialize gossip protocol
        self._gossip = GossipProtocol(
            node_id=self.node_id,
            gradient_table=self._gradient_table,
            local_capabilities=self._capabilities
        )
        
        # Add peers
        for peer in self.config.peers:
            self._gossip.add_peer(peer.name, peer.url)
        
        # Initialize learner
        if self.config.learning_enabled:
            self._learner = get_route_learner()
        
        # Register agents
        for agent in self.config.agents:
            self._agents[agent.name] = agent
            logger.info(f"Registered agent: {agent.name}")
        
        # Start background tasks
        if self.config.gossip_interval_sec > 0:
            task = asyncio.create_task(self._gossip_loop())
            self._background_tasks.append(task)
        
        self._started = True
        logger.info(f"RouterService started with {len(self._capabilities)} capabilities, "
                   f"{len(self.config.peers)} peers, {len(self._agents)} agents")
    
    async def stop(self) -> None:
        """Stop the router service and cleanup."""
        if not self._started:
            return
        
        logger.info("Stopping RouterService")
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._background_tasks.clear()
        
        # Close embedding engine
        if self._embedding_engine:
            await self._embedding_engine.close()
        
        self._started = False
        logger.info("RouterService stopped")
    
    async def _build_local_capabilities(self) -> None:
        """Build capability objects from config."""
        self._capabilities = []
        
        for cap_cfg in self.config.local_capabilities:
            if isinstance(cap_cfg, dict):
                name = cap_cfg.get("name", "")
                description = cap_cfg.get("description", "")
                examples = cap_cfg.get("examples", [])
                handler = cap_cfg.get("handler", "")
            else:
                # Simple string capability
                name = cap_cfg
                description = cap_cfg
                examples = []
                handler = ""
            
            # Generate embedding
            cap_text = f"{name}: {description}"
            if examples:
                cap_text += f" Examples: {', '.join(examples[:3])}"
            
            vec = await self._embedding_engine.embed(cap_text)
            
            cap = Capability(
                id=f"{self.node_id}:{name}",
                label=name,
                description=description,
                vector=vec,
                handler=handler,
                models=cap_cfg.get("models", []) if isinstance(cap_cfg, dict) else []
            )
            
            self._capabilities.append(cap)
            logger.debug(f"Built capability: {name}")
        
        # Also add agent capabilities
        for agent in self.config.agents:
            for cap_name in agent.capabilities:
                cap_text = f"{cap_name}: Agent {agent.name} - {agent.description}"
                vec = await self._embedding_engine.embed(cap_text)
                
                cap = Capability(
                    id=f"{self.node_id}:agent:{agent.name}:{cap_name}",
                    label=cap_name,
                    description=f"Handled by agent {agent.name}",
                    vector=vec,
                    handler=f"agent:{agent.name}",
                    models=[]
                )
                
                self._capabilities.append(cap)
    
    async def _gossip_loop(self) -> None:
        """Background gossip loop."""
        while True:
            try:
                await asyncio.sleep(self.config.gossip_interval_sec)
                await self._gossip.announce()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Gossip error: {e}")
    
    async def route_intent(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> MatchResult:
        """
        Route an intent to the best matching capability.
        
        Args:
            text: Natural language intent
            context: Optional context (user info, session, etc.)
        
        Returns:
            MatchResult with routing decision
        """
        if not self._started:
            raise RuntimeError("RouterService not started")
        
        # Generate intent embedding
        intent_vec = await self._embedding_engine.embed(text)
        
        # Get gradient entries for remote routing
        gradient_entries = self._gradient_table.to_routing_tuples()
        
        # Match with routing
        result = self._matcher.match_with_routing(
            intent_vec,
            self._capabilities,
            gradient_entries
        )
        
        # Record in learner if enabled
        if self._learner and result.matched:
            route_id = result.capability.id if result.capability else f"remote:{result.next_hop}"
            # Note: actual success/failure recorded via feedback endpoint
        
        return result
    
    async def embed_text(self, text: str) -> list:
        """Generate embedding for text."""
        if not self._started:
            raise RuntimeError("RouterService not started")
        vec = await self._embedding_engine.embed(text)
        return vec.tolist()
    
    def get_agent(self, name: str) -> Optional[AgentConfig]:
        """Get agent configuration by name."""
        return self._agents.get(name)
    
    def list_agents(self) -> List[str]:
        """List registered agent names."""
        return list(self._agents.keys())
    
    def get_capabilities(self) -> List[Dict[str, Any]]:
        """Get all known capabilities (local + remote)."""
        caps = []
        
        # Local capabilities
        for cap in self._capabilities:
            caps.append({
                "id": cap.id,
                "label": cap.label,
                "description": cap.description,
                "handler": cap.handler,
                "local": True,
                "node": self.node_id
            })
        
        # Remote capabilities from gradient table
        if self._gradient_table:
            for entry in self._gradient_table:
                caps.append({
                    "id": entry.capability_id,
                    "label": entry.capability_label,
                    "description": "",
                    "handler": "",
                    "local": False,
                    "node": entry.via_node,
                    "hops": entry.hops,
                    "confidence": entry.confidence
                })
        
        return caps
    
    def stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "node_id": self.node_id,
            "display_name": self.display_name,
            "started": self._started,
            "embedding_backend": self._embedding_engine.backend if self._embedding_engine else None,
            "local_capabilities": len(self._capabilities),
            "gradient_table_size": len(self._gradient_table) if self._gradient_table else 0,
            "registered_agents": len(self._agents),
            "peers": len(self.config.peers),
            "gossip_stats": self._gossip.stats() if self._gossip else {}
        }


# Global service instance
_service: Optional[RouterService] = None


def get_router_service() -> Optional[RouterService]:
    """Get the global router service instance."""
    return _service


async def init_router_service(config: Dict[str, Any]) -> RouterService:
    """Initialize the global router service from config."""
    global _service
    if _service is not None:
        await _service.stop()
    
    _service = RouterService.from_config(config)
    await _service.start()
    return _service


async def shutdown_router_service() -> None:
    """Shutdown the global router service."""
    global _service
    if _service is not None:
        await _service.stop()
        _service = None
