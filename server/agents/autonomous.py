"""
Autonomous Agent Loop with Memory and Router Integration.

Provides an autonomous agent that can:
- Execute tasks with memory persistence
- Delegate subtasks via semantic router
- Maintain conversation history across sessions
- Schedule follow-up tasks
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DELEGATED = "delegated"


@dataclass
class MemoryEntry:
    """A single memory entry with timestamp and metadata."""
    timestamp: float
    content: str
    entry_type: str = "observation"  # observation, action, thought, result
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "content": self.content,
            "type": self.entry_type,
            "metadata": self.metadata,
            "datetime": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat()
        }


@dataclass
class Task:
    """A task to be executed by the agent."""
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 5  # 1-10, higher = more urgent
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    result: Optional[str] = None
    error: Optional[str] = None
    subtasks: List[str] = field(default_factory=list)
    parent_task: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
            "subtasks": self.subtasks,
            "parent_task": self.parent_task,
            "metadata": self.metadata
        }


class AgentMemory:
    """
    Persistent memory for an autonomous agent.
    
    Stores:
    - Short-term memory (recent observations, actions)
    - Long-term memory (important facts, learned patterns)
    - Task history
    """
    
    def __init__(
        self,
        storage_path: Optional[Path] = None,
        max_short_term: int = 100,
        max_long_term: int = 1000
    ):
        self.storage_path = storage_path
        self.max_short_term = max_short_term
        self.max_long_term = max_long_term
        
        self.short_term: List[MemoryEntry] = []
        self.long_term: List[MemoryEntry] = []
        self.facts: Dict[str, str] = {}  # key -> value facts
        self.task_history: List[Task] = []
        
        if storage_path and storage_path.exists():
            self._load()
    
    def add_observation(self, content: str, metadata: Optional[Dict] = None) -> None:
        """Add an observation to short-term memory."""
        entry = MemoryEntry(
            timestamp=time.time(),
            content=content,
            entry_type="observation",
            metadata=metadata or {}
        )
        self.short_term.append(entry)
        self._trim_short_term()
    
    def add_action(self, content: str, metadata: Optional[Dict] = None) -> None:
        """Record an action taken."""
        entry = MemoryEntry(
            timestamp=time.time(),
            content=content,
            entry_type="action",
            metadata=metadata or {}
        )
        self.short_term.append(entry)
        self._trim_short_term()
    
    def add_thought(self, content: str, metadata: Optional[Dict] = None) -> None:
        """Record a thought/reasoning step."""
        entry = MemoryEntry(
            timestamp=time.time(),
            content=content,
            entry_type="thought",
            metadata=metadata or {}
        )
        self.short_term.append(entry)
        self._trim_short_term()
    
    def add_result(self, content: str, metadata: Optional[Dict] = None) -> None:
        """Record a result from an action."""
        entry = MemoryEntry(
            timestamp=time.time(),
            content=content,
            entry_type="result",
            metadata=metadata or {}
        )
        self.short_term.append(entry)
        self._trim_short_term()
    
    def memorize(self, content: str, metadata: Optional[Dict] = None) -> None:
        """Add something important to long-term memory."""
        entry = MemoryEntry(
            timestamp=time.time(),
            content=content,
            entry_type="memorized",
            metadata=metadata or {}
        )
        self.long_term.append(entry)
        self._trim_long_term()
        self._save()
    
    def set_fact(self, key: str, value: str) -> None:
        """Store a key-value fact."""
        self.facts[key] = value
        self._save()
    
    def get_fact(self, key: str) -> Optional[str]:
        """Retrieve a stored fact."""
        return self.facts.get(key)
    
    def get_context(self, max_entries: int = 20) -> str:
        """Get recent memory context for the agent."""
        entries = self.short_term[-max_entries:]
        lines = []
        for entry in entries:
            dt = datetime.fromtimestamp(entry.timestamp, tz=timezone.utc)
            lines.append(f"[{dt.strftime('%H:%M:%S')}] [{entry.entry_type.upper()}] {entry.content}")
        return "\n".join(lines)
    
    def get_relevant_memories(self, query: str, max_results: int = 5) -> List[MemoryEntry]:
        """
        Get memories relevant to a query.
        
        TODO: Implement semantic search using router embeddings.
        For now, simple keyword matching.
        """
        query_lower = query.lower()
        scored = []
        
        for entry in self.long_term + self.short_term:
            # Simple relevance score based on keyword overlap
            content_lower = entry.content.lower()
            words = query_lower.split()
            score = sum(1 for word in words if word in content_lower)
            if score > 0:
                scored.append((score, entry))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:max_results]]
    
    def _trim_short_term(self) -> None:
        if len(self.short_term) > self.max_short_term:
            # Move oldest to long-term if important
            overflow = self.short_term[:-self.max_short_term]
            for entry in overflow:
                if entry.entry_type in ("memorized", "result"):
                    self.long_term.append(entry)
            self.short_term = self.short_term[-self.max_short_term:]
    
    def _trim_long_term(self) -> None:
        if len(self.long_term) > self.max_long_term:
            self.long_term = self.long_term[-self.max_long_term:]
    
    def _save(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "short_term": [e.to_dict() for e in self.short_term],
            "long_term": [e.to_dict() for e in self.long_term],
            "facts": self.facts,
            "task_history": [t.to_dict() for t in self.task_history[-100:]]
        }
        self.storage_path.write_text(json.dumps(data, indent=2))
    
    def _load(self) -> None:
        if not self.storage_path or not self.storage_path.exists():
            return
        try:
            data = json.loads(self.storage_path.read_text())
            self.short_term = [
                MemoryEntry(
                    timestamp=e["timestamp"],
                    content=e["content"],
                    entry_type=e.get("type", "observation"),
                    metadata=e.get("metadata", {})
                )
                for e in data.get("short_term", [])
            ]
            self.long_term = [
                MemoryEntry(
                    timestamp=e["timestamp"],
                    content=e["content"],
                    entry_type=e.get("type", "observation"),
                    metadata=e.get("metadata", {})
                )
                for e in data.get("long_term", [])
            ]
            self.facts = data.get("facts", {})
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")


class AutonomousAgent:
    """
    An autonomous agent that executes tasks with memory and router integration.
    
    Features:
    - Task queue with priority scheduling
    - Memory persistence across sessions
    - Semantic router for capability delegation
    - Observe-Think-Act loop
    """
    
    def __init__(
        self,
        agent_id: str,
        memory: Optional[AgentMemory] = None,
        router_service: Optional[Any] = None,  # RouterService
        llm_client: Optional[Any] = None,  # LFAgentClient
        max_iterations: int = 100,
        think_budget: int = 5  # Max think steps per iteration
    ):
        self.agent_id = agent_id
        self.memory = memory or AgentMemory()
        self.router_service = router_service
        self.llm_client = llm_client
        self.max_iterations = max_iterations
        self.think_budget = think_budget
        
        self.task_queue: List[Task] = []
        self.current_task: Optional[Task] = None
        self.running = False
        self._stop_flag = False
    
    def add_task(
        self,
        description: str,
        priority: int = 5,
        metadata: Optional[Dict] = None
    ) -> Task:
        """Add a task to the queue."""
        import uuid
        task = Task(
            id=f"task-{uuid.uuid4().hex[:8]}",
            description=description,
            priority=priority,
            metadata=metadata or {}
        )
        self.task_queue.append(task)
        self.task_queue.sort(key=lambda t: t.priority, reverse=True)
        logger.info(f"Task added: {task.id} - {description[:50]}...")
        return task
    
    async def run(self) -> None:
        """
        Main agent loop.
        
        Continuously processes tasks from the queue using
        an Observe-Think-Act cycle.
        """
        self.running = True
        self._stop_flag = False
        iteration = 0
        
        logger.info(f"Agent {self.agent_id} starting autonomous loop")
        self.memory.add_observation(f"Agent loop started at {datetime.now(timezone.utc).isoformat()}")
        
        try:
            while not self._stop_flag and iteration < self.max_iterations:
                iteration += 1
                
                # Get next task
                if not self.current_task:
                    if not self.task_queue:
                        logger.debug("No tasks in queue, waiting...")
                        await asyncio.sleep(1)
                        continue
                    
                    self.current_task = self.task_queue.pop(0)
                    self.current_task.status = TaskStatus.RUNNING
                    self.memory.add_observation(f"Starting task: {self.current_task.description}")
                
                # Execute task iteration
                try:
                    completed = await self._execute_iteration()
                    if completed:
                        self.current_task.status = TaskStatus.COMPLETED
                        self.current_task.completed_at = time.time()
                        self.memory.task_history.append(self.current_task)
                        self.memory.add_result(f"Task completed: {self.current_task.result}")
                        self.current_task = None
                        
                except Exception as e:
                    logger.error(f"Task iteration failed: {e}")
                    self.memory.add_observation(f"Error: {str(e)}")
                    if self.current_task:
                        self.current_task.error = str(e)
                        # Retry logic could go here
                
                # Small delay between iterations
                await asyncio.sleep(0.1)
                
        finally:
            self.running = False
            logger.info(f"Agent {self.agent_id} stopped after {iteration} iterations")
    
    async def _execute_iteration(self) -> bool:
        """
        Execute one iteration of the Observe-Think-Act cycle.
        
        Returns True if the current task is complete.
        """
        if not self.current_task:
            return False
        
        # OBSERVE: Gather context
        context = self.memory.get_context(max_entries=10)
        relevant = self.memory.get_relevant_memories(self.current_task.description, max_results=3)
        
        # THINK: Decide what to do
        thought = await self._think(context, relevant)
        self.memory.add_thought(thought)
        
        # ACT: Execute the decided action
        action_result = await self._act(thought)
        self.memory.add_action(action_result.get("action", "unknown"))
        
        if action_result.get("result"):
            self.memory.add_result(action_result["result"])
        
        # Check if task is complete
        if action_result.get("complete"):
            self.current_task.result = action_result.get("result", "Completed")
            return True
        
        return False
    
    async def _think(self, context: str, relevant_memories: List[MemoryEntry]) -> str:
        """
        Think about what to do next.
        
        Uses LLM to reason about the current task given context.
        """
        if not self.llm_client:
            # Fallback: simple rule-based thinking
            return f"Proceeding with task: {self.current_task.description}"
        
        relevant_text = "\n".join([
            f"- {m.content}" for m in relevant_memories
        ]) if relevant_memories else "No relevant memories."
        
        prompt = f"""You are an autonomous agent working on a task.

CURRENT TASK: {self.current_task.description}

RECENT CONTEXT:
{context}

RELEVANT MEMORIES:
{relevant_text}

Think step-by-step about what action to take next. Consider:
1. What information do I need?
2. What capability should I use?
3. How do I know when the task is complete?

Respond with your reasoning and the next action to take."""

        # TODO: Call LLM client
        # For now, return placeholder
        return f"Analyzing task: {self.current_task.description[:50]}..."
    
    async def _act(self, thought: str) -> Dict[str, Any]:
        """
        Execute an action based on the thinking step.
        
        May delegate to router for capability-specific tasks.
        """
        # Check if we should delegate via router
        if self.router_service and "delegate" in thought.lower():
            try:
                match = await self.router_service.route_intent(thought)
                if match and match.get("action") == "route_forward":
                    return {
                        "action": "delegated",
                        "result": f"Delegated to {match.get('peer', 'unknown')}",
                        "complete": False
                    }
            except Exception as e:
                logger.warning(f"Router delegation failed: {e}")
        
        # Default: mark as needing manual implementation
        # In a real implementation, this would parse the thought
        # and execute appropriate tools
        return {
            "action": "pending",
            "result": f"Action pending for: {thought[:50]}...",
            "complete": True  # Mark complete for now
        }
    
    def stop(self) -> None:
        """Signal the agent to stop."""
        self._stop_flag = True
        logger.info(f"Stop signal sent to agent {self.agent_id}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {
            "agent_id": self.agent_id,
            "running": self.running,
            "current_task": self.current_task.to_dict() if self.current_task else None,
            "queue_length": len(self.task_queue),
            "completed_tasks": len(self.memory.task_history),
            "memory_size": {
                "short_term": len(self.memory.short_term),
                "long_term": len(self.memory.long_term),
                "facts": len(self.memory.facts)
            }
        }


# Global agent registry
_agents: Dict[str, AutonomousAgent] = {}


def get_or_create_agent(
    agent_id: str,
    storage_path: Optional[Path] = None,
    **kwargs
) -> AutonomousAgent:
    """Get or create an autonomous agent."""
    if agent_id not in _agents:
        memory = AgentMemory(storage_path=storage_path) if storage_path else AgentMemory()
        _agents[agent_id] = AutonomousAgent(agent_id=agent_id, memory=memory, **kwargs)
    return _agents[agent_id]


def list_agents() -> List[str]:
    """List all registered agents."""
    return list(_agents.keys())


def get_agent(agent_id: str) -> Optional[AutonomousAgent]:
    """Get an agent by ID."""
    return _agents.get(agent_id)
