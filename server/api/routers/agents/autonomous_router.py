"""
API endpoints for autonomous agents.

Provides REST API for:
- Creating and managing autonomous agents
- Adding tasks to agent queues
- Monitoring agent status
- Controlling agent execution
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from pathlib import Path
import logging

from agents.autonomous import (
    AutonomousAgent,
    get_or_create_agent,
    get_agent,
    list_agents,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agents", tags=["Autonomous Agents"])


class CreateAgentRequest(BaseModel):
    agent_id: str = Field(..., description="Unique identifier for the agent")
    storage_path: Optional[str] = Field(None, description="Path for memory persistence")
    max_iterations: int = Field(100, description="Maximum iterations before stopping")
    think_budget: int = Field(5, description="Max thinking steps per iteration")


class CreateAgentResponse(BaseModel):
    agent_id: str
    status: str
    message: str


class AddTaskRequest(BaseModel):
    description: str = Field(..., description="Task description")
    priority: int = Field(5, ge=1, le=10, description="Priority (1-10, higher = more urgent)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class AddTaskResponse(BaseModel):
    task_id: str
    agent_id: str
    description: str
    priority: int
    status: str


class AgentStatusResponse(BaseModel):
    agent_id: str
    running: bool
    current_task: Optional[Dict[str, Any]]
    queue_length: int
    completed_tasks: int
    memory_size: Dict[str, int]


class MemoryEntry(BaseModel):
    timestamp: float
    content: str
    entry_type: str
    metadata: Dict[str, Any]


class AgentMemoryResponse(BaseModel):
    agent_id: str
    context: str
    facts: Dict[str, str]
    recent_entries: List[Dict[str, Any]]


@router.post("/create", response_model=CreateAgentResponse)
async def create_agent(request: CreateAgentRequest):
    """Create a new autonomous agent."""
    try:
        storage = Path(request.storage_path) if request.storage_path else None
        agent = get_or_create_agent(
            agent_id=request.agent_id,
            storage_path=storage,
            max_iterations=request.max_iterations,
            think_budget=request.think_budget
        )
        return CreateAgentResponse(
            agent_id=agent.agent_id,
            status="created",
            message=f"Agent {agent.agent_id} created successfully"
        )
    except Exception as e:
        logger.error(f"Failed to create agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=List[str])
async def list_all_agents():
    """List all registered agents."""
    return list_agents()


@router.get("/{agent_id}/status", response_model=AgentStatusResponse)
async def get_agent_status(agent_id: str):
    """Get the status of an agent."""
    agent = get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    status = agent.get_status()
    return AgentStatusResponse(**status)


@router.post("/{agent_id}/tasks", response_model=AddTaskResponse)
async def add_task(agent_id: str, request: AddTaskRequest):
    """Add a task to an agent's queue."""
    agent = get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    task = agent.add_task(
        description=request.description,
        priority=request.priority,
        metadata=request.metadata
    )
    
    return AddTaskResponse(
        task_id=task.id,
        agent_id=agent_id,
        description=task.description,
        priority=task.priority,
        status=task.status.value
    )


@router.post("/{agent_id}/start")
async def start_agent(agent_id: str, background_tasks: BackgroundTasks):
    """Start an agent's autonomous loop."""
    agent = get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    if agent.running:
        return {"status": "already_running", "agent_id": agent_id}
    
    background_tasks.add_task(agent.run)
    return {"status": "started", "agent_id": agent_id}


@router.post("/{agent_id}/stop")
async def stop_agent(agent_id: str):
    """Stop an agent's autonomous loop."""
    agent = get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    if not agent.running:
        return {"status": "not_running", "agent_id": agent_id}
    
    agent.stop()
    return {"status": "stopping", "agent_id": agent_id}


@router.get("/{agent_id}/memory", response_model=AgentMemoryResponse)
async def get_agent_memory(agent_id: str, max_entries: int = 20):
    """Get an agent's memory context."""
    agent = get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    recent = agent.memory.short_term[-max_entries:]
    return AgentMemoryResponse(
        agent_id=agent_id,
        context=agent.memory.get_context(max_entries=max_entries),
        facts=agent.memory.facts,
        recent_entries=[e.to_dict() for e in recent]
    )


@router.post("/{agent_id}/memory/fact")
async def set_agent_fact(agent_id: str, key: str, value: str):
    """Set a fact in an agent's memory."""
    agent = get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    agent.memory.set_fact(key, value)
    return {"status": "ok", "key": key, "value": value}


@router.post("/{agent_id}/memory/memorize")
async def memorize(agent_id: str, content: str):
    """Add something to an agent's long-term memory."""
    agent = get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    agent.memory.memorize(content)
    return {"status": "memorized", "content": content[:100] + "..." if len(content) > 100 else content}


@router.get("/{agent_id}/tasks")
async def get_agent_tasks(agent_id: str):
    """Get an agent's task queue and history."""
    agent = get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    return {
        "agent_id": agent_id,
        "current_task": agent.current_task.to_dict() if agent.current_task else None,
        "queue": [t.to_dict() for t in agent.task_queue],
        "history": [t.to_dict() for t in agent.memory.task_history[-20:]]
    }
