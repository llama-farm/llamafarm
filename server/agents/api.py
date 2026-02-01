"""
FastAPI endpoints for the OpenClaw Lite Agent Framework.

Provides REST API for:
- Agent management
- Session management
- Scheduler/cron jobs
- Skill discovery
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from .autonomous import (
    AutonomousAgent,
    get_or_create_agent,
    get_agent,
    list_agents,
)
from .sessions import (
    Session,
    SessionConfig,
    SessionManager,
    SessionStatus,
    get_session_manager,
)
from .scheduler import (
    AgentScheduler,
    CronJob,
    JobType,
    get_scheduler,
)
from .skills import (
    SkillRegistry,
    get_skill_registry,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/agents", tags=["agents"])


# --- Request/Response Models ---

class CreateAgentRequest(BaseModel):
    agent_id: str
    storage_path: Optional[str] = None
    max_iterations: int = 100


class CreateSessionRequest(BaseModel):
    agent_id: str
    session_key: Optional[str] = None
    label: Optional[str] = None
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    channel: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SendMessageRequest(BaseModel):
    content: str
    role: str = "user"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SessionSendRequest(BaseModel):
    message: str
    timeout_sec: float = 30.0


class AddTaskRequest(BaseModel):
    description: str
    priority: int = 5
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CreateJobRequest(BaseModel):
    name: str
    task: str
    job_type: str = "once"  # once, cron, interval
    delay_sec: Optional[int] = None
    cron_expr: Optional[str] = None
    interval_sec: Optional[int] = None
    agent_id: Optional[str] = None
    session_key: Optional[str] = None
    channel: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UpdateJobRequest(BaseModel):
    name: Optional[str] = None
    task: Optional[str] = None
    cron_expr: Optional[str] = None
    interval_sec: Optional[int] = None
    enabled: Optional[bool] = None


# --- Agent Endpoints ---

@router.get("/")
async def list_all_agents() -> Dict[str, Any]:
    """List all registered agents."""
    agents = list_agents()
    return {
        "agents": agents,
        "count": len(agents)
    }


@router.post("/")
async def create_agent(request: CreateAgentRequest) -> Dict[str, Any]:
    """Create or get an agent."""
    storage_path = Path(request.storage_path) if request.storage_path else None
    agent = get_or_create_agent(
        agent_id=request.agent_id,
        storage_path=storage_path,
        max_iterations=request.max_iterations
    )
    return agent.get_status()


@router.get("/{agent_id}")
async def get_agent_status(agent_id: str) -> Dict[str, Any]:
    """Get agent status."""
    agent = get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
    return agent.get_status()


@router.post("/{agent_id}/tasks")
async def add_task_to_agent(
    agent_id: str,
    request: AddTaskRequest
) -> Dict[str, Any]:
    """Add a task to an agent's queue."""
    agent = get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
    
    task = agent.add_task(
        description=request.description,
        priority=request.priority,
        metadata=request.metadata
    )
    return task.to_dict()


@router.post("/{agent_id}/start")
async def start_agent(agent_id: str) -> Dict[str, Any]:
    """Start the agent's autonomous loop."""
    agent = get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
    
    if agent.running:
        return {"status": "already_running", "agent_id": agent_id}
    
    import asyncio
    asyncio.create_task(agent.run())
    
    return {"status": "started", "agent_id": agent_id}


@router.post("/{agent_id}/stop")
async def stop_agent(agent_id: str) -> Dict[str, Any]:
    """Stop the agent's autonomous loop."""
    agent = get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")
    
    agent.stop()
    return {"status": "stopping", "agent_id": agent_id}


# --- Session Endpoints ---

@router.get("/sessions")
async def list_sessions(
    agent_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(default=100, le=1000)
) -> Dict[str, Any]:
    """List all sessions."""
    manager = get_session_manager()
    
    status_enum = None
    if status:
        try:
            status_enum = SessionStatus(status)
        except ValueError:
            pass
    
    sessions = manager.list_sessions(
        agent_id=agent_id,
        status=status_enum,
        limit=limit
    )
    
    return {
        "sessions": [s.to_dict() for s in sessions],
        "count": len(sessions)
    }


@router.post("/sessions")
async def create_session(request: CreateSessionRequest) -> Dict[str, Any]:
    """Create a new session."""
    manager = get_session_manager()
    
    config = SessionConfig(
        model=request.model,
        system_prompt=request.system_prompt,
        channel=request.channel,
        metadata=request.metadata
    )
    
    session = manager.create_session(
        agent_id=request.agent_id,
        session_key=request.session_key,
        config=config,
        label=request.label
    )
    
    return session.to_dict()


@router.get("/sessions/{session_key}")
async def get_session(session_key: str) -> Dict[str, Any]:
    """Get a session by key or label."""
    manager = get_session_manager()
    
    session = manager.get_session(session_key)
    if not session:
        session = manager.get_session_by_label(session_key)
    
    if not session:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_key}")
    
    return session.to_dict()


@router.get("/sessions/{session_key}/history")
async def get_session_history(
    session_key: str,
    limit: int = Query(default=50, le=500),
    include_system: bool = False
) -> Dict[str, Any]:
    """Get session message history."""
    manager = get_session_manager()
    
    session = manager.get_session(session_key)
    if not session:
        session = manager.get_session_by_label(session_key)
    
    if not session:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_key}")
    
    messages = session.get_history(limit=limit, include_system=include_system)
    
    return {
        "session_key": session.session_key,
        "messages": [m.to_dict() for m in messages],
        "count": len(messages)
    }


@router.post("/sessions/{session_key}/messages")
async def add_session_message(
    session_key: str,
    request: SendMessageRequest
) -> Dict[str, Any]:
    """Add a message to a session."""
    manager = get_session_manager()
    
    session = manager.get_session(session_key)
    if not session:
        session = manager.get_session_by_label(session_key)
    
    if not session:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_key}")
    
    message = session.add_message(
        role=request.role,
        content=request.content,
        metadata=request.metadata
    )
    
    return message.to_dict()


@router.post("/sessions/{session_key}/send")
async def send_to_session(
    session_key: str,
    request: SessionSendRequest
) -> Dict[str, Any]:
    """Send a message to a session and get response."""
    manager = get_session_manager()
    
    response = await manager.send_to_session(
        target_key=session_key,
        message=request.message,
        timeout_sec=request.timeout_sec
    )
    
    return {
        "session_key": session_key,
        "response": response
    }


@router.delete("/sessions/{session_key}")
async def delete_session(session_key: str) -> Dict[str, Any]:
    """Delete a session."""
    manager = get_session_manager()
    
    if manager.delete_session(session_key):
        return {"status": "deleted", "session_key": session_key}
    
    raise HTTPException(status_code=404, detail=f"Session not found: {session_key}")


# --- Scheduler Endpoints ---

@router.get("/cron")
async def list_jobs(
    include_disabled: bool = False,
    agent_id: Optional[str] = None
) -> Dict[str, Any]:
    """List all scheduled jobs."""
    scheduler = get_scheduler()
    
    jobs = scheduler.list_jobs(
        include_disabled=include_disabled,
        agent_id=agent_id
    )
    
    return {
        "jobs": [j.to_dict() for j in jobs],
        "count": len(jobs)
    }


@router.post("/cron")
async def create_job(request: CreateJobRequest) -> Dict[str, Any]:
    """Create a scheduled job."""
    scheduler = get_scheduler()
    
    job_type = JobType(request.job_type)
    
    job = scheduler.add_job(
        name=request.name,
        task=request.task,
        job_type=job_type,
        delay_sec=request.delay_sec,
        cron_expr=request.cron_expr,
        interval_sec=request.interval_sec,
        agent_id=request.agent_id,
        session_key=request.session_key,
        channel=request.channel,
        metadata=request.metadata
    )
    
    return job.to_dict()


@router.get("/cron/{job_id}")
async def get_job(job_id: str) -> Dict[str, Any]:
    """Get a job by ID."""
    scheduler = get_scheduler()
    
    job = scheduler.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    return job.to_dict()


@router.patch("/cron/{job_id}")
async def update_job(
    job_id: str,
    request: UpdateJobRequest
) -> Dict[str, Any]:
    """Update a job."""
    scheduler = get_scheduler()
    
    updates = request.model_dump(exclude_none=True)
    job = scheduler.update_job(job_id, **updates)
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    return job.to_dict()


@router.delete("/cron/{job_id}")
async def remove_job(job_id: str) -> Dict[str, Any]:
    """Remove a job."""
    scheduler = get_scheduler()
    
    if scheduler.remove_job(job_id):
        return {"status": "deleted", "job_id": job_id}
    
    raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")


@router.post("/cron/{job_id}/run")
async def run_job_now(job_id: str) -> Dict[str, Any]:
    """Run a job immediately."""
    scheduler = get_scheduler()
    
    run = await scheduler.run_job(job_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    return run.to_dict()


@router.get("/cron/{job_id}/runs")
async def get_job_runs(
    job_id: str,
    limit: int = Query(default=20, le=100)
) -> Dict[str, Any]:
    """Get job run history."""
    scheduler = get_scheduler()
    
    runs = scheduler.get_runs(job_id=job_id, limit=limit)
    
    return {
        "job_id": job_id,
        "runs": [r.to_dict() for r in runs],
        "count": len(runs)
    }


# --- Skills Endpoints ---

@router.get("/skills")
async def list_skills(enabled_only: bool = True) -> Dict[str, Any]:
    """List all registered skills."""
    registry = get_skill_registry()
    
    skills = registry.list_skills(enabled_only=enabled_only)
    
    return {
        "skills": [s.to_dict() for s in skills],
        "count": len(skills)
    }


@router.get("/skills/{skill_name}")
async def get_skill(skill_name: str) -> Dict[str, Any]:
    """Get a skill by name."""
    registry = get_skill_registry()
    
    skill = registry.get_skill(skill_name)
    if not skill:
        raise HTTPException(status_code=404, detail=f"Skill not found: {skill_name}")
    
    return skill.to_dict()


@router.get("/skills/{skill_name}/tools")
async def get_skill_tools(skill_name: str) -> Dict[str, Any]:
    """Get tools for a skill."""
    registry = get_skill_registry()
    
    skill = registry.get_skill(skill_name)
    if not skill:
        raise HTTPException(status_code=404, detail=f"Skill not found: {skill_name}")
    
    return {
        "skill": skill_name,
        "tools": skill.get_tools_for_openai()
    }


@router.get("/tools")
async def list_all_tools() -> Dict[str, Any]:
    """List all tools from all skills."""
    registry = get_skill_registry()
    
    tools = registry.get_tools_for_openai()
    
    return {
        "tools": tools,
        "count": len(tools)
    }


# --- Health Endpoint ---

@router.get("/health")
async def agent_framework_health() -> Dict[str, Any]:
    """Get agent framework health status."""
    manager = get_session_manager()
    scheduler = get_scheduler()
    registry = get_skill_registry()
    
    return {
        "status": "healthy",
        "agents": len(list_agents()),
        "sessions": len(manager.list_sessions()),
        "jobs": len(scheduler.list_jobs()),
        "skills": len(registry.list_skills()),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
