"""
OpenClaw Lite Agent Framework for LlamaFarm.

This module provides a lightweight agent framework with:
- Agent loops with tool calling
- Memory (short-term + long-term)
- Sessions and instances
- Cron/scheduled tasks
- Channels (messaging surfaces)
- Skills (modular capabilities)

Architecture:
    ┌─────────────────────────────────────────────┐
    │              Agent Framework                │
    ├─────────────────────────────────────────────┤
    │  ┌─────────┐  ┌─────────┐  ┌──────────┐    │
    │  │ Sessions│  │Scheduler│  │ Channels │    │
    │  └─────────┘  └─────────┘  └──────────┘    │
    │  ┌─────────┐  ┌─────────┐  ┌──────────┐    │
    │  │ Skills  │  │ Memory  │  │  Router  │    │
    │  └─────────┘  └─────────┘  └──────────┘    │
    └─────────────────────────────────────────────┘
"""

from .autonomous import (
    AutonomousAgent,
    AgentMemory,
    Task,
    TaskStatus,
    MemoryEntry,
    get_or_create_agent,
    get_agent,
    list_agents,
)

from .sessions import (
    Session,
    SessionConfig,
    SessionManager,
    SessionMessage,
    SessionStatus,
    get_session_manager,
)

from .scheduler import (
    AgentScheduler,
    CronJob,
    JobRun,
    JobStatus,
    JobType,
    get_scheduler,
)

__all__ = [
    # Autonomous Agent
    "AutonomousAgent",
    "AgentMemory",
    "Task",
    "TaskStatus",
    "MemoryEntry",
    "get_or_create_agent",
    "get_agent",
    "list_agents",
    
    # Sessions
    "Session",
    "SessionConfig",
    "SessionManager",
    "SessionMessage",
    "SessionStatus",
    "get_session_manager",
    
    # Scheduler
    "AgentScheduler",
    "CronJob",
    "JobRun",
    "JobStatus",
    "JobType",
    "get_scheduler",
]
