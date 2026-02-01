#!/usr/bin/env python3
"""
OpenClaw Lite Agent Basics Demo

Demonstrates:
- Creating an autonomous agent
- Managing memory (short-term + long-term)
- Session management
- Task tracking
"""

import asyncio
import json
from pathlib import Path

from agents import (
    AutonomousAgent,
    AgentMemory,
    Session,
    SessionManager,
    Task,
    TaskStatus,
)


async def demo_agent_memory():
    """Demo: Agent memory management."""
    print("\n=== Agent Memory Demo ===")
    
    # Create memory with storage
    memory_path = Path("/tmp/demo_agent_memory.json")
    memory = AgentMemory(storage_path=memory_path, max_short_term=10)
    
    # Add some observations
    memory.add_observation("User asked about the weather")
    memory.add_action("Checking weather API")
    memory.add_thought("Need to determine user's location first")
    memory.add_result("Weather in NYC: 72°F, sunny")
    
    # Get recent history
    recent = memory.get_recent(limit=4)
    print(f"\n Recent memory ({len(recent)} entries):")
    for entry in recent:
        print(f"  [{entry.entry_type}] {entry.content}")
    
    # Add long-term memory
    memory.add_long_term("User prefers NYC for weather queries")
    
    # Save memory
    memory.save()
    print(f"\n Memory saved to: {memory_path}")
    
    return memory


async def demo_session_management():
    """Demo: Session lifecycle."""
    print("\n\n=== Session Management Demo ===")
    
    manager = SessionManager()
    
    # Create a new session
    session = await manager.create_session(
        session_id="demo-session-001",
        metadata={
            "user": "demo_user",
            "channel": "cli",
            "created": "2026-02-01"
        }
    )
    
    print(f"\n Created session: {session.session_id}")
    print(f" Status: {session.status.value}")
    print(f" Metadata: {json.dumps(session.metadata, indent=2)}")
    
    # Add messages to session
    await session.add_message("user", "Hello, agent!")
    await session.add_message("assistant", "Hello! How can I help you today?")
    await session.add_message("user", "What's the weather like?")
    
    # Get conversation history
    history = session.get_history()
    print(f"\n Conversation history ({len(history)} messages):")
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        print(f"  {role}: {content}")
    
    # List all sessions
    all_sessions = await manager.list_sessions()
    print(f"\n Total active sessions: {len(all_sessions)}")
    
    return session


async def demo_task_management():
    """Demo: Creating and tracking tasks."""
    print("\n\n=== Task Management Demo ===")
    
    # Create some tasks
    task1 = Task(
        id="task-001",
        description="Research weather API options",
        priority=8
    )
    
    task2 = Task(
        id="task-002",
        description="Implement weather skill",
        priority=7,
        parent_task="task-001"
    )
    
    task3 = Task(
        id="task-003",
        description="Test weather integration",
        priority=5,
        parent_task="task-002"
    )
    
    tasks = [task1, task2, task3]
    
    print("\n Created tasks:")
    for task in tasks:
        print(f"  [{task.id}] {task.description}")
        print(f"    Priority: {task.priority}, Status: {task.status.value}")
        if task.parent_task:
            print(f"    Parent: {task.parent_task}")
    
    # Simulate task progression
    print("\n Simulating task execution...")
    task1.status = TaskStatus.RUNNING
    await asyncio.sleep(0.5)
    task1.status = TaskStatus.COMPLETED
    task1.result = "Selected OpenWeatherMap API"
    
    task2.status = TaskStatus.RUNNING
    print(f"  Task {task2.id}: {task2.status.value}")
    
    # Delegate task 3
    task3.status = TaskStatus.DELEGATED
    task3.metadata["delegated_to"] = "specialist-agent"
    
    print("\n Final task states:")
    for task in tasks:
        print(f"  [{task.id}] {task.status.value}")
        if task.result:
            print(f"    Result: {task.result}")
        if task.status == TaskStatus.DELEGATED:
            delegated_to = task.metadata.get("delegated_to")
            print(f"    Delegated to: {delegated_to}")
    
    return tasks


async def demo_autonomous_agent():
    """Demo: Creating an autonomous agent."""
    print("\n\n=== Autonomous Agent Demo ===")
    
    # Create agent configuration
    agent_config = {
        "agent_id": "demo-agent-001",
        "name": "DemoBot",
        "description": "A demonstration autonomous agent",
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_iterations": 5
    }
    
    # Create agent
    agent = AutonomousAgent(config=agent_config)
    
    print(f"\n Created agent: {agent_config['name']}")
    print(f" Agent ID: {agent_config['agent_id']}")
    print(f" Model: {agent_config['model']}")
    
    # Create and bind session
    session = Session(session_id="agent-demo-session")
    agent.bind_session(session)
    
    print(f" Bound to session: {session.session_id}")
    
    # Add some context to agent memory
    agent.memory.add_observation("Agent initialized and ready")
    agent.memory.add_long_term("Agent purpose: Demonstrate OpenClaw Lite capabilities")
    
    print("\n Agent memory initialized:")
    for entry in agent.memory.get_recent(limit=5):
        print(f"  - {entry.content}")
    
    return agent


async def main():
    """Run all demos."""
    print("OpenClaw Lite Agent Framework Demo")
    print("=" * 60)
    
    # Run demos
    memory = await demo_agent_memory()
    session = await demo_session_management()
    tasks = await demo_task_management()
    agent = await demo_autonomous_agent()
    
    print("\n\n" + "=" * 60)
    print("Demo completed successfully! ✅")
    print("\nKey takeaways:")
    print("  • Agent memory persists observations, actions, and learnings")
    print("  • Sessions manage conversation state across interactions")
    print("  • Tasks can be hierarchical and delegated")
    print("  • Autonomous agents combine all components")


if __name__ == "__main__":
    asyncio.run(main())
