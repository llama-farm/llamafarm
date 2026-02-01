"""
Tests for agent basics - lifecycle, memory, sessions.

Tests:
- Agent creation and initialization
- Memory management (short-term + long-term)
- Session lifecycle
- Task creation and tracking
"""

import pytest
from pathlib import Path
import tempfile

from agents import (
    AutonomousAgent,
    AgentMemory,
    Session,
    SessionManager,
    SessionStatus,
    Task,
    TaskStatus,
)


@pytest.fixture
def temp_memory_path():
    """Create a temporary path for memory storage."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        yield Path(f.name)


@pytest.fixture
def agent_memory(temp_memory_path):
    """Create an agent memory instance."""
    return AgentMemory(storage_path=temp_memory_path, max_short_term=10)


@pytest.fixture
async def session_manager():
    """Create a session manager."""
    return SessionManager()


def test_memory_creation(agent_memory):
    """Test that memory can be created."""
    assert agent_memory is not None
    assert agent_memory.max_short_term == 10


def test_add_observation(agent_memory):
    """Test adding observations to memory."""
    agent_memory.add_observation("User asked about weather")
    
    recent = agent_memory.get_recent(limit=1)
    assert len(recent) == 1
    assert recent[0].entry_type == "observation"
    assert "weather" in recent[0].content


def test_add_action(agent_memory):
    """Test adding actions to memory."""
    agent_memory.add_action("Calling weather API")
    
    recent = agent_memory.get_recent(limit=1)
    assert len(recent) == 1
    assert recent[0].entry_type == "action"


def test_add_thought(agent_memory):
    """Test adding thoughts to memory."""
    agent_memory.add_thought("Need to determine user's location")
    
    recent = agent_memory.get_recent(limit=1)
    assert len(recent) == 1
    assert recent[0].entry_type == "thought"


def test_add_result(agent_memory):
    """Test adding results to memory."""
    agent_memory.add_result("Weather: 72°F, sunny")
    
    recent = agent_memory.get_recent(limit=1)
    assert len(recent) == 1
    assert recent[0].entry_type == "result"


def test_memory_limit(agent_memory):
    """Test that short-term memory respects limit."""
    # Add more entries than the limit
    for i in range(15):
        agent_memory.add_observation(f"Observation {i}")
    
    recent = agent_memory.get_recent()
    # Should only keep max_short_term entries
    assert len(recent) <= agent_memory.max_short_term


def test_long_term_memory(agent_memory):
    """Test adding and retrieving long-term memories."""
    agent_memory.add_long_term("User prefers metric units")
    agent_memory.add_long_term("User timezone: America/New_York")
    
    # Long-term memories should be persistent
    assert len(agent_memory.long_term) >= 2


def test_memory_save_load(agent_memory, temp_memory_path):
    """Test saving and loading memory."""
    # Add some data
    agent_memory.add_observation("Test observation")
    agent_memory.add_long_term("Test long-term memory")
    
    # Save
    agent_memory.save()
    
    # Create new memory instance and load
    new_memory = AgentMemory(storage_path=temp_memory_path)
    
    # Should have loaded the data
    assert len(new_memory.get_recent()) > 0
    assert len(new_memory.long_term) > 0


@pytest.mark.asyncio
async def test_session_creation(session_manager):
    """Test creating a session."""
    session = await session_manager.create_session(
        session_id="test-001",
        metadata={"user": "test_user"}
    )
    
    assert session is not None
    assert session.session_id == "test-001"
    assert session.status == SessionStatus.ACTIVE
    assert session.metadata["user"] == "test_user"


@pytest.mark.asyncio
async def test_session_add_message(session_manager):
    """Test adding messages to a session."""
    session = await session_manager.create_session(session_id="test-002")
    
    await session.add_message("user", "Hello!")
    await session.add_message("assistant", "Hi there!")
    
    history = session.get_history()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hello!"
    assert history[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_session_history(session_manager):
    """Test retrieving session history."""
    session = await session_manager.create_session(session_id="test-003")
    
    messages = [
        ("user", "Message 1"),
        ("assistant", "Response 1"),
        ("user", "Message 2"),
        ("assistant", "Response 2"),
    ]
    
    for role, content in messages:
        await session.add_message(role, content)
    
    history = session.get_history()
    assert len(history) == 4
    
    # Check order is preserved
    for i, (role, content) in enumerate(messages):
        assert history[i]["role"] == role
        assert history[i]["content"] == content


@pytest.mark.asyncio
async def test_session_status_transition(session_manager):
    """Test session status transitions."""
    session = await session_manager.create_session(session_id="test-004")
    
    # Start active
    assert session.status == SessionStatus.ACTIVE
    
    # Pause
    session.status = SessionStatus.PAUSED
    assert session.status == SessionStatus.PAUSED
    
    # Resume
    session.status = SessionStatus.ACTIVE
    assert session.status == SessionStatus.ACTIVE
    
    # Complete
    session.status = SessionStatus.COMPLETED
    assert session.status == SessionStatus.COMPLETED


@pytest.mark.asyncio
async def test_session_metadata(session_manager):
    """Test session metadata."""
    metadata = {
        "user": "alice",
        "channel": "web",
        "language": "en",
        "preferences": {"theme": "dark"}
    }
    
    session = await session_manager.create_session(
        session_id="test-005",
        metadata=metadata
    )
    
    assert session.metadata["user"] == "alice"
    assert session.metadata["preferences"]["theme"] == "dark"
    
    # Update metadata
    session.metadata["preferences"]["theme"] = "light"
    assert session.metadata["preferences"]["theme"] == "light"


@pytest.mark.asyncio
async def test_session_manager_list_sessions(session_manager):
    """Test listing sessions."""
    # Create multiple sessions
    await session_manager.create_session(session_id="test-006")
    await session_manager.create_session(session_id="test-007")
    
    sessions = await session_manager.list_sessions()
    
    assert len(sessions) >= 2
    session_ids = [s.session_id for s in sessions]
    assert "test-006" in session_ids
    assert "test-007" in session_ids


def test_task_creation():
    """Test creating a task."""
    task = Task(
        id="task-001",
        description="Test task",
        priority=5
    )
    
    assert task.id == "task-001"
    assert task.description == "Test task"
    assert task.priority == 5
    assert task.status == TaskStatus.PENDING


def test_task_status_transitions():
    """Test task status transitions."""
    task = Task(id="task-002", description="Test")
    
    # Start pending
    assert task.status == TaskStatus.PENDING
    
    # Running
    task.status = TaskStatus.RUNNING
    assert task.status == TaskStatus.RUNNING
    
    # Completed
    task.status = TaskStatus.COMPLETED
    assert task.status == TaskStatus.COMPLETED


def test_task_with_parent():
    """Test task with parent relationship."""
    parent = Task(id="parent", description="Parent task", priority=8)
    child = Task(
        id="child",
        description="Child task",
        priority=7,
        parent_task="parent"
    )
    
    assert child.parent_task == "parent"


def test_task_result():
    """Test task result storage."""
    task = Task(id="task-003", description="Calculate something")
    
    # No result initially
    assert task.result is None
    
    # Set result
    task.result = "42"
    assert task.result == "42"


def test_task_metadata():
    """Test task metadata."""
    task = Task(
        id="task-004",
        description="Test",
        metadata={"source": "user_request", "urgency": "high"}
    )
    
    assert task.metadata["source"] == "user_request"
    assert task.metadata["urgency"] == "high"
    
    # Update metadata
    task.metadata["processed"] = True
    assert task.metadata["processed"] is True


def test_autonomous_agent_creation():
    """Test creating an autonomous agent."""
    config = {
        "agent_id": "test-agent-001",
        "name": "TestBot",
        "description": "A test agent",
        "model": "gpt-4o-mini"
    }
    
    agent = AutonomousAgent(config=config)
    
    assert agent is not None


def test_autonomous_agent_memory():
    """Test that agent has memory."""
    config = {"agent_id": "test-agent-002", "name": "TestBot"}
    agent = AutonomousAgent(config=config)
    
    assert agent.memory is not None
    
    # Add to memory
    agent.memory.add_observation("Test observation")
    recent = agent.memory.get_recent(limit=1)
    assert len(recent) == 1


def test_autonomous_agent_bind_session():
    """Test binding a session to an agent."""
    config = {"agent_id": "test-agent-003", "name": "TestBot"}
    agent = AutonomousAgent(config=config)
    
    session = Session(session_id="agent-test-session")
    agent.bind_session(session)
    
    # Session should be bound
    assert agent.session is not None
    assert agent.session.session_id == "agent-test-session"
