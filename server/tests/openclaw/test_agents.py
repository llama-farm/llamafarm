"""Tests for OpenClaw Lite agent framework."""
import pytest
from agents import AutonomousAgent, Session, AgentScheduler, AgentMemory
from agents.skills import SkillRegistry


class TestAgentMemory:
    """Test AgentMemory storage and retrieval."""
    
    def test_memory_initialization(self):
        """Test memory initializes with empty stores."""
        memory = AgentMemory()
        assert memory.short_term == []
        assert memory.long_term == {}
        assert memory.context == {}
    
    def test_add_to_short_term(self):
        """Test adding items to short-term memory."""
        memory = AgentMemory()
        memory.add_short_term("user", "Hello")
        assert len(memory.short_term) == 1
        assert memory.short_term[0]["role"] == "user"
        assert memory.short_term[0]["content"] == "Hello"
    
    def test_short_term_max_length(self):
        """Test short-term memory maintains max length."""
        memory = AgentMemory(max_short_term=3)
        for i in range(5):
            memory.add_short_term("user", f"Message {i}")
        assert len(memory.short_term) == 3
        assert memory.short_term[0]["content"] == "Message 2"
    
    def test_long_term_storage(self):
        """Test long-term memory key-value storage."""
        memory = AgentMemory()
        memory.store_long_term("user_name", "Alice")
        memory.store_long_term("user_prefs", {"theme": "dark"})
        
        assert memory.recall_long_term("user_name") == "Alice"
        assert memory.recall_long_term("user_prefs") == {"theme": "dark"}
        assert memory.recall_long_term("nonexistent") is None


class TestSession:
    """Test Session management."""
    
    def test_session_creation(self):
        """Test creating a new session."""
        session = Session(session_id="test-123")
        assert session.session_id == "test-123"
        assert session.memory is not None
        assert session.state == "active"
    
    def test_session_with_metadata(self):
        """Test session with custom metadata."""
        metadata = {"user": "alice", "channel": "slack"}
        session = Session(session_id="test-456", metadata=metadata)
        assert session.metadata["user"] == "alice"
        assert session.metadata["channel"] == "slack"
    
    def test_session_message_history(self):
        """Test session maintains message history."""
        session = Session(session_id="test-789")
        session.add_message("user", "What's the weather?")
        session.add_message("assistant", "It's sunny!")
        
        history = session.get_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"


class TestAutonomousAgent:
    """Test AutonomousAgent core functionality."""
    
    def test_agent_initialization(self):
        """Test agent initializes with config."""
        config = {
            "name": "TestAgent",
            "model": "gpt-4",
            "temperature": 0.7
        }
        agent = AutonomousAgent(config=config)
        assert agent.config["name"] == "TestAgent"
        assert agent.config["model"] == "gpt-4"
    
    def test_agent_with_skills(self):
        """Test agent with skill registry."""
        registry = SkillRegistry()
        registry.register("search", lambda q: f"Results for {q}")
        
        agent = AutonomousAgent(
            config={"name": "SearchAgent"},
            skills=registry
        )
        assert agent.skills is not None
        assert "search" in agent.skills.list_skills()
    
    def test_agent_session_binding(self):
        """Test binding agent to session."""
        agent = AutonomousAgent(config={"name": "BoundAgent"})
        session = Session(session_id="bound-123")
        
        agent.bind_session(session)
        assert agent.session == session
        assert agent.session.session_id == "bound-123"


class TestAgentScheduler:
    """Test AgentScheduler for cron/background jobs."""
    
    def test_scheduler_initialization(self):
        """Test scheduler initializes empty."""
        scheduler = AgentScheduler()
        assert len(scheduler.jobs) == 0
    
    def test_add_cron_job(self):
        """Test adding a cron job."""
        scheduler = AgentScheduler()
        
        def task():
            return "Job executed"
        
        job_id = scheduler.add_job(
            cron="*/5 * * * *",  # Every 5 minutes
            task=task,
            name="test_job"
        )
        
        assert job_id is not None
        assert len(scheduler.jobs) == 1
        assert scheduler.jobs[job_id]["name"] == "test_job"
    
    def test_remove_job(self):
        """Test removing a scheduled job."""
        scheduler = AgentScheduler()
        
        job_id = scheduler.add_job(
            cron="0 * * * *",
            task=lambda: None,
            name="removable"
        )
        
        scheduler.remove_job(job_id)
        assert len(scheduler.jobs) == 0
    
    def test_list_jobs(self):
        """Test listing all jobs."""
        scheduler = AgentScheduler()
        
        scheduler.add_job("*/10 * * * *", lambda: None, "job1")
        scheduler.add_job("0 0 * * *", lambda: None, "job2")
        
        jobs = scheduler.list_jobs()
        assert len(jobs) == 2
        assert any(j["name"] == "job1" for j in jobs)
        assert any(j["name"] == "job2" for j in jobs)
