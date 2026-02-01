#!/usr/bin/env python3
"""
OpenClaw Lite Multi-Turn Session Demo

Demonstrates:
- Creating conversational sessions
- Managing multi-turn dialogue
- Context retention across messages
- Session state and metadata
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from agents import SessionManager, Session, SessionStatus


async def demo_simple_conversation():
    """Demo: Basic multi-turn conversation."""
    print("\n=== Simple Conversation Demo ===")
    
    manager = SessionManager()
    
    # Create session
    session = await manager.create_session(
        session_id="simple-chat-001",
        metadata={
            "user": "alice",
            "channel": "web",
            "language": "en"
        }
    )
    
    print(f"\n Created session: {session.session_id}")
    
    # Simulate conversation
    conversation = [
        ("user", "Hi! My name is Alice."),
        ("assistant", "Hello Alice! Nice to meet you. How can I help you today?"),
        ("user", "What's the weather like in New York?"),
        ("assistant", "Let me check that for you. The current weather in New York is 68°F and partly cloudy."),
        ("user", "Thanks! What about tomorrow?"),
        ("assistant", "Tomorrow in New York will be sunny with a high of 72°F."),
        ("user", "Perfect! Can you remind me to bring an umbrella just in case?"),
        ("assistant", "I've created a reminder for you to bring an umbrella tomorrow."),
    ]
    
    print("\n Conversation:")
    for role, content in conversation:
        await session.add_message(role, content)
        print(f"  {role:10s}: {content}")
    
    # Get full history
    history = session.get_history()
    print(f"\n Total messages in session: {len(history)}")
    
    return session


async def demo_context_retention():
    """Demo: Context retention across messages."""
    print("\n\n=== Context Retention Demo ===")
    
    manager = SessionManager()
    
    session = await manager.create_session(
        session_id="context-demo-001",
        metadata={"user": "bob"}
    )
    
    print(f"\n Session: {session.session_id}")
    
    # First exchange establishes context
    await session.add_message("user", "I'm planning a trip to Paris next month.")
    await session.add_message("assistant", "That sounds exciting! Paris in March is lovely. How long will you be staying?")
    
    # Later messages reference earlier context
    await session.add_message("user", "About a week. What's the weather usually like?")
    await session.add_message("assistant", "In Paris during March, temperatures typically range from 45-55°F. I'd recommend bringing layers and a light rain jacket.")
    
    await session.add_message("user", "Great! Any restaurant recommendations?")
    await session.add_message("assistant", "For your week in Paris, I recommend trying traditional bistros in the Marais district, or fine dining at L'Ambroisie if you want something special.")
    
    print("\n Context-aware conversation:")
    for msg in session.get_history():
        role = msg["role"]
        content = msg["content"]
        print(f"  {role:10s}: {content}")
    
    print("\n Notice how later messages reference:")
    print("  • The destination (Paris)")
    print("  • The duration (a week)")
    print("  • The time frame (March)")
    print("\n This demonstrates session context retention!")
    
    return session


async def demo_session_metadata():
    """Demo: Using session metadata for context."""
    print("\n\n=== Session Metadata Demo ===")
    
    manager = SessionManager()
    
    # Create sessions with rich metadata
    sessions = []
    
    # Customer support session
    support_session = await manager.create_session(
        session_id="support-001",
        metadata={
            "user_id": "user_12345",
            "user_name": "Charlie",
            "channel": "chat",
            "issue_type": "technical",
            "priority": "high",
            "account_tier": "premium",
            "timestamp": datetime.now().isoformat()
        }
    )
    sessions.append(("Support Session", support_session))
    
    # Sales inquiry session
    sales_session = await manager.create_session(
        session_id="sales-001",
        metadata={
            "user_id": "lead_67890",
            "user_name": "Diana",
            "channel": "email",
            "inquiry_type": "pricing",
            "company": "Acme Corp",
            "timestamp": datetime.now().isoformat()
        }
    )
    sessions.append(("Sales Session", sales_session))
    
    # Personal assistant session
    assistant_session = await manager.create_session(
        session_id="assistant-001",
        metadata={
            "user_id": "user_99999",
            "user_name": "Eve",
            "channel": "voice",
            "preferences": {
                "language": "en-US",
                "timezone": "America/New_York",
                "temperature_unit": "fahrenheit"
            },
            "timestamp": datetime.now().isoformat()
        }
    )
    sessions.append(("Assistant Session", assistant_session))
    
    print("\n Created sessions with metadata:")
    for name, session in sessions:
        print(f"\n  {name}:")
        print(f"    ID: {session.session_id}")
        print(f"    Status: {session.status.value}")
        print(f"    Metadata: {json.dumps(session.metadata, indent=6)}")
    
    print("\n Metadata enables:")
    print("  • Personalization (names, preferences)")
    print("  • Routing (channel, issue type)")
    print("  • Prioritization (account tier, priority)")
    print("  • Analytics (timestamps, user segments)")
    
    return sessions


async def demo_session_lifecycle():
    """Demo: Complete session lifecycle."""
    print("\n\n=== Session Lifecycle Demo ===")
    
    manager = SessionManager()
    
    # Create session
    session = await manager.create_session(session_id="lifecycle-demo-001")
    print(f"\n 1. Session created: {session.session_id}")
    print(f"    Status: {session.status.value}")
    
    # Add messages
    await session.add_message("user", "Hello!")
    await session.add_message("assistant", "Hi there!")
    print(f"\n 2. Messages added: {len(session.get_history())} total")
    
    # Update metadata
    session.metadata["user_rating"] = 5
    session.metadata["completed_action"] = "weather_query"
    print(f"\n 3. Metadata updated: {session.metadata}")
    
    # Pause session
    session.status = SessionStatus.PAUSED
    print(f"\n 4. Session paused: {session.status.value}")
    
    # Resume session
    session.status = SessionStatus.ACTIVE
    await session.add_message("user", "Thanks for your help!")
    print(f"\n 5. Session resumed and message added")
    
    # Complete session
    session.status = SessionStatus.COMPLETED
    print(f"\n 6. Session completed: {session.status.value}")
    print(f"    Final message count: {len(session.get_history())}")
    
    # Get session info
    all_sessions = await manager.list_sessions()
    print(f"\n 7. Total sessions in manager: {len(all_sessions)}")
    
    return session


async def demo_concurrent_sessions():
    """Demo: Managing multiple concurrent sessions."""
    print("\n\n=== Concurrent Sessions Demo ===")
    
    manager = SessionManager()
    
    # Create multiple sessions for different users
    user_sessions = {}
    
    for i in range(3):
        session = await manager.create_session(
            session_id=f"user-{i+1}-session",
            metadata={"user": f"user_{i+1}", "start_time": datetime.now().isoformat()}
        )
        user_sessions[f"user_{i+1}"] = session
    
    print(f"\n Created {len(user_sessions)} concurrent sessions:")
    
    # Simulate concurrent conversations
    conversations = {
        "user_1": [
            ("user", "Show me my calendar"),
            ("assistant", "You have 2 meetings today"),
        ],
        "user_2": [
            ("user", "Send an email to the team"),
            ("assistant", "Email draft created. What's the subject?"),
        ],
        "user_3": [
            ("user", "What's the weather?"),
            ("assistant", "Current weather: 70°F, sunny"),
        ]
    }
    
    for user, messages in conversations.items():
        session = user_sessions[user]
        print(f"\n  [{user}] Session: {session.session_id}")
        for role, content in messages:
            await session.add_message(role, content)
            print(f"    {role:10s}: {content}")
    
    # Show summary
    print("\n Session Summary:")
    all_sessions = await manager.list_sessions()
    for session in all_sessions:
        msg_count = len(session.get_history())
        user = session.metadata.get("user", "unknown")
        print(f"  {session.session_id}: {msg_count} messages from {user}")
    
    return user_sessions


async def main():
    """Run all demos."""
    print("OpenClaw Lite Multi-Turn Session Demo")
    print("=" * 70)
    
    # Run demos
    simple = await demo_simple_conversation()
    context = await demo_context_retention()
    metadata = await demo_session_metadata()
    lifecycle = await demo_session_lifecycle()
    concurrent = await demo_concurrent_sessions()
    
    print("\n\n" + "=" * 70)
    print("Demo completed successfully! ✅")
    print("\nKey takeaways:")
    print("  • Sessions maintain conversation history and context")
    print("  • Metadata enriches sessions with user info and preferences")
    print("  • Session status tracks lifecycle (active/paused/completed)")
    print("  • Multiple concurrent sessions can be managed independently")
    print("  • Context retention enables natural multi-turn dialogue")


if __name__ == "__main__":
    asyncio.run(main())
