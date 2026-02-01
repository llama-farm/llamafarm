#!/usr/bin/env python3
"""
OpenClaw Lite Semantic Router Demo

Demonstrates:
- Embedding engine for text vectorization
- Capability matching via semantic similarity
- Intent routing to best-fit capabilities
- Gradient-based routing optimization
"""

import asyncio
import numpy as np
from typing import Dict, List

from router import (
    EmbeddingEngine,
    CapabilityMatcher,
    MatchResult,
    Capability,
)


def demo_embedding_engine():
    """Demo: Text embeddings and similarity."""
    print("\n=== Embedding Engine Demo ===")
    
    engine = EmbeddingEngine()
    
    # Embed some example texts
    texts = [
        "What's the weather like today?",
        "Tell me about the forecast",
        "Search Google for python tutorials",
        "Find information about machine learning",
        "Send an email to my boss",
        "Calculate 15 times 23"
    ]
    
    print("\n Embedding texts...")
    embeddings = []
    for text in texts:
        emb = engine.embed(text)
        embeddings.append(emb)
        print(f"  '{text}' → {emb.shape[0]}-dim vector")
    
    # Check similarity between weather queries
    sim_weather = engine.cosine_similarity(embeddings[0], embeddings[1])
    sim_search = engine.cosine_similarity(embeddings[2], embeddings[3])
    sim_different = engine.cosine_similarity(embeddings[0], embeddings[4])
    
    print("\n Semantic similarity:")
    print(f"  Weather queries: {sim_weather:.3f}")
    print(f"  Search queries: {sim_search:.3f}")
    print(f"  Weather vs Email: {sim_different:.3f}")
    
    return engine


def demo_capability_matching():
    """Demo: Matching intents to capabilities."""
    print("\n\n=== Capability Matching Demo ===")
    
    # Define capabilities
    capabilities = {
        "weather": Capability(
            name="weather",
            description="Get weather information and forecasts",
            examples=[
                "What's the weather in NYC?",
                "Will it rain tomorrow?",
                "Temperature forecast for this week",
                "Is it sunny today?"
            ],
            node_id="weather-service-001"
        ),
        
        "search": Capability(
            name="search",
            description="Search the web for information",
            examples=[
                "Search Google for Python tutorials",
                "Find information about AI",
                "Look up the capital of France",
                "Google machine learning papers"
            ],
            node_id="search-service-001"
        ),
        
        "calculator": Capability(
            name="calculator",
            description="Perform mathematical calculations",
            examples=[
                "What is 15 times 23?",
                "Calculate the square root of 144",
                "Solve 5x + 3 = 18",
                "What's 25% of 200?"
            ],
            node_id="math-service-001"
        ),
        
        "email": Capability(
            name="email",
            description="Send and manage emails",
            examples=[
                "Send an email to my boss",
                "Check my inbox",
                "Reply to the last message",
                "Draft an email about the meeting"
            ],
            node_id="email-service-001"
        )
    }
    
    print(f"\n Registered {len(capabilities)} capabilities:")
    for cap_name, cap in capabilities.items():
        print(f"  • {cap_name}: {cap.description}")
        print(f"    Node: {cap.node_id}")
    
    # Create matcher
    matcher = CapabilityMatcher(capabilities)
    
    # Test queries
    test_queries = [
        "What's the temperature going to be tomorrow?",
        "Find me some articles about deep learning",
        "What is 42 divided by 7?",
        "I need to email the team about the project update",
        "Will it be sunny this weekend?",
        "Can you help me calculate compound interest?",
    ]
    
    print("\n\n Routing test queries:")
    print(" " + "=" * 70)
    
    for query in test_queries:
        result = matcher.match(query, top_k=3)
        
        print(f"\n Query: '{query}'")
        print(f" Top matches:")
        
        for i, match in enumerate(result, 1):
            cap = capabilities[match.capability_name]
            confidence_pct = match.confidence * 100
            print(f"   {i}. {match.capability_name} ({confidence_pct:.1f}%)")
            print(f"      Node: {cap.node_id}")
            if i == 1:
                print(f"      → ROUTE TO THIS")
    
    return matcher


def demo_route_decision():
    """Demo: Making routing decisions with thresholds."""
    print("\n\n=== Route Decision Demo ===")
    
    capabilities = {
        "general_chat": Capability(
            name="general_chat",
            description="General conversation and chitchat",
            examples=[
                "Hello, how are you?",
                "Tell me a joke",
                "What's your name?",
                "How's your day going?"
            ],
            node_id="chat-bot-001"
        ),
        
        "technical_support": Capability(
            name="technical_support",
            description="Technical troubleshooting and support",
            examples=[
                "My laptop won't turn on",
                "How do I reset my password?",
                "The application keeps crashing",
                "I can't connect to WiFi"
            ],
            node_id="support-agent-001"
        )
    }
    
    matcher = CapabilityMatcher(capabilities)
    
    # Test with varying confidence thresholds
    query = "The server is showing a 500 error"
    
    print(f"\n Query: '{query}'")
    print("\n Testing with different confidence thresholds:")
    
    for threshold in [0.3, 0.5, 0.7, 0.9]:
        result = matcher.match(query, threshold=threshold, top_k=1)
        
        if result and result[0].confidence >= threshold:
            match = result[0]
            print(f"\n  Threshold: {threshold:.1f}")
            print(f"   → Matched: {match.capability_name} ({match.confidence:.3f})")
        else:
            print(f"\n  Threshold: {threshold:.1f}")
            print(f"   → No match (confidence too low)")
            print(f"   → Would fallback to default handler")
    
    return matcher


def demo_multi_capability_routing():
    """Demo: Routing to multiple capabilities."""
    print("\n\n=== Multi-Capability Routing Demo ===")
    
    capabilities = {
        "weather": Capability(
            name="weather",
            description="Weather information",
            examples=["What's the weather?"],
            node_id="weather-001"
        ),
        "calendar": Capability(
            name="calendar",
            description="Calendar and scheduling",
            examples=["Check my calendar", "Schedule a meeting"],
            node_id="calendar-001"
        ),
        "reminder": Capability(
            name="reminder",
            description="Reminders and notifications",
            examples=["Remind me to call John", "Set a reminder"],
            node_id="reminder-001"
        )
    }
    
    matcher = CapabilityMatcher(capabilities)
    
    # Complex query that might match multiple capabilities
    query = "Remind me to check the weather before my meeting tomorrow"
    
    print(f"\n Query: '{query}'")
    print("\n This query involves multiple capabilities:")
    
    matches = matcher.match(query, top_k=3)
    
    for i, match in enumerate(matches, 1):
        confidence_pct = match.confidence * 100
        print(f"\n  {i}. {match.capability_name} ({confidence_pct:.1f}%)")
        
        if i == 1:
            print(f"     → Primary capability (execute first)")
        elif confidence_pct > 50:
            print(f"     → Secondary capability (possibly relevant)")
        else:
            print(f"     → Low relevance (skip)")
    
    print("\n Strategy: Execute capabilities in order of relevance")
    print("  1. Create reminder (primary)")
    print("  2. Query weather (if confidence > 50%)")
    print("  3. Check calendar (if confidence > 50%)")


async def main():
    """Run all demos."""
    print("OpenClaw Lite Semantic Router Demo")
    print("=" * 70)
    
    # Run demos
    engine = demo_embedding_engine()
    matcher = demo_capability_matching()
    decision_demo = demo_route_decision()
    demo_multi_capability_routing()
    
    print("\n\n" + "=" * 70)
    print("Demo completed successfully! ✅")
    print("\nKey takeaways:")
    print("  • Embeddings capture semantic meaning of text")
    print("  • Capabilities advertise what they can do via examples")
    print("  • Routing matches intents to capabilities via similarity")
    print("  • Confidence thresholds prevent incorrect routing")
    print("  • Complex queries can trigger multiple capabilities")


if __name__ == "__main__":
    asyncio.run(main())
