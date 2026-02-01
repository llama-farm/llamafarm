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
from dataclasses import dataclass

from router import EmbeddingEngine


@dataclass
class DemoCapability:
    """Simple capability for demo purposes."""
    name: str
    description: str
    examples: List[str]
    node_id: str


async def demo_embedding_engine():
    """Demo: Text embeddings and similarity."""
    print("\n=== Embedding Engine Demo ===")
    
    engine = EmbeddingEngine()
    await engine.initialize()
    
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
        emb = await engine.embed(text)
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


async def demo_capability_matching():
    """Demo: Matching intents to capabilities."""
    print("\n\n=== Capability Matching Demo ===")
    
    # Initialize embedding engine
    engine = EmbeddingEngine()
    await engine.initialize()
    
    # Define capabilities
    capabilities = {
        "weather": DemoCapability(
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
        
        "search": DemoCapability(
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
        
        "calculator": DemoCapability(
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
        
        "email": DemoCapability(
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
    
    # Create matcher (simplified for demo)
    # Note: Real matcher uses vectorized capabilities
    # This demo shows the conceptual flow
    print("\n Creating semantic matcher...")
    
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
        # For demo: simple semantic matching using embeddings
        query_emb = await engine.embed(query)
        
        print(f"\n Query: '{query}'")
        print(f" Top matches:")
        
        # Calculate similarity to each capability
        scores = []
        for cap_name, cap in capabilities.items():
            # Average embeddings of examples
            example_embs = []
            for example in cap.examples:
                ex_emb = await engine.embed(example)
                example_embs.append(ex_emb)
            
            if example_embs:
                cap_emb = np.mean(example_embs, axis=0)
                cap_emb = cap_emb / np.linalg.norm(cap_emb)  # normalize
                score = engine.cosine_similarity(query_emb, cap_emb)
                scores.append((cap_name, score, cap))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        for i, (cap_name, score, cap) in enumerate(scores[:3], 1):
            confidence_pct = score * 100
            print(f"   {i}. {cap_name} ({confidence_pct:.1f}%)")
            print(f"      Node: {cap.node_id}")
            if i == 1:
                print(f"      → ROUTE TO THIS")
    
    await engine.close()
    return None


async def demo_route_decision():
    """Demo: Making routing decisions with thresholds."""
    print("\n\n=== Route Decision Demo ===")
    
    engine = EmbeddingEngine()
    await engine.initialize()
    
    capabilities = {
        "general_chat": DemoCapability(
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
        
        "technical_support": DemoCapability(
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
    
    # Test with varying confidence thresholds
    query = "The server is showing a 500 error"
    query_emb = await engine.embed(query)
    
    print(f"\n Query: '{query}'")
    print("\n Testing with different confidence thresholds:")
    
    # Calculate scores
    scores = {}
    for cap_name, cap in capabilities.items():
        example_embs = [await engine.embed(ex) for ex in cap.examples]
        cap_emb = np.mean(example_embs, axis=0)
        cap_emb = cap_emb / np.linalg.norm(cap_emb)
        scores[cap_name] = engine.cosine_similarity(query_emb, cap_emb)
    
    best_cap = max(scores, key=scores.get)
    best_score = scores[best_cap]
    
    for threshold in [0.3, 0.5, 0.7, 0.9]:
        if best_score >= threshold:
            print(f"\n  Threshold: {threshold:.1f}")
            print(f"   → Matched: {best_cap} ({best_score:.3f})")
        else:
            print(f"\n  Threshold: {threshold:.1f}")
            print(f"   → No match (confidence too low)")
            print(f"   → Would fallback to default handler")
    
    await engine.close()
    return None


async def demo_multi_capability_routing():
    """Demo: Routing to multiple capabilities."""
    print("\n\n=== Multi-Capability Routing Demo ===")
    
    engine = EmbeddingEngine()
    await engine.initialize()
    
    capabilities = {
        "weather": DemoCapability(
            name="weather",
            description="Weather information",
            examples=["What's the weather?"],
            node_id="weather-001"
        ),
        "calendar": DemoCapability(
            name="calendar",
            description="Calendar and scheduling",
            examples=["Check my calendar", "Schedule a meeting"],
            node_id="calendar-001"
        ),
        "reminder": DemoCapability(
            name="reminder",
            description="Reminders and notifications",
            examples=["Remind me to call John", "Set a reminder"],
            node_id="reminder-001"
        )
    }
    
    # Complex query that might match multiple capabilities
    query = "Remind me to check the weather before my meeting tomorrow"
    query_emb = await engine.embed(query)
    
    print(f"\n Query: '{query}'")
    print("\n This query involves multiple capabilities:")
    
    # Calculate all scores
    scores = []
    for cap_name, cap in capabilities.items():
        example_embs = [await engine.embed(ex) for ex in cap.examples]
        cap_emb = np.mean(example_embs, axis=0)
        cap_emb = cap_emb / np.linalg.norm(cap_emb)
        score = engine.cosine_similarity(query_emb, cap_emb)
        scores.append((cap_name, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    
    for i, (cap_name, score) in enumerate(scores, 1):
        confidence_pct = score * 100
        print(f"\n  {i}. {cap_name} ({confidence_pct:.1f}%)")
        
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
    
    await engine.close()


async def main():
    """Run all demos."""
    print("OpenClaw Lite Semantic Router Demo")
    print("=" * 70)
    
    # Run demos
    engine = await demo_embedding_engine()
    await engine.close()
    
    await demo_capability_matching()
    await demo_route_decision()
    await demo_multi_capability_routing()
    
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
