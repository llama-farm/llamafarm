#!/usr/bin/env python3
"""
Simple OpenClaw Lite Routing Demo

Demonstrates basic capability matching without embeddings.
Shows the conceptual flow of semantic routing.
"""

from dataclasses import dataclass
from typing import List, Dict
import asyncio


@dataclass
class SimpleCapability:
    """A simple capability definition."""
    name: str
    description: str
    keywords: List[str]
    node_id: str


class SimpleRouter:
    """Keyword-based router for demonstration."""
    
    def __init__(self, capabilities: Dict[str, SimpleCapability]):
        self.capabilities = capabilities
    
    def route(self, query: str) -> List[tuple]:
        """Route query to capabilities based on keyword matching."""
        query_lower = query.lower()
        matches = []
        
        for cap_name, cap in self.capabilities.items():
            score = 0
            for keyword in cap.keywords:
                if keyword.lower() in query_lower:
                    score += 1
            
            if score > 0:
                matches.append((cap_name, score, cap))
        
        # Sort by score descending
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches


async def main():
    """Demo simple routing."""
    print("OpenClaw Lite Simple Routing Demo")
    print("=" * 70)
    
    # Define capabilities
    capabilities = {
        "weather": SimpleCapability(
            name="weather",
            description="Get weather information",
            keywords=["weather", "temperature", "forecast", "rain", "sunny", "cloud"],
            node_id="weather-service-001"
        ),
        
        "search": SimpleCapability(
            name="search",
            description="Web search",
            keywords=["search", "find", "google", "lookup", "information"],
            node_id="search-service-001"
        ),
        
        "calculator": SimpleCapability(
            name="calculator",
            description="Math calculations",
            keywords=["calculate", "math", "plus", "minus", "times", "divided", "square"],
            node_id="calc-service-001"
        ),
        
        "email": SimpleCapability(
            name="email",
            description="Email management",
            keywords=["email", "send", "inbox", "message", "reply"],
            node_id="email-service-001"
        )
    }
    
    print(f"\n Registered {len(capabilities)} capabilities:\n")
    for cap_name, cap in capabilities.items():
        print(f"  • {cap.name}: {cap.description}")
        print(f"    Keywords: {', '.join(cap.keywords[:5])}")
        print(f"    Node: {cap.node_id}\n")
    
    # Create router
    router = SimpleRouter(capabilities)
    
    # Test queries
    test_queries = [
        "What's the weather forecast for tomorrow?",
        "Search for Python tutorials",
        "Calculate 15 times 23",
        "Send an email to the team",
        "Will it rain this weekend?",
        "Find information about machine learning"
    ]
    
    print("\n" + "=" * 70)
    print("Routing Test Queries:\n")
    
    for query in test_queries:
        print(f" Query: '{query}'")
        
        matches = router.route(query)
        
        if matches:
            print(f" Matches:")
            for cap_name, score, cap in matches:
                print(f"   • {cap_name} (score: {score}) → {cap.node_id}")
                if matches.index((cap_name, score, cap)) == 0:
                    print(f"     ✓ ROUTE TO THIS")
        else:
            print(f" No matches → fallback to default handler")
        
        print()
    
    print("=" * 70)
    print("\nDemo completed! ✅")
    print("\nKey concepts demonstrated:")
    print("  • Capabilities advertise what they can do")
    print("  • Queries are matched to capabilities")
    print("  • Best match gets the request")
    print("  • Each capability runs on a specific node")
    print("\n In production:")
    print("  • Replace keyword matching with semantic embeddings")
    print("  • Add confidence thresholds")
    print("  • Support multi-capability routing")
    print("  • Enable gradient-based learning")


if __name__ == "__main__":
    asyncio.run(main())
