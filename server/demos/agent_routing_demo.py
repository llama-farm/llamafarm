#!/usr/bin/env python3
"""
Agent Routing Demo
==================

Demonstrates the new agent and node routing architecture:

1. SIMPLE USE CASE: Route between agents on the same device
   - Router agent analyzes intent
   - Spawns appropriate sub-agent (research, code, quick)
   - Returns aggregated result

2. COMPLEX USE CASE: Route across nodes in a mesh
   - Semantic routing finds best capability match
   - Credentials handled automatically
   - Multi-hop routing if needed

Usage:
    cd server
    uv run python demos/agent_routing_demo.py

Requirements:
    - LlamaFarm server running on port 14345
    - Embedding model: sentence-transformers/all-MiniLM-L6-v2 (384 dims)
"""

import asyncio
import sys
from pathlib import Path

import aiohttp
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from router.capability_store import CapabilityStore, get_embedding


# LlamaFarm server settings
LLAMAFARM_HOST = "localhost"
LLAMAFARM_PORT = 14345
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


async def demo_simple_routing():
    """
    Simple Use Case: Route between agents on the SAME device.
    
    Uses LlamaFarm's /v1/nlp/embeddings endpoint directly.
    Stores capabilities in ChromaDB via CapabilityStore.
    """
    print("\n" + "="*60)
    print("🎯 SIMPLE ROUTING: Same Device, Multiple Agents")
    print("="*60)
    
    # Initialize capability store
    store = CapabilityStore(
        persist_dir=Path("./data/demo_router"),
        collection_name="demo_capabilities",
        embedding_dimension=EMBEDDING_DIM
    )
    await store.initialize()
    print(f"\n📦 Store initialized: {store.stats()['backend']}")
    
    # Clear and rebuild for demo
    await store.clear()
    
    # Define capabilities with their semantic descriptions
    capabilities = [
        {
            "id": "research",
            "label": "research",
            "description": "Deep research and analysis on any topic. Thorough, well-reasoned responses with citations and sources. Historical analysis, literature review, fact-checking.",
            "handler": "agent:research"
        },
        {
            "id": "coding",
            "label": "coding", 
            "description": "Write, review, and debug code. Expert in Python, TypeScript, Go, Rust. Code generation, refactoring, bug fixes, code review, testing.",
            "handler": "agent:code"
        },
        {
            "id": "quick-answer",
            "label": "quick-answer",
            "description": "Fast answers to simple factual questions. Calculations, unit conversions, definitions, capital cities, basic facts.",
            "handler": "agent:quick"
        }
    ]
    
    # Add capabilities with embeddings from LlamaFarm
    print("\n🔄 Generating capability embeddings...")
    for cap in capabilities:
        # Get embedding from LlamaFarm
        vec = await get_embedding(
            f"{cap['label']}: {cap['description']}",
            host=LLAMAFARM_HOST,
            port=LLAMAFARM_PORT
        )
        
        await store.add_capability(
            cap_id=cap["id"],
            label=cap["label"],
            description=cap["description"],
            vector=vec,
            handler=cap["handler"],
            node_id="local"
        )
        print(f"   ✓ {cap['label']} ({len(vec)} dims)")
    
    # Test queries
    test_queries = [
        "Research the history of neural networks and their evolution",
        "Write a Python function to parse JSON files recursively",
        "What is the capital of France?",
        "Analyze the pros and cons of microservices architecture",
        "Debug this code: for i in range(10): print(i",
        "Calculate the compound interest on $1000 at 5% for 10 years"
    ]
    
    print("\n📊 Routing Decisions:")
    print("-" * 60)
    
    for query in test_queries:
        # Get query embedding
        query_vec = await get_embedding(query, host=LLAMAFARM_HOST, port=LLAMAFARM_PORT)
        
        # Search for matching capability (lower threshold for demo)
        results = await store.search(query, query_vec, top_k=1, min_score=0.2)
        
        if results:
            match = results[0]
            handler = match.capability.handler
            agent = handler.replace("agent:", "") if handler.startswith("agent:") else "model"
            print(f"\n  Query: \"{query[:50]}...\"")
            print(f"  → Agent: {agent} ({match.score:.0%} confidence)")
        else:
            print(f"\n  Query: \"{query[:50]}...\"")
            print(f"  → No match above threshold")
    
    # Show stats
    print("\n📈 Store Stats:")
    stats = store.stats()
    print(f"  - Backend: {stats['backend']}")
    print(f"  - Capabilities: {stats['count']}")
    print(f"  - Embedding dim: {stats['embedding_dimension']}")
    
    print("\n✅ Simple routing demo complete!")


async def demo_multi_node_routing():
    """
    Complex Use Case: Route across NODES in a mesh.
    
    Shows how semantic routing can:
    1. Find capabilities across multiple nodes
    2. Handle authentication automatically
    3. Route to the best available node
    
    This is a conceptual demo - real multi-node routing
    uses the gossip protocol and gradient tables.
    """
    print("\n" + "="*60)
    print("🌐 MULTI-NODE ROUTING: Distributed Mesh")
    print("="*60)
    
    # In production, this config would be in config/examples/multi-node-agents.yaml
    print("\n📄 Config structure (see config/examples/multi-node-agents.yaml):")
    print("""
    nodes:
      self:
        node_id: macbook-main
        capabilities: [text-generation, local-code]
      peers:
        - name: dell-gpu
          url: http://192.168.1.100:14345
          auth: {type: bearer, token: ${env:DELL_TOKEN}}
          capabilities: [vision, image-generation]
    """)
    
    print("\n🖥️  Simulated Node Mesh:")
    print("-" * 60)
    nodes = [
        ("macbook-main", "Local", ["text-generation", "local-code"]),
        ("dell-gpu", "192.168.1.100", ["vision", "image-generation", "video-processing"]),
        ("jetson-edge", "192.168.1.101", ["edge-inference", "real-time-detection"]),
        ("rpi-monitor", "192.168.1.102", ["monitoring", "data-collection"])
    ]
    
    for name, addr, caps in nodes:
        print(f"\n  📍 {name} ({addr})")
        for cap in caps:
            print(f"     └─ {cap}")
    
    print("\n💡 In production, the router would:")
    print("  1. Use gossip protocol to discover node capabilities")
    print("  2. Build gradient table for multi-hop routing")
    print("  3. Authenticate using configured credentials")
    print("  4. Route requests to the optimal node")
    
    print("\n📝 Example routing decisions:")
    examples = [
        ("Analyze this image for objects", "dell-gpu (vision capability)"),
        ("Monitor system health", "rpi-monitor (monitoring capability)"),
        ("Detect objects in real-time", "jetson-edge (real-time-detection)"),
        ("Write a Python script", "macbook-main (local-code capability)")
    ]
    
    for query, target in examples:
        print(f"\n  Query: \"{query}\"")
        print(f"  → Would route to: {target}")
    
    print("\n✅ Multi-node routing demo complete!")


async def demo_llamafarm_embeddings():
    """
    Test LlamaFarm's /v1/nlp/embeddings endpoint directly.
    
    Uses sentence-transformers/all-MiniLM-L6-v2 (384 dims).
    """
    print("\n" + "="*60)
    print("🧠 LLAMAFARM EMBEDDINGS: Direct API Test")
    print("="*60)
    
    print(f"\n📡 Connecting to LlamaFarm at {LLAMAFARM_HOST}:{LLAMAFARM_PORT}")
    print(f"   Model: {EMBEDDING_MODEL}")
    print(f"   Dimension: {EMBEDDING_DIM}")
    
    try:
        # Test embedding
        test_texts = [
            "Research the history of neural networks",
            "Write a Python function to sort data",
            "What is the weather today?"
        ]
        
        print("\n📊 Sample Embeddings:")
        for text in test_texts:
            vec = await get_embedding(text, host=LLAMAFARM_HOST, port=LLAMAFARM_PORT)
            print(f"\n  \"{text[:40]}...\"")
            print(f"  → Shape: {vec.shape}, First 5: [{', '.join(f'{v:.3f}' for v in vec[:5])}...]")
        
        # Test similarity
        print("\n🔍 Similarity Test:")
        research_vec = await get_embedding("Research the topic in depth", host=LLAMAFARM_HOST, port=LLAMAFARM_PORT)
        code_vec = await get_embedding("Write Python code", host=LLAMAFARM_HOST, port=LLAMAFARM_PORT)
        question_vec = await get_embedding("Research about programming languages", host=LLAMAFARM_HOST, port=LLAMAFARM_PORT)
        
        # Cosine similarity (vectors should be normalized)
        sim_research = float(np.dot(question_vec, research_vec))
        sim_code = float(np.dot(question_vec, code_vec))
        
        print(f"\n  Query: \"Research about programming languages\"")
        print(f"  → Similarity to 'Research the topic in depth': {sim_research:.3f}")
        print(f"  → Similarity to 'Write Python code': {sim_code:.3f}")
        print(f"  → Better match: {'Research' if sim_research > sim_code else 'Code'}")
        
        print("\n✅ LlamaFarm embeddings working!")
        
    except Exception as e:
        print(f"\n⚠️  Could not connect to LlamaFarm: {e}")
        print(f"   Make sure LlamaFarm server is running:")
        print(f"   cd server && uv run python main.py")
        return False
    
    return True


async def main():
    print("\n" + "🚀" * 30)
    print("\n    LlamaFarm Agent & Node Routing Demo")
    print("\n" + "🚀" * 30)
    
    # Check what's available
    print("\n📋 Demo Sections:")
    print("  1. LlamaFarm Embeddings (/v1/nlp/embeddings)")
    print("  2. Simple Routing (same device, multiple agents)")
    print("  3. Multi-Node Routing (distributed mesh)")
    
    # Run embedding demo first (core dependency)
    embeddings_ok = await demo_llamafarm_embeddings()
    
    if not embeddings_ok:
        print("\n⚠️  Skipping routing demos (embeddings not available)")
        return
    
    # Run simple routing demo
    try:
        await demo_simple_routing()
    except Exception as e:
        print(f"\n⚠️  Simple routing demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Run multi-node demo (mostly informational)
    await demo_multi_node_routing()
    
    print("\n" + "="*60)
    print("🎉 All demos complete!")
    print("="*60)
    print("\n📚 Architecture Summary:")
    print(f"  - Embedding Model: {EMBEDDING_MODEL}")
    print(f"  - Embedding Dims: {EMBEDDING_DIM}")
    print(f"  - LlamaFarm Port: {LLAMAFARM_PORT}")
    print(f"  - Endpoint: /v1/nlp/embeddings")
    print("\n📚 Next Steps:")
    print("  1. Ensure LlamaFarm server is running: cd server && uv run python main.py")
    print("  2. Create a config with agents/nodes sections (see config/examples/)")
    print("  3. Access router API at /router/* endpoints")
    print("  4. Use Designer UI at /chat/router for visual routing")


if __name__ == "__main__":
    asyncio.run(main())
