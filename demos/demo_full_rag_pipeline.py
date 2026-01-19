#!/usr/bin/env python3
"""Full RAG Pipeline Demo - Complete walkthrough of RAG-lite capabilities.

This demo walks through:
1. Database creation and management
2. Document ingestion with different chunking strategies
3. Metadata extraction (keywords, entities, summaries)
4. Vector embedding and storage
5. Search with different retrieval approaches
6. Cleanup

Requirements:
- Server running: nx start server
- Universal Runtime: nx start universal-runtime (for embeddings)
"""

import json
import os
import requests
import tempfile
import time

# Configuration
BASE_URL = "http://127.0.0.1:8000"
RAG_LITE_URL = f"{BASE_URL}/v1/rag-lite"

# Sample documents for testing
SAMPLE_DOCS = {
    "tech_article.md": """# AI Breakthroughs in 2024

## Introduction

Artificial Intelligence has made remarkable progress in 2024, with several key breakthroughs
transforming how we approach complex problems.

## Key Developments

### Large Language Models
The latest generation of large language models has achieved unprecedented capabilities in
reasoning, code generation, and multimodal understanding. These models can now process
images, audio, and text simultaneously.

### Robotics Integration
AI-powered robots have become more sophisticated, with improved dexterity and the ability
to learn from human demonstrations. Companies like Boston Dynamics and Tesla have unveiled
new humanoid robots capable of complex tasks.

### Scientific Discovery
AI systems are now contributing to scientific breakthroughs, including protein structure
prediction, drug discovery, and materials science. AlphaFold continues to revolutionize
biology research.

## Conclusion

The pace of AI advancement shows no signs of slowing down, with researchers pushing the
boundaries of what's possible with machine learning and neural networks.
""",
    "health_guide.txt": """Guide to Healthy Living

Maintaining good health requires attention to several key areas:

1. Nutrition
A balanced diet includes fruits, vegetables, whole grains, and lean proteins.
Limit processed foods and added sugars. Stay hydrated by drinking plenty of water.

2. Exercise
Regular physical activity is essential for cardiovascular health. Aim for at least
150 minutes of moderate exercise per week. Include both cardio and strength training.

3. Sleep
Quality sleep is crucial for mental and physical well-being. Adults need 7-9 hours
of sleep per night. Maintain a consistent sleep schedule for optimal rest.

4. Mental Health
Practice stress management through meditation, hobbies, or social activities.
Seek professional help when needed. Strong social connections support mental wellness.

5. Preventive Care
Regular check-ups help catch health issues early. Stay current on vaccinations
and screenings appropriate for your age and risk factors.
""",
    "cooking_recipes.json": json.dumps({
        "title": "Classic Pasta Recipes",
        "author": "Chef Maria",
        "recipes": [
            {
                "name": "Spaghetti Carbonara",
                "time": "30 minutes",
                "difficulty": "Medium",
                "ingredients": ["spaghetti", "eggs", "pecorino", "guanciale", "black pepper"],
                "instructions": "Cook pasta. Fry guanciale. Mix eggs with cheese. Combine all."
            },
            {
                "name": "Aglio e Olio",
                "time": "20 minutes",
                "difficulty": "Easy",
                "ingredients": ["spaghetti", "garlic", "olive oil", "chili flakes", "parsley"],
                "instructions": "Cook pasta. Sauté garlic in oil. Toss with pasta and parsley."
            }
        ]
    }, indent=2)
}


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_step(step: int, description: str):
    """Print a step indicator."""
    print(f"\n{'─' * 70}")
    print(f"  Step {step}: {description}")
    print(f"{'─' * 70}")


def check_server():
    """Verify server is running."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        health = response.json()
        return health["status"] == "healthy"
    except Exception:
        return False


def get_stats():
    """Get current RAG stats."""
    response = requests.get(f"{RAG_LITE_URL}/stats", timeout=10)
    return response.json()


def ingest_file(content: str, filename: str, chunk_strategy: str = "semantic"):
    """Ingest a file with specified chunking strategy."""
    # Create temp file
    suffix = os.path.splitext(filename)[1]
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False, encoding='utf-8') as f:
        f.write(content)
        temp_path = f.name

    try:
        with open(temp_path, "rb") as f:
            files = {"file": (filename, f)}
            params = {"chunk_strategy": chunk_strategy}
            response = requests.post(
                f"{RAG_LITE_URL}/ingest",
                files=files,
                params=params,
                timeout=60
            )
        return response.json()
    finally:
        os.unlink(temp_path)


def search(query: str, top_k: int = 3):
    """Search with basic similarity."""
    payload = {"query": query, "top_k": top_k}
    response = requests.post(f"{RAG_LITE_URL}/search", json=payload, timeout=30)
    return response.json()


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                     FULL RAG PIPELINE DEMONSTRATION                          ║
║                                                                              ║
║   This demo walks through the complete RAG-lite workflow:                    ║
║   • Document ingestion with different chunking strategies                    ║
║   • Metadata extraction (keywords, summaries)                                ║
║   • Vector embedding and storage                                             ║
║   • Search and retrieval                                                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    # Check server
    print_step(1, "Checking Server Status")

    if not check_server():
        print("✗ Server not responding!")
        print("  Please start: nx start server && nx start universal-runtime")
        return

    print("✓ Server is healthy!")

    # Get initial stats
    stats = get_stats()
    initial_count = stats["document_count"]
    print(f"\n  Current document count: {initial_count}")
    print(f"  Embedding model: {stats['embedding_model']}")
    print(f"  Embedding dimension: {stats['embedding_dimension']}")

    # Ingest documents with different strategies
    print_step(2, "Ingesting Documents with Different Chunking Strategies")

    strategies = {
        "tech_article.md": "section",    # Markdown -> section-based
        "health_guide.txt": "paragraph", # Plain text -> paragraph-based
        "cooking_recipes.json": "semantic" # JSON -> semantic (default)
    }

    ingested = []
    for filename, strategy in strategies.items():
        content = SAMPLE_DOCS[filename]
        print(f"\n  Ingesting {filename} with '{strategy}' chunking...")

        result = ingest_file(content, filename, strategy)

        if result.get("success"):
            print(f"    ✓ Success!")
            print(f"      Chunks created: {result['chunks']}")
            print(f"      Characters: {result['characters']}")
            ingested.append(filename)
        else:
            print(f"    ✗ Failed: {result.get('error', 'Unknown error')}")

    # Verify ingestion
    print_step(3, "Verifying Ingestion")

    stats = get_stats()
    new_count = stats["document_count"]
    added = new_count - initial_count

    print(f"  Documents added: {added}")
    print(f"  Total documents: {new_count}")

    # Test different search queries
    print_step(4, "Testing Search with Different Queries")

    test_queries = [
        ("What are the key AI developments?", "Looking for tech article content"),
        ("How much sleep do adults need?", "Looking for health guide content"),
        ("How to make carbonara pasta?", "Looking for recipe content"),
        ("robotics and machine learning", "Cross-document search")
    ]

    for query, description in test_queries:
        print(f"\n  Query: '{query}'")
        print(f"  Purpose: {description}")

        result = search(query, top_k=2)

        if result.get("success") and result["count"] > 0:
            print(f"  Results: {result['count']}")
            for i, r in enumerate(result["results"][:2], 1):
                print(f"\n    Result {i}:")
                print(f"      Score: {r['score']:.4f}")
                print(f"      Source: {r['metadata'].get('file_name', 'unknown')}")
                preview = r['content'][:150].replace('\n', ' ')
                print(f"      Preview: {preview}...")
        else:
            print("  No results found")

    # Demonstrate metadata
    print_step(5, "Examining Document Metadata")

    result = search("AI", top_k=1)
    if result.get("success") and result["count"] > 0:
        metadata = result["results"][0]["metadata"]
        print("\n  Sample document metadata:")
        for key, value in sorted(metadata.items()):
            if isinstance(value, str) and len(str(value)) > 50:
                value = str(value)[:50] + "..."
            print(f"    {key}: {value}")

    # Summary
    print_step(6, "Summary")

    print(f"""
  ╭─────────────────────────────────────────────────────────────╮
  │                    Pipeline Results                         │
  ├─────────────────────────────────────────────────────────────┤
  │  Documents ingested: {len(ingested):<3}                                     │
  │  Chunks created: ~{added:<3}                                       │
  │  Embedding model: {stats['embedding_model'][:30]:<30} │
  │  Search queries: {len(test_queries):<3}                                      │
  ╰─────────────────────────────────────────────────────────────╯
    """)

    print("""
  Chunking Strategies Used:
  • section  - Split on markdown headers (# ## ###)
  • paragraph - Split on double newlines
  • semantic - Token-aware intelligent chunking (SemChunk)

  Also available:
  • sentence - Split on sentence boundaries
  • page - Split on page breaks
  • fixed - Fixed-size character chunks with overlap

  To try different strategies, use:
    curl -X POST "http://localhost:8000/v1/rag-lite/ingest?chunk_strategy=paragraph" \\
         -F "file=@document.txt"
""")

    print_header("Demo Complete!")


if __name__ == "__main__":
    main()
