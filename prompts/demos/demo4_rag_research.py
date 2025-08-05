#!/usr/bin/env python3
"""
Demo 4: RAG-Enhanced Research Assistant

This demo shows:
- RAG (Retrieval-Augmented Generation) strategies
- Working with retrieved documents
- Citation styles and synthesis approaches
- Fallback handling when no context available
- Research vs simple Q&A workflows
"""

import sys
import os
from pathlib import Path
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from demo_helper import setup_demo_environment


def run_demo():
    """Run the demo using the strategy manager."""
    print("=== Demo 4: RAG-Enhanced Research Assistant ===\n")
    
    # Initialize the strategy manager with templates
    manager = setup_demo_environment()
    
    # Example 1: Simple RAG Q&A with inline citations
    print("Example 1: RAG Q&A with Inline Citations")
    print("-" * 50)
    
    retrieved_docs = [
        {
            "title": "Climate Change Overview",
            "content": "Global temperatures have risen by approximately 1.1 degrees Celsius since pre-industrial times.",
            "source": "IPCC Report 2023",
            "relevance_score": 0.92
        },
        {
            "title": "Environmental Impact",
            "content": "Rising temperatures are causing glacial retreat, sea level rise, and changes in precipitation patterns.",
            "source": "Nature Climate Journal",
            "relevance_score": 0.88
        },
        {
            "title": "Mitigation Strategies",
            "content": "Key strategies include renewable energy adoption, reforestation, and carbon capture technologies.",
            "source": "Environmental Science Review",
            "relevance_score": 0.75
        }
    ]
    
    result = manager.execute_strategy(
        strategy_name="rag_qa",
        inputs={
            "query": "What are the main impacts of climate change?",
            "retrieved_documents": retrieved_docs,
            "metadata": {"search_query": "climate change impacts", "num_results": 3}
        }
    )
    
    print("Generated Prompt:")
    print(result)
    print("\n")
    
    # Example 2: Academic research with formal citations
    print("Example 2: Academic Research Assistant")
    print("-" * 50)
    
    academic_docs = [
        {
            "title": "Machine Learning in Healthcare: A Systematic Review",
            "content": "ML applications in healthcare have shown promising results in diagnosis, with accuracy rates exceeding 90% in certain applications.",
            "source": "Journal of Medical AI, 2023, Vol 15, pp. 234-251",
            "authors": "Smith, J., Chen, L., Kumar, R.",
            "relevance_score": 0.95
        },
        {
            "title": "Deep Learning for Medical Image Analysis",
            "content": "Convolutional neural networks have revolutionized medical imaging, particularly in radiology and pathology.",
            "source": "Nature Medicine, 2023, Vol 29, pp. 102-115",
            "authors": "Johnson, M., Lee, S.",
            "relevance_score": 0.91
        }
    ]
    
    result = manager.execute_strategy(
        strategy_name="research_assistant",
        inputs={
            "query": "How is machine learning being applied in healthcare?",
            "retrieved_documents": academic_docs,
            "metadata": {"research_area": "healthcare AI", "publication_years": "2020-2023"}
        }
    )
    
    print("Generated Prompt:")
    print(result)
    print("\n")
    
    # Example 3: Handling no context (fallback)
    print("Example 3: No Context Fallback")
    print("-" * 50)
    
    result = manager.execute_strategy(
        strategy_name="rag_qa",
        inputs={
            "query": "What is quantum computing?",
            "retrieved_documents": [],  # No documents retrieved
            "metadata": {"search_query": "quantum computing", "num_results": 0}
        }
    )
    
    print("Generated Prompt (fallback to simple QA):")
    print(result)
    print("\n")
    
    # Example 4: Large context handling
    print("Example 4: Large Context with Focused Synthesis")
    print("-" * 50)
    
    # Simulate many retrieved documents
    many_docs = [
        {
            "title": f"Research Paper {i}",
            "content": f"Finding {i}: Various aspects of the research topic...",
            "source": f"Journal {i}",
            "relevance_score": 0.9 - (i * 0.05)
        }
        for i in range(1, 12)  # 11 documents
    ]
    
    result = manager.execute_strategy(
        strategy_name="rag_qa",
        inputs={
            "query": "Summarize the key findings",
            "retrieved_documents": many_docs,
            "metadata": {"search_query": "research findings", "num_results": 11}
        },
        context={
            "context_size": {"min": 10}  # Triggers focused synthesis
        }
    )
    
    print("Generated Prompt (focused synthesis for large context):")
    print(result[:500] + "...")  # Truncate for display
    print("\n")
    
    # Example 5: Custom relevance threshold
    print("Example 5: Custom Relevance Filtering")
    print("-" * 50)
    
    mixed_relevance_docs = [
        {"title": "Highly Relevant", "content": "Direct answer to the query", "relevance_score": 0.95},
        {"title": "Somewhat Relevant", "content": "Related information", "relevance_score": 0.65},
        {"title": "Barely Relevant", "content": "Tangential information", "relevance_score": 0.45},
    ]
    
    result = manager.execute_strategy(
        strategy_name="rag_qa",
        inputs={
            "query": "Focus on highly relevant information only",
            "retrieved_documents": mixed_relevance_docs,
            "metadata": {}
        },
        override_config={
            "relevance_threshold": 0.8,  # Only use docs with score >= 0.8
            "synthesis_approach": "focused"
        }
    )
    
    print("Generated Prompt (high relevance threshold):")
    print(result)
    print("\n")
    
    # Example 6: Comparing strategies
    print("Example 6: RAG Q&A vs Research Assistant")
    print("-" * 50)
    
    comparison_docs = [
        {
            "title": "Technical Report",
            "content": "Technical details about the implementation...",
            "source": "Tech Conference 2023",
            "relevance_score": 0.9
        }
    ]
    
    # RAG Q&A version
    rag_result = manager.execute_strategy(
        strategy_name="rag_qa",
        inputs={
            "query": "Explain the technical implementation",
            "retrieved_documents": comparison_docs,
            "metadata": {}
        }
    )
    
    # Research Assistant version
    research_result = manager.execute_strategy(
        strategy_name="research_assistant",
        inputs={
            "query": "Explain the technical implementation",
            "retrieved_documents": comparison_docs,
            "metadata": {}
        }
    )
    
    print("RAG Q&A Strategy (inline citations, comprehensive):")
    print(rag_result[:200] + "...")
    print("\nResearch Assistant Strategy (academic citations, analytical):")
    print(research_result[:200] + "...")
    print("\n")
    
    # Show configuration differences
    print("Strategy Comparison:")
    print("-" * 50)
    
    rag_strategy = manager.get_strategy("rag_qa")
    research_strategy = manager.get_strategy("research_assistant")
    
    print("RAG Q&A:")
    print(f"  - Citation style: {rag_strategy.templates.default.config.get('citation_style', 'inline')}")
    print(f"  - Relevance threshold: {rag_strategy.templates.default.config.get('relevance_threshold', 0.7)}")
    print(f"  - Temperature: {rag_strategy.global_config.temperature}")
    
    print("\nResearch Assistant:")
    print(f"  - Citation style: {research_strategy.templates.default.config.get('citation_style', 'academic')}")
    print(f"  - Include methodology: {research_strategy.templates.default.config.get('include_methodology', False)}")
    print(f"  - Temperature: {research_strategy.global_config.temperature}")
    
    print("\n")
    
    # Teaching moment
    print("💡 Key Concepts:")
    print("- RAG strategies enhance responses with retrieved context")
    print("- Different citation styles for different use cases (inline vs academic)")
    print("- Relevance thresholds filter low-quality documents")
    print("- Synthesis approaches adapt to context size")
    print("- Fallback templates handle cases with no retrieved documents")
    print("- Research assistant includes confidence scores and critical analysis")
    print("- System prompts ensure citations and context-grounded responses")


def main():
    """Main entry point with CLI integration example."""
    print("\nThis demo can also be run via CLI:")
    print("  python -m prompts.cli_strategy strategy execute rag_qa -q 'What are the main impacts?' -c '{\"retrieved_documents\": [...]}'")
    print("  python -m prompts.cli_strategy strategy execute research_assistant -q 'Analyze this research'")
    print("\nRunning demo...\n")
    
    run_demo()


if __name__ == "__main__":
    main()