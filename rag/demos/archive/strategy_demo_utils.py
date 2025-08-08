"""
Demo Strategy Utilities

Utility functions to help demos use the strategy-first approach.
This makes demos cleaner and showcases the power of the strategy system.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.manager import StrategyManager
from core.factories import create_embedder_from_config, create_vector_store_from_config, create_retrieval_strategy_from_config, create_parser_from_config, create_extractor_from_config
from core.base import Document


class StrategyDemoSystem:
    """
    A simplified system for demos that uses strategies to initialize components.
    
    This showcases how easy it is to set up a complete RAG system using 
    the strategy-first approach.
    """
    
    def __init__(self, strategy_name: str, strategies_file: Optional[str] = None):
        """
        Initialize demo system with a strategy.
        
        Args:
            strategy_name: Name of the strategy to use
            strategies_file: Optional custom strategies file (defaults to demo_strategies.yaml)
        """
        self.strategy_name = strategy_name
        
        # Use demo strategies file by default
        if strategies_file is None:
            strategies_file = str(Path(__file__).parent / "demo_strategies.yaml")
        
        self.strategy_manager = StrategyManager(strategies_file)
        
        # Convert strategy to configuration
        config = self.strategy_manager.convert_strategy_to_config(strategy_name)
        if not config:
            available = self.strategy_manager.get_available_strategies()
            raise ValueError(f"Strategy '{strategy_name}' not found. Available: {available}")
        
        self.config = config
        self.strategy_info = config.pop("_strategy_info", {})
        
        # Initialize components from strategy
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all components from strategy configuration."""
        
        # Create parser
        parser_config = self.config.get("parser", {})
        self.parser = create_parser_from_config(parser_config)
        
        # Create extractors
        extractors_config = self.config.get("extractors", [])
        self.extractors = []
        for extractor_config in extractors_config:
            extractor = create_extractor_from_config(extractor_config)
            self.extractors.append(extractor)
        
        # Create embedder
        embedder_config = self.config.get("embedder", {})
        self.embedder = create_embedder_from_config(embedder_config)
        
        # Create vector store
        store_config = self.config.get("vector_store", {})
        self.vector_store = create_vector_store_from_config(store_config)
        
        # Create retrieval strategy if specified
        retrieval_config = self.config.get("retrieval_strategy", {})
        self.retrieval_strategy = None
        if retrieval_config:
            self.retrieval_strategy = create_retrieval_strategy_from_config(retrieval_config)
    
    def process_documents(self, file_paths: List[str], show_progress: bool = True) -> List:
        """
        Process documents using strategy-configured components.
        
        Args:
            file_paths: List of file paths to process
            show_progress: Whether to show progress information
            
        Returns:
            List of processed documents
        """
        if show_progress:
            print(f"🔧 Processing documents with '{self.strategy_name}' strategy...")
        
        all_documents = []
        
        for file_path in file_paths:
            if not Path(file_path).exists():
                if show_progress:
                    print(f"⚠️  File not found: {file_path}")
                continue
            
            # Parse the document using file path (parsers expect file paths)
            parsed_result = self.parser.parse(Path(file_path))
            
            # Handle different return types from parsers
            if hasattr(parsed_result, 'documents'):
                parsed_docs = parsed_result.documents
            elif isinstance(parsed_result, list):
                parsed_docs = parsed_result
            else:
                parsed_docs = [parsed_result] if parsed_result else []
            
            # Apply extractors to each document
            for doc in parsed_docs:
                doc.source = file_path  # Set source
                
                # Apply each extractor
                for extractor in self.extractors:
                    try:
                        # Pass the document as a list - extractors expect List[Document]
                        extracted_docs = extractor.extract([doc])
                        
                        # Get the processed document back (should be the same doc, enhanced)
                        if extracted_docs and len(extracted_docs) > 0:
                            enhanced_doc = extracted_docs[0]
                            # The extractor should have added metadata to the doc directly
                            # So we just update our doc reference
                            doc = enhanced_doc
                        
                    except Exception as e:
                        if show_progress:
                            print(f"⚠️  Extractor {extractor.__class__.__name__} failed: {e}")
            
            all_documents.extend(parsed_docs)
            
            if show_progress:
                print(f"📄 Processed {Path(file_path).name}: {len(parsed_docs)} chunks")
        
        return all_documents
    
    def generate_embeddings(self, documents: List, show_progress: bool = True) -> List:
        """
        Generate embeddings for documents using strategy-configured embedder.
        
        Args:
            documents: List of documents to embed
            show_progress: Whether to show progress information
            
        Returns:
            Documents with embeddings added
        """
        if show_progress:
            print(f"🧠 Generating embeddings using {self.embedder.__class__.__name__}...")
        
        for doc in documents:
            if not doc.embeddings:
                embeddings = self.embedder.embed([doc.content])
                doc.embeddings = embeddings[0] if embeddings else []
        
        if show_progress:
            print(f"✅ Generated embeddings for {len(documents)} documents")
        
        return documents
    
    def store_documents(self, documents: List, show_progress: bool = True) -> bool:
        """
        Store documents in strategy-configured vector store.
        
        Args:
            documents: List of documents to store
            show_progress: Whether to show progress information
            
        Returns:
            True if successful, False otherwise
        """
        if show_progress:
            print(f"💾 Storing documents using {self.vector_store.__class__.__name__}...")
        
        success = self.vector_store.add_documents(documents)
        
        if show_progress:
            if success:
                print(f"✅ Stored {len(documents)} documents successfully")
            else:
                print("❌ Failed to store documents")
        
        return success
    
    def search(self, query: str, top_k: Optional[int] = None, show_progress: bool = True) -> List:
        """
        Search documents using strategy-configured components.
        
        Args:
            query: Search query
            top_k: Number of results (uses strategy default if None)
            show_progress: Whether to show progress information
            
        Returns:
            List of search results
        """
        if show_progress:
            print(f"🔍 Searching with query: '{query}'")
        
        # Use strategy's default top_k if not specified
        if top_k is None:
            retrieval_config = self.config.get("retrieval_strategy", {}).get("config", {})
            top_k = retrieval_config.get("top_k", 3)
        
        # Generate query embedding
        query_embeddings = self.embedder.embed([query])
        query_embedding = query_embeddings[0] if query_embeddings else []
        
        # Search using retrieval strategy or direct vector store search
        if self.retrieval_strategy:
            if show_progress:
                print(f"🎯 Using {self.retrieval_strategy.__class__.__name__} for search")
            
            # Use retrieval strategy
            retrieval_result = self.retrieval_strategy.retrieve(
                query_embedding=query_embedding,
                vector_store=self.vector_store,
                top_k=top_k
            )
            
            if hasattr(retrieval_result, 'documents'):
                results = retrieval_result.documents
            else:
                results = retrieval_result if isinstance(retrieval_result, list) else [retrieval_result]
        else:
            # Direct vector store search
            results = self.vector_store.search(query_embedding=query_embedding, top_k=top_k)
        
        if show_progress:
            print(f"📊 Found {len(results)} results")
        
        return results
    
    def cleanup(self, show_progress: bool = True):
        """Clean up vector database to prevent duplicate accumulation."""
        if show_progress:
            print("🧹 Cleaning up vector database...")
        
        try:
            self.vector_store.delete_collection()
            if show_progress:
                print("✅ Database cleaned successfully!")
        except Exception as e:
            if show_progress:
                print(f"⚠️  Note: Could not clean database: {e}")
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about the current strategy."""
        return self.strategy_info
    
    def print_strategy_info(self):
        """Print information about the current strategy."""
        info = self.get_strategy_info()
        print(f"\n🚀 Strategy: {info.get('name', self.strategy_name)}")
        print(f"📝 Description: {info.get('description', 'N/A')}")
        print(f"🎯 Use Cases: {', '.join(info.get('use_cases', []))}")
        print(f"⚡ Performance: {info.get('performance_priority', 'N/A')}")
        print(f"💾 Resources: {info.get('resource_usage', 'N/A')}")
        print(f"🔧 Complexity: {info.get('complexity', 'N/A')}")


def create_demo_system(strategy_name: str, strategies_file: Optional[str] = None) -> StrategyDemoSystem:
    """
    Convenience function to create a demo system with a strategy.
    
    Args:
        strategy_name: Name of the strategy to use
        strategies_file: Optional custom strategies file
        
    Returns:
        Initialized StrategyDemoSystem
    """
    return StrategyDemoSystem(strategy_name, strategies_file)


def list_demo_strategies(strategies_file: Optional[str] = None) -> List[str]:
    """
    List available demo strategies.
    
    Args:
        strategies_file: Optional custom strategies file
        
    Returns:
        List of strategy names
    """
    if strategies_file is None:
        strategies_file = str(Path(__file__).parent / "demo_strategies.yaml")
    
    manager = StrategyManager(strategies_file)
    return manager.get_available_strategies()


def print_demo_strategies(strategies_file: Optional[str] = None):
    """Print information about all demo strategies."""
    if strategies_file is None:
        strategies_file = str(Path(__file__).parent / "demo_strategies.yaml")
    
    manager = StrategyManager(strategies_file)
    
    print("\n🚀 Available Demo Strategies:")
    print("=" * 50)
    
    strategies = manager.get_available_strategies()
    for strategy_name in sorted(strategies):
        info = manager.get_strategy_info(strategy_name)
        if info:
            print(f"\n{strategy_name}:")
            print(f"  Description: {info['description']}")
            print(f"  Use Cases: {', '.join(info['use_cases'])}")
            print(f"  Performance: {info['performance_priority']} | Resources: {info['resource_usage']}")


# Example usage for demos:
if __name__ == "__main__":
    # Show available strategies
    print_demo_strategies()
    
    # Example of using a strategy
    print("\n" + "="*60)
    print("Example: Using the research_papers_demo strategy")
    print("="*60)
    
    try:
        # Create system with strategy
        system = create_demo_system("research_papers_demo")
        system.print_strategy_info()
        
        print("\n✅ Strategy system initialized successfully!")
        print("📝 This demonstrates how easy it is to set up a complete RAG system")
        print("🔧 All components are configured from the strategy file")
        print("🚀 No hardcoded parameters in the demo code!")
    
    except Exception as e:
        print(f"❌ Error: {e}")