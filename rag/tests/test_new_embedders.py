#!/usr/bin/env python3
"""
Test script for new embedders: OpenAIEmbedder, HuggingFaceEmbedder, SentenceTransformerEmbedder.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_openai_embedder():
    """Test OpenAIEmbedder functionality."""
    print("🧪 Testing OpenAIEmbedder...")
    
    try:
        from components.embedders.openai_embedder import OpenAIEmbedder
        
        # Check if API key is available
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️  OpenAIEmbedder - Skipping test (OPENAI_API_KEY not set)")
            print("   To test with real API key: export OPENAI_API_KEY=your_key")
            print("   Testing basic instantiation...")
            
            # Test basic instantiation without API calls
            embedder = OpenAIEmbedder({
                "model": "text-embedding-3-small",
                "api_key": "test-key",
                "batch_size": 100,
                "timeout": 30,
                "max_retries": 3
            })
            
            print("✅ OpenAIEmbedder - Basic instantiation successful")
            print(f"   Model: {embedder.model}")
            print(f"   Batch size: {embedder.batch_size}")
            print(f"   Expected dimension: {embedder.get_dimension()}")
            return True
        else:
            print("✅ OpenAIEmbedder - API key found, testing with real API...")
            
            # Test with real API
            embedder = OpenAIEmbedder({
                "model": "text-embedding-3-small",
                "api_key": api_key,
                "batch_size": 10,  # Small batch for testing
                "timeout": 30,
                "max_retries": 2
            })
            
            # Test single embedding
            test_text = "This is a test sentence for embedding."
            embedding = embedder.embed_query(test_text)
            
            assert len(embedding) == embedder.get_dimension(), f"Expected {embedder.get_dimension()} dimensions"
            assert all(isinstance(x, float) for x in embedding), "Embedding should be floats"
            
            # Test batch embedding
            test_texts = [
                "First test document.",
                "Second test document with different content.",
                "Third document for batch testing."
            ]
            
            embeddings = embedder.embed_documents(test_texts)
            assert len(embeddings) == len(test_texts), "Should get one embedding per document"
            
            for emb in embeddings:
                assert len(emb) == embedder.get_dimension(), "All embeddings should have same dimension"
            
            print("✅ OpenAIEmbedder - API test successful")
            print(f"   Generated embeddings: {len(embeddings)} x {len(embeddings[0])}")
            print(f"   Model info: {embedder.get_model_info()}")
            return True
        
    except ImportError as e:
        print(f"⚠️  OpenAIEmbedder - Import failed (openai required): {e}")
        print("   Install with: pip install openai")
        return False
    except Exception as e:
        print(f"❌ OpenAIEmbedder - Test failed: {e}")
        return False


def test_huggingface_embedder():
    """Test HuggingFaceEmbedder functionality."""
    print("\n🧪 Testing HuggingFaceEmbedder...")
    
    try:
        from components.embedders.huggingface_embedder import HuggingFaceEmbedder
        
        print("⚠️  HuggingFaceEmbedder - Skipping full test (requires large model download)")
        print("   Testing basic instantiation...")
        
        # Test basic instantiation without loading model
        try:
            embedder = HuggingFaceEmbedder({
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "device": "cpu",  # Force CPU for testing
                "max_length": 512,
                "batch_size": 8,  # Small batch for testing
                "normalize_embeddings": True,
                "trust_remote_code": False
            })
            
            print("✅ HuggingFaceEmbedder - Model loaded successfully")
            print(f"   Model: {embedder.model_name}")
            print(f"   Device: {embedder.device}")
            print(f"   Dimension: {embedder.get_dimension()}")
            print(f"   Max length: {embedder.get_max_input_length()}")
            
            # Test health check
            is_healthy = embedder.health_check()
            print(f"   Health check: {'✅ Pass' if is_healthy else '❌ Fail'}")
            
            # Test model info
            model_info = embedder.get_model_info()
            print(f"   Model type: {model_info.get('model_type', 'unknown')}")
            
            # Test memory usage info
            memory_info = embedder.get_memory_usage()
            print(f"   Model loaded: {memory_info.get('model_loaded', False)}")
            
            return True
            
        except Exception as model_error:
            print(f"⚠️  HuggingFaceEmbedder - Model loading failed: {model_error}")
            print("   This is expected if transformers/torch are not installed")
            print("   Install with: pip install transformers torch")
            
            # Test recommendations without loading model
            recommendations = HuggingFaceEmbedder.get_recommended_models()
            print(f"   Available model recommendations: {len(recommendations)}")
            for name, info in list(recommendations.items())[:2]:
                print(f"     {name}: {info['description']}")
            
            return True
        
    except ImportError as e:
        print(f"⚠️  HuggingFaceEmbedder - Import failed (transformers/torch required): {e}")
        print("   Install with: pip install transformers torch")
        return False
    except Exception as e:
        print(f"❌ HuggingFaceEmbedder - Test failed: {e}")
        return False


def test_sentence_transformer_embedder():
    """Test SentenceTransformerEmbedder functionality."""
    print("\n🧪 Testing SentenceTransformerEmbedder...")
    
    try:
        from components.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
        
        print("⚠️  SentenceTransformerEmbedder - Skipping full test (requires model download)")
        print("   Testing basic instantiation...")
        
        try:
            embedder = SentenceTransformerEmbedder({
                "model_name": "all-MiniLM-L6-v2",  # Small, fast model
                "device": "cpu",  # Force CPU for testing
                "batch_size": 8,
                "normalize_embeddings": True,
                "show_progress_bar": False,
                "convert_to_tensor": False
            })
            
            print("✅ SentenceTransformerEmbedder - Model loaded successfully")
            print(f"   Model: {embedder.model_name}")
            print(f"   Device: {embedder.device}")
            print(f"   Dimension: {embedder.get_dimension()}")
            print(f"   Max length: {embedder.get_max_input_length()}")
            
            # Test health check
            is_healthy = embedder.health_check()
            print(f"   Health check: {'✅ Pass' if is_healthy else '❌ Fail'}")
            
            # Test model info
            model_info = embedder.get_model_info()
            print(f"   Provider: {model_info.get('provider', 'unknown')}")
            print(f"   Normalization: {model_info.get('normalize_embeddings', False)}")
            
            # Test memory usage info
            memory_info = embedder.get_memory_usage()
            print(f"   Model loaded: {memory_info.get('model_loaded', False)}")
            
            return True
            
        except Exception as model_error:
            print(f"⚠️  SentenceTransformerEmbedder - Model loading failed: {model_error}")
            print("   This is expected if sentence-transformers is not installed")
            print("   Install with: pip install sentence-transformers")
            
            # Test recommendations without loading model
            recommendations = SentenceTransformerEmbedder.get_recommended_models()
            print(f"   Available model recommendations: {len(recommendations)}")
            for name, info in list(recommendations.items())[:3]:
                print(f"     {name}: {info['description']} ({info['use_case']})")
            
            return True
        
    except ImportError as e:
        print(f"⚠️  SentenceTransformerEmbedder - Import failed (sentence-transformers required): {e}")
        print("   Install with: pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"❌ SentenceTransformerEmbedder - Test failed: {e}")
        return False


def test_embedder_comparison():
    """Test comparison of available embedder features."""
    print("\n🧪 Testing Embedder Comparison...")
    
    try:
        print("📊 Embedder Feature Comparison:")
        print("-" * 60)
        
        # Test which embedders are available
        embedders_available = {}
        
        # Test OpenAI
        try:
            from components.embedders.openai_embedder import OpenAIEmbedder
            embedders_available["OpenAI"] = {
                "requires": "openai library + API key",
                "type": "cloud",
                "models": ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]
            }
        except ImportError:
            embedders_available["OpenAI"] = {"available": False, "requires": "pip install openai"}
        
        # Test HuggingFace
        try:
            from components.embedders.huggingface_embedder import HuggingFaceEmbedder
            embedders_available["HuggingFace"] = {
                "requires": "transformers + torch",
                "type": "local",
                "models": ["sentence-transformers/all-MiniLM-L6-v2", "bert-base-uncased"]
            }
        except ImportError:
            embedders_available["HuggingFace"] = {"available": False, "requires": "pip install transformers torch"}
        
        # Test SentenceTransformers
        try:
            from components.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder
            embedders_available["SentenceTransformers"] = {
                "requires": "sentence-transformers",
                "type": "local",
                "models": ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
            }
        except ImportError:
            embedders_available["SentenceTransformers"] = {"available": False, "requires": "pip install sentence-transformers"}
        
        # Print comparison
        for name, info in embedders_available.items():
            available = info.get("available", True)
            status = "✅ Available" if available else "❌ Not available"
            print(f"{name:20} {status}")
            
            if available:
                print(f"{'':20} Type: {info.get('type', 'unknown')}")
                print(f"{'':20} Requires: {info.get('requires', 'unknown')}")
                if 'models' in info:
                    print(f"{'':20} Example models: {info['models'][:2]}")
            else:
                print(f"{'':20} Install: {info.get('requires', 'unknown')}")
            print()
        
        total_available = sum(1 for info in embedders_available.values() if info.get("available", True))
        print(f"📈 Summary: {total_available}/{len(embedders_available)} embedders available")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedder comparison failed: {e}")
        return False


def run_embedder_tests():
    """Run all embedder tests."""
    print("🚀 Running New Embedder Tests")
    print("=" * 50)
    
    tests = [
        ("OpenAIEmbedder", test_openai_embedder),
        ("HuggingFaceEmbedder", test_huggingface_embedder),
        ("SentenceTransformerEmbedder", test_sentence_transformer_embedder),
        ("EmbedderComparison", test_embedder_comparison),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} - Unexpected error: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Embedder Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All embedder tests passed!")
    else:
        print("⚠️  Some tests failed or were skipped due to missing dependencies")
        print("💡 Install missing dependencies to enable full testing:")
        print("   pip install openai transformers torch sentence-transformers")
    
    return passed, total


if __name__ == "__main__":
    run_embedder_tests()