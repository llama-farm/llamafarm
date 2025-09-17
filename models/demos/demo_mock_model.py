#!/usr/bin/env python3
"""
Mock Model Demo - Fast Development and Testing
Shows how to use mock models for rapid prototyping without API costs or setup.
"""

import yaml
import time
from datetime import datetime
from pathlib import Path

def create_mock_response(query: str, model: str = "mock-gpt-4") -> str:
    """Create a mock response based on the query."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Estimate tokens (rough approximation)
    estimated_tokens = len(query.split()) * 15 + 50
    
    # Simulate response time
    response_time = int(time.time() * 1000) % 100 + 50  # 50-150ms
    
    template = """🤖 **Mock Response** - {timestamp}

**Your Query:** "{query}"

**Mock Answer:** This is a simulated response for development and testing purposes. 
In a production environment, this would be processed by the {model} model and provide 
a real, contextual response to your query.

**Metadata:**
- Model: {model}
- Estimated Tokens: ~{estimated_tokens}
- Response Time: {response_time}ms
- Status: ✅ Mock Success

💡 **Development Tip:** Use this mock setup to build and test your application logic 
before switching to real models!"""

    return template.format(
        timestamp=timestamp,
        query=query,
        model=model,
        estimated_tokens=estimated_tokens,
        response_time=response_time
    )

def demo_mock_model():
    """Demonstrate mock model capabilities."""
    print("=" * 60)
    print("🤖 MOCK MODEL DEMO - Fast Development & Testing")
    print("=" * 60)
    print()
    
    # Mock models are built-in and require no setup!
    print("✨ **No setup required!** Mock models are built into LlamaFarm.")
    print("   Perfect for immediate testing without any API keys or downloads.")
    print()
    
    # Load the mock strategy from main strategies file
    strategy_path = Path(__file__).parent / "strategies.yaml"
    with open(strategy_path) as f:
        strategies = yaml.safe_load(f)
    
    mock_strategy = strategies['strategies']['mock_development']
    # Use default test queries if not in strategy
    test_queries = mock_strategy.get('test_queries', [
        "Hello, how are you?",
        "Explain machine learning",
        "Write a Python function",
        "What's the weather like?",
        "Tell me a joke"
    ])
    
    print(f"📋 **Strategy:** {mock_strategy['name']}")
    print(f"📝 **Description:** {mock_strategy['description']}")
    print()
    
    # Get available models if configured
    model_config = mock_strategy['components']['model_app']['config']
    if 'models' in model_config:
        models = model_config['models']
        print("🤖 **Available Mock Models:**")
        for model in models:
            capabilities = ", ".join(model.get('capabilities', ['chat']))
            print(f"  • {model['name']}: {capabilities}")
    else:
        print("🤖 **Available Mock Models:**")
        print(f"  • {model_config.get('default_model', 'mock-gpt-4')}: chat, completion")
    print()
    
    print("🧪 **Running Test Queries:**")
    print()
    
    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}: {query}")
        print("-" * 40)
        
        # Simulate small delay for realism
        time.sleep(0.1)
        
        # Generate mock response
        response = create_mock_response(query)
        print(response)
        print()
    
    print("=" * 60)
    print("✅ **Mock Demo Complete**")
    print()
    print("🎯 **Key Benefits of Mock Models:**")
    print("  • ⚡ Instant responses - no API calls")
    print("  • 💰 Zero cost - no API usage fees") 
    print("  • 🔒 No API keys required")
    print("  • 🚀 Perfect for development and testing")
    print("  • 🧪 Consistent responses for unit tests")
    print("  • 📱 Works offline")
    print()
    print("💡 **Next Steps:**")
    print("  1. Use mock models to build your application logic")
    print("  2. Test your workflows without external dependencies")
    print("  3. Switch to real models when ready for production")
    print("  4. Keep mock as fallback for testing")

def interactive_mock():
    """Interactive mock model session."""
    print("\n🎮 **Interactive Mock Session** (type 'quit' to exit)")
    print("=" * 50)
    
    available_models = ["mock-gpt-4", "mock-claude-3", "mock-tiny"]
    current_model = "mock-gpt-4"
    
    while True:
        print(f"\n[{current_model}] Enter your query:")
        user_input = input("> ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
            
        if user_input.lower().startswith('model:'):
            # Allow model switching
            new_model = user_input[6:].strip()
            if new_model in available_models:
                current_model = new_model
                print(f"✅ Switched to {current_model}")
            else:
                print(f"❌ Unknown model. Available: {', '.join(available_models)}")
            continue
            
        if not user_input:
            continue
            
        # Generate and display response
        print()
        response = create_mock_response(user_input, current_model)
        print(response)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mock()
    else:
        demo_mock_model()
        
        # Offer interactive mode
        print("\n🎮 Want to try interactive mode?")
        response = input("Run interactive session? (y/n): ").lower()
        if response == 'y':
            interactive_mock()
            
    print("\n👋 Mock demo finished!")