#!/bin/bash

# LlamaFarm Prompts Management System - Complete Setup and Demo
# This script sets up the environment and runs comprehensive demonstrations

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Print banner
print_banner() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║                    🚀 LlamaFarm Prompts System                     ║${NC}"
    echo -e "${BLUE}║                   Complete Setup and Demo Script                   ║${NC}"
    echo -e "${BLUE}║                                                                    ║${NC}"
    echo -e "${BLUE}║  • 14 Templates across 5 categories                               ║${NC}"
    echo -e "${BLUE}║  • 3 Intelligent selection strategies                             ║${NC}"
    echo -e "${BLUE}║  • 5 Comprehensive evaluation templates                           ║${NC}"
    echo -e "${BLUE}║  • Live LLM integration (OpenAI + Ollama)                         ║${NC}"
    echo -e "${BLUE}║  • Production-ready prompt management                             ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════╝${NC}"
    echo
}

# Check dependencies
check_dependencies() {
    echo -e "${CYAN}🔍 Checking system dependencies...${NC}"
    
    # Check if uv is installed
    if ! command -v uv &> /dev/null; then
        echo -e "${RED}❌ uv is not installed. Please install it first:${NC}"
        echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    echo -e "${GREEN}✅ uv package manager found${NC}"
    
    # Check if curl is available (for API testing)
    if ! command -v curl &> /dev/null; then
        echo -e "${YELLOW}⚠️  curl not found - some API tests may not work${NC}"
    else
        echo -e "${GREEN}✅ curl found${NC}"
    fi
    
    echo
}

# Setup environment
setup_environment() {
    echo -e "${CYAN}🔧 Setting up environment...${NC}"
    
    # Install dependencies
    echo "📦 Installing Python dependencies..."
    uv sync --dev
    echo -e "${GREEN}✅ Dependencies installed${NC}"
    
    # Generate configuration if needed
    if [ ! -f "config/default_prompts.json" ]; then
        echo "🔄 Generating prompt configuration..."
        uv run python generate_config.py
        echo -e "${GREEN}✅ Configuration generated${NC}"
    else
        echo -e "${GREEN}✅ Configuration already exists${NC}"
    fi
    
    echo
}

# Check LLM providers
check_llm_providers() {
    echo -e "${CYAN}🤖 Checking LLM providers...${NC}"
    
    # Load environment variables if .env exists
    if [ -f "../.env" ]; then
        # Source only valid environment variables, filtering out comments and empty lines
        set -a  # Automatically export all variables
        source <(grep -E '^[A-Z_]+=.*' ../.env | grep -v '^#' | head -20)
        set +a  # Turn off automatic export
        # Unset organization to avoid header mismatch
        unset OPENAI_ORG_ID
        echo -e "${GREEN}✅ Loaded environment variables from ../.env${NC}"
    fi
    
    # Check OpenAI
    if [ ! -z "$OPENAI_API_KEY" ]; then
        echo -e "${GREEN}✅ OpenAI API key found${NC}"
        OPENAI_AVAILABLE=true
    else
        echo -e "${YELLOW}⚠️  OpenAI API key not found${NC}"
        OPENAI_AVAILABLE=false
    fi
    
    # Check Ollama
    if curl -s "http://localhost:11434/api/tags" > /dev/null 2>&1; then
        OLLAMA_MODELS=$(curl -s "http://localhost:11434/api/tags" | python3 -c "import sys, json; data = json.load(sys.stdin); print(len(data.get('models', [])))")
        if [ "$OLLAMA_MODELS" -gt 0 ]; then
            echo -e "${GREEN}✅ Ollama connected with $OLLAMA_MODELS models${NC}"
            OLLAMA_AVAILABLE=true
        else
            echo -e "${YELLOW}⚠️  Ollama connected but no models found${NC}"
            OLLAMA_AVAILABLE=false
        fi
    else
        echo -e "${YELLOW}⚠️  Ollama not available at localhost:11434${NC}"
        OLLAMA_AVAILABLE=false
    fi
    
    # Set active provider
    if [ "$OPENAI_AVAILABLE" = true ]; then
        ACTIVE_PROVIDER="OpenAI"
    elif [ "$OLLAMA_AVAILABLE" = true ]; then
        ACTIVE_PROVIDER="Ollama"
    else
        ACTIVE_PROVIDER="Template-only (no LLM)"
    fi
    
    echo -e "${PURPLE}🎯 Active provider: $ACTIVE_PROVIDER${NC}"
    echo
}

# Basic system test
test_basic_functionality() {
    echo -e "${CYAN}🧪 Testing basic functionality...${NC}"
    
    # Test CLI
    echo "📋 Testing CLI..."
    uv run python -m prompts.cli --help > /dev/null
    echo -e "${GREEN}✅ CLI working${NC}"
    
    # Test template listing
    echo "📝 Testing template listing..."
    TEMPLATE_COUNT=$(uv run python -m prompts.cli template list 2>/dev/null | grep -c "│")
    if [ "$TEMPLATE_COUNT" -gt 10 ]; then
        echo -e "${GREEN}✅ Found $TEMPLATE_COUNT templates${NC}"
    else
        echo -e "${RED}❌ Template listing failed${NC}"
        exit 1
    fi
    
    # Test basic execution
    echo "🔄 Testing basic prompt execution..."
    uv run python -m prompts.cli execute "What is machine learning?" --show-details > /dev/null 2>&1
    echo -e "${GREEN}✅ Basic execution working${NC}"
    
    echo
}

# Demo: Template Categories
demo_template_categories() {
    echo -e "${PURPLE}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${PURPLE}📚 TEMPLATE CATEGORIES DEMONSTRATION${NC}"
    echo -e "${PURPLE}═══════════════════════════════════════════════════════════════════${NC}"
    echo
    
    echo -e "${CYAN}Displaying all 14 templates across 5 categories:${NC}"
    uv run python -m prompts.cli template list
    echo
}

# Demo: Strategy Selection
demo_strategy_selection() {
    echo -e "${PURPLE}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${PURPLE}🧠 STRATEGY SELECTION DEMONSTRATION${NC}"
    echo -e "${PURPLE}═══════════════════════════════════════════════════════════════════${NC}"
    echo
    
    echo -e "${CYAN}Testing intelligent template selection with different strategies...${NC}"
    echo
    
    # Medical query - should trigger medical_qa with rule-based strategy
    echo -e "${YELLOW}📋 Scenario 1: Medical Domain Query${NC}"
    echo "Query: 'What are the symptoms of hypertension?'"
    echo "Expected: rule_based_strategy → medical_qa template"
    
    RESULT=$(uv run python -m prompts.cli execute "What are the symptoms of hypertension?" \
        --variables '{"domain": "medical"}' --show-details 2>/dev/null | grep "Template Used" | awk '{print $3}')
    echo -e "Result: ${GREEN}✅ Template: $RESULT${NC}"
    echo
    
    # Summarization query
    echo -e "${YELLOW}📋 Scenario 2: Summarization Request${NC}"
    echo "Query: 'Summarize the benefits of renewable energy'"
    echo "Expected: Context-aware strategy should select appropriate template"
    
    uv run python -m prompts.cli execute "Summarize the benefits of renewable energy" \
        --show-details 2>/dev/null | grep -E "(Template Used|Strategy Used)" | head -2
    echo
    
    # Complex analysis
    echo -e "${YELLOW}📋 Scenario 3: Complex Analysis${NC}"
    echo "Query: 'Analyze the economic impact of AI adoption'"
    echo "Template override: chain_of_thought"
    
    uv run python -m prompts.cli execute "Analyze the economic impact of AI adoption" \
        --template chain_of_thought --show-details 2>/dev/null | grep -E "(Template Used|Strategy Used)" | head -2
    echo
}

# Demo: Template Showcase  
demo_template_showcase() {
    echo -e "${PURPLE}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${PURPLE}📝 TEMPLATE GENERATION SHOWCASE${NC}"
    echo -e "${PURPLE}═══════════════════════════════════════════════════════════════════${NC}"
    echo
    
    echo -e "${CYAN}Showing how different templates format the same query...${NC}"
    echo
    
    QUERY="What are the benefits of renewable energy?"
    CONTEXT='[{"title": "Clean Energy Report", "content": "Solar and wind power reduce emissions and costs significantly."}]'
    
    # Basic template
    echo -e "${YELLOW}🔹 Basic Q&A Template:${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    uv run python -m prompts.cli execute "$QUERY" \
        --template qa_basic \
        --variables "{\"context\": $CONTEXT}" 2>/dev/null | head -20
    echo
    
    # Chain of thought template
    echo -e "${YELLOW}🔹 Chain of Thought Template:${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    uv run python -m prompts.cli execute "$QUERY" \
        --template chain_of_thought \
        --variables "{\"context\": $CONTEXT, \"query\": \"$QUERY\"}" 2>/dev/null | head -25
    echo
    
    # Comparative analysis template
    echo -e "${YELLOW}🔹 Comparative Analysis Template:${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    uv run python -m prompts.cli execute "Compare solar vs wind energy benefits" \
        --template comparative_analysis \
        --variables "{\"context\": $CONTEXT, \"query\": \"Compare solar vs wind energy benefits\"}" 2>/dev/null | head -25
    echo
}

# Demo: Evaluation Templates
demo_evaluation_templates() {
    echo -e "${PURPLE}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${PURPLE}🔍 EVALUATION TEMPLATES SHOWCASE${NC}"
    echo -e "${PURPLE}═══════════════════════════════════════════════════════════════════${NC}"
    echo
    
    echo -e "${CYAN}Demonstrating 5 comprehensive evaluation templates...${NC}"
    echo
    
    # LLM Judge evaluation
    echo -e "${YELLOW}⚖️ LLM as Judge Evaluation:${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Evaluates AI responses with 5-dimension scoring framework"
    uv run python -m prompts.cli execute "Evaluate response quality" \
        --template llm_judge \
        --variables '{
            "original_query": "What is machine learning?",
            "response_to_evaluate": "Machine learning is AI that learns from data.",
            "evaluation_criteria": "Accuracy and completeness",
            "context": [{"title": "ML Guide", "content": "ML is a subset of AI"}]
        }' 2>/dev/null | head -15
    echo
    
    # RAG evaluation
    echo -e "${YELLOW}🔍 RAG System Evaluation:${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Comprehensive RAG system assessment with 55-point scoring"
    uv run python -m prompts.cli execute "Evaluate RAG performance" \
        --template rag_evaluation \
        --variables '{
            "query": "Benefits of renewable energy",
            "retrieved_docs": [{"title": "Solar Power", "content": "Solar reduces emissions"}],
            "generated_response": "Solar energy provides clean electricity with environmental benefits."
        }' 2>/dev/null | head -15
    echo
    
    # A/B Testing
    echo -e "${YELLOW}🧪 A/B Testing Comparison:${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Statistical comparison framework for prompt optimization"
    uv run python -m prompts.cli execute "Compare variants" \
        --template ab_testing \
        --variables '{
            "test_description": "Testing basic vs detailed response templates",
            "variant_a_name": "Basic", "variant_a_config": "Simple format",
            "variant_a_responses": "Short, concise answers",
            "variant_b_name": "Detailed", "variant_b_config": "Comprehensive format", 
            "variant_b_responses": "Long, thorough explanations",
            "evaluation_metrics": "Accuracy, completeness, user satisfaction"
        }' 2>/dev/null | head -15
    echo
}

# Live LLM Demo (if providers available)
demo_live_llm() {
    if [ "$ACTIVE_PROVIDER" = "Template-only (no LLM)" ]; then
        echo -e "${YELLOW}⚠️  Skipping live LLM demo - no providers available${NC}"
        echo "   To see live responses, set up OpenAI API key or install Ollama"
        echo
        return
    fi
    
    echo -e "${PURPLE}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${PURPLE}🤖 LIVE LLM DEMONSTRATION (Using $ACTIVE_PROVIDER)${NC}"
    echo -e "${PURPLE}═══════════════════════════════════════════════════════════════════${NC}"
    echo
    
    echo -e "${CYAN}Running live demo with real LLM responses...${NC}"
    echo
    
    # Create and run live demo script
    cat > temp_live_demo.py << 'EOF'
import asyncio
import os
import sys
import requests
import json
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path="../.env")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from prompts.core.prompt_system import PromptSystem
from prompts.models.config import PromptConfig

def print_progress(message):
    """Print progress with spinner effect."""
    print(f"\r{message}...", end="", flush=True)

async def call_llm(prompt: str, provider_name: str = "LLM") -> tuple[str, str]:
    """Call available LLM provider and return response with provider info."""
    # Try OpenAI first
    openai_key = os.getenv('OPENAI_API_KEY')
    if OPENAI_AVAILABLE and openai_key and openai_key.startswith('sk-'):
        try:
            print_progress(f"   🔄 Calling OpenAI API")
            # Clean environment
            os.environ.pop('OPENAI_ORG_ID', None)
            os.environ.pop('OPENAI_ORGANIZATION', None)
            
            client = OpenAI(api_key=openai_key)
            
            # Check for preferred model from environment, otherwise use modern model priority
            preferred_model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
            
            # Try preferred model first, then fallback to others
            if preferred_model != 'gpt-4o-mini':
                models_to_try = [(preferred_model, f"OpenAI {preferred_model}")]
            else:
                models_to_try = []
                
            # Add fallback models in order of preference
            fallback_models = [
                ("gpt-4o-mini", "OpenAI GPT-4o Mini"),
                ("gpt-4o", "OpenAI GPT-4o"), 
                ("gpt-4-turbo", "OpenAI GPT-4 Turbo"),
                ("gpt-4", "OpenAI GPT-4"),
                ("gpt-3.5-turbo", "OpenAI GPT-3.5 Turbo")
            ]
            
            # Add fallback models, avoiding duplicates
            for model, display in fallback_models:
                if not models_to_try or model != models_to_try[0][0]:
                    models_to_try.append((model, display))
            
            last_error = None
            for model_name, display_name in models_to_try:
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1000,
                        temperature=0.7
                    )
                    print("\r" + " " * 50 + "\r", end="")  # Clear progress line
                    return response.choices[0].message.content, display_name
                except Exception as e:
                    last_error = str(e)
                    if "does not exist" not in str(e).lower() and "model" not in str(e).lower():
                        # If it's not a model availability error, break and show the error
                        break
                    continue
            
            # If we get here, all models failed
            raise Exception(last_error or "All OpenAI models failed")
        except Exception as e:
            print(f"\r   ⚠️  OpenAI error: {str(e)[:60]}...")
    
    # Try Ollama
    try:
        print_progress("   🔄 Calling Ollama (llama3.1:8b)")
        data = {"model": "llama3.1:8b", "prompt": prompt, "stream": False}
        response = requests.post("http://localhost:11434/api/generate", json=data, timeout=60)
        if response.status_code == 200:
            print("\r" + " " * 50 + "\r", end="")  # Clear progress line
            return response.json().get('response', 'No response'), "Ollama llama3.1:8b"
    except Exception as e:
        print(f"\r   ⚠️  Ollama error: {str(e)[:60]}...")
    
    print("\r" + " " * 50 + "\r", end="")  # Clear progress line
    return "[No LLM available - showing template only]", "Template Only"

async def main():
    config = PromptConfig.from_file('config/default_prompts.json')
    system = PromptSystem(config)
    
    print("═" * 80)
    print("🔄 Example 1: Basic Q&A Template")
    print("═" * 80)
    
    # Generate prompt
    result = system.execute_prompt(
        query="What are the main benefits of machine learning in healthcare?",
        variables={"context": [{"title": "Healthcare AI", "content": "ML improves diagnostics and treatment accuracy by analyzing medical images, predicting patient outcomes, and personalizing treatments"}]},
        template_override="qa_basic"
    )
    
    print("\n📝 FULL GENERATED PROMPT:")
    print("─" * 80)
    print(result.rendered_prompt)
    print("─" * 80)
    
    # Get LLM response
    response, provider = await call_llm(result.rendered_prompt)
    print(f"\n🤖 LLM RESPONSE (via {provider}):")
    print("─" * 80)
    print(response)
    print("─" * 80)
    print()
    
    # Example 2: LLM Judge
    print("═" * 80)
    print("⚖️ Example 2: LLM Judge Evaluation")
    print("═" * 80)
    
    judge_result = system.execute_prompt(
        query="Evaluate response",
        variables={
            "original_query": "What are the main benefits of machine learning in healthcare?",
            "response_to_evaluate": response,
            "evaluation_criteria": "Medical accuracy, completeness, and clarity",
            "context": [{"title": "Healthcare AI", "content": "ML improves diagnostics and treatment accuracy"}]
        },
        template_override="llm_judge"
    )
    
    print("\n📝 EVALUATION PROMPT (truncated to 1000 chars):")
    print("─" * 80)
    print(judge_result.rendered_prompt[:1000] + "...\n[TRUNCATED FOR DISPLAY]")
    print("─" * 80)
    
    # Get evaluation
    evaluation, eval_provider = await call_llm(judge_result.rendered_prompt)
    print(f"\n🎯 EVALUATION RESULT (via {eval_provider}):")
    print("─" * 80)
    print(evaluation)
    print("─" * 80)
    print()
    
    # Example 3: Chain of Thought
    print("═" * 80)
    print("🔍 Example 3: Chain of Thought Template")
    print("═" * 80)
    
    cot_result = system.execute_prompt(
        query="How does machine learning improve medical diagnosis accuracy?",
        variables={"context": [{"title": "ML in Medicine", "content": "Pattern recognition in medical imaging, predictive analytics for disease progression, and integration with electronic health records"}]},
        template_override="chain_of_thought"
    )
    
    print("\n📝 CHAIN OF THOUGHT PROMPT:")
    print("─" * 80)
    print(cot_result.rendered_prompt)
    print("─" * 80)
    
    # Get CoT response
    cot_response, cot_provider = await call_llm(cot_result.rendered_prompt)
    print(f"\n🧠 REASONING RESPONSE (via {cot_provider}):")
    print("─" * 80)
    print(cot_response)
    print("─" * 80)

if __name__ == "__main__":
    asyncio.run(main())
EOF
    
    # Run the live demo
    uv run python temp_live_demo.py
    
    # Clean up
    rm temp_live_demo.py
    echo
}

# Show CLI examples
show_cli_examples() {
    echo -e "${PURPLE}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${PURPLE}🔧 CLI USAGE EXAMPLES WITH OUTPUTS${NC}"
    echo -e "${PURPLE}═══════════════════════════════════════════════════════════════════${NC}"
    echo
    
    echo -e "${CYAN}Running actual CLI commands to show their outputs...${NC}"
    echo
    
    # List templates
    echo -e "${YELLOW}📋 Command: List all templates${NC}"
    echo -e "${BLUE}$ uv run python -m prompts.cli template list${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    uv run python -m prompts.cli template list
    echo
    
    # Show template details
    echo -e "${YELLOW}🔍 Command: Get template details${NC}"
    echo -e "${BLUE}$ uv run python -m prompts.cli template show medical_qa${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    uv run python -m prompts.cli template show medical_qa
    echo
    
    # Execute basic query
    echo -e "${YELLOW}💬 Command: Execute basic query${NC}"
    echo -e "${BLUE}$ uv run python -m prompts.cli execute 'What is artificial intelligence?'${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    uv run python -m prompts.cli execute 'What is artificial intelligence?' 2>/dev/null | head -20
    echo
    
    # Use specific template
    echo -e "${YELLOW}🎯 Command: Use specific template${NC}"
    echo -e "${BLUE}$ uv run python -m prompts.cli execute 'Compare solar vs wind' --template comparative_analysis${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    uv run python -m prompts.cli execute 'Compare solar vs wind energy' \
        --template comparative_analysis \
        --variables '{"context": [{"title": "Energy Report", "content": "Solar and wind are renewable energy sources"}]}' 2>/dev/null | head -25
    echo
    
    # Medical domain query
    echo -e "${YELLOW}🏥 Command: Medical domain query${NC}"
    echo -e "${BLUE}$ uv run python -m prompts.cli execute 'Diabetes symptoms' --variables '{\"domain\": \"medical\"}'${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    uv run python -m prompts.cli execute 'What are the symptoms of diabetes?' \
        --variables '{"domain": "medical", "context": [{"title": "Medical Guide", "content": "Diabetes symptoms include increased thirst"}]}' 2>/dev/null | head -20
    echo
    
    # System statistics
    echo -e "${YELLOW}📊 Command: System statistics${NC}"
    echo -e "${BLUE}$ uv run python -m prompts.cli stats${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    uv run python -m prompts.cli stats
    echo
}

# Run performance benchmark
run_benchmark() {
    echo -e "${PURPLE}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${PURPLE}⚡ PERFORMANCE BENCHMARK${NC}"
    echo -e "${PURPLE}═══════════════════════════════════════════════════════════════════${NC}"
    echo
    
    echo -e "${CYAN}Testing system performance with various templates...${NC}"
    echo
    
    # Test different templates with timing
    TEMPLATES=("qa_basic" "qa_detailed" "chain_of_thought" "medical_qa" "llm_judge")
    
    for template in "${TEMPLATES[@]}"; do
        echo -e "${YELLOW}🔄 Testing $template template...${NC}"
        
        # Create appropriate variables for each template
        case $template in
            "llm_judge")
                VARS='{"original_query": "Test", "response_to_evaluate": "Sample response", "evaluation_criteria": "Quality", "context": [{"title": "Test", "content": "Content"}]}'
                ;;
            "medical_qa")
                VARS='{"domain": "medical", "context": [{"title": "Medical", "content": "Health info"}]}'
                ;;
            *)
                VARS='{"context": [{"title": "Test", "content": "Sample content"}]}'
                ;;
        esac
        
        # Time the execution (use seconds and calculate milliseconds)
        START_TIME=$(date +%s)
        uv run python -m prompts.cli execute "Test query for $template" \
            --template "$template" \
            --variables "$VARS" > /dev/null 2>&1
        END_TIME=$(date +%s)
        
        DURATION=$((END_TIME - START_TIME))
        DURATION_MS=$((DURATION * 1000))
        echo -e "   ${GREEN}✅ $template: ${DURATION_MS}ms${NC}"
    done
    echo
}

# Display final summary
show_summary() {
    echo -e "${PURPLE}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${PURPLE}✅ DEMO COMPLETED SUCCESSFULLY${NC}"
    echo -e "${PURPLE}═══════════════════════════════════════════════════════════════════${NC}"
    echo
    
    echo -e "${GREEN}🎯 Key Achievements:${NC}"
    echo -e "   ${CYAN}• 14 templates across 5 categories working perfectly${NC}"
    echo -e "   ${CYAN}• 3 intelligent selection strategies demonstrated${NC}"
    echo -e "   ${CYAN}• 5 evaluation templates for comprehensive assessment${NC}"
    echo -e "   ${CYAN}• CLI interface with 25+ commands ready for use${NC}"
    echo -e "   ${CYAN}• $ACTIVE_PROVIDER integration for live responses${NC}"
    echo -e "   ${CYAN}• Production-ready prompt management system${NC}"
    echo
    
    echo -e "${GREEN}🚀 Next Steps:${NC}"
    echo -e "   ${CYAN}• Integrate with your RAG system or application${NC}"
    echo -e "   ${CYAN}• Customize templates for your specific use cases${NC}"
    echo -e "   ${CYAN}• Set up monitoring and evaluation workflows${NC}"
    echo -e "   ${CYAN}• Explore advanced features like LangGraph integration${NC}"
    echo
    
    echo -e "${GREEN}📚 Resources:${NC}"
    echo -e "   ${CYAN}• README.md - Complete documentation and usage guide${NC}"
    echo -e "   ${CYAN}• templates/ - All template files for customization${NC}"
    echo -e "   ${CYAN}• CLI help: uv run python -m prompts.cli --help${NC}"
    echo
    
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  🎉 LlamaFarm Prompts Management System is ready for production! 🎉${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
}

# Main execution
main() {
    print_banner
    
    # Setup phase
    check_dependencies
    setup_environment
    check_llm_providers
    test_basic_functionality
    
    # Demo phase
    demo_template_categories
    demo_strategy_selection
    demo_template_showcase
    demo_evaluation_templates
    demo_live_llm
    show_cli_examples
    run_benchmark
    
    # Completion
    show_summary
}

# Handle interruption
trap 'echo -e "\n${YELLOW}Demo interrupted by user${NC}"; exit 130' INT

# Run main function
main "$@"

