# LlamaFarm Integration Guide: Tying Models, RAG, and Prompts Together

This guide demonstrates how to integrate LlamaFarm's three core components (Models, RAG, Prompts) into powerful end-to-end AI workflows.

## 🎯 Quick Integration Examples

### Example 1: Simple Document Q&A System

This example shows the most basic integration - query documents and get answers.

```bash
# Step 1: Set up your configs (use Llama Brain's as examples)
cp designer/llama-brain/configs/llama_brain_models.yaml my_configs/models.yaml
cp designer/llama-brain/configs/llama_brain_rag.yaml my_configs/rag.yaml  
cp designer/llama-brain/configs/llama_brain_prompts.yaml my_configs/prompts.yaml

# Step 2: Ingest your documents
cd rag
uv run python cli.py --config ../my_configs/rag.yaml init
uv run python cli.py --config ../my_configs/rag.yaml ingest /path/to/your/documents/

# Step 3: Query the system
cd ../models
uv run python cli.py --config ../my_configs/models.yaml query \
  "Based on my documents, how do I configure authentication?" \
  --provider llama_brain_chat
```

### Example 2: Complete RAG Pipeline

This shows the full pipeline: search → prompt template → model response.

```bash
#!/bin/bash
# complete_rag_query.sh

QUERY="$1"
CONFIGS_DIR="my_configs"

echo "🔍 Searching documents for: $QUERY"

# Step 1: Search documents with RAG
cd rag
RAG_RESULTS=$(uv run python cli.py --config "../$CONFIGS_DIR/rag.yaml" \
  search "$QUERY" --top-k 5 --format json)

echo "📄 Found relevant documents"

# Step 2: Format context and apply prompt template
cd ../prompts
CONTEXT=$(echo "$RAG_RESULTS" | jq -r '[.[] | {title: .metadata.filename, content: .content}]')
VARIABLES="{\"query\": \"$QUERY\", \"context\": $CONTEXT}"

PROMPT_RESULT=$(uv run python -m prompts.cli --config "../$CONFIGS_DIR/prompts.yaml" \
  execute "$QUERY" --template config_assistant --variables "$VARIABLES")

echo "✨ Applied prompt template"

# Step 3: Generate final response
cd ../models
FINAL_RESPONSE=$(uv run python cli.py --config "../$CONFIGS_DIR/models.yaml" \
  query "$PROMPT_RESULT" --provider llama_brain_chat)

echo "🤖 Generated response:"
echo "$FINAL_RESPONSE"
```

Usage:
```bash
chmod +x complete_rag_query.sh
./complete_rag_query.sh "How do I set up vector search?"
```

## 🏗️ Architecture Patterns

### Pattern 1: Sequential Pipeline

Most straightforward - each component processes the output of the previous one.

```
User Query → RAG Search → Prompt Template → Model Generation → Response
```

**Use Cases:**
- Document Q&A
- Knowledge base queries  
- Technical support

**Configuration Strategy:**
```yaml
# rag.yaml - Optimize for accurate retrieval
rag:
  retrieval_strategies:
    default:
      type: "RerankedStrategy"  # Higher accuracy
  defaults:
    top_k: 3  # Focused results

# prompts.yaml - Clear, structured templates
templates:
  qa_template:
    template: |
      Context: {{ context | format_documents }}
      Question: {{ query }}
      Provide a clear, accurate answer:

# models.yaml - Balanced model for general use
providers:
  main_model:
    model: "llama3.2:3b"
    temperature: 0.7  # Balanced creativity/accuracy
```

### Pattern 2: Parallel Processing with Fusion

Run multiple retrieval strategies in parallel, then combine results.

```
         ┌─ RAG Strategy 1 ─┐
User Query ┤                 ├─ Fusion ─ Prompt ─ Model ─ Response
         └─ RAG Strategy 2 ─┘
```

**Use Cases:**
- High-stakes decisions
- Research and analysis
- Multi-domain queries

**Implementation:**
```python
# parallel_fusion.py
import asyncio
import subprocess
import json

async def multi_strategy_search(query, strategies):
    """Run multiple RAG strategies in parallel"""
    tasks = []
    
    for strategy in strategies:
        task = asyncio.create_task(
            run_rag_search(query, strategy)
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return fuse_results(results)

async def run_rag_search(query, strategy):
    """Run RAG search with specific strategy"""
    result = await asyncio.subprocess.create_subprocess_exec(
        "uv", "run", "python", "cli.py",
        "--config", "rag.yaml",
        "search", query,
        "--strategy", strategy,
        "--format", "json",
        stdout=asyncio.subprocess.PIPE
    )
    stdout, _ = await result.communicate()
    return json.loads(stdout)

def fuse_results(results_list):
    """Combine results using reciprocal rank fusion"""
    # Implementation details...
    pass
```

### Pattern 3: Agentic Workflow

Let different agents handle different types of queries.

```
User Query → Intent Detection → Route to Specialist Agent → Response
                                    ├─ Technical Agent
                                    ├─ Business Agent  
                                    └─ General Agent
```

**Use Cases:**
- Multi-domain systems
- Complex decision making
- Specialized expertise

**Configuration:**
```yaml
# prompts.yaml - Intent detection and routing
templates:
  intent_detector:
    template: |
      Analyze this query and determine the domain:
      Query: {{ query }}
      
      Domains: technical, business, legal, medical, general
      Domain:

  technical_agent:
    template: |
      You are a technical expert. Context: {{ context }}
      Technical Question: {{ query }}
      Provide detailed technical guidance:

  business_agent:
    template: |
      You are a business analyst. Context: {{ context }}
      Business Question: {{ query }}
      Provide strategic business advice:
```

### Pattern 4: Iterative Refinement

Improve responses through multiple iterations.

```
Query → Initial Response → Quality Check → Refinement → Final Response
  ↑                                           ↓
  └─────────── Feedback Loop ──────────────────┘
```

**Use Cases:**
- High-quality content generation
- Complex reasoning tasks
- Fact-checking requirements

## 🔧 Configuration Strategies

### Development Configuration

**Focus**: Fast iteration, easy debugging

```yaml
# dev_models.yaml
providers:
  dev_model:
    model: "llama3.2:3b"  # Fastest local model
    temperature: 0.3      # More deterministic for testing
    timeout: 30

# dev_rag.yaml  
rag:
  defaults:
    top_k: 3           # Fewer results for speed
  retrieval_strategies:
    default:
      type: "BasicSimilarityStrategy"  # Fastest retrieval

# dev_prompts.yaml
templates:
  debug_template:
    template: |
      [DEBUG MODE]
      Context: {{ context | format_documents | truncate(500) }}
      Query: {{ query }}
      Quick answer:
```

### Production Configuration

**Focus**: Accuracy, reliability, monitoring

```yaml
# prod_models.yaml
providers:
  primary:
    model: "llama3.1:8b"     # More capable model
    temperature: 0.7
    timeout: 120
    max_retries: 3
    
  fallback:
    model: "llama3.2:3b"     # Faster fallback
    temperature: 0.7

fallback_chain: ["primary", "fallback"]

monitoring:
  track_usage: true
  log_requests: true
  alert_on_errors: true

# prod_rag.yaml
rag:
  defaults:
    top_k: 5
  retrieval_strategies:
    default:
      type: "RerankedStrategy"      # Higher accuracy
      config:
        rerank_factors:
          recency_weight: 0.1
          authority_weight: 0.15

monitoring:
  log_level: "INFO"
  track_latency: true
  track_popular_queries: true

# prod_prompts.yaml
global_prompts:
  - global_id: "quality_assurance"
    prefix_prompt: "Provide accurate, well-sourced information."
    priority: 10
    
templates:
  production_qa:
    template: |
      Documentation: {{ context | format_documents }}
      
      User Question: {{ query }}
      
      Comprehensive Answer (cite sources when relevant):
```

### Specialized Domain Configurations

**Medical/Healthcare Domain:**

```yaml
# medical_rag.yaml
rag:
  retrieval_strategies:
    medical_filtered:
      type: "MetadataFilteredStrategy"
      config:
        default_filters:
          domain: ["medical", "clinical", "research"]
          verified: [true]

# medical_prompts.yaml
global_prompts:
  - global_id: "medical_disclaimer"
    suffix_prompt: |
      
      MEDICAL DISCLAIMER: This information is for educational purposes only. 
      Always consult qualified healthcare professionals for medical decisions.
    applies_to: ["medical_*"]

templates:
  medical_qa:
    template: |
      Medical Literature: {{ context | format_documents }}
      Medical Question: {{ query }}
      
      Evidence-based response with appropriate cautions:
```

**Legal Domain:**

```yaml
# legal_prompts.yaml
templates:
  legal_analysis:
    template: |
      Legal Sources: {{ context | format_documents }}
      
      Legal Question: {{ query }}
      
      Legal Analysis (include jurisdiction and limitations):
      
      LEGAL DISCLAIMER: This is not legal advice. Consult qualified attorneys.
```

## 🚀 Advanced Integration Patterns

### 1. Multi-Modal Integration

Combine text, images, and code processing:

```python
# multi_modal_pipeline.py
class MultiModalPipeline:
    def __init__(self, configs):
        self.text_rag = TextRAG(configs['text_rag'])
        self.image_rag = ImageRAG(configs['image_rag'])
        self.code_rag = CodeRAG(configs['code_rag'])
        self.models = ModelClient(configs['models'])
        
    async def process_query(self, query, media_files=None):
        # Detect query type
        query_type = await self.detect_query_type(query)
        
        # Process based on type
        if query_type == "code":
            context = await self.code_rag.search(query)
            template = "code_assistant"
        elif query_type == "visual":
            context = await self.image_rag.search(query, media_files)
            template = "visual_assistant"
        else:
            context = await self.text_rag.search(query)
            template = "general_assistant"
            
        # Generate response
        return await self.models.generate(
            query=query,
            context=context,
            template=template
        )
```

### 2. Streaming Integration

Real-time responses with progressive updates:

```python
# streaming_integration.py
async def streaming_rag_response(query):
    """Stream response as context is retrieved and processed"""
    
    # Start search immediately
    search_task = asyncio.create_task(search_documents(query))
    
    # Stream initial response
    yield "🔍 Searching relevant documents...\n"
    
    # Get search results
    documents = await search_task
    yield f"📄 Found {len(documents)} relevant documents\n\n"
    
    # Stream formatted context
    yield "📖 **Relevant Information:**\n"
    for doc in documents[:3]:
        yield f"- {doc['title']}: {doc['content'][:100]}...\n"
    
    yield "\n🤖 **AI Response:**\n"
    
    # Stream AI response
    async for chunk in stream_model_response(query, documents):
        yield chunk
```

### 3. Caching and Optimization

Optimize for performance with intelligent caching:

```python
# optimized_integration.py
class OptimizedRAG:
    def __init__(self):
        self.search_cache = LRUCache(maxsize=1000)
        self.embedding_cache = LRUCache(maxsize=5000)
        self.response_cache = LRUCache(maxsize=500)
    
    async def smart_search(self, query):
        # Check cache first
        cache_key = hash(query)
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]
        
        # Compute embeddings with caching
        embedding = await self.get_cached_embedding(query)
        
        # Search with cached results
        results = await self.vector_search(embedding)
        
        # Cache results
        self.search_cache[cache_key] = results
        return results
    
    async def get_cached_embedding(self, text):
        text_hash = hash(text)
        if text_hash not in self.embedding_cache:
            self.embedding_cache[text_hash] = await self.compute_embedding(text)
        return self.embedding_cache[text_hash]
```

## 📊 Monitoring and Analytics

### Comprehensive Logging

```python
# monitoring_integration.py
import logging
import time
from functools import wraps

class RAGMetrics:
    def __init__(self):
        self.query_count = 0
        self.avg_latency = 0
        self.error_rate = 0
        
    def log_query(self, query, latency, success):
        self.query_count += 1
        self.avg_latency = (self.avg_latency + latency) / 2
        if not success:
            self.error_rate += 1
        
        logging.info(f"Query: {query[:50]} | Latency: {latency:.2f}s | Success: {success}")

def monitor_performance(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            latency = time.time() - start_time
            metrics.log_query(kwargs.get('query', ''), latency, True)
            return result
        except Exception as e:
            latency = time.time() - start_time
            metrics.log_query(kwargs.get('query', ''), latency, False)
            raise
    return wrapper

@monitor_performance
async def complete_rag_pipeline(query):
    # Your integration logic here
    pass
```

### Health Checks

```bash
#!/bin/bash
# health_check.sh

echo "🏥 LlamaFarm Health Check"

# Check Models
echo "Checking Models..."
cd models
MODELS_STATUS=$(uv run python cli.py --config ../my_configs/models.yaml health-check)
echo "Models: $MODELS_STATUS"

# Check RAG
echo "Checking RAG..."
cd ../rag
RAG_STATUS=$(uv run python cli.py --config ../my_configs/rag.yaml info)
echo "RAG: $RAG_STATUS"

# Check Prompts
echo "Checking Prompts..."
cd ../prompts
PROMPTS_STATUS=$(uv run python -m prompts.cli --config ../my_configs/prompts.yaml stats)
echo "Prompts: $PROMPTS_STATUS"

# Integration test
echo "Running integration test..."
cd ..
./complete_rag_query.sh "test query" > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Integration: HEALTHY"
else
    echo "❌ Integration: FAILED"
fi
```

## 🎯 Use Case Implementations

### 1. Customer Support Bot

Complete implementation for customer support:

```yaml
# customer_support_models.yaml
providers:
  support_model:
    model: "llama3.2:3b"
    temperature: 0.5      # Balanced for support
    system_prompt: "You are a helpful customer support agent."

# customer_support_rag.yaml
rag:
  retrieval_strategies:
    support_focused:
      type: "MetadataFilteredStrategy"
      config:
        default_filters:
          category: ["support", "faq", "troubleshooting", "billing"]
  defaults:
    top_k: 3

# customer_support_prompts.yaml
templates:
  support_agent:
    template: |
      Support Documentation: {{ context | format_documents }}
      
      Customer Issue: {{ query }}
      
      As a support agent, provide a helpful, empathetic response:
      1. Acknowledge the customer's concern
      2. Provide clear steps to resolve the issue  
      3. Offer additional help if needed
      
      Response:
```

### 2. Technical Documentation Assistant

For internal engineering teams:

```yaml
# tech_docs_rag.yaml
rag:
  parsers:
    code:
      type: "TextParser"
      config:
        chunk_size: 1500        # Larger chunks for technical content
        preserve_code_blocks: true
  
  retrieval_strategies:
    code_aware:
      type: "RerankedStrategy"
      config:
        rerank_factors:
          code_similarity: 0.3   # Boost code relevance
          recency_weight: 0.2    # Prefer recent docs

# tech_docs_prompts.yaml
templates:
  technical_guide:
    template: |
      Technical Documentation: {{ context | format_documents }}
      
      Developer Question: {{ query }}
      
      Technical Response:
      - Include code examples when relevant
      - Reference specific API versions
      - Mention any breaking changes or deprecations
      
      Answer:
```

### 3. Research Assistant

For academic or business research:

```yaml
# research_rag.yaml
rag:
  retrieval_strategies:
    research_quality:
      type: "RerankedStrategy"
      config:
        rerank_factors:
          authority_weight: 0.4    # Boost authoritative sources
          citation_count: 0.2      # Boost well-cited content
          recency_weight: 0.1

# research_prompts.yaml
templates:
  research_analyst:
    template: |
      Research Sources: {{ context | format_documents }}
      
      Research Question: {{ query }}
      
      Comprehensive Analysis:
      1. **Key Findings**: What does the research show?
      2. **Methodologies**: What approaches were used?
      3. **Limitations**: What are the constraints or gaps?
      4. **Implications**: What does this mean for practice?
      5. **Further Research**: What questions remain?
      
      Analysis:
```

## 🔄 CI/CD Integration

### Automated Testing Pipeline

```yaml
# .github/workflows/llamafarm-integration.yml
name: LlamaFarm Integration Tests

on: [push, pull_request]

jobs:
  integration-test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Python and UV
      run: |
        curl -LsSf https://install.python-lang.org/3.11 | sh
        curl -LsSf https://astral.sh/uv/install.sh | sh
        
    - name: Start Ollama
      run: |
        curl https://ollama.ai/install.sh | sh
        ollama serve &
        sleep 10
        ollama pull llama3.2:3b
        ollama pull nomic-embed-text
        
    - name: Install Dependencies
      run: |
        cd designer/llama-brain
        uv sync
        
    - name: Test Individual Components
      run: |
        # Test Models
        cd models
        uv run python cli.py --config ../designer/llama-brain/configs/llama_brain_models.yaml list
        
        # Test RAG
        cd ../rag
        uv run python cli.py --config ../designer/llama-brain/configs/llama_brain_rag.yaml info
        
        # Test Prompts
        cd ../prompts
        uv run python -m prompts.cli --config ../designer/llama-brain/configs/llama_brain_prompts.yaml stats
        
    - name: Test Integration
      run: |
        cd designer/llama-brain
        uv run python -c "
        from llama_brain.integrations.llamafarm_client import LlamaFarmClient
        client = LlamaFarmClient()
        response = client.query('test integration')
        assert 'error' not in response.lower()
        print('Integration test passed!')
        "
```

### Configuration Validation

```python
# validate_integration.py
import yaml
import json
import subprocess
import sys

def validate_config_syntax(config_path):
    """Validate YAML syntax"""
    try:
        with open(config_path) as f:
            yaml.safe_load(f)
        return True, None
    except Exception as e:
        return False, str(e)

def validate_config_against_cli(config_path, cli_path, command):
    """Test config with actual CLI"""
    try:
        result = subprocess.run([
            "uv", "run", "python", cli_path,
            "--config", config_path,
            command
        ], capture_output=True, text=True, timeout=30)
        
        return result.returncode == 0, result.stderr
    except Exception as e:
        return False, str(e)

def main():
    configs = [
        ("models", "designer/llama-brain/configs/llama_brain_models.yaml", "models/cli.py", "list"),
        ("rag", "designer/llama-brain/configs/llama_brain_rag.yaml", "rag/cli.py", "info"),
        ("prompts", "designer/llama-brain/configs/llama_brain_prompts.yaml", "prompts", "stats")
    ]
    
    all_valid = True
    
    for component, config_path, cli_path, command in configs:
        print(f"Validating {component} config...")
        
        # Syntax check
        valid, error = validate_config_syntax(config_path)
        if not valid:
            print(f"❌ {component} syntax error: {error}")
            all_valid = False
            continue
            
        # CLI check
        if component == "prompts":
            cli_command = ["uv", "run", "python", "-m", "prompts.cli", "--config", config_path, command]
        else:
            cli_command = ["uv", "run", "python", cli_path, "--config", config_path, command]
            
        try:
            result = subprocess.run(cli_command, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(f"✅ {component} config valid")
            else:
                print(f"❌ {component} CLI error: {result.stderr}")
                all_valid = False
        except Exception as e:
            print(f"❌ {component} validation failed: {e}")
            all_valid = False
    
    if all_valid:
        print("🎉 All configurations valid!")
        sys.exit(0)
    else:
        print("💥 Some configurations invalid!")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## 📚 Best Practices Summary

### Configuration Management
1. **Version Control**: Always version your configurations
2. **Environment Separation**: Separate dev/staging/prod configs
3. **Validation**: Validate configs before deployment
4. **Documentation**: Document all custom settings

### Performance Optimization
1. **Caching**: Cache embeddings, search results, and responses
2. **Batching**: Process multiple queries together when possible
3. **Monitoring**: Track latency, error rates, and resource usage
4. **Scaling**: Use appropriate chunk sizes and retrieval limits

### Error Handling
1. **Graceful Degradation**: Implement fallback strategies
2. **Retry Logic**: Retry failed operations with backoff
3. **Circuit Breakers**: Prevent cascade failures
4. **Monitoring**: Alert on error rate thresholds

### Security
1. **API Keys**: Never commit API keys to version control
2. **Input Validation**: Validate all user inputs
3. **Access Control**: Implement proper authentication
4. **Audit Logging**: Log all queries and responses

---

This integration guide provides the foundation for building sophisticated AI systems with LlamaFarm. Start with the simple examples and gradually incorporate more advanced patterns as your needs grow.

For specific component documentation, refer to:
- [Models README](../models/README.md)
- [RAG README](../rag/README.md)
- [Prompts README](../prompts/README.md)
- [Llama Brain README](./README.md)