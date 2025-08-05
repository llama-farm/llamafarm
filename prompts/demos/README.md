# Prompt System Demos

This directory contains comprehensive demonstrations of the LlamaFarm Prompt System capabilities.

## Available Demos

### 1. Simple Q&A (`demo1_simple_qa.py`)
- Basic question answering with context
- Configuration overrides
- Strategy statistics
- Template selection basics

**Key Concepts:**
- Simple strategies for straightforward use cases
- Context document formatting
- Temperature and token configuration

### 2. Customer Support (`demo2_customer_support.py`)
- Dynamic template selection based on query type
- Selection rules in action
- Input transformations (trimming, normalization)
- Specialized handling for technical queries and complaints

**Key Concepts:**
- Conditional template selection
- Query type detection
- Personality adjustments
- System prompts for consistent tone

### 3. Code Assistant (`demo3_code_assistant.py`)
- Code analysis and review
- Debug vs optimization workflows
- Multi-language support
- Output transformations for code extraction

**Key Concepts:**
- Task-specific analysis (debug, optimize, review)
- Low temperature for accuracy
- Pattern-based checking
- Code block extraction

### 4. RAG Research (`demo4_rag_research.py`)
- Retrieval-Augmented Generation workflows
- Citation styles (inline, academic)
- Relevance filtering
- Fallback handling

**Key Concepts:**
- Working with retrieved documents
- Citation management
- Synthesis approaches
- Context-grounded responses

### 5. Advanced Reasoning (`demo5_advanced_reasoning.py`)
- Chain of thought problem solving
- Quality evaluation (LLM-as-judge)
- Strategy recommendations
- Custom strategy creation

**Key Concepts:**
- Step-by-step reasoning
- Comparative evaluation
- Performance profiling
- Strategy customization

## Running the Demos

### Prerequisites
```bash
# Ensure you're in the prompts directory
cd prompts

# Install dependencies with UV
uv sync
```

### Individual Demos
```bash
# Run a specific demo with UV
uv run python demos/demo1_simple_qa.py
uv run python demos/demo2_customer_support.py
uv run python demos/demo3_code_assistant.py
uv run python demos/demo4_rag_research.py
uv run python demos/demo5_advanced_reasoning.py
```

### Quick Start Demo
```bash
# Run a quick overview of capabilities
uv run python demos/quick_start.py
```

### All Demos (Master Demo)
```bash
# Run all demos in sequence
uv run python demos/master_demo.py

# Or use the run_all_demos script
uv run python demos/run_all_demos.py
```

## Demo Structure

Each demo follows a consistent structure:
1. **Introduction** - What the demo demonstrates
2. **Examples** - Multiple examples showing different features
3. **Configuration** - Strategy and template configuration details
4. **Key Concepts** - Important takeaways

## Learning Path

1. **Start with Demo 1** - Understand basic strategy usage
2. **Progress to Demo 2** - Learn about dynamic template selection
3. **Explore Demo 3** - See specialized domain handling
4. **Study Demo 4** - Understand RAG integration
5. **Master Demo 5** - Advanced features and customization

## Common Patterns

### Strategy Selection
```python
# Basic execution
result = manager.execute_strategy(
    strategy_name="simple_qa",
    inputs={"query": "...", "context": [...]}
)

# With context for template selection
result = manager.execute_strategy(
    strategy_name="customer_support",
    inputs={...},
    context={"query_type": "technical"}
)

# With configuration overrides
result = manager.execute_strategy(
    strategy_name="code_assistant",
    inputs={...},
    override_config={"temperature_hint": 0.2}
)
```

### Creating Custom Strategies
```python
# Create new strategy
custom_strategy = manager.create_strategy(
    name="My Strategy",
    description="Custom strategy for specific use case",
    default_template="qa_basic"
)

# Configure and save
manager.save_strategy("my_strategy", custom_strategy)
```

## Integration Examples

### With LangChain
```python
from langchain import PromptTemplate
from prompts.strategies import StrategyManager

manager = StrategyManager()
prompt_text = manager.execute_strategy("simple_qa", inputs)
langchain_prompt = PromptTemplate.from_template(prompt_text)
```

### With Native APIs
```python
# Get prompt for direct API usage
prompt = manager.execute_strategy("code_assistant", inputs)
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3
)
```

## Performance Tips

1. **Enable Caching** - Strategies have caching enabled by default
2. **Choose Right Strategy** - Use recommendation system
3. **Optimize Tokens** - Configure max_tokens appropriately
4. **Batch Processing** - Process multiple inputs together
5. **Monitor Usage** - Track execution statistics

## Troubleshooting

- **Template Not Found**: Ensure templates are properly registered
- **Strategy Load Error**: Check YAML syntax in strategy files
- **Transform Errors**: Verify input data types match expectations
- **Selection Rule Issues**: Enable debug logging to trace selection

## Next Steps

After running the demos:
1. Create your own templates in `templates/`
2. Design custom strategies for your use cases
3. Integrate with your application
4. Monitor and optimize performance
5. Contribute improvements back to the project!