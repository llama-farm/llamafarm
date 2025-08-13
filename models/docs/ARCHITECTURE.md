# LlamaFarm Models System - Developer Structure Guide

> **For Project Developers**: This document explains the internal architecture, organization, and development patterns of the models management system. For user documentation, see [README.md](README.md).

## 🏗️ System Architecture Overview

The LlamaFarm Models System follows a **unified CLI-based architecture** with provider abstraction:

```
┌─────────────────────────────────────────────────────┐
│                   CLI Interface                     │
│                    (cli.py)                         │
├─────────────────────────────────────────────────────┤
│              Provider Abstraction Layer            │
│   CloudProvider │ LocalProvider │ HFProvider       │  
├─────────────────────────────────────────────────────┤
│                 API Integration                     │
│  OpenAI │ Anthropic │ Ollama │ Groq │ Together     │
├─────────────────────────────────────────────────────┤
│             Configuration Management                │
│        JSON configs │ Environment │ Fallbacks      │
└─────────────────────────────────────────────────────┘
```

## 📁 Directory Structure Deep Dive

```
models/
├── cli.py                           # 2,169 lines - Main CLI application
├── README.md                        # User-facing documentation
├── STRUCTURE.md                     # This file - developer guide
├── pyproject.toml                   # Python package configuration
├── setup_and_demo.sh               # Automated setup and demo script
├── uv.lock                          # Dependency lock file (800k+ lines)
├── .env                             # Environment variables (not in git)
├── .coverage                        # Coverage report data
│
├── config/                          # Configuration management
│   ├── default.json                 # Default provider configuration
│   ├── development.json             # Development-specific settings
│   ├── real_models_example.json     # Production-ready real model configs
│   ├── use_case_examples.json       # Domain-specific optimized configs
│   ├── hf_models.json              # Hugging Face model definitions
│   ├── ollama_local.json           # Local Ollama model configs
│   ├── local_engines.json          # vLLM/TGI engine configurations
│   └── test_config.json            # Test environment configuration
│
├── docs/                           # Technical documentation
│   ├── ALL_WORKING_CONFIRMED.md    # Feature verification documentation
│   ├── WORKING_API_CALLS.md        # Real API integration examples
│   └── INTEGRATION.md              # Integration patterns and examples
│
├── examples/                       # Generated examples and demos
│   ├── demo_basic_config.json      # Basic configuration example
│   ├── demo_engines_config.json    # Local inference engines example
│   ├── demo_hf_config.json         # Hugging Face configuration
│   ├── demo_multi_config.json      # Multi-provider with fallbacks
│   ├── demo_ollama_config.json     # Ollama local model configuration
│   ├── demo_production_config.json # Production deployment example
│   │
│   ├── config_examples/            # Configuration pattern examples
│   │   └── README.md
│   ├── methods/                    # Method implementation examples
│   │   └── README.md
│   └── registry/                   # Model registry examples
│       └── README.md
│
├── tests/                          # Test suite
│   ├── test_models.py              # Unit tests (34 tests)
│   └── test_e2e.py                 # Integration tests (12 tests)
│
└── __pycache__/                    # Python bytecode cache
```

## 🔧 Core Architecture Components

### **1. CLI Application (cli.py) - 2,169 Lines**

The monolithic CLI application contains all functionality in a single file for simplicity and deployment efficiency.

#### **Command Categories:**
- **Core Interaction**: `query`, `chat`, `send`, `batch` (4 commands)
- **Testing & Health**: `test`, `compare`, `health-check`, `validate-config` (4 commands)  
- **Management**: `list`, `list-local`, `pull` (3 commands)
- **Configuration**: `generate-config`, `generate-ollama-config`, `generate-hf-config`, `generate-engines-config` (4 commands)
- **Hugging Face**: `hf-login`, `list-hf`, `download-hf`, `test-hf` (4 commands)
- **Local Engines**: `list-vllm`, `test-vllm`, `list-tgi`, `test-tgi` (4 commands)
- **Ollama Specific**: `test-local` (1 command)

#### **Architecture Patterns:**

```python
# Provider abstraction pattern
class ProviderInterface:
    def call_api(self, messages, **kwargs) -> dict
    def validate_config(self) -> bool
    def get_cost_info(self) -> dict

# Configuration validation pattern  
def validate_provider_config(provider_config: dict) -> List[str]:
    """Validate provider configuration and return errors"""

# Error handling pattern
def safe_api_call(func, *args, **kwargs):
    """Wrapper for safe API calls with error handling"""
```

### **2. Configuration System**

#### **Configuration Schema:**
```json
{
  "name": "Configuration Name",
  "version": "1.0.0", 
  "description": "Optional description",
  "default_provider": "provider_id",
  "fallback_chain": ["primary", "secondary", "backup"],
  "global_settings": {
    "timeout": 120,
    "max_retries": 3,
    "default_temperature": 0.7
  },
  "providers": {
    "provider_id": {
      "type": "cloud|local|hf",
      "provider": "openai|anthropic|ollama|huggingface",
      "model": "model_name",
      "api_key": "${ENV_VAR}",
      "base_url": "https://api.example.com",
      "temperature": 0.7,
      "max_tokens": 2048,
      "timeout": 60,
      "cost_per_token": 0.000001,
      "custom_headers": {},
      "model_params": {}
    }
  }
}
```

#### **Configuration Hierarchy:**
1. **Command-line arguments** (highest priority)
2. **Custom config file** (`--config path`)
3. **Default config file** (`config/default.json`)
4. **Environment variables** (for API keys)
5. **Built-in defaults** (lowest priority)

### **3. Provider Implementation Patterns**

#### **Cloud Providers (OpenAI, Anthropic, etc.)**
```python
async def call_openai_api(provider_config, messages, **kwargs):
    """OpenAI API implementation pattern"""
    client = AsyncOpenAI(
        api_key=provider_config['api_key'],
        base_url=provider_config.get('base_url')
    )
    
    # Parameter mapping and validation
    params = map_parameters(provider_config, kwargs)
    
    # API call with error handling
    response = await client.chat.completions.create(**params)
    
    # Response normalization
    return normalize_response(response)
```

#### **Local Providers (Ollama, vLLM)**
```python
async def call_ollama_api(provider_config, messages, **kwargs):
    """Local Ollama API implementation pattern"""
    base_url = f"http://{provider_config['host']}:11434"
    
    # Local API call pattern
    async with aiohttp.ClientSession() as session:
        response = await session.post(
            f"{base_url}/api/chat",
            json=format_ollama_request(messages, **kwargs)
        )
        
    return await response.json()
```

#### **Hugging Face Models**
```python
def load_hf_model(model_name, **kwargs):
    """Hugging Face model loading pattern"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if kwargs.get('gpu') else None
    )
    
    return model, tokenizer
```

## 🔄 Command Implementation Patterns

### **1. Standard Command Pattern**
```python
@click.command()
@click.argument('query')
@click.option('--provider', help='Provider to use')
@click.option('--temperature', type=float, help='Generation temperature')
@click.pass_context
def command_name(ctx, query, provider, temperature):
    """Command docstring"""
    # 1. Load and validate configuration
    config = load_configuration(ctx.obj.get('config'))
    
    # 2. Resolve provider (explicit, default, or fallback)
    provider_id = resolve_provider(provider, config)
    
    # 3. Execute operation with error handling
    try:
        result = execute_operation(provider_id, query, temperature)
        display_result(result)
    except Exception as e:
        handle_error(e)
```

### **2. Configuration Generation Pattern**
```python
def generate_config_template(config_type: str) -> dict:
    """Generate configuration template based on type"""
    templates = {
        'basic': generate_basic_config,
        'multi': generate_multi_provider_config,
        'production': generate_production_config
    }
    
    return templates[config_type]()
```

### **3. Provider Discovery Pattern**
```python
def discover_available_providers():
    """Discover what providers are available"""
    available = {}
    
    # Check cloud providers (API keys)
    if os.getenv('OPENAI_API_KEY'):
        available['openai'] = check_openai_connectivity()
    
    # Check local providers (services)
    if check_ollama_running():
        available['ollama'] = get_ollama_models()
        
    # Check HF models (local cache)
    available['huggingface'] = scan_hf_cache()
    
    return available
```

## 🧪 Testing Architecture

### **Test Organization**
```
tests/
├── test_models.py              # Unit tests (34 tests)
│   ├── TestConfigValidation    # Configuration validation tests
│   ├── TestProviderDiscovery   # Provider discovery tests  
│   ├── TestParameterMapping    # Parameter conversion tests
│   ├── TestErrorHandling       # Error handling tests
│   └── TestUtilityFunctions    # Helper function tests
│
└── test_e2e.py                # Integration tests (12 tests)
    ├── TestOpenAIIntegration   # Live OpenAI API tests
    ├── TestOllamaIntegration   # Live Ollama tests  
    ├── TestAnthropicIntegration # Live Anthropic tests
    └── TestFallbackChains      # Fallback mechanism tests
```

### **Test Patterns**

#### **Unit Test Pattern**
```python
class TestProviderValidation:
    def test_valid_openai_config(self):
        """Test valid OpenAI configuration"""
        config = {
            "type": "cloud",
            "provider": "openai", 
            "model": "gpt-4o-mini",
            "api_key": "sk-test123"
        }
        errors = validate_provider_config(config)
        assert errors == []
    
    def test_missing_api_key(self):
        """Test configuration missing API key"""
        config = {"type": "cloud", "provider": "openai"}
        errors = validate_provider_config(config)
        assert "api_key is required" in errors
```

#### **Integration Test Pattern**
```python
@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv('OPENAI_API_KEY'), 
                   reason="Requires OpenAI API key")
async def test_openai_api_call():
    """Test actual OpenAI API call"""
    config = load_test_config()
    response = await call_openai_api(
        config['providers']['openai_test'],
        [{"role": "user", "content": "Hello"}]
    )
    
    assert response['content']
    assert response['model'] 
    assert response['usage']['total_tokens'] > 0
```

## 🚀 Extension & Development Patterns

### **Adding New Cloud Providers**

1. **Add Provider Configuration Schema**
```python
PROVIDER_SCHEMAS = {
    'new_provider': {
        'required': ['api_key', 'model'],
        'optional': ['base_url', 'temperature', 'max_tokens'],
        'api_format': 'openai_compatible'  # or 'custom'
    }
}
```

2. **Implement API Client**
```python
async def call_new_provider_api(provider_config, messages, **kwargs):
    """New provider API implementation"""
    # Follow existing patterns in cli.py
    # Include error handling, response normalization
    pass
```

3. **Add to Provider Router**
```python
PROVIDER_HANDLERS = {
    'openai': call_openai_api,
    'anthropic': call_anthropic_api,
    'new_provider': call_new_provider_api,  # Add here
}
```

4. **Add Configuration Templates**
```python
def generate_new_provider_config():
    return {
        "new_provider_example": {
            "type": "cloud",
            "provider": "new_provider",
            "model": "default-model",
            "api_key": "${NEW_PROVIDER_API_KEY}"
        }
    }
```

### **Adding New Local Inference Engines**

1. **Define Engine Interface**
```python
class LocalEngine:
    def __init__(self, config):
        self.config = config
    
    def load_model(self, model_name):
        """Load model into memory"""
        pass
    
    def generate(self, prompt, **kwargs):
        """Generate response"""
        pass
    
    def unload_model(self):
        """Free memory"""
        pass
```

2. **Implement Engine-Specific Logic**
```python
class NewEngineProvider(LocalEngine):
    def load_model(self, model_name):
        # Engine-specific model loading
        self.model = load_with_new_engine(model_name)
    
    def generate(self, prompt, **kwargs):
        return self.model.generate(prompt, **kwargs)
```

### **Adding New Commands**

1. **Follow CLI Pattern**
```python
@cli.command()
@click.argument('required_arg')
@click.option('--optional-param', help='Description')
@click.pass_context
def new_command(ctx, required_arg, optional_param):
    """Command description for help text"""
    try:
        config = load_configuration(ctx.obj.get('config'))
        result = execute_new_functionality(required_arg, optional_param)
        click.echo(format_output(result))
    except Exception as e:
        handle_error(e, ctx)
```

2. **Add to Command Groups**
```python
# Core commands: query, chat, send, batch
# Management commands: list, test, health-check
# Config commands: generate-config, validate-config
# Provider-specific: ollama-*, hf-*, vllm-*, tgi-*
```

## 🔍 Debugging & Development Tools

### **Built-in Debugging**
```bash
# Enable verbose logging
export PYTHONPATH=. 
python -m pdb cli.py query "test" --provider openai_gpt4o_mini

# Configuration debugging
uv run python cli.py validate-config --verbose

# Provider connectivity testing
uv run python cli.py health-check --detailed
```

### **Development Environment Setup**
```bash
# Install in development mode
uv sync --dev

# Run tests with coverage
uv run python -m pytest tests/ --cov=. --cov-report=html

# Type checking (if using mypy)
uv run mypy cli.py

# Code formatting
uv run black cli.py
uv run isort cli.py
```

### **Configuration Validation**
```python
def validate_full_configuration(config_path: str) -> dict:
    """Comprehensive configuration validation"""
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'provider_status': {}
    }
    
    # Validate JSON structure
    # Check required fields
    # Validate provider configurations
    # Test API connectivity
    
    return results
```

## 📊 Performance & Monitoring

### **Performance Patterns**
```python
# Async operations for I/O bound tasks
async def batch_process_queries(queries, provider_config):
    """Process multiple queries concurrently"""
    semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
    
    async def process_single(query):
        async with semaphore:
            return await call_provider_api(provider_config, query)
    
    return await asyncio.gather(*[process_single(q) for q in queries])

# Caching for expensive operations
from functools import lru_cache

@lru_cache(maxsize=128)
def load_model_metadata(model_name: str) -> dict:
    """Cache model metadata to avoid repeated API calls"""
    return fetch_model_info(model_name)
```

### **Monitoring Integration Points**
- **API call timing and success rates**
- **Token usage and cost tracking**
- **Error rates by provider**
- **Model performance metrics**
- **Configuration validation results**

## 🎯 Best Practices for Contributors

### **Code Organization**
1. **Single File Architecture**: Keep CLI functionality in `cli.py` for deployment simplicity
2. **Configuration Driven**: Use JSON configs instead of hardcoding provider details
3. **Error Handling**: Comprehensive error handling with user-friendly messages
4. **Async Operations**: Use async/await for I/O bound operations
5. **Type Hints**: Add type hints for better code documentation

### **Configuration Management**
1. **Environment Variables**: Use env vars for secrets, configs for structure
2. **Fallback Chains**: Always provide fallback options for reliability
3. **Validation**: Validate configurations before using them
4. **Examples**: Provide working examples for all configuration types

### **Testing Strategy**
1. **Unit Tests**: Test individual functions and validation logic
2. **Integration Tests**: Test actual API calls (with API keys)
3. **Configuration Tests**: Test all example configurations
4. **Error Path Testing**: Test error handling and edge cases

### **Documentation**
1. **User Documentation**: Keep README.md focused on usage
2. **Developer Documentation**: Use this STRUCTURE.md for architecture
3. **API Examples**: Provide real, working examples
4. **Configuration Examples**: Include complete, tested configurations

## 🔄 Development Workflow

### **Adding New Features**
1. **Design**: Update this STRUCTURE.md with proposed changes
2. **Implementation**: Add code following existing patterns
3. **Testing**: Add unit and integration tests
4. **Configuration**: Update relevant config files
5. **Documentation**: Update README.md and examples
6. **Validation**: Run full test suite and demo script

### **Debugging Issues**
1. **Configuration**: Check config file syntax and completeness
2. **Environment**: Verify API keys and environment variables
3. **Connectivity**: Test network connectivity to providers
4. **Logs**: Enable verbose logging for detailed error information
5. **Tests**: Run specific tests to isolate issues

### **Performance Optimization**
1. **Profiling**: Use Python profilers to identify bottlenecks
2. **Async**: Convert I/O operations to async for better concurrency
3. **Caching**: Cache expensive operations like model loading
4. **Batching**: Batch similar operations for efficiency

## 🤝 Contributing Guidelines

### **Code Standards**
- Follow existing code patterns and naming conventions
- Add comprehensive error handling
- Include type hints where possible
- Write descriptive docstrings
- Test all new functionality

### **Configuration Standards**
- Validate all configuration schemas
- Provide working examples
- Use environment variables for secrets
- Include fallback options
- Document all configuration options

### **Testing Standards**
- Write unit tests for all new functions
- Add integration tests for new providers
- Test error conditions and edge cases
- Maintain >90% test coverage
- Ensure all tests pass before submitting

---

This system is designed for **simplicity**, **reliability**, and **extensibility**. The single-file architecture makes deployment easy while the configuration-driven approach makes it flexible for different use cases.

For user documentation and setup instructions, see [README.md](README.md).