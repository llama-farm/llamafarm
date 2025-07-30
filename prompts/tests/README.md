# Prompt System Tests

## 🧪 Overview

The tests directory contains comprehensive test suites for the LlamaFarm prompts system. Following the project's testing philosophy with >90% coverage for core components, this directory ensures reliability, performance, and correctness of all prompt system components.

## 📁 Directory Structure

```
tests/
├── __init__.py                 # Test package initialization (future)
├── conftest.py                # Pytest configuration and fixtures (future)
├── unit/                      # Unit tests for individual components
│   ├── test_templates.py      # Template processing tests
│   ├── test_strategies.py     # Strategy selection tests
│   ├── test_registry.py       # Registry system tests
│   └── test_utils.py          # Utility function tests
├── integration/               # Integration tests
│   ├── test_rag_integration.py # RAG pipeline integration
│   ├── test_end_to_end.py     # Complete workflow tests
│   └── test_config_loading.py # Configuration system tests
├── performance/               # Performance and load tests
│   ├── test_template_speed.py # Template rendering performance
│   ├── test_strategy_speed.py # Strategy selection performance
│   └── test_cache_performance.py # Caching system performance
├── fixtures/                  # Test data and fixtures
│   ├── templates/             # Sample templates for testing
│   ├── configs/               # Test configurations
│   └── sample_data/           # Test documents and data
└── utils/                     # Testing utilities and helpers
    ├── test_helpers.py        # Common test utilities
    ├── mock_components.py     # Mock objects for testing
    └── assertion_helpers.py   # Custom assertions
```

## 🎯 Testing Strategy

### **Testing Principles**
1. **Comprehensive Coverage**: >90% code coverage for core components
2. **Test Pyramid**: Unit tests (70%), integration tests (20%), E2E tests (10%)
3. **Fast Feedback**: Unit tests run in <1 second each
4. **Reliable**: Tests are deterministic and isolated
5. **Maintainable**: Clear, readable test code with good documentation

### **Test Categories**

#### **Unit Tests** (`unit/`)
**Purpose**: Test individual components in isolation
**Scope**: Single functions, classes, or small modules
**Speed**: Very fast (<1s per test)
**Coverage Target**: >95%

#### **Integration Tests** (`integration/`)
**Purpose**: Test component interactions and system integration
**Scope**: Multiple components working together
**Speed**: Fast (<10s per test)
**Coverage Target**: >80%

#### **Performance Tests** (`performance/`)
**Purpose**: Validate performance requirements and benchmarks
**Scope**: System performance under load
**Speed**: Variable (can be slow)
**Coverage Target**: Key performance paths

## 🔧 Test Implementation Specifications

### **Template Testing** (`unit/test_templates.py`)
```python
class TestTemplateEngine:
    """Test template processing and rendering"""
    
    def test_basic_variable_substitution(self):
        """Test simple variable replacement"""
        template = "Hello {name}, you have {count} messages"
        variables = {"name": "Alice", "count": 5}
        expected = "Hello Alice, you have 5 messages"
        # Test implementation
    
    def test_missing_variable_handling(self):
        """Test behavior with missing required variables"""
        # Should raise appropriate exception
    
    def test_template_validation(self):
        """Test template structure validation"""
        # Test valid and invalid template structures
    
    def test_complex_templates(self):
        """Test complex templates with conditionals"""
        # Test advanced template features
    
    def test_template_caching(self):
        """Test template compilation caching"""
        # Verify caching improves performance

class TestTemplateRegistry:
    """Test template registration and discovery"""
    
    def test_template_registration(self):
        """Test registering new templates"""
    
    def test_template_discovery(self):
        """Test automatic template discovery"""
    
    def test_version_management(self):
        """Test template versioning"""
    
    def test_duplicate_handling(self):
        """Test handling of duplicate template IDs"""
```

### **Strategy Testing** (`unit/test_strategies.py`)
```python
class TestRuleBasedStrategy:
    """Test rule-based prompt selection"""
    
    def test_simple_rule_matching(self):
        """Test basic rule condition matching"""
        strategy_config = {
            "rules": [
                {"condition": "document_type == 'medical'", "template": "medical_qa"}
            ]
        }
        context = {"document_type": "medical"}
        # Should select medical_qa template
    
    def test_multiple_rule_evaluation(self):
        """Test evaluation of multiple rules"""
    
    def test_rule_priority_handling(self):
        """Test rule priority and conflict resolution"""
    
    def test_fallback_behavior(self):
        """Test fallback when no rules match"""

class TestContextAwareStrategy:
    """Test context-aware selection strategy"""
    
    def test_context_factor_weighting(self):
        """Test context factor weight application"""
    
    def test_dynamic_adaptation(self):
        """Test adaptation to changing context"""
```

### **Integration Testing** (`integration/test_rag_integration.py`)
```python
class TestRAGIntegration:
    """Test prompts integration with RAG pipeline"""
    
    def test_retrieval_to_prompt_flow(self):
        """Test complete retrieval → prompt selection → response flow"""
        # Set up RAG pipeline with prompts
        # Execute query
        # Verify correct template selection
        # Verify response quality
    
    def test_metadata_integration(self):
        """Test document metadata usage in prompt selection"""
    
    def test_strategy_chaining_with_retrieval(self):
        """Test strategy chaining with retrieval strategies"""
    
    def test_performance_with_rag(self):
        """Test performance impact of prompts on RAG pipeline"""

class TestConfigurationIntegration:
    """Test configuration system integration"""
    
    def test_config_loading(self):
        """Test loading prompts from configuration files"""
    
    def test_environment_specific_configs(self):
        """Test environment-specific prompt configurations"""
    
    def test_config_validation(self):
        """Test configuration validation and error handling"""
```

### **Performance Testing** (`performance/test_template_speed.py`)
```python
class TestTemplatePerformance:
    """Test template system performance"""
    
    def test_template_compilation_speed(self):
        """Test template compilation performance"""
        # Measure compilation time for various template sizes
        # Assert performance thresholds
    
    def test_rendering_speed(self):
        """Test template rendering performance"""
        # Measure rendering time with different variable counts
    
    def test_cache_effectiveness(self):
        """Test caching impact on performance"""
        # Compare cached vs uncached performance
    
    def test_concurrent_template_processing(self):
        """Test performance under concurrent load"""
        # Simulate multiple concurrent template requests
    
    def test_memory_usage(self):
        """Test memory consumption during template processing"""
        # Monitor memory usage patterns

class TestStrategyPerformance:
    """Test strategy selection performance"""
    
    def test_strategy_selection_speed(self):
        """Test speed of strategy template selection"""
        # Measure selection time for different context sizes
    
    def test_scalability_with_template_count(self):
        """Test performance scaling with template library size"""
        # Test with 10, 100, 1000+ templates
```

## 📊 Test Data and Fixtures

### **Template Fixtures** (`fixtures/templates/`)
```json
{
  "test_templates": {
    "simple_qa": {
      "template": "Question: {question}\nAnswer: Based on {context}, the answer is:",
      "input_variables": ["question", "context"],
      "metadata": {"complexity": "low", "use_case": "qa"}
    },
    "complex_analysis": {
      "template": "Analyze the following:\n{context}\n\nFocus on:\n1. {aspect1}\n2. {aspect2}\n\nProvide detailed analysis:",
      "input_variables": ["context", "aspect1", "aspect2"],
      "metadata": {"complexity": "high", "use_case": "analysis"}
    }
  }
}
```

### **Context Fixtures** (`fixtures/sample_data/contexts.json`)
```json
{
  "test_contexts": {
    "medical_context": {
      "documents": [
        {"type": "medical", "domain": "cardiology", "content": "Patient presents with chest pain..."}
      ],
      "user": {"expertise": "doctor", "role": "physician"},
      "query": {"text": "What is the diagnosis?", "type": "diagnostic"}
    },
    "general_context": {
      "documents": [
        {"type": "general", "domain": "knowledge", "content": "The Earth revolves around the Sun..."}
      ],
      "user": {"expertise": "general", "role": "student"},
      "query": {"text": "How long is a year?", "type": "factual"}
    }
  }
}
```

### **Configuration Fixtures** (`fixtures/configs/`)
Multiple test configuration files for different scenarios:
- `minimal_config.json` - Bare minimum configuration
- `full_featured_config.json` - All features enabled
- `performance_config.json` - Performance-optimized settings
- `invalid_config.json` - Invalid configurations for error testing

## 🛠️ Testing Utilities

### **Test Helpers** (`utils/test_helpers.py`)
```python
class PromptTestHelper:
    """Helper utilities for prompt system tests"""
    
    @staticmethod
    def create_mock_context(doc_type: str = "general", 
                          user_expertise: str = "general") -> PromptContext:
        """Create standardized test context"""
    
    @staticmethod
    def assert_template_valid(template: Dict) -> None:
        """Assert template meets validation requirements"""
    
    @staticmethod
    def measure_performance(func: Callable, *args, **kwargs) -> PerformanceResult:
        """Measure function performance"""
    
    @staticmethod
    def create_test_templates(count: int) -> List[Dict]:
        """Generate test templates for performance testing"""
```

### **Mock Components** (`utils/mock_components.py`)
```python
class MockTemplateRegistry:
    """Mock template registry for testing"""
    
class MockStrategy:
    """Mock strategy for testing"""
    
class MockRAGPipeline:
    """Mock RAG pipeline for integration testing"""

class MockVectorStore:
    """Mock vector store for testing"""
```

### **Custom Assertions** (`utils/assertion_helpers.py`)
```python
def assert_template_selection_correct(context: PromptContext, 
                                    selected_template: str, 
                                    expected_criteria: Dict) -> None:
    """Assert template selection matches expected criteria"""

def assert_performance_within_bounds(actual_time: float, 
                                   max_time: float, 
                                   operation: str) -> None:
    """Assert operation completed within performance bounds"""

def assert_response_quality(response: str, 
                          quality_metrics: Dict) -> None:
    """Assert response meets quality requirements"""
```

## 🚀 Test Execution

### **Running Tests** (Future Commands)
```bash
# Run all tests
pytest prompts/tests/

# Run specific test category
pytest prompts/tests/unit/
pytest prompts/tests/integration/
pytest prompts/tests/performance/

# Run with coverage
pytest prompts/tests/ --cov=prompts --cov-report=html

# Run performance tests only
pytest prompts/tests/performance/ -m performance

# Run tests with specific markers
pytest prompts/tests/ -m "not slow"
```

### **Test Configuration** (`conftest.py`)
```python
import pytest
from prompts.registry import TemplateRegistry
from prompts.utils.cache_manager import CacheManager

@pytest.fixture
def template_registry():
    """Provide clean template registry for tests"""
    registry = TemplateRegistry()
    yield registry
    registry.clear()

@pytest.fixture
def sample_templates():
    """Provide standard test templates"""
    return load_test_templates()

@pytest.fixture
def mock_rag_pipeline():
    """Provide mock RAG pipeline"""
    return MockRAGPipeline()

@pytest.fixture(autouse=True)
def clear_cache():
    """Clear cache between tests"""
    CacheManager().clear_all_cache()
    yield
    CacheManager().clear_all_cache()
```

## 📈 Coverage and Quality Metrics

### **Coverage Targets**
- **Overall System**: >90%
- **Core Components**: >95%
- **Utilities**: >85%
- **Integration Points**: >80%

### **Quality Metrics**
- **Test Speed**: Unit tests <1s each
- **Test Reliability**: <1% flaky test rate
- **Maintainability**: Clear, readable test code
- **Documentation**: All tests have clear descriptions

### **Continuous Integration**
```yaml
# CI pipeline for prompt system tests
name: Prompt System Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements-test.txt
      - name: Run unit tests
        run: pytest prompts/tests/unit/ --cov=prompts
      - name: Run integration tests
        run: pytest prompts/tests/integration/
      - name: Run performance tests
        run: pytest prompts/tests/performance/ --benchmark-only
```

## 🔄 Test Development Roadmap

### **Phase 1: Foundation**
- [ ] Basic test structure and fixtures
- [ ] Core unit tests for templates and strategies
- [ ] Simple integration tests
- [ ] Test utilities and helpers

### **Phase 2: Comprehensive Coverage**
- [ ] Complete unit test coverage
- [ ] Advanced integration scenarios
- [ ] Performance benchmarks
- [ ] Mock system improvements

### **Phase 3: Advanced Testing**
- [ ] Load testing and stress tests
- [ ] Security testing
- [ ] A/B testing validation
- [ ] Automated test generation

### **Phase 4: Production Testing**
- [ ] Production monitoring integration
- [ ] Real-world performance validation
- [ ] User acceptance testing
- [ ] Regression test automation

The testing framework ensures the prompts system maintains high quality, reliability, and performance while enabling confident development and deployment within the LlamaFarm ecosystem.