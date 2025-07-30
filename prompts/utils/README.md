# Prompt Utilities Directory

## 🛠️ Overview

The utilities directory contains helper functions, shared components, and support tools for the LlamaFarm prompts system. These utilities provide common functionality used across templates, strategies, and the registry system.

## 📁 Directory Structure

```
utils/
├── __init__.py          # Package initialization (future)
├── template_engine.py   # Template processing and rendering (future)
├── variable_parser.py   # Variable extraction and validation (future)
├── context_builder.py   # Context object construction (future)
├── validators.py        # Input and output validation (future)
├── formatters.py        # Output formatting utilities (future)
├── cache_manager.py     # Caching utilities (future)
├── metrics.py          # Performance metrics collection (future)
└── exceptions.py       # Custom exception classes (future)
```

## 🔧 Utility Components

### **Template Engine** (`template_engine.py`)
**Purpose**: Core template processing and variable substitution
**Responsibilities**:
- Template parsing and compilation
- Variable substitution with type checking
- Conditional logic processing
- Template composition and inheritance
- Performance optimization

**Expected Interface**:
```python
class TemplateEngine:
    def compile_template(self, template: str) -> CompiledTemplate
    def render(self, template: CompiledTemplate, variables: Dict) -> str
    def validate_template(self, template: str) -> ValidationResult
    def extract_variables(self, template: str) -> List[str]
    def optimize_template(self, template: str) -> str
```

### **Variable Parser** (`variable_parser.py`)
**Purpose**: Handle template variables and parameter management
**Responsibilities**:
- Variable extraction from templates
- Type validation and conversion
- Default value management
- Required vs optional variable handling
- Variable dependency resolution

**Expected Interface**:
```python
class VariableParser:
    def extract_variables(self, template: str) -> List[Variable]
    def validate_inputs(self, variables: Dict, requirements: Dict) -> ValidationResult
    def apply_defaults(self, variables: Dict, defaults: Dict) -> Dict
    def resolve_dependencies(self, variables: Dict) -> Dict
    def sanitize_inputs(self, variables: Dict) -> Dict
```

### **Context Builder** (`context_builder.py`)
**Purpose**: Construct comprehensive prompt context objects
**Responsibilities**:
- Aggregate context from multiple sources
- Document metadata integration
- User preference incorporation
- System state information
- Context serialization and caching

**Expected Interface**:
```python
class ContextBuilder:
    def build_context(self, query: str, documents: List[Document], 
                     user_info: Dict, system_state: Dict) -> PromptContext
    def enrich_context(self, context: PromptContext, enrichers: List[Enricher]) -> PromptContext
    def serialize_context(self, context: PromptContext) -> str
    def deserialize_context(self, data: str) -> PromptContext
    def cache_context(self, context: PromptContext, ttl: int) -> str
```

### **Validators** (`validators.py`)
**Purpose**: Input and output validation utilities
**Responsibilities**:
- Template structure validation
- Input parameter validation
- Output format verification
- Content safety checking
- Schema compliance validation

**Expected Interface**:
```python
class InputValidator:
    def validate_query(self, query: str) -> ValidationResult
    def validate_context(self, context: str) -> ValidationResult
    def validate_variables(self, variables: Dict, schema: Dict) -> ValidationResult
    def check_content_safety(self, content: str) -> SafetyResult
    def sanitize_input(self, input_data: str) -> str

class OutputValidator:
    def validate_response(self, response: str, expected_format: str) -> ValidationResult
    def check_completeness(self, response: str, requirements: Dict) -> ValidationResult
    def verify_safety(self, response: str) -> SafetyResult
    def format_compliance(self, response: str, format_spec: Dict) -> ValidationResult
```

### **Formatters** (`formatters.py`)
**Purpose**: Output formatting and presentation utilities
**Responsibilities**:
- Response formatting (JSON, XML, Markdown)
- Template output standardization
- Multi-format support
- Custom formatting rules
- Presentation optimization

**Expected Interface**:
```python
class ResponseFormatter:
    def format_as_json(self, response: str, schema: Dict) -> str
    def format_as_markdown(self, response: str, style: str) -> str
    def format_as_xml(self, response: str, schema: Dict) -> str
    def apply_custom_format(self, response: str, formatter: Callable) -> str
    def extract_structured_data(self, response: str) -> Dict

class TemplateFormatter:
    def normalize_template(self, template: str) -> str
    def optimize_whitespace(self, template: str) -> str
    def add_formatting_hints(self, template: str) -> str
    def apply_style_guide(self, template: str, style: Dict) -> str
```

### **Cache Manager** (`cache_manager.py`)
**Purpose**: Caching utilities for templates and contexts
**Responsibilities**:
- Template compilation caching
- Context object caching
- Response caching with TTL
- Cache invalidation strategies
- Multi-level cache management

**Expected Interface**:
```python
class CacheManager:
    def cache_template(self, template_id: str, compiled_template: CompiledTemplate, ttl: int) -> bool
    def get_cached_template(self, template_id: str) -> Optional[CompiledTemplate]
    def cache_context(self, context_hash: str, context: PromptContext, ttl: int) -> bool
    def get_cached_context(self, context_hash: str) -> Optional[PromptContext]
    def cache_response(self, query_hash: str, response: str, ttl: int) -> bool
    def get_cached_response(self, query_hash: str) -> Optional[str]
    def invalidate_cache(self, pattern: str) -> int
    def clear_all_cache(self) -> bool
```

### **Metrics Collector** (`metrics.py`)
**Purpose**: Performance metrics and analytics collection
**Responsibilities**:
- Template performance tracking
- Strategy effectiveness measurement
- System performance monitoring
- User interaction analytics
- Cost and resource tracking

**Expected Interface**:
```python
class MetricsCollector:
    def track_template_usage(self, template_id: str, duration_ms: int, success: bool) -> None
    def track_strategy_selection(self, strategy_id: str, context: Dict, template_selected: str) -> None
    def track_user_satisfaction(self, template_id: str, rating: float, feedback: str) -> None
    def track_system_performance(self, component: str, metrics: Dict) -> None
    def get_template_metrics(self, template_id: str, timeframe: str) -> TemplateMetrics
    def get_strategy_metrics(self, strategy_id: str, timeframe: str) -> StrategyMetrics
    def export_metrics(self, format: str, destination: str) -> bool
```

### **Exception Handling** (`exceptions.py`)
**Purpose**: Custom exception classes for error handling
**Responsibilities**:
- Template processing errors
- Variable validation errors
- Strategy selection errors
- Context building errors
- System integration errors

**Expected Classes**:
```python
class PromptSystemError(Exception):
    """Base exception for prompt system errors"""

class TemplateError(PromptSystemError):
    """Template-related errors"""

class VariableError(PromptSystemError):
    """Variable validation and processing errors"""

class StrategyError(PromptSystemError):
    """Strategy selection and execution errors"""

class ContextError(PromptSystemError):
    """Context building and validation errors"""

class ValidationError(PromptSystemError):
    """Input/output validation errors"""

class CacheError(PromptSystemError):
    """Caching system errors"""

class ConfigurationError(PromptSystemError):
    """Configuration and setup errors"""
```

## 🔗 Integration Patterns

### **Template Engine Integration**
```python
# Template processing with utilities
from prompts.utils.template_engine import TemplateEngine
from prompts.utils.variable_parser import VariableParser
from prompts.utils.validators import InputValidator, OutputValidator

class PromptProcessor:
    def __init__(self):
        self.engine = TemplateEngine()
        self.parser = VariableParser()
        self.input_validator = InputValidator()
        self.output_validator = OutputValidator()
    
    def process_prompt(self, template: str, variables: Dict) -> str:
        # Validate inputs
        validation = self.input_validator.validate_variables(variables, template)
        if not validation.is_valid:
            raise ValidationError(validation.errors)
        
        # Compile and render template
        compiled = self.engine.compile_template(template)
        response = self.engine.render(compiled, variables)
        
        # Validate output
        output_validation = self.output_validator.validate_response(response, "text")
        if not output_validation.is_valid:
            raise ValidationError(output_validation.errors)
        
        return response
```

### **Caching Integration**
```python
# Caching integration with template system
from prompts.utils.cache_manager import CacheManager
from prompts.utils.context_builder import ContextBuilder

class CachedPromptSystem:
    def __init__(self):
        self.cache = CacheManager()
        self.context_builder = ContextBuilder()
    
    def get_response(self, query: str, documents: List[Document]) -> str:
        # Build context
        context = self.context_builder.build_context(query, documents, {}, {})
        context_hash = hash(str(context))
        
        # Check cache first
        cached_response = self.cache.get_cached_response(context_hash)
        if cached_response:
            return cached_response
        
        # Generate new response
        response = self.generate_response(context)
        
        # Cache the response
        self.cache.cache_response(context_hash, response, ttl=3600)
        
        return response
```

### **Metrics Integration**
```python
# Metrics collection integration
from prompts.utils.metrics import MetricsCollector
import time

class MonitoredPromptSystem:
    def __init__(self):
        self.metrics = MetricsCollector()
    
    def process_with_metrics(self, template_id: str, variables: Dict) -> str:
        start_time = time.time()
        success = False
        
        try:
            response = self.process_prompt(template_id, variables)
            success = True
            return response
        except Exception as e:
            raise
        finally:
            duration_ms = int((time.time() - start_time) * 1000)
            self.metrics.track_template_usage(template_id, duration_ms, success)
```

## 🛡️ Security and Safety Utilities

### **Content Safety**
```python
class ContentSafetyChecker:
    def check_prompt_injection(self, input_text: str) -> SafetyResult
    def detect_pii(self, content: str) -> PIIDetectionResult
    def validate_content_policy(self, content: str) -> PolicyResult
    def sanitize_user_input(self, input_text: str) -> str
```

### **Input Sanitization**
```python
class InputSanitizer:
    def remove_harmful_content(self, input_text: str) -> str
    def escape_special_characters(self, input_text: str) -> str
    def validate_length_limits(self, input_text: str, max_length: int) -> bool
    def normalize_encoding(self, input_text: str) -> str
```

## 🎯 Performance Optimization

### **Template Optimization**
```python
class TemplateOptimizer:
    def optimize_token_usage(self, template: str) -> str
    def minimize_redundancy(self, template: str) -> str
    def optimize_variable_placement(self, template: str) -> str
    def compress_template(self, template: str) -> str
```

### **Batch Processing**
```python
class BatchProcessor:
    def process_batch_templates(self, templates: List[str], variables_list: List[Dict]) -> List[str]
    def parallel_template_processing(self, templates: List[str], max_workers: int) -> List[str]
    def optimize_batch_size(self, total_items: int, system_resources: Dict) -> int
```

## 🚀 Development Roadmap

### **Phase 1: Core Utilities**
- [ ] Template engine implementation
- [ ] Variable parser and validator
- [ ] Basic exception classes
- [ ] Simple caching utilities

### **Phase 2: Advanced Features**
- [ ] Context builder system
- [ ] Comprehensive validators
- [ ] Output formatters
- [ ] Metrics collection

### **Phase 3: Performance & Security**
- [ ] Content safety checkers
- [ ] Performance optimizers
- [ ] Batch processing utilities
- [ ] Advanced caching strategies

### **Phase 4: Integration & Tools**
- [ ] CLI utility tools
- [ ] Development helpers
- [ ] Testing utilities
- [ ] Debugging tools

## 🤝 Contributing Guidelines

### **Adding New Utilities**
1. **Clear Purpose**: Each utility should have a single, well-defined purpose
2. **Consistent Interface**: Follow established patterns and naming conventions
3. **Comprehensive Testing**: Include unit tests for all functionality
4. **Documentation**: Provide clear documentation and usage examples
5. **Error Handling**: Implement proper error handling and logging

### **Utility Design Principles**
- **Reusability**: Design for use across multiple components
- **Performance**: Optimize for common usage patterns
- **Extensibility**: Enable customization and extension
- **Reliability**: Handle edge cases gracefully
- **Observability**: Include logging and monitoring hooks

The utilities directory provides the foundational tools and helpers that enable the prompts system to operate efficiently, safely, and reliably within the LlamaFarm RAG framework.