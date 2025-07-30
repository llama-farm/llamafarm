# Prompt Templates Directory

## 📋 Overview

This directory contains the core prompt template definitions for the LlamaFarm prompts system. Templates are organized by complexity and use case, following a modular architecture that enables easy composition and reuse.

## 🗂️ Directory Structure

```
templates/
├── basic/              # Simple, single-purpose templates
├── chat/              # Conversational and multi-turn templates
├── few_shot/          # Example-driven learning templates
├── advanced/          # Complex, multi-step templates
└── domain_specific/   # Specialized domain templates
```

## 🔧 Template Categories

### **Basic Templates** (`basic/`)
**Purpose**: Simple, focused templates for common tasks
**Use Cases**:
- Question answering
- Text summarization
- Classification
- Simple generation

**Expected Files**:
- `qa_simple.json` - Basic Q&A template
- `summarize_document.json` - Document summarization
- `classify_text.json` - Text classification
- `extract_entities.json` - Named entity extraction

### **Chat Templates** (`chat/`)
**Purpose**: Conversational interfaces and multi-turn dialogues
**Use Cases**:
- Chatbots and assistants
- Customer support
- Interactive tutorials
- Role-playing scenarios

**Expected Files**:
- `assistant_basic.json` - General assistant persona
- `customer_support.json` - Support conversation templates
- `tutor_interactive.json` - Educational dialogue
- `technical_expert.json` - Domain expert persona

### **Few-Shot Templates** (`few_shot/`)
**Purpose**: Example-driven templates that teach through demonstration
**Use Cases**:
- Pattern recognition
- Format standardization
- Style mimicking
- Domain-specific transformations

**Expected Files**:
- `format_transform.json` - Transform data formats
- `style_mimic.json` - Mimic writing styles
- `code_convert.json` - Convert between code languages
- `translation_pattern.json` - Translation with examples

### **Advanced Templates** (`advanced/`)
**Purpose**: Complex, multi-step templates with sophisticated logic
**Use Cases**:
- Chain-of-thought reasoning
- Multi-step workflows
- Self-correction loops
- Composite analysis

**Expected Files**:
- `chain_of_thought.json` - Step-by-step reasoning
- `multi_step_analysis.json` - Complex analysis workflow
- `self_correction.json` - Error detection and correction
- `comparative_analysis.json` - Compare multiple sources

### **Domain-Specific Templates** (`domain_specific/`)
**Purpose**: Specialized templates for specific domains
**Use Cases**:
- Medical document analysis
- Legal text processing
- Financial analysis
- Code documentation
- Scientific research

**Expected Files**:
- `medical/` - Medical domain templates
- `legal/` - Legal domain templates
- `financial/` - Financial analysis templates
- `code/` - Software development templates
- `scientific/` - Research and academic templates

## 📄 Template File Format

### **Standard Template Structure**
```json
{
  "template_id": "unique_identifier",
  "name": "Human Readable Name",
  "version": "1.0.0",
  "type": "basic|chat|few_shot|advanced|domain_specific",
  "description": "Detailed description of template purpose",
  
  "template": "Template content with {variables}",
  "input_variables": ["variable1", "variable2"],
  "optional_variables": ["optional1"],
  
  "metadata": {
    "use_case": ["qa", "summarization", "analysis"],
    "complexity": "low|medium|high",
    "domain": "general|medical|legal|financial|technical",
    "language": "en",
    "max_tokens": 4000,
    "min_context_length": 100
  },
  
  "configuration": {
    "temperature": 0.7,
    "max_tokens": 1000,
    "stop_sequences": ["\\n\\n", "---"],
    "format_output": true
  },
  
  "validation": {
    "required_fields": ["context", "question"],
    "min_input_length": 10,
    "max_input_length": 10000,
    "input_validation_regex": "^[A-Za-z0-9\\s.,!?-]+$"
  },
  
  "examples": [
    {
      "input": {"context": "...", "question": "..."},
      "expected_output": "Example output format"
    }
  ],
  
  "tags": ["retrieval", "qa", "general"],
  "author": "template_author",
  "created_date": "2024-01-01",
  "last_modified": "2024-01-01"
}
```

### **Chat Template Structure**
```json
{
  "template_id": "chat_assistant_basic",
  "type": "chat",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant specialized in {domain}. Always provide {response_style} responses."
    },
    {
      "role": "user", 
      "content": "{user_input}"
    }
  ],
  "input_variables": ["domain", "response_style", "user_input"],
  // ... rest of standard structure
}
```

### **Few-Shot Template Structure**
```json
{
  "template_id": "few_shot_transform",
  "type": "few_shot",
  "prefix": "Transform the following examples from format A to format B:",
  "examples": [
    {
      "input": "Example input 1",
      "output": "Transformed output 1"
    },
    {
      "input": "Example input 2", 
      "output": "Transformed output 2"
    }
  ],
  "suffix": "Now transform: {input}",
  "input_variables": ["input"],
  // ... rest of standard structure
}
```

## 🎯 Template Design Guidelines

### **1. Clarity and Purpose**
- Each template should have a single, clear purpose
- Use descriptive names and comprehensive documentation
- Include usage examples and expected outcomes

### **2. Variable Management**
- Use clear, descriptive variable names
- Distinguish between required and optional variables
- Provide default values where appropriate

### **3. Flexibility and Reusability**
- Design templates to be modular and composable
- Avoid hardcoded values; use variables instead
- Enable customization through configuration options

### **4. Performance Considerations**
- Optimize token usage without sacrificing quality
- Consider prompt length and complexity
- Include performance metadata (estimated tokens, processing time)

### **5. Validation and Testing**
- Include input validation rules
- Provide test examples with expected outputs
- Document edge cases and error conditions

## 🔄 Template Lifecycle

### **1. Creation**
- Define template purpose and scope
- Create template file following standard format
- Add comprehensive metadata and examples
- Write validation rules

### **2. Testing**
- Validate template syntax and structure
- Test with various input combinations
- Verify output quality and consistency
- Performance testing for token usage

### **3. Registration**
- Register template in the template registry
- Add to appropriate category index
- Update documentation and examples
- Version control integration

### **4. Optimization**
- Monitor template performance metrics
- A/B test template variations
- Optimize based on usage patterns
- Update based on user feedback

### **5. Maintenance**
- Regular review and updates
- Deprecation of outdated templates
- Migration guides for breaking changes
- Documentation updates

## 🚀 Next Steps for Implementation

### **Phase 1: Basic Templates**
1. Create essential Q&A templates
2. Document summarization templates
3. Basic classification templates
4. Simple generation templates

### **Phase 2: Chat Templates**
1. General assistant personas
2. Domain-specific experts
3. Conversation management
4. Context handling

### **Phase 3: Advanced Templates**
1. Chain-of-thought templates
2. Multi-step reasoning
3. Self-correction logic
4. Composite workflows

### **Phase 4: Domain Specialization**
1. Medical document templates
2. Legal analysis templates
3. Financial data templates
4. Code documentation templates

## 📊 Template Metrics

Templates will be tracked for:
- **Usage Frequency**: How often each template is used
- **Performance**: Response quality and user satisfaction
- **Efficiency**: Token usage and processing time
- **Success Rate**: Percentage of successful completions

## 🤝 Contributing New Templates

1. **Choose Appropriate Category**: Select the right subdirectory
2. **Follow Naming Conventions**: Use descriptive, consistent names
3. **Complete Metadata**: Include all required metadata fields
4. **Provide Examples**: Include realistic usage examples
5. **Test Thoroughly**: Validate template functionality
6. **Document Integration**: Update relevant documentation

Templates are the foundation of the prompts system - well-designed templates enable powerful, flexible, and maintainable prompt engineering workflows.