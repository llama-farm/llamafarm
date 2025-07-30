# Prompt Strategies Directory

## 🎯 Overview

The strategies directory contains the intelligent prompt selection and routing logic for the LlamaFarm prompts system. Strategies determine which prompt template to use based on context, user requirements, document characteristics, and system state.

## 🧠 Strategy Architecture

Following the same pattern as the universal retrieval strategies in `rag/retrieval/strategies/`, prompt strategies are:
- **Database-Agnostic**: Work with any vector store or retrieval system
- **Context-Aware**: Make decisions based on document metadata, user context, and system state
- **Composable**: Can be combined and chained for complex decision-making
- **Configurable**: Behavior controlled through JSON configuration
- **Plugin-Based**: Easy addition of new strategies

## 📁 Directory Structure

```
strategies/
├── base.py                 # Base strategy interface (future)
├── registry.py            # Strategy registration system (future)
├── rule_based/            # Rule-based selection strategies
├── ml_driven/             # Machine learning prompt selection
├── context_aware/         # Context-dependent strategies  
├── performance_optimized/ # Performance-focused strategies
├── domain_specific/       # Domain-specialized strategies
└── hybrid/               # Combined strategy approaches
```

## 🔧 Strategy Categories

### **Rule-Based Strategies** (`rule_based/`)
**Purpose**: Select prompts based on explicit rules and conditions
**Use Cases**:
- Domain-specific routing (medical → medical templates)
- User role-based selection (expert → detailed prompts)
- Document type matching (PDF → document analysis prompts)
- Response length requirements (brief → concise templates)

**Expected Implementation**:
```json
{
  "strategy_type": "rule_based",
  "rules": [
    {
      "condition": "document_metadata.type == 'medical'",
      "template": "medical_qa_specialized",
      "priority": 10
    },
    {
      "condition": "user_context.expertise == 'expert'",
      "template": "technical_detailed",
      "priority": 8
    }
  ],
  "fallback": "qa_basic"
}
```

### **ML-Driven Strategies** (`ml_driven/`)
**Purpose**: Learn optimal prompt selection from usage patterns and feedback
**Use Cases**:
- Personalized prompt selection
- Performance-based optimization
- User satisfaction maximization
- Automated A/B testing

**Expected Components**:
- Feature extraction from context
- Model training pipelines
- Real-time inference systems
- Feedback integration loops

### **Context-Aware Strategies** (`context_aware/`)
**Purpose**: Dynamic prompt selection based on contextual information
**Use Cases**:
- Retrieved document characteristics
- User query complexity analysis
- Historical conversation context
- System performance state

**Context Sources**:
- Document metadata (type, domain, length, complexity)
- Retrieval results (relevance scores, result count)
- User information (role, preferences, history)
- System state (load, performance metrics)

### **Performance-Optimized Strategies** (`performance_optimized/`)
**Purpose**: Select prompts optimized for speed, cost, or resource usage
**Use Cases**:
- Low-latency applications
- Cost-sensitive deployments
- Resource-constrained environments
- High-throughput scenarios

**Optimization Targets**:
- Token count minimization
- Processing time reduction
- Memory usage optimization
- API cost reduction

### **Domain-Specific Strategies** (`domain_specific/`)
**Purpose**: Specialized selection logic for specific domains
**Domains**:
- Medical: HIPAA compliance, clinical terminology
- Legal: Citation requirements, jurisdiction awareness
- Financial: Regulatory compliance, numerical accuracy
- Technical: Code complexity, language-specific needs

### **Hybrid Strategies** (`hybrid/`)
**Purpose**: Combine multiple strategy approaches for optimal selection
**Combinations**:
- Rule-based + ML-driven
- Context-aware + Performance-optimized
- Domain-specific + Rule-based
- Multi-strategy voting systems

## 🏗️ Strategy Interface Design

### **Base Strategy Class** (Future Implementation)
```python
class PromptStrategy:
    """Base class for all prompt selection strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
    
    def select_template(self, context: PromptContext) -> str:
        """Select the best template ID for given context"""
        pass
    
    def validate_config(self) -> bool:
        """Validate strategy configuration"""
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return strategy metadata for monitoring"""
        pass
```

### **Prompt Context Structure**
```json
{
  "query": {
    "text": "User query text",
    "type": "question|request|command",
    "complexity": "low|medium|high", 
    "language": "en",
    "intent": "qa|summarization|analysis"
  },
  "user": {
    "role": "expert|general|beginner",
    "preferences": {
      "response_style": "detailed|concise|technical",
      "format": "text|structured|json"
    },
    "history": ["previous_templates_used"]
  },
  "documents": {
    "count": 5,
    "types": ["pdf", "text"],
    "domains": ["medical", "general"],
    "metadata": {
      "avg_relevance": 0.85,
      "total_length": 10000,
      "languages": ["en"]
    }
  },
  "system": {
    "retrieval_strategy": "metadata_filtered",
    "performance_constraints": {
      "max_tokens": 4000,
      "max_latency_ms": 2000,
      "cost_limit": 0.01
    },
    "current_load": "low|medium|high"
  }
}
```

## 🎪 Strategy Examples

### **Simple Rule-Based Strategy**
```json
{
  "strategy_id": "domain_router",
  "name": "Domain-Based Template Router",
  "type": "rule_based",
  "description": "Routes prompts based on document domain",
  
  "rules": [
    {
      "name": "medical_documents",
      "condition": "any(doc.domain == 'medical' for doc in context.documents)",
      "template": "medical_qa_hipaa_compliant",
      "confidence": 0.9
    },
    {
      "name": "legal_documents", 
      "condition": "any(doc.domain == 'legal' for doc in context.documents)",
      "template": "legal_analysis_detailed",
      "confidence": 0.9
    },
    {
      "name": "technical_queries",
      "condition": "context.query.complexity == 'high' and context.user.role == 'expert'",
      "template": "technical_detailed_analysis",
      "confidence": 0.8
    }
  ],
  
  "fallback": {
    "template": "qa_basic",
    "reason": "no_rules_matched"
  },
  
  "metadata": {
    "domains": ["medical", "legal", "technical"],
    "complexity": "low",
    "performance_impact": "minimal"
  }
}
```

### **Performance-Optimized Strategy**
```json
{
  "strategy_id": "speed_optimizer",
  "name": "Speed-Optimized Template Selection",
  "type": "performance_optimized",
  "description": "Selects fastest templates based on performance constraints",
  
  "optimization_target": "latency",
  "constraints": {
    "max_tokens": 2000,
    "max_processing_time_ms": 1000,
    "acceptable_quality_threshold": 0.8
  },
  
  "template_performance_map": {
    "qa_basic": {"avg_tokens": 500, "avg_time_ms": 200, "quality_score": 0.75},
    "qa_detailed": {"avg_tokens": 1500, "avg_time_ms": 800, "quality_score": 0.9},
    "chain_of_thought": {"avg_tokens": 2500, "avg_time_ms": 1500, "quality_score": 0.95}
  },
  
  "selection_logic": "maximize_quality_within_constraints"
}
```

### **Context-Aware Strategy**
```json
{
  "strategy_id": "adaptive_selector",
  "name": "Adaptive Context-Aware Selection",
  "type": "context_aware",
  "description": "Dynamically adapts template selection based on full context",
  
  "context_weights": {
    "document_relevance": 0.3,
    "user_expertise": 0.2,
    "query_complexity": 0.2,
    "response_requirements": 0.15,
    "system_performance": 0.15
  },
  
  "decision_tree": {
    "high_relevance_expert_user": "technical_detailed_analysis",
    "high_relevance_general_user": "detailed_explanation",
    "medium_relevance_any_user": "balanced_qa",
    "low_relevance_any_user": "clarifying_questions"
  },
  
  "learning_enabled": true,
  "feedback_integration": true
}
```

## 🚀 Integration with RAG System

### **Pipeline Integration**
```json
{
  "rag_pipeline": {
    "retrieval_strategy": "metadata_filtered",
    "prompt_strategy": "domain_router",
    "strategy_config": {
      "fallback_chain": ["domain_router", "performance_optimizer", "basic_selector"],
      "override_conditions": {
        "emergency_mode": "qa_basic",
        "high_load": "speed_optimizer"
      }
    }
  }
}
```

### **Strategy Chaining**
Multiple strategies can be chained for complex decision-making:
1. **Primary Strategy**: Domain-based routing
2. **Secondary Strategy**: Performance optimization
3. **Fallback Strategy**: Basic rule-based selection

## 📊 Strategy Monitoring

### **Key Metrics**
- **Selection Accuracy**: How often the strategy selects the optimal template
- **Response Quality**: User satisfaction with strategy-selected templates
- **Performance Impact**: Strategy overhead on system performance
- **Coverage**: Percentage of contexts handled by primary vs fallback strategies

### **A/B Testing Framework**
```json
{
  "ab_test": {
    "name": "strategy_comparison",
    "variants": {
      "control": "rule_based_domain",
      "treatment": "ml_driven_selector"
    },
    "traffic_split": 0.5,
    "success_metrics": ["user_satisfaction", "response_quality", "system_performance"],
    "duration_days": 14
  }
}
```

## 🔄 Development Roadmap

### **Phase 1: Foundation**
- [ ] Base strategy interface
- [ ] Strategy registry system
- [ ] Basic rule-based strategies
- [ ] Configuration validation

### **Phase 2: Core Strategies**
- [ ] Context-aware selection
- [ ] Performance-optimized routing
- [ ] Domain-specific strategies
- [ ] Strategy chaining

### **Phase 3: Advanced Features**
- [ ] ML-driven selection
- [ ] Real-time learning
- [ ] A/B testing framework
- [ ] Advanced monitoring

### **Phase 4: Optimization**
- [ ] Performance tuning
- [ ] Predictive selection
- [ ] Multi-objective optimization
- [ ] Automated strategy generation

## 🤝 Contributing Strategies

### **Adding New Strategies**
1. **Define Strategy Purpose**: Clear use case and objectives
2. **Implement Interface**: Follow base strategy pattern
3. **Configuration Schema**: JSON schema for strategy config
4. **Testing**: Unit tests and integration tests
5. **Documentation**: Strategy documentation and examples
6. **Registration**: Add to strategy registry

### **Strategy Design Guidelines**
- **Single Responsibility**: Each strategy should have one clear purpose
- **Composability**: Enable combination with other strategies
- **Configurability**: Behavior controlled through configuration
- **Performance**: Minimize selection overhead
- **Monitoring**: Include metrics and logging

Strategies are the intelligence layer of the prompts system, enabling sophisticated, adaptive prompt selection that improves system performance and user experience.