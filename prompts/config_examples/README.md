# Prompt Configuration Examples

## 📋 Overview

This directory contains example configurations demonstrating how to integrate the prompts system with the LlamaFarm RAG framework. Following the project's configuration-driven philosophy, these examples show how to set up prompts for various use cases and domains.

## 🎯 Configuration Philosophy

The prompts system extends the existing RAG configuration format with:
- **Seamless Integration**: Prompts configs merge with existing RAG configs
- **Environment Specificity**: Different prompts for dev/staging/production
- **A/B Testing Support**: Multiple prompt variants in single configuration
- **Strategy Selection**: Intelligent prompt routing based on context

## 📁 Configuration Categories

```
config_examples/
├── basic/                  # Simple, single-strategy configs
├── advanced/              # Complex, multi-strategy configs
├── domain_specific/       # Specialized domain configurations
├── integration/           # RAG pipeline integration examples
├── ab_testing/           # A/B testing configurations
└── production/           # Production-ready configurations
```

## 🔧 Basic Configuration Examples

### **Simple Q&A Configuration**
*File: `basic/simple_qa_config.json`*

```json
{
  "name": "Simple Q&A Pipeline with Prompts",
  "version": "1.0",
  
  "rag_pipeline": {
    "retrieval_strategy": "basic_similarity",
    "retrieval_config": {
      "top_k": 5,
      "similarity_threshold": 0.7
    }
  },
  
  "prompts": {
    "enabled": true,
    "strategy": "static_selection",
    "config": {
      "default_template": "qa_basic",
      "fallback_template": "qa_simple"
    },
    
    "templates": {
      "qa_basic": {
        "type": "basic",
        "template": "Based on the following context:\\n{context}\\n\\nAnswer this question: {question}\\n\\nProvide a clear and concise answer:",
        "input_variables": ["context", "question"],
        "metadata": {
          "use_case": "general_qa",
          "complexity": "low"
        }
      }
    }
  }
}
```

### **Multi-Template Configuration**
*File: `basic/multi_template_config.json`*

```json
{
  "name": "Multi-Template Prompt System",
  
  "prompts": {
    "strategy": "rule_based",
    "config": {
      "rules": [
        {
          "condition": "query_type == 'summary'",
          "template": "document_summary"
        },
        {
          "condition": "query_type == 'analysis'", 
          "template": "detailed_analysis"
        }
      ],
      "fallback": "qa_basic"
    },
    
    "templates": {
      "document_summary": {
        "template": "Summarize the following document:\\n{context}\\n\\nProvide a concise summary highlighting the key points:",
        "input_variables": ["context"],
        "metadata": {"use_case": "summarization"}
      },
      
      "detailed_analysis": {
        "template": "Analyze the following content:\\n{context}\\n\\nProvide a detailed analysis considering:\\n1. Key themes\\n2. Important insights\\n3. Actionable conclusions",
        "input_variables": ["context"],
        "metadata": {"use_case": "analysis"}
      }
    }
  }
}
```

## 🏗️ Advanced Configuration Examples

### **Context-Aware Strategy**
*File: `advanced/context_aware_config.json`*

```json
{
  "name": "Context-Aware Prompt Selection",
  
  "prompts": {
    "strategy": "context_aware",
    "config": {
      "context_factors": {
        "document_type": 0.3,
        "user_expertise": 0.25,
        "query_complexity": 0.2,
        "response_requirements": 0.15,
        "system_load": 0.1
      },
      
      "selection_rules": {
        "expert_technical": {
          "conditions": {
            "user_expertise": "expert",
            "document_type": "technical"
          },
          "template": "technical_expert_analysis"
        },
        
        "general_simple": {
          "conditions": {
            "user_expertise": "general", 
            "query_complexity": "low"
          },
          "template": "simple_explanation"
        }
      },
      
      "fallback_chain": ["qa_detailed", "qa_basic"]
    }
  }
}
```

### **Performance-Optimized Configuration**
*File: `advanced/performance_optimized_config.json`*

```json
{
  "name": "Performance-Optimized Prompt System",
  
  "prompts": {
    "strategy": "performance_optimized",
    "config": {
      "optimization_target": "speed",
      "constraints": {
        "max_tokens": 2000,
        "max_latency_ms": 1000,
        "min_quality_score": 0.8
      },
      
      "template_performance_profiles": {
        "qa_fast": {
          "avg_tokens": 500,
          "avg_latency_ms": 200,
          "quality_score": 0.8
        },
        "qa_balanced": {
          "avg_tokens": 1000,
          "avg_latency_ms": 500,
          "quality_score": 0.85
        }
      },
      
      "load_balancing": {
        "high_load": "qa_fast",
        "medium_load": "qa_balanced", 
        "low_load": "qa_detailed"
      }
    }
  }
}
```

## 🏥 Domain-Specific Examples

### **Medical Document Processing**
*File: `domain_specific/medical_config.json`*

```json
{
  "name": "Medical Document Analysis Pipeline",
  "compliance": ["HIPAA"],
  
  "rag_pipeline": {
    "retrieval_strategy": "metadata_filtered",
    "retrieval_config": {
      "filters": {
        "document_type": "medical",
        "privacy_level": "compliant"  
      }
    }
  },
  
  "prompts": {
    "strategy": "domain_specific",
    "domain": "medical",
    "config": {
      "compliance_mode": true,
      "pii_handling": "strict",
      
      "templates": {
        "medical_qa": {
          "template": "Based on the medical documentation provided:\\n{context}\\n\\nMedical Query: {question}\\n\\nIMPORTANT: This response is for informational purposes only and should not replace professional medical advice.\\n\\nAnalysis:",
          "metadata": {
            "domain": "medical",
            "compliance": ["HIPAA"],
            "disclaimer_required": true
          }
        },
        
        "clinical_summary": {
          "template": "Summarize the following clinical information:\\n{context}\\n\\nProvide a structured summary including:\\n- Key findings\\n- Relevant symptoms\\n- Treatment considerations\\n\\nNote: This is a summary of provided information only.",
          "metadata": {
            "domain": "medical",
            "output_format": "structured"
          }
        }
      }
    }
  }
}
```

### **Legal Document Analysis**
*File: `domain_specific/legal_config.json`*

```json
{
  "name": "Legal Document Analysis System",
  
  "prompts": {
    "strategy": "domain_specific",
    "domain": "legal",
    "config": {
      "citation_requirements": true,
      "jurisdiction_awareness": true,
      
      "templates": {
        "legal_analysis": {
          "template": "Legal Document Analysis:\\n\\nDocument Context:\\n{context}\\n\\nQuery: {question}\\n\\nAnalysis:\\n1. Relevant Legal Principles:\\n2. Document Provisions:\\n3. Legal Implications:\\n\\nDISCLAIMER: This analysis is for informational purposes only and does not constitute legal advice.",
          "metadata": {
            "domain": "legal",
            "requires_disclaimer": true,
            "citation_format": "bluebook"
          }
        },
        
        "contract_review": {
          "template": "Contract Review:\\n{context}\\n\\nReview Focus: {question}\\n\\nContract Analysis:\\n- Key Terms:\\n- Potential Issues:\\n- Recommendations:\\n\\nNote: This is a preliminary analysis only.",
          "metadata": {
            "domain": "legal",
            "output_format": "structured_review"
          }
        }
      }
    }
  }
}
```

## 🔗 RAG Integration Examples

### **Complete RAG + Prompts Pipeline**
*File: `integration/full_pipeline_config.json`*

```json
{
  "name": "Complete RAG Pipeline with Intelligent Prompts",
  "version": "2.0",
  
  "parser": {
    "type": "PDFParser",
    "config": {
      "extract_metadata": true,
      "preserve_structure": true
    }
  },
  
  "embedder": {
    "type": "OllamaEmbedder", 
    "config": {
      "model": "nomic-embed-text",
      "dimension": 768
    }
  },
  
  "vector_store": {
    "type": "ChromaStore",
    "config": {
      "collection_name": "documents_with_prompts",
      "persistence_path": "./chroma_prompts_db"
    }
  },
  
  "retrieval": {
    "strategy": "hybrid_universal",
    "config": {
      "strategies": [
        {
          "name": "metadata_filtered",
          "weight": 0.6
        },
        {
          "name": "reranked", 
          "weight": 0.4
        }
      ],
      "fusion_method": "reciprocal_rank"
    }
  },
  
  "prompts": {
    "strategy": "context_aware",
    "config": {
      "adapt_to_retrieval": true,
      "use_document_metadata": true,
      "consider_retrieval_quality": true,
      
      "retrieval_quality_thresholds": {
        "high_quality": {
          "min_relevance": 0.8,
          "min_results": 3,
          "template": "confident_answer"
        },
        "medium_quality": {
          "min_relevance": 0.6, 
          "template": "cautious_answer"
        },
        "low_quality": {
          "template": "clarifying_questions"
        }
      }
    }
  }
}
```

## 🧪 A/B Testing Examples

### **Template Comparison Test**
*File: `ab_testing/template_comparison_config.json`*

```json
{
  "name": "Template A/B Test Configuration",
  
  "ab_testing": {
    "enabled": true,
    "test_name": "qa_template_optimization",
    "duration_days": 14,
    "traffic_split": 0.5,
    
    "variants": {
      "control": {
        "template": "qa_basic",
        "description": "Current basic Q&A template"
      },
      "treatment": {
        "template": "qa_enhanced", 
        "description": "Enhanced Q&A with reasoning steps"
      }
    },
    
    "success_metrics": [
      "user_satisfaction",
      "response_relevance", 
      "task_completion_rate"
    ],
    
    "monitoring": {
      "track_user_feedback": true,
      "measure_response_quality": true,
      "monitor_system_performance": true
    }
  },
  
  "templates": {
    "qa_enhanced": {
      "template": "Context: {context}\\n\\nQuestion: {question}\\n\\nLet me analyze this step by step:\\n1. Key information from context:\\n2. Direct answer:\\n3. Supporting details:\\n\\nFinal Answer:",
      "metadata": {
        "variant": "treatment",
        "approach": "structured_reasoning"
      }
    }
  }
}
```

### **Strategy Comparison Test**
*File: `ab_testing/strategy_comparison_config.json`*

```json
{
  "name": "Prompt Strategy A/B Test",
  
  "ab_testing": {
    "test_name": "selection_strategy_comparison",
    "variants": {
      "rule_based": {
        "strategy": "rule_based",
        "config": {
          "rules": [
            {
              "condition": "document_type == 'technical'",
              "template": "technical_analysis"
            }
          ]
        }
      },
      "context_aware": {
        "strategy": "context_aware",
        "config": {
          "context_factors": {
            "document_type": 0.4,
            "user_expertise": 0.3,
            "query_complexity": 0.3
          }
        }
      }
    }
  }
}
```

## 🚀 Production Examples

### **Production-Ready Configuration**
*File: `production/enterprise_config.json`*

```json
{
  "name": "Enterprise Production Prompt System",
  "environment": "production",
  
  "prompts": {
    "strategy": "multi_tier",
    "config": {
      "primary_strategy": "ml_driven",
      "fallback_strategies": ["context_aware", "rule_based", "static"],
      
      "performance_monitoring": {
        "enabled": true,
        "metrics": ["latency", "quality", "cost", "user_satisfaction"],
        "alerting": {
          "quality_threshold": 0.8,
          "latency_threshold_ms": 2000,
          "error_rate_threshold": 0.05
        }
      },
      
      "security": {
        "input_validation": true,
        "output_filtering": true,
        "audit_logging": true,
        "rate_limiting": {
          "requests_per_minute": 1000,
          "requests_per_user": 100
        }
      },
      
      "caching": {
        "enabled": true,
        "backend": "redis",
        "ttl_seconds": 3600,
        "max_cache_size": "10GB"
      }
    }
  }
}
```

## 📊 Configuration Validation

All configuration files follow JSON Schema validation:

### **Schema Location**
- Base schema: `prompts/schemas/prompt_config.schema.json`
- Strategy schemas: `prompts/schemas/strategies/`
- Template schemas: `prompts/schemas/templates/`

### **Validation Commands** (Future)
```bash
# Validate configuration
llamafarm prompts validate config_examples/basic/simple_qa_config.json

# Test configuration
llamafarm prompts test config_examples/basic/simple_qa_config.json

# Deploy configuration
llamafarm prompts deploy config_examples/production/enterprise_config.json
```

## 🔄 Configuration Evolution

### **Version Migration**
Configuration files include version information for migration support:
```json
{
  "config_version": "2.0",
  "migration_from": "1.x",
  "breaking_changes": ["template_format", "strategy_interface"]
}
```

### **Environment Inheritance**
```json
{
  "base_config": "basic/simple_qa_config.json",
  "environment_overrides": {
    "production": {
      "prompts.config.performance_monitoring.enabled": true,
      "prompts.config.caching.enabled": true
    }
  }
}
```

These configuration examples provide a comprehensive guide for integrating the prompts system with the LlamaFarm RAG framework, enabling powerful, flexible, and maintainable prompt engineering workflows.