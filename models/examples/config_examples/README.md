# Model Configuration Examples

## 🎯 Overview

This directory contains comprehensive configuration examples for the LlamaFarm Models system. Following the project's config-driven philosophy, these examples demonstrate how to set up fine-tuning jobs, dataset creation, and model deployment for various use cases and scenarios.

## 📁 Configuration Categories

```
config_examples/
├── basic/                 # Simple, getting-started configurations
├── production/           # Production-ready configurations
├── domain_specific/      # Domain-specialized configurations
├── method_comparison/    # Compare different fine-tuning methods
├── integration/         # RAG system integration examples
├── distributed/         # Multi-GPU and distributed training
├── deployment/          # Model deployment configurations
└── advanced/           # Advanced and experimental configurations
```

## 🚀 Quick Start Examples

### **Basic LoRA Fine-tuning** (`basic/lora_quickstart.json`)
Perfect for getting started with fine-tuning:

```json
{
  "name": "QuickStart LoRA Fine-tuning",
  "description": "Simple LoRA fine-tuning for general purpose enhancement",
  
  "model": {
    "base_model": "meta-llama/Llama-2-7b-hf",
    "method": "lora"
  },
  
  "fine_tuning": {
    "method_config": {
      "r": 16,
      "lora_alpha": 32,
      "lora_dropout": 0.1,
      "target_modules": ["q_proj", "v_proj"]
    },
    "training": {
      "epochs": 3,
      "batch_size": 4,
      "learning_rate": 2e-4,
      "warmup_steps": 100
    }
  },
  
  "dataset": {
    "source": {
      "type": "file",
      "path": "./datasets/alpaca_sample.json",
      "format": "alpaca"
    },
    "validation_split": 0.1
  },
  
  "output": {
    "save_path": "./trained_models/quickstart_lora",
    "save_steps": 500,
    "eval_steps": 100
  }
}
```

### **QLoRA for Large Models** (`basic/qlora_70b.json`)
Memory-efficient fine-tuning for large models:

```json
{
  "name": "QLoRA 70B Fine-tuning",
  "description": "Memory-efficient fine-tuning of 70B model using QLoRA",
  
  "model": {
    "base_model": "meta-llama/Llama-2-70b-hf",
    "method": "qlora"
  },
  
  "fine_tuning": {
    "method_config": {
      "load_in_4bit": true,
      "bnb_4bit_compute_dtype": "float16",
      "bnb_4bit_quant_type": "nf4",
      "bnb_4bit_use_double_quant": true,
      "r": 64,
      "lora_alpha": 128,
      "lora_dropout": 0.1,
      "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
    },
    "training": {
      "epochs": 2,
      "batch_size": 1,
      "gradient_accumulation_steps": 8,
      "learning_rate": 1e-4,
      "max_grad_norm": 1.0
    },
    "hardware": {
      "mixed_precision": "bf16",
      "gradient_checkpointing": true,
      "dataloader_num_workers": 4
    }
  },
  
  "monitoring": {
    "wandb_project": "llamafarm_qlora_70b",
    "logging_steps": 10,
    "eval_steps": 50
  }
}
```

## 🏭 Production Configurations

### **Production LoRA Pipeline** (`production/production_lora.json`)
Enterprise-ready configuration with comprehensive monitoring:

```json
{
  "name": "Production LoRA Pipeline",
  "version": "1.0.0",
  "environment": "production",
  
  "model": {
    "base_model": "meta-llama/Llama-2-13b-hf",
    "method": "lora",
    "version_control": {
      "track_changes": true,
      "backup_original": true
    }
  },
  
  "fine_tuning": {
    "method_config": {
      "r": 32,
      "lora_alpha": 64,
      "lora_dropout": 0.05,
      "target_modules": [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
      ],
      "bias": "none",
      "task_type": "CAUSAL_LM"
    },
    "training": {
      "epochs": 5,
      "batch_size": 8,
      "gradient_accumulation_steps": 4,
      "learning_rate": 1e-4,
      "warmup_steps": 500,
      "lr_scheduler_type": "cosine",
      "weight_decay": 0.01,
      "max_grad_norm": 1.0
    },
    "optimization": {
      "mixed_precision": "bf16",
      "gradient_checkpointing": true,
      "dataloader_num_workers": 8,
      "pin_memory": true
    }
  },
  
  "dataset": {
    "source": {
      "type": "rag_generated",
      "config": "./datasets/creation/production_qa_config.json"
    },
    "preprocessing": {
      "max_length": 2048,
      "truncation_strategy": "context_first",
      "padding": "max_length"
    },
    "validation": {
      "split_ratio": 0.15,
      "stratified": true,
      "quality_threshold": 0.8
    }
  },
  
  "evaluation": {
    "metrics": ["perplexity", "bleu", "rouge", "rag_accuracy"],
    "benchmarks": ["hellaswag", "mmlu"],
    "custom_evaluations": ["domain_qa", "instruction_following"],
    "evaluation_frequency": "every_epoch"
  },
  
  "monitoring": {
    "wandb_project": "llamafarm_production",
    "logging_steps": 10,
    "eval_steps": 100,
    "save_steps": 200,
    "alerting": {
      "loss_threshold": 0.1,
      "quality_degradation": 0.05,
      "training_time_limit": "24h"
    }
  },
  
  "deployment": {
    "target": "kubernetes",
    "replicas": 2,
    "resources": {
      "gpu": "1x_a100_40gb",
      "memory": "64Gi",
      "cpu": "8"
    },
    "auto_scaling": {
      "enabled": true,
      "min_replicas": 1,
      "max_replicas": 5,
      "cpu_threshold": 70,
      "memory_threshold": 80
    },
    "health_checks": {
      "liveness_probe": "/health",
      "readiness_probe": "/ready",
      "startup_probe": "/startup"
    }
  },
  
  "backup_and_recovery": {
    "checkpoint_frequency": "every_100_steps",
    "backup_location": "s3://llamafarm-models/backups/",
    "retention_policy": "30_days",
    "disaster_recovery": {
      "cross_region_backup": true,
      "recovery_time_objective": "1h"
    }
  }
}
```

## 🎯 Domain-Specific Examples

### **Medical RAG Enhancement** (`domain_specific/medical_rag.json`)
Specialized configuration for medical domain:

```json
{
  "name": "Medical RAG Enhancement Model",
  "domain": "healthcare",
  "compliance": ["HIPAA", "GDPR"],
  
  "model": {
    "base_model": "meta-llama/Llama-2-7b-hf",
    "method": "lora"
  },
  
  "fine_tuning": {
    "method_config": {
      "r": 32,
      "lora_alpha": 64,
      "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
    },
    "training": {
      "epochs": 4,
      "batch_size": 6,
      "learning_rate": 1.5e-4
    }
  },
  
  "dataset": {
    "source": {
      "type": "domain_corpus",
      "domain": "medical",
      "sources": [
        {
          "type": "rag_documents",
          "path": "./rag/data/medical/",
          "filters": {
            "document_types": ["clinical_notes", "research_papers", "guidelines"],
            "min_quality_score": 0.9
          }
        },
        {
          "type": "synthetic_medical_qa",
          "generation_model": "gpt-4",
          "medical_specialties": ["cardiology", "neurology", "oncology"],
          "complexity_levels": ["resident", "attending", "specialist"]
        }
      ]
    },
    "preprocessing": {
      "medical_entity_preservation": true,
      "terminology_normalization": true,
      "pii_removal": {
        "enabled": true,
        "entities": ["PATIENT_ID", "MRN", "DOB", "SSN"]
      }
    },
    "quality_control": {
      "medical_accuracy_check": true,
      "clinical_coherence": true,
      "ethical_review": true
    }
  },
  
  "evaluation": {
    "medical_benchmarks": [
      "MedQA",
      "PubMedQA", 
      "BioASQ",
      "MMLU-Medical"
    ],
    "custom_medical_eval": {
      "clinical_reasoning": true,
      "diagnostic_accuracy": true,
      "treatment_recommendations": true
    }
  },
  
  "deployment": {
    "security": {
      "encryption_at_rest": true,
      "encryption_in_transit": true,
      "access_controls": ["role_based", "attribute_based"],
      "audit_logging": true
    },
    "compliance_features": {
      "hipaa_logging": true,
      "data_retention_policy": "7_years",
      "patient_consent_tracking": true
    }
  }
}
```

### **Legal Document Analysis** (`domain_specific/legal_specialist.json`)
Configuration for legal domain specialization:

```json
{
  "name": "Legal Document Specialist",
  "domain": "legal",
  
  "model": {
    "base_model": "meta-llama/Llama-2-13b-hf",
    "method": "lora"
  },
  
  "dataset": {
    "source": {
      "type": "legal_corpus",
      "sources": [
        {
          "type": "case_law",
          "jurisdictions": ["federal", "state"],
          "areas_of_law": ["contract", "tort", "constitutional", "criminal"]
        },
        {
          "type": "legal_documents",
          "document_types": ["contracts", "briefs", "opinions", "statutes"]
        }
      ]
    },
    "preprocessing": {
      "citation_preservation": true,
      "legal_entity_recognition": true,
      "jurisdiction_tagging": true
    }
  },
  
  "fine_tuning": {
    "method_config": {
      "r": 48,
      "lora_alpha": 96,
      "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj"]
    },
    "legal_specific_training": {
      "citation_format_training": true,
      "legal_reasoning_emphasis": true,
      "precedent_analysis": true
    }
  },
  
  "evaluation": {
    "legal_benchmarks": ["LegalBench", "LexGLUE"],
    "custom_legal_eval": {
      "citation_accuracy": true,
      "legal_reasoning": true,
      "contract_analysis": true,
      "case_law_application": true
    }
  }
}
```

## 🔄 RAG Integration Examples

### **RAG Enhancement Pipeline** (`integration/rag_enhancement.json`)
Fine-tuning specifically for RAG system improvement:

```json
{
  "name": "RAG Enhancement Fine-tuning",
  "purpose": "Improve model performance within RAG pipeline",
  
  "integration": {
    "rag_system": {
      "retrieval_strategy": "hybrid_universal",
      "vector_store": "ChromaStore",
      "embedding_model": "nomic-embed-text"
    },
    "enhancement_targets": [
      "context_utilization",
      "retrieval_quality_assessment",
      "response_grounding",
      "citation_generation"
    ]
  },
  
  "dataset": {
    "source": {
      "type": "rag_optimized",
      "generation_strategy": "context_aware_qa",
      "context_sources": [
        {
          "type": "existing_rag_corpus",
          "path": "./rag/data/chroma_db/",
          "include_retrieval_metadata": true
        }
      ]
    },
    "rag_specific_processing": {
      "context_truncation_strategy": "relevance_based",
      "citation_training": true,
      "retrieval_quality_labels": true,
      "context_utilization_training": true
    }
  },
  
  "fine_tuning": {
    "rag_optimized_training": {
      "context_attention_bias": 1.2,
      "citation_loss_weight": 0.3,
      "retrieval_quality_loss": 0.2
    },
    "method_config": {
      "r": 24,
      "lora_alpha": 48,
      "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
    }
  },
  
  "evaluation": {
    "rag_specific_metrics": [
      "context_utilization_score",
      "citation_accuracy",
      "retrieval_relevance_improvement",
      "response_groundedness"
    ],
    "integration_testing": {
      "end_to_end_rag_pipeline": true,
      "retrieval_strategy_compatibility": ["basic_similarity", "metadata_filtered", "reranked"],
      "performance_benchmarks": {
        "latency_impact": "<10%",
        "quality_improvement": ">15%"  
      }
    }
  }
}
```

## 📊 Method Comparison Examples

### **LoRA vs QLoRA Comparison** (`method_comparison/lora_vs_qlora.json`)
Side-by-side comparison configuration:

```json
{
  "comparison_study": {
    "name": "LoRA vs QLoRA Performance Study",
    "variants": [
      {
        "name": "lora_baseline",
        "method": "lora",
        "config": {
          "r": 16,
          "lora_alpha": 32,
          "target_modules": ["q_proj", "v_proj"]
        }
      },
      {
        "name": "qlora_memory_efficient",
        "method": "qlora",
        "config": {
          "load_in_4bit": true,
          "r": 32,
          "lora_alpha": 64,
          "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
        }
      }
    ],
    "comparison_metrics": [
      "training_time",
      "memory_usage",
      "final_performance",
      "inference_speed",
      "model_size"
    ],
    "shared_config": {
      "base_model": "meta-llama/Llama-2-7b-hf",
      "dataset": "./datasets/comparison_dataset.json",
      "training": {
        "epochs": 3,
        "batch_size": 4,
        "learning_rate": 2e-4
      }
    }
  }
}
```

## 🌐 Distributed Training Examples

### **Multi-GPU Training** (`distributed/multi_gpu_lora.json`)
Configuration for distributed training:

```json
{
  "name": "Multi-GPU LoRA Training",
  "distributed_training": {
    "strategy": "deepspeed_zero2",
    "num_gpus": 4,
    "gradient_synchronization": "allreduce"
  },
  
  "model": {
    "base_model": "meta-llama/Llama-2-13b-hf",
    "method": "lora"
  },
  
  "fine_tuning": {
    "method_config": {
      "r": 32,
      "lora_alpha": 64,
      "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
    },
    "distributed_config": {
      "zero_optimization": {
        "stage": 2,
        "offload_optimizer": false,
        "overlap_comm": true,
        "contiguous_gradients": true
      },
      "gradient_accumulation_steps": 8,
      "effective_batch_size": 128
    }
  },
  
  "hardware": {
    "nodes": 1,
    "gpus_per_node": 4,
    "cpu_cores_per_node": 32,
    "memory_per_node": "256GB",
    "interconnect": "infiniband"
  },
  
  "monitoring": {
    "distributed_metrics": [
      "communication_overhead",
      "load_balancing",
      "synchronization_time",
      "per_gpu_utilization"
    ]
  }
}
```

## 🚀 Deployment Configurations

### **Kubernetes Deployment** (`deployment/kubernetes_production.json`)
Production deployment configuration:

```json
{
  "deployment": {
    "name": "llamafarm-custom-model-production",
    "platform": "kubernetes",
    
    "model_config": {
      "model_path": "./trained_models/production_lora_v2",
      "adapter_path": "./adapters/domain_specialist_v1",
      "hot_swap_enabled": true
    },
    
    "kubernetes": {
      "namespace": "llamafarm-models",
      "deployment": {
        "replicas": 3,
        "strategy": {
          "type": "RollingUpdate",
          "max_unavailable": 1,
          "max_surge": 1
        }
      },
      "service": {
        "type": "LoadBalancer",
        "ports": [
          {"name": "http", "port": 80, "target_port": 8000},
          {"name": "grpc", "port": 9000, "target_port": 9000}
        ]
      },
      "ingress": {
        "enabled": true,
        "tls_enabled": true,
        "hosts": ["models.llamafarm.internal"]
      }
    },
    
    "resources": {
      "requests": {
        "cpu": "4",
        "memory": "32Gi",
        "nvidia.com/gpu": "1"
      },
      "limits": {
        "cpu": "8", 
        "memory": "64Gi",
        "nvidia.com/gpu": "1"
      }
    },
    
    "auto_scaling": {
      "enabled": true,
      "min_replicas": 2,
      "max_replicas": 10,
      "target_cpu_utilization": 70,
      "target_memory_utilization": 80,
      "custom_metrics": [
        {
          "name": "model_queue_length",
          "target_value": 50
        }
      ]
    },
    
    "monitoring": {
      "prometheus": {
        "enabled": true,
        "scrape_interval": "15s",
        "metrics_path": "/metrics"
      },
      "logging": {
        "level": "INFO",
        "format": "json",
        "output": "stdout"
      },
      "health_checks": {
        "liveness_probe": {
          "http_get": {"path": "/health", "port": 8000},
          "initial_delay_seconds": 30,
          "period_seconds": 10
        },
        "readiness_probe": {
          "http_get": {"path": "/ready", "port": 8000},
          "initial_delay_seconds": 5,
          "period_seconds": 5
        }
      }
    }
  }
}
```

## 🔧 Configuration Best Practices

### **Configuration Validation**
All configurations include schema validation:

```json
{
  "config_metadata": {
    "schema_version": "1.0.0",
    "validation": {
      "required_fields": ["name", "model", "fine_tuning"],
      "field_types": {
        "name": "string",
        "model.base_model": "string",
        "fine_tuning.method_config.r": "integer"
      },
      "constraints": {
        "fine_tuning.method_config.r": {"min": 1, "max": 256},
        "fine_tuning.training.learning_rate": {"min": 1e-6, "max": 1e-2}
      }
    }
  }
}
```

### **Environment-Specific Configurations**
Support for different environments:

```json
{
  "environments": {
    "development": {
      "model.base_model": "microsoft/DialoGPT-small",
      "fine_tuning.training.epochs": 1,
      "monitoring.wandb_project": "llamafarm_dev"
    },
    "staging": {
      "fine_tuning.training.epochs": 2,
      "evaluation.benchmarks": ["hellaswag"],
      "monitoring.wandb_project": "llamafarm_staging"
    },
    "production": {
      "fine_tuning.training.epochs": 5,
      "evaluation.benchmarks": ["hellaswag", "mmlu", "truthfulqa"],
      "monitoring.wandb_project": "llamafarm_production",
      "backup_and_recovery.enabled": true
    }
  }
}
```

## 📋 Configuration Templates

### **Template Structure**
```json
{
  "template_metadata": {
    "name": "Template Name",
    "description": "Template description",
    "use_case": "Specific use case",
    "difficulty": "beginner|intermediate|advanced",
    "estimated_time": "2h",
    "gpu_requirements": "1x RTX 4090 (24GB)"
  },
  
  "variables": {
    "BASE_MODEL": "meta-llama/Llama-2-7b-hf",
    "DATASET_PATH": "./datasets/my_dataset.json",
    "OUTPUT_PATH": "./trained_models/my_model",
    "LEARNING_RATE": 2e-4
  },
  
  "configuration": {
    // Template configuration using variables
  }
}
```

## 🚀 Quick Commands Reference

### **Using Configuration Examples**
```bash
# Use example configuration directly
llamafarm models train --config ./config_examples/basic/lora_quickstart.json

# Override specific parameters
llamafarm models train \
  --config ./config_examples/basic/lora_quickstart.json \
  --override base_model=meta-llama/Llama-2-13b-hf \
  --override training.epochs=5

# Validate configuration before training
llamafarm models validate-config \
  --config ./config_examples/production/production_lora.json

# Generate configuration from template
llamafarm models generate-config \
  --template basic_lora \
  --base-model llama-2-7b \
  --dataset ./my_dataset.json \
  --output ./my_config.json

# Compare configurations
llamafarm models compare-configs \
  --config1 ./config_examples/method_comparison/lora_vs_qlora.json \
  --metrics training_time,memory_usage,performance
```

These configuration examples provide a comprehensive starting point for various fine-tuning scenarios while demonstrating the flexibility and power of the LlamaFarm Models system's config-driven approach.