# Model Registry System

## 🎯 Overview

The Model Registry provides comprehensive version control, metadata management, and lifecycle management for fine-tuned models and adapters. Following the LlamaFarm philosophy of configuration-driven development, the registry enables sophisticated model management while maintaining simplicity and production readiness.

## 🏗️ Registry Architecture

### **Core Components**
```
registry/
├── README.md           # This documentation
├── schemas/           # Registry data schemas
│   ├── model.schema.json      # Model metadata schema
│   ├── adapter.schema.json    # Adapter metadata schema
│   └── deployment.schema.json # Deployment record schema
├── storage/           # Storage backend configurations
│   ├── local.json     # Local filesystem storage
│   ├── s3.json        # AWS S3 storage
│   ├── gcs.json       # Google Cloud Storage
│   └── azure.json     # Azure Blob Storage
├── versioning/        # Version control strategies
│   ├── semantic.json  # Semantic versioning
│   ├── timestamp.json # Timestamp-based versioning
│   └── hash.json      # Content-based hashing
└── access_control/    # Permission and access management
    ├── rbac.json      # Role-based access control
    ├── policies.json  # Access policies
    └── audit.json     # Audit logging configuration
```

## 📊 Registry Data Model

### **Model Record Schema**
```json
{
  "model_record": {
    "metadata": {
      "id": "rag_enhancement_v1.2.0",
      "name": "RAG Enhancement Model",
      "version": "1.2.0",
      "description": "Fine-tuned model for improved RAG performance",
      "created_at": "2024-01-15T10:30:00Z",
      "updated_at": "2024-01-15T10:30:00Z",
      "created_by": "user@llamafarm.com",
      "tags": ["rag", "enhancement", "production"]
    },
    
    "model_info": {
      "base_model": "meta-llama/Llama-2-7b-hf",
      "fine_tuning_method": "lora",
      "parameters": {
        "total_params": "6,738,415,616",
        "trainable_params": "4,194,304",
        "trainable_percentage": "0.062%"
      },
      "architecture": {
        "layers": 32,
        "hidden_size": 4096,
        "attention_heads": 32,
        "vocab_size": 32000
      }
    },
    
    "training_info": {
      "dataset": {
        "name": "rag_qa_dataset_v2",
        "size": 25000,
        "source": "rag_documents_generated"
      },
      "method_config": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "v_proj"]
      },
      "training_config": {
        "epochs": 3,
        "batch_size": 4,
        "learning_rate": 2e-4,
        "total_steps": 18750
      },
      "training_time": "4h 32m",
      "gpu_hours": 18.1
    },
    
    "performance_metrics": {
      "training_metrics": {
        "final_loss": 0.847,
        "perplexity": 2.33,
        "convergence_step": 15000
      },
      "evaluation_metrics": {
        "hellaswag_accuracy": 0.721,
        "mmlu_accuracy": 0.645,
        "rag_accuracy": 0.843,
        "bleu_score": 0.267,
        "rouge_l": 0.445
      },
      "benchmark_comparison": {
        "vs_base_model": {
          "rag_accuracy_improvement": "+12.3%",
          "general_accuracy_change": "+2.1%"
        },
        "vs_previous_version": {
          "rag_accuracy_improvement": "+3.4%",
          "training_efficiency": "+8.7%"
        }
      }
    },
    
    "storage": {
      "location": "s3://llamafarm-models/rag_enhancement_v1.2.0/",
      "size_mb": 24.7,
      "checksum": "sha256:a1b2c3d4e5f6...",
      "compression": "gzip",
      "encryption": "AES-256"
    },
    
    "deployment": {
      "status": "production",
      "environments": {
        "development": {
          "deployed_at": "2024-01-15T11:00:00Z",
          "endpoint": "https://dev-models.llamafarm.internal/rag_enhancement",
          "replicas": 1
        },
        "production": {
          "deployed_at": "2024-01-16T09:00:00Z", 
          "endpoint": "https://models.llamafarm.com/rag_enhancement",
          "replicas": 3,
          "traffic_percentage": 100
        }
      }
    },
    
    "quality_gates": {
      "passed": true,
      "checks": [
        {"name": "performance_threshold", "status": "passed", "score": 0.843},
        {"name": "safety_evaluation", "status": "passed", "score": 0.95},
        {"name": "bias_assessment", "status": "passed", "score": 0.88},
        {"name": "integration_tests", "status": "passed", "coverage": 0.92}
      ]
    },
    
    "lineage": {
      "parent_model": "rag_enhancement_v1.1.0",
      "training_config_hash": "config_abc123",
      "dataset_version": "rag_qa_v2.1",
      "code_commit": "commit_def456"
    }
  }
}
```

### **Adapter Record Schema**
```json
{
  "adapter_record": {
    "metadata": {
      "id": "medical_specialist_v2.0.0",
      "name": "Medical Domain Specialist",
      "version": "2.0.0",
      "type": "lora_adapter",
      "base_model_id": "rag_enhancement_v1.2.0"
    },
    
    "adapter_info": {
      "method": "lora",
      "parameters": {
        "r": 32,
        "lora_alpha": 64,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "trainable_params": "8,388,608"
      },
      "specialization": {
        "domain": "medical",
        "use_case": "clinical_qa",
        "expertise_level": "specialist"
      }
    },
    
    "training_info": {
      "dataset": "medical_qa_curated_v3",
      "training_samples": 15000,
      "validation_samples": 2000,
      "medical_specialties": ["cardiology", "neurology", "oncology"]
    },
    
    "performance_metrics": {
      "medical_qa_accuracy": 0.891,
      "clinical_reasoning": 0.834,
      "safety_score": 0.976,
      "ethical_alignment": 0.945
    },
    
    "compatibility": {
      "base_models": ["rag_enhancement_v1.2.0", "rag_enhancement_v1.1.0"],
      "frameworks": ["transformers", "peft", "vllm"],
      "inference_engines": ["llamafarm_native", "huggingface", "tgi"]
    },
    
    "hot_swap": {
      "enabled": true,
      "swap_time_ms": 150,
      "memory_overhead_mb": 24.7,
      "compatibility_check": "automatic"
    }
  }
}
```

## 🔧 Registry Operations

### **Model Management Commands**
```bash
# Register new model
llamafarm models register \
  --path ./trained_models/my_model \
  --name "my_custom_model" \
  --version "1.0.0" \
  --base-model "llama-2-7b" \
  --method "lora" \
  --description "Custom model for specific use case" \
  --tags "production,rag,custom"

# List available models
llamafarm models list \
  --filter "tags:production AND method:lora" \
  --sort "created_at:desc" \
  --limit 10

# Get model information
llamafarm models info my_custom_model:1.0.0

# Download model
llamafarm models pull my_custom_model:1.0.0 --output ./downloaded_models/

# Update model metadata
llamafarm models update my_custom_model:1.0.0 \
  --description "Updated description" \
  --add-tag "validated" \
  --set-status "production"

# Delete model version
llamafarm models delete my_custom_model:1.0.0 --confirm

# Promote model between environments
llamafarm models promote my_custom_model:1.0.0 \
  --from staging \
  --to production \
  --traffic-percentage 20
```

### **Adapter Management Commands**
```bash
# Register adapter
llamafarm adapters register \
  --path ./adapters/medical_specialist \
  --name "medical_specialist" \
  --version "2.0.0" \
  --base-model-id "rag_enhancement_v1.2.0" \
  --domain "medical" \
  --specialization "clinical_qa"

# List adapters for specific base model
llamafarm adapters list \
  --base-model "rag_enhancement_v1.2.0" \
  --domain "medical"

# Hot-swap adapter in production
llamafarm adapters swap \
  --current "general_qa_v1.0.0" \
  --target "medical_specialist_v2.0.0" \
  --environment "production" \
  --validate-compatibility

# Create adapter bundle for deployment
llamafarm adapters bundle \
  --adapters "medical_v2.0.0,legal_v1.5.0,technical_v1.2.0" \
  --output ./adapter_bundle.tar.gz
```

## 📊 Version Management

### **Semantic Versioning Strategy**
```json
{
  "versioning": {
    "strategy": "semantic",
    "format": "MAJOR.MINOR.PATCH",
    "rules": {
      "major_increment": [
        "breaking_changes_to_api",
        "significant_architecture_changes",
        "base_model_change"
      ],
      "minor_increment": [
        "new_capabilities",
        "performance_improvements",
        "new_adapters"
      ],
      "patch_increment": [
        "bug_fixes",
        "minor_optimizations",
        "metadata_updates"
      ]
    },
    "auto_increment": {
      "enabled": true,
      "based_on": "training_config_changes"
    }
  }
}
```

### **Version Lifecycle**
```json
{
  "version_lifecycle": {
    "states": [
      {
        "name": "development",
        "description": "In development, not ready for production",
        "allowed_operations": ["update", "delete", "test"]
      },
      {
        "name": "testing",
        "description": "Ready for testing and validation",
        "allowed_operations": ["test", "validate", "promote"]
      },
      {
        "name": "staging",
        "description": "Deployed to staging environment",
        "allowed_operations": ["promote", "rollback", "clone"]
      },
      {
        "name": "production",
        "description": "Live in production",
        "allowed_operations": ["rollback", "clone", "archive"],
        "immutable": true
      },
      {
        "name": "deprecated",
        "description": "No longer recommended for use",
        "allowed_operations": ["archive", "delete"]
      },
      {
        "name": "archived",
        "description": "Archived for historical reference",
        "allowed_operations": ["restore", "delete"]
      }
    ],
    "transitions": {
      "development": ["testing", "deprecated"],
      "testing": ["staging", "development", "deprecated"],
      "staging": ["production", "testing", "deprecated"],
      "production": ["deprecated", "archived"],
      "deprecated": ["archived"],
      "archived": ["development"]
    }
  }
}
```

## 🔍 Model Discovery & Search

### **Search Capabilities**
```python
class ModelSearch:
    """Advanced model search and filtering"""
    
    def search_models(self, query):
        """
        Search models using various criteria
        
        Examples:
        - "name:medical AND version:>=2.0.0"
        - "performance.rag_accuracy:>0.85"
        - "tags:production AND method:lora"
        - "created_at:last_7_days AND base_model:llama-2"
        """
        pass
    
    def recommend_models(self, use_case, performance_requirements):
        """Recommend models based on use case and requirements"""
        pass
    
    def find_compatible_adapters(self, model_id):
        """Find adapters compatible with specific model"""
        pass
```

### **Search API Examples**
```bash
# Text-based search
llamafarm models search "medical domain specialist with rag accuracy > 0.85"

# Structured search
llamafarm models search \
  --filter "domain:medical AND performance.rag_accuracy:>0.85" \
  --sort "performance.rag_accuracy:desc"

# Performance-based recommendations
llamafarm models recommend \
  --use-case "medical_qa" \
  --min-accuracy 0.85 \
  --max-latency 200ms \
  --max-memory 16GB

# Find similar models
llamafarm models similar my_model:1.0.0 \
  --similarity-metric "performance" \
  --limit 5
```

## 🔒 Access Control & Security

### **Role-Based Access Control**
```json
{
  "rbac": {
    "roles": [
      {
        "name": "model_developer",
        "permissions": [
          "models:create",
          "models:read",
          "models:update",
          "models:delete_own",
          "adapters:create",
          "adapters:read"
        ]
      },
      {
        "name": "ml_engineer",
        "permissions": [
          "models:read",
          "models:deploy",
          "models:promote",
          "adapters:read",
          "adapters:deploy",
          "registry:search"
        ]
      },
      {
        "name": "model_admin",
        "permissions": [
          "models:*",
          "adapters:*",
          "registry:*",
          "access_control:manage"
        ]
      },
      {
        "name": "viewer",
        "permissions": [
          "models:read",
          "adapters:read",
          "registry:search"
        ]
      }
    ],
    "user_assignments": {
      "user@example.com": ["model_developer"],
      "admin@example.com": ["model_admin"]
    }
  }
}
```

### **Audit Logging**
```json
{
  "audit_logging": {
    "enabled": true,
    "log_level": "INFO",
    "events": [
      "model_registered",
      "model_downloaded",
      "model_deployed",
      "model_deleted",
      "adapter_swapped",
      "permission_changed"
    ],
    "log_format": {
      "timestamp": "2024-01-15T10:30:00Z",
      "user": "user@example.com",
      "action": "model_deployed",
      "resource": "my_model:1.0.0",
      "environment": "production",
      "source_ip": "192.168.1.100",
      "user_agent": "llamafarm-cli/1.0.0"
    },
    "retention": {
      "duration": "7_years",
      "archive_after": "1_year",
      "compression": true
    }
  }
}
```

## 🗄️ Storage Backends

### **Local Storage Configuration**
```json
{
  "storage_backend": {
    "type": "local",
    "config": {
      "base_path": "/var/lib/llamafarm/registry",
      "structure": "hierarchical",
      "compression": {
        "enabled": true,
        "algorithm": "gzip",
        "level": 6
      },
      "backup": {
        "enabled": true,
        "frequency": "daily",
        "retention": "30_days",
        "location": "/backup/llamafarm/registry"
      }
    }
  }
}
```

### **S3 Storage Configuration**
```json
{
  "storage_backend": {
    "type": "s3",
    "config": {
      "bucket": "llamafarm-model-registry",
      "region": "us-west-2",
      "access_key_id": "${AWS_ACCESS_KEY_ID}",
      "secret_access_key": "${AWS_SECRET_ACCESS_KEY}",
      "prefix": "models/",
      "storage_class": "STANDARD_IA",
      "encryption": {
        "enabled": true,
        "kms_key_id": "arn:aws:kms:us-west-2:123456789012:key/12345678-1234-1234-1234-123456789012"
      },
      "versioning": {
        "enabled": true,
        "lifecycle_policy": {
          "delete_non_current_versions_after": "90_days",
          "transition_to_glacier_after": "365_days"
        }
      }
    }
  }
}
```

## 📈 Registry Analytics

### **Usage Analytics**
```json
{
  "analytics": {
    "tracking": {
      "model_downloads": true,
      "deployment_frequency": true,
      "performance_trends": true,
      "user_patterns": true
    },
    "reports": {
      "most_popular_models": {
        "period": "last_30_days",
        "metrics": ["downloads", "deployments", "ratings"]
      },
      "performance_trends": {
        "models": ["rag_enhancement_*", "medical_specialist_*"],
        "metrics": ["accuracy_improvement", "training_efficiency"]
      },
      "resource_utilization": {
        "storage_usage": "daily",
        "compute_costs": "monthly",
        "user_activity": "weekly"
      }
    },
    "alerts": {
      "storage_quota": "80%",
      "inactive_models": "90_days",
      "failed_deployments": "5_in_1_hour"
    }
  }
}
```

### **Model Lineage Tracking**
```python
class ModelLineage:
    """Track model development lineage and dependencies"""
    
    def track_training_lineage(self, model_id):
        """Track complete training lineage"""
        return {
            "base_model": "meta-llama/Llama-2-7b-hf",
            "parent_model": "rag_enhancement_v1.1.0",
            "training_dataset": "rag_qa_v2.1",
            "training_config": "config_hash_abc123",
            "code_version": "commit_def456",
            "environment": "training_cluster_gpu_8x_a100"
        }
    
    def track_deployment_lineage(self, deployment_id):
        """Track deployment lineage"""
        return {
            "model_version": "rag_enhancement_v1.2.0",
            "adapter_versions": ["medical_v2.0.0", "legal_v1.5.0"],
            "deployment_config": "deploy_config_xyz789",
            "infrastructure": "kubernetes_cluster_prod",
            "deployment_time": "2024-01-16T09:00:00Z"
        }
```

## 🔄 Registry Integration

### **CI/CD Integration**
```yaml
# Example GitHub Actions workflow
name: Model Registry Integration
on:
  push:
    paths: ['models/**']

jobs:
  register_model:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Train Model
        run: |
          llamafarm models train --config models/configs/ci_training.json
      
      - name: Evaluate Model
        run: |
          llamafarm models evaluate \
            --model ./trained_models/ci_model \
            --benchmarks hellaswag,mmlu
      
      - name: Register Model
        if: ${{ success() }}
        run: |
          llamafarm models register \
            --path ./trained_models/ci_model \
            --name "ci_model_$(date +%Y%m%d)" \
            --version "1.0.${GITHUB_RUN_NUMBER}" \
            --environment "staging" \
            --metadata "commit=${GITHUB_SHA}"
      
      - name: Deploy to Staging
        run: |
          llamafarm models deploy \
            --model "ci_model_$(date +%Y%m%d):1.0.${GITHUB_RUN_NUMBER}" \
            --environment "staging"
```

### **API Integration**
```python
# Python SDK integration
from llamafarm.registry import ModelRegistry

registry = ModelRegistry(config_path="./registry_config.json")

# Register model programmatically
model_id = registry.register_model(
    path="./trained_models/my_model",
    name="api_registered_model",
    version="1.0.0",
    metadata={
        "base_model": "llama-2-7b",
        "method": "lora",
        "use_case": "rag_enhancement"
    }
)

# Deploy model
deployment_id = registry.deploy_model(
    model_id=model_id,
    environment="production",
    config={
        "replicas": 3,
        "resources": {"gpu": 1, "memory": "32Gi"}
    }
)

# Monitor deployment
status = registry.get_deployment_status(deployment_id)
```

The Model Registry provides a comprehensive foundation for managing the complete lifecycle of fine-tuned models and adapters, enabling sophisticated model management while maintaining the LlamaFarm principles of simplicity, configurability, and production readiness.