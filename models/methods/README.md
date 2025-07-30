# Fine-Tuning Methods Directory

## 🎯 Overview

The methods directory contains implementations and configurations for different fine-tuning approaches supported by the LlamaFarm Models system. Each method provides a config-driven approach to model customization with CLI integration and production-ready capabilities.

## 📁 Directory Structure

```
methods/
├── lora/              # Low-Rank Adaptation (Most Popular)
├── qlora/             # Quantized LoRA (Memory Efficient)
├── full_finetune/     # Traditional Full Fine-tuning
├── adapters/          # Adapter-based Methods
└── prefix_tuning/     # Prefix Tuning Implementation
```

## 🏗️ Method Architecture

Each fine-tuning method follows a consistent structure:

### **Standard Method Structure**
```
method_name/
├── README.md          # Method-specific documentation
├── config_schema.json # JSON schema for configuration validation
├── defaults.json      # Default configuration values
├── examples/          # Example configurations
│   ├── basic.json     # Simple use case
│   ├── advanced.json  # Complex configuration
│   └── production.json # Production-ready config
├── implementation.py  # Core method implementation (future)
└── tests/            # Method-specific tests
```

## 🔧 Method Selection Guide

### **Quick Selection Matrix**

| Criteria | LoRA | QLoRA | Full FT | Adapters | Prefix |
|----------|------|-------|---------|----------|--------|
| **GPU Memory (7B model)** | 16GB | 8GB | 56GB | 20GB | 12GB |
| **Training Speed** | Fast | Medium | Slow | Fast | Very Fast |
| **Quality vs Full FT** | 95-98% | 93-96% | 100% | 94-97% | 85-92% |
| **Multiple Tasks** | ✅ | ✅ | ❌ | ✅ | ✅ |
| **Hot Swapping** | ✅ | ✅ | ❌ | ✅ | ✅ |
| **Production Ready** | ✅ | ✅ | ✅ | ✅ | ⚠️ |

### **Use Case Recommendations**

| Use Case | Primary Method | Secondary Option | Configuration Notes |
|----------|----------------|------------------|-------------------|
| **General RAG Enhancement** | LoRA | QLoRA | r=16, standard config |
| **Domain Specialization** | LoRA | Full Fine-tune | r=32-64, all modules |
| **Memory Constrained** | QLoRA | LoRA | 4-bit quantization |
| **Multiple Domains** | Adapters | LoRA | Hot-swappable setup |
| **Rapid Prototyping** | Prefix Tuning | LoRA | Minimal parameters |
| **Maximum Quality** | Full Fine-tune | LoRA (r=64+) | Large dataset required |

## 📊 Configuration Schema Standards

### **Base Configuration Schema**
All methods inherit from a base configuration schema:

```json
{
  "method_config": {
    "method_name": "string",
    "base_model": "string",
    "version": "string",
    
    "training": {
      "epochs": "integer",
      "batch_size": "integer", 
      "learning_rate": "float",
      "warmup_steps": "integer",
      "gradient_accumulation_steps": "integer",
      "max_grad_norm": "float",
      "weight_decay": "float",
      "lr_scheduler": "string"
    },
    
    "hardware": {
      "device": "string",
      "mixed_precision": "boolean",
      "gradient_checkpointing": "boolean",
      "dataloader_num_workers": "integer"
    },
    
    "monitoring": {
      "logging_steps": "integer",
      "eval_steps": "integer",
      "save_steps": "integer",
      "wandb_project": "string",
      "tensorboard_log_dir": "string"
    },
    
    "method_specific": {
      // Method-specific parameters defined in each method
    }
  }
}
```

### **Method-Specific Extensions**

Each method extends the base schema with specific parameters:

#### **LoRA Extension**
```json
{
  "method_specific": {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "target_modules": ["q_proj", "v_proj"],
    "bias": "none",
    "modules_to_save": null,
    "fan_in_fan_out": false
  }
}
```

#### **QLoRA Extension**
```json
{
  "method_specific": {
    "load_in_4bit": true,
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": true,
    // Plus all LoRA parameters
    "r": 64,
    "lora_alpha": 128
  }
}
```

## 🚀 Implementation Standards

### **CLI Integration Pattern**
Each method supports consistent CLI operations:

```bash
# Training with method-specific config
llamafarm models train \
  --method lora \
  --config methods/lora/examples/basic.json \
  --dataset ./datasets/my_dataset.json

# Quick start with defaults
llamafarm models quick-train \
  --method qlora \
  --base-model llama-2-7b \
  --dataset ./datasets/my_dataset.json

# Method-specific validation
llamafarm models validate-config \
  --method lora \
  --config ./my_lora_config.json
```

### **Configuration Validation**
Each method includes JSON schema validation:

```python
# Example validation implementation
def validate_lora_config(config):
    """Validate LoRA configuration parameters"""
    
    # Check required parameters
    required_params = ['r', 'lora_alpha', 'target_modules']
    for param in required_params:
        if param not in config['method_specific']:
            raise ValueError(f"Missing required parameter: {param}")
    
    # Validate parameter ranges
    r = config['method_specific']['r']
    if not (1 <= r <= 256):
        raise ValueError(f"LoRA rank 'r' must be between 1 and 256, got {r}")
    
    # Validate relationships
    lora_alpha = config['method_specific']['lora_alpha']
    if lora_alpha < r:
        warnings.warn(f"lora_alpha ({lora_alpha}) is less than r ({r}), this may reduce training effectiveness")
    
    return True
```

### **Progress Monitoring**
Consistent monitoring across all methods:

```python
class TrainingMonitor:
    """Standard training monitoring for all methods"""
    
    def __init__(self, method_name, config):
        self.method_name = method_name
        self.config = config
        self.metrics = []
    
    def log_step(self, step, loss, lr, method_specific_metrics=None):
        """Log training step with method-specific metrics"""
        log_entry = {
            "step": step,
            "loss": loss,
            "learning_rate": lr,
            "method": self.method_name,
            "timestamp": datetime.now().isoformat()
        }
        
        if method_specific_metrics:
            log_entry.update(method_specific_metrics)
        
        self.metrics.append(log_entry)
        
        # Optional external logging
        if self.config.get("monitoring", {}).get("wandb_project"):
            self._log_to_wandb(log_entry)
    
    def generate_report(self):
        """Generate training summary report"""
        return {
            "method": self.method_name,
            "total_steps": len(self.metrics),
            "final_loss": self.metrics[-1]["loss"] if self.metrics else None,
            "training_time": self._calculate_training_time(),
            "convergence_analysis": self._analyze_convergence()
        }
```

## 📈 Performance Benchmarking

### **Standard Benchmarks**
All methods are evaluated against consistent benchmarks:

1. **Training Efficiency**: Time to convergence, memory usage
2. **Model Quality**: Perplexity, downstream task performance
3. **Resource Utilization**: GPU utilization, memory efficiency
4. **Deployment Metrics**: Inference latency, model size

### **Benchmark Configuration**
```json
{
  "benchmarks": {
    "training_efficiency": {
      "max_training_time": "24h",
      "memory_budget": "24GB",
      "convergence_threshold": 0.01
    },
    "quality_metrics": {
      "perplexity_threshold": 10.0,
      "downstream_tasks": ["hellaswag", "mmlu"],
      "minimum_accuracy": 0.7
    },
    "resource_metrics": {
      "max_gpu_memory": "24GB",
      "min_gpu_utilization": 0.8,
      "max_inference_latency": "200ms"
    }
  }
}
```

## 🔄 Method Development Guidelines

### **Adding New Methods**
To add a new fine-tuning method:

1. **Create Method Directory**: Follow standard structure
2. **Define Configuration Schema**: Extend base schema with method-specific parameters
3. **Implement Core Logic**: Method-specific training and inference code
4. **Add CLI Integration**: Support standard CLI operations
5. **Create Examples**: Basic, advanced, and production configurations
6. **Write Tests**: Comprehensive test coverage
7. **Update Documentation**: Method-specific documentation and integration guide

### **Method Quality Standards**
- **Configuration Validation**: JSON schema validation for all parameters
- **Error Handling**: Graceful handling of invalid configurations and training failures
- **Monitoring Integration**: Standard metrics and logging
- **Documentation**: Clear documentation with examples
- **Testing**: Unit and integration tests
- **Performance**: Benchmarking against established baselines

### **Backward Compatibility**
- **Config Versioning**: Support for configuration migration
- **API Stability**: Maintain stable interfaces for CLI and programmatic access
- **Deprecation Process**: Clear deprecation timeline for method changes

## 🔧 Development Roadmap

### **Phase 1: Foundation Methods**
- [x] **Directory Structure**: Standard method organization
- [ ] **LoRA Implementation**: Most popular method first
- [ ] **QLoRA Implementation**: Memory-efficient alternative
- [ ] **Base CLI Integration**: Standard commands for all methods

### **Phase 2: Advanced Methods**
- [ ] **Adapter Implementation**: Multi-task support
- [ ] **Full Fine-tuning**: Traditional approach for comparison
- [ ] **Prefix Tuning**: Lightweight alternative
- [ ] **Hot-Swapping Support**: Runtime method switching

### **Phase 3: Optimization**
- [ ] **Distributed Training**: Multi-GPU and multi-node support
- [ ] **Mixed Precision**: Automatic mixed precision training
- [ ] **Gradient Checkpointing**: Memory optimization
- [ ] **Advanced Monitoring**: Comprehensive metrics and alerting

### **Phase 4: Research Methods**
- [ ] **AdaLoRA**: Adaptive rank allocation
- [ ] **IA³**: Infused adapter implementation
- [ ] **Multi-Method Fusion**: Combining different approaches
- [ ] **AutoML Integration**: Automated method selection

## 📊 Configuration Examples by Use Case

### **Quick Start - General Purpose**
```json
{
  "method": "lora",
  "base_model": "meta-llama/Llama-2-7b-hf",
  "method_specific": {
    "r": 8,
    "lora_alpha": 16,
    "target_modules": ["q_proj", "v_proj"]
  },
  "training": {
    "epochs": 3,
    "batch_size": 4,
    "learning_rate": 2e-4
  }
}
```

### **Memory Constrained - Large Model**
```json
{
  "method": "qlora", 
  "base_model": "meta-llama/Llama-2-70b-hf",
  "method_specific": {
    "load_in_4bit": true,
    "r": 64,
    "lora_alpha": 128,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
  },
  "hardware": {
    "gradient_checkpointing": true,
    "mixed_precision": true
  }
}
```

### **Production - High Quality**
```json
{
  "method": "lora",
  "base_model": "meta-llama/Llama-2-13b-hf",
  "method_specific": {
    "r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  },
  "training": {
    "epochs": 5,
    "batch_size": 8,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1e-4,
    "warmup_steps": 500
  },
  "monitoring": {
    "wandb_project": "llamafarm_production",
    "eval_steps": 100,
    "logging_steps": 10
  }
}
```

## 🧪 Testing Strategy

### **Method Testing Framework**
```python
class MethodTestSuite:
    """Standard test suite for all fine-tuning methods"""
    
    def test_config_validation(self):
        """Test configuration validation"""
        pass
    
    def test_training_smoke(self):
        """Test basic training functionality"""
        pass
    
    def test_memory_usage(self):
        """Test memory consumption stays within bounds"""
        pass
    
    def test_convergence(self):
        """Test training converges on toy dataset"""
        pass
    
    def test_inference(self):
        """Test inference with trained model"""
        pass
    
    def test_serialization(self):
        """Test model saving and loading"""
        pass
```

### **Integration Testing**
- **RAG Pipeline Integration**: Verify fine-tuned models work with existing RAG system
- **CLI Integration**: Test all CLI commands work correctly
- **Configuration Migration**: Test config version upgrades
- **Performance Regression**: Ensure new changes don't degrade performance

The methods directory provides a solid foundation for implementing and managing various fine-tuning approaches while maintaining consistency, quality, and ease of use across the LlamaFarm ecosystem.