# 🎯 LlamaFarm Fine-Tuning Demo Showcase

This directory contains **6 comprehensive fine-tuning demonstrations** showcasing different use cases, strategies, and capabilities of the LlamaFarm system. Each demo uses real data, actual training, and demonstrates measurable improvements.

## 🚀 Demo Overview

| Demo | Use Case | Dataset Size | Model | Method | Strategy | Duration |
|------|----------|--------------|-------|---------|----------|----------|
| **OpenAI Models** | Dynamic model switching | 15 examples | GPT-4o-mini/GPT-4o/GPT-4-turbo | Strategy Switching | `openai_model_switching` | 4-5 min |
| **Customer Support** | E-commerce support | 150 examples | DialoGPT-medium | LoRA | `customer_support_lora` | 3-5 min |
| **Medical Assistant** | Healthcare Q&A | 300+ examples | TinyLlama-Medical-1.1B | QLoRA | `medical_qlora_efficient` | 5-8 min |
| **Code Helper** | Programming assistance | 200 examples | Liquid-Llama-3-8B-Coding | LoRA | `python_coding_specialist` | 4-6 min |
| **Creative Writing** | Story & content generation | 50 examples | GPT2-medium | LoRA | `creative_lora_diverse` | 3-4 min |
| **Technical Q&A** | Engineering documentation | 30 examples | T5-base | QLoRA | `technical_qlora_large` | 2-3 min |

## 🎪 **End-to-End Showcase**

Run all 6 demos in sequence with educational commentary:

```bash
uv run python run_all_demos.py
```

This comprehensive showcase demonstrates:
- **Different strategies** for different use cases
- **Model selection rationale** (why TinyLlama-Medical for healthcare, Liquid-Llama for programming)
- **Method comparison** (LoRA vs QLoRA vs Full Fine-tuning)
- **Dataset considerations** (size, quality, domain-specific needs)
- **Hardware optimization** (memory usage, training time)
- **Real-world applications** and deployment scenarios

## 📋 Individual Demos

### 1. 🛒 **Customer Support Demo**
```bash
cd customer_support && uv run python run_demo.py
```

**Scenario**: E-commerce customer service assistant  
**Challenge**: Handle returns, shipping, product questions professionally  
**Strategy**: `customer_support_lora` - balanced efficiency and quality  
**Key Learning**: Professional tone, policy adherence, empathy

**Before**: Generic, unhelpful responses  
**After**: Professional, policy-aware, customer-focused answers

---

### 2. 🏥 **Medical Assistant Demo**
```bash
cd medical_assistant && uv run python run_demo.py
```

**Scenario**: Healthcare information assistant  
**Challenge**: Accurate medical information with safety disclaimers  
**Strategy**: `medical_qlora_efficient` - large dataset, memory-efficient training  
**Key Learning**: Domain expertise, safety protocols, ethical considerations

**Before**: General health advice, no safety awareness  
**After**: Medically accurate, safety-first, disclaimer-aware responses

---

### 3. 💻 **Code Helper Demo**
```bash
cd code_helper && uv run python run_demo.py
```

**Scenario**: Programming assistant for Python development  
**Challenge**: Generate working code, explain concepts, debug issues  
**Strategy**: `code_full_quality` - maximum adaptation for technical accuracy  
**Key Learning**: Code generation, best practices, debugging assistance

**Before**: Broken code, poor explanations  
**After**: Working code, clear explanations, best practices

---

### 4. ✨ **Creative Writing Demo**
```bash
cd creative_writing && uv run python run_demo.py
```

**Scenario**: Creative writing assistant for authors  
**Challenge**: Generate engaging stories, maintain style consistency  
**Strategy**: `creative_lora_diverse` - preserve creativity while adding structure  
**Key Learning**: Style adaptation, narrative coherence, creative enhancement

**Before**: Generic, formulaic writing  
**After**: Engaging, stylistically consistent, creative content

---

### 5. 🔧 **Technical Q&A Demo**
```bash
cd technical_qa && uv run python run_demo.py
```

**Scenario**: Engineering documentation assistant  
**Challenge**: Complex technical explanations, accurate specifications  
**Strategy**: `technical_qlora_large` - handle large technical dataset efficiently  
**Key Learning**: Technical accuracy, detailed explanations, specification adherence

**Before**: Vague technical responses  
**After**: Precise, detailed, specification-compliant answers

## 🧠 Educational Focus

Each demo includes detailed commentary explaining:

### **Strategy Selection**
- **Why this model?** (Domain alignment, size considerations)
- **Why this method?** (LoRA for efficiency, Full for quality, QLoRA for large datasets)
- **Why these parameters?** (Learning rate, batch size, epochs)

### **Dataset Considerations**
- **Size impact**: How dataset size affects training time and quality
- **Quality vs Quantity**: When to prioritize one over the other
- **Domain specificity**: Industry-specific vocabulary and concepts

### **Hardware Optimization**
- **Memory management**: GPU vs CPU, batch size optimization
- **Training time**: Balancing speed with quality
- **Production deployment**: Model size and inference speed

### **Real-World Applications**
- **Business value**: ROI of fine-tuning for specific use cases
- **Deployment patterns**: How to integrate fine-tuned models
- **Maintenance**: Updating models with new data

## 📊 Comparison Matrix

| Aspect | Customer Support | Medical Assistant | Code Helper | Creative Writing | Technical Q&A |
|--------|------------------|-------------------|-------------|------------------|---------------|
| **Dataset Size** | 150 (Medium) | 300+ (Large) | 200 (Medium) | 100 (Small) | 400+ (XLarge) |
| **Training Time** | 3-5 min | 5-8 min | 8-12 min | 4-6 min | 6-10 min |
| **Memory Usage** | ~2GB | ~3GB | ~4GB | ~2GB | ~3GB |
| **Quality Focus** | Professional tone | Medical accuracy | Code correctness | Creative flow | Technical precision |
| **Business Impact** | Customer satisfaction | Patient safety | Developer productivity | Content quality | Documentation accuracy |

## 🎓 Learning Objectives

After running all demos, you'll understand:

1. **Strategy Selection**: How to choose the right approach for your use case
2. **Model Architecture**: Why different models excel in different domains  
3. **Training Methods**: When to use LoRA, QLoRA, or Full Fine-tuning
4. **Dataset Engineering**: How to create effective training data
5. **Performance Optimization**: Balancing quality, speed, and resources
6. **Production Readiness**: Deploying and maintaining fine-tuned models

## 🚀 Quick Start

### Run Individual Demo:
```bash
cd demos/customer_support
./run_demo.sh
```

### Run Complete Showcase:
```bash
cd demos
./run_all_demos.sh
```

### Compare Strategies:
```bash
cd demos
./compare_strategies.sh
```

## 📁 Directory Structure

```
demos/
├── README.md                          # This file
├── run_all_demos.sh                   # Complete showcase
├── compare_strategies.sh               # Strategy comparison
│
├── customer_support/                   # E-commerce support demo
│   ├── datasets/
│   │   └── ecommerce_support.jsonl    # 150 support examples
│   ├── strategies/
│   │   └── customer_support_lora.yaml # Strategy configuration
│   ├── run_demo.sh                    # Demo runner
│   └── README.md                      # Specific instructions
│
├── medical_assistant/                  # Healthcare Q&A demo
│   ├── datasets/
│   │   └── medical_qa.jsonl           # 300+ medical examples
│   ├── strategies/
│   │   └── medical_qlora_efficient.yaml
│   ├── run_demo.sh
│   └── README.md
│
├── code_helper/                        # Programming assistant demo
│   ├── datasets/
│   │   └── python_coding.jsonl        # 200 coding examples
│   ├── strategies/
│   │   └── code_full_quality.yaml
│   ├── run_demo.sh
│   └── README.md
│
├── creative_writing/                   # Writing assistant demo
│   ├── datasets/
│   │   └── creative_stories.jsonl     # 100 creative examples
│   ├── strategies/
│   │   └── creative_lora_diverse.yaml
│   ├── run_demo.sh
│   └── README.md
│
└── technical_qa/                       # Engineering Q&A demo
    ├── datasets/
    │   └── engineering_qa.jsonl        # 400+ technical examples
    ├── strategies/
    │   └── technical_qlora_large.yaml
    ├── run_demo.sh
    └── README.md
```

## 🏆 Success Metrics

Each demo demonstrates measurable improvements:

- **Training Loss Reduction**: 20-40% decrease
- **Response Relevance**: Dramatically improved domain focus
- **Professional Quality**: Industry-appropriate tone and accuracy
- **Technical Correctness**: Factual accuracy in specialized domains
- **User Experience**: More helpful, contextual responses

## 🔄 Continuous Learning

The demos are designed to be:
- **Reproducible**: Same results every time
- **Educational**: Clear explanations of decisions
- **Extensible**: Easy to modify for your use cases
- **Production-Ready**: Models can be deployed immediately

## 💡 Next Steps

After exploring the demos:

1. **Adapt for Your Use Case**: Modify datasets and strategies
2. **Scale Up**: Try larger models and datasets
3. **Production Deployment**: Use the saved models in applications
4. **Advanced Techniques**: Experiment with ensemble methods, multi-stage training
5. **Contribute**: Share your successful strategies and datasets

---

**🎯 Ready to see fine-tuning in action? Start with `./run_all_demos.sh` for the complete experience!**