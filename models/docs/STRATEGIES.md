# LlamaFarm Strategies Collection

This directory contains **strategy-based configurations** for different use cases, deployment scenarios, and requirements. Instead of manually configuring individual components, you can choose a strategy that matches your needs and let LlamaFarm optimize the entire stack for you.

## 🎯 Strategy-First Philosophy

**Why strategies over traditional config?**
- **Opinionated excellence**: Each strategy represents best practices for specific use cases
- **Hardware optimization**: Automatically adapts to your hardware (M1, NVIDIA, CPU-only)
- **Intelligent fallbacks**: Built-in fallback chains ensure reliability
- **Cost awareness**: Strategies include cost optimization and budget management
- **Performance tuning**: Pre-configured for optimal performance in each scenario

## 📚 Available Strategies

### 🚀 **API-First & Quick Start**

#### [`api_first_cost_optimized.yaml`](./api_first_cost_optimized.yaml)
**Perfect for: Prototyping, MVPs, low-volume production**
- **Cost**: $10-50/month
- **Setup time**: < 30 minutes  
- **Complexity**: Easy
- Uses GPT-4o-mini (60x cheaper than GPT-4) with intelligent routing
- Built-in cost monitoring and budget alerts
- Great for startups and proof-of-concepts

```yaml
# Quick start example
strategy: "api_first_cost_optimized"
environments:
  development:
    daily_budget: 2.0
    max_tokens: 1000
```

#### [`startup_budget_conscious.yaml`](./startup_budget_conscious.yaml)
**Perfect for: Bootstrapped startups, MVPs, extreme budget constraints**
- **Cost**: $0-25/month
- **Setup time**: < 1 hour
- **Complexity**: Easy
- Maximizes free tiers and local models
- Aggressive caching and cost optimization
- Scaling path from $0 to $1000/month planned

### 🏠 **Local & Privacy-First**

#### [`local_first_privacy.yaml`](./local_first_privacy.yaml)
**Perfect for: Healthcare, legal, financial, sensitive enterprise data**
- **Privacy**: Maximum (data never leaves device)
- **Setup time**: 30-60 minutes
- **Complexity**: Intermediate
- HIPAA/GDPR/SOX compliant
- Complete network isolation option
- Optimized for Apple Silicon, NVIDIA, and CPU-only

```yaml
# Privacy-first example
strategy: "local_first_privacy"
privacy_settings:
  network_isolation: true
  encrypted_storage: true
  audit_trail: true
```

### 🔬 **Fine-Tuning & Specialization**

#### [`domain_specialist_finetuning.yaml`](./domain_specialist_finetuning.yaml)
**Perfect for: Medical AI, legal assistants, technical support, specialized consulting**
- **Training time**: 2-8 hours
- **Setup time**: 1-2 hours
- **Complexity**: Advanced
- QLoRA/LoRA optimization for M1 and NVIDIA hardware
- Domain-specific configurations (medical, legal, technical, financial)
- Built-in evaluation and safety checks

```yaml
# Medical specialist example  
strategy: "domain_specialist_finetuning"
domain_configs:
  medical:
    safety_checks: true
    disclaimer_required: true
    special_tokens: ["<PATIENT>", "<DIAGNOSIS>"]
```

### ⚡ **Performance & Production**

#### [`performance_optimized_throughput.yaml`](./performance_optimized_throughput.yaml)
**Perfect for: Real-time APIs, live chat, high-frequency trading, gaming**
- **Throughput**: 1000+ requests/second
- **Latency**: < 100ms p95
- **Complexity**: Advanced
- Multi-tier model architecture
- Advanced caching and optimization
- Auto-scaling and load balancing

#### [`hybrid_cloud_local.yaml`](./hybrid_cloud_local.yaml)
**Perfect for: Production applications, growing startups, privacy-conscious enterprises**
- **Cost optimization**: Intelligent routing
- **Privacy**: Configurable levels
- **Complexity**: Intermediate
- Smart routing between cloud and local models
- Privacy classification system
- Cost management with budget controls

### 🔬 **Research & Experimentation**

#### [`research_experimentation.yaml`](./research_experimentation.yaml)
**Perfect for: Academic research, model evaluation, prototyping, AI education**
- **Flexibility**: Maximum
- **Reproducibility**: High
- **Complexity**: Intermediate
- Multi-model comparison framework
- Academic benchmark integration
- Experiment tracking and collaboration tools

## 🎯 How to Choose a Strategy

### Quick Decision Tree

```
🤔 What's your primary goal?

├── 💰 Minimize costs
│   ├── Have $0 budget → startup_budget_conscious
│   └── Have small budget → api_first_cost_optimized
│
├── 🔒 Maximum privacy/security
│   └── → local_first_privacy
│
├── 🚀 Production performance
│   ├── Need high throughput → performance_optimized_throughput
│   └── Need balanced approach → hybrid_cloud_local
│
├── 🧠 Domain expertise
│   └── → domain_specialist_finetuning
│
└── 🔬 Research/experiments
    └── → research_experimentation
```

### By Use Case

| Use Case | Strategy | Why |
|----------|----------|-----|
| **Startup MVP** | `startup_budget_conscious` | Maximum value for minimal cost |
| **Enterprise API** | `hybrid_cloud_local` | Balance of performance, cost, privacy |
| **Healthcare App** | `local_first_privacy` | HIPAA compliance, data protection |
| **Trading Bot** | `performance_optimized_throughput` | Ultra-low latency requirements |
| **Legal Assistant** | `domain_specialist_finetuning` | Specialized knowledge and accuracy |
| **Research Project** | `research_experimentation` | Flexibility and reproducibility |
| **Customer Support** | `api_first_cost_optimized` | Quick deployment, cost control |

### By Budget

| Monthly Budget | Recommended Strategy | Features |
|----------------|---------------------|----------|
| **$0-25** | `startup_budget_conscious` | Free local models, minimal APIs |
| **$25-100** | `api_first_cost_optimized` | Smart API usage, cost monitoring |
| **$100-500** | `hybrid_cloud_local` | Best of both worlds |
| **$500+** | `performance_optimized_throughput` | Maximum performance |

### By Technical Complexity

| Complexity | Strategies | Setup Time |
|------------|-----------|------------|
| **Easy** | `api_first_cost_optimized`, `startup_budget_conscious` | < 30 min |
| **Intermediate** | `local_first_privacy`, `hybrid_cloud_local`, `research_experimentation` | 30-90 min |
| **Advanced** | `domain_specialist_finetuning`, `performance_optimized_throughput` | 1-4 hours |

## 🚀 Quick Start

### 1. Choose Your Strategy
```bash
# List available strategies
ls /models/strategies/

# View strategy details
cat /models/strategies/api_first_cost_optimized.yaml
```

### 2. Use Strategy in Demo
```bash
# Run with specific strategy
cd /models/demos/creative_writing
STRATEGY=api_first_cost_optimized python run_demo.py

# Or configure in your application
strategy: "api_first_cost_optimized"
```

### 3. Customize as Needed
```yaml
# Override specific settings while keeping strategy benefits
strategy: "hybrid_cloud_local"
strategy_overrides:
  cost_management:
    daily_budget: 50.0  # Increase budget
  privacy_classifier:
    sensitivity_levels:
      internal:
        route_to: "local_only"  # More conservative privacy
```

## 🔧 Strategy Components

Each strategy includes:

### 📋 **Strategy Info**
- Name, description, use case
- Difficulty level and setup time
- Cost estimates and requirements

### 🤖 **Model Selection**
- Primary and fallback models
- Hardware-optimized configurations
- Intelligent routing rules

### ⚙️ **Environment Configs**
- Apple Silicon (M1/M2/M3) optimization
- NVIDIA GPU configurations
- CPU-only fallback options

### 💰 **Cost Management**
- Budget controls and alerts
- Cost optimization techniques
- Usage monitoring and reporting

### 🔒 **Privacy & Security**
- Data handling policies
- Compliance configurations
- Audit and monitoring features

### 📊 **Performance Tuning**
- Latency optimization
- Throughput maximization
- Resource utilization

## 🎨 Creating Custom Strategies

### Strategy Template
```yaml
# My Custom Strategy
version: "v1"

strategy_info:
  name: "my_custom_strategy"
  description: "Custom strategy for my specific needs"
  use_case: "My specific use case"
  difficulty: "intermediate"

# Your custom configuration...
model_selection:
  primary: "your_preferred_model"
  fallback_chain: ["model1", "model2", "model3"]

environments:
  my_environment:
    active: true
    # Your environment config...
```

### Best Practices
1. **Start with existing strategy**: Copy the closest match and modify
2. **Include fallbacks**: Always have backup options
3. **Document decisions**: Explain why choices were made
4. **Test thoroughly**: Validate performance and costs
5. **Version control**: Track changes and improvements

## 🔄 Migration Between Strategies

### From API-Only to Hybrid
```bash
# Current: api_first_cost_optimized
# Target: hybrid_cloud_local

# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Pull required models
ollama pull llama3.1:8b

# 3. Update strategy
sed -i 's/api_first_cost_optimized/hybrid_cloud_local/' config.yaml

# 4. Test and monitor
python test_strategy.py --validate
```

### Cost Migration Path
```
startup_budget_conscious ($0-25)
         ↓
api_first_cost_optimized ($25-100)  
         ↓
hybrid_cloud_local ($100-500)
         ↓
performance_optimized_throughput ($500+)
```

## 📈 Strategy Evolution

### Automatic Upgrades
Strategies can suggest upgrades when:
- Usage exceeds current tier limits
- Cost efficiency would improve
- New capabilities are needed
- Performance requirements change

### Community Strategies
- Submit your strategies via PR
- Share successful configurations
- Collaborate on domain-specific optimizations
- Learn from production deployments

## 🆘 Getting Help

### Strategy Selection Help
- Use the decision tree above
- Check example use cases
- Consider your constraints (budget, privacy, performance)
- Start simple and evolve

### Configuration Help
- Each strategy includes detailed comments
- Example applications are provided
- Hardware requirements are specified
- Migration paths are documented

### Community Support
- GitHub Discussions for strategy questions
- Discord for real-time help
- Strategy showcase for inspiration
- Best practices documentation

---

**Remember**: Strategies are starting points, not rigid rules. Customize them to fit your specific needs while maintaining the benefits of opinionated excellence.