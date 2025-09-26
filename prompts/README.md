# LlamaFarm Prompts - Atomic Agents Pattern

A clean, structured prompt system following the Atomic Agents pattern with reusable components.

## 🎯 Overview

This system implements the **Atomic Agents pattern** for prompt engineering, featuring:
- **Background**: Defines the agent's role and expertise
- **Steps**: Sequential instructions for task execution
- **Output Instructions**: Guidelines for response formatting

## 📁 Structure

```
prompts/
├── schema.yaml           # JSON-Schema definition (v2.0.0)
├── default-prompts.yaml  # Default agents with all types
└── examples/            # Domain-specific agent collections
    ├── customer-service.yaml  # 8 customer service agents
    ├── code-review.yaml      # 8 code review agents
    └── data-analysis.yaml    # 8 data analysis agents
```

## 🏗️ Architecture

### Prompts (Agent Definitions) - Listed First
```yaml
prompts:
  - id: agent_name
    type: agent|specialist|analyzer|coordinator|...
    background: [...]      # Role definition
    steps: [...]          # Execution steps
    output_instructions: [...] # Output format
```

### Components (Reusable Building Blocks) - Listed After Prompts
```yaml
components:
  backgrounds:       # Agent role definitions
  steps:            # Task execution sequences
  output_instructions: # Response formatting rules
```

**Note:** Components are now placed at the bottom of YAML files for better readability, with prompts taking the primary position at the top.

## 🔄 Component Reusability

Components can be:
1. **Referenced**: `"@components.backgrounds.expert_analyst"`
2. **Combined**: `combine: ["@components.x", "@components.y"]`
3. **Extended**: Mix references with inline definitions

Example:
```yaml
background:
  combine:
    - "@components.backgrounds.base_expert"
    - "Additional specialized knowledge"
```

## 🤖 Agent Types

- **agent**: General purpose agents
- **specialist**: Domain experts
- **analyzer**: Data and code analysis
- **coordinator**: Multi-step orchestration
- **validator**: Verification and testing
- **generator**: Content creation
- **assistant**: User interaction

## 📝 Atomic Pattern

Each agent follows this structure:

### 1. Background
Establishes the agent's identity:
- Role and expertise
- Domain knowledge
- Operating principles

### 2. Steps
Numbered sequence of actions:
1. Understand the request
2. Gather information
3. Process and analyze
4. Generate solution
5. Validate results

### 3. Output Instructions
Formatting guidelines:
- Response structure
- Language style
- Level of detail
- Special formatting

## 🎨 Examples

### Customer Service
- `first_contact_agent`: Initial customer inquiries
- `complaint_specialist`: Complaint resolution
- `technical_support_agent`: Technical troubleshooting
- `retention_specialist`: Customer retention
- `onboarding_agent`: New customer setup
- `billing_specialist`: Payment issues
- `feedback_coordinator`: Feedback management
- `vip_concierge`: Premium support

### Code Review
- `pr_reviewer`: Pull request reviews
- `security_auditor`: Security analysis
- `performance_optimizer`: Performance tuning
- `refactoring_specialist`: Code refactoring
- `bug_detective`: Bug detection
- `test_coverage_analyst`: Test analysis
- `api_reviewer`: API design
- `dependency_auditor`: Dependency management

### Data Analysis
- `business_analyst`: Business insights
- `statistical_analyst`: Statistical tests
- `ml_engineer`: Machine learning
- `visualization_expert`: Data visualization
- `trend_analyst`: Trend detection
- `anomaly_detector`: Outlier detection
- `segment_analyst`: Segmentation
- `experiment_analyst`: A/B testing

## 🔧 Usage

### Basic Agent
```yaml
- id: simple_agent
  name: Simple Agent
  type: agent
  background:
    - "You are a helpful assistant."
  steps:
    - "1. Understand the request"
    - "2. Provide a response"
  output_instructions:
    - "Be clear and concise"
```

### With Components
```yaml
- id: advanced_agent
  name: Advanced Agent
  type: specialist
  background: "@components.backgrounds.expert_base"
  steps: "@components.steps.analysis_workflow"
  output_instructions: "@components.output_instructions.detailed_report"
```

### Combined Components
```yaml
- id: hybrid_agent
  name: Hybrid Agent
  type: coordinator
  background:
    combine:
      - "@components.backgrounds.analyst"
      - "@components.backgrounds.coordinator"
  steps:
    combine:
      - "@components.steps.analysis"
      - "@components.steps.coordination"
  output_instructions:
    - "Custom instruction 1"
    - "@components.output_instructions.standard"
```

## 📊 Variables

Agents support variables with:
- **Types**: string, number, boolean, array, object
- **Validation**: min/max, patterns, enums
- **Defaults**: Default values
- **Requirements**: Required/optional flags

Example:
```yaml
variables:
  - name: domain
    type: string
    required: false
    default: "general"
    description: "Domain expertise"
    validation:
      enum: ["tech", "business", "science"]
```

## 🚀 Getting Started

1. **Choose a base**: Start with `default-prompts.yaml`
2. **Select an agent**: Pick one that matches your use case
3. **Customize**: Modify background, steps, or output instructions
4. **Add variables**: Define dynamic parameters
5. **Test**: Validate against `schema.yaml`

## 📐 Schema Validation

The `schema.yaml` file is JSON-Schema compliant and can be used with any JSON-Schema validator to ensure your prompt configurations are valid.

## 🎯 Design Philosophy

1. **Modularity**: Reusable components reduce duplication
2. **Clarity**: Clear structure makes agents understandable
3. **Flexibility**: Mix and match components as needed
4. **Standardization**: Consistent pattern across all agents
5. **Extensibility**: Easy to add new agents and components

## 📈 Benefits

- **Consistency**: All agents follow the same pattern
- **Reusability**: Share components across agents
- **Maintainability**: Update components in one place
- **Scalability**: Easy to add new agents
- **Clarity**: Self-documenting structure

## 🔮 Future Enhancements

- Runtime implementation
- Dynamic component loading
- A/B testing framework
- Performance metrics
- Agent composition tools

---

*Based on the Atomic Agents pattern for structured, reusable prompt engineering*