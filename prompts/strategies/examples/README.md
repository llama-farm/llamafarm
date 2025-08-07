# Strategy Examples

This directory contains example strategies demonstrating various use cases and configurations for the LlamaFarm Prompt System.

## Available Strategies

### 1. Data Analysis (`data_analysis.yaml`)
**Use Cases**: Statistical analysis, data visualization, exploratory data analysis
- Multiple analysis templates for different data scenarios
- Visualization detection and recommendations
- Statistical test configurations
- Big data handling strategies

### 2. Content Moderation (`content_moderation.yaml`)
**Use Cases**: Content safety, toxicity detection, policy compliance
- Multi-level moderation with severity levels
- Platform-specific configurations
- Automated bulk processing
- Legal compliance checking

### 3. Educational Tutor (`educational_tutor.yaml`)
**Use Cases**: Personalized tutoring, homework help, test preparation
- Adaptive difficulty levels
- Multiple learning style support
- Subject-specific templates (math, writing, etc.)
- Progress tracking and encouragement

### 4. Product Assistant (`product_assistant.yaml`)
**Use Cases**: E-commerce recommendations, product comparisons, shopping assistance
- Budget-aware recommendations
- Technical product guidance
- Gift recommendations
- Quick decision support

### 5. Travel Planner (`travel_planner.yaml`)
**Use Cases**: Itinerary creation, destination research, travel tips
- Budget and luxury travel options
- Family and adventure travel specializations
- Day-by-day planning with local insights
- Weather and safety considerations

### 6. Health & Wellness (`health_wellness.yaml`)
**Use Cases**: Wellness tips, nutrition guidance, fitness planning
- Evidence-based information only
- Emergency detection and resources
- Age-specific wellness advice
- Mental health support resources

### 7. Financial Advisor (`financial_advisor.yaml`)
**Use Cases**: Financial education, budgeting, investment basics
- Educational focus (not financial advice)
- Life-stage specific guidance
- Debt management strategies
- Retirement planning education

### 8. Software Architect (`software_architect.yaml`)
**Use Cases**: System design, architecture patterns, technology selection
- Cloud and microservices architectures
- Scalability and performance planning
- API and database design
- Startup vs enterprise approaches

### 9. Language Learning (`language_learning.yaml`)
**Use Cases**: Language practice, grammar help, cultural context
- Adaptive proficiency levels
- Conversation and pronunciation practice
- Cultural integration
- Multiple language support

### 10. Creative Workshop (`creative_workshop.yaml`)
**Use Cases**: Creative writing, art concepts, overcoming creative blocks
- Multi-domain creative support
- Genre-specific guidance
- Collaboration features
- Experimental art encouragement

## Strategy Structure

Each strategy demonstrates:
- **Multiple Templates**: Default, fallback, and specialized templates
- **Selection Rules**: Automatic template selection based on context
- **Global Configuration**: System prompts, temperature, model preferences
- **Optimization Settings**: Caching, compression, parallel processing
- **Input/Output Transforms**: Data preprocessing and postprocessing

## Using These Examples

### 1. As Templates
Copy and modify any strategy for your specific needs:
```bash
cp data_analysis.yaml my_custom_analysis.yaml
# Edit to customize
```

### 2. Direct Usage
Load strategies in your code:
```python
from prompts.strategies import StrategyManager

manager = StrategyManager(strategies_dir="strategies/examples")
result = manager.execute_strategy("data_analysis", inputs={...})
```

### 3. Learning Resources
Study strategies to understand:
- Complex conditional logic
- Multi-template coordination
- Performance optimization
- Domain-specific configurations

## Key Patterns

### Conditional Templates
```yaml
specialized:
  - condition:
      query_type: "specific_type"
    template: "specialized_template"
    priority: 20
```

### Selection Rules
```yaml
selection_rules:
  - name: "pattern_detection"
    condition:
      expression: "'keyword' in context.get('query', '').lower()"
    template: "specific_template"
    priority: 30
```

### Performance Profiles
- **Speed**: Quick responses, simpler processing
- **Accuracy**: Lower temperature, detailed analysis
- **Balanced**: Moderate settings for general use

### Safety Features
- Health/medical disclaimers
- Financial education notices
- Emergency resource directions
- Content moderation levels

## Creating Your Own Strategy

1. **Choose a Base**: Start with the most similar example
2. **Define Use Cases**: List specific scenarios
3. **Configure Templates**: Set up default and specialized templates
4. **Add Selection Logic**: Create rules for template selection
5. **Set Global Config**: Define system prompts and parameters
6. **Test Thoroughly**: Validate with various inputs

## Best Practices

1. **Clear Naming**: Use descriptive strategy and template names
2. **Comprehensive Conditions**: Cover edge cases in selection rules
3. **Appropriate Temperature**: Lower for accuracy, higher for creativity
4. **Safety First**: Include necessary disclaimers and warnings
5. **User-Centric**: Design for the end user's needs
6. **Performance Aware**: Configure caching and optimization appropriately

## Contributing

To contribute a new example strategy:
1. Create a well-documented YAML file
2. Include diverse use cases
3. Add comprehensive selection rules
4. Test with various scenarios
5. Submit with example usage

## Support

For questions about these strategies:
- Review the main documentation
- Check the demos directory
- Examine the schema definitions
- Test with the CLI tools