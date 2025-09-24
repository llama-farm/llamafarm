# Multi-Model Support Implementation Plan for LlamaFarm

## Executive Summary

This plan outlines the implementation of multi-model support in LlamaFarm, allowing users to configure and switch between multiple models (initially for Ollama) using named configurations similar to the existing strategies pattern. The implementation will leverage the existing `models/` directory infrastructure and integrate with the config schema compilation process.

## Current State Analysis

### Existing Structure
- **Runtime Configuration**: Single `runtime` block in `llamafarm.yaml` with one model
- **Models Directory**: Existing `models/` with strategy pattern, schema.yaml, and StrategyManager
- **Config System**: Schema compilation via `config/compile_schema.py` and type generation
- **CLI**: No `--model` flag support in `lf run` command
- **Strategies Pattern**: Well-established in both RAG and models systems

### Key Files to Modify
- `models/runtime_schema.yaml` - NEW: Runtime models schema definition
- `config/schema.yaml` - Add $ref to models/runtime_schema.yaml
- `config/templates/default.yaml` - Update default template with multi-model
- `config/compile_schema.py` - Already handles $refs properly
- `cli/cmd/run.go` - Add --model flag support
- `cli/cmd/models.go` - NEW: Model management commands
- NO MIGRATION: Fresh start with new schema

## Proposed Architecture

### Key Design Decisions
1. **Leverage models/ directory** - Use existing strategy infrastructure
2. **Schema integration** - Create `models/runtime_schema.yaml` and reference via $ref
3. **No migration** - Start fresh with new multi-model structure
4. **Config compilation** - Use existing compile_schema.py process
5. **Single default_model field** - One place to change default, no dual-write needed

### 1. Configuration Structure

#### models/runtime_schema.yaml (NEW)
```yaml
# Runtime Models Schema - For multi-model support
$schema: http://json-schema.org/draft-07/schema#

type: object
properties:
  default_model:
    type: string
    description: Name of the default model to use
    pattern: "^[a-z0-9][a-z0-9-_]*$"
    
  runtime_models:
    type: array
    description: Named model configurations for runtime use
    minItems: 1
    items:
      type: object
      required: [name, provider, model]
      properties:
        name:
          type: string
          pattern: "^[a-z0-9][a-z0-9-_]*$"
          description: Unique model configuration name
        
        provider:
          type: string
          enum: [openai, ollama, anthropic]
          
        model:
          type: string
          description: Model identifier
          
        base_url:
          type: string
          description: API base URL
          
        api_key:
          type: string
          description: API key (if required)
          
        instructor_mode:
          type: string
          enum: [json, markdown, text]
          
        parameters:
          type: object
          properties:
            temperature:
              type: number
              minimum: 0
              maximum: 2
            top_p:
              type: number
            top_k:
              type: integer
            max_tokens:
              type: integer
          additionalProperties: true
```

#### Updated config/schema.yaml
```yaml
# Add reference to models schema
properties:
  # ... existing properties ...
  
  # NEW: Multi-model support via $ref
  default_model:
    $ref: "../models/runtime_schema.yaml#/properties/default_model"
    
  runtime_models:
    $ref: "../models/runtime_schema.yaml#/properties/runtime_models"
  
  # KEEP for backward compatibility (deprecated)
  runtime:
    type: object
    deprecated: true
    # ... existing runtime schema ...
```

#### Updated llamafarm.yaml structure
```yaml
version: v1
name: project_name
namespace: llamafarm

# NEW: Default model selection (single write to change)
default_model: "primary"

# NEW: Named model configurations
runtime_models:
  - name: "primary"
    provider: "ollama"
    model: "llama3.1:8b"
    base_url: "http://localhost:11434"
    parameters:
      temperature: 0.5
    instructor_mode: "json"
    
  - name: "backup"
    provider: "ollama"
    model: "qwen3:8b"
    base_url: "http://localhost:11434"
    parameters:
      temperature: 0.7
      
  - name: "creative"
    provider: "ollama"
    model: "mixtral:8x7b"
    base_url: "http://localhost:11434"
    parameters:
      temperature: 0.9
      top_p: 0.95
      
  - name: "coding"
    provider: "ollama"
    model: "codellama:13b"
    base_url: "http://localhost:11434"
    parameters:
      temperature: 0.3
      max_tokens: 2048
```

### 2. CLI Interface Design

```bash
# Use default model
lf run "Hello world"

# Use named model configuration
lf run --model backup "Hello world"
lf run --model creative "Write a story"
lf run --model coding "Write a Python function"

# List configured models
lf models list
# Output:
# NAME        PROVIDER    MODEL            DEFAULT
# default     ollama      llama3.1:8b      ✓
# backup      ollama      qwen3:8b         
# creative    ollama      mixtral:8x7b     
# coding      ollama      codellama:13b    

# Show model details
lf models show creative
# Output:
# Name: creative
# Provider: ollama
# Model: mixtral:8x7b
# Temperature: 0.9
# Top-p: 0.95
# Base URL: http://localhost:11434

# Set default model (single config change)
lf models set-default backup
# Updates: default_model: "backup"

# NEW: Auto-import all Ollama models
lf models import-ollama
# Output:
# Discovering Ollama models...
# Found 8 models:
#   ✓ llama3.1:8b (already configured as 'primary')
#   ✓ qwen3:8b (already configured as 'backup')
#   + Adding llama3.2:3b as 'llama3-2-3b'
#   + Adding mistral:7b as 'mistral-7b'
#   + Adding phi3:mini as 'phi3-mini'
#   + Adding gemma2:2b as 'gemma2-2b'
#   ! Skipping mixtral:8x7b (already configured)
#   ! Skipping codellama:13b (already configured)
# Added 4 new model configurations to llamafarm.yaml
# Current default: primary

# Import with custom naming pattern and set default
lf models import-ollama --prefix ollama- --set-default llama3.2:3b
# Output:
# Added: ollama-llama3-2-3b, ollama-mistral-7b, ollama-phi3-mini
# Updated default_model: "ollama-llama3-2-3b"

# Import specific models only
lf models import-ollama --filter "llama*" --filter "mistral*"
```

## Implementation Steps

### Phase 1: Schema Integration in models/

1. **Create Runtime Schema** (`models/runtime_schema.yaml`)
   - Define runtime_models array schema
   - Include validation patterns for names
   - Support provider-specific parameters
   - Mark one model as default

2. **Update Config Schema** (`config/schema.yaml`)
   - Add $ref to models/runtime_schema.yaml
   - Keep runtime block as deprecated
   - NO MIGRATION - fresh start only

3. **Compile and Generate Types**
   - Run `config/generate-types.sh`
   - Verify schema compilation works with $ref
   - Check generated datamodel.py includes RuntimeModels

### Phase 2: CLI Command Updates

3. **Update `run` Command** (`cli/cmd/run.go`)
   ```go
   // Add model flag
   var runModelName string
   runCmd.Flags().StringVar(&runModelName, "model", "", "Named model configuration to use")
   ```

4. **Model Resolution Logic**
   - If `--model` specified, lookup by name
   - Otherwise, use model named in `default_model` field
   - Fallback to first model if `default_model` not specified

5. **Add `models` Command** (`cli/cmd/models.go`)
   - Subcommands: `list`, `show`, `set-default`, `import-ollama`
   - Similar structure to existing `rag` command group
   
6. **Implement Ollama Discovery** (`cli/cmd/models_import.go`)
   - Call `ollama list` to get available models
   - Parse model names and tags
   - Generate safe configuration names (e.g., `llama3.2:3b` → `llama3-2-3b`)
   - Skip already configured models
   - Add new models with sensible defaults

### Phase 3: Template Updates

6. **Update Default Template** (`config/templates/default.yaml`)
   - Replace runtime block with runtime_models array
   - Include 2-3 example model configurations
   - Set one as default

7. **Runtime Model Manager** (`models/core/runtime_manager.py`)
   - NEW: Create RuntimeManager class
   - Handle model selection by name
   - Resolve default model
   - Validate model configurations

### Phase 4: CLI Runtime Integration

8. **Update Chat Client** (`cli/cmd/chat_client.go`)
   - Accept model name parameter
   - Look up model config from runtime_models
   - Pass to server API

9. **Server Integration**
   - Update project seed to use single model (no changes needed)
   - Server continues to receive runtime config as before

### Phase 5: Validation & Testing

9. **Validation Rules**
   - Unique model names
   - Valid provider/model combinations
   - At least one model defined
   - `default_model` must reference existing model name
   - Validate name patterns (lowercase, alphanumeric, hyphens, underscores)

10. **Test Coverage**
    - Unit tests for model selection logic
    - Integration tests for CLI commands
    - Backward compatibility tests

## NO MIGRATION STRATEGY

### Design Decision: Fresh Start
- **No automatic migration** from runtime to runtime_models
- Users must explicitly update their configs
- Clear documentation and examples provided
- Simpler implementation, fewer edge cases

### Transition Path
1. **New projects**: Use runtime_models by default
2. **Existing projects**: Continue using runtime block
3. **Documentation**: Show clear before/after examples
4. **Future**: Deprecate runtime in v2.0 (6+ months)

## Error Handling

### Common Error Scenarios
1. **Model not found**: "Model 'creative' not found. Available models: default, backup"
2. **No default model**: "No default model specified. Use --model flag or set default with 'lf models set-default'"
3. **Duplicate names**: "Model name 'default' is already in use"
4. **Invalid provider**: "Provider 'openai' not supported. Use: ollama"
5. **Ollama not running**: "Cannot connect to Ollama. Ensure it's running with 'ollama serve'"
6. **Import name conflict**: "Model name 'llama3' already exists. Use --prefix to avoid conflicts"

## Future Enhancements

### Phase 2 Features (3-6 months)
- Model-specific prompt templates
- Automatic model selection based on task type
- Model performance benchmarking
- Cost tracking per model
- Multi-provider support (OpenAI, Anthropic, etc.)

### Phase 3 Features (6-12 months)
- Model routing strategies (similar to RAG strategies)
- Ensemble model support
- A/B testing framework
- Model versioning and rollback

## Success Metrics

1. **User Adoption**
   - 50% of users configure 2+ models within first month
   - 80% use named models vs raw model strings

2. **Developer Experience**
   - Zero breaking changes for existing users
   - CLI commands complete in <100ms
   - Clear error messages with actionable fixes

3. **System Performance**
   - No regression in response times
   - Model switching overhead <50ms
   - Configuration parsing <10ms

## Implementation Order

### Day 1-2: Schema Foundation
1. Create `models/runtime_schema.yaml`
2. Update `config/schema.yaml` with $ref
3. Run schema compilation and type generation
4. Verify types are correctly generated

### Day 3-4: Configuration Templates
5. Update `config/templates/default.yaml`
6. Create `models/core/runtime_manager.py`
7. Add unit tests for RuntimeManager

### Day 5-6: CLI Integration
8. Add --model flag to run.go
9. Implement model selection logic
10. Create models command (list, show, set-default)
11. Implement import-ollama subcommand with discovery

### Day 7: Testing & Documentation
11. Integration tests
12. Update README and examples
13. Create migration guide

## Documentation Updates Required

1. **README.md**: Add multi-model examples
2. **CLI Help Text**: Update all relevant commands
3. **Configuration Guide**: New models section
4. **Migration Guide**: For existing users
5. **API Documentation**: Model selection endpoints

## Review Checklist

- [ ] Schema validates all edge cases
- [ ] CLI commands follow existing patterns
- [ ] Backward compatibility maintained
- [ ] Error messages are helpful
- [ ] Tests cover all scenarios
- [ ] Documentation is complete
- [ ] Performance benchmarks pass

## Appendix: Example Configurations

### Simple Setup (2 models) - config/templates/default.yaml
```yaml
version: v1
name: my-project
namespace: default

# Single place to change default model
default_model: "primary"

runtime_models:
  - name: "primary"
    provider: "ollama"
    model: "llama3.1:8b"
    base_url: "http://localhost:11434"
    parameters:
      temperature: 0.7
    
  - name: "fallback"
    provider: "ollama"  
    model: "mistral:7b"
    base_url: "http://localhost:11434"
    parameters:
      temperature: 0.5
```

### Advanced Setup (task-specific)
```yaml
# Easy to switch defaults - just change this line
default_model: "general"

runtime_models:
  - name: "general"
    provider: "ollama"
    model: "llama3.1:8b"
    base_url: "http://localhost:11434"
    instructor_mode: "json"
    parameters:
      temperature: 0.5
    
  - name: "analysis"
    provider: "ollama"
    model: "mixtral:8x7b"
    base_url: "http://localhost:11434"
    parameters:
      temperature: 0.3
      top_k: 10
    
  - name: "creative"
    provider: "ollama"
    model: "llama3.1:70b"
    base_url: "http://localhost:11434"
    instructor_mode: "markdown"
    parameters:
      temperature: 0.9
      top_p: 0.95
    
  - name: "code"
    provider: "ollama"
    model: "codellama:34b"
    base_url: "http://localhost:11434"
    parameters:
      temperature: 0.2
      max_tokens: 2048

# Swap defaults on the fly:
# default_model: "creative"  # for creative tasks
# default_model: "code"      # for coding sessions
# default_model: "analysis"  # for data analysis
```

## Key Implementation Notes

### Schema Integration
- The `models/runtime_schema.yaml` will be referenced via $ref in `config/schema.yaml`
- The existing `config/compile_schema.py` already handles $refs properly
- Type generation via `config/generate-types.sh` will create proper Python models

### No Migration Complexity
- Fresh start approach eliminates migration bugs
- Users explicitly opt-in to new multi-model config
- Server seed remains simple with single model
- Default template shows multi-model best practices

### Ollama Import Feature
- **Discovery**: Uses `ollama list` command to find local models
- **Name Generation**: Converts model tags to safe config names
  - `llama3.2:3b` → `llama3-2-3b`
  - `mistral:7b-instruct` → `mistral-7b-instruct`
  - `phi3:mini` → `phi3-mini`
- **Conflict Resolution**: Skip models already in config
- **Defaults**: Apply sensible temperature based on model type
  - Code models: 0.3
  - Chat models: 0.7
  - Creative models: 0.9
- **Batch Operations**: Import all at once or filter by pattern

### Leveraging Existing Infrastructure
- Use `models/` directory's existing strategy patterns
- RuntimeManager follows same pattern as StrategyManager
- Schema compilation process unchanged
- CLI patterns match existing RAG commands

### Future Extensions
- Model routing based on prompt analysis
- Fallback chains for reliability
- A/B testing between models
- Cost/performance optimization strategies
- Auto-import from other providers (OpenAI, Anthropic, etc.)