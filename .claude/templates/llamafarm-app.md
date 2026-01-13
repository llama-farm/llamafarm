# LlamaFarm Application Template

Follow these steps to build an application using LlamaFarm:

## Phase 1: Project Setup
1. Initialize project structure
2. Run `lf init` to create llamafarm.yaml
3. Configure models, RAG, and prompts
4. Start LlamaFarm services: `lf start`

## Phase 2: Configuration
1. Edit llamafarm.yaml with required models
2. Set up RAG databases if needed
3. Configure embedding and retrieval strategies
4. Create system prompts for your use case

### Example llamafarm.yaml
```yaml
version: v1
name: my-app
namespace: default

runtime:
  default_model: main
  models:
    - name: main
      provider: ollama
      model: qwen3:8b
      base_url: http://127.0.0.1:11434

rag:
  databases:
    - name: knowledge
      type: ChromaStore
      default_embedding_strategy: embeddings
      default_retrieval_strategy: search

prompts:
  - name: assistant
    messages:
      - role: system
        content: "You are a helpful assistant."
```

## Phase 3: External Databases (if needed)
1. Determine if external DBs are required
2. Time-series → TimescaleDB
3. Relational → PostgreSQL
4. Cache → Redis
5. Set up connections and schemas

## Phase 4: Implementation
1. Create application structure
2. Implement LlamaFarm API calls
3. Integrate external databases if used
4. Add error handling and logging

### Key LlamaFarm APIs
- Chat: `POST /v1/projects/{ns}/{proj}/chat/completions`
- RAG: `POST /v1/projects/{ns}/{proj}/rag/query`
- Anomaly: `POST http://localhost:11540/v1/anomaly/fit`
- Classify: `POST http://localhost:11540/v1/classifier/fit`

## Phase 5: Testing
1. Write tests for each component
2. Test LlamaFarm API integration
3. Test error handling
4. Run full test suite

## Phase 6: Demo
1. Create comprehensive demo script
2. Include mock data if needed
3. Show all features working together
4. Make it impressive and runnable

## Phase 7: Documentation
1. Document API usage
2. Document configuration options
3. Update README with setup instructions

## Reference
See `.claude/docs/LLAMAFARM-REFERENCE.md` for:
- Full API documentation
- Configuration options
- Example configurations
- Troubleshooting guides
