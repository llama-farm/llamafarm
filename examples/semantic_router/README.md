# Semantic Router Demo

This example demonstrates LlamaFarm's semantic router, which routes queries to specialized LLM models based on topic similarity using sentence-transformer embeddings.

## What This Demo Shows

- **Topic-based routing**: Queries about billing go to the billing model, support queries go to tech support, etc.
- **Sub-millisecond routing**: Routing decisions are made in under 10ms without calling an LLM
- **Automatic fallback**: Queries that don't match any route go to a general assistant
- **Transparent integration**: Works through the standard chat/completions API

## Prerequisites

1. **Ollama** running locally with `llama3.2:3b` model:
   ```bash
   ollama pull llama3.2:3b
   ollama serve  # if not already running
   ```

2. **LlamaFarm services** started:
   ```bash
   # From this example directory
   lf start
   # Or from project root:
   # lf services start
   ```

## Running the Demo

### Option 1: Using the run script

```bash
./run_example.sh
```

### Option 2: Manual testing

1. Start services from this directory:
   ```bash
   lf start
   ```

2. Test routing with different queries:

   **Billing queries** (routes to billing_specialist):
   ```bash
   lf chat "What is my current account balance?"
   lf chat "Can I set up automatic payments?"
   ```

   **Support queries** (routes to tech_support):
   ```bash
   lf chat "I can't log in to my account"
   lf chat "The app keeps crashing on startup"
   ```

   **Sales queries** (routes to sales_team):
   ```bash
   lf chat "How much does the enterprise plan cost?"
   lf chat "What's the difference between the plans?"
   ```

   **General queries** (routes to general_assistant):
   ```bash
   lf chat "What is the capital of France?"
   lf chat "Tell me about machine learning"
   ```

## How It Works

1. The router is configured in `llamafarm.yaml` with `provider: router`
2. Each route has a name, target model, and example utterances
3. When a query arrives, the router:
   - Embeds the query using sentence-transformers
   - Computes cosine similarity with all route utterances
   - Routes to the best matching model (if above threshold)
   - Falls back to default model otherwise

## Configuration

The router is defined in `llamafarm.yaml`:

```yaml
- name: smart_router
  provider: router
  embedder_model: sentence-transformers/all-MiniLM-L6-v2
  default_model: general_assistant
  similarity_threshold: 0.6
  routes:
    - name: billing
      target_model: billing_specialist
      utterances:
        - "what is my bill"
        - "payment options"
        ...
```

## Customization

- **Add routes**: Add new routes with different target models and utterances
- **Adjust threshold**: Lower `similarity_threshold` for more permissive matching
- **Change embedder**: Use different sentence-transformer models for accuracy/speed tradeoffs
- **Add complexity routing**: Configure `complexity_classifier` to route by query complexity

## Router Models Storage

Router models are saved to `~/.llamafarm/models/router/` alongside other ML models (anomaly detectors, classifiers).

## API Endpoints

The router also exposes direct API endpoints:

- `POST /v1/router/train` - Train a router
- `POST /v1/router/route` - Test routing decision
- `GET /v1/router/models` - List saved routers
- `DELETE /v1/router/models/{name}` - Delete a router

See the LlamaFarm API documentation for more details.
