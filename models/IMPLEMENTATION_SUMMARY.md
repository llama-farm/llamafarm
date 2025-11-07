# Model Farm - Runtime Selection & Recommended Models

## Summary of New Features

This update adds comprehensive runtime selection, download commands, and recommended models to the Model Farm catalog.

### 1. Recommended Models
Families can now designate recommended models that appear at the top of the selection UI.

**Example** (from qwen3.yaml):
```yaml
recommended:
  - category: "Small & Fast"
    description: "Efficient models for quick responses"
    models:
      - variant_id: "qwen3:0.6b"
        priority: 1
      - variant_id: "qwen3:1.7b"
        priority: 2
```

### 2. Runtime & Format Information
Each provider now includes:
- `runtime`: Which runtime executes the model (universal, ollama, lemonade, openai)
- `format`: Model format type (transformers, gguf, onnx, api)
- `download_command`: Complete command to download the model
- `backend`: Lemonade-specific backend (llamacpp, transformers, onnx)

**Example**:
```yaml
providers:
  universal:
    provider: universal
    runtime: universal
    format: transformers
    model_id: "Qwen/Qwen3-4B"
    download_command: "Auto-downloads from HuggingFace on first use"

  ollama:
    provider: ollama
    runtime: ollama
    format: gguf
    model_id: "qwen3:4b"
    download_command: "ollama pull qwen3:4b"

  lemonade:
    provider: lemonade
    runtime: lemonade
    format: gguf
    backend: llamacpp
    model_id: "user.Qwen3-4B"
    checkpoint: "unsloth/Qwen3-4B-GGUF:Q4_K_M"
    recipe: "llamacpp"
    download_command: "uv run lemonade-server-dev pull user.Qwen3-4B --checkpoint unsloth/Qwen3-4B-GGUF:Q4_K_M --recipe llamacpp"
```

### 3. Runtime Compatibility Logic

| Format       | Universal | Ollama | Lemonade (llamacpp) | Lemonade (transformers) | Lemonade (onnx) | OpenAI |
|--------------|-----------|--------|---------------------|-------------------------|-----------------|--------|
| transformers | ✅         | ❌      | ❌                   | ✅                       | ❌               | ❌      |
| gguf         | ❌         | ✅      | ✅                   | ❌                       | ❌               | ❌      |
| onnx         | ❌         | ❌      | ❌                   | ❌                       | ✅               | ❌      |
| api          | ❌         | ❌      | ❌                   | ❌                       | ❌               | ✅      |

**Key Points**:
- **Universal**: Only transformers (HuggingFace models)
- **Ollama**: Only GGUF (quantized models)
- **Lemonade**: All formats depending on backend
  - `llamacpp` backend: GGUF models
  - `transformers` backend: HuggingFace models
  - `onnx` backend: ONNX models
- **OpenAI**: Only API format (cloud models via OpenAI-compatible endpoints)

## Schema Changes

### New Definitions
- `recommended_model`: A recommended model variant reference
- `recommended_category`: Category of recommended models

### Updated Fields
- `provider_config`: Now requires `runtime` and `format`
- `model_family`: New optional `recommended` array

### New Required Fields (provider_config)
- `runtime`: Runtime type (universal, ollama, lemonade, openai)
- `format`: Model format (transformers, gguf, onnx, api)

### New Optional Fields (provider_config)
- `download_command`: Complete download command string
- `backend`: Lemonade backend type
- `recipe`: Lemonade recipe (same as backend)

## Files Modified

1. **models/schema.yaml** - Updated with new definitions and required fields
2. **models/text-generation/qwen3.yaml** - Complete example with all new features
3. **models/README.md** - Updated documentation
4. **models/RUNTIME_COMPATIBILITY.md** (NEW) - Comprehensive runtime guide

## Next Steps for Implementation

### UI Changes Needed

1. **Recommended Models Section**
   - Add "Recommended" section at top of model selection
   - Group by category (Small & Fast, Balanced, Powerful, etc.)
   - Show 1-3 top picks per category

2. **Runtime Filter/Selector**
   - Add dropdown/tabs to filter by runtime:
     - All Runtimes
     - Universal (transformers)
     - Ollama (GGUF)
     - Lemonade (all formats)
   - Show only compatible models for selected runtime
   - Display format badge (GGUF, Transformers, ONNX)

3. **Provider Selection on Add**
   - When user clicks "Add" on a variant, show provider selection modal
   - List available providers for that variant
   - Show:
     - Runtime name
     - Format type
     - Download command (copy button)
     - Notes/requirements
   - User selects provider before adding to project

4. **Download Command Display**
   - Show download command for selected provider
   - Include copy-to-clipboard button
   - For Universal: Show "Auto-downloads" message
   - For Ollama: Show `ollama pull` command
   - For Lemonade: Show full `uv run lemonade-server-dev pull` command

### Backend/Type Generation

The schema changes will require:
1. Regenerating types: `cd designer && bash generate-types.sh`
2. Updating `transformCatalogToLocalGroups()` to include runtime/format info
3. Adding filtering utilities for runtime compatibility

## Example Usage in UI

### Scenario 1: User wants a small model
1. User opens "Add or change models" tab
2. Sees "Recommended: Small & Fast" section at top
3. Clicks on "Qwen3 0.6B"
4. Modal shows 3 provider options:
   - **Universal** (transformers) - Auto-downloads
   - **Ollama** (GGUF) - Copy command: `ollama pull qwen3:0.6b`
   - **Lemonade** (GGUF, llamacpp) - Copy command: `uv run...`
5. User selects "Universal" (easiest)
6. Model added to project with universal provider

### Scenario 2: User has Ollama installed
1. User filters by "Ollama (GGUF)" runtime
2. Sees only GGUF-compatible models
3. Each model shows "ollama pull" command
4. User can copy command and run it manually
5. Or click "Add" to add to project config

### Scenario 3: User wants optimal NPU performance
1. User filters by "Lemonade" runtime
2. Sees models with lemonade providers
3. Qwen3 4B marked as "Recommended for most use cases"
4. Shows full lemonade-server-dev pull command
5. Notes mention "Excellent NPU performance"

## Migration from Old Format

Old models (without runtime/format fields) will need:
1. Backward compatibility in type generation
2. Default runtime based on provider:
   - ollama provider → ollama runtime, gguf format
   - universal provider → universal runtime, transformers format
   - lemonade provider → lemonade runtime, format from recipe

## Testing Checklist

- [ ] Schema validates correctly
- [ ] Type generation works with new fields
- [ ] Old models still load (backward compat)
- [ ] Qwen3 example displays properly
- [ ] Recommended models appear first
- [ ] Runtime filtering works
- [ ] Provider selection modal works
- [ ] Download commands are correct
- [ ] Copy-to-clipboard works

## Benefits

✅ **User-friendly**: Clear guidance on which models to try first
✅ **Flexible**: Users can choose their preferred runtime
✅ **Informative**: Download commands provided for each option
✅ **Logical**: Only shows compatible runtime options
✅ **Future-proof**: Easy to add new runtimes/formats

