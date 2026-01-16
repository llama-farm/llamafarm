# Transformers Model Patterns Checklist

HuggingFace Transformers model loading, tokenization, and inference patterns.

---

## Category: Model Loading

### Use trust_remote_code=True Consistently

**What to check**: Enable trust_remote_code for custom model architectures

**Search pattern**:
```bash
rg "trust_remote_code=" --type py runtimes/universal/models/
```

**Pass criteria**:
- All AutoModel/AutoTokenizer calls include trust_remote_code=True
- Enables loading of custom model code (required for many popular models)

**Severity**: High

**Good pattern**:
```python
self.tokenizer = AutoTokenizer.from_pretrained(
    self.model_id,
    trust_remote_code=True,
    token=self.token,
)

self.model = AutoModelForCausalLM.from_pretrained(
    self.model_id,
    trust_remote_code=True,
    token=self.token,
)
```

---

### Load Config Before Model for Capability Detection

**What to check**: Use AutoConfig to detect model capabilities before loading

**Search pattern**:
```bash
rg "AutoConfig\.from_pretrained" --type py runtimes/universal/models/
```

**Pass criteria**:
- Config loaded first to detect max_length, Flash Attention support
- Avoids loading model with incompatible settings

**Severity**: Medium

**Good pattern** (from models/encoder_model.py):
```python
async def load(self) -> None:
    # Load config first to detect capabilities
    config = AutoConfig.from_pretrained(
        self.model_id, trust_remote_code=True, token=self.token
    )

    # Detect max sequence length
    self._detected_max_length = self._detect_max_length(config)

    # Check Flash Attention support
    if self._supports_flash_attention(config):
        model_kwargs["attn_implementation"] = "flash_attention_2"
```

---

### Use Correct AutoModel Class for Task

**What to check**: Select the right AutoModel variant for the task

**Search pattern**:
```bash
rg "AutoModel|AutoModelFor" --type py runtimes/universal/models/
```

**Pass criteria**:
- `AutoModelForCausalLM` for text generation
- `AutoModel` for embeddings (no classification head)
- `AutoModelForSequenceClassification` for classification/reranking
- `AutoModelForTokenClassification` for NER

**Severity**: High

**Good pattern**:
```python
if self.task == "classification":
    self.model = AutoModelForSequenceClassification.from_pretrained(...)
elif self.task == "reranking":
    self.model = AutoModelForSequenceClassification.from_pretrained(...)
elif self.task == "ner":
    self.model = AutoModelForTokenClassification.from_pretrained(...)
else:  # embedding
    self.model = AutoModel.from_pretrained(...)
```

---

### Set Model to Eval Mode for Inference

**What to check**: Call model.eval() after loading for inference

**Search pattern**:
```bash
rg "\.eval\(\)" --type py runtimes/universal/models/
```

**Pass criteria**:
- model.eval() called after loading
- Disables dropout and batch normalization training behavior

**Severity**: High

**Good pattern**:
```python
if self.model is not None:
    self.model = self.model.to(self.device)
    self.model.eval()  # Critical for inference
```

---

## Category: Tokenization

### Use Tokenizer's Chat Template

**What to check**: Apply model-specific chat template for message formatting

**Search pattern**:
```bash
rg "apply_chat_template" --type py runtimes/universal/models/
```

**Pass criteria**:
- Use tokenizer.apply_chat_template() when available
- Fallback to simple format if not available
- Set add_generation_prompt=True for assistant response

**Severity**: High

**Good pattern**:
```python
def format_messages(self, messages: list[dict]) -> str:
    if self.tokenizer and hasattr(self.tokenizer, "apply_chat_template"):
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass

    # Fallback to simple concatenation
    return "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
```

---

### Configure Tokenizer Padding and Truncation

**What to check**: Set padding, truncation, and max_length consistently

**Search pattern**:
```bash
rg "padding=|truncation=|max_length=" --type py runtimes/universal/models/
```

**Pass criteria**:
- `padding=True` for batch processing
- `truncation=True` to handle long inputs
- `max_length` set from model's detected capability
- `return_tensors="pt"` for PyTorch tensors

**Severity**: High

**Good pattern**:
```python
encoded = self.tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=self.max_length,
    return_tensors="pt",
)
encoded = {k: v.to(self.device) for k, v in encoded.items()}
```

---

### Handle Tokenizer Output Properly

**What to check**: Move tokenizer output to device correctly

**Search pattern**:
```bash
rg "\.to\(self\.device\)" --type py runtimes/universal/models/
```

**Pass criteria**:
- All tensor values moved to device
- Use dict comprehension for clean handling
- Remove non-tensor fields before model forward pass

**Severity**: High

**Good pattern**:
```python
# For embeddings - keep all fields
encoded = {k: v.to(self.device) for k, v in encoded.items()}

# For NER - remove offset_mapping before model call
offset_mapping = encoded.pop("offset_mapping")[0].tolist()
encoded = {k: v.to(self.device) for k, v in encoded.items()}
```

---

### Detect Max Sequence Length from Config

**What to check**: Extract max_length from model config attributes

**Search pattern**:
```bash
rg "max_position_embeddings|max_seq_length|n_positions" --type py runtimes/universal/
```

**Pass criteria**:
- Check multiple config attributes for compatibility
- Handle extended context models (ModernBERT: 8192)
- Default to 512 for classic BERT

**Severity**: Medium

**Good pattern**:
```python
def _detect_max_length(self, config: AutoConfig) -> int:
    # Check for known extended context models
    for prefix, length in self.EXTENDED_CONTEXT_MODELS.items():
        if self.model_id.startswith(prefix):
            return length

    # Try config attributes
    if hasattr(config, "max_position_embeddings"):
        return config.max_position_embeddings
    if hasattr(config, "max_seq_length"):
        return config.max_seq_length
    if hasattr(config, "n_positions"):
        return config.n_positions

    return 512  # Default for classic BERT
```

---

## Category: Text Generation

### Use pad_token_id from Tokenizer

**What to check**: Set pad_token_id in generate() to avoid warnings

**Search pattern**:
```bash
rg "pad_token_id=" --type py runtimes/universal/models/
```

**Pass criteria**:
- pad_token_id set to eos_token_id (common pattern)
- Prevents "Setting pad_token_id to eos_token_id" warning

**Severity**: Low

**Good pattern**:
```python
outputs = self.model.generate(
    **inputs,
    max_new_tokens=max_tokens,
    temperature=temperature,
    pad_token_id=self.tokenizer.eos_token_id,
)
```

---

### Decode Only New Tokens

**What to check**: Skip input tokens when decoding generation output

**Search pattern**:
```bash
rg "skip_special_tokens" --type py runtimes/universal/models/
```

**Pass criteria**:
- Slice output to skip input_ids: `outputs[0][inputs.input_ids.shape[1]:]`
- Use skip_special_tokens=True to clean output

**Severity**: Medium

**Good pattern**:
```python
# Generate
outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)

# Decode only the NEW tokens (skip input)
generated_text = self.tokenizer.decode(
    outputs[0][inputs.input_ids.shape[1]:],
    skip_special_tokens=True
)
return generated_text.strip()
```

---

### Use TextIteratorStreamer for Streaming

**What to check**: Stream tokens using TextIteratorStreamer

**Search pattern**:
```bash
rg "TextIteratorStreamer" --type py runtimes/universal/models/
```

**Pass criteria**:
- TextIteratorStreamer instantiated with tokenizer
- skip_prompt=True to not re-emit input
- Generation runs in separate thread

**Severity**: Medium

**Good pattern**:
```python
from transformers import TextIteratorStreamer
from threading import Thread

streamer = TextIteratorStreamer(
    cast(AutoTokenizer, self.tokenizer),
    skip_prompt=True,
    skip_special_tokens=True,
)

thread = Thread(target=self.model.generate, kwargs={**inputs, "streamer": streamer})
thread.start()

for text in streamer:
    yield text

thread.join()
```

---

## Category: Embeddings

### Mean Pooling for Embeddings

**What to check**: Apply mean pooling with attention mask weighting

**Search pattern**:
```bash
rg "_mean_pooling|mean.*pooling" --type py runtimes/universal/models/
```

**Pass criteria**:
- Expand attention mask to match embedding dimensions
- Sum weighted by attention mask
- Clamp denominator to avoid division by zero

**Severity**: High

**Good pattern**:
```python
def _mean_pooling(self, model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
```

---

### L2 Normalize Embeddings

**What to check**: Normalize embeddings for cosine similarity

**Search pattern**:
```bash
rg "F\.normalize|normalize.*p=2" --type py runtimes/universal/models/
```

**Pass criteria**:
- Use F.normalize(embeddings, p=2, dim=1)
- Default to normalize=True for embedding endpoints

**Severity**: Medium

**Good pattern**:
```python
if normalize:
    embeddings = F.normalize(embeddings, p=2, dim=1)
return embeddings.cpu().tolist()
```

---

## Category: Flash Attention

### Check Flash Attention Compatibility

**What to check**: Enable Flash Attention 2 only when supported

**Search pattern**:
```bash
rg "flash_attention|attn_implementation" --type py runtimes/universal/
```

**Pass criteria**:
- Only enable on CUDA devices
- Check torch version >= 2.0
- Check if flash_attn package is installed
- Set attn_implementation="flash_attention_2" in model kwargs

**Severity**: Medium

**Good pattern**:
```python
def _supports_flash_attention(self, config: AutoConfig) -> bool:
    if self.device != "cuda":
        return False

    torch_version = tuple(map(int, torch.__version__.split(".")[:2]))
    if torch_version < (2, 0):
        return False

    try:
        import flash_attn
        return True
    except ImportError:
        return False

# In load():
if self._use_flash_attention and self._supports_flash_attention(config):
    model_kwargs["attn_implementation"] = "flash_attention_2"
```

---

## Category: GGUF Models

### Use llama-cpp-python's Chat Template

**What to check**: Use create_chat_completion() for GGUF models

**Search pattern**:
```bash
rg "create_chat_completion" --type py runtimes/universal/
```

**Pass criteria**:
- Use create_chat_completion() not manual prompt formatting
- GGUF metadata contains proper chat template
- Essential for models with special tokens (Qwen, Llama)

**Severity**: High

**Good pattern**:
```python
# GGUF models use embedded chat template
result = self.llama.create_chat_completion(
    messages=messages,
    max_tokens=max_tokens,
    temperature=temperature,
    stream=False,
)
return result["choices"][0]["message"]["content"]
```

---

### Configure GPU Layers for llama-cpp

**What to check**: Set n_gpu_layers based on device

**Search pattern**:
```bash
rg "n_gpu_layers" --type py runtimes/universal/
```

**Pass criteria**:
- n_gpu_layers=-1 for GPU (all layers)
- n_gpu_layers=0 for CPU only

**Severity**: High

**Good pattern**:
```python
if self.device != "cpu":
    n_gpu_layers = -1  # All layers on GPU
else:
    n_gpu_layers = 0  # CPU only

self.llama = Llama(
    model_path=gguf_path,
    n_ctx=context_size,
    n_gpu_layers=n_gpu_layers,
)
```
