"""Quick standalone test of KV cache save/load on Llama."""
import sys
sys.path.insert(0, "/Users/robthelen/clawd/projects/llamafarm-core/packages/llamafarm-llama/src")

from llamafarm_llama import Llama

model_path = "/Users/robthelen/.cache/huggingface/hub/models--Qwen--Qwen3-8B-GGUF/snapshots/7c41481f57cb95916b40956ab2f0b139b296d974/Qwen3-8B-Q4_K_M.gguf"

print("Loading model...")
llm = Llama(model_path=model_path, n_ctx=4096, n_gpu_layers=-1)
print("Model loaded.")

# Tokenize a system prompt
prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
tokens = llm.tokenize(prompt, add_special=False, parse_special=True)
print(f"Tokenized: {len(tokens)} tokens")

# Decode to populate KV cache
llm.reset()
ok = llm._decode_batch(tokens)
print(f"Decode OK: {ok}")

# Get state size
size = llm.state_seq_get_size(0)
print(f"State seq size: {size} bytes ({size/1024/1024:.1f}MB)")

# Save state
print("Saving state...")
data = llm.state_seq_save(0)
print(f"Saved: {len(data)} bytes ({len(data)/1024/1024:.1f}MB)")

# Reset and verify empty
llm.reset()
print("Reset done.")

# Restore state
print("Restoring state...")
consumed = llm.state_seq_load(data, 0)
print(f"Restored: {consumed} bytes consumed")

# Generate a token to verify it works
llm._create_sampler(temperature=0.7, top_p=0.95, top_k=40)
tok = llm._sample_token()
print(f"Sampled token after restore: {tok}")

print("\nSUCCESS: KV cache save/load works!")
