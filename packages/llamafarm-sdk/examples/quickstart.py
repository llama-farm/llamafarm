"""LlamaFarm SDK quickstart — run against a live server."""

from llamafarm import LlamaFarm

# Connect (auto-discovers localhost:14345)
lf = LlamaFarm()
print(f"Connected: {lf}")

# Health check
print("Health:", lf.health())

# Create a project
project = lf.create_project(
    "default",
    "quickstart-demo",
    model="unsloth/Qwen3-1.7B-GGUF:Q4_K_M",
    system_prompt="You are a concise, helpful assistant.",
    temperature=0.7,
)
print(f"Created: {project}")

# Chat
response = project.chat("What is LlamaFarm?")
print(f"Response: {response.text}")

# Streaming
print("Streaming: ", end="", flush=True)
for chunk in project.chat_stream("Tell me a short joke"):
    if chunk.delta_content:
        print(chunk.delta_content, end="", flush=True)
print()

# RAG query (if configured)
try:
    results = project.rag.query("How do I configure models?")
    for r in results.results:
        print(f"  [{r.score:.2f}] {r.content[:80]}...")
except Exception as e:
    print(f"RAG not configured: {e}")

# List projects
for p in lf.projects("default"):
    print(f"  Project: {p.name} ({p.id})")

# Clean up
project.delete()
print("Deleted project")
