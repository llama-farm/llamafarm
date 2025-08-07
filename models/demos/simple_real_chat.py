#!/usr/bin/env python3
"""
Simple chat with the REAL fine-tuned model.
This uses the actual trained weights from your checkpoint.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

def chat_with_real_model():
    """Load and chat with the real fine-tuned model."""
    
    print("🏥 Medical AI Chat - Using REAL Fine-tuned Weights")
    print("=" * 50)
    
    # Find the checkpoint
    checkpoint = Path("fine_tuned_models/pytorch/medical_demo/checkpoint-24")
    if not checkpoint.exists():
        print(f"❌ Model not found at {checkpoint}")
        print("Please train the model first with: python demos/demo_pytorch.py")
        return
    
    print(f"✓ Found model at: {checkpoint}")
    print("Loading model (this will take a moment)...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Determine device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    
    # Load the REAL fine-tuned model
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint),
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    
    print(f"✓ Model loaded on {device}")
    print("\n" + "=" * 50)
    print("Chat with your fine-tuned medical AI!")
    print("Type 'quit' to exit")
    print("=" * 50 + "\n")
    
    while True:
        # Get user input
        question = input("\n👤 You: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye! 👋")
            break
        
        # Format prompt
        prompt = f"<|system|>You are a helpful medical AI assistant.</s><|user|>{question}</s><|assistant|>"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response
        print("\n🤖 AI: ", end="", flush=True)
        
        with torch.no_grad():
            # Stream the response token by token
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                streamer=None  # Could add streaming here
            )
        
        # Decode and print response
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        print(response)
        
        print("\n" + "-" * 50)
        print("⚠️  Remember: Always consult real healthcare professionals!")

if __name__ == "__main__":
    chat_with_real_model()