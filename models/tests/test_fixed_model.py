#!/usr/bin/env python3
"""
Simple test of the fixed medical model.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

def test_model():
    print("Testing fixed medical model...")
    
    # Model paths
    base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    adapter_path = Path("fine_tuned_models/pytorch/medical_fixed")
    
    if not adapter_path.exists():
        print(f"❌ Model not found at {adapter_path}")
        print("Train it with: uv run python demos/train_with_fixed_strategy.py --quick")
        return
    
    print(f"✓ Found adapter at {adapter_path}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    print("Loading base model...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    ).to(device)
    
    # Load LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()
    
    print(f"✓ Model loaded on {device}")
    
    # Test questions
    test_questions = [
        "What are the symptoms of diabetes?",
        "How should I treat a headache?",
        "When should I see a doctor for a fever?"
    ]
    
    print("\n" + "="*50)
    print("Testing model responses:")
    print("="*50)
    
    for question in test_questions:
        # Format prompt properly
        prompt = f"""<|system|>
You are a helpful medical AI assistant. Provide accurate, detailed medical information while always reminding users to consult healthcare professionals.</s>
<|user|>
{question}</s>
<|assistant|>"""
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        print(f"\nQ: {question}")
        print("A: ", end="", flush=True)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode response (only the generated part)
        response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        print(response)
    
    print("\n" + "="*50)
    print("✓ Test complete!")
    
    # Interactive mode
    print("\nWant to chat? (y/n): ", end="")
    if input().lower() == 'y':
        print("\nInteractive mode (type 'quit' to exit):")
        while True:
            question = input("\nYou: ")
            if question.lower() in ['quit', 'exit']:
                break
            
            prompt = f"""<|system|>
You are a helpful medical AI assistant.</s>
<|user|>
{question}</s>
<|assistant|>"""
            
            inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
            print(f"AI: {response}")

if __name__ == "__main__":
    test_model()