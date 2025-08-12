#!/usr/bin/env python3
"""
Quick test to load and use the fine-tuned medical model.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def test_finetuned_model():
    """Test the fine-tuned medical model."""
    print("Loading fine-tuned medical model...")
    
    # Load base model
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, "./fine_tuned_models/medical/final_model/")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_models/medical/final_model/")
    
    print("✅ Model loaded successfully!")
    
    # Test with a medical question
    question = "What are the symptoms of diabetes?"
    
    print(f"\nQuestion: {question}")
    print("Generating response...")
    
    # Tokenize input
    inputs = tokenizer(question, return_tensors="pt")
    
    # Move inputs to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nResponse: {response}")
    
    return True

if __name__ == "__main__":
    try:
        test_finetuned_model()
        print("\n✅ Test completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")