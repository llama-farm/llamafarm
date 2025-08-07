#!/usr/bin/env python3
"""
Chat with the REAL fine-tuned medical model using actual weights.
This script loads the fine-tuned checkpoint and allows interactive chat.
"""

import sys
import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()

class RealModelChat:
    """Chat interface for the real fine-tuned model."""
    
    def __init__(self, checkpoint_path: str = None):
        """Initialize with model checkpoint."""
        self.device = self._get_device()
        self.dtype = torch.float16 if self.device != "cpu" else torch.float32
        
        # Find checkpoint if not specified
        if checkpoint_path is None:
            checkpoint_path = self._find_latest_checkpoint()
        
        if not Path(checkpoint_path).exists():
            console.print(f"[red]Error: Checkpoint not found at {checkpoint_path}[/red]")
            sys.exit(1)
        
        self.checkpoint_path = checkpoint_path
        console.print(f"[green]Using checkpoint: {checkpoint_path}[/green]")
        
        # Load model and tokenizer
        self._load_model()
    
    def _get_device(self):
        """Determine the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _find_latest_checkpoint(self):
        """Find the latest checkpoint from training."""
        base_path = Path("fine_tuned_models/pytorch/medical_demo")
        if not base_path.exists():
            console.print("[red]No fine-tuned model found![/red]")
            console.print("Please run the training first: python demos/demo_pytorch.py")
            sys.exit(1)
        
        checkpoints = list(base_path.glob("checkpoint-*"))
        if not checkpoints:
            console.print("[red]No checkpoints found![/red]")
            sys.exit(1)
        
        # Get the latest checkpoint
        latest = max(checkpoints, key=lambda p: int(p.name.split('-')[-1]))
        return str(latest)
    
    def _load_model(self):
        """Load the fine-tuned model and tokenizer."""
        console.print("[yellow]Loading model (this may take a moment)...[/yellow]")
        
        # Load tokenizer from base model
        base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load fine-tuned model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True
        ).to(self.device)
        
        self.model.eval()
        console.print(f"[green]✓ Model loaded successfully on {self.device}[/green]")
    
    def format_prompt(self, question: str) -> str:
        """Format the prompt to match training data format."""
        # Match the exact training format
        instruction = "Provide a comprehensive and medically accurate response to the following health question. Include relevant symptoms, causes, and general guidance while emphasizing the importance of professional medical consultation."
        
        # Use the alpaca format that the model was trained on
        prompt = f"""### Instruction:
{instruction}

### Input:
{question}

### Response:"""
        return prompt
    
    def generate_response(self, question: str) -> str:
        """Generate response from the model."""
        # Try simpler prompt first
        prompt = f"Question: {question}\nAnswer:"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=256,
            truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate with simpler parameters
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.5,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the response part
        if "### Response:" in full_response:
            response = full_response.split("### Response:")[-1].strip()
        else:
            response = full_response[len(prompt):].strip()
        
        return response
    
    def chat(self):
        """Start interactive chat session."""
        console.print(Panel.fit(
            "[bold cyan]Medical AI Assistant - REAL Fine-tuned Model[/bold cyan]\n"
            "[yellow]Using actual trained weights from checkpoint[/yellow]\n"
            "[dim]Type 'quit' or 'exit' to end the conversation[/dim]",
            title="💊 Real Model Chat",
            border_style="cyan"
        ))
        
        console.print("\n[green]Model loaded and ready![/green]")
        console.print("[dim]Note: This is the ACTUAL fine-tuned model with your trained weights.[/dim]\n")
        
        while True:
            # Get user input
            question = Prompt.ask("\n[bold cyan]Your medical question[/bold cyan]")
            
            if question.lower() in ['quit', 'exit', 'q']:
                console.print("\n[yellow]Thank you for using Medical AI Assistant![/yellow]")
                break
            
            # Generate response
            console.print("\n[dim]Thinking...[/dim]")
            
            try:
                response = self.generate_response(question)
                
                console.print(Panel(
                    response,
                    title="[green]Medical AI Response[/green]",
                    border_style="green",
                    padding=(1, 2)
                ))
                
                console.print("\n[dim]⚠️ Remember: Always consult healthcare professionals for medical advice.[/dim]")
                
            except Exception as e:
                console.print(f"[red]Error generating response: {e}[/red]")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Chat with real fine-tuned medical model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint (auto-detects if not specified)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a quick test instead of interactive chat"
    )
    
    args = parser.parse_args()
    
    # Create chat interface
    chat_interface = RealModelChat(checkpoint_path=args.checkpoint)
    
    if args.test:
        # Test mode - just generate one response
        console.print("\n[cyan]Test Mode - Generating sample response[/cyan]")
        test_question = "What are the symptoms of diabetes?"
        console.print(f"\n[bold]Question:[/bold] {test_question}")
        
        response = chat_interface.generate_response(test_question)
        console.print(f"\n[bold]Response:[/bold] {response}\n")
        
        console.print("[green]✓ Test complete! The model is working.[/green]")
    else:
        # Interactive chat
        chat_interface.chat()

if __name__ == "__main__":
    main()