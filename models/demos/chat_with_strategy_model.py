#!/usr/bin/env python3
"""
Chat with a model trained using any strategy configuration.
Loads the strategy info and uses proper generation parameters.
"""

import sys
import yaml
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()

class StrategyModelChat:
    """Chat interface for strategy-trained models."""
    
    def __init__(self, strategy_name: str = None, model_path: str = None):
        """Initialize with strategy or model path."""
        
        if model_path:
            self.model_path = Path(model_path)
        elif strategy_name:
            # Load strategy to get model path
            self.load_strategy(strategy_name)
            self.model_path = Path(self.strategy['training_args']['output_dir'])
        else:
            # Try to find the fixed model
            self.model_path = Path("fine_tuned_models/pytorch/medical_fixed")
        
        if not self.model_path.exists():
            console.print(f"[red]Model not found at: {self.model_path}[/red]")
            console.print("\n[yellow]Available models:[/yellow]")
            self.list_available_models()
            sys.exit(1)
        
        # Load strategy info if available
        strategy_info_path = self.model_path / "strategy_info.yaml"
        if strategy_info_path.exists():
            with open(strategy_info_path, 'r') as f:
                info = yaml.safe_load(f)
                self.strategy = info.get('strategy', {})
                console.print(f"[green]✓ Loaded strategy: {info.get('strategy_name')}[/green]")
        else:
            console.print("[yellow]No strategy info found, using defaults[/yellow]")
            self.strategy = self.get_default_strategy()
        
        self.load_model()
    
    def load_strategy(self, strategy_name: str):
        """Load strategy from strategies.yaml."""
        strategy_file = Path("demos/strategies.yaml")
        
        with open(strategy_file, 'r') as f:
            strategies = yaml.safe_load(f)
        
        if strategy_name not in strategies:
            console.print(f"[red]Strategy '{strategy_name}' not found![/red]")
            sys.exit(1)
        
        self.strategy = strategies[strategy_name]
    
    def get_default_strategy(self):
        """Get default strategy for models without strategy info."""
        return {
            'generation': {
                'max_new_tokens': 150,
                'temperature': 0.7,
                'top_p': 0.9,
                'repetition_penalty': 1.1,
                'do_sample': True
            },
            'prompt_template': {
                'system': "You are a helpful AI assistant.",
                'format': "<|system|>\n{system}</s>\n<|user|>\n{input}</s>\n<|assistant|>"
            }
        }
    
    def list_available_models(self):
        """List available trained models."""
        base_path = Path("fine_tuned_models")
        if base_path.exists():
            for model_dir in base_path.rglob("*/"):
                if (model_dir / "config.json").exists():
                    console.print(f"  - {model_dir}")
    
    def load_model(self):
        """Load the model and tokenizer."""
        console.print(f"[cyan]Loading model from: {self.model_path}[/cyan]")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Determine device
        if torch.cuda.is_available():
            self.device = "cuda"
            dtype = torch.float16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            dtype = torch.float16
        else:
            self.device = "cpu"
            dtype = torch.float32
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        ).to(self.device)
        
        self.model.eval()
        console.print(f"[green]✓ Model loaded on {self.device}[/green]")
    
    def generate_response(self, question: str) -> str:
        """Generate response using strategy parameters."""
        # Format prompt using strategy template
        prompt_template = self.strategy.get('prompt_template', {})
        
        if 'format' in prompt_template and '{input}' in prompt_template['format']:
            # Use template format
            prompt = prompt_template['format'].format(
                system=prompt_template.get('system', 'You are a helpful AI assistant.'),
                input=question,
                output=""  # Empty for generation
            ).rstrip()  # Remove trailing output placeholder
        else:
            # Fallback format
            prompt = f"""<|system|>
{prompt_template.get('system', 'You are a helpful AI assistant.')}</s>
<|user|>
{question}</s>
<|assistant|>"""
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get generation config from strategy
        gen_config = self.strategy.get('generation', {})
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=gen_config.get('max_new_tokens', 150),
                temperature=gen_config.get('temperature', 0.7),
                do_sample=gen_config.get('do_sample', True),
                top_p=gen_config.get('top_p', 0.9),
                top_k=gen_config.get('top_k', 50),
                repetition_penalty=gen_config.get('repetition_penalty', 1.1),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def chat(self):
        """Interactive chat session."""
        # Display info panel
        strategy_name = self.strategy.get('name', 'Unknown Strategy')
        strategy_desc = self.strategy.get('description', 'Strategy-trained model')
        
        console.print(Panel.fit(
            f"[bold cyan]{strategy_name}[/bold cyan]\n"
            f"[yellow]{strategy_desc}[/yellow]\n"
            f"[dim]Model: {self.model_path.name}[/dim]",
            title="🤖 Strategy Model Chat",
            border_style="cyan"
        ))
        
        console.print("\n[bold]Chat with your trained model![/bold]")
        console.print("[dim]Type 'quit' to exit[/dim]\n")
        
        while True:
            # Get user input
            question = Prompt.ask("\n[bold cyan]You[/bold cyan]")
            
            if question.lower() in ['quit', 'exit', 'q']:
                console.print("\n[yellow]Goodbye! 👋[/yellow]")
                break
            
            # Generate response
            console.print("\n[dim]Thinking...[/dim]", end="\r")
            
            try:
                response = self.generate_response(question)
                
                # Clear "Thinking..." and show response
                console.print(" " * 20, end="\r")  # Clear line
                console.print(Panel(
                    response,
                    title="[green]AI Response[/green]",
                    border_style="green",
                    padding=(1, 2)
                ))
                
                # Show disclaimer if medical strategy
                if 'medical' in self.strategy.get('name', '').lower():
                    console.print("\n[dim]⚠️ Always consult healthcare professionals for medical advice[/dim]")
                
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Chat with strategy-trained models"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        help="Strategy name (loads model from strategy's output_dir)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Direct path to model directory"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode - generate one response and exit"
    )
    
    args = parser.parse_args()
    
    # Create chat interface
    chat = StrategyModelChat(
        strategy_name=args.strategy,
        model_path=args.model_path
    )
    
    if args.test:
        # Test mode
        console.print("\n[cyan]Test Mode[/cyan]")
        test_q = "What are the symptoms of a cold?"
        console.print(f"\n[bold]Q:[/bold] {test_q}")
        
        response = chat.generate_response(test_q)
        console.print(f"\n[bold]A:[/bold] {response}\n")
        
        console.print("[green]✓ Model is working![/green]")
    else:
        # Interactive chat
        chat.chat()

if __name__ == "__main__":
    main()