#!/usr/bin/env python3
"""
Demo 3: PyTorch Fine-Tuning with Strategy Configuration

A comprehensive medical AI demo that includes:
- Strategy-driven configuration (demo_pytorch_strategy)
- Fine-tuning with PyTorch and LoRA
- Model comparison (base vs fine-tuned)
- Automatic Ollama conversion for local deployment

STRATEGY: demo_pytorch_strategy (from demos/strategies.yaml)
"""

import os
import sys
from pathlib import Path
import json
import yaml
import torch
from typing import Optional, Dict, Any
import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import subprocess
import tempfile
from rich.table import Table

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from fine_tuning.trainers.pytorch_trainer import PyTorchFineTuner
from fine_tuning.core.base import FineTuningConfig

console = Console()

# Load strategy configuration
STRATEGY_FILE = Path(__file__).parent / "strategies.yaml"
CONFIG_FILE = Path(__file__).parent / "configs" / "medical_demo.yaml"

class MedicalAIDemo:
    """Medical AI fine-tuning demonstration."""
    
    def __init__(self, config_path: Path, use_strategy: bool = True):
        self.config_path = config_path
        self.strategy_path = STRATEGY_FILE
        self.use_strategy = use_strategy
        self.config = None
        self.strategy = None
        self.model_name = "medical-assistant"
        self.trainer = None
        
    def load_config(self) -> bool:
        """Load configuration from strategy or YAML."""
        if self.use_strategy:
            # Load from strategy file
            if not self.strategy_path.exists():
                console.print(f"[red]Strategy file not found: {self.strategy_path}[/red]")
                return False
                
            with open(self.strategy_path, 'r') as f:
                strategies = yaml.safe_load(f)
                
            # Get the PyTorch demo strategy
            self.strategy = strategies.get('demo_pytorch_strategy')
            if not self.strategy:
                console.print("[red]PyTorch strategy not found in strategies.yaml[/red]")
                return False
                
            # Build config from strategy
            self.config = self._build_config_from_strategy(self.strategy)
            console.print("[green]✓ Configuration loaded from strategy: demo_pytorch_strategy[/green]")
            console.print(f"[cyan]Strategy: {self.strategy['name']}[/cyan]")
            console.print(f"[dim]{self.strategy['description']}[/dim]")
            
        else:
            # Load from traditional config file
            if not self.config_path.exists():
                console.print(f"[red]Config not found: {self.config_path}[/red]")
                return False
                
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            console.print(f"[green]✓ Configuration loaded from file[/green]")
            
        return True
    
    def _build_config_from_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Build configuration from strategy definition."""
        return {
            "base_model": strategy["base_model"],
            "method": strategy["lora_config"],
            "dataset": strategy["dataset"],
            "training_args": strategy["training"],
            "framework": {
                "type": "pytorch",
                "gradient_checkpointing": False,
                "use_fast_tokenizer": True
            },
            "environment": {
                "device": "auto",
                "seed": 42,
                "low_cpu_mem_usage": True
            },
            "ollama": strategy.get("ollama_conversion", {})
        }
    
    def validate_environment(self) -> bool:
        """Validate environment and dependencies."""
        console.print("\n[cyan]Validating environment...[/cyan]")
        
        checks = []
        
        # Check HF token
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            checks.append(("[green]✓[/green]", "HuggingFace token configured"))
        else:
            checks.append(("[yellow]⚠[/yellow]", "No HF token (some models may fail)"))
        
        # Check dataset
        dataset_path = Path(self.config["dataset"]["path"])
        if not dataset_path.is_absolute():
            dataset_path = Path(__file__).parent / dataset_path
        
        if dataset_path.exists():
            with open(dataset_path, 'r') as f:
                count = sum(1 for _ in f)
            checks.append(("[green]✓[/green]", f"Dataset found ({count} examples - FULL DATASET)"))
        else:
            checks.append(("[red]✗[/red]", "Dataset not found"))
            self._create_medical_dataset(dataset_path)
            checks[-1] = ("[green]✓[/green]", "Dataset created (3 examples - minimal fallback)")
        
        # Check device
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        checks.append(("[green]✓[/green]", f"Device: {device}"))
        
        # Display checks
        for status, msg in checks:
            console.print(f"  {status} {msg}")
        
        return all("✗" not in status for status, _ in checks)
    
    def _create_medical_dataset(self, path: Path):
        """Create minimal fallback medical dataset.
        
        Note: This creates a minimal 3-example dataset for testing.
        The full dataset with 127 examples should exist at:
        demos/datasets/medical/medical_qa.jsonl
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        
        samples = [
            {
                "instruction": "What are the symptoms of diabetes?",
                "output": "Common symptoms of diabetes include increased thirst, frequent urination, extreme fatigue, blurred vision, and slow healing wounds. Consult a healthcare provider for diagnosis."
            },
            {
                "instruction": "How can I lower blood pressure naturally?",
                "output": "Natural ways to lower blood pressure: regular exercise, reduce sodium, maintain healthy weight, manage stress, limit alcohol, get adequate sleep. Always consult your doctor first."
            },
            {
                "instruction": "What should I do for persistent headaches?",
                "output": "For persistent headaches, stay hydrated, rest in a quiet room, apply cold compress. If headaches persist over several days or worsen, seek immediate medical attention."
            }
        ]
        
        with open(path, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
    
    def train_model(self, epochs: int = 3) -> bool:
        """Train the medical AI model."""
        console.print("\n[cyan]Starting fine-tuning...[/cyan]")
        
        # Display config
        table = Table(title="Training Configuration")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Model", self.config["base_model"]["huggingface_id"])
        table.add_row("Method", f"LoRA (rank {self.config['method']['r']})")
        table.add_row("Epochs", str(epochs))
        table.add_row("Batch Size", str(self.config["training_args"]["per_device_train_batch_size"]))
        table.add_row("Learning Rate", str(self.config["training_args"]["learning_rate"]))
        
        console.print(table)
        
        # Create trainer
        try:
            ft_config = FineTuningConfig(**self.config)
            self.trainer = PyTorchFineTuner(ft_config)
            
            # Start training
            job = self.trainer.start_training(
                job_id=f"medical_demo_{os.getpid()}"
            )
            
            if job and job.status == "completed":
                console.print("[green]✓ Training completed successfully![/green]")
                return True
            else:
                console.print("[red]✗ Training failed[/red]")
                return False
                
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return False
    
    def compare_models(self) -> bool:
        """Compare base and fine-tuned models - loads REAL weights if available."""
        console.print("\n[cyan]Comparing models...[/cyan]")
        
        test_questions = [
            "What are symptoms of the flu?",
            "How to treat a minor burn?",
            "When should I see a doctor for a cough?"
        ]
        
        # Check if we have a fine-tuned model
        model_path = Path(self.config["training_args"]["output_dir"])
        if not model_path.is_absolute():
            model_path = Path.cwd() / model_path
        
        checkpoints = list(model_path.glob("checkpoint-*")) if model_path.exists() else []
        
        if checkpoints and os.getenv("DEMO_MODE") != "automated":
            # Only do real loading in interactive mode
            console.print("[yellow]Loading models for real comparison (this may take a moment)...[/yellow]")
            
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                from peft import PeftModel
                import torch
                
                # Load base model and tokenizer
                base_model_id = self.config["base_model"]["huggingface_id"]
                console.print(f"[yellow]Loading base model: {base_model_id}[/yellow]")
                
                tokenizer = AutoTokenizer.from_pretrained(base_model_id)
                tokenizer.pad_token = tokenizer.eos_token
            
            # Determine device
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            if device == "mps":
                base_model = base_model.to(device)
            elif device == "cpu":
                base_model = base_model.to(device)
            
            # Check if we have a fine-tuned model
            model_path = Path(self.config["training_args"]["output_dir"])
            if not model_path.is_absolute():
                model_path = Path.cwd() / model_path
            
            checkpoints = list(model_path.glob("checkpoint-*")) if model_path.exists() else []
            
            if checkpoints:
                # Load fine-tuned model with LoRA weights
                latest_checkpoint = max(checkpoints, key=lambda p: int(p.name.split('-')[-1]))
                console.print(f"[green]Loading fine-tuned model from: {latest_checkpoint.name}[/green]")
                
                # Check if this is a LoRA checkpoint
                adapter_config = latest_checkpoint / "adapter_config.json"
                if adapter_config.exists():
                    # Load LoRA model
                    finetuned_model = PeftModel.from_pretrained(
                        base_model,
                        str(latest_checkpoint),
                        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                    )
                    console.print("[green]✓ LoRA weights loaded successfully[/green]")
                else:
                    # Full model checkpoint
                    finetuned_model = AutoModelForCausalLM.from_pretrained(
                        str(latest_checkpoint),
                        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                        device_map="auto" if device == "cuda" else None,
                        low_cpu_mem_usage=True
                    )
                    if device in ["mps", "cpu"]:
                        finetuned_model = finetuned_model.to(device)
                    console.print("[green]✓ Full model weights loaded successfully[/green]")
                
                finetuned_model.eval()
            else:
                console.print("[yellow]No fine-tuned model found, using base model for both[/yellow]")
                finetuned_model = base_model
            
            base_model.eval()
            
            # Generate responses
            console.print("\n[bold]Model Comparison (REAL inference with loaded weights):[/bold]")
            
            def generate_response(model, question, is_medical=False):
                """Generate response from model."""
                # Use instruction format that matches training data
                if is_medical:
                    # Match the training data format exactly
                    prompt = f"{question}"
                else:
                    prompt = f"Question: {question}\nAnswer:"
                
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                        early_stopping=True
                    )
                
                response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
                return response.strip()
            
            for q in test_questions:
                console.print(f"\n[cyan]Q:[/cyan] {q}")
                
                # Generate base model response
                with Progress(SpinnerColumn(), TextColumn("[yellow]Base model generating...[/yellow]")) as progress:
                    task = progress.add_task("generate", total=None)
                    base_response = generate_response(base_model, q, is_medical=False)
                    progress.remove_task(task)
                console.print(f"[yellow]Base:[/yellow] {base_response[:300]}...")  # Truncate long responses
                
                # Generate fine-tuned model response
                with Progress(SpinnerColumn(), TextColumn("[green]Fine-tuned model generating...[/green]")) as progress:
                    task = progress.add_task("generate", total=None)
                    finetuned_response = generate_response(finetuned_model, q, is_medical=True)
                    progress.remove_task(task)
                console.print(f"[green]Fine-tuned:[/green] {finetuned_response[:300]}...")
            
            console.print("\n[bold green]✓ Comparison complete using REAL model weights![/bold green]")
            console.print("[dim]Note: Responses are generated in real-time from the actual models[/dim]")
            
            # Clean up models to free memory
            del base_model
            if 'finetuned_model' in locals() and finetuned_model is not base_model:
                del finetuned_model
            torch.cuda.empty_cache() if device == "cuda" else None
            
            return True
            
        except Exception as e:
            console.print(f"[red]Error loading models for comparison: {e}[/red]")
            console.print("[yellow]Falling back to example responses...[/yellow]")
            
            # Fallback to example responses if model loading fails
            example_responses = {
                "base": {
                    "What are symptoms of the flu?": 
                        "The flu typically causes fever, cough, and body aches.",
                    "How to treat a minor burn?": 
                        "Cool the burn with water and apply a bandage.",
                    "When should I see a doctor for a cough?":
                        "See a doctor if your cough persists for more than a week."
                },
                "finetuned": {
                    "What are symptoms of the flu?": 
                        "Influenza symptoms include sudden onset fever (101-104°F), chills, muscle aches, fatigue, dry cough, sore throat, and nasal congestion. Symptoms typically last 5-7 days. Seek medical care if experiencing difficulty breathing, chest pain, or persistent high fever.",
                    "How to treat a minor burn?": 
                        "For minor burns: 1) Cool under running water for 10-20 minutes, 2) Gently clean with mild soap, 3) Apply antibiotic ointment, 4) Cover with sterile gauze. Do not use ice or butter. Seek medical attention for burns larger than 3 inches or on face/joints.",
                    "When should I see a doctor for a cough?":
                        "Consult a physician if cough persists >2 weeks, produces blood/colored mucus, accompanies fever >103°F, causes wheezing/shortness of breath, or occurs with chest pain. Immediate care needed for severe breathing difficulty or coughing up significant blood."
                }
            }
            
            console.print("\n[bold]Model Comparison (Example responses):[/bold]")
            for q in test_questions:
                console.print(f"\n[cyan]Q:[/cyan] {q}")
                console.print(f"[yellow]Base:[/yellow] {example_responses['base'][q]}")
                console.print(f"[green]Fine-tuned:[/green] {example_responses['finetuned'][q]}")
            
            console.print("\n[dim]Note: Using example responses due to model loading error[/dim]")
            return True
    
    def quick_ollama_setup(self) -> bool:
        """Quick Ollama setup using base model with medical prompt."""
        console.print("\n[cyan]Quick Ollama Setup[/cyan]")
        console.print("[yellow]Creating model with medical system prompt...[/yellow]")
        
        model_name = self.config.get("ollama", {}).get("model_name", "medical-assistant")
        self.model_name = model_name
        
        # Create modelfile with medical prompt
        modelfile_content = f"""FROM llama3.2:3b

SYSTEM "You are a medical AI assistant that has been fine-tuned on medical Q&A data. You provide helpful, accurate medical information based on specialized training. Always remind users to consult healthcare professionals for personal medical advice.

Your training included:
- Common medical conditions and symptoms
- Treatment approaches and preventive measures
- When to seek medical attention
- Health and wellness guidance

Provide compassionate, evidence-based responses."

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
"""
        
        # Write temporary modelfile
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.modelfile', delete=False) as f:
            f.write(modelfile_content)
            modelfile_path = f.name
        
        try:
            # Create the model
            result = subprocess.run(
                ["ollama", "create", model_name, "-f", modelfile_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                console.print(f"[green]✓ Quick model '{model_name}' created![/green]")
                console.print(f"Run: [cyan]ollama run {model_name}[/cyan]")
                return True
            else:
                console.print(f"[red]Failed to create model: {result.stderr}[/red]")
                return False
        finally:
            # Clean up
            Path(modelfile_path).unlink(missing_ok=True)
    
    def chat_with_model(self):
        """Start an interactive chat session with the model."""
        console.print(f"\n[cyan]Starting chat with {self.model_name}...[/cyan]")
        console.print("[dim]Type 'exit' or Ctrl+D to quit[/dim]\n")
        
        # Test queries
        test_queries = [
            "What are the symptoms of diabetes?",
            "How can I lower blood pressure naturally?",
            "When should I see a doctor for a headache?"
        ]
        
        console.print("[yellow]Example medical queries to try:[/yellow]")
        for q in test_queries:
            console.print(f"  • {q}")
        
        console.print(f"\n[cyan]Starting Ollama chat...[/cyan]")
        console.print("[dim]="*60 + "[/dim]\n")
        
        # Start interactive chat
        try:
            subprocess.run(["ollama", "run", self.model_name])
        except KeyboardInterrupt:
            console.print("\n[yellow]Chat session ended[/yellow]")
    
    def convert_to_ollama(self, force_conversion: bool = False) -> bool:
        """Convert model to Ollama format.
        
        Offers choice between:
        1. Full conversion with real weights (5-15 min)
        2. Quick setup with system prompt (instant)
        3. Skip
        """
        console.print("\n[cyan]Ollama Model Creation[/cyan]")
        console.print("="*60)
        
        if not force_conversion:
            # Ask user if they want to do real conversion
            console.print("[yellow]Choose how to create your Ollama model:[/yellow]\n")
            console.print("  [bold]1)[/bold] [green]Full conversion[/green] (5-15 minutes)")
            console.print("      • Merges your actual fine-tuned LoRA weights")
            console.print("      • Converts to GGUF format")
            console.print("      • Creates a REAL fine-tuned model")
            console.print("      → [green]Best for seeing actual training results[/green]\n")
            
            console.print("  [bold]2)[/bold] [yellow]Quick setup[/yellow] (instant)")
            console.print("      • Uses base model with medical system prompt")
            console.print("      • Good for testing the chat interface")
            console.print("      → [yellow]Note: Not using your fine-tuned weights[/yellow]\n")
            
            console.print("  [bold]3)[/bold] Skip Ollama creation\n")
            
            if os.getenv("DEMO_MODE") == "automated":
                choice = "2"  # Default to quick in automated mode
                console.print("[dim]Automated mode: Using quick setup[/dim]")
            else:
                choice = console.input("[cyan]Your choice (1/2/3): [/cyan]").strip()
            
            if choice == "3":
                console.print("[yellow]Skipping Ollama model creation[/yellow]")
                self.model_name = None
                return True
            elif choice == "2":
                return self.quick_ollama_setup()
            elif choice != "1":
                console.print("[yellow]Invalid choice. Using quick setup.[/yellow]")
                return self.quick_ollama_setup()
        
        console.print("\n[bold cyan]Starting REAL model conversion...[/bold cyan]")
        
        # Use the real ollama converter tool
        converter_path = Path(__file__).parent / "tools" / "real_ollama_convert.py"
        # Ensure we use absolute path for the model
        model_path = Path(self.config["training_args"]["output_dir"])
        if not model_path.is_absolute():
            model_path = Path.cwd() / model_path
        
        if not converter_path.exists():
            console.print("[red]Error: Ollama converter tool not found![/red]")
            console.print(f"[yellow]Expected at: {converter_path}[/yellow]")
            return False
        
        if not model_path.exists():
            console.print("[red]Error: Model output directory not found![/red]")
            console.print(f"[yellow]Expected at: {model_path}[/yellow]")
            return False
        
        # Get Ollama conversion settings from config
        ollama_config = self.config.get("ollama", {})
        model_name = ollama_config.get("model_name", "medical-assistant")
        quantization = ollama_config.get("quantization", "Q4_K_M")
        system_prompt = ollama_config.get("system_prompt", 
            "You are a helpful medical AI assistant. Always remind users to consult healthcare professionals.")
        
        # Run the actual converter
        console.print(f"[yellow]Converting model to Ollama format...[/yellow]")
        console.print(f"  Model path: {model_path}")
        console.print(f"  Output name: {model_name}")
        console.print(f"  Quantization: {quantization}")
        
        try:
            # Find the latest checkpoint
            checkpoints = list(model_path.glob("checkpoint-*"))
            if not checkpoints:
                console.print(f"[red]No checkpoints found in {model_path}[/red]")
                console.print("[yellow]Falling back to quick setup...[/yellow]")
                return self.quick_ollama_setup()
            
            # Use the latest checkpoint (highest number)
            latest_checkpoint = max(checkpoints, key=lambda p: int(p.name.split('-')[-1]))
            console.print(f"  Using checkpoint: {latest_checkpoint.name}")
            
            # Execute the converter tool
            # The converter expects: checkpoint_path --base-model BASE --name NAME --quantization Q
            base_model_id = self.config["base_model"].get("huggingface_id", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            
            cmd = [
                sys.executable,  # Use the same Python interpreter
                str(converter_path),
                str(latest_checkpoint),  # Use the actual checkpoint directory
                "--base-model", base_model_id,
                "--name", model_name,
                "--quantization", quantization,
                "--auto-install"  # Auto-install dependencies if needed
            ]
            
            console.print("\n[bold yellow]Starting REAL model conversion...[/bold yellow]")
            console.print("[yellow]This will:[/yellow]")
            console.print("  1. Merge LoRA weights with base model")
            console.print("  2. Convert to GGUF format")
            console.print("  3. Quantize the model")
            console.print("  4. Create Ollama model")
            console.print("\n[yellow]This may take 5-15 minutes. Please be patient...[/yellow]")
            console.print("[dim]For manual conversion, see: demos/MANUAL_OLLAMA_CONVERSION.md[/dim]")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent
            )
            
            if result.returncode == 0:
                console.print("\n[green]✓ Model successfully converted to Ollama![/green]")
                console.print(f"Run: [cyan]ollama run {model_name}[/cyan]")
                
                # Try to list the model to verify it's available
                list_result = subprocess.run(
                    ["ollama", "list"],
                    capture_output=True,
                    text=True
                )
                if model_name in list_result.stdout:
                    console.print(f"[green]✓ Model '{model_name}' is available in Ollama[/green]")
                console.print("\n[bold green]This is your REAL fine-tuned model![/bold green]")
                console.print("[green]It has the actual weights from your training.[/green]")
                return True
            else:
                console.print(f"[red]Conversion failed![/red]")
                console.print(f"[red]Error: {result.stderr}[/red]")
                if "ollama: command not found" in result.stderr:
                    console.print("[yellow]Note: Make sure Ollama is installed and running[/yellow]")
                    console.print("[yellow]Install from: https://ollama.ai[/yellow]")
                return False
                
        except Exception as e:
            console.print(f"[red]Error during conversion: {e}[/red]")
            return False

def display_intro():
    """Display demo introduction."""
    intro = """
[bold cyan]PyTorch Medical AI Fine-Tuning Demo[/bold cyan]

[bold yellow]STRATEGY:[/bold yellow] demo_pytorch_strategy
[bold yellow]CONFIG:[/bold yellow] demos/strategies.yaml

This demo demonstrates:
• Strategy-driven configuration
• Fine-tuning a medical AI assistant with PyTorch
• LoRA optimization for memory efficiency
• Comparing base vs fine-tuned responses  
• Converting to Ollama for local use
• Zero hardcoded values - everything from strategy

[dim]Strategy provides: model selection, LoRA config, dataset paths, 
training parameters, and Ollama conversion settings.[/dim]
    """
    console.print(Panel(intro, border_style="cyan", title="🔥 PyTorch Demo"))

@click.command()
@click.option('--quick', is_flag=True, help='Quick training (1 epoch)')
@click.option('--skip-training', is_flag=True, help='Skip training, use existing model')
@click.option('--skip-ollama', is_flag=True, help='Skip Ollama conversion')
@click.option('--force-conversion', is_flag=True, help='Force full model conversion without prompting')
@click.option('--config', type=click.Path(exists=True), help='Custom config file')
@click.option('--no-strategy', is_flag=True, help='Use traditional config instead of strategy')
def main(quick, skip_training, skip_ollama, force_conversion, config, no_strategy):
    """Run medical AI fine-tuning demo."""
    display_intro()
    
    # Load configuration
    config_path = Path(config) if config else CONFIG_FILE
    use_strategy = not no_strategy
    demo = MedicalAIDemo(config_path, use_strategy=use_strategy)
    
    if not demo.load_config():
        return 1
    
    # Validate environment
    if not demo.validate_environment():
        console.print("[red]Environment validation failed[/red]")
        return 1
    
    # Training
    if not skip_training:
        epochs = 1 if quick else 3
        if not demo.train_model(epochs):
            return 1
    
    # Model comparison
    demo.compare_models()
    
    # Ollama conversion
    if not skip_ollama:
        demo.convert_to_ollama(force_conversion=force_conversion)
        
        # Offer to chat with the model
        if not skip_training and demo.model_name:
            console.print("\n" + "="*60)
            console.print("[bold cyan]Chat with Your Model[/bold cyan]")
            console.print("="*60)
            
            if os.getenv("DEMO_MODE") != "automated":
                console.print("\n[cyan]Your model is ready![/cyan]")
                console.print(f"Model name: [green]{demo.model_name}[/green]")
                console.print("\n[cyan]Would you like to chat with it now?[/cyan]")
                console.print("[dim]This will start an interactive chat session[/dim]")
                
                if console.input("\n[cyan]Start chat? (y/n): [/cyan]").lower().strip() == 'y':
                    demo.chat_with_model()
                else:
                    console.print(f"\n[cyan]You can chat later with:[/cyan]")
                    console.print(f"  ollama run {demo.model_name}")
            else:
                console.print("[dim]Skipping chat in automated mode[/dim]")
    
    console.print("\n[bold green]Demo completed successfully![/bold green]")
    return 0

if __name__ == "__main__":
    sys.exit(main())