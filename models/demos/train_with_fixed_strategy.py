#!/usr/bin/env python3
"""
Train the medical model using the fixed strategy configuration.
Everything is driven by the strategy - NO HARDCODED VALUES.
"""

import os
import sys
import json
import yaml
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

class StrategyDrivenTrainer:
    """Train model using strategy configuration - zero hardcoding."""
    
    def __init__(self, strategy_name: str = "medical_training_fixed"):
        """Initialize with strategy name."""
        self.strategy_name = strategy_name
        self.strategy = None
        self.load_strategy()
        
    def load_strategy(self):
        """Load strategy from YAML configuration."""
        strategy_file = Path("demos/strategies.yaml")
        
        if not strategy_file.exists():
            console.print(f"[red]Strategy file not found: {strategy_file}[/red]")
            sys.exit(1)
            
        with open(strategy_file, 'r') as f:
            strategies = yaml.safe_load(f)
            
        if self.strategy_name not in strategies:
            console.print(f"[red]Strategy '{self.strategy_name}' not found![/red]")
            console.print(f"Available strategies: {list(strategies.keys())}")
            sys.exit(1)
            
        self.strategy = strategies[self.strategy_name]
        console.print(f"[green]✓ Loaded strategy: {self.strategy['name']}[/green]")
        
    def load_and_prepare_data(self):
        """Load dataset based on strategy configuration."""
        dataset_config = self.strategy['dataset']
        dataset_path = Path(dataset_config['path'])
        
        console.print(f"[cyan]Loading dataset from: {dataset_path}[/cyan]")
        
        # Load the data
        data = []
        with open(dataset_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                data.append(item)
        
        if dataset_config.get('use_full_dataset', True):
            console.print(f"[green]✓ Using full dataset: {len(data)} examples[/green]")
        else:
            # Could add logic to limit dataset size here
            pass
        
        # Format data according to prompt template
        prompt_template = self.strategy['prompt_template']
        formatted_data = []
        
        for item in data:
            # Use the format from strategy
            text = prompt_template['format'].format(
                system=prompt_template['system'],
                input=item['input'],
                output=item['output']
            )
            formatted_data.append({"text": text})
        
        dataset = Dataset.from_list(formatted_data)
        
        # Split dataset
        train_split = dataset_config.get('train_split', 0.8)
        split = dataset.train_test_split(
            test_size=1-train_split, 
            seed=dataset_config.get('seed', 42)
        )
        
        return split["train"], split["test"]
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer from strategy configuration."""
        base_model_config = self.strategy['base_model']
        lora_config_dict = self.strategy['lora_config']
        tokenizer_config = self.strategy['tokenizer']
        
        console.print(f"[cyan]Loading model: {base_model_config['huggingface_id']}[/cyan]")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_config['huggingface_id']
        )
        
        # Configure tokenizer from strategy
        if tokenizer_config['pad_token'] == "eos_token":
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = tokenizer_config['padding_side']
        
        # Determine device and dtype
        if torch.cuda.is_available():
            device_map = base_model_config.get('device_map', 'auto')
            dtype = torch.float16 if base_model_config.get('dtype') == 'float16' else torch.float32
        elif torch.backends.mps.is_available():
            device_map = None
            dtype = torch.float16 if base_model_config.get('dtype') == 'float16' else torch.float32
        else:
            device_map = None
            dtype = torch.float32
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            base_model_config['huggingface_id'],
            torch_dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=base_model_config.get('low_cpu_mem_usage', True)
        )
        
        # Move to device if needed (for MPS)
        if torch.backends.mps.is_available() and device_map is None:
            model = model.to("mps")
        
        # Apply LoRA from strategy configuration
        console.print("[cyan]Applying LoRA configuration from strategy...[/cyan]")
        
        # Convert task_type string to enum
        task_type = getattr(TaskType, lora_config_dict['task_type'])
        
        lora_config = LoraConfig(
            r=lora_config_dict['r'],
            lora_alpha=lora_config_dict['lora_alpha'],
            target_modules=lora_config_dict['target_modules'],
            lora_dropout=lora_config_dict['lora_dropout'],
            bias=lora_config_dict['bias'],
            task_type=task_type,
        )
        
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        console.print(
            f"[green]✓ Trainable params: {trainable_params:,} / {all_params:,} "
            f"({100 * trainable_params / all_params:.2f}%)[/green]"
        )
        
        return model, tokenizer
    
    def tokenize_function(self, examples, tokenizer):
        """Tokenize examples using strategy configuration."""
        tokenizer_config = self.strategy['tokenizer']
        dataset_config = self.strategy['dataset']
        
        return tokenizer(
            examples["text"],
            truncation=tokenizer_config.get('truncation', True),
            padding="max_length",
            max_length=dataset_config.get('max_length', 512),
        )
    
    def train(self):
        """Run training using strategy configuration."""
        console.print(Panel.fit(
            f"[bold cyan]{self.strategy['name']}[/bold cyan]\n"
            f"[yellow]{self.strategy['description']}[/yellow]",
            title="🏥 Strategy-Driven Training",
            border_style="cyan"
        ))
        
        # Load data
        train_dataset, eval_dataset = self.load_and_prepare_data()
        
        # Setup model and tokenizer
        model, tokenizer = self.setup_model_and_tokenizer()
        
        # Tokenize datasets
        console.print("[cyan]Tokenizing datasets...[/cyan]")
        
        train_tokenized = train_dataset.map(
            lambda x: self.tokenize_function(x, tokenizer),
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        eval_tokenized = eval_dataset.map(
            lambda x: self.tokenize_function(x, tokenizer),
            batched=True,
            remove_columns=eval_dataset.column_names
        )
        
        console.print(f"[green]✓ Train: {len(train_tokenized)}, Eval: {len(eval_tokenized)} examples[/green]")
        
        # Get training arguments from strategy
        training_args_dict = self.strategy['training_args'].copy()
        
        # Handle device-specific settings
        if torch.backends.mps.is_available():
            training_args_dict['fp16'] = False  # MPS doesn't support fp16 well
            training_args_dict['use_mps_device'] = True
        
        # Create TrainingArguments
        training_args = TrainingArguments(**training_args_dict)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=eval_tokenized,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        
        # Display training configuration
        self.display_training_config()
        
        # Train
        console.print("\n[bold cyan]Starting training...[/bold cyan]")
        console.print("[dim]This will take 5-15 minutes depending on your hardware[/dim]\n")
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("[cyan]Training model...", total=None)
                
                # Train the model
                train_result = trainer.train()
                
                progress.update(task, description="[green]Training complete!")
            
            # Save the model
            console.print("\n[cyan]Saving model...[/cyan]")
            trainer.save_model()
            tokenizer.save_pretrained(training_args_dict['output_dir'])
            
            # Save strategy info with model
            strategy_info_path = Path(training_args_dict['output_dir']) / "strategy_info.yaml"
            with open(strategy_info_path, 'w') as f:
                yaml.dump({
                    'strategy_name': self.strategy_name,
                    'strategy': self.strategy
                }, f)
            
            console.print(f"\n[bold green]✓ Training complete![/bold green]")
            console.print(f"[green]Model saved to: {training_args_dict['output_dir']}[/green]")
            
            # Validate the model
            if self.strategy.get('validation'):
                self.validate_model(model, tokenizer)
            
            return True
            
        except Exception as e:
            console.print(f"\n[red]Training failed: {e}[/red]")
            import traceback
            traceback.print_exc()
            return False
    
    def display_training_config(self):
        """Display training configuration from strategy."""
        table = Table(title="Training Configuration (from strategy)")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="yellow")
        
        # Model info
        table.add_row("Model", self.strategy['base_model']['huggingface_id'])
        table.add_row("Strategy", self.strategy_name)
        
        # LoRA config
        lora = self.strategy['lora_config']
        table.add_row("LoRA Rank", str(lora['r']))
        table.add_row("LoRA Alpha", str(lora['lora_alpha']))
        table.add_row("Target Modules", ", ".join(lora['target_modules']))
        
        # Training config
        training = self.strategy['training_args']
        table.add_row("Epochs", str(training['num_train_epochs']))
        table.add_row("Batch Size", str(training['per_device_train_batch_size']))
        table.add_row("Learning Rate", str(training['learning_rate']))
        table.add_row("Output Dir", training['output_dir'])
        
        # Dataset config
        dataset = self.strategy['dataset']
        table.add_row("Dataset", dataset['path'])
        table.add_row("Train Split", f"{dataset['train_split']*100:.0f}%")
        
        console.print(table)
    
    def validate_model(self, model, tokenizer):
        """Validate the trained model using strategy test questions."""
        validation_config = self.strategy.get('validation', {})
        test_questions = validation_config.get('test_questions', [])
        
        if not test_questions:
            return
        
        console.print("\n[cyan]Validating trained model...[/cyan]")
        
        model.eval()
        device = next(model.parameters()).device
        generation_config = self.strategy.get('generation', {})
        
        for question in test_questions[:2]:  # Test first 2 questions
            # Format prompt using strategy template
            prompt_template = self.strategy['prompt_template']
            prompt = f"""<|system|>
{prompt_template['system']}</s>
<|user|>
{question}</s>
<|assistant|>"""
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=generation_config.get('max_new_tokens', 100),
                    temperature=generation_config.get('temperature', 0.7),
                    do_sample=generation_config.get('do_sample', True),
                    top_p=generation_config.get('top_p', 0.9),
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            response = tokenizer.decode(
                outputs[0][len(inputs["input_ids"][0]):], 
                skip_special_tokens=True
            )
            
            # Check response quality
            min_length = validation_config.get('min_response_length', 20)
            if len(response) < min_length:
                console.print(f"[yellow]⚠ Response too short ({len(response)} chars)[/yellow]")
            
            console.print(f"\n[bold]Q:[/bold] {question}")
            console.print(f"[bold]A:[/bold] {response[:200]}...")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train medical model using strategy configuration"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="medical_training_fixed",
        help="Strategy name from strategies.yaml"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode - reduce epochs for testing"
    )
    
    args = parser.parse_args()
    
    trainer = StrategyDrivenTrainer(strategy_name=args.strategy)
    
    if args.quick:
        # Override epochs for quick testing
        console.print("[yellow]Quick mode: Reducing epochs to 1[/yellow]")
        trainer.strategy['training_args']['num_train_epochs'] = 1
        trainer.strategy['training_args']['save_steps'] = 20
        trainer.strategy['training_args']['eval_steps'] = 20
    
    success = trainer.train()
    
    if success:
        console.print("\n[bold green]🎉 Success![/bold green]")
        console.print("\nTo chat with the trained model, run:")
        console.print(f"[cyan]uv run python demos/chat_with_strategy_model.py --strategy {args.strategy}[/cyan]")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())