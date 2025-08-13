"""
PyTorch-based fine-tuning implementation.

This module provides a PyTorch/Transformers-based fine-tuner with support for
LoRA, QLoRA, and full fine-tuning methods.
"""

import os
import uuid
import json
import shutil
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from datetime import datetime
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    import torch
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
    from peft import (
        LoraConfig, 
        TaskType, 
        get_peft_model, 
        PeftModel,
        prepare_model_for_kbit_training
    )
    from datasets import load_dataset, Dataset
    import transformers
    PYTORCH_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    PYTORCH_AVAILABLE = False
    IMPORT_ERROR = e
    # Create dummy classes for type hints when imports fail
    class Dataset:
        pass

logger = logging.getLogger(__name__)


class TrainingJob:
    """Represents a training job with metadata."""
    def __init__(self, job_id: str, name: str, status: str, config: Dict[str, Any], 
                 started_at: datetime, total_epochs: int):
        self.job_id = job_id
        self.name = name
        self.status = status
        self.config = config
        self.started_at = started_at
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.current_step = 0
        self.total_steps = 0
        self.metrics = {}
        self.completed_at = None
        self.error_message = None
        self.output_dir = None
        self.checkpoint_dir = None


class PyTorchFineTuner:
    """PyTorch-based fine-tuning implementation."""
    
    def __init__(self, config: Dict[str, Any], verbose: bool = False):
        """Initialize PyTorch fine-tuner."""
        if not PYTORCH_AVAILABLE:
            raise ImportError(f"PyTorch dependencies not available: {IMPORT_ERROR}")
        
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.dataset = None
        self._training_stopped = False
        self._current_job = None
        self._verbose_mode = verbose
        
    def validate_config(self) -> List[str]:
        """Validate the PyTorch configuration."""
        errors = []
        
        # Check base model configuration
        base_model = self.config.get("base_model", {})
        if "name" not in base_model:
            errors.append("Base model 'name' is required")
        
        if "huggingface_id" not in base_model:
            errors.append("Base model 'huggingface_id' is required")
        
        # Check method configuration
        method = self.config.get("method", {})
        method_type = method.get("type")
        if method_type not in self.get_supported_methods():
            errors.append(f"Unsupported method: {method_type}")
        
        # Check LoRA specific configuration
        if method_type in ["lora", "qlora"]:
            if "r" not in method:
                errors.append("LoRA rank 'r' is required for LoRA methods")
            if "alpha" not in method:
                errors.append("LoRA alpha is required for LoRA methods")
        
        # Check training arguments
        training_args = self.config.get("training_args", {})
        if "output_dir" not in training_args:
            errors.append("Training argument 'output_dir' is required")
        
        # Check dataset configuration
        dataset = self.config.get("dataset", {})
        if "path" not in dataset and "dataset_name" not in dataset:
            errors.append("Dataset 'path' or 'dataset_name' is required")
        
        # Hardware compatibility checks
        environment = self.config.get("environment", {})
        device = environment.get("device", "auto")
        
        if method_type == "full_finetune" and device == "cpu":
            errors.append("Full fine-tuning on CPU is very slow and not recommended")
        
        return errors
    
    def prepare_model(self) -> Any:
        """Prepare the base model for fine-tuning."""
        logger.info("Preparing model for fine-tuning...")
        
        model_config = self.config.get("base_model", {})
        huggingface_id = model_config["huggingface_id"]
        
        # Load tokenizer
        logger.info(f"Loading tokenizer: {huggingface_id}")
        
        # Get HF token from environment (only use if available)
        hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN')
        
        tokenizer_kwargs = {
            "cache_dir": model_config.get("cache_dir"),
            "trust_remote_code": model_config.get("trust_remote_code", False),
            "use_fast": self.config.get("framework", {}).get("use_fast_tokenizer", True)
        }
        
        # Only add token if it's actually set
        if hf_token and hf_token != "your_hf_token_here":
            tokenizer_kwargs["token"] = hf_token
            
        self.tokenizer = AutoTokenizer.from_pretrained(huggingface_id, **tokenizer_kwargs)
        
        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        logger.info(f"Loading model: {huggingface_id}")
        # Get HF token from environment (only use if available)
        hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN')
        
        # Determine torch dtype based on hardware
        hardware_config = self.config.get("hardware", {})
        device = hardware_config.get("device", "auto")
        
        # Auto-detect device if needed
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        # Set dtype based on device
        if model_config.get("torch_dtype") and model_config.get("torch_dtype") != "auto":
            torch_dtype = getattr(torch, model_config.get("torch_dtype"), torch.float32)
        else:
            # Auto-select dtype based on device
            if device == "cuda":
                torch_dtype = torch.float16  # Good default for CUDA
            elif device == "mps":
                torch_dtype = torch.float32  # MPS is more stable with float32
            else:
                torch_dtype = torch.float32  # CPU needs float32
        
        model_kwargs = {
            "cache_dir": model_config.get("cache_dir"),
            "trust_remote_code": model_config.get("trust_remote_code", False),
            "torch_dtype": torch_dtype,
            "device_map": "auto" if device != "cpu" else None,  # Auto device map for GPU
            "low_cpu_mem_usage": True
        }
        
        # Only add token if it's actually set and not placeholder
        if hf_token and hf_token != "your_hf_token_here":
            model_kwargs["token"] = hf_token
        
        # Add quantization settings if using QLoRA
        method = self.config.get("method", {})
        method_type = method.get("type")
        if method_type == "qlora":
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs["quantization_config"] = quantization_config
        
        self.model = AutoModelForCausalLM.from_pretrained(huggingface_id, **model_kwargs)
        
        # Prepare model for training based on method
        if method_type in ["lora", "qlora"]:
            self._prepare_peft_model()
        elif method_type == "full_finetune":
            self._prepare_full_finetune()
        
        return self.model
    
    def _prepare_peft_model(self) -> None:
        """Prepare model for PEFT (LoRA/QLoRA) training."""
        method_config = self.config.get("method", {})
        
        # Prepare model for k-bit training if using quantization
        if method_config.get("type") == "qlora":
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=method_config.get("r", 16),
            lora_alpha=method_config.get("alpha", 32),
            lora_dropout=method_config.get("dropout", 0.1),
            target_modules=method_config.get("target_modules", ["q_proj", "v_proj"]),
            bias=method_config.get("bias", "none")
        )
        
        # Apply PEFT to model
        self.model = get_peft_model(self.model, peft_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        logger.info("Model prepared for PEFT training")
    
    def _prepare_full_finetune(self) -> None:
        """Prepare model for full fine-tuning."""
        # Enable gradient computation for all parameters
        for param in self.model.parameters():
            param.requires_grad = True
        
        logger.info("Model prepared for full fine-tuning")
    
    def prepare_dataset(self) -> Any:
        """Prepare the dataset for training."""
        logger.info("Preparing dataset...")
        
        dataset_config = self.config.get("dataset", {})
        training_args = self.config.get("training_args", {})
        
        # Check if we need evaluation dataset
        needs_eval = (
            training_args.get("eval_strategy", "no") != "no" or
            training_args.get("evaluation_strategy", "no") != "no" or
            training_args.get("load_best_model_at_end", False)
        )
        
        # Load dataset
        if "path" in dataset_config:
            dataset_path = Path(dataset_config["path"])
            
            # Check for separate train/eval files
            dataset_dir = dataset_path.parent
            dataset_stem = dataset_path.stem
            train_file = dataset_dir / f"{dataset_stem}_train.jsonl"
            eval_file = dataset_dir / f"{dataset_stem}_eval.jsonl"
            
            if train_file.exists() and eval_file.exists() and needs_eval:
                # Load separate train and eval datasets
                logger.info(f"Loading train dataset from: {train_file}")
                logger.info(f"Loading eval dataset from: {eval_file}")
                datasets = load_dataset("json", data_files={
                    "train": str(train_file),
                    "validation": str(eval_file)
                })
                self.dataset = datasets["train"]
                self.eval_dataset = datasets["validation"]
            else:
                # Load single file
                if dataset_path.suffix == ".jsonl":
                    dataset = load_dataset("json", data_files=str(dataset_path))["train"]
                    
                    # Split for evaluation if needed
                    if needs_eval:
                        eval_split = dataset_config.get("eval_split", 0.1)
                        logger.info(f"Splitting dataset with {eval_split:.0%} for evaluation")
                        split_datasets = dataset.train_test_split(test_size=eval_split, seed=42)
                        self.dataset = split_datasets["train"]
                        self.eval_dataset = split_datasets["test"]
                    else:
                        self.dataset = dataset
                        self.eval_dataset = None
                else:
                    raise ValueError(f"Unsupported file format: {dataset_path.suffix}")
        
        elif "dataset_name" in dataset_config:
            # Load from HuggingFace datasets
            self.dataset = load_dataset(
                dataset_config["dataset_name"],
                dataset_config.get("dataset_config_name"),
                split=dataset_config.get("train_split", "train")
            )
            
            if needs_eval:
                # Try to load eval split or create one
                try:
                    self.eval_dataset = load_dataset(
                        dataset_config["dataset_name"],
                        dataset_config.get("dataset_config_name"),
                        split=dataset_config.get("eval_split", "validation")
                    )
                except:
                    # Create eval split from train
                    eval_split = dataset_config.get("eval_split", 0.1)
                    logger.info(f"Creating eval split ({eval_split:.0%}) from train dataset")
                    split_datasets = self.dataset.train_test_split(test_size=eval_split, seed=42)
                    self.dataset = split_datasets["train"]
                    self.eval_dataset = split_datasets["test"]
            else:
                self.eval_dataset = None
        
        else:
            raise ValueError("No dataset path or name specified")
        
        # Tokenize datasets
        self.dataset = self._tokenize_dataset(self.dataset)
        if self.eval_dataset is not None:
            self.eval_dataset = self._tokenize_dataset(self.eval_dataset)
            logger.info(f"Dataset prepared with {len(self.dataset)} train and {len(self.eval_dataset)} eval examples")
        else:
            logger.info(f"Dataset prepared with {len(self.dataset)} examples (no evaluation)")
        
        return self.dataset
    
    def _tokenize_dataset(self, dataset):
        """Tokenize the dataset."""
        training_args = self.config.get("training_args", {})
        max_length = training_args.get("max_seq_length", 512)
        
        def tokenize_function(examples):
            # Handle different dataset formats
            if "text" in examples:
                texts = examples["text"]
            elif "instruction" in examples and "output" in examples:
                # Alpaca format
                texts = []
                for i in range(len(examples["instruction"])):
                    instruction = examples["instruction"][i]
                    input_text = examples.get("input", [""] * len(examples["instruction"]))[i]
                    output = examples["output"][i]
                    
                    if input_text:
                        text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                    else:
                        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                    texts.append(text)
            else:
                raise ValueError("Unsupported dataset format")
            
            # Tokenize with proper padding for batched processing
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",  # Pad to max_length for consistent tensor sizes
                max_length=max_length,
                return_tensors=None
            )
            
            # Add labels for causal LM
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        # Apply tokenization
        dataset_config = self.config.get("dataset", {})
        # Use num_proc=1 for stability, especially on macOS
        num_workers = dataset_config.get("preprocessing_num_workers", 1)
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=num_workers,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def start_training(self, job_id: Optional[str] = None):
        """Start the fine-tuning process."""
        if job_id is None:
            job_id = str(uuid.uuid4())
        
        logger.info(f"Starting training job: {job_id}")
        
        # Create training job
        training_args = self.config.get("training_args", {})
        job = TrainingJob(
            job_id=job_id,
            name=f"PyTorch Fine-tuning {job_id[:8]}",
            status="running",
            config=self.config,
            started_at=datetime.now(),
            total_epochs=training_args.get("num_train_epochs", 3)
        )
        
        self._current_job = job
        self._training_stopped = False
        
        try:
            # Prepare model and dataset if not already done
            if self.model is None:
                self.prepare_model()
            if self.dataset is None:
                self.prepare_dataset()
            
            # Set up training arguments
            training_args = self._create_training_arguments()
            
            # Create data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,  # Causal LM, not masked LM
            )
            
            # Create trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dataset,
                eval_dataset=getattr(self, 'eval_dataset', None),  # Pass eval dataset if available
                data_collator=data_collator,
                tokenizer=self.tokenizer,
            )
            
            # Add callback to track progress
            self.trainer.add_callback(self._create_progress_callback(verbose=getattr(self, '_verbose_mode', False)))
            
            # Start training
            logger.info("Starting training...")
            train_result = self.trainer.train()
            
            # Update job status
            job.status = "completed"
            job.completed_at = datetime.now()
            job.metrics = train_result.metrics
            
            # Save final model to dedicated directory
            base_output_dir = Path(self.config.get("training_args", {}).get("output_dir", "./fine_tuned_models"))
            final_model_dir = base_output_dir / "final_model"
            
            logger.info(f"Saving final model to: {final_model_dir}")
            self.trainer.save_model(str(final_model_dir))
            
            # Save tokenizer too
            self.tokenizer.save_pretrained(str(final_model_dir))
            
            # Update job info
            job.output_dir = final_model_dir
            job.checkpoint_dir = Path(training_args.output_dir)
            
            logger.info(f"Training completed successfully: {job_id}")
            logger.info(f"Final model saved to: {final_model_dir}")
            logger.info(f"Checkpoints saved to: {training_args.output_dir}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.now()
            raise
        
        return job
    
    def _create_training_arguments(self):
        """Create TrainingArguments from configuration."""
        import torch
        config = self.config.get("training_args", {})
        hardware_config = self.config.get("hardware", {})
        
        # Ensure output directory exists
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organization
        (output_dir / "checkpoints").mkdir(exist_ok=True)
        (output_dir / "logs").mkdir(exist_ok=True)
        (output_dir / "final_model").mkdir(exist_ok=True)
        
        logger.info(f"Training output directory: {output_dir}")
        logger.info(f"Checkpoints will be saved to: {output_dir / 'checkpoints'}")
        
        # Detect device if set to auto
        device = hardware_config.get("device", "auto")
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info("Detected CUDA device")
            elif torch.backends.mps.is_available():
                device = "mps"
                logger.info("Detected Apple Silicon (MPS) device")
            else:
                device = "cpu"
                logger.info("Using CPU (no GPU detected)")
        
        # Get precision settings for the detected/specified device
        precision_config = hardware_config.get("precision", {}).get(device, {})
        optimization_config = hardware_config.get("optimization", {})
        
        # Set precision based on hardware
        if device == "cuda":
            fp16 = precision_config.get("use_fp16", True)
            bf16 = precision_config.get("use_bf16", False)
            tf32 = precision_config.get("use_tf32", True)
        elif device == "mps":
            fp16 = precision_config.get("use_fp16", False)  # Not stable on MPS
            bf16 = precision_config.get("use_bf16", False)  # Not supported
            tf32 = False  # Not applicable to MPS
        else:  # CPU
            fp16 = precision_config.get("use_fp16", False)
            bf16 = False
            tf32 = False
        
        # Base arguments
        eval_strategy = config.get("eval_strategy", "no")
        save_strategy = config.get("save_strategy", "steps")
        
        # If load_best_model_at_end is requested but eval is disabled, disable it
        # since we can't load best model without evaluation
        load_best_model = config.get("load_best_model_at_end", False)
        if load_best_model and eval_strategy == "no":
            logger.warning("load_best_model_at_end requires evaluation dataset. Disabling.")
            load_best_model = False
            
        args = TrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            num_train_epochs=config.get("num_train_epochs", 3),
            per_device_train_batch_size=config.get("per_device_train_batch_size", 4),
            per_device_eval_batch_size=config.get("per_device_eval_batch_size", 4),
            gradient_accumulation_steps=optimization_config.get("gradient_accumulation_steps", 1),
            learning_rate=config.get("learning_rate", 2e-4),
            lr_scheduler_type=config.get("lr_scheduler_type", "linear"),
            warmup_ratio=config.get("warmup_ratio", 0.03),
            max_grad_norm=config.get("max_grad_norm", 1.0),
            optim=config.get("optim", "adamw_torch"),
            weight_decay=config.get("weight_decay", 0.01),
            adam_beta1=config.get("adam_beta1", 0.9),
            adam_beta2=config.get("adam_beta2", 0.999),
            adam_epsilon=config.get("adam_epsilon", 1e-8),
            fp16=fp16,
            bf16=bf16,
            tf32=tf32,
            logging_steps=config.get("logging_steps", 1),  # Log every step for immediate feedback
            eval_strategy=eval_strategy,
            eval_steps=config.get("eval_steps", 500) if eval_strategy == "steps" else None,
            save_strategy=save_strategy,
            save_steps=config.get("save_steps", 500),
            save_total_limit=config.get("save_total_limit", 3),
            load_best_model_at_end=load_best_model,
            metric_for_best_model=config.get("metric_for_best_model", "loss"),
            greater_is_better=config.get("greater_is_better", False),
            report_to=config.get("report_to", []),
            run_name=config.get("run_name"),
            seed=self.config.get("environment", {}).get("seed", 42),
            dataloader_num_workers=config.get("dataloader_num_workers", 0),
            remove_unused_columns=config.get("remove_unused_columns", False),
            # Disable gradient checkpointing on MPS due to compatibility issues
            gradient_checkpointing=optimization_config.get("gradient_checkpointing", False) if device != "mps" else False,
        )
        
        return args
    
    def _create_progress_callback(self, verbose=False):
        """Create callback to track training progress."""
        class ProgressCallback(transformers.TrainerCallback):
            def __init__(self, job: TrainingJob, verbose: bool = False):
                self.job = job
                self.verbose = verbose
                self.last_logged_step = 0
            
            def on_train_begin(self, args, state, control, **kwargs):
                if self.verbose:
                    print(f"\n{'='*60}")
                    print(f"🏋️  TRAINING STARTED")
                    print(f"{'='*60}")
                    print(f"📊 Total Steps: {state.max_steps}")
                    print(f"📊 Epochs: {args.num_train_epochs}")
                    print(f"📊 Batch Size: {args.per_device_train_batch_size}")
                    print(f"📊 Learning Rate: {args.learning_rate}")
                    print(f"{'='*60}\n")
            
            def on_epoch_begin(self, args, state, control, **kwargs):
                epoch = int(state.epoch) + 1  # 1-indexed for display
                self.job.current_epoch = epoch
                if self.verbose:
                    print(f"\n📊 Epoch {epoch}/{args.num_train_epochs} started")
            
            def on_step_end(self, args, state, control, **kwargs):
                self.job.current_step = state.global_step
                self.job.total_steps = state.max_steps
                
                # Show progress EVERY step for immediate feedback in verbose mode
                if self.verbose:
                    progress_pct = (state.global_step / state.max_steps) * 100
                    # Use \r for updating the same line
                    print(f"\r   Step {state.global_step}/{state.max_steps} ({progress_pct:.1f}%) ", end='', flush=True)
                    
                    # Print newline every 5 steps to show loss
                    if state.global_step % 5 == 0 or state.global_step == state.max_steps:
                        print()  # New line to preserve the progress
            
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    # Update job metrics
                    if not hasattr(self.job, 'metrics'):
                        self.job.metrics = {}
                    self.job.metrics.update(logs)
                    
                    # Show training metrics immediately for verbose mode
                    if self.verbose and 'loss' in logs:
                        print(f"\n   📉 Training Loss: {logs['loss']:.4f}", flush=True)
                        if 'learning_rate' in logs:
                            print(f"   🎯 Learning Rate: {logs['learning_rate']:.2e}", flush=True)
            
            def on_epoch_end(self, args, state, control, **kwargs):
                epoch = int(state.epoch)
                if self.verbose:
                    loss_info = f" - Loss: {self.job.metrics.get('loss', 'N/A')}" if hasattr(self.job, 'metrics') else ""
                    print(f"✅ Epoch {epoch} completed{loss_info}")
        
        return ProgressCallback(self._current_job, verbose=getattr(self, '_verbose_mode', False))
    
    def stop_training(self) -> None:
        """Stop the current training process."""
        if self.trainer is not None:
            self.trainer.should_save = False
            self._training_stopped = True
            logger.info("Training stop requested")
            
            if self._current_job:
                self._current_job.status = "cancelled"
                self._current_job.completed_at = datetime.now()
    
    def get_training_status(self):
        """Get the current training status."""
        return self._current_job
    
    def resume_training(self, checkpoint_path: Path):
        """Resume training from a checkpoint."""
        logger.info(f"Resuming training from: {checkpoint_path}")
        
        # Update config to resume from checkpoint
        self.config["training_args"]["resume_from_checkpoint"] = str(checkpoint_path)
        
        # Start training
        return self.start_training()
    
    def evaluate_model(self, checkpoint_path: Optional[Path] = None) -> Dict[str, Any]:
        """Evaluate the fine-tuned model."""
        logger.info("Evaluating model...")
        
        if checkpoint_path:
            # Load specific checkpoint
            model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        elif self.model:
            model = self.model
        else:
            raise ValueError("No model available for evaluation")
        
        # TODO: Implement proper evaluation metrics
        # For now, return basic info
        return {
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "evaluation_timestamp": datetime.now().isoformat()
        }
    
    def export_model(self, output_path: Path, format: str = "pytorch") -> None:
        """Export the fine-tuned model."""
        logger.info(f"Exporting model to: {output_path}")
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if format == "pytorch":
            # Save model and tokenizer
            if self.model:
                self.model.save_pretrained(output_path)
            if self.tokenizer:
                self.tokenizer.save_pretrained(output_path)
            
            # Save configuration
            config_path = output_path / "fine_tuning_config.json"
            with open(config_path, "w") as f:
                json.dump(self.config, f, indent=2, default=str)
            
            logger.info(f"Model exported to {output_path}")
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_supported_methods(self) -> List[str]:
        """Get supported fine-tuning methods."""
        return ["lora", "qlora", "full_finetune"]
    
    def get_supported_models(self) -> List[str]:
        """Get supported model architectures."""
        return ["llama", "mistral", "gpt2", "codellama", "phi", "qwen"]