"""
LlamaFactory integration for fine-tuning.

This module provides integration with LlamaFactory for easy fine-tuning
with optimized settings and configurations.
"""

import os
import uuid
import json
import subprocess
import tempfile
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from datetime import datetime
import logging

from ..core.base import BaseFineTuner, FineTuningConfig, TrainingJob

logger = logging.getLogger(__name__)

# Try to detect LlamaFactory
try:
    import llamafactory
    LLAMAFACTORY_AVAILABLE = True
except ImportError:
    try:
        # Try alternative import
        from llamafactory_cli import main as llamafactory_main
        LLAMAFACTORY_AVAILABLE = True
    except ImportError:
        LLAMAFACTORY_AVAILABLE = False


class LlamaFactoryFineTuner(BaseFineTuner):
    """LlamaFactory-based fine-tuning implementation."""
    
    def __init__(self, config: FineTuningConfig):
        """Initialize LlamaFactory fine-tuner."""
        if not LLAMAFACTORY_AVAILABLE:
            raise ImportError(
                "LlamaFactory not available. Install with: "
                "pip install llamafactory-cli"
            )
        
        super().__init__(config)
        self._process = None
        self._config_file = None
        
    def validate_config(self) -> List[str]:
        """Validate the LlamaFactory configuration."""
        errors = []
        
        # Check base model configuration
        if "name" not in self.config.base_model:
            errors.append("Base model 'name' is required")
        
        # Check method compatibility
        method_type = self.config.method.get("type")
        if method_type not in self.get_supported_methods():
            errors.append(f"Unsupported method: {method_type}")
        
        # Check dataset configuration
        if "path" not in self.config.dataset:
            errors.append("Dataset 'path' is required")
        
        # Check output directory
        if "output_dir" not in self.config.training_args:
            errors.append("Training argument 'output_dir' is required")
        
        # Hardware compatibility
        device = self.config.environment.get("device", "auto")
        if device == "cpu":
            errors.append("LlamaFactory requires GPU for optimal performance")
        
        return errors
    
    def prepare_model(self) -> Any:
        """Prepare model configuration for LlamaFactory."""
        logger.info("Preparing model configuration for LlamaFactory...")
        
        # LlamaFactory handles model loading internally
        # We just need to ensure the configuration is correct
        model_name = self.config.base_model.get("name", "llama3.2-3b")
        
        # Map our model names to LlamaFactory model names
        model_mapping = {
            "llama3.2-3b": "llama3_2_3b_instruct",
            "llama3.1-8b": "llama3_1_8b_instruct", 
            "llama3.1-70b": "llama3_1_70b_instruct",
            "mistral-7b": "mistral_7b_instruct_v0_3",
            "codellama-13b": "codellama_13b_instruct"
        }
        
        llamafactory_model = model_mapping.get(model_name, model_name)
        logger.info(f"Using LlamaFactory model: {llamafactory_model}")
        
        return llamafactory_model
    
    def prepare_dataset(self) -> Any:
        """Prepare dataset configuration for LlamaFactory."""
        logger.info("Preparing dataset configuration for LlamaFactory...")
        
        dataset_path = Path(self.config.dataset["path"])
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        # LlamaFactory expects datasets in specific formats
        # We'll create a dataset configuration
        dataset_config = {
            "file_name": str(dataset_path.absolute()),
            "formatting": self.config.dataset.get("conversation_template", "alpaca"),
            "columns": {
                "prompt": "instruction",
                "query": "input", 
                "response": "output"
            }
        }
        
        return dataset_config
    
    def start_training(self, job_id: Optional[str] = None) -> TrainingJob:
        """Start training with LlamaFactory."""
        if job_id is None:
            job_id = str(uuid.uuid4())
        
        logger.info(f"Starting LlamaFactory training job: {job_id}")
        
        # Create training job
        job = TrainingJob(
            job_id=job_id,
            name=f"LlamaFactory Fine-tuning {job_id[:8]}",
            status="running",
            config=self.config,
            started_at=datetime.now(),
            total_epochs=self.config.training_args.get("num_train_epochs", 3)
        )
        
        self._current_job = job
        
        try:
            # Prepare components
            model_name = self.prepare_model()
            dataset_config = self.prepare_dataset()
            
            # Create LlamaFactory configuration
            llamafactory_config = self._create_llamafactory_config(
                model_name, dataset_config
            )
            
            # Write config to temporary file
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.yaml', delete=False
            ) as f:
                import yaml
                yaml.dump(llamafactory_config, f)
                self._config_file = f.name
            
            # Start training process
            self._start_training_process()
            
            # Monitor training (this would be more sophisticated in practice)
            self._monitor_training(job)
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.now()
            raise
        
        return job
    
    def _create_llamafactory_config(self, model_name: str, dataset_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create LlamaFactory configuration."""
        method_type = self.config.method.get("type", "lora")
        training_args = self.config.training_args
        
        config = {
            # Model configuration
            "model_name": model_name,
            "template": dataset_config.get("formatting", "alpaca"),
            
            # Dataset configuration  
            "dataset": "custom",
            "dataset_dir": "data",
            "dataset_info": {
                "custom": {
                    "file_name": dataset_config["file_name"],
                    "formatting": dataset_config["formatting"],
                    "columns": dataset_config["columns"]
                }
            },
            
            # Training configuration
            "stage": "sft",  # Supervised fine-tuning
            "do_train": True,
            "finetuning_type": method_type,
            
            # Training arguments
            "output_dir": training_args.get("output_dir", "./fine_tuned_models"),
            "num_train_epochs": training_args.get("num_train_epochs", 3),
            "per_device_train_batch_size": training_args.get("per_device_train_batch_size", 4),
            "gradient_accumulation_steps": training_args.get("gradient_accumulation_steps", 4),
            "learning_rate": training_args.get("learning_rate", 2e-4),
            "lr_scheduler_type": training_args.get("lr_scheduler_type", "cosine"),
            "warmup_ratio": training_args.get("warmup_ratio", 0.03),
            "max_grad_norm": training_args.get("max_grad_norm", 1.0),
            "weight_decay": training_args.get("weight_decay", 0.01),
            
            # Precision settings
            "fp16": training_args.get("fp16", False),
            "bf16": training_args.get("bf16", True),
            
            # Logging and saving
            "logging_steps": training_args.get("logging_steps", 10),
            "save_steps": training_args.get("save_steps", 500),
            "save_total_limit": training_args.get("save_total_limit", 3),
            
            # Sequence length
            "cutoff_len": training_args.get("max_seq_length", 1024),
            
            # Device settings
            "use_fast_tokenizer": self.config.framework.get("use_fast_tokenizer", True),
            "ddp_find_unused_parameters": False,
        }
        
        # Add method-specific settings
        if method_type in ["lora", "qlora"]:
            config.update({
                "lora_rank": self.config.method.get("r", 16),
                "lora_alpha": self.config.method.get("alpha", 32),
                "lora_dropout": self.config.method.get("dropout", 0.1),
                "lora_target": self.config.method.get("target_modules", ["q_proj", "v_proj"])
            })
            
            if method_type == "qlora":
                config.update({
                    "quantization_bit": 4,
                    "double_quantization": True,
                    "quantization_type": "nf4"
                })
        
        return config
    
    def _start_training_process(self) -> None:
        """Start the LlamaFactory training process."""
        cmd = [
            "llamafactory-cli", "train",
            "--config", self._config_file
        ]
        
        logger.info(f"Starting LlamaFactory: {' '.join(cmd)}")
        
        # Start process
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
    
    def _monitor_training(self, job: TrainingJob) -> None:
        """Monitor the training process."""
        if not self._process:
            return
        
        try:
            # Read output and update job status
            for line in iter(self._process.stdout.readline, ''):
                logger.info(f"LlamaFactory: {line.strip()}")
                
                # Parse progress from output (simplified)
                if "epoch" in line.lower():
                    # Try to extract epoch information
                    try:
                        # This is a simplified parser - real implementation would be more robust
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if "epoch" in part.lower() and i + 1 < len(parts):
                                epoch_info = parts[i + 1]
                                if "/" in epoch_info:
                                    current, total = epoch_info.split("/")
                                    job.current_epoch = int(float(current))
                                    job.total_epochs = int(float(total))
                    except:
                        pass  # Ignore parsing errors
                
                # Check if process is still running
                if self._process.poll() is not None:
                    break
            
            # Wait for process to complete
            return_code = self._process.wait()
            
            if return_code == 0:
                job.status = "completed"
                job.completed_at = datetime.now()
                logger.info("Training completed successfully")
            else:
                job.status = "failed"
                job.error_message = f"Process exited with code {return_code}"
                job.completed_at = datetime.now()
                logger.error(f"Training failed with return code: {return_code}")
        
        except Exception as e:
            logger.error(f"Error monitoring training: {e}")
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.now()
        
        finally:
            # Clean up config file
            if self._config_file and os.path.exists(self._config_file):
                os.unlink(self._config_file)
                self._config_file = None
    
    def stop_training(self) -> None:
        """Stop the current training process."""
        if self._process:
            logger.info("Stopping LlamaFactory training...")
            self._process.terminate()
            
            # Wait a bit for graceful shutdown
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Force killing LlamaFactory process")
                self._process.kill()
            
            if self._current_job:
                self._current_job.status = "cancelled"
                self._current_job.completed_at = datetime.now()
    
    def get_training_status(self) -> Optional[TrainingJob]:
        """Get current training status."""
        return self._current_job
    
    def resume_training(self, checkpoint_path: Path) -> TrainingJob:
        """Resume training from checkpoint."""
        logger.info(f"Resuming training from: {checkpoint_path}")
        
        # Update config to resume from checkpoint
        self.config.training_args["resume_from_checkpoint"] = str(checkpoint_path)
        
        return self.start_training()
    
    def evaluate_model(self, checkpoint_path: Optional[Path] = None) -> Dict[str, Any]:
        """Evaluate the fine-tuned model."""
        logger.info("Evaluating model with LlamaFactory...")
        
        # Create evaluation config
        eval_config = self._create_llamafactory_config("", {})
        eval_config.update({
            "stage": "sft",
            "do_train": False,
            "do_eval": True,
            "model_name_or_path": str(checkpoint_path) if checkpoint_path else self.config.training_args.get("output_dir")
        })
        
        # TODO: Implement actual evaluation
        return {
            "evaluation_timestamp": datetime.now().isoformat(),
            "note": "Evaluation not yet implemented for LlamaFactory"
        }
    
    def export_model(self, output_path: Path, format: str = "pytorch") -> None:
        """Export the fine-tuned model."""
        logger.info(f"Exporting model to: {output_path}")
        
        if format == "pytorch":
            # LlamaFactory saves models in PyTorch format by default
            # Just copy from training output directory
            training_output = Path(self.config.training_args.get("output_dir", "./fine_tuned_models"))
            
            if training_output.exists():
                import shutil
                shutil.copytree(training_output, output_path, dirs_exist_ok=True)
                logger.info(f"Model exported to {output_path}")
            else:
                raise FileNotFoundError(f"Training output not found: {training_output}")
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_supported_methods(self) -> List[str]:
        """Get supported fine-tuning methods."""
        return ["lora", "qlora", "full_finetune"]
    
    def get_supported_models(self) -> List[str]:
        """Get supported model architectures."""
        return ["llama", "mistral", "codellama", "baichuan", "chatglm", "qwen"]
    
    def estimate_resources(self) -> Dict[str, Any]:
        """Estimate resources for LlamaFactory training."""
        estimates = super().estimate_resources()
        
        # LlamaFactory is generally more memory efficient
        estimates["memory_gb"] *= 0.8
        estimates["training_time_hours"] *= 0.7  # Usually faster
        
        return estimates