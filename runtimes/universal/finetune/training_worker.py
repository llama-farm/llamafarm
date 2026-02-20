#!/usr/bin/env python3
"""Training worker subprocess - runs actual fine-tuning with OOM isolation."""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ── GGUF conversion helpers ──────────────────────────────────────────────

_GGUF_QUANT_MAP = {
    "f16": "f16", "f32": "f32",
    "q8_0": "q8_0", "q4_0": "q4_0", "q4_1": "q4_1",
    "q5_0": "q5_0", "q5_1": "q5_1",
    "q4_k_m": "q8_0",  # convert as q8_0, quantize separately
    "q4_k_s": "q8_0",
}


def _quant_to_gguf_type(quant: str) -> str:
    """Map quantization name to convert_hf_to_gguf.py --outtype value."""
    return _GGUF_QUANT_MAP.get(quant.lower(), "q8_0")


def _find_hf_to_gguf_converter() -> Path | None:
    """Find convert_hf_to_gguf.py from llama.cpp."""
    import os

    # Check env var first
    llama_dir = os.environ.get("LLAMA_CPP_DIR")
    if llama_dir:
        p = Path(llama_dir) / "convert_hf_to_gguf.py"
        if p.exists():
            return p

    # Check common locations
    candidates = [
        Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
        Path("/tmp/llama_cpp_conv/convert_hf_to_gguf.py"),
        Path("/opt/llama.cpp/convert_hf_to_gguf.py"),
    ]
    for p in candidates:
        if p.exists():
            return p

    # Try to find via pip-installed gguf package
    try:
        import importlib.util
        spec = importlib.util.find_spec("gguf")
        if spec and spec.origin:
            gguf_dir = Path(spec.origin).parent.parent
            p = gguf_dir / "convert_hf_to_gguf.py"
            if p.exists():
                return p
    except Exception:
        pass

    return None


def main():
    """Main worker entry point."""
    if len(sys.argv) < 2:
        logger.error("Usage: training_worker.py <config_json_path>")
        sys.exit(1)
    
    config_path = Path(sys.argv[1])
    
    try:
        # Load config
        with open(config_path) as f:
            config = json.load(f)
        
        logger.info(f"Starting training job {config['job_id']}")
        logger.info(f"Job type: {config['type']}")
        logger.info(f"Model: {config['model']}")
        logger.info(f"Dataset size: {len(config['dataset'])} examples")
        
        start_time = time.time()
        
        # Import training libraries
        from finetune.data_prep import (
            auto_detect_format,
            format_raw_text,
            format_sharegpt,
        )
        from finetune.helpers import get_lora_target_modules, resolve_gguf_to_hf
        from finetune.trainer import detect_backend, import_training_libs
        
        backend = detect_backend()
        FastLanguageModel, SFTTrainer, actual_backend = import_training_libs(backend)
        
        logger.info(f"Using backend: {actual_backend}")
        
        # Resolve model name (handle GGUF references)
        model_name = config['model']
        if 'gguf' in model_name.lower():
            model_name = resolve_gguf_to_hf(model_name)
            logger.info(f"Resolved GGUF reference to: {model_name}")
        
        # Load model
        logger.info(f"Loading model {model_name}...")
        
        max_seq_length = config.get('max_seq_length', 2048)
        dtype = None  # Auto-detect
        load_in_4bit = True  # Use 4-bit quantization during training
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        
        logger.info("Model loaded successfully")
        
        # Get LoRA target modules
        if config.get('target_modules'):
            target_modules = config['target_modules']
        else:
            target_modules = get_lora_target_modules(model_name, config['type'])
        
        logger.info(f"LoRA target modules: {target_modules}")
        
        # Apply LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.get('lora_rank', 16),
            target_modules=target_modules,
            lora_alpha=config.get('lora_alpha', 16),
            lora_dropout=0,  # Optimized for speed
            bias="none",
            use_gradient_checkpointing="unsloth",  # Unsloth optimization
            random_state=3407,
        )
        
        logger.info("LoRA applied to model")
        
        # Prepare dataset
        dataset = config['dataset']
        
        if config['type'] == 'sft':
            # Format dataset for SFT
            dataset_format = config.get('dataset_format') or auto_detect_format(dataset)
            logger.info(f"Dataset format: {dataset_format}")
            
            if dataset_format == 'sharegpt' or dataset_format == 'chat':
                formatted_data = format_sharegpt(dataset, config.get('chat_template'))
            else:
                formatted_data = dataset  # Assume already formatted
            
            # Convert to HF dataset format
            from datasets import Dataset
            hf_dataset = Dataset.from_list(formatted_data)
            
            # Determine chat template
            chat_template = config.get('chat_template', 'chatml')
            
            # Define formatting function
            def formatting_func(examples):
                texts = []
                for convs in examples['conversations']:
                    text_parts = []
                    for turn in convs:
                        if turn['from'] == 'human':
                            text_parts.append(f"<|im_start|>user\n{turn['value']}<|im_end|>")
                        elif turn['from'] == 'gpt':
                            text_parts.append(f"<|im_start|>assistant\n{turn['value']}<|im_end|>")
                    texts.append('\n'.join(text_parts))
                return {"text": texts}
            
            # Try using Unsloth's standardize method if available
            try:
                if hasattr(FastLanguageModel, 'standardize_sharegpt'):
                    hf_dataset = FastLanguageModel.standardize_sharegpt(hf_dataset)
                    logger.info("Using Unsloth standardize_sharegpt")
            except Exception as e:
                logger.warning(f"Could not use standardize_sharegpt: {e}")
        
        elif config['type'] == 'cpt':
            # Format dataset for CPT (raw text)
            formatted_data = format_raw_text(dataset)
            
            from datasets import Dataset
            hf_dataset = Dataset.from_list(formatted_data)
        
        else:
            raise ValueError(f"Unknown job type: {config['type']}")
        
        logger.info(f"Dataset prepared: {len(hf_dataset)} examples")
        
        # Training arguments
        from transformers import TrainingArguments
        
        output_dir = Path(config['output_dir']) / "checkpoints"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        epochs = config.get('epochs', 3)
        batch_size = config.get('batch_size', 2)
        learning_rate = config.get('learning_rate', 2e-4)
        max_steps = config.get('max_steps') or -1
        warmup_steps = config.get('warmup_steps', 10)
        gradient_accumulation_steps = config.get('gradient_accumulation_steps', 2)
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=epochs,
            max_steps=max_steps,
            learning_rate=learning_rate,
            fp16=actual_backend != "mlx",  # MLX doesn't use fp16 flag
            bf16=False,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            save_strategy="epoch",
            save_total_limit=2,
        )
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=hf_dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            args=training_args,
        )
        
        # Train
        logger.info("Starting training...")
        train_result = trainer.train()
        
        logger.info("Training completed!")
        # unsloth-mlx returns dict or CompletedProcess, unsloth returns TrainOutput
        if isinstance(train_result, dict):
            final_loss = train_result.get("training_loss", train_result.get("loss", 0.0))
            global_step = train_result.get("global_step", 0)
        elif hasattr(train_result, "training_loss"):
            final_loss = train_result.training_loss
            global_step = getattr(train_result, "global_step", 0)
        else:
            final_loss = 0.0
            global_step = 0

        # Parse actual loss from training log if not captured from return value
        if final_loss == 0.0:
            log_path = Path(config['output_dir']) / "training.log"
            if log_path.exists():
                import re
                for line in reversed(log_path.read_text().splitlines()):
                    m = re.search(r'Train loss (\d+\.\d+)', line)
                    if m:
                        final_loss = float(m.group(1))
                        break
                    m = re.search(r'Val loss (\d+\.\d+)', line)
                    if m:
                        final_loss = float(m.group(1))
                        break
                # Parse step count
                for line in reversed(log_path.read_text().splitlines()):
                    m = re.search(r'Iter (\d+):', line)
                    if m:
                        global_step = int(m.group(1))
                        break
        logger.info(f"Training loss: {final_loss:.4f}")
        
        # Save LoRA adapter
        lora_dir = Path(config['output_dir']) / "lora_adapter"
        lora_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving LoRA adapter to {lora_dir}")
        model.save_pretrained(str(lora_dir))
        tokenizer.save_pretrained(str(lora_dir))
        
        # Export to GGUF if requested
        gguf_path = None
        if config.get('output_gguf', True):
            quantization = config.get('quantization', 'q8_0')
            gguf_dir = Path(config['output_dir']) / "gguf"
            gguf_dir.mkdir(parents=True, exist_ok=True)
            gguf_file = gguf_dir / f"model-{quantization}.gguf"

            # Strategy 1: Try unsloth's built-in GGUF export
            try:
                logger.info(f"Exporting to GGUF via unsloth ({quantization})...")
                model.save_pretrained_gguf(
                    str(gguf_dir),
                    tokenizer,
                    quantization_method=quantization,
                )
                # Find the output file
                for f in gguf_dir.iterdir():
                    if f.suffix == ".gguf":
                        gguf_path = str(f)
                        break
                if gguf_path:
                    logger.info(f"GGUF export completed: {gguf_path}")
            except Exception as e:
                logger.warning(f"Unsloth GGUF export failed: {e}")

            # Strategy 2: Merge to HF format, then use convert_hf_to_gguf.py
            if not gguf_path:
                try:
                    logger.info("Trying merge → convert_hf_to_gguf.py fallback...")

                    # Step 1: Merge LoRA into full HF model
                    merged_dir = Path(config['output_dir']) / "merged_hf"
                    merged_dir.mkdir(parents=True, exist_ok=True)
                    logger.info("Merging LoRA adapter into base model...")

                    if hasattr(model, 'save_pretrained_merged'):
                        model.save_pretrained_merged(str(merged_dir), tokenizer)
                    else:
                        # CUDA unsloth: merge_and_unload
                        merged_model = model.merge_and_unload()
                        merged_model.save_pretrained(str(merged_dir))
                        tokenizer.save_pretrained(str(merged_dir))

                    logger.info(f"Merged model saved to {merged_dir}")

                    # Step 2: Find convert_hf_to_gguf.py
                    converter = _find_hf_to_gguf_converter()
                    if converter is None:
                        raise FileNotFoundError(
                            "convert_hf_to_gguf.py not found. "
                            "Install llama.cpp or set LLAMA_CPP_DIR."
                        )

                    # Step 3: Convert to GGUF
                    logger.info(f"Converting to GGUF ({quantization})...")
                    import subprocess as sp
                    cmd = [
                        sys.executable, str(converter),
                        str(merged_dir),
                        "--outfile", str(gguf_file),
                        "--outtype", _quant_to_gguf_type(quantization),
                    ]
                    # Ensure converter can find matching gguf package
                    env = dict(os.environ)
                    converter_gguf_py = converter.parent / "gguf-py"
                    if converter_gguf_py.exists():
                        pp = env.get("PYTHONPATH", "")
                        env["PYTHONPATH"] = f"{converter_gguf_py}:{pp}"

                    result = sp.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=300,
                        env=env,
                    )
                    if result.returncode == 0 and gguf_file.exists():
                        gguf_path = str(gguf_file)
                        size_mb = gguf_file.stat().st_size / (1024 * 1024)
                        logger.info(
                            f"GGUF export completed: {gguf_path} "
                            f"({size_mb:.1f} MB)"
                        )
                    else:
                        logger.error(
                            f"convert_hf_to_gguf.py failed: {result.stderr[-500:]}"
                        )
                except Exception as e:
                    logger.warning(f"GGUF fallback export failed: {e}")

            if not gguf_path:
                logger.warning(
                    "GGUF export not available. LoRA adapter saved — "
                    "convert manually with: convert_hf_to_gguf.py"
                )
        
        # Save training log
        log_path = Path(config['output_dir']) / "training_log.jsonl"
        with open(log_path, "w") as f:
            log_entry = {
                "final_loss": float(final_loss),
                "epochs": epochs,
                "steps": global_step,
            }
            json.dump(log_entry, f)
            f.write("\n")
        
        # Save result
        duration = time.time() - start_time
        
        result = {
            "status": "completed",
            "metrics": {
                "final_loss": float(final_loss),
                "epochs": epochs,
                "steps": global_step,
                "lora_adapter": str(lora_dir),
                "gguf_path": gguf_path,
            },
            "duration_seconds": duration,
        }
        
        result_path = Path(config['output_dir']) / "result.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Training completed successfully in {duration:.1f}s")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        
        # Save error result
        result = {
            "status": "failed",
            "error": str(e),
            "duration_seconds": time.time() - start_time if 'start_time' in locals() else 0
        }
        
        try:
            output_dir = Path(config['output_dir']) if 'config' in locals() else Path(".")
            result_path = output_dir / "result.json"
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
        except Exception:
            pass
        
        sys.exit(1)


if __name__ == "__main__":
    main()
