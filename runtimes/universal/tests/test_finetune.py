"""Unit tests for fine-tuning addon."""

import importlib.util

import pytest

from finetune.data_prep import (
    auto_detect_format,
    format_alpaca,
    format_raw_text,
    format_sharegpt,
    validate_cpt_dataset,
    validate_sft_dataset,
)
from finetune.helpers import (
    check_template_compatibility,
    estimate_training_time,
    get_lora_target_modules,
    get_supported_templates,
    resolve_gguf_to_hf,
    validate_model_for_finetune,
)

HAS_UNSLOTH = (
    importlib.util.find_spec("unsloth_mlx") is not None
    or importlib.util.find_spec("unsloth") is not None
)


class TestDataPrep:
    """Tests for data preparation and validation."""
    
    def test_auto_detect_format_sharegpt(self):
        """Test ShareGPT format detection."""
        data = [
            {
                "conversations": [
                    {"from": "human", "value": "Hello"},
                    {"from": "gpt", "value": "Hi there!"}
                ]
            }
        ]
        assert auto_detect_format(data) == "sharegpt"
    
    def test_auto_detect_format_chat(self):
        """Test chat format detection."""
        data = [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"}
                ]
            }
        ]
        assert auto_detect_format(data) == "chat"
    
    def test_auto_detect_format_alpaca(self):
        """Test Alpaca format detection."""
        data = [
            {
                "instruction": "Translate to French",
                "input": "Hello",
                "output": "Bonjour"
            }
        ]
        assert auto_detect_format(data) == "alpaca"
    
    def test_auto_detect_format_raw(self):
        """Test raw text format detection."""
        data = [{"text": "This is raw text."}]
        assert auto_detect_format(data) == "raw"
    
    def test_auto_detect_format_unknown(self):
        """Test unknown format detection."""
        data = [{"unknown_field": "value"}]
        assert auto_detect_format(data) == "unknown"
    
    def test_validate_sft_dataset_valid_sharegpt(self):
        """Test validation of valid ShareGPT dataset."""
        data = [
            {
                "conversations": [
                    {"from": "human", "value": "Hello"},
                    {"from": "gpt", "value": "Hi!"}
                ]
            }
        ]
        result = validate_sft_dataset(data)
        assert result.valid
        assert result.format == "sharegpt"
        assert result.num_examples == 1
    
    def test_validate_sft_dataset_valid_chat(self):
        """Test validation of valid chat dataset."""
        data = [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"}
                ]
            }
        ]
        result = validate_sft_dataset(data)
        assert result.valid
        assert result.format == "chat"
    
    def test_validate_sft_dataset_invalid_sharegpt(self):
        """Test validation of invalid ShareGPT dataset."""
        data = [
            {
                "conversations": [
                    {"from": "human"}  # Missing 'value'
                ]
            }
        ]
        result = validate_sft_dataset(data)
        assert not result.valid
        assert result.errors is not None
    
    def test_validate_sft_dataset_empty(self):
        """Test validation of empty dataset."""
        result = validate_sft_dataset([])
        assert not result.valid
        assert "empty" in str(result.errors).lower()
    
    def test_validate_cpt_dataset_valid(self):
        """Test validation of valid CPT dataset."""
        data = [{"text": "This is training text."}]
        result = validate_cpt_dataset(data)
        assert result.valid
        assert result.format == "raw"
    
    def test_validate_cpt_dataset_invalid(self):
        """Test validation of invalid CPT dataset."""
        data = [{"not_text": "Missing text field"}]
        result = validate_cpt_dataset(data)
        assert not result.valid
        assert result.errors is not None
    
    def test_format_sharegpt_already_formatted(self):
        """Test formatting data already in ShareGPT format."""
        data = [
            {
                "conversations": [
                    {"from": "human", "value": "Hello"},
                    {"from": "gpt", "value": "Hi!"}
                ]
            }
        ]
        result = format_sharegpt(data)
        assert result == data
    
    def test_format_sharegpt_from_chat(self):
        """Test converting chat format to ShareGPT."""
        data = [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"}
                ]
            }
        ]
        result = format_sharegpt(data)
        assert result[0]["conversations"][0]["from"] == "human"
        assert result[0]["conversations"][0]["value"] == "Hello"
        assert result[0]["conversations"][1]["from"] == "gpt"
    
    def test_format_alpaca(self):
        """Test Alpaca format conversion."""
        data = [
            {
                "messages": [
                    {"role": "user", "content": "Translate this"},
                    {"role": "assistant", "content": "Translated"}
                ]
            }
        ]
        result = format_alpaca(data)
        assert "instruction" in result[0]
        assert "output" in result[0]
    
    def test_format_raw_text_from_text(self):
        """Test raw text formatting from text field."""
        data = [{"text": "Raw text here"}]
        result = format_raw_text(data)
        assert result[0]["text"] == "Raw text here"
    
    def test_format_raw_text_from_conversations(self):
        """Test raw text formatting from conversations."""
        data = [
            {
                "conversations": [
                    {"from": "human", "value": "Hello"},
                    {"from": "gpt", "value": "Hi!"}
                ]
            }
        ]
        result = format_raw_text(data)
        assert "text" in result[0]
        assert "human: Hello" in result[0]["text"]
        assert "gpt: Hi!" in result[0]["text"]


class TestHelpers:
    """Tests for helper utilities."""
    
    def test_get_supported_templates(self):
        """Test getting list of supported templates."""
        templates = get_supported_templates()
        assert isinstance(templates, list)
        assert len(templates) > 0
        assert "llama-3" in templates
        assert "chatml" in templates
    
    def test_check_template_compatibility_llama3(self):
        """Test template compatibility check for Llama-3."""
        result = check_template_compatibility("meta-llama/Llama-3-8B")
        assert result["template"] == "llama-3"
        assert result["confidence"] == "high"
    
    def test_check_template_compatibility_qwen(self):
        """Test template compatibility check for Qwen."""
        result = check_template_compatibility("Qwen/Qwen2-7B")
        assert result["template"] == "qwen"
        assert result["confidence"] == "high"
    
    def test_check_template_compatibility_unknown(self):
        """Test template compatibility check for unknown model."""
        result = check_template_compatibility("unknown/model")
        assert result["template"] == "chatml"  # Default
        assert result["confidence"] == "low"
    
    def test_estimate_training_time(self):
        """Test training time estimation."""
        result = estimate_training_time(
            dataset_size=1000,
            model_size="7B",
            epochs=3,
            backend="mlx"
        )
        assert "estimated_minutes" in result
        assert "estimated_hours" in result
        assert result["dataset_size"] == 1000
        assert result["epochs"] == 3
        assert result["model_size"] == "7B"
    
    def test_estimate_training_time_int_size(self):
        """Test training time estimation with integer size."""
        result = estimate_training_time(
            dataset_size=500,
            model_size=8,
            epochs=2
        )
        assert result["model_size"] == "8B"
    
    def test_validate_model_for_finetune_valid(self):
        """Test model validation for valid HF model."""
        result = validate_model_for_finetune("meta-llama/Llama-3-8B")
        assert result["valid"]
        assert not result["is_gguf"]
    
    def test_validate_model_for_finetune_gguf_name(self):
        """Test model validation for GGUF model name."""
        result = validate_model_for_finetune("Qwen/Qwen3-8B-GGUF")
        assert not result["valid"]
        assert result["is_gguf"]
        assert "warnings" in result
        assert "suggested_model" in result
        assert result["suggested_model"] == "Qwen/Qwen3-8B"
    
    def test_resolve_gguf_to_hf(self):
        """Test GGUF model name resolution."""
        resolved = resolve_gguf_to_hf("Qwen/Qwen3-8B-GGUF")
        assert resolved == "Qwen/Qwen3-8B"
        
        resolved = resolve_gguf_to_hf("model.gguf")
        assert resolved == "model"
        
        resolved = resolve_gguf_to_hf("regular-model")
        assert resolved == "regular-model"
    
    def test_get_lora_target_modules_sft(self):
        """Test LoRA target modules for SFT."""
        modules = get_lora_target_modules("meta-llama/Llama-3-8B", task="sft")
        assert isinstance(modules, list)
        assert "q_proj" in modules
        assert "v_proj" in modules
        assert "embed_tokens" not in modules
        assert "lm_head" not in modules
    
    def test_get_lora_target_modules_cpt(self):
        """Test LoRA target modules for CPT."""
        modules = get_lora_target_modules("meta-llama/Llama-3-8B", task="cpt")
        assert isinstance(modules, list)
        assert "q_proj" in modules
        assert "embed_tokens" in modules
        assert "lm_head" in modules
    
    def test_get_lora_target_modules_phi(self):
        """Test LoRA target modules for Phi models."""
        modules = get_lora_target_modules("microsoft/phi-2", task="sft")
        assert "dense" in modules or "fc1" in modules


class TestTrainerConfig:
    """Tests for trainer configuration."""
    
    def test_sft_job_config_creation(self):
        """Test SFT job config creation."""
        from finetune.trainer import SFTJobConfig
        
        config = SFTJobConfig(
            job_id="test-123",
            model="meta-llama/Llama-3-8B",
            dataset=[{"text": "test"}],
            epochs=3,
            batch_size=2
        )
        
        assert config.job_id == "test-123"
        assert config.model == "meta-llama/Llama-3-8B"
        assert config.epochs == 3
        assert config.batch_size == 2
        assert config.output_gguf  # Default True
    
    def test_cpt_job_config_creation(self):
        """Test CPT job config creation."""
        from finetune.trainer import CPTJobConfig
        
        config = CPTJobConfig(
            job_id="test-456",
            model="meta-llama/Llama-3-8B",
            dataset=[{"text": "training data"}],
            epochs=1
        )
        
        assert config.job_id == "test-456"
        assert config.epochs == 1
        assert config.embedding_learning_rate == 5e-6


@pytest.mark.skipif(not HAS_UNSLOTH, reason="unsloth or unsloth_mlx not installed")
class TestTrainerWithUnsloth:
    """Tests that require unsloth libraries."""
    
    def test_detect_backend(self):
        """Test backend detection."""
        from finetune.trainer import detect_backend
        
        backend = detect_backend()
        assert backend in ["mlx", "cuda"]
    
    def test_import_training_libs(self):
        """Test importing training libraries."""
        from finetune.trainer import detect_backend, import_training_libs
        
        backend = detect_backend()
        FastLanguageModel, SFTTrainer, actual_backend = import_training_libs(backend)
        
        assert actual_backend == backend
        assert FastLanguageModel is not None
        assert SFTTrainer is not None


class TestJobQueue:
    """Tests for job queue management (without actually training)."""
    
    @pytest.mark.asyncio
    async def test_trainer_lifecycle(self):
        """Test trainer start/stop lifecycle."""
        from finetune.trainer import FineTuneTrainer
        
        trainer = FineTuneTrainer()
        
        # Start
        await trainer.start()
        assert trainer._running
        
        # Stop
        await trainer.stop()
        assert not trainer._running
    
    @pytest.mark.asyncio
    async def test_list_jobs_empty(self):
        """Test listing jobs when queue is empty."""
        from finetune.trainer import FineTuneTrainer
        
        trainer = FineTuneTrainer()
        jobs = trainer.list_jobs()
        
        assert jobs == []
    
    @pytest.mark.asyncio
    async def test_get_job_not_found(self):
        """Test getting non-existent job."""
        from finetune.trainer import FineTuneTrainer
        
        trainer = FineTuneTrainer()
        job = trainer.get_job("nonexistent")
        
        assert job is None
    
    @pytest.mark.asyncio
    async def test_cancel_job_not_found(self):
        """Test cancelling non-existent job."""
        from finetune.trainer import FineTuneTrainer
        
        trainer = FineTuneTrainer()
        success = await trainer.cancel_job("nonexistent")
        
        assert not success
