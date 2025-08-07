"""
Tests for fine-tuning base classes.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from ..core.base import FineTuningConfig, TrainingJob, BaseFineTuner


class TestFineTuningConfig:
    """Test FineTuningConfig model."""
    
    def test_basic_config_creation(self):
        """Test creating a basic configuration."""
        config = FineTuningConfig(
            base_model={"name": "llama3.2-3b"},
            method={"type": "lora"},
            framework={"type": "pytorch"},
            training_args={"output_dir": "./test_output"},
            dataset={"path": "./test_data.jsonl"}
        )
        
        assert config.base_model["name"] == "llama3.2-3b"
        assert config.method["type"] == "lora"
        assert config.framework["type"] == "pytorch"
        assert config.training_args["output_dir"] == "./test_output"
        assert config.dataset["path"] == "./test_data.jsonl"
    
    def test_config_with_optional_fields(self):
        """Test configuration with optional fields."""
        config = FineTuningConfig(
            base_model={"name": "llama3.2-3b"},
            method={"type": "lora"},
            framework={"type": "pytorch"},
            training_args={"output_dir": "./test_output"},
            dataset={"path": "./test_data.jsonl"},
            environment={"device": "cuda"},
            advanced={"use_flash_attention": True}
        )
        
        assert config.environment["device"] == "cuda"
        assert config.advanced["use_flash_attention"] is True


class TestTrainingJob:
    """Test TrainingJob model."""
    
    def test_job_creation(self):
        """Test creating a training job."""
        config = FineTuningConfig(
            base_model={"name": "llama3.2-3b"},
            method={"type": "lora"},
            framework={"type": "pytorch"},
            training_args={"output_dir": "./test_output"},
            dataset={"path": "./test_data.jsonl"}
        )
        
        job = TrainingJob(
            job_id="test-job-123",
            name="Test Job",
            config=config,
            total_epochs=3
        )
        
        assert job.job_id == "test-job-123"
        assert job.name == "Test Job"
        assert job.status == "pending"
        assert job.total_epochs == 3
        assert job.current_epoch == 0
    
    def test_job_status_validation(self):
        """Test job status validation."""
        config = FineTuningConfig(
            base_model={"name": "llama3.2-3b"},
            method={"type": "lora"},
            framework={"type": "pytorch"},
            training_args={"output_dir": "./test_output"},
            dataset={"path": "./test_data.jsonl"}
        )
        
        # Valid status
        job = TrainingJob(
            job_id="test-job-123",
            config=config,
            status="running"
        )
        assert job.status == "running"
        
        # Invalid status should raise validation error
        with pytest.raises(ValueError):
            TrainingJob(
                job_id="test-job-123",
                config=config,
                status="invalid_status"
            )


class MockFineTuner(BaseFineTuner):
    """Mock fine-tuner for testing."""
    
    def validate_config(self):
        return []
    
    def prepare_model(self):
        return Mock()
    
    def prepare_dataset(self):
        return Mock()
    
    def start_training(self, job_id=None):
        config = FineTuningConfig(
            base_model={"name": "test"},
            method={"type": "lora"},
            framework={"type": "pytorch"},
            training_args={"output_dir": "./test"},
            dataset={"path": "./test.jsonl"}
        )
        return TrainingJob(job_id=job_id or "mock-job", config=config)
    
    def stop_training(self):
        pass
    
    def get_training_status(self):
        return None
    
    def resume_training(self, checkpoint_path):
        config = FineTuningConfig(
            base_model={"name": "test"},
            method={"type": "lora"},
            framework={"type": "pytorch"},
            training_args={"output_dir": "./test"},
            dataset={"path": "./test.jsonl"}
        )
        return TrainingJob(job_id="resumed-job", config=config)
    
    def evaluate_model(self, checkpoint_path=None):
        return {"accuracy": 0.85}
    
    def export_model(self, output_path, format="pytorch"):
        pass


class TestBaseFineTuner:
    """Test BaseFineTuner abstract class."""
    
    def test_fine_tuner_initialization(self):
        """Test fine-tuner initialization."""
        config = FineTuningConfig(
            base_model={"name": "llama3.2-3b"},
            method={"type": "lora"},
            framework={"type": "pytorch"},
            training_args={"output_dir": "./test_output"},
            dataset={"path": "./test_data.jsonl"}
        )
        
        tuner = MockFineTuner(config)
        assert tuner.config == config
        assert tuner._current_job is None
    
    def test_supported_methods(self):
        """Test getting supported methods."""
        config = FineTuningConfig(
            base_model={"name": "llama3.2-3b"},
            method={"type": "lora"},
            framework={"type": "pytorch"},
            training_args={"output_dir": "./test_output"},
            dataset={"path": "./test_data.jsonl"}
        )
        
        tuner = MockFineTuner(config)
        methods = tuner.get_supported_methods()
        assert isinstance(methods, list)
        assert "lora" in methods
    
    def test_supported_models(self):
        """Test getting supported models."""
        config = FineTuningConfig(
            base_model={"name": "llama3.2-3b"},
            method={"type": "lora"},
            framework={"type": "pytorch"},
            training_args={"output_dir": "./test_output"},
            dataset={"path": "./test_data.jsonl"}
        )
        
        tuner = MockFineTuner(config)
        models = tuner.get_supported_models()
        assert isinstance(models, list)
        assert "llama" in models
    
    def test_estimate_resources(self):
        """Test resource estimation."""
        config = FineTuningConfig(
            base_model={"name": "llama3.2-3b"},
            method={"type": "lora"},
            framework={"type": "pytorch"},
            training_args={"output_dir": "./test_output"},
            dataset={"path": "./test_data.jsonl"}
        )
        
        tuner = MockFineTuner(config)
        estimates = tuner.estimate_resources()
        
        assert "memory_gb" in estimates
        assert "training_time_hours" in estimates
        assert "storage_gb" in estimates
        assert "gpu_required" in estimates
        
        assert isinstance(estimates["memory_gb"], (int, float))
        assert isinstance(estimates["training_time_hours"], (int, float))
        assert isinstance(estimates["gpu_required"], bool)
    
    def test_hardware_recommendations(self):
        """Test hardware recommendations."""
        config = FineTuningConfig(
            base_model={"name": "llama3.2-3b"},
            method={"type": "lora"},
            framework={"type": "pytorch"},
            training_args={"output_dir": "./test_output"},
            dataset={"path": "./test_data.jsonl"}
        )
        
        tuner = MockFineTuner(config)
        recommendations = tuner.get_hardware_recommendations()
        
        assert "min_memory_gb" in recommendations
        assert "recommended_memory_gb" in recommendations
        assert "gpu_required" in recommendations
        assert "suitable_hardware" in recommendations
        
        assert isinstance(recommendations["suitable_hardware"], list)
    
    def test_training_workflow(self):
        """Test basic training workflow."""
        config = FineTuningConfig(
            base_model={"name": "llama3.2-3b"},
            method={"type": "lora"},
            framework={"type": "pytorch"},
            training_args={"output_dir": "./test_output"},
            dataset={"path": "./test_data.jsonl"}
        )
        
        tuner = MockFineTuner(config)
        
        # Validate config
        errors = tuner.validate_config()
        assert isinstance(errors, list)
        
        # Prepare components
        model = tuner.prepare_model()
        dataset = tuner.prepare_dataset()
        assert model is not None
        assert dataset is not None
        
        # Start training
        job = tuner.start_training("test-job-123")
        assert job.job_id == "test-job-123"
        assert isinstance(job, TrainingJob)
        
        # Evaluate model
        metrics = tuner.evaluate_model()
        assert isinstance(metrics, dict)
        
        # Export model
        tuner.export_model(Path("./test_export"))  # Should not raise


if __name__ == "__main__":
    pytest.main([__file__])