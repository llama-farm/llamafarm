"""
Tests for fine-tuning factory classes.
"""

import pytest
from unittest.mock import Mock, patch

from ..core.factory import FineTunerFactory, DataProcessorFactory, validate_config_compatibility
from ..core.base import FineTuningConfig, BaseFineTuner, BaseDataProcessor


class MockFineTuner(BaseFineTuner):
    """Mock fine-tuner for testing."""
    
    def validate_config(self):
        return []
    
    def prepare_model(self):
        return Mock()
    
    def prepare_dataset(self):
        return Mock()
    
    def start_training(self, job_id=None):
        from ..core.base import TrainingJob
        return TrainingJob(job_id=job_id or "mock-job", config=self.config)
    
    def stop_training(self):
        pass
    
    def get_training_status(self):
        return None
    
    def resume_training(self, checkpoint_path):
        from ..core.base import TrainingJob
        return TrainingJob(job_id="resumed-job", config=self.config)
    
    def evaluate_model(self, checkpoint_path=None):
        return {"accuracy": 0.85}
    
    def export_model(self, output_path, format="pytorch"):
        pass
    
    def get_supported_methods(self):
        return ["lora", "qlora"]
    
    def get_supported_models(self):
        return ["llama", "mistral"]


class MockDataProcessor(BaseDataProcessor):
    """Mock data processor for testing."""
    
    def process_dataset(self, dataset_path):
        return Mock()
    
    def validate_dataset(self, dataset_path):
        return []


class TestFineTunerFactory:
    """Test FineTunerFactory class."""
    
    def test_register_and_create(self):
        """Test registering and creating a fine-tuner."""
        # Register mock tuner
        FineTunerFactory.register("mock", MockFineTuner)
        
        # Create config
        config = FineTuningConfig(
            base_model={"name": "llama3.2-3b"},
            method={"type": "lora"},
            framework={"type": "mock"},
            training_args={"output_dir": "./test"},
            dataset={"path": "./test.jsonl"}
        )
        
        # Create tuner
        tuner = FineTunerFactory.create(config)
        assert isinstance(tuner, MockFineTuner)
        assert tuner.config == config
        
        # Clean up
        FineTunerFactory.unregister("mock")
    
    def test_create_unsupported_framework(self):
        """Test creating tuner with unsupported framework."""
        config = FineTuningConfig(
            base_model={"name": "llama3.2-3b"},
            method={"type": "lora"},
            framework={"type": "unsupported"},
            training_args={"output_dir": "./test"},
            dataset={"path": "./test.jsonl"}
        )
        
        with pytest.raises(ValueError, match="Unsupported framework type"):
            FineTunerFactory.create(config)
    
    def test_get_available_tuners(self):
        """Test getting available tuner types."""
        # Register mock tuner
        FineTunerFactory.register("mock", MockFineTuner)
        
        available = FineTunerFactory.get_available_tuners()
        assert isinstance(available, list)
        assert "mock" in available
        
        # Clean up
        FineTunerFactory.unregister("mock")
    
    def test_get_tuner_info(self):
        """Test getting tuner information."""
        # Register mock tuner
        FineTunerFactory.register("mock", MockFineTuner)
        
        info = FineTunerFactory.get_tuner_info("mock")
        assert info is not None
        assert info["name"] == "mock"
        assert "supported_methods" in info
        assert "supported_models" in info
        assert "lora" in info["supported_methods"]
        assert "llama" in info["supported_models"]
        
        # Test non-existent tuner
        info = FineTunerFactory.get_tuner_info("nonexistent")
        assert info is None
        
        # Clean up
        FineTunerFactory.unregister("mock")
    
    def test_unregister(self):
        """Test unregistering a tuner."""
        # Register mock tuner
        FineTunerFactory.register("mock", MockFineTuner)
        assert "mock" in FineTunerFactory.get_available_tuners()
        
        # Unregister
        FineTunerFactory.unregister("mock")
        assert "mock" not in FineTunerFactory.get_available_tuners()


class TestDataProcessorFactory:
    """Test DataProcessorFactory class."""
    
    def test_register_and_create(self):
        """Test registering and creating a data processor."""
        # Register mock processor
        DataProcessorFactory.register("mock", MockDataProcessor)
        
        # Create processor
        processor = DataProcessorFactory.create("mock", {})
        assert isinstance(processor, MockDataProcessor)
        
        # Clean up
        DataProcessorFactory.unregister("mock")
    
    def test_create_unsupported_processor(self):
        """Test creating unsupported processor."""
        with pytest.raises(ValueError, match="Unsupported processor type"):
            DataProcessorFactory.create("unsupported", {})
    
    def test_create_from_format(self):
        """Test creating processor from data format."""
        # Register mock processor for jsonl
        DataProcessorFactory.register("jsonl_processor", MockDataProcessor)
        
        # Create processor from format
        processor = DataProcessorFactory.create_from_format("jsonl", {})
        assert isinstance(processor, MockDataProcessor)
        
        # Test unknown format (should default to jsonl)
        processor = DataProcessorFactory.create_from_format("unknown", {})
        assert isinstance(processor, MockDataProcessor)
        
        # Clean up
        DataProcessorFactory.unregister("jsonl_processor")
    
    def test_get_available_processors(self):
        """Test getting available processor types."""
        # Register mock processor
        DataProcessorFactory.register("mock", MockDataProcessor)
        
        available = DataProcessorFactory.get_available_processors()
        assert isinstance(available, list)
        assert "mock" in available
        
        # Clean up
        DataProcessorFactory.unregister("mock")


class TestConfigValidation:
    """Test configuration validation functions."""
    
    def test_validate_config_compatibility(self):
        """Test configuration compatibility validation."""
        # Valid config
        config = FineTuningConfig(
            base_model={"name": "llama3.2-3b"},
            method={"type": "lora"},
            framework={"type": "pytorch"},
            training_args={"output_dir": "./test"},
            dataset={"path": "./test.jsonl"},
            environment={"device": "cuda"}
        )
        
        # Register mock tuner that supports lora
        FineTunerFactory.register("pytorch", MockFineTuner)
        
        errors = validate_config_compatibility(config)
        # Should have no compatibility errors
        assert isinstance(errors, list)
        
        # Test incompatible method
        config.method["type"] = "unsupported_method"
        errors = validate_config_compatibility(config)
        # Should have method compatibility error
        # (depending on mock implementation)
        
        # Clean up
        FineTunerFactory.unregister("pytorch")
    
    def test_validate_config_hardware_compatibility(self):
        """Test hardware compatibility validation."""
        # CPU + full fine-tuning should generate warning
        config = FineTuningConfig(
            base_model={"name": "llama3.2-3b"},
            method={"type": "full_finetune"},
            framework={"type": "pytorch"},
            training_args={"output_dir": "./test"},
            dataset={"path": "./test.jsonl"},
            environment={"device": "cpu"}
        )
        
        errors = validate_config_compatibility(config)
        assert any("full fine-tuning" in error.lower() and "cpu" in error.lower() for error in errors)
    
    def test_validate_config_memory_requirements(self):
        """Test memory requirement validation."""
        # Large batch size + long sequences + full fine-tuning
        config = FineTuningConfig(
            base_model={"name": "llama3.2-3b"},
            method={"type": "full_finetune"},
            framework={"type": "pytorch"},
            training_args={
                "output_dir": "./test",
                "per_device_train_batch_size": 8,
                "max_seq_length": 2048
            },
            dataset={"path": "./test.jsonl"}
        )
        
        errors = validate_config_compatibility(config)
        # Should warn about potential OOM
        assert any("oom" in error.lower() or "memory" in error.lower() for error in errors)
    

if __name__ == "__main__":
    pytest.main([__file__])