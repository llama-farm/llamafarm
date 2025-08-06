"""Integration tests for complete workflows."""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from core import ModelManager
from core.strategy_manager import StrategyManager
from components.factory import ModelAppFactory, CloudAPIFactory, FineTunerFactory


class TestCompleteWorkflows:
    """Test complete end-to-end workflows."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def mock_ollama(self):
        """Mock Ollama responses."""
        with patch('requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "response": "This is a test response from Ollama."
            }
            mock_response.status_code = 200
            mock_post.return_value = mock_response
            yield mock_post
    
    @pytest.fixture
    def mock_openai(self):
        """Mock OpenAI responses."""
        with patch('openai.OpenAI') as mock_client:
            instance = MagicMock()
            mock_client.return_value = instance
            
            # Mock chat completion
            completion = MagicMock()
            completion.choices = [MagicMock(message=MagicMock(content="Test response from OpenAI"))]
            instance.chat.completions.create.return_value = completion
            
            yield instance
    
    def test_strategy_loading_and_execution(self):
        """Test loading a strategy and executing a query."""
        # Load strategy
        strategy_manager = StrategyManager()
        strategies = strategy_manager.list_strategies()
        
        # Should have strategies available
        assert len(strategies) > 0
        
        # Find a local strategy
        local_strategies = [s for s in strategies if s.get('source') == 'default' and 'model_app' in s.get('components', [])]
        assert len(local_strategies) > 0
        
        # Load the local development strategy
        strategy_config = strategy_manager.load_strategy('local_development')
        assert 'model_app' in strategy_config
        assert strategy_config['model_app']['type'] == 'ollama'
    
    def test_cloud_fallback_workflow(self, mock_openai, mock_ollama):
        """Test cloud API with local fallback workflow."""
        # Initialize with hybrid strategy
        manager = ModelManager.from_strategy('hybrid_with_fallback')
        
        # Test query - should try cloud first, then fallback
        response = manager.generate("What is AI?")
        assert response is not None
        assert len(response) > 0
    
    def test_medical_strategy_workflow(self, mock_ollama):
        """Test medical specialist strategy workflow."""
        # Initialize with medical strategy
        manager = ModelManager.from_strategy('medical_specialist')
        
        # Test medical query
        query = "What are the symptoms of the common cold?"
        response = manager.generate(query)
        
        assert response is not None
        # Medical responses should include disclaimer
        assert any(word in response.lower() for word in ['consult', 'medical', 'professional', 'advice'])
    
    def test_code_assistant_strategy_workflow(self, mock_ollama):
        """Test code assistant strategy workflow."""
        # Initialize with code assistant strategy
        manager = ModelManager.from_strategy('code_assistant')
        
        # Test code generation query
        query = "Write a Python function to calculate factorial"
        response = manager.generate(query)
        
        assert response is not None
        assert 'def' in response or 'function' in response.lower()
    
    def test_fine_tuning_strategy_loading(self):
        """Test loading fine-tuning strategies."""
        strategy_manager = StrategyManager()
        
        # Load M1 fine-tuning strategy
        strategy = strategy_manager.load_strategy('m1_fine_tuning')
        assert 'fine_tuner' in strategy
        assert strategy['fine_tuner']['type'] == 'pytorch'
        assert strategy['fine_tuner']['config']['training_args']['device'] == 'mps'
        
        # Load CUDA fine-tuning strategy
        strategy = strategy_manager.load_strategy('cuda_fine_tuning')
        assert strategy['fine_tuner']['config']['training_args']['device'] == 'cuda'
        
        # Load CPU fine-tuning strategy
        strategy = strategy_manager.load_strategy('cpu_fine_tuning')
        assert strategy['fine_tuner']['config']['training_args']['device'] == 'cpu'
    
    def test_multi_model_workflow(self, mock_openai):
        """Test using different models for different tasks."""
        # This simulates demo2 functionality
        strategy_manager = StrategyManager()
        
        # Different task types should use different models
        tasks = [
            {"type": "simple", "query": "What is 2+2?"},
            {"type": "complex", "query": "Explain quantum computing"},
            {"type": "creative", "query": "Write a poem about AI"}
        ]
        
        for task in tasks:
            # In real implementation, would select model based on task type
            manager = ModelManager.from_strategy('cloud_production')
            response = manager.generate(task['query'])
            assert response is not None
    
    def test_component_registration(self):
        """Test that all required components are registered."""
        # Check model apps
        assert 'ollama' in ModelAppFactory.list_components()
        
        # Check cloud APIs
        assert 'openai' in CloudAPIFactory.list_components()
        
        # Check fine-tuners
        assert 'pytorch' in FineTunerFactory.list_components()
        assert 'llamafactory' in FineTunerFactory.list_components()
    
    def test_fallback_chain_execution(self, mock_ollama):
        """Test fallback chain execution when primary fails."""
        manager = ModelManager.from_strategy('hybrid_with_fallback')
        
        # Mock first call to fail
        with patch.object(manager, '_try_cloud_api', return_value=None):
            # Should fall back to local model
            response = manager.generate("Test query")
            assert response is not None
    
    def test_strategy_override_functionality(self):
        """Test overriding strategy configurations."""
        strategy_manager = StrategyManager()
        
        # Load strategy with overrides
        overrides = {
            "model_app": {
                "config": {
                    "default_model": "llama3.2:1b"  # Override to smaller model
                }
            }
        }
        
        strategy = strategy_manager.load_strategy('local_development', overrides)
        assert strategy['model_app']['config']['default_model'] == 'llama3.2:1b'
    
    def test_model_catalog_integration(self):
        """Test model catalog functionality."""
        from model_catalog import ModelCatalog
        
        catalog = ModelCatalog()
        
        # Test getting model info
        model_info = catalog.get_model_info('llama3.2:3b')
        assert model_info is not None
        assert 'category' in model_info
        
        # Test fallback chains
        medical_chain = catalog.get_fallback_chain('medical_chain')
        assert len(medical_chain) > 0
        assert medical_chain[0] == 'hf.co/mradermacher/DeepSeek-R1-Medicalai-923-i1-GGUF:Q4_K_M'
    
    @pytest.mark.parametrize("strategy_name", [
        "local_development",
        "cloud_production", 
        "hybrid_with_fallback",
        "medical_specialist",
        "code_assistant"
    ])
    def test_all_strategies_loadable(self, strategy_name):
        """Test that all documented strategies can be loaded."""
        strategy_manager = StrategyManager()
        strategy = strategy_manager.load_strategy(strategy_name)
        assert strategy is not None
        assert 'strategy' in strategy
        assert strategy['strategy'] == strategy_name


class TestErrorHandling:
    """Test error handling in workflows."""
    
    def test_missing_api_key_handling(self):
        """Test handling missing API keys."""
        # Remove API key if set
        old_key = os.environ.pop('OPENAI_API_KEY', None)
        
        try:
            manager = ModelManager.from_strategy('cloud_production')
            # Should handle gracefully
            response = manager.generate("Test")
            # Should either fail gracefully or use fallback
            assert response is None or "error" in response.lower()
        finally:
            if old_key:
                os.environ['OPENAI_API_KEY'] = old_key
    
    def test_invalid_strategy_handling(self):
        """Test handling invalid strategy names."""
        strategy_manager = StrategyManager()
        
        with pytest.raises(ValueError) as exc_info:
            strategy_manager.load_strategy('non_existent_strategy')
        
        assert "Strategy not found" in str(exc_info.value)
    
    def test_component_initialization_failure(self):
        """Test handling component initialization failures."""
        # Try to create with invalid config
        with pytest.raises(ValueError):
            ModelAppFactory.create({"type": "invalid_type"})


class TestCLIIntegration:
    """Test CLI command integration."""
    
    def test_cli_strategy_listing(self):
        """Test CLI strategy listing functionality."""
        from cli import manage_finetune_strategies
        
        # Mock args for list command
        class Args:
            strategies_command = 'list'
        
        # Should not raise any exceptions
        # In real test would capture output
        try:
            manage_finetune_strategies(Args())
        except SystemExit:
            pass  # CLI might exit normally
    
    def test_cli_model_catalog(self):
        """Test CLI model catalog functionality."""
        from cli import catalog_command
        
        # Mock args for catalog list
        class Args:
            catalog_command = 'list'
            category = None
            detailed = False
        
        # Should not raise any exceptions
        try:
            catalog_command(Args())
        except SystemExit:
            pass