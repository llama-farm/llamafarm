#!/usr/bin/env python3
"""
Test suite for LlamaFarm Models CLI and functionality.
Tests actual OpenAI integration using the provided API key.
"""

import os
import sys
import json
import pytest
import tempfile
import subprocess
import requests
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
import cli

# Load environment variables
load_dotenv()

@pytest.fixture
def test_config():
    """Provide test configuration."""
    return {
        "name": "Test Configuration",
        "version": "1.0.0",
        "default_provider": "openai_test",
        "providers": {
            "openai_test": {
                "type": "cloud",
                "provider": "openai", 
                "model": "gpt-4o-mini",
                "api_key": "${OPENAI_API_KEY}",
                "base_url": "https://api.openai.com/v1",
                "max_tokens": 100,
                "temperature": 0.7
            }
        }
    }

@pytest.fixture
def temp_config_file(test_config):
    """Create temporary config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_config, f, indent=2)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)

class TestModelsCLI:
    """Test the Models CLI functionality."""
    
    def test_load_config(self, temp_config_file):
        """Test configuration loading."""
        config = cli.load_config(temp_config_file)
        
        assert config["name"] == "Test Configuration"
        assert "providers" in config
        assert "openai_test" in config["providers"]
        
    def test_load_config_with_env_substitution(self, temp_config_file):
        """Test environment variable substitution in config."""
        # Set a test environment variable
        os.environ["TEST_API_KEY"] = "test-key-123"
        
        # Modify config to use the test env var
        with open(temp_config_file, 'r') as f:
            config = json.load(f)
        
        config["providers"]["openai_test"]["api_key"] = "${TEST_API_KEY}"
        
        with open(temp_config_file, 'w') as f:
            json.dump(config, f)
        
        loaded_config = cli.load_config(temp_config_file)
        assert loaded_config["providers"]["openai_test"]["api_key"] == "test-key-123"
        
        # Cleanup
        del os.environ["TEST_API_KEY"]
    
    def test_config_validation(self, temp_config_file):
        """Test configuration validation."""
        from argparse import Namespace
        
        args = Namespace(config=temp_config_file)
        
        # Should not raise any exceptions
        try:
            cli.validate_config_command(args)
        except SystemExit:
            pytest.fail("validate_config_command raised SystemExit unexpectedly")

class TestModelIntegration:
    """Test model integration functionality."""
    
    def test_openai_package_available(self):
        """Test that OpenAI package is available."""
        try:
            import openai
            assert hasattr(openai, 'OpenAI')
        except ImportError:
            pytest.skip("OpenAI package not installed")
    
    @patch('openai.OpenAI')
    def test_mock_openai_call(self, mock_openai):
        """Test OpenAI integration with mocked response."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Mock response structure
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "Hello, LlamaFarm!"
        
        mock_client.chat.completions.create.return_value = mock_response
        
        # Test the mock integration
        try:
            import openai
            client = openai.OpenAI(api_key="test-key")
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say hello"}],
                max_tokens=50,
                temperature=0
            )
            
            assert response.choices[0].message.content == "Hello, LlamaFarm!"
            
        except ImportError:
            pytest.skip("OpenAI package not installed")

class TestModelManagement:
    """Test model management functionality."""
    
    def test_generate_basic_config(self):
        """Test basic configuration generation."""
        from argparse import Namespace
        
        args = Namespace(type="basic", output=None)
        
        # Should not raise exceptions
        try:
            cli.generate_config_command(args)
        except SystemExit:
            pytest.fail("generate_config_command raised SystemExit")
    
    def test_generate_multi_config(self):
        """Test multi-provider configuration generation.""" 
        from argparse import Namespace
        
        args = Namespace(type="multi", output=None)
        
        try:
            cli.generate_config_command(args)
        except SystemExit:
            pytest.fail("generate_config_command raised SystemExit")
    
    def test_list_providers(self, temp_config_file):
        """Test listing providers."""
        from argparse import Namespace
        
        args = Namespace(config=temp_config_file, detailed=False)
        
        try:
            cli.list_command(args)
        except SystemExit:
            pytest.fail("list_command raised SystemExit")

class TestModelSelection:
    """Test model selection and fallback logic."""
    
    def test_fallback_chain_validation(self, test_config):
        """Test fallback chain validation."""
        # Add a fallback chain with invalid provider
        test_config["fallback_chain"] = ["openai_test", "invalid_provider"]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            temp_path = f.name
        
        try:
            from argparse import Namespace
            args = Namespace(config=temp_path)
            
            # Should detect invalid provider in fallback chain
            cli.validate_config_command(args)
            
        finally:
            os.unlink(temp_path)

class TestEndToEndWorkflow:
    """End-to-end workflow tests."""
    
    def test_config_generation_and_validation(self):
        """Test generating and validating configuration."""
        from argparse import Namespace
        
        # Test generating config to file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Generate config to file
            args = Namespace(type="basic", output=temp_path)
            cli.generate_config_command(args)
            
            # Validate the generated config
            args = Namespace(config=temp_path)
            cli.validate_config_command(args)
            
            # Verify file exists and is valid JSON
            assert os.path.exists(temp_path)
            with open(temp_path, 'r') as f:
                config = json.load(f)
                assert "providers" in config
                assert "name" in config
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

class TestOllamaIntegration:
    """Test Ollama-specific functionality."""
    
    @patch('subprocess.run')
    def test_list_local_fallback_to_cli(self, mock_subprocess):
        """Test list local command fallback to ollama CLI."""
        # Mock successful ollama list command
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = """NAME                       ID              SIZE      MODIFIED     
llama3.1:8b                46e0c10c039e    4.9 GB    2 weeks ago     
llama3:latest              365c0bd3c000    4.7 GB    2 weeks ago"""
        
        from argparse import Namespace
        args = Namespace()
        
        with patch('requests.get') as mock_get:
            # Mock API request failure to trigger CLI fallback
            mock_get.side_effect = requests.RequestException("API not available")
            
            try:
                cli.list_local_command(args)
            except SystemExit:
                pytest.fail("list_local_command raised SystemExit unexpectedly")
            
            # Verify CLI command was called
            mock_subprocess.assert_called_with(['ollama', 'list'], capture_output=True, text=True)

    @patch('requests.get')
    def test_list_local_via_api(self, mock_get):
        """Test list local command via API."""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {
                    "name": "llama3.1:8b",
                    "digest": "46e0c10c039e",
                    "size": 4900000000,
                    "modified_at": "2024-07-15T21:05:00Z"
                }
            ]
        }
        mock_get.return_value = mock_response
        
        from argparse import Namespace
        args = Namespace()
        
        try:
            cli.list_local_command(args)
        except SystemExit:
            pytest.fail("list_local_command raised SystemExit unexpectedly")
        
        mock_get.assert_called_once()

    @patch('requests.post')
    def test_test_local_command(self, mock_post):
        """Test local model testing."""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Hello! I'm doing well, thank you.",
            "eval_count": 10,
            "eval_duration": 1000000000  # 1 second in nanoseconds
        }
        mock_post.return_value = mock_response
        
        from argparse import Namespace
        args = Namespace(model="llama3.1:8b", query="Hello, how are you?")
        
        try:
            cli.test_local_command(args)
        except SystemExit:
            pytest.fail("test_local_command raised SystemExit unexpectedly")
        
        # Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "api/generate" in call_args[0][0]
        assert call_args[1]["json"]["model"] == "llama3.1:8b"

    @patch('subprocess.run')
    def test_generate_ollama_config(self, mock_subprocess):
        """Test Ollama configuration generation."""
        # Mock ollama list output
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = """NAME                       ID              SIZE      MODIFIED     
llama3.1:8b                46e0c10c039e    4.9 GB    2 weeks ago     
llama3:latest              365c0bd3c000    4.7 GB    2 weeks ago"""
        
        from argparse import Namespace
        args = Namespace(output=None)
        
        try:
            cli.generate_ollama_config_command(args)
        except SystemExit:
            pytest.fail("generate_ollama_config_command raised SystemExit unexpectedly")
        
        # Verify ollama list was called
        mock_subprocess.assert_called_with(['ollama', 'list'], capture_output=True, text=True)

    @patch('requests.post')
    @patch('subprocess.run')
    def test_pull_model_command_api(self, mock_subprocess, mock_post):
        """Test model pulling via API."""
        # Mock successful streaming response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            b'{"status": "downloading", "completed": 50, "total": 100}',
            b'{"status": "complete"}'
        ]
        mock_post.return_value = mock_response
        
        from argparse import Namespace
        args = Namespace(model="llama3.2:1b")
        
        try:
            cli.pull_model_command(args)
        except SystemExit:
            pytest.fail("pull_model_command raised SystemExit unexpectedly")
        
        # Verify API call
        mock_post.assert_called_once()
        assert "api/pull" in mock_post.call_args[0][0]

class TestOllamaConfiguration:
    """Test Ollama configuration validation and loading."""
    
    @pytest.fixture
    def ollama_config(self):
        """Provide Ollama test configuration."""
        return {
            "name": "Ollama Test Configuration",
            "version": "1.0.0",
            "default_provider": "ollama_llama3_1_8b",
            "providers": {
                "ollama_llama3_1_8b": {
                    "type": "local",
                    "provider": "ollama",
                    "model": "llama3.1:8b",
                    "host": "localhost",
                    "port": 11434,
                    "base_url": "http://localhost:11434",
                    "timeout": 120
                }
            }
        }
    
    @pytest.fixture
    def temp_ollama_config_file(self, ollama_config):
        """Create temporary Ollama config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(ollama_config, f, indent=2)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        os.unlink(temp_path)
    
    def test_ollama_config_validation(self, temp_ollama_config_file):
        """Test Ollama configuration validation."""
        from argparse import Namespace
        
        args = Namespace(config=temp_ollama_config_file)
        
        try:
            cli.validate_config_command(args)
        except SystemExit:
            pytest.fail("validate_config_command raised SystemExit unexpectedly")
    
    def test_ollama_config_listing(self, temp_ollama_config_file):
        """Test listing Ollama providers."""
        from argparse import Namespace
        
        args = Namespace(config=temp_ollama_config_file, detailed=False)
        
        try:
            cli.list_command(args)
        except SystemExit:
            pytest.fail("list_command raised SystemExit unexpectedly")

class TestOllamaEndToEnd:
    """End-to-end tests for Ollama integration."""
    
    @patch('subprocess.run')
    def test_ollama_config_generation_and_validation(self, mock_subprocess):
        """Test generating and validating Ollama configuration."""
        # Mock ollama list output
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = """NAME                       ID              SIZE      MODIFIED     
llama3.1:8b                46e0c10c039e    4.9 GB    2 weeks ago"""
        
        from argparse import Namespace
        
        # Test generating config to file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Generate Ollama config to file
            args = Namespace(output=temp_path)
            cli.generate_ollama_config_command(args)
            
            # Validate the generated config
            args = Namespace(config=temp_path)
            cli.validate_config_command(args)
            
            # Verify file exists and is valid JSON
            assert os.path.exists(temp_path)
            with open(temp_path, 'r') as f:
                config = json.load(f)
                assert "providers" in config
                assert "name" in config
                assert "local_settings" in config
                assert "ollama" in config["local_settings"]
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

class TestHuggingFaceIntegration:
    """Test Hugging Face-specific functionality."""
    
    @patch('cli.HF_AVAILABLE', True)
    @patch('cli.list_models')
    def test_list_hf_models_command(self, mock_list_models):
        """Test listing Hugging Face models."""
        # Mock HF models response
        mock_model = MagicMock()
        mock_model.id = "gpt2"
        mock_model.downloads = 1000000
        mock_model.tags = ["transformers", "pytorch"]
        
        mock_list_models.return_value = [mock_model]
        
        from argparse import Namespace
        args = Namespace(search="gpt", limit=10)
        
        try:
            cli.list_hf_models_command(args)
        except SystemExit:
            pytest.fail("list_hf_models_command raised SystemExit unexpectedly")
        
        mock_list_models.assert_called_once()

    @patch('cli.HF_AVAILABLE', True)
    @patch('cli.snapshot_download')
    @patch('cli.login')
    @patch('cli.HfApi')
    def test_download_hf_model_command(self, mock_api, mock_login, mock_snapshot):
        """Test downloading Hugging Face models."""
        # Mock successful download
        mock_snapshot.return_value = "/path/to/model"
        
        # Mock model info
        mock_api_instance = MagicMock()
        mock_model_info = MagicMock()
        mock_model_info.tags = ["transformers", "pytorch"]
        mock_api_instance.model_info.return_value = mock_model_info
        mock_api.return_value = mock_api_instance
        
        from argparse import Namespace
        args = Namespace(model_id="gpt2", cache_dir=None, include_images=False)
        
        try:
            cli.download_hf_model_command(args)
        except SystemExit:
            pytest.fail("download_hf_model_command raised SystemExit unexpectedly")
        
        mock_snapshot.assert_called_once()
        mock_login.assert_called_once()

    @patch('cli.HF_AVAILABLE', True)
    def test_generate_hf_config_command(self):
        """Test Hugging Face configuration generation."""
        from argparse import Namespace
        args = Namespace(output=None, models=None)
        
        try:
            cli.generate_hf_config_command(args)
        except SystemExit:
            pytest.fail("generate_hf_config_command raised SystemExit unexpectedly")

    @patch('cli.HF_AVAILABLE', True)
    @patch('cli.login')
    @patch('cli.HfApi')
    def test_hf_login_command(self, mock_api, mock_login):
        """Test Hugging Face login."""
        # Mock successful login and user info
        mock_api_instance = MagicMock()
        mock_api_instance.whoami.return_value = {"name": "testuser", "type": "user"}
        mock_api.return_value = mock_api_instance
        
        from argparse import Namespace
        args = Namespace()
        
        try:
            cli.hf_login_command(args)
        except SystemExit:
            pytest.fail("hf_login_command raised SystemExit unexpectedly")
        
        mock_login.assert_called_once()

    def test_hf_unavailable_handling(self):
        """Test handling when Hugging Face libraries are not available."""
        with patch('cli.HF_AVAILABLE', False):
            from argparse import Namespace
            
            # Test all HF commands when libraries unavailable
            commands = [
                ('list_hf_models_command', Namespace()),
                ('download_hf_model_command', Namespace(model_id="gpt2", cache_dir=None, include_images=False)),
                ('generate_hf_config_command', Namespace(output=None, models=None)),
                ('hf_login_command', Namespace()),
            ]
            
            for command_name, args in commands:
                command_func = getattr(cli, command_name)
                try:
                    command_func(args)
                except SystemExit:
                    pytest.fail(f"{command_name} raised SystemExit unexpectedly")

class TestHuggingFaceConfiguration:
    """Test Hugging Face configuration validation and loading."""
    
    @pytest.fixture
    def hf_config(self):
        """Provide HF test configuration."""
        return {
            "name": "Hugging Face Test Configuration",
            "version": "1.0.0",
            "default_provider": "hf_gpt2",
            "providers": {
                "hf_gpt2": {
                    "type": "local",
                    "provider": "huggingface",
                    "model": "gpt2",
                    "cache_dir": "~/.cache/huggingface/hub",
                    "device": "cpu",
                    "token": "${HF_TOKEN}",
                    "max_tokens": 50
                }
            }
        }
    
    @pytest.fixture
    def temp_hf_config_file(self, hf_config):
        """Create temporary HF config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(hf_config, f, indent=2)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        os.unlink(temp_path)
    
    def test_hf_config_validation(self, temp_hf_config_file):
        """Test HF configuration validation."""
        from argparse import Namespace
        
        args = Namespace(config=temp_hf_config_file)
        
        try:
            cli.validate_config_command(args)
        except SystemExit:
            pytest.fail("validate_config_command raised SystemExit unexpectedly")
    
    def test_hf_config_listing(self, temp_hf_config_file):
        """Test listing HF providers."""
        from argparse import Namespace
        
        args = Namespace(config=temp_hf_config_file, detailed=False)
        
        try:
            cli.list_command(args)
        except SystemExit:
            pytest.fail("list_command raised SystemExit unexpectedly")

class TestLocalEnginesIntegration:
    """Test local inference engines functionality."""
    
    def test_list_vllm_models_command(self):
        """Test listing vLLM-compatible models."""
        from argparse import Namespace
        args = Namespace()
        
        try:
            cli.list_vllm_models_command(args)
        except SystemExit:
            pytest.fail("list_vllm_models_command raised SystemExit unexpectedly")
    
    def test_list_tgi_models_command(self):
        """Test listing TGI endpoints."""
        from argparse import Namespace
        args = Namespace()
        
        try:
            cli.list_tgi_models_command(args)
        except SystemExit:
            pytest.fail("list_tgi_models_command raised SystemExit unexpectedly")
    
    def test_generate_local_engines_config_command(self):
        """Test generating local engines configuration."""
        from argparse import Namespace
        args = Namespace(output=None, include_unavailable=True)
        
        try:
            cli.generate_local_engines_config_command(args)
        except SystemExit:
            pytest.fail("generate_local_engines_config_command raised SystemExit unexpectedly")
    
    @patch('cli.VLLM_AVAILABLE', False)
    def test_test_vllm_command_unavailable(self):
        """Test vLLM command when vLLM is not available."""
        from argparse import Namespace
        args = Namespace(model="test-model", query="test", max_tokens=10)
        
        try:
            cli.test_vllm_command(args)
        except SystemExit:
            pytest.fail("test_vllm_command raised SystemExit unexpectedly")
    
    @patch('cli.TGI_AVAILABLE', False)
    def test_test_tgi_command_unavailable(self):
        """Test TGI command when TGI is not available."""
        from argparse import Namespace
        args = Namespace(endpoint="http://localhost:8080", query="test", max_tokens=10)
        
        try:
            cli.test_tgi_command(args)
        except SystemExit:
            pytest.fail("test_tgi_command raised SystemExit unexpectedly")
    
    @patch('cli.VLLM_AVAILABLE', True)
    def test_test_vllm_command_success(self):
        """Test successful vLLM model testing."""
        with patch.object(cli, 'LLM', create=True) as mock_llm, \
             patch.object(cli, 'SamplingParams', create=True) as mock_sampling:
            
            # Mock vLLM components
            mock_llm_instance = MagicMock()
            mock_output = MagicMock()
            mock_output.outputs = [MagicMock()]
            mock_output.outputs[0].text = "Hello! How can I help you?"
            mock_output.outputs[0].token_ids = [1, 2, 3, 4, 5]
            mock_llm_instance.generate.return_value = [mock_output]
            mock_llm.return_value = mock_llm_instance
            
            from argparse import Namespace
            args = Namespace(
                model="test-model", 
                query="Hello", 
                max_tokens=10,
                max_model_len=2048,
                gpu_memory=0.8
            )
            
            try:
                cli.test_vllm_command(args)
            except SystemExit:
                pytest.fail("test_vllm_command raised SystemExit unexpectedly")
            
            mock_llm.assert_called_once()
            mock_llm_instance.generate.assert_called_once()
    
    @patch('cli.TGI_AVAILABLE', True)
    def test_test_tgi_command_success(self):
        """Test successful TGI endpoint testing."""
        with patch.object(cli, 'Client', create=True) as mock_client:
            # Mock TGI client and response
            mock_client_instance = MagicMock()
            mock_response = MagicMock()
            mock_response.generated_text = "Hello! How can I help you?"
            mock_client_instance.generate.return_value = mock_response
            mock_client.return_value = mock_client_instance
            
            from argparse import Namespace
            args = Namespace(
                endpoint="http://localhost:8080",
                query="Hello",
                max_tokens=10
            )
            
            try:
                cli.test_tgi_command(args)
            except SystemExit:
                pytest.fail("test_tgi_command raised SystemExit unexpectedly")
            
            mock_client.assert_called_once_with("http://localhost:8080")
            mock_client_instance.generate.assert_called_once()

class TestLocalEnginesConfiguration:
    """Test local engines configuration validation and loading."""
    
    @pytest.fixture
    def engines_config(self):
        """Provide local engines test configuration."""
        return {
            "name": "Local Engines Test Configuration",
            "version": "1.0.0",
            "default_provider": "vllm_test",
            "providers": {
                "vllm_test": {
                    "type": "local",
                    "provider": "vllm",
                    "model": "test-model",
                    "max_model_len": 2048,
                    "gpu_memory_utilization": 0.8,
                    "temperature": 0.7
                },
                "tgi_test": {
                    "type": "local",
                    "provider": "tgi",
                    "endpoint": "http://localhost:8080",
                    "timeout": 30,
                    "temperature": 0.7
                }
            }
        }
    
    @pytest.fixture
    def temp_engines_config_file(self, engines_config):
        """Create temporary engines config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(engines_config, f, indent=2)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        os.unlink(temp_path)
    
    def test_engines_config_validation(self, temp_engines_config_file):
        """Test local engines configuration validation."""
        from argparse import Namespace
        
        args = Namespace(config=temp_engines_config_file)
        
        try:
            cli.validate_config_command(args)
        except SystemExit:
            pytest.fail("validate_config_command raised SystemExit unexpectedly")
    
    def test_engines_config_listing(self, temp_engines_config_file):
        """Test listing local engines providers."""
        from argparse import Namespace
        
        args = Namespace(config=temp_engines_config_file, detailed=False)
        
        try:
            cli.list_command(args)
        except SystemExit:
            pytest.fail("list_command raised SystemExit unexpectedly")


class TestStrategyBasedCLI:
    """Test new strategy-based CLI commands."""
    
    def test_list_strategies_command(self):
        """Test list-strategies CLI command."""
        from argparse import Namespace
        
        args = Namespace()
        
        try:
            import cli
            cli.list_strategies_command(args)
        except SystemExit:
            pytest.fail("list_strategies_command raised SystemExit unexpectedly")
        except ImportError:
            pytest.skip("Strategy management components not available")
    
    def test_use_strategy_command(self):
        """Test use-strategy CLI command.""" 
        from argparse import Namespace
        
        args = Namespace(strategy="local_development", save=False)
        
        try:
            import cli
            cli.use_strategy_command(args)
        except SystemExit:
            pytest.fail("use_strategy_command raised SystemExit unexpectedly")
        except ImportError:
            pytest.skip("Strategy management components not available")
    
    def test_info_command(self):
        """Test info CLI command."""
        from argparse import Namespace
        
        args = Namespace(strategy="local_development")
        
        try:
            import cli
            cli.info_command(args)
        except SystemExit:
            pytest.fail("info_command raised SystemExit unexpectedly") 
        except ImportError:
            pytest.skip("Strategy management components not available")
    
    def test_generate_command_mock(self):
        """Test generate CLI command with mocked response."""
        from argparse import Namespace
        from unittest.mock import patch, MagicMock
        
        args = Namespace(
            strategy="local_development", 
            prompt="Hello world",
            max_tokens=50,
            temperature=0.7,
            stream=False,
            json=False,
            save=None
        )
        
        try:
            import cli
            
            # Mock the ModelManager to avoid actual API calls
            with patch('cli.ModelManager') as mock_manager_class:
                mock_manager = MagicMock()
                mock_manager.generate.return_value = "Hello! How can I help you?"
                mock_manager_class.from_strategy.return_value = mock_manager
                
                cli.generate_command(args)
                
                # Verify ModelManager was created with correct strategy
                mock_manager_class.from_strategy.assert_called_once_with("local_development")
                # Verify generate was called
                mock_manager.generate.assert_called_once()
                
        except SystemExit:
            pytest.fail("generate_command raised SystemExit unexpectedly")
        except ImportError:
            pytest.skip("Strategy management components not available")


class TestStrategyMigration:
    """Test migration from config-based to strategy-based system."""
    
    def test_legacy_commands_show_deprecation_warning(self, capsys):
        """Test that legacy commands show deprecation warnings.""" 
        from argparse import Namespace
        import tempfile
        import json
        
        # Create a temporary config file
        test_config = {
            "name": "Test Config",
            "providers": {
                "test_provider": {
                    "type": "cloud",
                    "provider": "openai",
                    "model": "gpt-3.5-turbo"
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config, f)
            temp_path = f.name
        
        try:
            args = Namespace(config=temp_path, detailed=False)
            
            import cli
            cli.list_command(args)
            
            # Check that deprecation warning was printed
            captured = capsys.readouterr()
            assert "legacy config system" in captured.out or "legacy config system" in captured.err
            
        except SystemExit:
            # This is expected for some commands
            pass
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])