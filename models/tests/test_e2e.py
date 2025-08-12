#!/usr/bin/env python3
"""
End-to-end integration tests for LlamaFarm Models.
These tests use actual APIs and models, so they require:
- Valid API keys in environment
- Running Ollama instance with models
- Network connectivity
"""

import os
import sys
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
import cli

# Load environment variables
load_dotenv()

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


class TestOpenAIEndToEnd:
    """End-to-end tests with actual OpenAI API."""
    
    def test_openai_api_key_available(self):
        """Test that OpenAI API key is available."""
        api_key = os.getenv('OPENAI_API_KEY')
        assert api_key is not None, "OPENAI_API_KEY environment variable not set"
        assert api_key.startswith('sk-'), "Invalid OpenAI API key format"
    
    @pytest.mark.skipif(not os.getenv('OPENAI_API_KEY'), reason="OpenAI API key not available")
    def test_openai_real_api_call(self):
        """Test actual OpenAI API call."""
        try:
            import openai
            
            # Create client without organization header to avoid mismatch
            client = openai.OpenAI(
                api_key=os.getenv('OPENAI_API_KEY'),
                organization=None  # Don't set organization header
            )
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": "Say 'Hello LlamaFarm!' and nothing else."}
                ],
                max_tokens=10,
                temperature=0
            )
            
            assert response.choices[0].message.content is not None
            assert "LlamaFarm" in response.choices[0].message.content
            
        except ImportError:
            pytest.skip("OpenAI package not installed")
        except openai.AuthenticationError as e:
            if "organization" in str(e).lower():
                pytest.skip("OpenAI organization configuration issue - test environment specific")
            else:
                pytest.fail(f"OpenAI API authentication failed: {e}")
        except Exception as e:
            pytest.fail(f"OpenAI API call failed: {e}")


class TestOllamaEndToEnd:
    """End-to-end tests with actual Ollama models."""
    
    def test_ollama_installed(self):
        """Test that Ollama is installed and available."""
        import subprocess
        
        try:
            result = subprocess.run(['ollama', '--version'], 
                                  capture_output=True, text=True, timeout=30)
            assert result.returncode == 0, "Ollama not installed or not working"
        except FileNotFoundError:
            pytest.skip("Ollama not installed")
        except subprocess.TimeoutExpired:
            pytest.fail("Ollama command timed out")
    
    def test_ollama_has_models(self):
        """Test that Ollama has models available."""
        import subprocess
        
        try:
            result = subprocess.run(['ollama', 'list'], 
                                  capture_output=True, text=True, timeout=30)
            assert result.returncode == 0, "Failed to list Ollama models"
            
            lines = result.stdout.strip().split('\n')
            # Should have at least header + 1 model
            assert len(lines) >= 2, "No models found in Ollama"
            
        except FileNotFoundError:
            pytest.skip("Ollama not installed")
        except subprocess.TimeoutExpired:
            pytest.fail("Ollama list command timed out")
    
    def test_ollama_api_connectivity(self):
        """Test Ollama API connectivity."""
        import requests
        
        ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        
        try:
            response = requests.get(f"{ollama_url}/api/tags", timeout=30)
            assert response.status_code == 200, f"Ollama API not responding: {response.status_code}"
            
            data = response.json()
            assert 'models' in data, "Invalid response from Ollama API"
            assert len(data['models']) > 0, "No models available in Ollama"
            
        except requests.RequestException as e:
            pytest.skip(f"Ollama API not accessible: {e}")
    
    @pytest.mark.skipif(not os.system('which ollama > /dev/null 2>&1') == 0, 
                       reason="Ollama not installed")
    def test_ollama_model_generation(self):
        """Test actual generation with an Ollama model."""
        import subprocess
        import requests
        
        # Get first available model
        try:
            result = subprocess.run(['ollama', 'list'], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                pytest.skip("Cannot list Ollama models")
            
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            if not lines or not lines[0].strip():
                pytest.skip("No Ollama models available")
            
            # Find first generative (non-embedding) model
            model_name = None
            for line in lines:
                if line.strip():
                    candidate = line.split()[0]
                    # Skip embedding models
                    if 'embed' not in candidate.lower():
                        model_name = candidate
                        break
            
            if not model_name:
                pytest.skip("No generative Ollama models available (only embedding models found)")
            
        except Exception as e:
            pytest.skip(f"Cannot determine Ollama models: {e}")
        
        # Test generation via API
        ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        
        try:
            # First check if model exists and is loaded
            models_response = requests.get(f"{ollama_url}/api/tags", timeout=30)
            if models_response.status_code != 200:
                pytest.skip("Cannot access Ollama API")
            
            models_data = models_response.json()
            available_models = [m['name'] for m in models_data.get('models', [])]
            
            if model_name not in available_models:
                pytest.skip(f"Model {model_name} not found in Ollama")
            
            # Test generation with the model
            response = requests.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": "Say 'Hello from Ollama!' and nothing else.",
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 400:
                # Check if it's a model loading issue
                error_text = response.text
                if "model" in error_text.lower() and ("not found" in error_text.lower() or "load" in error_text.lower()):
                    pytest.skip(f"Model {model_name} not properly loaded in Ollama")
                else:
                    pytest.fail(f"Ollama API error (400): {error_text}")
            
            assert response.status_code == 200, f"Generation failed: {response.status_code} - {response.text}"
            
            data = response.json()
            assert 'response' in data, "No response in generation result"
            assert data['response'], "Empty response from model"
            
        except requests.RequestException as e:
            pytest.skip(f"Cannot test Ollama generation: {e}")


class TestEndToEndWorkflows:
    """End-to-end workflow tests combining multiple components."""
    
    @pytest.fixture
    def test_config_with_real_providers(self):
        """Create test config with real providers."""
        config = {
            "name": "E2E Test Configuration",
            "version": "1.0.0",
            "default_provider": "openai_test",
            "providers": {
                "openai_test": {
                    "type": "cloud",
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "api_key": "${OPENAI_API_KEY}",
                    "base_url": "https://api.openai.com/v1",
                    "max_tokens": 50,
                    "temperature": 0.1
                }
            }
        }
        
        # Add Ollama if available
        if os.system('which ollama > /dev/null 2>&1') == 0:
            config["providers"]["ollama_test"] = {
                "type": "local",
                "provider": "ollama",
                "model": "llama3.2:3b",  # Assume this model exists
                "host": "localhost",
                "port": 11434,
                "timeout": 30
            }
            config["fallback_chain"] = ["openai_test", "ollama_test"]
        
        return config
    
    @pytest.fixture
    def temp_real_config_file(self, test_config_with_real_providers):
        """Create temporary config file with real providers."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_config_with_real_providers, f, indent=2)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        os.unlink(temp_path)
    
    @pytest.mark.skipif(not os.getenv('OPENAI_API_KEY'), reason="OpenAI API key not available")
    def test_openai_provider_real_test(self, temp_real_config_file):
        """Test the OpenAI provider with real API."""
        from argparse import Namespace
        
        args = Namespace(
            config=temp_real_config_file,
            provider="openai_test",
            query="Say hello in exactly 3 words."
        )
        
        # This will make a real API call
        try:
            cli.test_command(args)
        except SystemExit:
            pytest.fail("test_command failed with real OpenAI API")
    
    def test_config_validation_with_real_providers(self, temp_real_config_file):
        """Test configuration validation with real providers."""
        from argparse import Namespace
        
        args = Namespace(config=temp_real_config_file)
        
        try:
            cli.validate_config_command(args)
        except SystemExit:
            pytest.fail("Configuration validation failed")
    
    @pytest.mark.skipif(not os.system('which ollama > /dev/null 2>&1') == 0, 
                       reason="Ollama not available")
    def test_ollama_integration_e2e(self):
        """End-to-end test of Ollama integration."""
        from argparse import Namespace
        
        # Test list-local command
        args = Namespace()
        try:
            cli.list_local_command(args)
        except SystemExit:
            pytest.fail("list_local_command failed")
        
        # Get available models and test one if available
        import subprocess
        try:
            result = subprocess.run(['ollama', 'list'], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                if lines and lines[0].strip():
                    model_name = lines[0].split()[0]
                    
                    # Test the model
                    args = Namespace(
                        model=model_name,
                        query="Hello"
                    )
                    
                    try:
                        cli.test_local_command(args)
                    except SystemExit:
                        pytest.fail(f"test_local_command failed for model {model_name}")
        except Exception:
            pytest.skip("Cannot test Ollama models")
    
    @pytest.mark.skipif(not os.getenv('HF_TOKEN'), reason="HF_TOKEN not available")
    def test_huggingface_integration_e2e(self):
        """End-to-end test of Hugging Face integration."""
        if not cli.HF_AVAILABLE:
            pytest.skip("Hugging Face libraries not available")
        
        from argparse import Namespace
        
        # Test HF login
        args = Namespace()
        try:
            cli.hf_login_command(args)
        except SystemExit:
            pytest.fail("hf_login_command failed")
        
        # Test listing HF models
        args = Namespace(search="gpt2", limit=5)
        try:
            cli.list_hf_models_command(args)
        except SystemExit:
            pytest.fail("list_hf_models_command failed")


class TestPerformanceAndLimits:
    """Test performance characteristics and limits."""
    
    @pytest.mark.skipif(not os.getenv('OPENAI_API_KEY'), reason="OpenAI API key not available")
    def test_openai_response_time(self):
        """Test OpenAI API response time is reasonable."""
        import time
        import openai
        
        try:
            client = openai.OpenAI(
                api_key=os.getenv('OPENAI_API_KEY'),
                organization=None  # Avoid organization header issues
            )
            
            start_time = time.time()
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
                temperature=0
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Should respond within 10 seconds for a simple query
            assert response_time < 10.0, f"OpenAI API response too slow: {response_time:.2f}s"
            assert response.choices[0].message.content is not None
            
        except ImportError:
            pytest.skip("OpenAI package not installed")
        except openai.AuthenticationError as e:
            if "organization" in str(e).lower():
                pytest.skip("OpenAI organization configuration issue - test environment specific")
            else:
                pytest.fail(f"OpenAI API authentication failed: {e}")
        except Exception as e:
            pytest.fail(f"OpenAI performance test failed: {e}")
    
    @pytest.mark.skipif(not os.system('which ollama > /dev/null 2>&1') == 0, 
                       reason="Ollama not available")
    def test_ollama_response_time(self):
        """Test Ollama response time is reasonable."""
        import time
        import requests
        import subprocess
        
        # Get first available model
        try:
            result = subprocess.run(['ollama', 'list'], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                pytest.skip("Cannot list Ollama models")
            
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            if not lines or not lines[0].strip():
                pytest.skip("No Ollama models available")
            
            # Find first generative (non-embedding) model
            model_name = None
            for line in lines:
                if line.strip():
                    candidate = line.split()[0]
                    # Skip embedding models
                    if 'embed' not in candidate.lower():
                        model_name = candidate
                        break
            
            if not model_name:
                pytest.skip("No generative Ollama models available (only embedding models found)")
            
        except Exception:
            pytest.skip("Cannot determine Ollama models")
        
        ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        
        try:
            # First verify model is available
            models_response = requests.get(f"{ollama_url}/api/tags", timeout=30)
            if models_response.status_code != 200:
                pytest.skip("Cannot access Ollama API")
            
            models_data = models_response.json()
            available_models = [m['name'] for m in models_data.get('models', [])]
            
            if model_name not in available_models:
                pytest.skip(f"Model {model_name} not found in Ollama")
            
            start_time = time.time()
            
            response = requests.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": "Hi",
                    "stream": False
                },
                timeout=30
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 400:
                error_text = response.text
                if "model" in error_text.lower() and ("not found" in error_text.lower() or "load" in error_text.lower()):
                    pytest.skip(f"Model {model_name} not properly loaded in Ollama")
                else:
                    pytest.skip(f"Ollama API error (400): {error_text}")
            
            # Should respond within 30 seconds for local model
            assert response_time < 30.0, f"Ollama response too slow: {response_time:.2f}s"
            assert response.status_code == 200, f"Ollama request failed: {response.status_code} - {response.text}"
            
        except requests.RequestException:
            pytest.skip("Cannot test Ollama performance")


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-m", "integration", "--tb=short"])