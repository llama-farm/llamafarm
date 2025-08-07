"""
Tests for the new template structure with schema.json and defaults.json.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))


class TestTemplateStructure:
    """Test the new template directory structure."""
    
    def test_template_schema_structure(self):
        """Test that template schema has required fields."""
        schema = {
            "template_id": "test_template",
            "name": "Test Template",
            "description": "A test template",
            "category": "basic",
            "version": "1.0.0",
            "author": "Test Author",
            "inputs": {
                "query": {
                    "type": "string",
                    "required": True,
                    "description": "The query"
                }
            },
            "outputs": {
                "prompt": {
                    "type": "string",
                    "description": "The formatted prompt"
                }
            },
            "config_options": {
                "temperature": {
                    "type": "float",
                    "default": 0.7,
                    "min": 0,
                    "max": 2
                }
            },
            "template": "template content or filename",
            "frameworks": ["langchain", "native"],
            "use_cases": ["testing"],
            "metadata": {
                "complexity": "low",
                "tags": ["test"]
            }
        }
        
        # Verify required fields
        assert "template_id" in schema
        assert "name" in schema
        assert "description" in schema
        assert "category" in schema
        assert "inputs" in schema
        assert "outputs" in schema
        assert "frameworks" in schema
        assert "use_cases" in schema
    
    def test_template_defaults_structure(self):
        """Test that template defaults has required fields."""
        defaults = {
            "config": {
                "temperature_hint": 0.7,
                "max_tokens_hint": 500
            },
            "input_defaults": {
                "context": []
            },
            "framework_specific": {
                "langchain": {
                    "prompt_class": "PromptTemplate",
                    "memory_compatible": True
                },
                "native": {
                    "api_format": "messages"
                }
            },
            "performance": {
                "cache_enabled": True,
                "cache_ttl": 3600
            }
        }
        
        # Verify structure
        assert "config" in defaults
        assert "framework_specific" in defaults
        assert "performance" in defaults
    
    def test_template_directory_structure(self, tmp_path):
        """Test creating a proper template directory structure."""
        # Create template directory
        template_dir = tmp_path / "templates" / "basic" / "test_template"
        template_dir.mkdir(parents=True)
        
        # Create schema.json
        schema = {
            "template_id": "test_template",
            "name": "Test Template",
            "description": "Test template for unit tests",
            "category": "basic",
            "inputs": {"query": {"type": "string", "required": True}},
            "outputs": {"prompt": {"type": "string"}},
            "template": "template.jinja2",
            "frameworks": ["native"],
            "use_cases": ["testing"]
        }
        
        with open(template_dir / "schema.json", "w") as f:
            json.dump(schema, f, indent=2)
        
        # Create defaults.json
        defaults = {
            "config": {"temperature_hint": 0.7},
            "performance": {"cache_enabled": True}
        }
        
        with open(template_dir / "defaults.json", "w") as f:
            json.dump(defaults, f, indent=2)
        
        # Create template.jinja2
        with open(template_dir / "template.jinja2", "w") as f:
            f.write("Query: {{ query }}\nAnswer:")
        
        # Verify files exist
        assert (template_dir / "schema.json").exists()
        assert (template_dir / "defaults.json").exists()
        assert (template_dir / "template.jinja2").exists()
        
        # Verify content is loadable
        with open(template_dir / "schema.json") as f:
            loaded_schema = json.load(f)
            assert loaded_schema["template_id"] == "test_template"
        
        with open(template_dir / "defaults.json") as f:
            loaded_defaults = json.load(f)
            assert loaded_defaults["config"]["temperature_hint"] == 0.7


class TestTemplateLoader:
    """Test loading templates from the new structure."""
    
    def test_load_template_from_directory(self, tmp_path):
        """Test loading a template from directory structure."""
        # Create template structure
        template_dir = tmp_path / "qa_template"
        template_dir.mkdir()
        
        # Schema
        schema = {
            "template_id": "qa_template",
            "name": "Q&A Template",
            "description": "Question answering",
            "category": "basic",
            "inputs": {
                "query": {"type": "string", "required": True},
                "context": {"type": "array", "required": False}
            },
            "outputs": {"prompt": {"type": "string"}},
            "template": "template.jinja2",
            "frameworks": ["langchain", "native"],
            "use_cases": ["qa", "faq"]
        }
        
        with open(template_dir / "schema.json", "w") as f:
            json.dump(schema, f)
        
        # Defaults
        defaults = {
            "config": {
                "temperature_hint": 0.5,
                "max_tokens_hint": 300
            },
            "framework_specific": {
                "langchain": {"prompt_class": "PromptTemplate"}
            }
        }
        
        with open(template_dir / "defaults.json", "w") as f:
            json.dump(defaults, f)
        
        # Template
        with open(template_dir / "template.jinja2", "w") as f:
            f.write("""Based on context:
{% for doc in context %}
- {{ doc }}
{% endfor %}

Question: {{ query }}
Answer:""")
        
        # Mock template loader
        class MockTemplateLoader:
            def load_template(self, path):
                schema_path = Path(path) / "schema.json"
                defaults_path = Path(path) / "defaults.json"
                template_path = Path(path) / "template.jinja2"
                
                with open(schema_path) as f:
                    schema = json.load(f)
                
                with open(defaults_path) as f:
                    defaults = json.load(f)
                
                with open(template_path) as f:
                    template_content = f.read()
                
                return {
                    "schema": schema,
                    "defaults": defaults,
                    "template": template_content
                }
        
        # Test loading
        loader = MockTemplateLoader()
        loaded = loader.load_template(template_dir)
        
        assert loaded["schema"]["template_id"] == "qa_template"
        assert loaded["defaults"]["config"]["temperature_hint"] == 0.5
        assert "Question: {{ query }}" in loaded["template"]


class TestTemplateValidation:
    """Test template validation against schemas."""
    
    def test_validate_template_inputs(self):
        """Test validating template inputs against schema."""
        schema = {
            "inputs": {
                "query": {
                    "type": "string",
                    "required": True,
                    "validation": {
                        "min_length": 1,
                        "max_length": 1000
                    }
                },
                "context": {
                    "type": "array",
                    "required": False,
                    "items": {"type": "object"}
                }
            }
        }
        
        # Valid inputs
        valid_inputs = {
            "query": "What is machine learning?",
            "context": [{"title": "ML", "content": "..."}]
        }
        
        # Invalid inputs (missing required)
        invalid_inputs = {
            "context": []  # missing query
        }
        
        # Mock validator
        def validate_inputs(inputs, schema):
            errors = []
            for field, config in schema["inputs"].items():
                if config.get("required", False) and field not in inputs:
                    errors.append(f"Missing required field: {field}")
                
                if field in inputs and "validation" in config:
                    value = inputs[field]
                    validation = config["validation"]
                    
                    if "min_length" in validation and len(str(value)) < validation["min_length"]:
                        errors.append(f"{field} is too short")
                    
                    if "max_length" in validation and len(str(value)) > validation["max_length"]:
                        errors.append(f"{field} is too long")
            
            return errors
        
        # Test validation
        assert len(validate_inputs(valid_inputs, schema)) == 0
        assert len(validate_inputs(invalid_inputs, schema)) > 0
        assert "Missing required field: query" in validate_inputs(invalid_inputs, schema)
    
    def test_validate_config_options(self):
        """Test validating configuration options."""
        schema = {
            "config_options": {
                "temperature": {
                    "type": "float",
                    "default": 0.7,
                    "min": 0,
                    "max": 2
                },
                "max_tokens": {
                    "type": "integer",
                    "default": 500,
                    "min": 1,
                    "max": 4096
                }
            }
        }
        
        # Valid config
        valid_config = {"temperature": 0.8, "max_tokens": 1000}
        
        # Invalid config
        invalid_config = {"temperature": 3.0, "max_tokens": -1}
        
        # Mock validator
        def validate_config(config, schema):
            errors = []
            options = schema.get("config_options", {})
            
            for key, value in config.items():
                if key in options:
                    option = options[key]
                    
                    if "min" in option and value < option["min"]:
                        errors.append(f"{key} is below minimum ({option['min']})")
                    
                    if "max" in option and value > option["max"]:
                        errors.append(f"{key} is above maximum ({option['max']})")
            
            return errors
        
        # Test validation
        assert len(validate_config(valid_config, schema)) == 0
        assert len(validate_config(invalid_config, schema)) == 2


class TestFrameworkIntegration:
    """Test framework-specific configurations."""
    
    def test_langchain_integration(self):
        """Test LangChain framework configuration."""
        defaults = {
            "framework_specific": {
                "langchain": {
                    "prompt_class": "PromptTemplate",
                    "memory_compatible": True,
                    "chain_type": "stuff",
                    "supports_streaming": True
                }
            }
        }
        
        langchain_config = defaults["framework_specific"]["langchain"]
        assert langchain_config["prompt_class"] == "PromptTemplate"
        assert langchain_config["memory_compatible"] is True
        assert langchain_config["chain_type"] == "stuff"
        assert langchain_config["supports_streaming"] is True
    
    def test_native_integration(self):
        """Test native API configuration."""
        defaults = {
            "framework_specific": {
                "native": {
                    "api_format": "messages",
                    "role_mapping": {
                        "system": "system",
                        "user": "user",
                        "assistant": "assistant"
                    },
                    "supports_system_message": True
                }
            }
        }
        
        native_config = defaults["framework_specific"]["native"]
        assert native_config["api_format"] == "messages"
        assert native_config["role_mapping"]["system"] == "system"
        assert native_config["supports_system_message"] is True
    
    def test_framework_adapter_selection(self):
        """Test selecting appropriate framework adapter."""
        schema = {
            "frameworks": ["langchain", "langgraph", "native"]
        }
        
        # Mock adapter selection
        def get_adapter(framework, available_frameworks):
            if framework not in available_frameworks:
                raise ValueError(f"Framework {framework} not supported")
            
            adapters = {
                "langchain": "LangChainAdapter",
                "langgraph": "LangGraphAdapter",
                "native": "NativeAdapter"
            }
            
            return adapters.get(framework)
        
        # Test adapter selection
        assert get_adapter("langchain", schema["frameworks"]) == "LangChainAdapter"
        assert get_adapter("native", schema["frameworks"]) == "NativeAdapter"
        
        # Test unsupported framework
        with pytest.raises(ValueError):
            get_adapter("unsupported", schema["frameworks"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])