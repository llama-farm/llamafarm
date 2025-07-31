"""Tests for the main prompt system."""

import pytest
from pathlib import Path

from prompts.core.prompt_system import PromptSystem
from prompts.models.config import PromptConfig, GlobalPromptConfig
from prompts.models.context import PromptContext
from prompts.models.template import PromptTemplate, TemplateType, TemplateComplexity, TemplateMetadata
from prompts.models.strategy import PromptStrategy, StrategyType, StrategyRule, RuleOperator


@pytest.fixture
def sample_template():
    """Create a sample template for testing."""
    return PromptTemplate(
        template_id="test_template",
        name="Test Template",
        type=TemplateType.BASIC,
        template="Query: {{ query }}\\nContext: {{ context }}\\nAnswer:",
        input_variables=["query", "context"],
        metadata=TemplateMetadata(
            use_case="testing",
            complexity=TemplateComplexity.LOW,
            domain="test"
        )
    )


@pytest.fixture
def sample_strategy():
    """Create a sample strategy for testing."""
    rule = StrategyRule(
        rule_id="test_rule",
        field="domain",
        operator=RuleOperator.EQUALS,
        value="test",
        template_id="test_template",
        priority=10
    )
    
    return PromptStrategy(
        strategy_id="test_strategy",
        name="Test Strategy",
        type=StrategyType.RULE_BASED,
        rules=[rule],
        fallback_template="test_template"
    )


@pytest.fixture
def sample_global_prompt():
    """Create a sample global prompt for testing."""
    return GlobalPromptConfig(
        global_id="test_global",
        name="Test Global Prompt",
        system_prompt="You are a test assistant.",
        applies_to=["*"],
        priority=50
    )


@pytest.fixture
def sample_config(sample_template, sample_strategy, sample_global_prompt):
    """Create a sample configuration for testing."""
    return PromptConfig(
        name="Test Configuration",
        version="1.0.0",
        default_strategy="test_strategy",
        templates={"test_template": sample_template},
        strategies={"test_strategy": sample_strategy},
        global_prompts=[sample_global_prompt]
    )


@pytest.fixture
def prompt_system(sample_config):
    """Create a prompt system for testing."""
    return PromptSystem(sample_config)


class TestPromptSystem:
    """Test the main PromptSystem class."""
    
    def test_initialization(self, prompt_system):
        """Test prompt system initialization."""
        assert prompt_system is not None
        assert prompt_system.execution_count == 0
        assert prompt_system.error_count == 0
        assert prompt_system.fallback_count == 0
    
    def test_template_management(self, prompt_system, sample_template):
        """Test template management operations."""
        # Test getting template
        template = prompt_system.get_template("test_template")
        assert template is not None
        assert template.template_id == "test_template"
        
        # Test listing templates
        templates = prompt_system.list_templates()
        assert len(templates) == 1
        assert templates[0].template_id == "test_template"
        
        # Test adding new template
        new_template = PromptTemplate(
            template_id="new_template",
            name="New Template",
            type=TemplateType.BASIC,
            template="New template: {{ query }}",
            input_variables=["query"],
            metadata=TemplateMetadata(use_case="testing")
        )
        
        success = prompt_system.add_template(new_template)
        assert success is True
        
        templates = prompt_system.list_templates()
        assert len(templates) == 2
    
    def test_template_search(self, prompt_system):
        """Test template search functionality."""
        results = prompt_system.search_templates("test")
        assert len(results) == 1
        assert results[0].template_id == "test_template"
        
        results = prompt_system.search_templates("nonexistent")
        assert len(results) == 0
    
    def test_template_validation(self, prompt_system):
        """Test template validation."""
        # Valid template
        template = prompt_system.get_template("test_template")
        errors = prompt_system.validate_template(template)
        assert len(errors) == 0
        
        # Invalid template (missing content)
        invalid_template = PromptTemplate(
            template_id="invalid",
            name="Invalid",
            type=TemplateType.BASIC,
            template="",  # Empty template
            input_variables=[],
            metadata=TemplateMetadata(use_case="testing")
        )
        
        errors = prompt_system.validate_template(invalid_template)
        assert len(errors) > 0
    
    def test_basic_execution(self, prompt_system):
        """Test basic prompt execution."""
        query = "What is testing?"
        context = PromptContext(
            query=query,
            domain="test",
            documents=[{"content": "Testing is important"}]
        )
        
        exec_context = prompt_system.execute_prompt(query, context)
        
        assert exec_context is not None
        assert not exec_context.has_errors()
        assert exec_context.selected_template_id == "test_template"
        assert exec_context.selected_strategy_id == "test_strategy"
        assert exec_context.rendered_prompt is not None
        assert query in exec_context.rendered_prompt
    
    def test_execution_with_variables(self, prompt_system):
        """Test execution with additional variables."""
        query = "Test query"
        variables = {"custom_var": "custom_value"}
        context = PromptContext(query=query, domain="test")
        
        exec_context = prompt_system.execute_prompt(
            query, 
            context, 
            variables=variables
        )
        
        assert not exec_context.has_errors()
        assert "custom_var" in exec_context.input_variables
        assert exec_context.input_variables["custom_var"] == "custom_value"
    
    def test_template_override(self, prompt_system):
        """Test template override functionality."""
        query = "Test query"
        context = PromptContext(query=query, domain="other")  # Different domain
        
        # Without override, should use fallback
        exec_context = prompt_system.execute_prompt(query, context)
        assert exec_context.selected_template_id == "test_template"  # Fallback
        
        # With override, should use specified template
        exec_context = prompt_system.execute_prompt(
            query, 
            context, 
            template_override="test_template"
        )
        assert exec_context.selected_template_id == "test_template"
    
    def test_strategy_override(self, prompt_system):
        """Test strategy override functionality."""
        query = "Test query"
        context = PromptContext(query=query, domain="test")
        
        exec_context = prompt_system.execute_prompt(
            query, 
            context, 
            strategy_override="test_strategy"
        )
        
        assert exec_context.selected_strategy_id == "test_strategy"
    
    def test_global_prompts_application(self, prompt_system):
        """Test global prompts are applied correctly."""
        query = "Test query"
        context = PromptContext(query=query, domain="test")
        
        exec_context = prompt_system.execute_prompt(query, context)
        
        assert "test_global" in exec_context.applied_global_prompts
        # The rendered prompt should be longer due to global prompt application
        assert len(exec_context.rendered_prompt) > len("Query: Test query\\nContext: \\nAnswer:")
    
    def test_error_handling(self, prompt_system):
        """Test error handling."""
        # Test with nonexistent template override
        query = "Test query"
        context = PromptContext(query=query, domain="test")
        
        exec_context = prompt_system.execute_prompt(
            query, 
            context, 
            template_override="nonexistent_template"
        )
        
        # Should not error, but should use fallback
        assert not exec_context.has_errors()
        assert exec_context.selected_template_id == "test_template"
    
    def test_system_stats(self, prompt_system):
        """Test system statistics."""
        # Initial stats
        stats = prompt_system.get_system_stats()
        assert stats["total_executions"] == 0
        assert stats["total_errors"] == 0
        assert stats["templates_count"] == 1
        assert stats["strategies_count"] == 1
        
        # Execute a prompt
        query = "Test query"
        context = PromptContext(query=query, domain="test")
        prompt_system.execute_prompt(query, context)
        
        # Updated stats
        stats = prompt_system.get_system_stats()
        assert stats["total_executions"] == 1
        assert stats["error_rate"] == 0.0
    
    def test_template_testing(self, prompt_system):
        """Test template testing functionality."""
        test_variables = {
            "query": "Test question",
            "context": "Test context"
        }
        
        exec_context = prompt_system.test_template(
            "test_template", 
            test_variables
        )
        
        assert not exec_context.has_errors()
        assert exec_context.selected_template_id == "test_template"
        assert "Test question" in exec_context.rendered_prompt


class TestPromptSystemIntegration:
    """Integration tests for the prompt system."""
    
    def test_real_config_loading(self):
        """Test loading a real configuration file."""
        # This would test with the actual config file
        config_path = Path(__file__).parent.parent / "config" / "default_prompts.yaml"
        
        if config_path.exists():
            config = PromptConfig.from_file(str(config_path))
            prompt_system = PromptSystem(config)
            
            assert len(prompt_system.list_templates()) > 0
            assert len(prompt_system.strategy_engine.list_strategies()) > 0
            
            # Test basic execution
            query = "What is machine learning?"
            context = PromptContext(query=query, domain="general")
            
            exec_context = prompt_system.execute_prompt(query, context)
            assert not exec_context.has_errors()
            assert exec_context.rendered_prompt is not None
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        config_path = Path(__file__).parent.parent / "config" / "default_prompts.yaml"
        
        if config_path.exists():
            config = PromptConfig.from_file(str(config_path))
            errors = config.validate_config()
            
            # Should have no validation errors
            assert len(errors) == 0
    
    def test_multiple_executions(self, prompt_system):
        """Test multiple executions to ensure system stability."""
        queries = [
            "What is testing?",
            "How does automation work?",
            "Explain the concept of quality assurance."
        ]
        
        for query in queries:
            context = PromptContext(query=query, domain="test")
            exec_context = prompt_system.execute_prompt(query, context)
            
            assert not exec_context.has_errors()
            assert exec_context.rendered_prompt is not None
        
        # Check final stats
        stats = prompt_system.get_system_stats()
        assert stats["total_executions"] == len(queries)
        assert stats["error_rate"] == 0.0


# Helper functions for testing
def create_test_template(template_id: str, content: str = None) -> PromptTemplate:
    """Create a test template with minimal configuration."""
    return PromptTemplate(
        template_id=template_id,
        name=f"Test Template {template_id}",
        type=TemplateType.BASIC,
        template=content or f"Template {template_id}: {{{{ query }}}}",
        input_variables=["query"],
        metadata=TemplateMetadata(use_case="testing")
    )


def create_test_context(domain: str = "test", **kwargs) -> PromptContext:
    """Create a test context with common defaults."""
    return PromptContext(
        query=kwargs.get("query", "Test query"),
        domain=domain,
        user_id=kwargs.get("user_id", "test_user"),
        **{k: v for k, v in kwargs.items() if k not in ["query", "user_id"]}
    )