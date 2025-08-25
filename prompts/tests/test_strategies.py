"""
Tests for the strategy system.
"""

import pytest
import json
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from strategies import StrategyManager, StrategyLoader, StrategyConfig
from strategies.config import (
    TemplateConfig, TemplatesConfig, SelectionRule, 
    SpecializedTemplate, ConditionConfig, PerformanceProfile, 
    Complexity, TransformType, InputTransform, Transform
)


class TestStrategyConfig:
    """Test StrategyConfig and related models."""
    
    def test_template_config_creation(self):
        """Test creating a template configuration."""
        config = TemplateConfig(
            template="qa_basic",
            config={"temperature": 0.7, "max_tokens": 500}
        )
        
        assert config.template == "qa_basic"
        assert config.config["temperature"] == 0.7
        assert config.config["max_tokens"] == 500
    
    def test_template_config_with_transforms(self):
        """Test template config with input/output transforms."""
        config = TemplateConfig(
            template="qa_basic",
            config={},
            input_transforms=[
                InputTransform(
                    input="query",
                    transform=TransformType.LOWERCASE
                )
            ],
            output_transforms=[
                {"transform": TransformType.CLEAN_MARKDOWN}
            ]
        )
        
        assert len(config.input_transforms) == 1
        assert config.input_transforms[0].input == "query"
        assert config.input_transforms[0].transform == TransformType.LOWERCASE
    
    def test_specialized_template(self):
        """Test specialized template with conditions."""
        spec = SpecializedTemplate(
            template="technical_qa",
            condition=ConditionConfig(
                query_type="technical",
                has_context=True,
                context_size={"min": 1, "max": 10}
            ),
            config={"analysis_depth": "comprehensive"},
            priority=20
        )
        
        assert spec.template == "technical_qa"
        assert spec.condition.query_type == "technical"
        assert spec.condition.has_context is True
        assert spec.condition.context_size["min"] == 1
        assert spec.priority == 20
    
    def test_selection_rule(self):
        """Test selection rule creation."""
        rule = SelectionRule(
            name="error_detection",
            condition={
                "expression": "'error' in context.get('query', '').lower()"
            },
            template="debug_template",
            priority=30,
            stop_on_match=True
        )
        
        assert rule.name == "error_detection"
        assert "expression" in rule.condition
        assert rule.template == "debug_template"
        assert rule.priority == 30
        assert rule.stop_on_match is True
    
    def test_strategy_config_minimal(self):
        """Test minimal strategy configuration."""
        strategy = StrategyConfig(
            name="Test Strategy",
            description="A test strategy",
            templates=TemplatesConfig(
                default=TemplateConfig(template="qa_basic")
            )
        )
        
        assert strategy.name == "Test Strategy"
        assert strategy.description == "A test strategy"
        assert strategy.templates.default.template == "qa_basic"
        assert strategy.performance_profile == PerformanceProfile.BALANCED
        assert strategy.complexity == Complexity.MODERATE
    
    def test_strategy_config_complete(self):
        """Test complete strategy configuration."""
        strategy = StrategyConfig(
            name="Advanced Strategy",
            description="An advanced test strategy",
            use_cases=["testing", "qa", "validation"],
            templates=TemplatesConfig(
                default=TemplateConfig(template="qa_basic"),
                fallback=TemplateConfig(template="simple_qa"),
                specialized=[
                    SpecializedTemplate(
                        template="technical_qa",
                        condition=ConditionConfig(query_type="technical"),
                        priority=10
                    )
                ]
            ),
            selection_rules=[
                SelectionRule(
                    name="test_rule",
                    condition={"has_context": True},
                    template="context_qa",
                    priority=20
                )
            ],
            performance_profile=PerformanceProfile.ACCURACY,
            complexity=Complexity.COMPLEX
        )
        
        assert strategy.name == "Advanced Strategy"
        assert len(strategy.use_cases) == 3
        assert strategy.templates.fallback is not None
        assert len(strategy.templates.specialized) == 1
        assert len(strategy.selection_rules) == 1
        assert strategy.performance_profile == PerformanceProfile.ACCURACY
        assert strategy.complexity == Complexity.COMPLEX
    
    def test_strategy_from_dict(self):
        """Test creating strategy from dictionary."""
        data = {
            "name": "Dict Strategy",
            "description": "Created from dict",
            "templates": {
                "default": {"template": "qa_basic", "config": {"temp": 0.5}}
            }
        }
        
        strategy = StrategyConfig.from_dict(data)
        assert strategy.name == "Dict Strategy"
        assert strategy.templates.default.template == "qa_basic"
        assert strategy.templates.default.config["temp"] == 0.5
    
    def test_strategy_template_selection(self):
        """Test template selection based on context."""
        strategy = StrategyConfig(
            name="Selection Test",
            description="Test template selection",
            templates=TemplatesConfig(
                default=TemplateConfig(template="default_qa"),
                specialized=[
                    SpecializedTemplate(
                        template="technical_qa",
                        condition=ConditionConfig(query_type="technical"),
                        priority=10
                    ),
                    SpecializedTemplate(
                        template="simple_qa",
                        condition=ConditionConfig(has_context=False),
                        priority=5
                    )
                ]
            )
        )
        
        # Test default selection - empty context matches has_context=False condition
        template = strategy.get_template_for_context({})
        assert template.template == "simple_qa"  # This matches has_context=False condition
        
        # Test specialized selection
        template = strategy.get_template_for_context({"query_type": "technical"})
        assert template.template == "technical_qa"
        
        # Test no context selection
        template = strategy.get_template_for_context({"context": None})
        assert template.template == "simple_qa"


class TestStrategyLoader:
    """Test StrategyLoader functionality."""
    
    @pytest.fixture
    def temp_strategy_file(self, tmp_path):
        """Create a temporary strategy file."""
        strategy_file = tmp_path / "test_strategies.yaml"
        strategies = {
            "test_strategy": {
                "name": "Test Strategy",
                "description": "A test strategy",
                "templates": {
                    "default": {"template": "qa_basic"}
                }
            },
            "another_strategy": {
                "name": "Another Strategy",
                "description": "Another test strategy",
                "use_cases": ["testing"],
                "templates": {
                    "default": {"template": "qa_detailed"}
                }
            }
        }
        
        with open(strategy_file, 'w') as f:
            yaml.dump(strategies, f)
        
        return strategy_file
    
    def test_load_strategies_from_file(self, temp_strategy_file):
        """Test loading strategies from YAML file."""
        loader = StrategyLoader(strategies_file=str(temp_strategy_file))
        strategies = loader.load_strategies()
        
        assert len(strategies) == 2
        assert "test_strategy" in strategies
        assert "another_strategy" in strategies
        
        test_strategy = strategies["test_strategy"]
        assert test_strategy.name == "Test Strategy"
        assert test_strategy.templates.default.template == "qa_basic"
    
    def test_get_strategy(self, temp_strategy_file):
        """Test getting a specific strategy."""
        loader = StrategyLoader(strategies_file=str(temp_strategy_file))
        
        strategy = loader.get_strategy("test_strategy")
        assert strategy is not None
        assert strategy.name == "Test Strategy"
        
        # Non-existent strategy
        strategy = loader.get_strategy("non_existent")
        assert strategy is None
    
    def test_list_strategies(self, temp_strategy_file):
        """Test listing strategy names."""
        loader = StrategyLoader(strategies_file=str(temp_strategy_file))
        strategy_names = loader.list_strategies()
        
        assert len(strategy_names) == 2
        assert "test_strategy" in strategy_names
        assert "another_strategy" in strategy_names
    
    def test_get_strategies_by_use_case(self, temp_strategy_file):
        """Test filtering strategies by use case."""
        loader = StrategyLoader(strategies_file=str(temp_strategy_file))
        
        testing_strategies = loader.get_strategies_by_use_case("testing")
        assert len(testing_strategies) == 1
        assert testing_strategies[0].name == "Another Strategy"
        
        # Non-existent use case
        no_strategies = loader.get_strategies_by_use_case("non_existent")
        assert len(no_strategies) == 0
    
    def test_validate_strategy(self, temp_strategy_file):
        """Test strategy validation."""
        loader = StrategyLoader(strategies_file=str(temp_strategy_file))
        
        # Valid strategy
        valid_strategy = StrategyConfig(
            name="Valid",
            description="A valid strategy",
            templates=TemplatesConfig(
                default=TemplateConfig(template="qa_basic")
            )
        )
        errors = loader.validate_strategy(valid_strategy)
        assert len(errors) == 0
        
        # Invalid strategy (missing name)
        invalid_strategy = StrategyConfig(
            name="",
            description="Invalid",
            templates=TemplatesConfig(
                default=TemplateConfig(template="qa_basic")
            )
        )
        errors = loader.validate_strategy(invalid_strategy)
        assert len(errors) > 0
        assert any("name" in error for error in errors)


class TestStrategyManager:
    """Test StrategyManager functionality."""
    
    @pytest.fixture
    def mock_template_registry(self):
        """Create mock template registry."""
        registry = Mock()
        registry.get_template = Mock(return_value=Mock(config={"default": "config"}))
        return registry
    
    @pytest.fixture
    def mock_template_engine(self):
        """Create mock template engine."""
        engine = Mock()
        engine.render_template = Mock(return_value="Rendered prompt")
        return engine
    
    @pytest.fixture
    def manager(self, mock_template_registry, mock_template_engine):
        """Create StrategyManager with mocks."""
        manager = StrategyManager(
            template_registry=mock_template_registry,
            template_engine=mock_template_engine
        )
        
        # Add test strategy
        test_strategy = StrategyConfig(
            name="Test Strategy",
            description="Test",
            templates=TemplatesConfig(
                default=TemplateConfig(
                    template="qa_basic",
                    config={"temperature": 0.7}
                )
            )
        )
        manager._strategies_cache["test_strategy"] = test_strategy
        
        return manager
    
    def test_execute_strategy_basic(self, manager):
        """Test basic strategy execution."""
        result = manager.execute_strategy(
            strategy_name="test_strategy",
            inputs={"query": "Test question", "context": []}
        )
        
        # The actual result might be a mock object due to the template engine implementation
        # Just verify the execution succeeded
        assert result == "Rendered prompt"
        # Verify template_engine.render_template was called
        manager.template_engine.render_template.assert_called_once()
    
    def test_execute_strategy_with_overrides(self, manager):
        """Test strategy execution with config overrides."""
        result = manager.execute_strategy(
            strategy_name="test_strategy",
            inputs={"query": "Test question"},
            override_config={"temperature": 0.9, "max_tokens": 1000}
        )
        
        # The actual result might be a mock object due to the template engine implementation
        assert result == "Rendered prompt"
        
        # Check that template_engine.render_template was called
        manager.template_engine.render_template.assert_called_once()
    
    def test_execute_strategy_not_found(self, manager):
        """Test executing non-existent strategy."""
        with pytest.raises(ValueError, match="Strategy 'non_existent' not found"):
            manager.execute_strategy(
                strategy_name="non_existent",
                inputs={"query": "Test"}
            )
    
    def test_input_transforms(self, manager):
        """Test input transformations."""
        # Add strategy with transforms
        transform_strategy = StrategyConfig(
            name="Transform Strategy",
            description="Test transforms",
            templates=TemplatesConfig(
                default=TemplateConfig(
                    template="qa_basic",
                    input_transforms=[
                        InputTransform(
                            input="query",
                            transform=TransformType.LOWERCASE
                        ),
                        InputTransform(
                            input="title",
                            transform=TransformType.UPPERCASE
                        )
                    ]
                )
            )
        )
        manager._strategies_cache["transform_strategy"] = transform_strategy
        
        # Execute with transforms
        result = manager.execute_strategy(
            strategy_name="transform_strategy",
            inputs={
                "query": "TEST QUESTION",
                "title": "test title",
                "other": "unchanged"
            }
        )
        
        # Check that execution succeeded
        assert result == "Rendered prompt"
        
        # Check transformed inputs were passed to render_template
        call_args = manager.template_engine.render_template.call_args
        if call_args:
            # The second argument should be the render vars dict
            render_vars = call_args[0][1] if len(call_args[0]) > 1 else {}
            assert render_vars.get("query") == "test question"  # lowercased
            assert render_vars.get("title") == "TEST TITLE"     # uppercased
            assert render_vars.get("other") == "unchanged"      # unchanged
    
    def test_output_transforms(self, manager):
        """Test output transformations."""
        # Add strategy with output transforms
        transform_strategy = StrategyConfig(
            name="Output Transform Strategy",
            description="Test output transforms",
            templates=TemplatesConfig(
                default=TemplateConfig(
                    template="qa_basic",
                    output_transforms=[
                        Transform(
                            transform=TransformType.UPPERCASE
                        )
                    ]
                )
            )
        )
        manager._strategies_cache["output_transform_strategy"] = transform_strategy
        
        # Mock template engine to return a lowercase string
        manager.template_engine.render_template.return_value = "this is a test output"
        
        # Execute with output transforms
        result = manager.execute_strategy(
            strategy_name="output_transform_strategy",
            inputs={"query": "test"}
        )
        
        # Check that output was transformed to uppercase
        assert result == "THIS IS A TEST OUTPUT"
    
    def test_recommend_strategies(self, manager):
        """Test strategy recommendations with mocked strategies (no file dependency)."""
        # Create mock strategies
        mock_strategies = {
            "simple_qa": StrategyConfig(
                name="Simple Question Answering",
                description="Fast QA strategy",
                use_cases=["qa"],
                performance_profile=PerformanceProfile.SPEED,
                complexity=Complexity.SIMPLE,
                templates=TemplatesConfig(
                    default=TemplateConfig(template="qa_basic")
                )
            ),
            "complex_reasoning": StrategyConfig(
                name="Complex Reasoning",
                description="Accurate reasoning strategy",
                use_cases=["reasoning"],
                performance_profile=PerformanceProfile.ACCURACY,
                complexity=Complexity.COMPLEX,
                templates=TemplatesConfig(
                    default=TemplateConfig(template="chain_of_thought")
                )
            ),
            "general_qa": StrategyConfig(
                name="General QA",
                description="Another QA strategy",
                use_cases=["qa", "general"],
                performance_profile=PerformanceProfile.SPEED,
                complexity=Complexity.SIMPLE,
                templates=TemplatesConfig(
                    default=TemplateConfig(template="qa_basic")
                )
            )
        }
        
        # Mock the load_strategies method to return our mock strategies
        with patch.object(manager, 'load_strategies', return_value=mock_strategies):
            recommendations = manager.recommend_strategies(
                use_case="qa",
                performance="speed",
                complexity="simple"
            )
            
            assert len(recommendations) >= 1
            # Should recommend strategies with matching criteria
            assert any("qa" in r.name.lower() or "question" in r.name.lower() for r in recommendations)
            # Should prioritize strategies that match all criteria
            for rec in recommendations[:2]:  # Check top recommendations
                assert "qa" in rec.use_cases
                assert rec.performance_profile == PerformanceProfile.SPEED
                assert rec.complexity == Complexity.SIMPLE
    
    def test_create_strategy(self, manager):
        """Test creating a new strategy."""
        strategy = manager.create_strategy(
            name="New Strategy",
            description="A new test strategy",
            default_template="new_template",
            use_cases=["testing"],
            performance_profile="accuracy"
        )
        
        assert strategy.name == "New Strategy"
        assert strategy.description == "A new test strategy"
        assert strategy.templates.default.template == "new_template"
        assert strategy.use_cases == ["testing"]
        assert strategy.performance_profile == PerformanceProfile.ACCURACY
    
    def test_execution_stats(self, manager):
        """Test execution statistics tracking."""
        # Execute strategy multiple times
        for _ in range(3):
            manager.execute_strategy(
                strategy_name="test_strategy",
                inputs={"query": "Test"}
            )
        
        stats = manager.get_execution_stats()
        assert "test_strategy" in stats
        assert "qa_basic" in stats["test_strategy"]
        assert stats["test_strategy"]["qa_basic"] == 3
    
    def test_template_usage(self, manager):
        """Test getting template usage across strategies."""
        # get_template_usage loads from files, not cache
        usage = manager.get_template_usage()
        
        # The test should check that qa_basic is used by some strategies
        assert "qa_basic" in usage
        # At least simple_qa and custom_template use qa_basic from default_strategies.yaml
        assert len(usage["qa_basic"]) >= 2
        assert "simple_qa" in usage["qa_basic"]
        assert "custom_template" in usage["qa_basic"]


class TestStrategyIntegration:
    """Integration tests for the strategy system."""
    
    def test_full_workflow(self, tmp_path):
        """Test complete workflow from loading to execution."""
        # Create test strategy file
        strategy_file = tmp_path / "integration_strategies.yaml"
        strategies = {
            "integration_test": {
                "name": "Integration Test",
                "description": "Full workflow test",
                "use_cases": ["testing"],
                "templates": {
                    "default": {
                        "template": "qa_basic",
                        "config": {"temperature": 0.5}
                    },
                    "specialized": [
                        {
                            "template": "technical_qa",
                            "condition": {"query_type": "technical"},
                            "config": {"depth": "comprehensive"},
                            "priority": 10
                        }
                    ]
                },
                "selection_rules": [
                    {
                        "name": "error_rule",
                        "condition": {
                            "expression": "'error' in context.get('query', '').lower()"
                        },
                        "template": "debug_qa",
                        "priority": 20
                    }
                ],
                "global_config": {
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
            }
        }
        
        with open(strategy_file, 'w') as f:
            yaml.dump(strategies, f)
        
        # Create manager with real loader
        manager = StrategyManager(strategies_file=str(strategy_file))
        
        # Mock template components
        manager.template_registry = Mock()
        manager.template_registry.get_template = Mock(
            return_value=Mock(config={"base": "config"})
        )
        manager.template_engine = Mock()
        manager.template_engine.render_template = Mock(
            return_value="Integrated prompt result"
        )
        
        # Test loading
        strategies = manager.load_strategies()
        assert "integration_test" in strategies
        
        # Test execution with default template
        result = manager.execute_strategy(
            strategy_name="integration_test",
            inputs={"query": "Simple question", "context": []}
        )
        assert result == "Integrated prompt result"
        
        # Test execution with specialized template
        result = manager.execute_strategy(
            strategy_name="integration_test",
            inputs={"query": "Technical question"},
            context={"query_type": "technical"}
        )
        assert result == "Integrated prompt result"
        
        # Verify template engine was called
        manager.template_engine.render_template.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])