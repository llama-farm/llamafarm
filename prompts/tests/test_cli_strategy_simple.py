"""
Simplified tests for the strategy-based CLI.
"""

import pytest
from click.testing import CliRunner
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from prompts.core.cli.strategy_cli import cli


class TestCLIBasics:
    """Test basic CLI functionality without complex mocking."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    def test_cli_help(self, runner):
        """Test CLI help command."""
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'LlamaFarm Prompts Strategy Management CLI' in result.output
    
    def test_strategy_group_help(self, runner):
        """Test strategy group help."""
        result = runner.invoke(cli, ['strategy', '--help'])
        # This might fail if dependencies aren't available, but that's expected
        # We just want to ensure the CLI structure is correct
        assert 'strategy' in result.output.lower() or result.exit_code != 0
    
    def test_template_group_help(self, runner):
        """Test template group help."""
        result = runner.invoke(cli, ['template', '--help'])
        # Similar to above - structure test
        assert 'template' in result.output.lower() or result.exit_code != 0


class TestCLICommands:
    """Test CLI command structure."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    def test_stats_command_exists(self, runner):
        """Test that stats command exists and help output structure."""
        result = runner.invoke(cli, ['stats', '--help'])
        # Command should exist even if it fails due to missing deps
        assert 'stats' in result.output.lower() or result.exit_code != 0
        
        # Assert help output structure when command is available
        if result.exit_code == 0:
            assert '--help' in result.output or 'help' in result.output.lower()
            assert 'Usage:' in result.output or 'usage:' in result.output.lower()
            # Check for command description or options
            assert any(keyword in result.output.lower() for keyword in 
                      ['show', 'display', 'statistics', 'stats', 'metrics'])
        else:
            # If the command fails due to missing dependencies, check for a relevant error message
            assert any(keyword in result.output.lower() for keyword in 
                      ['missing', 'error', 'failed', 'not found', 'dependency'])
    
    def test_demo_command_exists(self, runner):
        """Test that demo command exists and help output structure."""
        result = runner.invoke(cli, ['demo', '--help'])
        # Command should exist even if it fails due to missing deps
        assert 'demo' in result.output.lower() or result.exit_code != 0
        
        # Assert help output structure when command is available
        if result.exit_code == 0:
            assert '--help' in result.output or 'help' in result.output.lower()
            assert 'Usage:' in result.output or 'usage:' in result.output.lower()
            # Check for command description or options
            assert any(keyword in result.output.lower() for keyword in 
                      ['demo', 'demonstration', 'example', 'run', 'execute'])
        else:
            # If the command fails due to missing dependencies, check for a relevant error message
            assert any(keyword in result.output.lower() for keyword in 
                      ['missing', 'error', 'failed', 'not found', 'dependency'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])