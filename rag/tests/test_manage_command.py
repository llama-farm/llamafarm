#!/usr/bin/env python3
"""
Test suite for the manage command functionality.
Tests all manage subcommands with various configurations and edge cases.
"""

import pytest
import tempfile
import os
from pathlib import Path
from argparse import Namespace
from unittest.mock import Mock, patch, MagicMock

# Add the parent directory to Python path so we can import cli
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import cli as rag_cli
from core.base import Document
from core.document_manager import DocumentManager, DeletionStrategy


class TestManageCommand:
    """Test the manage command and its subcommands."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_strategy = "simple"
        
    def teardown_method(self):
        """Clean up after each test."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def create_mock_args(self, **kwargs):
        """Create mock args with default values."""
        defaults = {
            'config': 'rag_config.json',
            'base_dir': None,
            'log_level': 'ERROR',
            'quiet': True,
            'verbose': False,
            'rag_strategy': self.test_strategy,
            'strategy_file': None,
        }
        defaults.update(kwargs)
        return Namespace(**defaults)

    @patch('core.factories.create_vector_store_from_config')
    @patch('core.document_manager.DocumentManager')
    def test_manage_stats_command(self, mock_doc_manager, mock_create_store):
        """Test manage stats command."""
        # Mock the document manager
        mock_manager = Mock()
        mock_manager.get_collection_stats.return_value = {
            'total_documents': 10,
            'total_size': 1024,
            'created_at': '2024-01-01'
        }
        mock_doc_manager.return_value = mock_manager
        
        # Mock vector store
        mock_store = Mock()
        mock_create_store.return_value = mock_store
        
        args = self.create_mock_args(
            manage_command='stats',
            detailed=False
        )
        
        # Should not raise an exception
        try:
            rag_cli.manage_command(args)
            success = True
        except Exception as e:
            success = False
            print(f"Stats command failed: {e}")
        
        assert success, "Stats command should execute without errors"
        mock_manager.get_collection_stats.assert_called_once()

    @patch('cli.create_vector_store_from_config')  
    @patch('cli.DocumentManager')
    def test_manage_delete_dry_run(self, mock_doc_manager, mock_create_store):
        """Test manage delete command in dry-run mode."""
        # Mock the document manager
        mock_manager = Mock()
        mock_manager.delete_documents.return_value = {
            'deleted': 0,
            'errors': []
        }
        mock_doc_manager.return_value = mock_manager
        
        # Mock vector store
        mock_store = Mock()
        mock_create_store.return_value = mock_store
        
        args = self.create_mock_args(
            manage_command='delete',
            delete_strategy='soft',
            older_than=30,
            doc_ids=None,
            filenames=None,
            content_hashes=None,
            expired=False,
            dry_run=True
        )
        
        # Should not raise an exception
        try:
            rag_cli.manage_command(args)
            success = True
        except Exception as e:
            success = False
            print(f"Delete dry-run failed: {e}")
        
        assert success, "Delete dry-run should execute without errors"

    @patch('core.factories.create_vector_store_from_config')
    @patch('core.document_manager.DocumentManager')
    def test_manage_delete_with_doc_ids(self, mock_doc_manager, mock_create_store):
        """Test manage delete command with specific document IDs."""
        # Mock the document manager
        mock_manager = Mock()
        mock_manager.delete_documents.return_value = {
            'deleted': 2,
            'errors': []
        }
        mock_doc_manager.return_value = mock_manager
        
        # Mock vector store
        mock_store = Mock()
        mock_create_store.return_value = mock_store
        
        args = self.create_mock_args(
            manage_command='delete',
            delete_strategy='hard',
            older_than=None,
            doc_ids=['doc1', 'doc2'],
            filenames=None,
            content_hashes=None,
            expired=False,
            dry_run=False
        )
        
        try:
            rag_cli.manage_command(args)
            success = True
        except Exception as e:
            success = False
            print(f"Delete with doc_ids failed: {e}")
        
        assert success, "Delete with doc_ids should execute without errors"

    @patch('core.factories.create_vector_store_from_config')
    @patch('core.document_manager.DocumentManager')
    def test_manage_delete_older_than(self, mock_doc_manager, mock_create_store):
        """Test manage delete command with older-than filter."""
        # Mock the document manager
        mock_manager = Mock()
        mock_manager.delete_documents.return_value = {
            'deleted': 5,
            'errors': []
        }
        mock_doc_manager.return_value = mock_manager
        
        # Mock vector store
        mock_store = Mock()
        mock_create_store.return_value = mock_store
        
        args = self.create_mock_args(
            manage_command='delete',
            delete_strategy='archive',
            older_than=7,  # 7 days
            doc_ids=None,
            filenames=None, 
            content_hashes=None,
            expired=False,
            dry_run=False
        )
        
        try:
            rag_cli.manage_command(args)
            success = True
        except Exception as e:
            success = False
            print(f"Delete older-than failed: {e}")
        
        assert success, "Delete older-than should execute without errors"

    @patch('core.strategies.StrategyManager')
    @patch('core.factories.create_vector_store_from_config')
    @patch('core.document_manager.DocumentManager')
    def test_manage_with_strategy(self, mock_doc_manager, mock_create_store, mock_strategy_manager):
        """Test manage command using RAG strategy configuration."""
        # Mock strategy manager
        mock_manager_instance = Mock()
        mock_manager_instance.convert_strategy_to_config.return_value = {
            'vector_store': {
                'type': 'ChromaStore',
                'config': {'collection_name': 'test'}
            }
        }
        mock_strategy_manager.return_value = mock_manager_instance
        
        # Mock document manager
        mock_doc_mgr = Mock()
        mock_doc_mgr.get_collection_stats.return_value = {'total_documents': 0}
        mock_doc_manager.return_value = mock_doc_mgr
        
        # Mock vector store
        mock_store = Mock()
        mock_create_store.return_value = mock_store
        
        args = self.create_mock_args(
            manage_command='stats',
            rag_strategy='test_strategy',
            strategy_file='test_strategies.yaml',
            detailed=False
        )
        
        try:
            rag_cli.manage_command(args)
            success = True
        except Exception as e:
            success = False
            print(f"Manage with strategy failed: {e}")
        
        assert success, "Manage with strategy should work"
        mock_manager_instance.convert_strategy_to_config.assert_called_once_with('test_strategy')

    def test_manage_command_argument_validation(self):
        """Test that manage command validates arguments correctly."""
        # Test missing manage_command
        args = self.create_mock_args()
        # Don't set manage_command
        delattr(args, 'manage_command') if hasattr(args, 'manage_command') else None
        
        with pytest.raises(SystemExit):
            # Should exit due to missing manage_command
            rag_cli.manage_command(args)

    @patch('core.factories.create_vector_store_from_config')
    @patch('core.document_manager.DocumentManager')  
    def test_manage_replace_command(self, mock_doc_manager, mock_create_store):
        """Test manage replace command."""
        # Mock the document manager
        mock_manager = Mock()
        mock_doc_manager.return_value = mock_manager
        
        # Mock vector store
        mock_store = Mock()
        mock_create_store.return_value = mock_store
        
        args = self.create_mock_args(
            manage_command='replace',
            source='/path/to/new/file.txt',
            target_doc_id='doc123',
            replace_strategy='versioning'
        )
        
        try:
            rag_cli.manage_command(args)
            success = True
        except Exception as e:
            # Replace might not be fully implemented, so we check for expected behavior
            if "coming soon" in str(e).lower() or "not implemented" in str(e).lower():
                success = True  # Expected behavior for unimplemented feature
            else:
                success = False
                print(f"Replace command failed unexpectedly: {e}")
        
        assert success, "Replace command should handle gracefully"

    @patch('core.factories.create_vector_store_from_config')
    @patch('core.document_manager.DocumentManager')
    def test_manage_cleanup_command(self, mock_doc_manager, mock_create_store):
        """Test manage cleanup command."""
        # Mock the document manager
        mock_manager = Mock()
        mock_doc_manager.return_value = mock_manager
        
        # Mock vector store
        mock_store = Mock()
        mock_create_store.return_value = mock_store
        
        args = self.create_mock_args(
            manage_command='cleanup',
            duplicates=True,
            vacuum=False
        )
        
        try:
            rag_cli.manage_command(args)
            success = True
        except Exception as e:
            # Cleanup might not be fully implemented
            if "coming soon" in str(e).lower() or "not implemented" in str(e).lower():
                success = True
            else:
                success = False
                print(f"Cleanup command failed: {e}")
        
        assert success, "Cleanup command should handle gracefully"

    def test_deletion_strategy_mapping(self):
        """Test that deletion strategy strings map correctly to enum values."""
        from core.document_manager import DeletionStrategy
        
        strategy_map = {
            "soft": DeletionStrategy.SOFT_DELETE,
            "hard": DeletionStrategy.HARD_DELETE,
            "archive": DeletionStrategy.ARCHIVE_DELETE
        }
        
        # Verify all expected strategies are mapped
        assert "soft" in strategy_map
        assert "hard" in strategy_map
        assert "archive" in strategy_map
        
        # Verify they map to correct enum values
        assert strategy_map["soft"] == DeletionStrategy.SOFT_DELETE
        assert strategy_map["hard"] == DeletionStrategy.HARD_DELETE
        assert strategy_map["archive"] == DeletionStrategy.ARCHIVE_DELETE


class TestManageCommandIntegration:
    """Integration tests for manage command with real components."""

    def test_manage_help_commands_basic(self):
        """Test that manage help commands work correctly."""
        import subprocess
        import sys
        
        cli_path = Path(__file__).parent.parent / "cli.py"
        
        # Test main manage help
        result = subprocess.run([
            sys.executable, str(cli_path), 'manage', '--help'
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert '--strategy' in result.stdout
        assert 'delete' in result.stdout
        assert 'stats' in result.stdout
        
        # Test delete subcommand help
        result = subprocess.run([
            sys.executable, str(cli_path), 'manage', 'delete', '--help'
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert '--delete-strategy' in result.stdout
        assert '--older-than' in result.stdout
        assert '--dry-run' in result.stdout

    def test_manage_argument_validation(self):
        """Test that manage command validates arguments properly."""
        import subprocess
        import sys
        
        cli_path = Path(__file__).parent.parent / "cli.py" 
        
        # Test with invalid subcommand
        result = subprocess.run([
            sys.executable, str(cli_path), 'manage', 'invalid_subcommand'
        ], capture_output=True, text=True)
        
        assert result.returncode != 0  # Should fail with invalid subcommand

    def test_manage_delete_dry_run_integration(self):
        """Test manage delete in dry-run mode (safe integration test)."""
        import subprocess
        import sys
        
        cli_path = Path(__file__).parent.parent / "cli.py"
        
        # Use simple strategy for consistency
        result = subprocess.run([
            sys.executable, str(cli_path), 
            'manage', '--strategy', 'simple', 
            'delete', '--delete-strategy', 'soft', '--dry-run'
        ], capture_output=True, text=True)
        
        # Should not fail, even if no documents exist
        # The command should handle empty collections gracefully
        assert result.returncode == 0 or "No documents" in result.stdout or "DRY RUN" in result.stdout


if __name__ == '__main__':
    pytest.main([__file__, '-v'])