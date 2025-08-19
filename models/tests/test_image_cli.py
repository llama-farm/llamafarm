"""Tests for image recognition CLI commands."""

import unittest
import subprocess
import json
import tempfile
import shutil
from pathlib import Path
import sys
import argparse
from unittest.mock import Mock, patch, MagicMock
import io

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.cli.image_commands import (
    setup_command, info_command, download_sample_command,
    detect_command, classify_command, batch_detect_command,
    benchmark_command, list_models_command
)


class TestCLICommands(unittest.TestCase):
    """Test CLI commands directly."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.sample_image = self.test_dir / "test.jpg"
        
        # Create a dummy image file
        self.sample_image.write_text("dummy image")
    
    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_setup_command(self):
        """Test setup command."""
        args = argparse.Namespace(
            download_models=False,
            test=False  # Don't actually test to avoid dependencies
        )
        
        # Should complete without error
        result = setup_command(args)
        self.assertIn(result, [0, 1])  # Either success or missing deps
    
    def test_info_command(self):
        """Test info command."""
        args = argparse.Namespace(
            device=True,
            models=False,
            strategies=False
        )
        
        # Capture output
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        result = info_command(args)
        
        output = buffer.getvalue()
        sys.stdout = old_stdout
        
        self.assertEqual(result, 0)
        self.assertIn("Hardware Information", output)
    
    def test_info_command_models(self):
        """Test info command with models."""
        args = argparse.Namespace(
            device=False,
            models=True,
            strategies=False
        )
        
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        result = info_command(args)
        
        output = buffer.getvalue()
        sys.stdout = old_stdout
        
        self.assertEqual(result, 0)
        self.assertIn("Available Models", output)
        self.assertIn("yolov8n", output)
    
    def test_info_command_strategies(self):
        """Test info command with strategies."""
        args = argparse.Namespace(
            device=False,
            models=False,
            strategies=True
        )
        
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        result = info_command(args)
        
        output = buffer.getvalue()
        sys.stdout = old_stdout
        
        self.assertEqual(result, 0)
        self.assertIn("Available Strategies", output)
        self.assertIn("default_yolo", output)
    
    @patch('requests.get')
    def test_download_sample_command(self, mock_get):
        """Test download sample command."""
        # Mock successful download
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"fake image data"
        mock_get.return_value = mock_response
        
        args = argparse.Namespace(
            output_dir=str(self.test_dir / "samples"),
            type='street'
        )
        
        with patch('PIL.Image.open') as mock_open:
            mock_img = Mock()
            mock_open.return_value = mock_img
            
            result = download_sample_command(args)
        
        self.assertEqual(result, 0)
        self.assertTrue((self.test_dir / "samples").exists())
    
    def test_list_models_command(self):
        """Test list models command."""
        args = argparse.Namespace(
            type='detection',
            show_size=True
        )
        
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        result = list_models_command(args)
        
        output = buffer.getvalue()
        sys.stdout = old_stdout
        
        self.assertEqual(result, 0)
        self.assertIn("Detection Models", output)
        self.assertIn("yolov8n", output)
        self.assertIn("MB", output)  # Size should be shown
    
    def test_list_models_all(self):
        """Test list all models."""
        args = argparse.Namespace(
            type='all',
            show_size=False
        )
        
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        result = list_models_command(args)
        
        output = buffer.getvalue()
        sys.stdout = old_stdout
        
        self.assertEqual(result, 0)
        self.assertIn("Detection Models", output)
        self.assertIn("Classification Models", output)
        self.assertIn("Segmentation Models", output)
    
    @patch('core.cli.image_commands.ImageRecognizerFactory')
    def test_detect_command_summary(self, mock_factory):
        """Test detect command with summary output."""
        # Mock recognizer
        mock_recognizer = Mock()
        mock_factory.create.return_value = mock_recognizer
        
        # Mock detection results
        mock_detection = Mock()
        mock_detection.label = "person"
        mock_detection.confidence = 0.95
        mock_detection.bbox = [10, 20, 100, 200]
        mock_detection.dict.return_value = {
            "label": "person",
            "confidence": 0.95,
            "bbox": [10, 20, 100, 200]
        }
        
        mock_recognizer.detect.return_value = [mock_detection, mock_detection]
        
        args = argparse.Namespace(
            image_path=str(self.sample_image),
            strategy='default_yolo',
            confidence=0.5,
            output_format='summary',
            visualize=False,
            output_path=None,
            measure_time=False
        )
        
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        result = detect_command(args)
        
        output = buffer.getvalue()
        sys.stdout = old_stdout
        
        self.assertEqual(result, 0)
        self.assertIn("Detected 2 objects", output)
        self.assertIn("2 person(s)", output)
    
    @patch('core.cli.image_commands.ImageRecognizerFactory')
    def test_detect_command_json(self, mock_factory):
        """Test detect command with JSON output."""
        # Mock recognizer
        mock_recognizer = Mock()
        mock_factory.create.return_value = mock_recognizer
        
        # Mock detection
        mock_detection = Mock()
        mock_detection.dict.return_value = {
            "label": "car",
            "confidence": 0.88,
            "bbox": [50, 50, 200, 150],
            "metadata": {}
        }
        
        mock_recognizer.detect.return_value = [mock_detection]
        
        args = argparse.Namespace(
            image_path=str(self.sample_image),
            strategy='default_yolo',
            confidence=0.5,
            output_format='json',
            visualize=False,
            output_path=None,
            measure_time=False
        )
        
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        result = detect_command(args)
        
        output = buffer.getvalue()
        sys.stdout = old_stdout
        
        self.assertEqual(result, 0)
        
        # Parse JSON output
        data = json.loads(output)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["label"], "car")
        self.assertEqual(data[0]["confidence"], 0.88)
    
    @patch('core.cli.image_commands.ImageRecognizerFactory')
    def test_detect_command_with_timing(self, mock_factory):
        """Test detect command with timing."""
        mock_recognizer = Mock()
        mock_factory.create.return_value = mock_recognizer
        mock_recognizer.detect.return_value = []
        
        args = argparse.Namespace(
            image_path=str(self.sample_image),
            strategy='default_yolo',
            confidence=0.5,
            output_format='summary',
            visualize=False,
            output_path=None,
            measure_time=True
        )
        
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        result = detect_command(args)
        
        output = buffer.getvalue()
        sys.stdout = old_stdout
        
        self.assertEqual(result, 0)
        self.assertIn("Detection time:", output)
        self.assertIn("FPS", output)
    
    @patch('core.cli.image_commands.ImageRecognizerFactory')
    @patch('requests.get')
    def test_detect_command_url(self, mock_get, mock_factory):
        """Test detect command with URL input."""
        # Mock URL download
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"fake image"
        mock_get.return_value = mock_response
        
        # Mock recognizer
        mock_recognizer = Mock()
        mock_factory.create.return_value = mock_recognizer
        mock_recognizer.detect.return_value = []
        
        with patch('PIL.Image.open') as mock_open:
            mock_img = Mock()
            mock_open.return_value = mock_img
            
            args = argparse.Namespace(
                image_path="https://example.com/image.jpg",
                strategy='default_yolo',
                confidence=0.5,
                output_format='summary',
                visualize=False,
                output_path=None,
                measure_time=False
            )
            
            result = detect_command(args)
        
        self.assertEqual(result, 0)
        mock_get.assert_called_once_with("https://example.com/image.jpg")
    
    @patch('core.cli.image_commands.ImageRecognizerFactory')
    def test_batch_detect_command(self, mock_factory):
        """Test batch detect command."""
        # Create test images
        (self.test_dir / "img1.jpg").write_text("img1")
        (self.test_dir / "img2.jpg").write_text("img2")
        
        # Mock recognizer
        mock_recognizer = Mock()
        mock_factory.create.return_value = mock_recognizer
        
        mock_detection = Mock()
        mock_detection.label = "object"
        mock_detection.dict.return_value = {"label": "object"}
        mock_recognizer.detect.return_value = [mock_detection]
        
        args = argparse.Namespace(
            input_dir=str(self.test_dir),
            output_dir=str(self.test_dir / "output"),
            strategy='default_yolo',
            parallel=False,
            batch_size=32,
            visualize=False,
            summary=True
        )
        
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        result = batch_detect_command(args)
        
        output = buffer.getvalue()
        sys.stdout = old_stdout
        
        self.assertEqual(result, 0)
        self.assertIn("Processing 2 images", output)
        self.assertIn("Summary", output)
        self.assertTrue((self.test_dir / "output").exists())
    
    @patch('core.cli.image_commands.ImageRecognizerFactory')
    @patch('requests.get')
    def test_benchmark_command(self, mock_get, mock_factory):
        """Test benchmark command."""
        # Mock image download
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"fake image"
        mock_get.return_value = mock_response
        
        # Mock recognizer
        mock_recognizer = Mock()
        mock_recognizer.device = "cpu"
        mock_factory.create.return_value = mock_recognizer
        mock_recognizer.detect.return_value = []
        
        with patch('PIL.Image.open') as mock_open:
            mock_img = Mock()
            mock_open.return_value = mock_img
            
            args = argparse.Namespace(
                image=None,
                runs=2,
                models=['yolov8n'],
                compare_devices=False
            )
            
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            
            result = benchmark_command(args)
            
            output = buffer.getvalue()
            sys.stdout = old_stdout
        
        self.assertEqual(result, 0)
        self.assertIn("Performance Benchmark", output)
        self.assertIn("yolov8n", output)
        self.assertIn("FPS", output)


class TestCLIIntegration(unittest.TestCase):
    """Test CLI integration with subprocess."""
    
    def test_cli_help(self):
        """Test CLI help command."""
        result = subprocess.run(
            ["uv", "run", "python", "-c", 
             "from core.cli.image_commands_enhanced import main; import sys; sys.argv = ['prog', '--help']; main()"],
            capture_output=True,
            text=True
        )
        
        # Should show help without error
        self.assertIn("LlamaFarm Image Recognition CLI", result.stdout + result.stderr)
    
    def test_cli_info_device(self):
        """Test CLI info command via subprocess."""
        result = subprocess.run(
            ["uv", "run", "python", "-c",
             """
from core.cli.image_commands_enhanced import info_command
import argparse
args = argparse.Namespace(device=True, models=False, strategies=False)
info_command(args)
             """],
            capture_output=True,
            text=True
        )
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("Hardware Information", result.stdout)


class TestCLIEndToEnd(unittest.TestCase):
    """End-to-end CLI tests."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_full_workflow(self):
        """Test a complete workflow using CLI commands."""
        import importlib
        
        # Step 1: Check info
        from core.cli.image_commands_enhanced import info_command
        args = argparse.Namespace(device=True, models=False, strategies=False)
        result = info_command(args)
        self.assertEqual(result, 0)
        
        # Step 2: List models
        from core.cli.image_commands_enhanced import list_models_command
        args = argparse.Namespace(type='detection', show_size=False)
        result = list_models_command(args)
        self.assertEqual(result, 0)
        
        # Step 3: Download samples (mocked)
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = b"fake image"
            mock_get.return_value = mock_response
            
            with patch('PIL.Image.open') as mock_open:
                mock_img = Mock()
                mock_open.return_value = mock_img
                
                from core.cli.image_commands_enhanced import download_sample_command
                args = argparse.Namespace(
                    output_dir=str(self.test_dir / "samples"),
                    type='street'
                )
                result = download_sample_command(args)
                self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()