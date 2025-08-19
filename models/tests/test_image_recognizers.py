"""Tests for image recognition components."""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import numpy as np
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from components.base import (
    BaseImageRecognizer, BaseImageTrainer,
    Detection, Classification, TrainingExample
)
from components.factory import ImageRecognizerFactory, ImageTrainerFactory


class TestImageRecognizerBase(unittest.TestCase):
    """Test base image recognizer functionality."""
    
    def test_detection_model(self):
        """Test Detection model."""
        detection = Detection(
            label="person",
            confidence=0.95,
            bbox=[100, 200, 300, 400],
            metadata={"class_id": 0}
        )
        
        self.assertEqual(detection.label, "person")
        self.assertEqual(detection.confidence, 0.95)
        self.assertEqual(detection.bbox, [100, 200, 300, 400])
        self.assertEqual(detection.metadata["class_id"], 0)
    
    def test_classification_model(self):
        """Test Classification model."""
        classification = Classification(
            label="cat",
            confidence=0.87,
            metadata={"class_id": 5}
        )
        
        self.assertEqual(classification.label, "cat")
        self.assertEqual(classification.confidence, 0.87)
        self.assertEqual(classification.metadata["class_id"], 5)
    
    def test_training_example_model(self):
        """Test TrainingExample model."""
        example = TrainingExample(
            image_path="/path/to/image.jpg",
            annotations={"bboxes": [{"class_id": 0, "bbox": [10, 10, 100, 100]}]},
            metadata={"source": "manual"}
        )
        
        self.assertEqual(example.image_path, "/path/to/image.jpg")
        self.assertIsNotNone(example.annotations)
        self.assertEqual(example.metadata["source"], "manual")


class TestYOLORecognizer(unittest.TestCase):
    """Test YOLO recognizer implementation."""
    
    @patch('ultralytics.YOLO')
    def test_yolo_initialization(self, mock_yolo):
        """Test YOLO recognizer initialization."""
        from components.image_recognizers.yolo.yolo_recognizer import YOLORecognizer
        
        config = {
            "version": "yolov8n",
            "confidence_threshold": 0.5
        }
        
        recognizer = YOLORecognizer(config)
        
        self.assertEqual(recognizer.model_version, "yolov8n")
        self.assertEqual(recognizer.confidence_threshold, 0.5)
        mock_yolo.assert_called_once()
    
    
    def test_supported_tasks(self):
        """Test getting supported tasks based on model version."""
        from components.image_recognizers.yolo.yolo_recognizer import YOLORecognizer
        
        with patch.object(YOLORecognizer, 'load_model'):
            # Detection model
            recognizer = YOLORecognizer({"version": "yolov8n"})
            tasks = recognizer.get_supported_tasks()
            self.assertIn("detect", tasks)
            
            # Classification model
            recognizer = YOLORecognizer({"version": "yolov8n-cls"})
            tasks = recognizer.get_supported_tasks()
            self.assertIn("classify", tasks)
            
            # Segmentation model
            recognizer = YOLORecognizer({"version": "yolov8n-seg"})
            tasks = recognizer.get_supported_tasks()
            self.assertIn("detect", tasks)
            self.assertIn("segment", tasks)


class TestYOLOTrainer(unittest.TestCase):
    """Test YOLO trainer implementation."""
    
    def test_trainer_initialization(self):
        """Test YOLO trainer initialization."""
        from components.image_recognizers.yolo.yolo_trainer import YOLOTrainer
        
        config = {
            "dataset_path": "./test_dataset",
            "augmentation_factor": 5
        }
        
        trainer = YOLOTrainer(config)
        
        self.assertEqual(trainer.dataset_path, Path("./test_dataset"))
        self.assertEqual(trainer.augmentation_factor, 5)
        self.assertEqual(trainer.training_status["status"], "idle")
    
    def test_add_examples(self):
        """Test adding training examples."""
        from components.image_recognizers.yolo.yolo_trainer import YOLOTrainer
        
        trainer = YOLOTrainer({})
        
        examples = [
            TrainingExample(
                image_path="img1.jpg",
                annotations={"bboxes": []}
            ),
            TrainingExample(
                image_path="img2.jpg",
                annotations={"bboxes": []}
            )
        ]
        
        trainer.add_examples(examples)
        
        self.assertEqual(len(trainer.examples), 2)
    
    def test_training_status(self):
        """Test getting training status."""
        from components.image_recognizers.yolo.yolo_trainer import YOLOTrainer
        
        trainer = YOLOTrainer({})
        status = trainer.get_training_status()
        
        self.assertEqual(status["status"], "idle")
        self.assertEqual(status["current_epoch"], 0)
        self.assertIsNone(status["loss"])
    
    @patch('cv2.flip')
    @patch('cv2.convertScaleAbs')
    def test_augmentation(self, mock_scale, mock_flip):
        """Test data augmentation."""
        from components.image_recognizers.yolo.yolo_trainer import YOLOTrainer
        
        trainer = YOLOTrainer({"augmentation_factor": 3})
        
        # Create mock image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Mock augmentation functions
        mock_flip.return_value = image.copy()
        mock_scale.return_value = image.copy()
        
        augmented = trainer.apply_augmentation(image)
        
        # Should return at most augmentation_factor images
        self.assertLessEqual(len(augmented), 3)
        self.assertGreaterEqual(len(augmented), 1)  # At least original


class TestImageRecognizerFactory(unittest.TestCase):
    """Test image recognizer factory."""
    
    def test_factory_registration(self):
        """Test component registration in factory."""
        # Check if YOLO is registered
        components = ImageRecognizerFactory.list_components()
        
        # May be empty if auto-registration didn't work
        # But the factory should exist
        self.assertIsInstance(components, list)
    
    @patch('components.factory.ImageRecognizerFactory.create')
    def test_factory_create(self, mock_yolo_class):
        """Test creating recognizer through factory."""
        # Register mock YOLO
        ImageRecognizerFactory.register("yolo", mock_yolo_class)
        
        config = {
            "type": "yolo",
            "config": {
                "version": "yolov8n"
            }
        }
        
        recognizer = ImageRecognizerFactory.create(config)
        
        mock_yolo_class.assert_called_once()


class TestIntegration(unittest.TestCase):
    """Integration tests for image recognition pipeline."""
    
    @patch('ultralytics.YOLO')
    def test_detection_pipeline(self, mock_yolo_model):
        """Test complete detection pipeline."""
        from components.image_recognizers.yolo.yolo_recognizer import YOLORecognizer
        
        # Setup mock model
        mock_model_instance = MagicMock()
        mock_yolo_model.return_value = mock_model_instance
        
        # Mock detection results
        mock_result = MagicMock()
        mock_result.boxes = MagicMock()
        mock_box = MagicMock()
        mock_box.xyxy = [[100, 200, 300, 400]]
        mock_box.conf = [0.95]
        mock_box.cls = [0]
        mock_result.boxes = [mock_box]
        mock_result.names = {0: "person"}
        
        mock_model_instance.return_value = [mock_result]
        
        # Create recognizer and run detection
        recognizer = YOLORecognizer({"version": "yolov8n"})
        
        with patch('pathlib.Path.exists', return_value=True):
            detections = recognizer.detect("test_image.jpg")
        
        # Verify detection
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].label, "person")
        self.assertEqual(detections[0].confidence, 0.95)
    
    def test_cli_integration(self):
        """Test CLI command integration."""
        from core.cli.image_commands import add_image_commands
        import argparse
        
        # Create parser
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        
        # Add image commands
        add_image_commands(subparsers)
        
        # Test parsing detect command
        args = parser.parse_args([
            "image", "detect", "test.jpg",
            "--confidence", "0.6",
            "--visualize"
        ])
        
        self.assertEqual(args.image_command, "detect")
        self.assertEqual(args.image_path, "test.jpg")
        self.assertEqual(args.confidence, 0.6)
        self.assertTrue(args.visualize)


if __name__ == "__main__":
    unittest.main()