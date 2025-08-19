"""YOLO few-shot trainer implementation."""

import logging
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import random

import numpy as np
from PIL import Image

from ...base import BaseImageTrainer, TrainingExample

logger = logging.getLogger(__name__)


class YOLOTrainer(BaseImageTrainer):
    """YOLO-based few-shot trainer with data augmentation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize YOLO trainer with configuration."""
        super().__init__(config)
        
        self.dataset_path = Path(config.get("dataset_path", "./yolo_dataset"))
        self.augmentation_factor = config.get("augmentation_factor", 3)
        self.training_status = {
            "status": "idle",
            "current_epoch": 0,
            "total_epochs": 0,
            "loss": None,
            "metrics": {}
        }
    
    def add_examples(self, examples: List[TrainingExample]) -> None:
        """Add training examples."""
        self.examples.extend(examples)
        logger.info(f"Added {len(examples)} training examples. Total: {len(self.examples)}")
    
    def prepare_dataset(self) -> Path:
        """Prepare YOLO format dataset from examples."""
        try:
            # Create dataset structure
            train_dir = self.dataset_path / "train"
            val_dir = self.dataset_path / "val"
            
            for dir_path in [train_dir / "images", train_dir / "labels",
                            val_dir / "images", val_dir / "labels"]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # Split examples (80/20 train/val)
            random.shuffle(self.examples)
            split_idx = int(0.8 * len(self.examples))
            train_examples = self.examples[:split_idx]
            val_examples = self.examples[split_idx:]
            
            # Process training examples
            self._process_examples(train_examples, train_dir, augment=True)
            
            # Process validation examples (no augmentation)
            self._process_examples(val_examples, val_dir, augment=False)
            
            # Create data.yaml for YOLO
            self._create_data_yaml()
            
            logger.info(f"Dataset prepared at {self.dataset_path}")
            return self.dataset_path
            
        except Exception as e:
            logger.error(f"Dataset preparation failed: {e}")
            raise
    
    def _process_examples(self, examples: List[TrainingExample], 
                         output_dir: Path, augment: bool = False) -> None:
        """Process examples and save in YOLO format."""
        for idx, example in enumerate(examples):
            try:
                # Load image
                img = Image.open(example.image_path)
                img_array = np.array(img)
                
                # Base image
                base_name = f"img_{idx:06d}"
                img.save(output_dir / "images" / f"{base_name}.jpg")
                
                # Save annotations if available
                if example.annotations:
                    self._save_yolo_annotations(
                        example.annotations,
                        output_dir / "labels" / f"{base_name}.txt",
                        img.size
                    )
                
                # Apply augmentation if requested
                if augment:
                    aug_images = self.apply_augmentation(img_array)
                    for aug_idx, aug_img in enumerate(aug_images):
                        aug_name = f"img_{idx:06d}_aug{aug_idx}"
                        
                        # Save augmented image
                        Image.fromarray(aug_img).save(
                            output_dir / "images" / f"{aug_name}.jpg"
                        )
                        
                        # Transform and save annotations
                        if example.annotations:
                            self._save_yolo_annotations(
                                example.annotations,
                                output_dir / "labels" / f"{aug_name}.txt",
                                img.size
                            )
                
            except Exception as e:
                logger.warning(f"Failed to process example {idx}: {e}")
    
    def _save_yolo_annotations(self, annotations: Dict[str, Any],
                               output_path: Path, img_size: tuple) -> None:
        """Save annotations in YOLO format."""
        width, height = img_size
        
        with open(output_path, 'w') as f:
            if "bboxes" in annotations:
                for bbox_data in annotations["bboxes"]:
                    class_id = bbox_data.get("class_id", 0)
                    bbox = bbox_data.get("bbox", [])
                    
                    if len(bbox) == 4:
                        # Convert to YOLO format (normalized xywh)
                        x1, y1, x2, y2 = bbox
                        cx = (x1 + x2) / 2 / width
                        cy = (y1 + y2) / 2 / height
                        w = (x2 - x1) / width
                        h = (y2 - y1) / height
                        
                        f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    
    def _create_data_yaml(self) -> None:
        """Create data.yaml configuration for YOLO training."""
        # Extract unique classes from annotations
        classes = set()
        for example in self.examples:
            if example.annotations and "classes" in example.annotations:
                classes.update(example.annotations["classes"])
        
        # If no classes specified, use default
        if not classes:
            classes = ["object"]
        
        classes = sorted(list(classes))
        
        data_config = {
            "path": str(self.dataset_path.absolute()),
            "train": "train/images",
            "val": "val/images",
            "nc": len(classes),
            "names": classes
        }
        
        yaml_path = self.dataset_path / "data.yaml"
        with open(yaml_path, 'w') as f:
            import yaml
            yaml.dump(data_config, f, default_flow_style=False)
        
        logger.info(f"Created data.yaml with {len(classes)} classes")
    
    def train_few_shot(self, base_model: str = "yolov8n.pt",
                      epochs: int = 10,
                      batch_size: int = 4,
                      learning_rate: float = 0.001) -> Any:
        """Train model with few-shot learning."""
        try:
            from ultralytics import YOLO
            
            # Prepare dataset
            dataset_path = self.prepare_dataset()
            
            # Load base model
            self.model = YOLO(base_model)
            
            # Update training status
            self.training_status = {
                "status": "training",
                "current_epoch": 0,
                "total_epochs": epochs,
                "loss": None,
                "metrics": {}
            }
            
            # Configure training parameters
            train_args = {
                "data": str(dataset_path / "data.yaml"),
                "epochs": epochs,
                "batch": batch_size,
                "lr0": learning_rate,
                "lrf": 0.1,  # Final learning rate factor
                "imgsz": 640,
                "device": self._detect_best_device(),
                "workers": 4,
                "patience": 5,  # Early stopping
                "save": True,
                "project": str(self.dataset_path / "runs"),
                "name": "few_shot_train",
                "exist_ok": True,
                "pretrained": True,
                "optimizer": "AdamW",
                "cos_lr": True,  # Cosine learning rate schedule
                "augment": True,  # Additional augmentation during training
                "cache": True,  # Cache images for faster training
            }
            
            # Adjust for few-shot scenario
            if len(self.examples) < 50:
                train_args["patience"] = 10  # More patience for small datasets
                train_args["warmup_epochs"] = 1  # Shorter warmup
            
            # Train model
            logger.info(f"Starting few-shot training with {len(self.examples)} examples")
            results = self.model.train(**train_args)
            
            # Update status
            self.training_status["status"] = "completed"
            self.training_status["current_epoch"] = epochs
            
            # Extract metrics
            if results:
                self.training_status["metrics"] = {
                    "mAP50": float(results.box.map50) if hasattr(results.box, "map50") else None,
                    "mAP50-95": float(results.box.map) if hasattr(results.box, "map") else None,
                    "precision": float(results.box.p) if hasattr(results.box, "p") else None,
                    "recall": float(results.box.r) if hasattr(results.box, "r") else None,
                }
            
            logger.info(f"Training completed. Metrics: {self.training_status['metrics']}")
            return self.model
            
        except ImportError:
            logger.error("ultralytics package not installed. Install with: pip install ultralytics")
            raise
        except Exception as e:
            self.training_status["status"] = "failed"
            logger.error(f"Training failed: {e}")
            raise
    
    def _detect_best_device(self) -> str:
        """Detect the best available device for training."""
        try:
            import torch
            import platform
            
            # Check for Apple Silicon
            if platform.system() == "Darwin" and platform.processor() == "arm":
                if torch.backends.mps.is_available():
                    return "mps"
            
            # Check for NVIDIA GPU
            if torch.cuda.is_available():
                return 0  # Use first GPU
            
            return "cpu"
            
        except ImportError:
            return "cpu"
    
    def evaluate(self, test_data: List[Union[Path, str]]) -> Dict[str, float]:
        """Evaluate model performance."""
        if not self.model:
            raise ValueError("No model loaded. Train a model first.")
        
        try:
            # Run validation
            metrics = self.model.val(data=str(self.dataset_path / "data.yaml"))
            
            # Extract key metrics
            results = {
                "mAP50": float(metrics.box.map50),
                "mAP50-95": float(metrics.box.map),
                "precision": float(metrics.box.p),
                "recall": float(metrics.box.r),
                "num_test_images": len(test_data)
            }
            
            logger.info(f"Evaluation results: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def save_model(self, output_path: Path) -> None:
        """Save trained model."""
        if not self.model:
            raise ValueError("No model to save. Train a model first.")
        
        try:
            # Save model weights
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Export to PyTorch format
            if output_path.suffix == ".pt":
                self.model.save(str(output_path))
            else:
                # Export to other formats
                format_map = {
                    ".onnx": "onnx",
                    ".tflite": "tflite",
                    ".mlmodel": "coreml"
                }
                export_format = format_map.get(output_path.suffix, "torchscript")
                self.model.export(format=export_format, imgsz=640)
            
            logger.info(f"Model saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return self.training_status.copy()
    
    def apply_augmentation(self, image: np.ndarray) -> List[np.ndarray]:
        """Apply data augmentation to increase training data."""
        augmented = []
        
        try:
            import cv2
            
            # Original image
            augmented.append(image)
            
            # Horizontal flip
            augmented.append(cv2.flip(image, 1))
            
            # Brightness adjustment
            bright = cv2.convertScaleAbs(image, alpha=1.2, beta=30)
            augmented.append(bright)
            
            # Slight rotation
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            
            for angle in [-10, 10]:
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(image, matrix, (width, height))
                augmented.append(rotated)
            
            # Gaussian blur
            blurred = cv2.GaussianBlur(image, (5, 5), 0)
            augmented.append(blurred)
            
            # Random crop and resize
            if height > 100 and width > 100:
                crop_h = int(height * 0.8)
                crop_w = int(width * 0.8)
                y = random.randint(0, height - crop_h)
                x = random.randint(0, width - crop_w)
                cropped = image[y:y+crop_h, x:x+crop_w]
                resized = cv2.resize(cropped, (width, height))
                augmented.append(resized)
            
            # Limit augmentations based on factor
            return augmented[:self.augmentation_factor]
            
        except ImportError:
            logger.warning("OpenCV not installed. Augmentation limited.")
            return [image]
        except Exception as e:
            logger.warning(f"Augmentation failed: {e}")
            return [image]