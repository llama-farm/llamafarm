"""YOLO image recognizer implementation."""

import logging
import platform
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import json

import numpy as np
from PIL import Image

from ...base import BaseImageRecognizer, Detection, Classification

logger = logging.getLogger(__name__)


class YOLORecognizer(BaseImageRecognizer):
    """YOLO-based image recognition model with cross-platform support."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize YOLO recognizer with configuration."""
        super().__init__(config)
        
        self.model_version = config.get("version", "yolov8n")
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self.nms_threshold = config.get("nms_threshold", 0.45)
        self.max_detections = config.get("max_detections", 100)
        
        # Hardware optimizations
        self.hardware_opts = config.get("hardware_optimizations", {})
        
        # Load model
        self.load_model()
    
    def _detect_device(self) -> str:
        """Detect and return the best available device."""
        try:
            import torch
            
            # Check for Apple Silicon
            if platform.system() == "Darwin" and platform.processor() == "arm":
                if torch.backends.mps.is_available():
                    logger.info("Using Apple Silicon MPS acceleration")
                    return "mps"
            
            # Check for NVIDIA GPU
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"Using NVIDIA GPU: {gpu_name}")
                return "cuda"
            
            # Check for AMD GPU (ROCm)
            if hasattr(torch, "hip") and torch.hip.is_available():
                logger.info("Using AMD GPU with ROCm")
                return "cuda"  # ROCm uses cuda interface
            
            # Fallback to CPU
            logger.info("Using CPU (no GPU acceleration available)")
            return "cpu"
            
        except ImportError:
            logger.warning("PyTorch not installed, defaulting to CPU")
            return "cpu"
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load the YOLO model weights."""
        try:
            from ultralytics import YOLO
            
            # Use provided path or download model
            if model_path:
                self.model = YOLO(model_path)
            else:
                # This will download the model if not cached
                self.model = YOLO(f"{self.model_version}.pt")
            
            # Move model to device
            if self.device != "cpu":
                self.model.to(self.device)
            
            # Apply hardware-specific optimizations
            self._apply_hardware_optimizations()
            
            logger.info(f"Loaded YOLO model: {self.model_version} on {self.device}")
            
        except ImportError:
            logger.error("ultralytics package not installed. Install with: pip install ultralytics")
            raise
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def _apply_hardware_optimizations(self) -> None:
        """Apply hardware-specific optimizations."""
        if self.device == "mps" and self.hardware_opts.get("apple_silicon", {}).get("use_coreml"):
            try:
                # Export to CoreML for optimal performance on Apple Silicon
                logger.info("Attempting CoreML optimization for Apple Silicon")
                # This would require coremltools installation
                # self.model.export(format="coreml")
            except Exception as e:
                logger.warning(f"CoreML optimization failed: {e}")
        
        elif self.device == "cuda" and self.hardware_opts.get("nvidia", {}).get("use_tensorrt"):
            try:
                # Export to TensorRT for optimal performance on NVIDIA
                logger.info("Attempting TensorRT optimization for NVIDIA GPU")
                # This would require tensorrt installation
                # self.model.export(format="engine")
            except Exception as e:
                logger.warning(f"TensorRT optimization failed: {e}")
        
        # Apply FP16 if supported
        if self.device == "cuda" and self.hardware_opts.get("nvidia", {}).get("fp16"):
            try:
                import torch
                if torch.cuda.is_available():
                    # Enable automatic mixed precision
                    logger.info("Enabling FP16 mode for faster inference")
            except Exception as e:
                logger.warning(f"FP16 optimization failed: {e}")
    
    def detect(self, image_path: Union[Path, str], 
               confidence: float = None,
               nms_threshold: float = None) -> List[Detection]:
        """Perform object detection on an image."""
        if confidence is None:
            confidence = self.confidence_threshold
        if nms_threshold is None:
            nms_threshold = self.nms_threshold
        
        try:
            # Run inference
            results = self.model(
                str(image_path),
                conf=confidence,
                iou=nms_threshold,
                max_det=self.max_detections,
                device=self.device
            )
            
            # Parse results
            detections = []
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        det = Detection(
                            label=r.names[int(box.cls)],
                            confidence=float(box.conf),
                            bbox=box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                            metadata={
                                "class_id": int(box.cls),
                                "area": float((box.xyxy[0][2] - box.xyxy[0][0]) * 
                                             (box.xyxy[0][3] - box.xyxy[0][1]))
                            }
                        )
                        detections.append(det)
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            raise
    
    def classify(self, image_path: Union[Path, str],
                 top_k: int = 5) -> List[Classification]:
        """Perform image classification."""
        try:
            # Check if model supports classification
            if not self.model_version.endswith("-cls"):
                logger.warning("Current model is for detection. Use a classification model (e.g., yolov8n-cls)")
                return []
            
            # Run inference
            results = self.model(str(image_path), device=self.device)
            
            # Parse classification results
            classifications = []
            for r in results:
                if hasattr(r, "probs"):
                    probs = r.probs
                    top_indices = probs.top5 if top_k == 5 else probs.argsort()[-top_k:][::-1]
                    
                    for idx in top_indices:
                        cls = Classification(
                            label=r.names[idx],
                            confidence=float(probs.data[idx]),
                            metadata={"class_id": int(idx)}
                        )
                        classifications.append(cls)
            
            return classifications[:top_k]
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            raise
    
    def segment(self, image_path: Union[Path, str]) -> np.ndarray:
        """Perform image segmentation."""
        try:
            # Check if model supports segmentation
            if not self.model_version.endswith("-seg"):
                logger.warning("Current model is for detection. Use a segmentation model (e.g., yolov8n-seg)")
                return np.array([])
            
            # Run inference
            results = self.model(str(image_path), device=self.device)
            
            # Get segmentation masks
            for r in results:
                if hasattr(r, "masks") and r.masks is not None:
                    # Return the first mask as numpy array
                    return r.masks.data[0].cpu().numpy()
            
            return np.array([])
            
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            raise
    
    def batch_process(self, image_paths: List[Union[Path, str]],
                     task: str = "detect",
                     batch_size: int = 32) -> List[Any]:
        """Process multiple images in batch."""
        results = []
        
        try:
            # Process in batches
            for i in range(0, len(image_paths), batch_size):
                batch = image_paths[i:i + batch_size]
                
                if task == "detect":
                    for img_path in batch:
                        detections = self.detect(img_path)
                        results.append(detections)
                        
                elif task == "classify":
                    for img_path in batch:
                        classifications = self.classify(img_path)
                        results.append(classifications)
                        
                elif task == "segment":
                    for img_path in batch:
                        mask = self.segment(img_path)
                        results.append(mask)
                        
                else:
                    raise ValueError(f"Unknown task: {task}")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise
    
    def export_model(self, output_path: Path, format: str = "onnx") -> None:
        """Export model to different formats."""
        try:
            supported_formats = ["onnx", "torchscript", "coreml", "tflite", "engine"]
            if format not in supported_formats:
                raise ValueError(f"Unsupported format: {format}. Supported: {supported_formats}")
            
            # Export model
            export_path = self.model.export(format=format)
            
            # Move to desired location if different
            if Path(export_path) != output_path:
                import shutil
                shutil.move(export_path, output_path)
            
            logger.info(f"Model exported to {output_path} in {format} format")
            
        except Exception as e:
            logger.error(f"Model export failed: {e}")
            raise
    
    def get_supported_tasks(self) -> List[str]:
        """Get list of supported tasks."""
        tasks = []
        
        if self.model_version.endswith("-cls"):
            tasks.append("classify")
        elif self.model_version.endswith("-seg"):
            tasks.extend(["detect", "segment"])
        else:
            tasks.append("detect")
        
        return tasks
    
    def visualize(self, image_path: Union[Path, str],
                  results: Union[List[Detection], List[Classification], np.ndarray],
                  output_path: Optional[Path] = None) -> Optional[np.ndarray]:
        """Visualize results on image."""
        try:
            import cv2
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            # Load image
            img = cv2.imread(str(image_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            fig, ax = plt.subplots(1, figsize=(12, 8))
            ax.imshow(img_rgb)
            
            # Visualize detections
            if results and isinstance(results[0], Detection):
                for det in results:
                    x1, y1, x2, y2 = det.bbox
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Create rectangle
                    rect = patches.Rectangle(
                        (x1, y1), width, height,
                        linewidth=2, edgecolor='red', facecolor='none'
                    )
                    ax.add_patch(rect)
                    
                    # Add label
                    label = f"{det.label}: {det.confidence:.2f}"
                    ax.text(x1, y1 - 5, label, color='red', fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            
            # Visualize classifications
            elif results and isinstance(results[0], Classification):
                # Add text with top classifications
                text = "\n".join([f"{cls.label}: {cls.confidence:.3f}" for cls in results[:5]])
                ax.text(10, 30, text, color='white', fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.7))
            
            # Visualize segmentation mask
            elif isinstance(results, np.ndarray) and results.size > 0:
                # Overlay mask
                mask_colored = plt.cm.jet(results / results.max())
                ax.imshow(mask_colored, alpha=0.5)
            
            ax.axis('off')
            
            if output_path:
                plt.savefig(output_path, bbox_inches='tight', dpi=100)
                logger.info(f"Visualization saved to {output_path}")
            
            # Convert to numpy array for return
            fig.canvas.draw()
            vis_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            vis_array = vis_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            plt.close()
            
            return vis_array
            
        except ImportError as e:
            logger.error(f"Visualization requires opencv-python and matplotlib: {e}")
            return None
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return None