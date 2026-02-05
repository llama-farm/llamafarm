# LlamaFarm Vision Capabilities - Comprehensive Plan

> **Status**: Planning Document (DO NOT IMPLEMENT)  
> **Branch**: `feat/vision`  
> **Created**: 2026-02-04  
> **Author**: AI Architect Agent

> ⛔ **CRITICAL: GIT RULES** ⛔
> - **ONLY commit to branch `feat/vision`**
> - **NEVER commit to `main` — EVER**
> - Always verify your branch before committing: `git branch --show-current`

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Requirements Analysis](#requirements-analysis)
3. [Research Findings](#research-findings)
4. [Architecture Design](#architecture-design)
5. [Component Specifications](#component-specifications)
6. [Training Pipeline](#training-pipeline)
7. [Storage & Retention](#storage--retention)
8. [Multi-Model Validation](#multi-model-validation)
9. [Human-in-the-Loop](#human-in-the-loop)
10. [Edge & Mobile Considerations](#edge--mobile-considerations)
11. [API Design](#api-design)
12. [API & Router Structure](#api--router-structure)
13. [Image RAG Integration](#image-rag-integration) ← **NEW: Multimodal RAG**
14. [Implementation Phases](#implementation-phases)
15. [Testing Strategy](#testing-strategy)
16. [Future Considerations](#future-considerations)
17. [Development Workflow](#development-workflow)

---

## Executive Summary

This plan outlines the addition of comprehensive vision capabilities to LlamaFarm, focusing on edge-first deployment with optional cloud/server escalation. The system will support:

- **Streaming Vision** (1 FPS anomaly detection)
- **Image Classification** (few-shot learning)
- **Object Detection/Recognition** (YOLO-based)
- **Image Segmentation** (SAM/MobileSAM)
- **Incremental Training** (continual learning without catastrophic forgetting)
- **Model Export/Import** (ONNX, CoreML, TensorRT)
- **Human-in-the-Loop Verification** (simple click-and-drag correction)

### Key Design Principles

1. **Non-breaking**: All changes are additive; existing APIs remain unchanged
2. **Edge-first**: Optimized for Mac (MPS), Linux (CUDA/CPU), with mobile research
3. **Privacy-preserving**: Local processing by default, optional escalation
4. **Incremental**: Models improve over time without full retraining
5. **Explainable**: Confidence scores, bounding boxes, and audit trails

---

## Requirements Analysis

### Functional Requirements

| Requirement | Description | Priority |
|-------------|-------------|----------|
| Streaming Vision | Process 1 image/second, return action only on anomaly | P0 |
| Image Classification | Classify images into custom categories | P0 |
| Object Detection | Detect and localize objects with bounding boxes | P0 |
| Image Segmentation | Pixel-level segmentation masks | P1 |
| Few-Shot Training | Train with 8-16 examples per class | P0 |
| Incremental Updates | Add new training data without full retrain | P0 |
| Model Export | Export to ONNX, CoreML, TensorRT | P1 |
| Model Import | Load pre-trained or transferred models | P0 |
| Image Retention | Configurable retention based on confidence | P0 |
| Multi-Model Validation | Escalate low-confidence to secondary model | P1 |
| Human Verification | Simple UI for correction/labeling | P1 |

### Non-Functional Requirements

| Requirement | Target |
|-------------|--------|
| Inference Latency | < 100ms per image (edge) |
| Training Time (few-shot) | < 5 minutes for 100 images |
| Memory Usage | < 2GB VRAM for inference |
| Storage Efficiency | < 1MB per retained image (compressed) |
| Model Size | < 50MB for edge deployment |

---

## Research Findings

### 1. Object Detection Models

**Recommendation: Ultralytics YOLO (v8/v11)**

| Model | mAP | Inference (GPU) | Size | Best For |
|-------|-----|-----------------|------|----------|
| YOLOv8n | 37.3 | 1.2ms | 6MB | Edge/Mobile |
| YOLOv8s | 44.9 | 1.4ms | 22MB | Balanced |
| YOLOv8m | 50.2 | 2.2ms | 52MB | Accuracy |
| YOLOv11n | 39.5 | 1.5ms | 5.4MB | Latest/Fastest |

**Why Ultralytics YOLO:**
- Unified Python API for detection, segmentation, classification, pose
- Built-in export to ONNX, CoreML, TensorRT, TFLite
- Active development, excellent documentation
- Supports training, fine-tuning, and transfer learning
- Apache 2.0 license (AGPL for some components - verify)

**Source**: [Ultralytics Docs](https://docs.ultralytics.com/)

### 2. Image Segmentation

**Recommendation: MobileSAM for edge, SAM2 for server**

| Model | Speed (GPU) | Memory | Use Case |
|-------|-------------|--------|----------|
| MobileSAM | 10-12ms | <50MB RAM | Edge/Mobile |
| FastSAM | 40ms | ~200MB | Balanced |
| SAM2-t | 100ms | ~400MB | Server/Quality |
| SAM2-b | 200ms | ~800MB | Highest Quality |

**Why MobileSAM:**
- 60x faster than original SAM
- Runs on ARM CPUs at <300ms/image
- Compatible with SAM prompts (points, boxes)
- Ultralytics integration available

**Source**: [MobileSAM Paper](https://arxiv.org/abs/2306.14289)

### 3. Image Classification (Few-Shot)

**Recommendation: CLIP embeddings + lightweight classifier**

**Approach:**
1. Use CLIP (ViT-B/32) or SigLIP for image embeddings
2. Pre-compute text embeddings for class names
3. Train simple classifier head (MLP or SetFit-style)
4. 8-16 examples sufficient for good accuracy

**Why CLIP-based:**
- Zero-shot capability out of the box
- Few-shot improves accuracy significantly
- Small embedding model (~150MB)
- Text-guided classification possible

**Alternative**: SetFit-style contrastive learning for purely visual classification

**Source**: [Few-Shot Learning on Edge with CLIP](https://itc.ktu.lt/index.php/ITC/article/view/36943)

### 4. Continual Learning (Avoiding Catastrophic Forgetting)

**Recommendation: Hybrid approach - EWC + Experience Replay**

**Strategies Evaluated:**

| Strategy | Pros | Cons | Recommendation |
|----------|------|------|----------------|
| Elastic Weight Consolidation (EWC) | Protects important weights | Requires Fisher info | Use for core model |
| Experience Replay | Simple, effective | Requires storage | Use for corrections |
| Progressive Networks | No forgetting | Model grows | Not recommended |
| Knowledge Distillation | Compact | Complex | Future consideration |

**Hybrid Approach:**
1. **EWC** for protecting learned features when adding new classes
2. **Experience Replay** buffer (1000 samples) for corrected examples
3. **Selective Replay** - prioritize low-confidence corrections

**Source**: [EWC Paper](https://arxiv.org/abs/1612.00796), [Avalanche Library](https://github.com/ContinualAI/avalanche)

### 5. Active Learning & Uncertainty Sampling

**Key Concepts:**
- **Uncertainty Sampling**: Prioritize samples where model is least confident
- **Diversity Sampling**: Ensure coverage of feature space
- **Query-by-Committee**: Multiple models disagree = high value sample

**Implementation:**
```
confidence_threshold = 0.7  # Below this, flag for review
uncertainty_threshold = 0.3  # entropy or 1 - max_prob
```

**Retention Policy Based on Confidence:**
| Confidence | Retention | Action |
|------------|-----------|--------|
| > 0.9 | 1 hour | Auto-process, minimal storage |
| 0.7 - 0.9 | 24 hours | Store for batch review |
| 0.5 - 0.7 | 7 days | Flag for secondary model |
| < 0.5 | 30 days | Require human verification |

**Source**: [Human-in-the-Loop ML Book](https://www.oreilly.com/library/view/human-in-the-loop-machine-learning/9781617296741/)

### 6. Model Export Formats

| Format | Platform | Use Case | Tool |
|--------|----------|----------|------|
| ONNX | Universal | Cross-platform inference | `model.export(format='onnx')` |
| CoreML | Apple (iOS/macOS) | Native Apple integration | `model.export(format='coreml')` |
| TensorRT | NVIDIA GPU | Maximum inference speed | `model.export(format='engine')` |
| TFLite | Android/Edge | Mobile deployment | `model.export(format='tflite')` |
| OpenVINO | Intel | Intel hardware optimization | `model.export(format='openvino')` |
| NCNN | Mobile | Lightweight mobile inference | `model.export(format='ncnn')` |

**Source**: [Ultralytics Export Docs](https://docs.ultralytics.com/modes/export/)

### 7. Annotation & Labeling Tools

**Recommendation: Embedded lightweight UI + CVAT export compatibility**

| Tool | Hosting | Strengths | Integration |
|------|---------|-----------|-------------|
| CVAT | Self-hosted | Video, interpolation | Export YOLO format |
| Label Studio | Self-hosted | Multi-modal | REST API |
| Custom UI | Embedded | Simple corrections | Native |

**For LlamaFarm:**
- Build simple correction UI (click to confirm/reject, drag to adjust bbox)
- Export to CVAT/Label Studio format for complex labeling
- Import from standard formats (COCO, YOLO, Pascal VOC)

### 8. Data Versioning & MLOps

**Recommendation: DVC + internal tracking**

| Tool | Purpose | Integration |
|------|---------|-------------|
| DVC | Dataset versioning | Git-like workflow |
| MLflow | Experiment tracking | Model registry |
| Internal | Lightweight tracking | SQLite metadata |

**Minimal Viable Approach:**
- SQLite database for model/dataset metadata
- Filesystem with structured directories for images
- JSON manifests for dataset versions
- Optional DVC integration for advanced users

---

## Architecture Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LlamaFarm Server                            │
│                         (Port 14345)                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Vision API Router                         │   │
│  │  /v1/vision/stream    - Streaming anomaly detection         │   │
│  │  /v1/vision/classify  - Image classification                │   │
│  │  /v1/vision/detect    - Object detection                    │   │
│  │  /v1/vision/segment   - Image segmentation                  │   │
│  │  /v1/vision/train     - Training endpoints                  │   │
│  │  /v1/vision/models    - Model management                    │   │
│  │  /v1/vision/review    - Human review queue                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                 Vision Service Layer                         │   │
│  │  - Request validation & routing                              │   │
│  │  - Confidence threshold management                           │   │
│  │  - Retention policy enforcement                              │   │
│  │  - Multi-model cascade orchestration                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼ HTTP/gRPC
┌─────────────────────────────────────────────────────────────────────┐
│                      Universal Runtime                              │
│                      (Port 11540)                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   Vision Router                              │   │
│  │  /v1/vision/infer     - Run inference                       │   │
│  │  /v1/vision/train     - Train/fine-tune                     │   │
│  │  /v1/vision/export    - Export model                        │   │
│  │  /v1/vision/backends  - List available backends             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Vision Models                             │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │   │
│  │  │ YOLO Model   │ │ SAM Model    │ │ CLIP Model   │        │   │
│  │  │ (Detection)  │ │ (Segment)    │ │ (Classify)   │        │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘        │   │
│  │  ┌──────────────┐ ┌──────────────┐                          │   │
│  │  │ Streaming    │ │ Training     │                          │   │
│  │  │ Detector     │ │ Pipeline     │                          │   │
│  │  └──────────────┘ └──────────────┘                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Storage Layer                             │   │
│  │  ~/.llamafarm/vision/                                        │   │
│  │    ├── models/          # Trained models                    │   │
│  │    ├── datasets/        # Training datasets                 │   │
│  │    ├── review_queue/    # Images pending review             │   │
│  │    ├── replay_buffer/   # Experience replay samples         │   │
│  │    └── exports/         # Exported models                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Streaming Vision Flow

```
Camera/Source                LlamaFarm                    Action
     │                           │                           │
     │ ──── Image (1 FPS) ────►  │                           │
     │                           │                           │
     │                    ┌──────┴──────┐                    │
     │                    │  Inference  │                    │
     │                    │  (< 100ms)  │                    │
     │                    └──────┬──────┘                    │
     │                           │                           │
     │                    ┌──────┴──────┐                    │
     │                    │ Confidence  │                    │
     │                    │   Check     │                    │
     │                    └──────┬──────┘                    │
     │                           │                           │
     │              ┌────────────┼────────────┐              │
     │              │            │            │              │
     │         conf > 0.9   0.5 < conf   conf < 0.5         │
     │              │            │            │              │
     │              ▼            ▼            ▼              │
     │         HTTP 200    Flag Review   Escalate to        │
     │         (no body)   + HTTP 200    Secondary Model    │
     │                                         │              │
     │                                         ▼              │
     │                                  ┌──────────────┐      │
     │                                  │ If anomaly:  │      │
     │                                  │ Return action│      │
     │ ◄─────────────────────────────── └──────────────┘      │
     │    {action: "alert", bbox: [...], confidence: 0.87}    │
```

---

## Component Specifications

### 1. VisionModel Base Class

```python
# runtimes/universal/models/vision_model.py

from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal
from .base import BaseModel

@dataclass
class VisionResult:
    """Base result for all vision operations."""
    confidence: float
    inference_time_ms: float
    model_name: str

@dataclass
class DetectionResult(VisionResult):
    """Object detection result."""
    boxes: list[dict]  # [{x1, y1, x2, y2, class, confidence}]
    class_names: list[str]

@dataclass
class ClassificationResult(VisionResult):
    """Classification result."""
    class_name: str
    class_id: int
    all_scores: dict[str, float]

@dataclass
class SegmentationResult(VisionResult):
    """Segmentation result."""
    masks: list[np.ndarray]  # Binary masks
    boxes: list[dict]  # Bounding boxes for each mask

class VisionModel(BaseModel):
    """Base class for all vision models."""
    
    @abstractmethod
    async def infer(self, image: bytes | np.ndarray) -> VisionResult:
        """Run inference on a single image."""
        pass
    
    @abstractmethod
    async def train(
        self,
        dataset_path: str,
        epochs: int = 10,
        batch_size: int = 16,
    ) -> dict:
        """Train or fine-tune the model."""
        pass
    
    @abstractmethod
    async def export(
        self,
        format: Literal["onnx", "coreml", "tensorrt", "tflite"],
        output_path: str,
    ) -> str:
        """Export model to specified format."""
        pass
```

### 2. YOLO Detection Model

```python
# runtimes/universal/models/yolo_model.py

from ultralytics import YOLO
from .vision_model import VisionModel, DetectionResult

class YOLOModel(VisionModel):
    """YOLO-based object detection model."""
    
    SUPPORTED_VARIANTS = ["yolov8n", "yolov8s", "yolov8m", "yolov11n", "yolov11s"]
    
    def __init__(
        self,
        model_id: str = "yolov8n",
        device: str = "auto",
        confidence_threshold: float = 0.5,
    ):
        self.model_id = model_id
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.model: YOLO | None = None
    
    async def load(self) -> None:
        """Load YOLO model."""
        # Load from HuggingFace, local path, or Ultralytics hub
        self.model = YOLO(self.model_id)
        if self.device != "auto":
            self.model.to(self.device)
    
    async def infer(self, image: bytes | np.ndarray) -> DetectionResult:
        """Run detection on image."""
        results = self.model(image, conf=self.confidence_threshold)
        # Process results...
        return DetectionResult(...)
    
    async def train(
        self,
        dataset_path: str,
        epochs: int = 10,
        batch_size: int = 16,
        resume: bool = False,
    ) -> dict:
        """Fine-tune YOLO on custom dataset."""
        results = self.model.train(
            data=dataset_path,
            epochs=epochs,
            batch=batch_size,
            resume=resume,
            # Continual learning settings
            freeze=10,  # Freeze first 10 layers
        )
        return {"metrics": results.results_dict}
    
    async def export(self, format: str, output_path: str) -> str:
        """Export to deployment format."""
        return self.model.export(format=format)
```

### 3. Classification Model (CLIP-based)

```python
# runtimes/universal/models/clip_classifier.py

import torch
from transformers import CLIPModel, CLIPProcessor
from .vision_model import VisionModel, ClassificationResult

class CLIPClassifier(VisionModel):
    """CLIP-based few-shot image classifier."""
    
    def __init__(
        self,
        model_id: str = "openai/clip-vit-base-patch32",
        device: str = "auto",
    ):
        self.model_id = model_id
        self.device = device
        self.model: CLIPModel | None = None
        self.processor: CLIPProcessor | None = None
        self.class_embeddings: torch.Tensor | None = None
        self.class_names: list[str] = []
    
    async def set_classes(self, class_names: list[str]) -> None:
        """Set classification classes (zero-shot)."""
        self.class_names = class_names
        text_inputs = self.processor(
            text=[f"a photo of a {c}" for c in class_names],
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            self.class_embeddings = self.model.get_text_features(**text_inputs)
            self.class_embeddings = self.class_embeddings / self.class_embeddings.norm(
                dim=-1, keepdim=True
            )
    
    async def train_few_shot(
        self,
        images: list[bytes],
        labels: list[str],
        epochs: int = 5,
    ) -> dict:
        """Train classifier head with few examples."""
        # Extract image embeddings
        # Train simple MLP classifier head
        # Use contrastive loss (SetFit-style)
        pass
    
    async def infer(self, image: bytes | np.ndarray) -> ClassificationResult:
        """Classify image."""
        image_input = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.model.get_image_features(**image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ self.class_embeddings.T).squeeze()
            probs = similarity.softmax(dim=-1)
        
        best_idx = probs.argmax().item()
        return ClassificationResult(
            class_name=self.class_names[best_idx],
            class_id=best_idx,
            confidence=probs[best_idx].item(),
            all_scores=dict(zip(self.class_names, probs.tolist())),
        )
```

### 4. Streaming Vision Detector

```python
# runtimes/universal/models/streaming_vision.py

import asyncio
from dataclasses import dataclass
from typing import Callable, Any
from .yolo_model import YOLOModel

@dataclass
class StreamingConfig:
    """Configuration for streaming vision."""
    target_fps: float = 1.0
    confidence_threshold: float = 0.7
    escalation_threshold: float = 0.5
    action_classes: list[str] = None  # Classes that trigger action
    cooldown_seconds: float = 5.0  # Min time between actions

class StreamingVisionDetector:
    """Streaming vision detector for anomaly-style detection."""
    
    def __init__(
        self,
        primary_model: YOLOModel,
        secondary_model: YOLOModel | None = None,
        config: StreamingConfig = None,
    ):
        self.primary = primary_model
        self.secondary = secondary_model
        self.config = config or StreamingConfig()
        self.last_action_time = 0
        self._running = False
    
    async def process_frame(
        self,
        image: bytes,
        callback: Callable[[dict], Any] | None = None,
    ) -> dict:
        """Process a single frame.
        
        Returns:
            - {"status": "ok"} for normal frames
            - {"status": "action", "detections": [...]} for anomalies
            - {"status": "review", "image_id": "..."} for uncertain frames
        """
        result = await self.primary.infer(image)
        
        # Check if any detection triggers action
        action_detections = [
            d for d in result.boxes
            if d["class"] in self.config.action_classes
            and d["confidence"] >= self.config.confidence_threshold
        ]
        
        if action_detections:
            # Check cooldown
            if time.time() - self.last_action_time < self.config.cooldown_seconds:
                return {"status": "ok", "suppressed": True}
            
            self.last_action_time = time.time()
            response = {
                "status": "action",
                "detections": action_detections,
                "confidence": max(d["confidence"] for d in action_detections),
            }
            if callback:
                await callback(response)
            return response
        
        # Check for low-confidence detections requiring review
        uncertain = [
            d for d in result.boxes
            if d["confidence"] < self.config.escalation_threshold
        ]
        
        if uncertain and self.secondary:
            # Escalate to secondary model
            secondary_result = await self.secondary.infer(image)
            # Combine results...
        
        if uncertain:
            # Queue for review
            image_id = await self._queue_for_review(image, result)
            return {"status": "review", "image_id": image_id}
        
        return {"status": "ok"}
```

---

## Training Pipeline

### Incremental Training Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Training Pipeline                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Data Ingestion                                               │
│     ┌──────────┐    ┌──────────┐    ┌──────────┐               │
│     │ New Data │ ─► │ Validate │ ─► │ Augment  │               │
│     └──────────┘    └──────────┘    └──────────┘               │
│                                           │                      │
│  2. Experience Replay Buffer              ▼                      │
│     ┌─────────────────────────────────────────┐                 │
│     │ Replay Buffer (max 1000 samples)        │                 │
│     │ - Priority: corrections > low-conf      │                 │
│     │ - Stratified by class                   │                 │
│     └─────────────────────────────────────────┘                 │
│                           │                                      │
│  3. Training Loop         ▼                                      │
│     ┌─────────────────────────────────────────┐                 │
│     │ Mixed Batch:                            │                 │
│     │ - 70% new data                          │                 │
│     │ - 30% replay buffer                     │                 │
│     └─────────────────────────────────────────┘                 │
│                           │                                      │
│  4. EWC Regularization    ▼                                      │
│     ┌─────────────────────────────────────────┐                 │
│     │ Loss = Task_Loss + λ * EWC_Loss         │                 │
│     │ EWC_Loss = Σ F_i * (θ_i - θ*_i)²       │                 │
│     └─────────────────────────────────────────┘                 │
│                           │                                      │
│  5. Validation & Save     ▼                                      │
│     ┌─────────────────────────────────────────┐                 │
│     │ - Validate on held-out set              │                 │
│     │ - Check for regression on old classes   │                 │
│     │ - Save if improved, else rollback       │                 │
│     └─────────────────────────────────────────┘                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Training Configuration

```yaml
# Example training config
training:
  # Base settings
  epochs: 10
  batch_size: 16
  learning_rate: 0.001
  
  # Continual learning
  continual:
    enabled: true
    strategy: "ewc_replay"  # ewc, replay, ewc_replay
    ewc_lambda: 0.4  # EWC regularization strength
    replay_buffer_size: 1000
    replay_ratio: 0.3  # 30% of batch from replay
    
  # Data augmentation
  augmentation:
    horizontal_flip: true
    vertical_flip: false
    rotation_range: 15
    brightness_range: [0.8, 1.2]
    mosaic: true  # YOLO mosaic augmentation
    
  # Validation
  validation:
    split: 0.2
    check_regression: true
    regression_threshold: 0.05  # Max allowed drop on old classes
```

### Correction Workflow

```
User Correction                 System Response
      │                              │
      │ ── Reject Detection ──────►  │
      │    (false positive)          │
      │                              │
      │                    ┌─────────┴─────────┐
      │                    │ Add to replay     │
      │                    │ buffer as         │
      │                    │ negative example  │
      │                    └─────────┬─────────┘
      │                              │
      │ ── Add Detection ─────────►  │
      │    (false negative)          │
      │                              │
      │                    ┌─────────┴─────────┐
      │                    │ Add to replay     │
      │                    │ buffer with       │
      │                    │ correct label     │
      │                    └─────────┬─────────┘
      │                              │
      │ ── Adjust BBox ───────────►  │
      │    (localization error)      │
      │                              │
      │                    ┌─────────┴─────────┐
      │                    │ Add corrected     │
      │                    │ annotation to     │
      │                    │ replay buffer     │
      │                    └─────────┬─────────┘
      │                              │
      │                              ▼
      │                    ┌───────────────────┐
      │                    │ When buffer full  │
      │                    │ or manual trigger:│
      │                    │ Incremental Train │
      │                    └───────────────────┘
```

---

## Storage & Retention

### Directory Structure

```
~/.llamafarm/vision/
├── models/
│   ├── detection/
│   │   ├── yolov8n_base.pt           # Base model
│   │   ├── yolov8n_custom_v1.pt      # Fine-tuned v1
│   │   ├── yolov8n_custom_v2.pt      # Fine-tuned v2 (latest)
│   │   └── metadata.json             # Model registry
│   ├── classification/
│   │   ├── clip_base/
│   │   └── custom_classifier_v1/
│   └── segmentation/
│       └── mobilesam/
│
├── datasets/
│   ├── {dataset_name}/
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── val/
│   │   ├── labels/
│   │   │   ├── train/
│   │   │   └── val/
│   │   ├── dataset.yaml              # YOLO format config
│   │   └── manifest.json             # Version info
│   └── ...
│
├── review_queue/
│   ├── pending/
│   │   ├── {timestamp}_{uuid}.jpg
│   │   └── {timestamp}_{uuid}.json   # Metadata + predictions
│   ├── approved/                      # Moved after human approval
│   └── rejected/                      # Moved after rejection
│
├── replay_buffer/
│   ├── corrections/                   # Human-corrected samples
│   ├── low_confidence/                # Auto-flagged samples
│   └── buffer_state.json              # Buffer metadata
│
├── exports/
│   ├── onnx/
│   ├── coreml/
│   ├── tensorrt/
│   └── tflite/
│
└── config/
    ├── retention_policy.yaml
    ├── training_config.yaml
    └── model_registry.json
```

### Retention Policy Configuration

```yaml
# ~/.llamafarm/vision/config/retention_policy.yaml

retention:
  # Based on confidence levels
  high_confidence:  # > 0.9
    retention_hours: 1
    storage: "temp"
    action: "auto_delete"
    
  medium_confidence:  # 0.7 - 0.9
    retention_hours: 24
    storage: "review_queue"
    action: "batch_review"
    
  low_confidence:  # 0.5 - 0.7
    retention_days: 7
    storage: "review_queue"
    action: "flag_for_secondary"
    
  very_low_confidence:  # < 0.5
    retention_days: 30
    storage: "review_queue"
    action: "require_human"
    
  # Special cases
  corrections:
    retention_days: 90
    storage: "replay_buffer"
    max_samples: 1000
    
  false_positives:
    retention_days: 90
    storage: "replay_buffer"
    
  # Storage limits
  limits:
    max_review_queue_gb: 10
    max_replay_buffer_gb: 5
    max_total_vision_gb: 50
    
  # Cleanup
  cleanup:
    schedule: "daily"
    time: "03:00"
    compress_after_days: 7
    delete_after_days: 90
```

### Image Storage Format

```python
# Efficient image storage with metadata

@dataclass
class StoredImage:
    """Image stored for review or replay."""
    id: str  # UUID
    timestamp: datetime
    image_path: str  # Relative path to JPEG
    thumbnail_path: str  # 128x128 thumbnail
    
    # Original prediction
    prediction: dict
    confidence: float
    model_name: str
    model_version: str
    
    # Review status
    status: Literal["pending", "approved", "rejected", "corrected"]
    reviewed_at: datetime | None
    reviewed_by: str | None
    
    # Correction data (if corrected)
    correction: dict | None
    
    # Metadata
    source: str  # e.g., "stream:camera1", "upload:batch123"
    tags: list[str]

# Storage as compressed JPEG + JSON sidecar
# image: {id}.jpg (quality=85, max 1280px)
# metadata: {id}.json
```

---

## Multi-Model Validation

### Cascade Architecture

```
                    Input Image
                         │
                         ▼
              ┌─────────────────────┐
              │   Primary Model     │
              │   (Fast, Edge)      │
              │   YOLOv8n / v11n    │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Confidence Check   │
              └──────────┬──────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    conf > 0.9      0.5-0.9         conf < 0.5
         │               │               │
         ▼               ▼               ▼
    ┌─────────┐   ┌───────────┐   ┌───────────────┐
    │ Accept  │   │  Log for  │   │   Secondary   │
    │ Result  │   │  Review   │   │    Model      │
    └─────────┘   └───────────┘   │  (YOLOv8m or  │
                                  │   cloud API)  │
                                  └───────┬───────┘
                                          │
                                          ▼
                                  ┌───────────────┐
                                  │   Ensemble    │
                                  │   Decision    │
                                  └───────┬───────┘
                                          │
                         ┌────────────────┼────────────────┐
                         │                │                │
                    Agreement         Disagree        Both Low
                         │           (flag)           Conf
                         │                │                │
                         ▼                ▼                ▼
                    ┌─────────┐   ┌───────────┐   ┌───────────┐
                    │ Accept  │   │  Human    │   │  Human    │
                    │ Higher  │   │  Review   │   │  Review   │
                    │ Conf    │   │  Queue    │   │  Priority │
                    └─────────┘   └───────────┘   └───────────┘
```

### Ensemble Configuration

```yaml
# Multi-model validation config

cascade:
  primary:
    model: "yolov8n"
    device: "auto"
    confidence_threshold: 0.5
    
  secondary:
    model: "yolov8m"  # Larger model
    device: "auto"
    # Or use cloud API
    # type: "api"
    # endpoint: "https://api.example.com/v1/detect"
    
  escalation:
    # When to escalate to secondary
    threshold: 0.5
    # Rate limit for secondary (cost control)
    max_per_minute: 10
    
  ensemble:
    strategy: "confidence_weighted"  # or "voting", "union"
    agreement_threshold: 0.7  # IoU for box matching
    
  cloud_fallback:
    enabled: false
    endpoint: null
    api_key_env: "VISION_CLOUD_API_KEY"
    timeout_seconds: 5
```

---

## Human-in-the-Loop

### Review Interface Requirements

**Simple Correction UI (Embedded in Designer)**

```
┌─────────────────────────────────────────────────────────────────┐
│  Review Queue                                    [Filter ▼] [▶] │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                          │   │
│  │                    [Image Preview]                       │   │
│  │                                                          │   │
│  │        ┌──────────────────────┐                         │   │
│  │        │  person (0.67)       │ ← Draggable bbox        │   │
│  │        │                      │                          │   │
│  │        └──────────────────────┘                         │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Prediction: person (67%)          Model: yolov8n v2            │
│  Source: camera:front_door         Time: 2 hours ago            │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ ✓ Correct│  │ ✗ Wrong  │  │ ✎ Adjust │  │ + Add Box│       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│                                                                  │
│  Class: [person     ▼]    [Skip] [Save & Next]                  │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  Queue: 47 pending │ Today: 12 reviewed │ Accuracy: 94%         │
└─────────────────────────────────────────────────────────────────┘
```

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `1` | Mark as Correct |
| `2` | Mark as Wrong (False Positive) |
| `3` | Enter Adjust Mode |
| `A` | Add New Bounding Box |
| `D` | Delete Selected Box |
| `←/→` | Previous/Next Image |
| `Enter` | Save & Next |
| `Esc` | Cancel/Exit |

### API Endpoints for Review

```yaml
# Review Queue API

/v1/vision/review:
  GET:
    description: "List pending reviews"
    params:
      - status: pending|approved|rejected
      - limit: int
      - offset: int
      - sort: timestamp|confidence
    response:
      items: [ReviewItem]
      total: int
      
  POST:
    description: "Submit review decision"
    body:
      image_id: str
      decision: correct|wrong|adjusted
      corrections:  # If adjusted
        boxes: [{x1, y1, x2, y2, class}]
    response:
      success: bool
      next_image_id: str | null

/v1/vision/review/batch:
  POST:
    description: "Batch approve/reject"
    body:
      image_ids: [str]
      decision: correct|wrong
```

---

## Edge & Mobile Considerations

### Platform Support Matrix

| Platform | Runtime | Model Format | Notes |
|----------|---------|--------------|-------|
| macOS (Apple Silicon) | MPS | PyTorch, CoreML | Native MPS acceleration |
| macOS (Intel) | CPU | PyTorch, ONNX | Slower, but functional |
| Linux (NVIDIA) | CUDA | PyTorch, TensorRT | Best performance |
| Linux (CPU) | CPU | PyTorch, ONNX | OpenVINO for Intel |
| Linux (ARM) | CPU | ONNX, TFLite | Raspberry Pi, Jetson |
| iOS | CoreML | CoreML | Requires separate app |
| Android | TFLite | TFLite | Requires separate app |

### Mobile Research (Future Project)

**Approach 1: Native App with LlamaFarm Sync**

```
┌─────────────────┐          ┌─────────────────┐
│   Mobile App    │          │   LlamaFarm     │
│   (iOS/Android) │  ◄────►  │   Server        │
│                 │   WiFi   │                 │
│  - CoreML/TFLite│          │  - Model sync   │
│  - Local infer  │          │  - Training     │
│  - Offline mode │          │  - Corrections  │
└─────────────────┘          └─────────────────┘
```

**Approach 2: React Native + ONNX Runtime**

- Single codebase for iOS/Android
- ONNX Runtime Mobile for inference
- Sync with LlamaFarm for model updates

**Approach 3: Flutter + TFLite**

- Cross-platform with TensorFlow Lite
- Google ML Kit integration
- Good performance on both platforms

**Recommended Libraries for Mobile:**

| Library | Platform | Purpose |
|---------|----------|---------|
| [Ultralytics iOS App](https://github.com/ultralytics/yolo-ios-app) | iOS | YOLO inference |
| [ONNX Runtime Mobile](https://onnxruntime.ai/docs/build/android.html) | Both | General inference |
| [TensorFlow Lite](https://www.tensorflow.org/lite) | Both | Mobile inference |
| [ML Kit](https://developers.google.com/ml-kit) | Both | Pre-built models |
| [Core ML](https://developer.apple.com/documentation/coreml) | iOS | Native Apple ML |

### Edge Deployment Checklist

```yaml
# Edge deployment config

edge_deployment:
  target_platforms:
    - macos_arm64
    - macos_x86_64
    - linux_x86_64
    - linux_arm64
    
  model_optimization:
    quantization: int8  # or fp16
    pruning: false
    distillation: false
    
  export_formats:
    macos: ["coreml", "onnx"]
    linux_nvidia: ["tensorrt", "onnx"]
    linux_cpu: ["onnx", "openvino"]
    
  resource_limits:
    max_model_size_mb: 50
    max_memory_mb: 2048
    target_latency_ms: 100
    
  offline_mode:
    enabled: true
    cache_models: true
    queue_uploads: true
```

---

## API Design

### Server API (Port 14345)

```yaml
# /server/api/routers/vision/

# Streaming (anomaly-style detection)
POST /v1/vision/stream/start:
  description: "Start streaming detection session"
  body:
    source: str  # camera URL, file path, or "upload"
    model: str
    config: StreamingConfig
  response:
    session_id: str

POST /v1/vision/stream/frame:
  description: "Process single frame"
  body:
    session_id: str
    image: str  # base64
  response:
    status: ok|action|review
    detections?: [Detection]
    
DELETE /v1/vision/stream/{session_id}:
  description: "End streaming session"

# Classification
POST /v1/vision/classify:
  description: "Classify image"
  body:
    image: str  # base64
    model: str
    classes?: [str]  # Optional, for zero-shot
  response:
    class: str
    confidence: float
    all_scores: {str: float}

# Detection
POST /v1/vision/detect:
  description: "Detect objects in image"
  body:
    image: str
    model: str
    confidence_threshold?: float
  response:
    detections: [Detection]

# Segmentation
POST /v1/vision/segment:
  description: "Segment image"
  body:
    image: str
    model: str
    prompts?: [Point|Box]  # SAM-style prompts
  response:
    masks: [Mask]
    
# Training
POST /v1/vision/train:
  description: "Train or fine-tune model"
  body:
    model: str
    dataset: str
    config: TrainingConfig
  response:
    job_id: str  # Async training
    
GET /v1/vision/train/{job_id}:
  description: "Get training status"
  
# Models
GET /v1/vision/models:
  description: "List available models"
  
POST /v1/vision/models/export:
  description: "Export model to format"
  body:
    model: str
    format: onnx|coreml|tensorrt|tflite
  response:
    path: str
    
POST /v1/vision/models/import:
  description: "Import model"
  body:
    path: str
    name: str
    type: detection|classification|segmentation
```

### Universal Runtime API (Port 11540)

```yaml
# /runtimes/universal/routers/vision/

# Low-level inference
POST /v1/vision/infer:
  description: "Run inference (internal)"
  body:
    model_id: str
    image: str
    task: detect|classify|segment
    params: dict
  response:
    result: VisionResult

# Model management
GET /v1/vision/backends:
  description: "List available vision backends"
  response:
    backends: [BackendInfo]

POST /v1/vision/load:
  description: "Load model into memory"
  body:
    model_id: str
    device: str
    
POST /v1/vision/unload:
  description: "Unload model from memory"
  body:
    model_id: str
```

---

## API & Router Structure

> **IMPORTANT**: Vision APIs should be modular and NOT added to `universal_runtime_service.py`. 
> Create dedicated service modules to keep the codebase maintainable.

### Design Principles

1. **Mirror Structure**: Server and Universal Runtime should have matching router/type structures
2. **Modular Files**: Split into focused modules (detection, classification, segmentation, training, review)
3. **Shared Types**: Types should be nearly identical between server and runtime
4. **Dedicated Service**: Create `vision_service.py` instead of adding to `universal_runtime_service.py`
5. **No Giant Files**: Keep each router file < 500 LOC

### Directory Structure Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SERVER (Port 14345)                              │
│                        /server/                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  api/routers/vision/                    services/                       │
│  ├── __init__.py                        ├── vision/                     │
│  ├── types.py          ◄─── mirrors ───►│   ├── __init__.py            │
│  ├── detection.py                       │   ├── detection_service.py   │
│  ├── classification.py                  │   ├── classification_service.py│
│  ├── segmentation.py                    │   ├── segmentation_service.py │
│  ├── streaming.py                       │   ├── streaming_service.py   │
│  ├── training.py                        │   ├── training_service.py    │
│  ├── review.py                          │   ├── review_service.py      │
│  └── models.py                          │   └── models_service.py      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP calls via vision_service.py
                              │ (NOT universal_runtime_service.py)
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    UNIVERSAL RUNTIME (Port 11540)                       │
│                    /runtimes/universal/                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  api_types/                             routers/vision/                 │
│  └── vision/           ◄─── mirrors ───►├── __init__.py                │
│      ├── __init__.py                    ├── detection.py               │
│      ├── common.py                      ├── classification.py          │
│      ├── detection.py                   ├── segmentation.py            │
│      ├── classification.py              ├── streaming.py               │
│      ├── segmentation.py                ├── training.py                │
│      ├── streaming.py                   └── models.py                  │
│      ├── training.py                                                   │
│      └── review.py                      services/vision/               │
│                                         ├── __init__.py                │
│  models/                                ├── inference_service.py       │
│  └── vision/                            ├── training_service.py        │
│      ├── __init__.py                    └── model_manager.py           │
│      ├── base.py                                                       │
│      ├── yolo_model.py                                                 │
│      ├── clip_classifier.py                                            │
│      ├── sam_model.py                                                  │
│      └── streaming_detector.py                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Server Router Structure

#### File: `server/api/routers/vision/__init__.py`

```python
"""Vision API routers for LlamaFarm server."""

from fastapi import APIRouter

from .detection import router as detection_router
from .classification import router as classification_router
from .segmentation import router as segmentation_router
from .streaming import router as streaming_router
from .training import router as training_router
from .review import router as review_router
from .models import router as models_router

# Main vision router - combines all sub-routers
vision_router = APIRouter(prefix="/v1/vision", tags=["vision"])

# Mount sub-routers
vision_router.include_router(detection_router)
vision_router.include_router(classification_router)
vision_router.include_router(segmentation_router)
vision_router.include_router(streaming_router)
vision_router.include_router(training_router)
vision_router.include_router(review_router)
vision_router.include_router(models_router)

__all__ = ["vision_router"]
```

#### File: `server/api/routers/vision/types.py`

```python
"""Shared types for vision API endpoints.

These types mirror the Universal Runtime types for consistency.
Import from here in all server-side vision routers.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# =============================================================================
# Common Types (shared across all vision endpoints)
# =============================================================================

class BoundingBox(BaseModel):
    """Bounding box coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    

class Detection(BaseModel):
    """Single object detection result."""
    box: BoundingBox
    class_name: str
    class_id: int
    confidence: float


class Point(BaseModel):
    """Point prompt for segmentation."""
    x: float
    y: float
    label: Literal[0, 1] = 1  # 0=background, 1=foreground


class Mask(BaseModel):
    """Segmentation mask result."""
    mask_base64: str  # Base64-encoded binary mask
    box: BoundingBox
    confidence: float
    area: int


# =============================================================================
# Detection Types
# =============================================================================

class DetectRequest(BaseModel):
    """Object detection request."""
    image: str = Field(..., description="Base64-encoded image")
    model: str = Field(default="yolov8n", description="Model ID")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    classes: list[str] | None = Field(default=None, description="Filter to specific classes")


class DetectResponse(BaseModel):
    """Object detection response."""
    detections: list[Detection]
    model: str
    inference_time_ms: float


# =============================================================================
# Classification Types
# =============================================================================

class ClassifyRequest(BaseModel):
    """Image classification request."""
    image: str = Field(..., description="Base64-encoded image")
    model: str = Field(default="clip-vit-base", description="Model ID")
    classes: list[str] | None = Field(default=None, description="Classes for zero-shot")
    top_k: int = Field(default=5, ge=1, le=100)


class ClassifyResponse(BaseModel):
    """Image classification response."""
    class_name: str
    class_id: int
    confidence: float
    all_scores: dict[str, float]
    model: str
    inference_time_ms: float


# =============================================================================
# Segmentation Types
# =============================================================================

class SegmentRequest(BaseModel):
    """Image segmentation request."""
    image: str = Field(..., description="Base64-encoded image")
    model: str = Field(default="mobilesam", description="Model ID")
    points: list[Point] | None = None
    boxes: list[BoundingBox] | None = None
    multimask_output: bool = False


class SegmentResponse(BaseModel):
    """Image segmentation response."""
    masks: list[Mask]
    model: str
    inference_time_ms: float


# =============================================================================
# Streaming Types
# =============================================================================

class StreamingConfig(BaseModel):
    """Configuration for streaming vision detection."""
    target_fps: float = Field(default=1.0, ge=0.1, le=30.0)
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    escalation_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    action_classes: list[str] | None = None
    cooldown_seconds: float = Field(default=5.0, ge=0.0)


class StreamStartRequest(BaseModel):
    """Start streaming session request."""
    model: str = Field(default="yolov8n")
    config: StreamingConfig = Field(default_factory=StreamingConfig)


class StreamStartResponse(BaseModel):
    """Start streaming session response."""
    session_id: str
    config: StreamingConfig


class StreamFrameRequest(BaseModel):
    """Process single frame request."""
    session_id: str
    image: str  # Base64-encoded


class StreamFrameResponse(BaseModel):
    """Process single frame response."""
    status: Literal["ok", "action", "review"]
    detections: list[Detection] | None = None
    confidence: float | None = None
    image_id: str | None = None  # For review queue


# =============================================================================
# Training Types
# =============================================================================

class TrainingConfig(BaseModel):
    """Training configuration."""
    epochs: int = Field(default=10, ge=1, le=1000)
    batch_size: int = Field(default=16, ge=1, le=256)
    learning_rate: float = Field(default=0.001, ge=0.0)
    # Continual learning
    use_ewc: bool = True
    ewc_lambda: float = Field(default=0.4, ge=0.0)
    use_replay: bool = True
    replay_ratio: float = Field(default=0.3, ge=0.0, le=1.0)


class TrainRequest(BaseModel):
    """Training request."""
    model: str
    dataset: str
    task: Literal["detection", "classification", "segmentation"]
    config: TrainingConfig = Field(default_factory=TrainingConfig)
    base_model: str | None = None  # For fine-tuning


class TrainResponse(BaseModel):
    """Training job response."""
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    progress: float = 0.0
    metrics: dict | None = None


# =============================================================================
# Review Types
# =============================================================================

class ReviewItem(BaseModel):
    """Item in review queue."""
    image_id: str
    image_url: str  # URL to fetch image
    thumbnail_url: str
    timestamp: datetime
    prediction: Detection | None
    confidence: float
    model: str
    source: str
    status: Literal["pending", "approved", "rejected", "corrected"]


class ReviewDecision(BaseModel):
    """Human review decision."""
    image_id: str
    decision: Literal["correct", "wrong", "adjusted"]
    corrections: list[Detection] | None = None  # If adjusted


class ReviewListResponse(BaseModel):
    """List of review items."""
    items: list[ReviewItem]
    total: int
    pending: int


# =============================================================================
# Model Management Types
# =============================================================================

class ModelInfo(BaseModel):
    """Information about a vision model."""
    model_id: str
    name: str
    task: Literal["detection", "classification", "segmentation"]
    version: str
    size_mb: float
    loaded: bool
    device: str | None


class ModelExportRequest(BaseModel):
    """Model export request."""
    model_id: str
    format: Literal["onnx", "coreml", "tensorrt", "tflite", "openvino"]
    quantization: Literal["fp32", "fp16", "int8"] = "fp16"


class ModelExportResponse(BaseModel):
    """Model export response."""
    export_path: str
    format: str
    size_mb: float


class ModelImportRequest(BaseModel):
    """Model import request."""
    path: str
    name: str
    task: Literal["detection", "classification", "segmentation"]
    metadata: dict | None = None
```

#### File: `server/api/routers/vision/detection.py`

```python
"""Detection router - object detection endpoints."""

from fastapi import APIRouter, HTTPException

from .types import DetectRequest, DetectResponse
from services.vision.detection_service import VisionDetectionService

router = APIRouter(tags=["vision-detection"])


@router.post("/detect", response_model=DetectResponse)
async def detect_objects(request: DetectRequest) -> DetectResponse:
    """Detect objects in an image.
    
    Uses YOLO or similar models to detect and localize objects
    with bounding boxes and confidence scores.
    """
    return await VisionDetectionService.detect(request)
```

#### File: `server/services/vision/__init__.py`

```python
"""Vision services for LlamaFarm server.

These services handle communication with the Universal Runtime
for all vision-related operations. Each service is dedicated
to a specific vision capability.

NOTE: Do NOT add vision methods to universal_runtime_service.py.
      Keep vision services modular and separate.
"""

from .detection_service import VisionDetectionService
from .classification_service import VisionClassificationService
from .segmentation_service import VisionSegmentationService
from .streaming_service import VisionStreamingService
from .training_service import VisionTrainingService
from .review_service import VisionReviewService
from .models_service import VisionModelsService

__all__ = [
    "VisionDetectionService",
    "VisionClassificationService",
    "VisionSegmentationService",
    "VisionStreamingService",
    "VisionTrainingService",
    "VisionReviewService",
    "VisionModelsService",
]
```

#### File: `server/services/vision/detection_service.py`

```python
"""Detection service - handles object detection requests."""

import logging
from typing import Any

import httpx

from core.settings import settings
from api.routers.vision.types import DetectRequest, DetectResponse

logger = logging.getLogger(__name__)


class VisionDetectionService:
    """Service for object detection operations.
    
    Communicates with Universal Runtime's vision detection endpoint.
    """
    
    # Persistent client (initialized on first use)
    _client: httpx.AsyncClient | None = None
    
    @classmethod
    async def _get_client(cls) -> httpx.AsyncClient:
        """Get or create HTTP client for runtime communication."""
        if cls._client is None or cls._client.is_closed:
            base_url = f"http://{settings.universal_host}:{settings.universal_port}"
            cls._client = httpx.AsyncClient(
                base_url=base_url,
                timeout=httpx.Timeout(connect=10.0, read=60.0, write=30.0),
            )
        return cls._client
    
    @classmethod
    async def detect(cls, request: DetectRequest) -> DetectResponse:
        """Run object detection on an image.
        
        Args:
            request: Detection request with image and parameters
            
        Returns:
            DetectResponse with list of detections
        """
        client = await cls._get_client()
        
        # Call Universal Runtime
        response = await client.post(
            "/v1/vision/detect",
            json=request.model_dump(),
        )
        response.raise_for_status()
        
        return DetectResponse(**response.json())
```

### Universal Runtime Router Structure

#### File: `runtimes/universal/api_types/vision/__init__.py`

```python
"""Vision API types for Universal Runtime.

These types should mirror the server types for consistency.
"""

from .common import BoundingBox, Point, Mask
from .detection import (
    DetectRequest,
    DetectResponse,
    Detection,
)
from .classification import (
    ClassifyRequest,
    ClassifyResponse,
)
from .segmentation import (
    SegmentRequest,
    SegmentResponse,
)
from .streaming import (
    StreamingConfig,
    StreamStartRequest,
    StreamStartResponse,
    StreamFrameRequest,
    StreamFrameResponse,
)
from .training import (
    TrainingConfig,
    TrainRequest,
    TrainResponse,
    TrainStatusResponse,
)
from .models import (
    ModelInfo,
    ModelLoadRequest,
    ModelUnloadRequest,
    BackendInfo,
)

__all__ = [
    # Common
    "BoundingBox",
    "Point",
    "Mask",
    # Detection
    "DetectRequest",
    "DetectResponse",
    "Detection",
    # Classification
    "ClassifyRequest",
    "ClassifyResponse",
    # Segmentation
    "SegmentRequest",
    "SegmentResponse",
    # Streaming
    "StreamingConfig",
    "StreamStartRequest",
    "StreamStartResponse",
    "StreamFrameRequest",
    "StreamFrameResponse",
    # Training
    "TrainingConfig",
    "TrainRequest",
    "TrainResponse",
    "TrainStatusResponse",
    # Models
    "ModelInfo",
    "ModelLoadRequest",
    "ModelUnloadRequest",
    "BackendInfo",
]
```

#### File: `runtimes/universal/routers/vision/__init__.py`

```python
"""Vision routers for Universal Runtime."""

from fastapi import APIRouter

from .detection import router as detection_router
from .classification import router as classification_router
from .segmentation import router as segmentation_router
from .streaming import router as streaming_router
from .training import router as training_router
from .models import router as models_router

# Main vision router
router = APIRouter(prefix="/v1/vision", tags=["vision"])

# Mount sub-routers
router.include_router(detection_router)
router.include_router(classification_router)
router.include_router(segmentation_router)
router.include_router(streaming_router)
router.include_router(training_router)
router.include_router(models_router)

__all__ = ["router"]
```

#### File: `runtimes/universal/routers/vision/detection.py`

```python
"""Detection router for Universal Runtime."""

import logging
from typing import Callable, Coroutine, Any

from fastapi import APIRouter, HTTPException

from api_types.vision import DetectRequest, DetectResponse
from services.error_handler import handle_endpoint_errors

logger = logging.getLogger(__name__)

router = APIRouter(tags=["vision-detection"])

# Dependency injection for model loader
_load_detection_model_fn: Callable[..., Coroutine[Any, Any, Any]] | None = None


def set_detection_loader(
    load_fn: Callable[..., Coroutine[Any, Any, Any]] | None
) -> None:
    """Set the detection model loader function.
    
    Called during app initialization to inject the model loading
    dependency from the main server.
    """
    global _load_detection_model_fn
    _load_detection_model_fn = load_fn


def _get_loader():
    """Get detection loader or raise if not initialized."""
    if _load_detection_model_fn is None:
        raise HTTPException(
            status_code=500,
            detail="Detection model loader not initialized",
        )
    return _load_detection_model_fn


@router.post("/detect", response_model=DetectResponse)
@handle_endpoint_errors("vision_detect")
async def detect_objects(request: DetectRequest) -> DetectResponse:
    """Detect objects in an image.
    
    Supports YOLO models (v8, v11) for real-time object detection.
    Returns bounding boxes with class labels and confidence scores.
    
    Args:
        request: Detection request with base64 image
        
    Returns:
        List of detections with bounding boxes
    """
    import time
    start_time = time.perf_counter()
    
    # Load model
    loader = _get_loader()
    model = await loader(request.model)
    
    # Run inference
    result = await model.detect(
        image=request.image,
        confidence_threshold=request.confidence_threshold,
        classes=request.classes,
    )
    
    inference_time_ms = (time.perf_counter() - start_time) * 1000
    
    return DetectResponse(
        detections=result.detections,
        model=request.model,
        inference_time_ms=inference_time_ms,
    )
```

### API Endpoint Mapping (Server ↔ Runtime)

| Server Endpoint | Runtime Endpoint | Description |
|-----------------|------------------|-------------|
| `POST /v1/vision/detect` | `POST /v1/vision/detect` | Object detection |
| `POST /v1/vision/classify` | `POST /v1/vision/classify` | Image classification |
| `POST /v1/vision/segment` | `POST /v1/vision/segment` | Image segmentation |
| `POST /v1/vision/stream/start` | `POST /v1/vision/stream/start` | Start streaming session |
| `POST /v1/vision/stream/frame` | `POST /v1/vision/stream/frame` | Process frame |
| `DELETE /v1/vision/stream/{id}` | `DELETE /v1/vision/stream/{id}` | End session |
| `POST /v1/vision/train` | `POST /v1/vision/train` | Start training job |
| `GET /v1/vision/train/{id}` | `GET /v1/vision/train/{id}` | Get training status |
| `GET /v1/vision/models` | `GET /v1/vision/models` | List models |
| `POST /v1/vision/models/load` | `POST /v1/vision/models/load` | Load model to memory |
| `POST /v1/vision/models/unload` | `POST /v1/vision/models/unload` | Unload model |
| `POST /v1/vision/models/export` | `POST /v1/vision/models/export` | Export model |
| `POST /v1/vision/models/import` | `POST /v1/vision/models/import` | Import model |
| `GET /v1/vision/review` | N/A (server-only) | List review queue |
| `POST /v1/vision/review` | N/A (server-only) | Submit review |
| `GET /v1/vision/backends` | `GET /v1/vision/backends` | List vision backends |

### Server-Only Endpoints

These endpoints exist only on the server (not mirrored in runtime):

```python
# server/api/routers/vision/review.py

@router.get("/review", response_model=ReviewListResponse)
async def list_review_queue(
    status: str = "pending",
    limit: int = 50,
    offset: int = 0,
) -> ReviewListResponse:
    """List items in the review queue.
    
    Server-only: Review queue is managed by the server,
    not the runtime.
    """
    return await VisionReviewService.list_queue(status, limit, offset)


@router.post("/review", response_model=dict)
async def submit_review(decision: ReviewDecision) -> dict:
    """Submit a human review decision.
    
    Server-only: Reviews are stored and managed by the server.
    Corrections are forwarded to runtime for replay buffer.
    """
    return await VisionReviewService.submit_review(decision)
```

### File Size Guidelines

| File | Max LOC | Purpose |
|------|---------|---------|
| `types.py` | 300 | Request/Response models |
| `detection.py` | 200 | Detection endpoints |
| `classification.py` | 200 | Classification endpoints |
| `segmentation.py` | 200 | Segmentation endpoints |
| `streaming.py` | 300 | Streaming session management |
| `training.py` | 300 | Training job management |
| `review.py` | 200 | Human review endpoints |
| `models.py` | 250 | Model CRUD operations |

### Integration with Existing Code

**DO NOT modify these files for vision:**
- `server/services/universal_runtime_service.py` - Keep vision separate
- `runtimes/universal/server.py` - Only add router imports

**DO modify these files:**
```python
# server/api/main.py - Add vision router
from api.routers.vision import vision_router
app.include_router(vision_router)

# runtimes/universal/server.py - Add vision router import
from routers.vision import router as vision_router
app.include_router(vision_router)

# runtimes/universal/routers/__init__.py - Add export
from .vision import router as vision_router
__all__ = [..., "vision_router"]
```

---

## Image RAG Integration

> **Goal**: Enable retrieval-augmented generation with images alongside text,
> integrating into the existing LlamaFarm RAG framework.

### Overview

Image RAG extends LlamaFarm's existing RAG capabilities to support:
- **Image Embedding**: Generate vector embeddings for images using CLIP
- **Multimodal Search**: Find images by text query or by image similarity
- **Metadata Indexing**: Track image annotations, detections, and labels
- **Integration with Vision Pipeline**: Feed detected objects/classifications into RAG

### How Existing RAG Works in LlamaFarm

Before designing Image RAG, let's understand the current architecture:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Current Text RAG Architecture                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Document Upload                                                        │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────┐                                                   │
│  │  RAG Worker     │  (Celery tasks in /rag/)                          │
│  │  - Parsers      │  PDFParser, MarkdownParser, UniversalParser       │
│  │  - Extractors   │  KeywordExtractor, EntityExtractor, etc.          │
│  │  - Chunking     │  SemChunk for semantic boundaries                 │
│  └────────┬────────┘                                                   │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                   │
│  │  Embedder       │  UniversalEmbedder (calls Universal Runtime)      │
│  │  (text → vec)   │  sentence-transformers/all-MiniLM-L6-v2           │
│  └────────┬────────┘                                                   │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                   │
│  │  Vector Store   │  ChromaDB, FAISS, Qdrant, Pinecone                │
│  │  (ChromaStore)  │  ~/.llamafarm/data/projects/{ns}/{proj}/chroma/   │
│  └────────┬────────┘                                                   │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                   │
│  │  Retriever      │  BasicSimilarityStrategy, CrossEncoderReranked    │
│  │                 │  HybridUniversalStrategy                          │
│  └─────────────────┘                                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Files:**
- `rag/core/factories.py` - Creates embedders, stores, retrievers
- `rag/components/stores/chroma_store/` - ChromaDB implementation
- `rag/components/embedders/universal_embedder/` - Calls Universal Runtime
- `server/services/database_service.py` - Database CRUD operations
- `server/services/rag_service.py` - RAG query handling
- `server/services/rag_health_cache.py` - Health monitoring

### Image RAG Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Image RAG Architecture (NEW)                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Image Input                                                            │
│  (upload, stream, vision detection result)                              │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────┐                                                   │
│  │ Image Processor │  (NEW: /rag/components/parsers/image_parser/)     │
│  │  - Resize/norm  │                                                   │
│  │  - OCR text     │  Extract text from images                         │
│  │  - Metadata     │  EXIF, dimensions, format                         │
│  └────────┬────────┘                                                   │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────┐                                                   │
│  │ CLIP Embedder   │  (NEW: /rag/components/embedders/clip_embedder/)  │
│  │ (image → vec)   │  openai/clip-vit-base-patch32 (512 dim)           │
│  │                 │  OR via Universal Runtime                         │
│  └────────┬────────┘                                                   │
│           │                                                             │
│           ├───────────────────────────┐                                 │
│           │                           │                                 │
│           ▼                           ▼                                 │
│  ┌─────────────────┐        ┌─────────────────┐                        │
│  │  Vector Store   │        │  Metadata Store │  (NEW)                 │
│  │  (ChromaDB)     │        │  (SQLite)       │                        │
│  │  - image_embed  │        │  - image_id     │                        │
│  │  - collection:  │        │  - file_path    │                        │
│  │    "images"     │        │  - detections   │                        │
│  └────────┬────────┘        │  - labels       │                        │
│           │                 │  - confidence   │                        │
│           │                 │  - reviewed     │                        │
│           │                 │  - timestamps   │                        │
│           │                 └────────┬────────┘                        │
│           │                          │                                  │
│           └──────────┬───────────────┘                                  │
│                      │                                                  │
│                      ▼                                                  │
│             ┌─────────────────┐                                        │
│             │ Image Retriever │  (NEW)                                 │
│             │ - text → images │  "Find photos of cats"                 │
│             │ - image → images│  Similar image search                  │
│             │ - hybrid        │  Combined with metadata filters        │
│             └─────────────────┘                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Database Options

#### Option 1: Extend ChromaDB (Recommended)

ChromaDB supports multimodal embeddings natively:

```python
# Using ChromaDB's multimodal capability
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

client = chromadb.PersistentClient(path="./chroma_db")

# Create image collection with CLIP embeddings
embedding_function = OpenCLIPEmbeddingFunction()
image_collection = client.get_or_create_collection(
    name="images",
    embedding_function=embedding_function,
    metadata={"hnsw:space": "cosine"}
)

# Add images
image_collection.add(
    ids=["img1", "img2"],
    images=[image_bytes_1, image_bytes_2],  # Raw image bytes
    metadatas=[
        {"source": "camera:front", "labels": ["person", "car"]},
        {"source": "upload", "labels": ["dog"]}
    ]
)

# Query by text
results = image_collection.query(
    query_texts=["a photo of a dog"],
    n_results=5
)

# Query by image
results = image_collection.query(
    query_images=[query_image_bytes],
    n_results=5
)
```

**Pros:**
- Already used in LlamaFarm for text RAG
- Native multimodal support
- No additional database to manage
- Cosine similarity built-in

**Cons:**
- Limited metadata querying (no complex SQL)
- Need separate collection for images

#### Option 2: SQLite for Metadata + ChromaDB for Vectors

Use SQLite for rich metadata and ChromaDB only for vector search:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Hybrid Storage Architecture                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  SQLite (Metadata)                      ChromaDB (Vectors)              │
│  ~/.llamafarm/vision/images.db          ~/.llamafarm/vision/chroma/     │
│                                                                         │
│  ┌─────────────────────────────┐       ┌─────────────────────────────┐ │
│  │ images                      │       │ collection: "image_vectors" │ │
│  │ ─────────────────────────── │       │ ───────────────────────────│ │
│  │ id          TEXT PRIMARY    │◄─────►│ id: image_id               │ │
│  │ file_path   TEXT            │       │ embedding: [512 floats]    │ │
│  │ thumbnail   TEXT            │       │ metadata: {source, ...}    │ │
│  │ created_at  DATETIME        │       └─────────────────────────────┘ │
│  │ source      TEXT            │                                       │
│  │ width       INTEGER         │                                       │
│  │ height      INTEGER         │                                       │
│  │ format      TEXT            │                                       │
│  │ size_bytes  INTEGER         │                                       │
│  │ hash        TEXT UNIQUE     │  (for deduplication)                  │
│  │ ocr_text    TEXT            │                                       │
│  │ reviewed    BOOLEAN         │                                       │
│  │ reviewed_at DATETIME        │                                       │
│  │ reviewed_by TEXT            │                                       │
│  └─────────────────────────────┘                                       │
│                                                                         │
│  ┌─────────────────────────────┐                                       │
│  │ detections                  │  (Vision detection results)           │
│  │ ─────────────────────────── │                                       │
│  │ id          INTEGER PRIMARY │                                       │
│  │ image_id    TEXT FOREIGN    │                                       │
│  │ class_name  TEXT            │                                       │
│  │ confidence  REAL            │                                       │
│  │ x1, y1      REAL            │                                       │
│  │ x2, y2      REAL            │                                       │
│  │ model       TEXT            │                                       │
│  │ verified    BOOLEAN         │                                       │
│  └─────────────────────────────┘                                       │
│                                                                         │
│  ┌─────────────────────────────┐                                       │
│  │ labels                      │  (Classification results)             │
│  │ ─────────────────────────── │                                       │
│  │ id          INTEGER PRIMARY │                                       │
│  │ image_id    TEXT FOREIGN    │                                       │
│  │ label       TEXT            │                                       │
│  │ confidence  REAL            │                                       │
│  │ source      TEXT            │  (model, human)                       │
│  └─────────────────────────────┘                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Pros:**
- Rich SQL queries on metadata
- Complex filtering (date ranges, labels, confidence)
- Audit trail for reviews
- Fast metadata-only queries

**Cons:**
- Two databases to manage
- Sync complexity

### Recommended Approach: Hybrid with SQLite

Given LlamaFarm's needs for:
- Tracking review state
- Storing detection results
- Complex filtering
- Audit trails

**Use SQLite for metadata + ChromaDB for vectors:**

#### File: `runtimes/universal/services/vision/image_store.py`

```python
"""Image metadata storage using SQLite."""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Any
from dataclasses import dataclass
import hashlib

from utils.safe_home import get_data_dir

# Database location
VISION_DB_PATH = get_data_dir() / "vision" / "images.db"


@dataclass
class ImageRecord:
    """Image metadata record."""
    id: str
    file_path: str
    thumbnail_path: str | None
    created_at: datetime
    source: str
    width: int
    height: int
    format: str
    size_bytes: int
    content_hash: str
    ocr_text: str | None
    reviewed: bool
    reviewed_at: datetime | None
    reviewed_by: str | None


class ImageMetadataStore:
    """SQLite-based image metadata storage."""
    
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or VISION_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS images (
                    id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    thumbnail_path TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    source TEXT,
                    width INTEGER,
                    height INTEGER,
                    format TEXT,
                    size_bytes INTEGER,
                    content_hash TEXT UNIQUE,
                    ocr_text TEXT,
                    reviewed BOOLEAN DEFAULT FALSE,
                    reviewed_at DATETIME,
                    reviewed_by TEXT
                );
                
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id TEXT NOT NULL,
                    class_name TEXT NOT NULL,
                    confidence REAL,
                    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
                    model TEXT,
                    verified BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (image_id) REFERENCES images(id)
                );
                
                CREATE TABLE IF NOT EXISTS labels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    confidence REAL,
                    source TEXT,  -- 'model' or 'human'
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (image_id) REFERENCES images(id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_images_source ON images(source);
                CREATE INDEX IF NOT EXISTS idx_images_reviewed ON images(reviewed);
                CREATE INDEX IF NOT EXISTS idx_detections_image ON detections(image_id);
                CREATE INDEX IF NOT EXISTS idx_detections_class ON detections(class_name);
                CREATE INDEX IF NOT EXISTS idx_labels_image ON labels(image_id);
                CREATE INDEX IF NOT EXISTS idx_labels_label ON labels(label);
            """)
    
    def add_image(self, record: ImageRecord) -> str:
        """Add image metadata record."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO images 
                (id, file_path, thumbnail_path, created_at, source, 
                 width, height, format, size_bytes, content_hash, ocr_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.id, record.file_path, record.thumbnail_path,
                record.created_at, record.source, record.width, record.height,
                record.format, record.size_bytes, record.content_hash, record.ocr_text
            ))
        return record.id
    
    def add_detection(
        self, 
        image_id: str, 
        class_name: str, 
        confidence: float,
        box: tuple[float, float, float, float],
        model: str
    ) -> int:
        """Add detection result for an image."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO detections (image_id, class_name, confidence, x1, y1, x2, y2, model)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (image_id, class_name, confidence, *box, model))
            return cursor.lastrowid
    
    def get_pending_review(self, limit: int = 50) -> list[ImageRecord]:
        """Get images pending human review."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM images 
                WHERE reviewed = FALSE 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (limit,)).fetchall()
            return [ImageRecord(**dict(row)) for row in rows]
    
    def mark_reviewed(self, image_id: str, reviewed_by: str) -> None:
        """Mark image as reviewed."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE images 
                SET reviewed = TRUE, reviewed_at = CURRENT_TIMESTAMP, reviewed_by = ?
                WHERE id = ?
            """, (reviewed_by, image_id))
```

### CLIP Embedder for Images

#### File: `rag/components/embedders/clip_embedder/clip_embedder.py`

```python
"""CLIP-based image embedder for multimodal RAG."""

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from core.base import Embedder
from core.logging import RAGStructLogger
from core.settings import settings

logger = RAGStructLogger("rag.components.embedders.clip_embedder")


class CLIPEmbedder(Embedder):
    """Image embedder using CLIP via Universal Runtime."""
    
    def __init__(
        self,
        name: str = "CLIPEmbedder",
        config: dict[str, Any] | None = None,
        project_dir: Path | None = None,
    ):
        super().__init__(name, config, project_dir)
        config = config or {}
        
        self.model = config.get("model", "openai/clip-vit-base-patch32")
        self.api_base = config.get(
            "base_url", 
            f"http://127.0.0.1:{settings.UNIVERSAL_PORT}/v1"
        )
        self.batch_size = config.get("batch_size", 8)
        self.timeout = config.get("timeout", 60)
    
    def embed_images(self, images: list[bytes]) -> list[list[float]]:
        """Generate embeddings for a batch of images.
        
        Args:
            images: List of image bytes (JPEG/PNG)
            
        Returns:
            List of embedding vectors (512 dimensions for CLIP)
        """
        embeddings = []
        
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            batch_b64 = [base64.b64encode(img).decode() for img in batch]
            
            response = requests.post(
                f"{self.api_base}/vision/embed",
                json={
                    "model": self.model,
                    "images": batch_b64,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            
            result = response.json()
            embeddings.extend(result["embeddings"])
        
        return embeddings
    
    def embed_text(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for text (for cross-modal search).
        
        CLIP embeds text and images into the same vector space,
        enabling text-to-image search.
        """
        response = requests.post(
            f"{self.api_base}/vision/embed",
            json={
                "model": self.model,
                "texts": texts,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["embeddings"]
```

### Integration with Existing RAG Framework

#### Update: `rag/core/factories.py`

```python
# Add to imports
from components.embedders.clip_embedder.clip_embedder import CLIPEmbedder

# Add to EMBEDDER_REGISTRY
EMBEDDER_REGISTRY = {
    # ... existing embedders
    "CLIPEmbedder": CLIPEmbedder,
}
```

#### Update: `config/schema.yaml` (Add Image Database Type)

```yaml
# Add to database types
DatabaseType:
  type: string
  enum:
    - ChromaStore
    - FAISSStore
    - QdrantStore
    - PineconeStore
    - ImageStore  # NEW

# Add ImageStore config
ImageStoreConfig:
  type: object
  properties:
    collection_name:
      type: string
      default: "images"
    embedding_model:
      type: string
      default: "openai/clip-vit-base-patch32"
    metadata_db:
      type: string
      description: "Path to SQLite metadata database"
```

### Health Checks for Image RAG

Following the pattern in `rag_health_cache.py`:

#### File: `server/services/vision/image_rag_health.py`

```python
"""Health monitoring for Image RAG service."""

import logging
import time
from datetime import datetime
from typing import Any

from services.vision.image_store import ImageMetadataStore, VISION_DB_PATH

logger = logging.getLogger(__name__)


class ImageRAGHealthChecker:
    """Health checker for Image RAG components."""
    
    def __init__(self):
        self._last_check: datetime | None = None
        self._last_status: dict[str, Any] | None = None
    
    def check_health(self) -> dict[str, Any]:
        """Check health of all Image RAG components."""
        start_time = time.time()
        
        health = {
            "status": "healthy",
            "components": {},
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        # Check SQLite metadata store
        try:
            store = ImageMetadataStore()
            # Simple query to verify DB is accessible
            with sqlite3.connect(store.db_path) as conn:
                conn.execute("SELECT COUNT(*) FROM images").fetchone()
            health["components"]["metadata_store"] = {
                "status": "healthy",
                "path": str(store.db_path),
            }
        except Exception as e:
            health["status"] = "unhealthy"
            health["components"]["metadata_store"] = {
                "status": "unhealthy",
                "error": str(e),
            }
        
        # Check ChromaDB image collection
        try:
            import chromadb
            client = chromadb.PersistentClient(
                path=str(VISION_DB_PATH.parent / "chroma")
            )
            collections = client.list_collections()
            image_collection = next(
                (c for c in collections if c.name == "images"), None
            )
            health["components"]["vector_store"] = {
                "status": "healthy",
                "collection_exists": image_collection is not None,
                "document_count": image_collection.count() if image_collection else 0,
            }
        except Exception as e:
            health["status"] = "degraded"
            health["components"]["vector_store"] = {
                "status": "unhealthy",
                "error": str(e),
            }
        
        # Check CLIP embedder availability
        try:
            import requests
            response = requests.get(
                "http://127.0.0.1:11540/health",
                timeout=5,
            )
            runtime_health = response.json()
            health["components"]["clip_embedder"] = {
                "status": "healthy" if response.ok else "unhealthy",
                "runtime_status": runtime_health.get("status"),
            }
        except Exception as e:
            health["status"] = "degraded"
            health["components"]["clip_embedder"] = {
                "status": "unhealthy",
                "error": str(e),
            }
        
        health["check_duration_ms"] = (time.time() - start_time) * 1000
        
        self._last_check = datetime.utcnow()
        self._last_status = health
        
        return health
```

### API Endpoints for Image RAG

#### File: `server/api/routers/vision/image_rag.py`

```python
"""Image RAG endpoints."""

from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from pydantic import BaseModel

router = APIRouter(tags=["vision-rag"])


class ImageSearchRequest(BaseModel):
    """Search for images by text or image."""
    query_text: str | None = None
    query_image: str | None = None  # base64
    filters: dict | None = None  # {"source": "camera:front", "labels": ["person"]}
    top_k: int = 10


class ImageSearchResult(BaseModel):
    """Single search result."""
    image_id: str
    score: float
    thumbnail_url: str
    metadata: dict


class ImageSearchResponse(BaseModel):
    """Search response."""
    results: list[ImageSearchResult]
    total: int
    query_type: str  # "text" or "image"


@router.post("/rag/search", response_model=ImageSearchResponse)
async def search_images(request: ImageSearchRequest):
    """Search images by text query or similar image.
    
    Examples:
        - Text: "photos of dogs playing"
        - Image: Upload reference image to find similar
        - Filters: {"labels": ["person"], "source": "camera:front"}
    """
    from services.vision.image_rag_service import ImageRAGService
    return await ImageRAGService.search(request)


@router.post("/rag/index")
async def index_image(
    file: UploadFile = File(...),
    source: str = Query(default="upload"),
    auto_detect: bool = Query(default=True, description="Run object detection"),
    auto_classify: bool = Query(default=True, description="Run classification"),
):
    """Index an image for RAG retrieval.
    
    This will:
    1. Generate CLIP embedding
    2. Store in vector database
    3. Optionally run detection/classification
    4. Store metadata in SQLite
    """
    from services.vision.image_rag_service import ImageRAGService
    return await ImageRAGService.index_image(
        file=file,
        source=source,
        auto_detect=auto_detect,
        auto_classify=auto_classify,
    )


@router.get("/rag/stats")
async def get_image_rag_stats():
    """Get Image RAG statistics."""
    from services.vision.image_rag_service import ImageRAGService
    return await ImageRAGService.get_stats()


@router.get("/rag/health")
async def get_image_rag_health():
    """Get Image RAG health status."""
    from services.vision.image_rag_health import ImageRAGHealthChecker
    checker = ImageRAGHealthChecker()
    return checker.check_health()
```

### Configuration in llamafarm.yaml

```yaml
# Example project configuration with Image RAG

version: v1
name: security-monitor
namespace: home

# Vision configuration
vision:
  # Streaming detection config
  streaming:
    model: yolov8n
    confidence_threshold: 0.7
    action_classes: ["person", "car", "package"]
    
  # Image RAG config
  image_rag:
    enabled: true
    embedding_model: openai/clip-vit-base-patch32
    auto_index_detections: true  # Index frames with detections
    
    # Retention policy
    retention:
      default_hours: 24
      with_detections_days: 7
      reviewed_days: 30
    
    # Vector store
    vector_store:
      type: ChromaStore
      config:
        collection_name: security_images
        
# Standard text RAG (can coexist)
rag:
  databases:
    - name: documents
      type: ChromaStore
```

### Integration Flow: Vision → Image RAG

```
Vision Detection                    Image RAG
     │                                 │
     │ Detection with conf > 0.7       │
     │ ────────────────────────────►   │
     │                                 │
     │                    ┌────────────┴────────────┐
     │                    │  1. Generate CLIP embed │
     │                    │  2. Store in ChromaDB   │
     │                    │  3. Save metadata       │
     │                    │  4. Create thumbnail    │
     │                    └────────────┬────────────┘
     │                                 │
     │ Low confidence (review queue)   │
     │ ────────────────────────────►   │
     │                                 │
     │                    ┌────────────┴────────────┐
     │                    │  Store with             │
     │                    │  reviewed=False         │
     │                    │  for human review       │
     │                    └────────────┬────────────┘
     │                                 │
     │                                 ▼
     │                         Image RAG Search
     │                         "Find all people
     │                          near front door"
```

### Files to Create for Image RAG

```
rag/
├── components/
│   ├── embedders/
│   │   └── clip_embedder/
│   │       ├── __init__.py
│   │       └── clip_embedder.py      # CLIP embedding via Universal Runtime
│   └── parsers/
│       └── image_parser/
│           ├── __init__.py
│           └── image_parser.py       # Extract metadata, OCR, thumbnails

runtimes/universal/
├── routers/
│   └── vision/
│       └── embedding.py              # /v1/vision/embed endpoint
├── services/
│   └── vision/
│       └── clip_service.py           # CLIP model management
└── models/
    └── vision/
        └── clip_model.py             # CLIP model wrapper

server/
├── api/routers/
│   └── vision/
│       └── image_rag.py              # Image RAG API endpoints
└── services/
    └── vision/
        ├── image_store.py            # SQLite metadata store
        ├── image_rag_service.py      # Image RAG business logic
        └── image_rag_health.py       # Health monitoring
```

---

## Implementation Phases

### Phase 1: Foundation (2-3 weeks)

**Deliverables:**
- [ ] Vision model base classes
- [ ] YOLO model wrapper (detection)
- [ ] Basic inference API
- [ ] Model loading/unloading
- [ ] Tests for core functionality

**Files to Create:**
```
runtimes/universal/
├── models/
│   ├── vision_model.py      # Base classes
│   └── yolo_model.py        # YOLO wrapper
├── routers/
│   └── vision/
│       ├── __init__.py
│       ├── router.py        # API endpoints
│       └── types.py         # Request/Response types
└── tests/
    └── test_vision_model.py

server/
├── api/routers/
│   └── vision/
│       ├── __init__.py
│       ├── router.py        # Server-side API
│       └── types.py
└── services/
    └── vision_service.py    # Business logic
```

### Phase 2: Streaming & Classification (2 weeks)

**Deliverables:**
- [ ] Streaming vision detector
- [ ] CLIP-based classifier
- [ ] Few-shot training for classification
- [ ] Confidence-based routing

**Files to Create:**
```
runtimes/universal/models/
├── streaming_vision.py
└── clip_classifier.py

server/services/
└── streaming_vision_service.py
```

### Phase 3: Storage & Retention (1-2 weeks)

**Deliverables:**
- [ ] Image storage system
- [ ] Retention policy engine
- [ ] Review queue management
- [ ] Replay buffer implementation

**Files to Create:**
```
runtimes/universal/
├── storage/
│   ├── __init__.py
│   ├── image_store.py
│   ├── retention_policy.py
│   └── replay_buffer.py
└── utils/
    └── image_compression.py
```

### Phase 4: Training Pipeline (2-3 weeks)

**Deliverables:**
- [ ] Incremental training system
- [ ] EWC implementation
- [ ] Experience replay
- [ ] Training job management

**Files to Create:**
```
runtimes/universal/
├── training/
│   ├── __init__.py
│   ├── incremental_trainer.py
│   ├── ewc.py
│   └── replay_sampler.py
└── routers/
    └── vision/
        └── training_router.py
```

### Phase 5: Multi-Model & Review UI (2 weeks)

**Deliverables:**
- [ ] Model cascade system
- [ ] Review UI in Designer
- [ ] Human correction workflow
- [ ] Batch review functionality

**Files to Create:**
```
server/services/
├── cascade_service.py
└── review_service.py

designer/
└── src/components/
    └── VisionReview/
        ├── ReviewQueue.tsx
        ├── ImageReviewer.tsx
        └── BoundingBoxEditor.tsx
```

### Phase 6: Export & Polish (1-2 weeks)

**Deliverables:**
- [ ] Model export (ONNX, CoreML, TensorRT)
- [ ] Documentation
- [ ] Integration tests
- [ ] Performance optimization

---

## Testing Strategy

### Unit Tests

```python
# tests/test_yolo_model.py

import pytest
from models.yolo_model import YOLOModel

@pytest.fixture
async def yolo_model():
    model = YOLOModel("yolov8n")
    await model.load()
    yield model
    await model.unload()

async def test_detection(yolo_model):
    image = load_test_image("person.jpg")
    result = await yolo_model.infer(image)
    
    assert result.confidence > 0.5
    assert len(result.boxes) > 0
    assert "person" in [b["class"] for b in result.boxes]

async def test_empty_image(yolo_model):
    image = load_test_image("empty.jpg")
    result = await yolo_model.infer(image)
    
    assert len(result.boxes) == 0
```

### Integration Tests

```python
# tests/integration/test_vision_api.py

async def test_detection_endpoint(client):
    response = await client.post(
        "/v1/vision/detect",
        json={
            "image": base64_encode(load_test_image("dog.jpg")),
            "model": "yolov8n",
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert len(data["detections"]) > 0
```

### Performance Tests

```python
# tests/performance/test_inference_speed.py

async def test_inference_latency(yolo_model):
    image = load_test_image("standard.jpg")
    
    times = []
    for _ in range(100):
        start = time.perf_counter()
        await yolo_model.infer(image)
        times.append(time.perf_counter() - start)
    
    avg_ms = statistics.mean(times) * 1000
    p99_ms = statistics.quantiles(times, n=100)[98] * 1000
    
    assert avg_ms < 50, f"Average latency {avg_ms}ms exceeds 50ms"
    assert p99_ms < 100, f"P99 latency {p99_ms}ms exceeds 100ms"
```

---

## Future Considerations

### Not In Scope (This Phase)

1. **Video file processing** - Focus on streaming/single images
2. **3D vision** - Depth estimation, point clouds
3. **Face recognition** - Privacy concerns, separate effort
4. **License plate recognition** - Specialized model needed
5. **Federated learning** - Complex, future consideration

### Future Enhancements

1. **Model Distillation** - Create smaller models from larger ones
2. **AutoML** - Automatic model selection and hyperparameter tuning
3. **Federated Learning** - Privacy-preserving distributed training
4. **Video Understanding** - Action recognition, temporal modeling
5. **Multi-Camera Fusion** - Combine views from multiple cameras

### Research Topics to Monitor

- YOLO version updates (v12+)
- SAM improvements (SAM3+)
- Efficient Vision Transformers (EfficientViT)
- On-device training advances (CoreML 4+)
- TinyML for extreme edge deployment

---

## Development Workflow

> ⚠️ **IMPORTANT**: LlamaFarm services do NOT auto-reload on code changes.  
> **You MUST kill and restart each service after making changes!**

### Starting the Services

LlamaFarm consists of three main services that must be run separately during development:

#### Terminal 1: Server (Port 14345)

```bash
cd server
uv run python main.py
```

The server provides:
- REST API endpoints
- Designer web UI
- Project management
- Proxies requests to Universal Runtime

#### Terminal 2: RAG Worker

```bash
cd rag
uv run python main.py
```

The RAG worker handles:
- Document processing
- Embedding generation
- Vector store operations
- Celery async tasks

#### Terminal 3: Universal Runtime (Port 11540)

```bash
cd runtimes/universal
uv run python server.py
```

The Universal Runtime provides:
- ML model inference
- Vision processing (NEW)
- OCR and document extraction
- Anomaly detection
- Text classification

### Restarting After Code Changes

```
┌─────────────────────────────────────────────────────────────────┐
│  ⚠️  CHANGES ARE NOT AUTO-RELOADED!                            │
│                                                                 │
│  After modifying any Python file, you MUST:                    │
│                                                                 │
│  1. Find the terminal running the affected service             │
│  2. Press Ctrl+C to kill the process                           │
│  3. Re-run the start command                                   │
│                                                                 │
│  This applies to ALL services (server, rag, universal)         │
└─────────────────────────────────────────────────────────────────┘
```

#### Quick Restart Commands

```bash
# Kill all LlamaFarm processes (nuclear option)
pkill -f "python main.py" ; pkill -f "python server.py"

# Or find specific processes
ps aux | grep -E "(main.py|server.py)" | grep -v grep

# Kill by PID
kill <PID>
```

#### Which Service to Restart?

| Changed File Location | Service to Restart |
|-----------------------|-------------------|
| `server/` | Server (Terminal 1) |
| `server/api/routers/vision/` | Server (Terminal 1) |
| `server/services/vision/` | Server (Terminal 1) |
| `rag/` | RAG Worker (Terminal 2) |
| `runtimes/universal/` | Universal Runtime (Terminal 3) |
| `runtimes/universal/routers/vision/` | Universal Runtime (Terminal 3) |
| `runtimes/universal/models/` | Universal Runtime (Terminal 3) |
| `config/` | ALL services |

### Development Tips

#### 1. Use Multiple Terminal Tabs/Panes

Recommended setup with tmux or terminal tabs:
```
┌─────────────────┬─────────────────┬─────────────────┐
│    Server       │      RAG        │    Runtime      │
│   (14345)       │                 │    (11540)      │
├─────────────────┴─────────────────┴─────────────────┤
│                    Editor/Code                       │
└─────────────────────────────────────────────────────┘
```

#### 2. Check Service Health

```bash
# Server health
curl http://localhost:14345/health

# Universal Runtime health
curl http://localhost:11540/health

# Check if ports are in use
lsof -i :14345
lsof -i :11540
```

#### 3. View Logs

Services log to stdout by default. For more verbose output:

```bash
# Server with debug logging
LOG_LEVEL=DEBUG uv run python main.py

# Universal Runtime with debug logging
cd runtimes/universal
LOG_LEVEL=DEBUG uv run python server.py
```

#### 4. Testing Vision Changes

When working on vision features, typical workflow:

```bash
# 1. Make changes to runtimes/universal/routers/vision/detection.py

# 2. Kill Universal Runtime (Ctrl+C in Terminal 3)

# 3. Restart Universal Runtime
cd runtimes/universal
uv run python server.py

# 4. Test the endpoint
curl -X POST http://localhost:11540/v1/vision/detect \
  -H "Content-Type: application/json" \
  -d '{"image": "base64...", "model": "yolov8n"}'

# 5. If also changed server/api/routers/vision/, restart server too
```

#### 5. Running Tests

```bash
# Server tests
cd server
uv run pytest tests/ -v

# Universal Runtime tests
cd runtimes/universal
uv run pytest tests/ -v

# Specific test file
uv run pytest tests/test_vision_detection.py -v

# With coverage
uv run pytest tests/ --cov=. --cov-report=html
```

### Common Issues

| Issue | Solution |
|-------|----------|
| Port already in use | Kill existing process: `lsof -ti :14345 | xargs kill` |
| Module not found | Run from correct directory, ensure `uv sync` was run |
| Changes not reflected | **Restart the service!** |
| Connection refused | Service not running, check terminal for errors |
| CUDA out of memory | Reduce batch size or model size in config |

### Service Dependencies

```
                    ┌─────────────┐
                    │   Client    │
                    │  (Browser/  │
                    │    CLI)     │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Server    │ ◄── Start first
                    │  (14345)    │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
       ┌───────────┐ ┌───────────┐ ┌───────────┐
       │    RAG    │ │ Universal │ │  Redis/   │
       │  Worker   │ │  Runtime  │ │  Celery   │
       │           │ │  (11540)  │ │ (optional)│
       └───────────┘ └───────────┘ └───────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │   Models    │
                    │  (HF Cache) │
                    └─────────────┘
```

**Recommended Start Order:**
1. Universal Runtime (loads models)
2. Server (connects to runtime)
3. RAG Worker (if using RAG features)

---

## Appendix

### A. Library Dependencies

```toml
# pyproject.toml additions

[project.optional-dependencies]
vision = [
    "ultralytics>=8.0.0",
    "transformers>=4.30.0",
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "opencv-python-headless>=4.8.0",
    "pillow>=10.0.0",
    "onnx>=1.14.0",
    "onnxruntime>=1.15.0",
    # For continual learning
    "avalanche-lib>=0.4.0",
]
```

### B. Hardware Requirements

| Task | Minimum | Recommended |
|------|---------|-------------|
| Inference (edge) | 4GB RAM, 2 CPU cores | 8GB RAM, Apple M1/NVIDIA GPU |
| Training (few-shot) | 8GB RAM, GPU optional | 16GB RAM, 8GB VRAM |
| Training (full) | 16GB RAM, 8GB VRAM | 32GB RAM, 16GB VRAM |

### C. References

1. Ultralytics YOLO Documentation: https://docs.ultralytics.com/
2. Segment Anything (SAM): https://segment-anything.com/
3. CLIP Paper: https://arxiv.org/abs/2103.00020
4. EWC Paper: https://arxiv.org/abs/1612.00796
5. Avalanche (Continual Learning): https://avalanche.continualai.org/
6. Human-in-the-Loop ML Book: https://www.manning.com/books/human-in-the-loop-machine-learning
7. Active Learning Survey: https://arxiv.org/abs/2009.09820

---

*Document Version: 1.0*  
*Last Updated: 2026-02-04*
