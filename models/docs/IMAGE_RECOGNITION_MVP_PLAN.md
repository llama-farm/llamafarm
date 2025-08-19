# Image Recognition Models MVP Plan

## Overview
This document outlines the plan to add image recognition capabilities to the LlamaFarm models framework, starting with YOLO and ensuring extensibility for other models and few-shot training.

## Architecture Design

### 1. New Component Type: Image Recognizers

We'll create a new component type alongside the existing ones (fine_tuners, model_apps, etc.):

```
components/
├── image_recognizers/          # NEW
│   ├── __init__.py
│   ├── yolo/
│   │   ├── __init__.py
│   │   ├── yolo_recognizer.py
│   │   ├── defaults.json
│   │   └── schema.json
│   ├── clip/                   # Future
│   └── sam/                    # Future
```

### 2. Base Classes

Create new base classes in `components/base.py`:
- `BaseImageRecognizer` - For image recognition models
- `BaseImageTrainer` - For few-shot training capability

### 3. Cross-Platform Support

#### Hardware Detection & Optimization
- **Apple Silicon (M1/M2/M3)**: Use CoreML and Metal Performance Shaders
- **NVIDIA GPUs**: Use CUDA with TensorRT optimization
- **Linux/CPU**: Use ONNX Runtime with CPU optimizations
- **AMD GPUs**: Use ROCm (future support)

## Implementation Steps

### Phase 1: Core Infrastructure (Week 1)

#### Task 1.1: Create Base Classes
```python
# components/base.py additions
class BaseImageRecognizer(ABC):
    """Base class for image recognition models."""
    
    @abstractmethod
    def detect(self, image_path: Union[Path, str], 
               confidence: float = 0.5) -> List[Detection]
    
    @abstractmethod
    def classify(self, image_path: Union[Path, str]) -> List[Classification]
    
    @abstractmethod
    def segment(self, image_path: Union[Path, str]) -> np.ndarray
    
    @abstractmethod
    def batch_process(self, image_paths: List[Path]) -> List[Any]

class BaseImageTrainer(ABC):
    """Base class for few-shot image model training."""
    
    @abstractmethod
    def add_examples(self, examples: List[TrainingExample])
    
    @abstractmethod
    def train_few_shot(self, base_model: str, epochs: int = 10)
    
    @abstractmethod
    def evaluate(self, test_data: List[Path]) -> Dict[str, float]
```

#### Task 1.2: Update Factory System
```python
# components/factory.py additions
class ImageRecognizerFactory(ComponentFactory):
    """Factory for creating image recognizer instances."""
    _registry: Dict[str, Type[BaseImageRecognizer]] = {}
    _component_type = "image_recognizers"
```

### Phase 2: YOLO Implementation (Week 1-2)

#### Task 2.1: YOLO Recognizer Component
```python
# components/image_recognizers/yolo/yolo_recognizer.py
class YOLORecognizer(BaseImageRecognizer):
    def __init__(self, config: Dict[str, Any]):
        self.model_version = config.get("version", "yolov8n")
        self.device = self._detect_device()
        self._setup_model()
    
    def _detect_device(self) -> str:
        """Auto-detect best available device."""
        if torch.backends.mps.is_available():  # Apple Silicon
            return "mps"
        elif torch.cuda.is_available():  # NVIDIA
            return "cuda"
        else:
            return "cpu"
```

#### Task 2.2: Configuration Files
```json
// components/image_recognizers/yolo/defaults.json
{
    "version": "yolov8n",
    "confidence_threshold": 0.5,
    "nms_threshold": 0.45,
    "max_detections": 100,
    "hardware_optimizations": {
        "apple_silicon": {
            "use_coreml": true,
            "use_metal": true
        },
        "nvidia": {
            "use_tensorrt": false,
            "fp16": true
        }
    }
}
```

### Phase 3: CLI Integration (Week 2)

#### Task 3.1: New CLI Commands
```python
# cli.py additions
@app.command()
def detect(
    image_path: str,
    strategy: str = "default_yolo",
    output_format: str = "json",
    confidence: float = 0.5,
    visualize: bool = False
):
    """Run object detection on an image."""
    pass

@app.command()
def train_detector(
    examples_dir: str,
    base_model: str = "yolov8n",
    epochs: int = 10,
    output_dir: str = "./fine_tuned_detector"
):
    """Train a detector with few-shot learning."""
    pass

@app.command()
def batch_detect(
    input_dir: str,
    output_dir: str,
    strategy: str = "default_yolo",
    parallel: bool = True
):
    """Process multiple images in batch."""
    pass
```

### Phase 4: Strategy System Integration (Week 2)

#### Task 4.1: Image Recognition Strategies
```yaml
# default_strategies.yaml additions
image_recognition:
  default_yolo:
    type: yolo
    config:
      version: yolov8n
      confidence_threshold: 0.5
      device: auto
  
  high_accuracy:
    type: yolo
    config:
      version: yolov8x
      confidence_threshold: 0.7
      use_tensorrt: true
  
  mobile_optimized:
    type: yolo
    config:
      version: yolov8n
      optimize_for: mobile
      quantization: int8

  multi_model_ensemble:
    models:
      - type: yolo
        weight: 0.5
      - type: clip
        weight: 0.3
      - type: sam
        weight: 0.2
```

### Phase 5: Few-Shot Training (Week 3)

#### Task 5.1: Training Pipeline
```python
# components/image_recognizers/yolo/few_shot_trainer.py
class YOLOFewShotTrainer(BaseImageTrainer):
    def train_few_shot(self, base_model: str, epochs: int = 10):
        """
        Implement few-shot training using:
        - Transfer learning from pre-trained YOLO
        - Data augmentation for small datasets
        - Cross-platform optimization
        """
        pass
```

#### Task 5.2: Data Format Support
```python
# Support multiple annotation formats
- YOLO format (.txt)
- COCO format (.json)
- Pascal VOC (.xml)
- Custom JSON format
```

### Phase 6: Pipeline Integration (Week 3)

#### Task 6.1: Combined Pipelines
```yaml
# Example: Document Processing Pipeline
document_analysis:
  steps:
    - name: detect_layout
      type: image_recognition
      model: yolo_document
      
    - name: extract_text
      type: ocr
      model: tesseract
      
    - name: analyze_content
      type: llm
      model: llama3.2
```

## API Design

### Python API
```python
from models import ImageRecognizer

# Simple detection
recognizer = ImageRecognizer(strategy="default_yolo")
results = recognizer.detect("image.jpg")

# Few-shot training
trainer = ImageRecognizer.create_trainer("yolo")
trainer.add_examples([
    {"image": "cat1.jpg", "label": "cat", "bbox": [10, 20, 100, 150]},
    {"image": "cat2.jpg", "label": "cat", "bbox": [30, 40, 120, 180]}
])
model = trainer.train_few_shot(epochs=10)

# Batch processing
results = recognizer.batch_detect(
    images=["img1.jpg", "img2.jpg"],
    parallel=True
)
```

### CLI API
```bash
# Basic detection
llamafarm detect image.jpg --strategy default_yolo --visualize

# Few-shot training
llamafarm train-detector \
  --examples ./my_dataset \
  --base-model yolov8n \
  --epochs 20 \
  --output ./my_detector

# Batch processing
llamafarm batch-detect \
  --input-dir ./images \
  --output-dir ./results \
  --parallel
```

## Testing Strategy

### Unit Tests
```python
# tests/test_image_recognizers.py
def test_yolo_detection():
    """Test YOLO detection on sample image."""
    pass

def test_device_detection():
    """Test hardware detection logic."""
    pass

def test_few_shot_training():
    """Test few-shot training pipeline."""
    pass
```

### Integration Tests
```python
# tests/test_image_pipeline.py
def test_document_processing_pipeline():
    """Test full document analysis pipeline."""
    pass

def test_cross_platform_compatibility():
    """Test on different hardware configurations."""
    pass
```

## Demo Applications

### 1. Object Detection Demo
```python
# demos/demo_object_detection.py
"""
Demonstrates:
- Loading YOLO model
- Detecting objects in images
- Visualizing results
- Exporting to different formats
"""
```

### 2. Few-Shot Custom Detector
```python
# demos/demo_few_shot_training.py
"""
Demonstrates:
- Training custom detector with 5-10 examples
- Fine-tuning for specific objects
- Evaluating performance
"""
```

### 3. Document Analysis Pipeline
```python
# demos/demo_document_pipeline.py
"""
Demonstrates:
- Layout detection with YOLO
- Text extraction with OCR
- Content analysis with LLM
- End-to-end document processing
"""
```

## Performance Targets

### Detection Performance
- **Latency**: < 50ms per image (on GPU)
- **Throughput**: > 20 images/second (batch mode)
- **Accuracy**: mAP > 0.5 for COCO dataset

### Training Performance
- **Few-shot training**: < 5 minutes for 10 examples
- **Memory usage**: < 4GB for training
- **Cross-platform**: Works on M1 Mac, NVIDIA GPU, CPU

## Extensibility Considerations

### Adding New Models
1. Create new directory under `image_recognizers/`
2. Implement `BaseImageRecognizer` interface
3. Add configuration files
4. Register in factory

### Adding New Features
- **Video processing**: Extend to handle video streams
- **Real-time detection**: Add streaming capability
- **Model optimization**: Support quantization, pruning
- **Multi-modal**: Combine with text models

## Dependencies

### Core Dependencies
```toml
# pyproject.toml additions
[dependencies]
ultralytics = "^8.0.0"  # YOLO
torch = "^2.0.0"        # PyTorch
torchvision = "^0.15.0" # Vision utilities
opencv-python = "^4.8.0" # Image processing
pillow = "^10.0.0"      # Image I/O
numpy = "^1.24.0"       # Numerical operations

[optional-dependencies]
apple = ["coremltools", "pyobjc-framework-Metal"]
nvidia = ["tensorrt", "pycuda"]
visualization = ["matplotlib", "seaborn"]
```

## Timeline

### Week 1
- [x] Architecture design
- [ ] Base classes implementation
- [ ] YOLO component basic implementation
- [ ] Hardware detection logic

### Week 2
- [ ] CLI integration
- [ ] Strategy system updates
- [ ] Basic testing
- [ ] Documentation

### Week 3
- [ ] Few-shot training
- [ ] Pipeline integration
- [ ] Demo applications
- [ ] Performance optimization

### Week 4
- [ ] Cross-platform testing
- [ ] Performance benchmarking
- [ ] Documentation completion
- [ ] Release preparation

## Success Criteria

1. **Functionality**
   - YOLO detection works on Mac M1/M2, NVIDIA, Linux
   - Few-shot training produces usable models
   - CLI commands are intuitive

2. **Performance**
   - Meets latency/throughput targets
   - Efficient memory usage
   - Hardware acceleration utilized

3. **Extensibility**
   - Easy to add new image models
   - Clear integration patterns
   - Well-documented APIs

4. **Integration**
   - Works with existing strategy system
   - Compatible with current CLI
   - Can be combined with LLM models

## Risks & Mitigations

### Risk 1: Hardware Compatibility
**Mitigation**: Implement fallback to CPU, extensive testing on different platforms

### Risk 2: Large Model Files
**Mitigation**: Use model caching, support for lightweight versions

### Risk 3: Training Complexity
**Mitigation**: Provide sensible defaults, automated hyperparameter tuning

### Risk 4: API Breaking Changes
**Mitigation**: Careful design of interfaces, versioning strategy

## Next Steps

1. Review and approve this plan
2. Set up development branch
3. Implement Phase 1 (Core Infrastructure)
4. Create initial YOLO component
5. Test on target hardware platforms