# OCR Preprocessors

High-accuracy OCR with layout analysis, table extraction, and formula recognition using PaddleOCR.

## Quick Start

### Installation

```bash
# Install OCR dependencies (CPU version)
uv pip install --extra ocr

# For GPU acceleration (optional)
uv pip install --extra ocr-gpu
```

### Pre-Download Models (Recommended)

OCR models are downloaded automatically on first use, but this can take 5-10 minutes. To speed up your first run, pre-download models:

```bash
# Run the warmup script to cache all models
cd rag
uv run python scripts/warmup_ocr_models.py
```

This downloads ~1-2GB of models that are cached in `~/.paddlex/official_models/`.

**Models downloaded:**
- Document layout detection (PP-DocLayout_plus-L)
- Text detection & recognition (PP-OCRv5)
- Table extraction (SLANeXt, SLANet_plus, RT-DETR)
- Formula recognition (PP-FormulaNet_plus-L)
- Chart parsing (PP-Chart2Table)

### Configuration

Add OCR preprocessing to your `llamafarm.yaml`:

```yaml
rag:
  data_processing_strategies:
    - name: ocr_with_markitdown
      description: OCR for scanned documents with table extraction

      preprocessors:
        - type: PaddleOCRPreprocessor
          config:
            language: en              # en, ch, fr, german, korean, japan
            extract_tables: true      # Enable table extraction
            output_format: markdown   # markdown (recommended) or text
            table_confidence_threshold: 0.7
            scanned_threshold: 50     # chars/page threshold for scanned detection
          file_include_patterns:
            - '*.pdf'
            - '*.png'
            - '*.jpg'
            - '*.jpeg'
            - '*.tiff'
          priority: 10

      parsers:
        - type: MarkdownParser_Python
          config:
            chunk_size: 1000
            chunk_strategy: sections
            chunk_overlap: 100
          file_include_patterns: ['*']
          priority: 100
```

## Available Preprocessors

### PaddleOCRPreprocessor (Primary)

**Capabilities:**
- Text extraction with 95%+ accuracy
- Layout analysis (detect headers, paragraphs, tables, formulas)
- Table extraction with structure preservation
- Formula recognition (LaTeX output)
- 80+ language support
- Orientation detection

**Configuration Options:**

```yaml
config:
  # Language settings
  language: en                    # en, ch, fr, german, korean, japan, etc.

  # Table extraction (MVP critical)
  extract_tables: true            # Enable table detection & extraction
  table_confidence_threshold: 0.7 # Filter low-confidence tables
  merge_tables_inline: false      # Append tables at end vs. inline

  # Output format
  output_format: markdown         # markdown (best for RAG), text

  # Scanned PDF detection
  scanned_threshold: 50           # chars/page threshold
  min_confidence: 0.6             # Minimum OCR confidence
```

**Performance:**
- Speed: ~0.5s per page (CPU), ~0.1s (GPU)
- Accuracy: >95% for printed text, >85% for handwritten
- Memory: ~500MB base + 1GB for table extraction

### TesseractPreprocessor (Fallback)

Classic OCR engine with 100+ language support. Lower accuracy but faster installation.

```yaml
preprocessors:
  - type: TesseractPreprocessor
    config:
      language: eng               # eng, fra, deu, spa, etc.
      psm: 3                      # Page segmentation mode
      oem: 3                      # OCR Engine mode
      min_confidence: 0.5
      scanned_threshold: 50
    file_include_patterns: ['*.pdf', '*.png', '*.jpg']
    priority: 50  # Lower priority than PaddleOCR
```

## Docker Deployment

### Option 1: Pre-download During Build (Production)

Build Docker image with models included:

```bash
# Build with model pre-download (adds 5-10 minutes to build, saves time at runtime)
docker build --build-arg WARMUP_OCR_MODELS=1 -t llamafarm/rag:latest .

# Models are baked into the image at /root/.paddlex/official_models/
```

**Pros:**
- Instant startup (no model download delay)
- Predictable container size
- Better for production deployments

**Cons:**
- Longer build time
- Larger image size (~3GB vs ~1GB)

### Option 2: Download on First Use (Development)

```bash
# Build without models (default)
docker build -t llamafarm/rag:latest .

# Models download automatically on first OCR operation
# Cache persists if you mount ~/.paddlex as a volume:
docker run -v ~/.paddlex:/root/.paddlex llamafarm/rag:latest
```

**Pros:**
- Faster builds
- Smaller images

**Cons:**
- First OCR operation takes 5-10 minutes
- Requires internet access at runtime

## Language Support

PaddleOCR supports 80+ languages. Change the `language` config:

```yaml
config:
  language: ch      # Chinese
  language: fr      # French
  language: german  # German
  language: japan   # Japanese
  language: korean  # Korean
  language: en      # English (default)
```

Full list: https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/multi_languages_en.md

## Troubleshooting

### "Models downloading on first use"

This is normal. Pre-download with `scripts/warmup_ocr_models.py` to avoid delays.

### "Out of memory" errors

PPStructureV3 requires ~1.5GB RAM. Reduce memory usage:

```yaml
config:
  extract_tables: false  # Disable table extraction (saves ~1GB)
```

### "Table extraction not working"

Ensure `paddlex[ocr]` is installed:

```bash
uv pip install "paddlex[ocr]>=3.0.0"
```

### GPU acceleration not working

Install GPU-enabled PaddlePaddle:

```bash
# CUDA version
uv pip install paddlepaddle-gpu

# Or use ocr-gpu extra
uv pip install --extra ocr-gpu
```

## Model Files Location

Models are cached in:
- **Linux/macOS:** `~/.paddlex/official_models/`
- **Docker:** `/root/.paddlex/official_models/`

To clear cache and re-download:

```bash
rm -rf ~/.paddlex/official_models/
```

## Architecture

```
Input (PDF/Image)
    ↓
PaddleOCRPreprocessor
    ├── Is PDF? → Convert pages to images (PyMuPDF)
    ├── Run PPStructureV3
    │   ├── Layout Detection → Identify regions (text, tables, formulas)
    │   ├── Table Extraction → Extract table structure as HTML
    │   ├── Formula Recognition → Convert to LaTeX
    │   └── Text OCR → Extract text with bboxes
    ↓
PreprocessorResult
    ├── content: Markdown text with tables
    ├── metadata: {layout, tables, formulas, confidence}
    └── output_format: "markdown"
    ↓
MarkdownParser (chunking)
    ↓
Vector Database
```

## Examples

### Example 1: Scanned PDF with Tables

```python
from components.preprocessors.factory import PreprocessorFactory

# Create preprocessor
preprocessor = PreprocessorFactory.create(
    "PaddleOCRPreprocessor",
    config={
        "language": "en",
        "extract_tables": True,
        "output_format": "markdown",
    }
)

# Process document
result = preprocessor.preprocess(
    "invoice_scan.pdf",
    metadata={"source": "scanned_invoice"}
)

print(result.content)  # Markdown with tables
print(result.metadata["table_count"])  # Number of tables found
```

### Example 2: Multilingual Document

```python
preprocessor = PreprocessorFactory.create(
    "PaddleOCRPreprocessor",
    config={
        "language": "ch",  # Chinese
        "extract_tables": True,
    }
)

result = preprocessor.preprocess("chinese_contract.pdf", {})
```

## Performance Tips

1. **Pre-download models** before production deployment
2. **Use GPU** if available (5-10x faster)
3. **Disable table extraction** if not needed (saves memory & time)
4. **Increase DPI** for better accuracy on low-quality scans (edit `_process_pdf` method)
5. **Batch process** multiple documents in parallel

## References

- PaddleOCR Documentation: https://github.com/PaddlePaddle/PaddleOCR
- PP-StructureV3 Paper: https://arxiv.org/abs/2301.07820
- Model Zoo: https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/models_list_en.md
