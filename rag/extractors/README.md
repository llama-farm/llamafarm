# Local-Only Extractors

This module provides extractors that work without requiring external language models, making them suitable for local processing and privacy-sensitive environments.

## Implemented Extractors (Top 5)

### 1. Keyword Extractors (`keyword_extractors.py`)

#### RAKE (Rapid Automatic Keyword Extraction)
- **Best for**: Quick phrase extraction from single documents
- **Method**: Uses stop words as delimiters, scores phrases by word co-occurrence
- **Output**: Ranked phrases with scores
- **Dependencies**: None

```python
from extractors import registry
rake = registry.create("rake", {
    "min_length": 1,
    "max_length": 4, 
    "max_keywords": 10
})
```

#### YAKE (Yet Another Keyword Extractor)
- **Best for**: High-quality keywords considering position and context
- **Method**: Statistical features including position, frequency, co-occurrence
- **Output**: Keywords/phrases with relevance scores (lower = better)
- **Dependencies**: None

```python
yake = registry.create("yake", {
    "max_ngram_size": 3,
    "max_keywords": 10,
    "deduplication_threshold": 0.9
})
```

#### TF-IDF (Term Frequency-Inverse Document Frequency)
- **Best for**: Finding unique terms relative to document corpus
- **Method**: Statistical term weighting across document collection
- **Output**: Terms ranked by TF-IDF scores
- **Dependencies**: None

```python
tfidf = registry.create("tfidf", {
    "max_features": 10,
    "ngram_range": [1, 2]
})
```

### 2. Entity Extractor (`entity_extractor.py`)

- **Best for**: Named entity recognition (persons, organizations, dates, etc.)
- **Method**: spaCy NER with regex fallbacks for emails, phones, URLs, etc.
- **Output**: Categorized entities with positions and confidence scores
- **Dependencies**: spaCy (optional), regex fallback available

```python
entities = registry.create("entities", {
    "model": "en_core_web_sm",
    "entity_types": ["PERSON", "ORG", "GPE", "DATE", "MONEY"],
    "use_fallback": True
})
```

### 3. DateTime Extractor (`datetime_extractor.py`)

- **Best for**: Extracting dates, times, and relative date expressions
- **Method**: dateutil parsing with comprehensive regex patterns
- **Output**: Parsed dates/times with interpretations
- **Dependencies**: python-dateutil (optional), regex fallback available

```python
datetime_ext = registry.create("datetime", {
    "fuzzy_parsing": True,
    "extract_relative": True,
    "extract_times": True
})
```

### 4. Content Statistics Extractor (`statistics_extractor.py`)

- **Best for**: Document analysis, readability metrics, content profiling
- **Method**: Mathematical analysis of text properties
- **Output**: Comprehensive statistics including readability scores
- **Dependencies**: None

```python
stats = registry.create("statistics", {
    "include_readability": True,
    "include_vocabulary": True,
    "include_sentiment_indicators": True
})
```

## Usage Examples

### Basic Usage
```python
from extractors import registry
from core.base import Document

# Create extractors
rake = registry.create("rake")
entities = registry.create("entities")
stats = registry.create("statistics")

# Process documents
documents = [Document(id="1", content="Your text here...")]

# Extract metadata
documents = rake.extract(documents)
documents = entities.extract(documents) 
documents = stats.extract(documents)

# Access results
doc = documents[0]
print(doc.metadata["rake_keywords"])
print(doc.metadata["entities_person"])
print(doc.metadata["word_count"])
```

### Pipeline Usage
```python
from extractors.base import ExtractorPipeline

# Create pipeline
pipeline = ExtractorPipeline([
    registry.create("yake"),
    registry.create("entities"),
    registry.create("datetime"),
    registry.create("statistics")
])

# Process documents
enhanced_docs = pipeline.run(documents)
```

### Configuration Example
```python
# Custom extractor configuration
config = {
    "extractors": [
        {
            "type": "yake",
            "config": {
                "max_keywords": 15,
                "max_ngram_size": 3,
                "language": "en"
            }
        },
        {
            "type": "entities", 
            "config": {
                "entity_types": ["PERSON", "ORG", "DATE", "MONEY"],
                "min_entity_length": 3
            }
        },
        {
            "type": "statistics",
            "config": {
                "include_readability": True,
                "include_sentiment_indicators": True
            }
        }
    ]
}
```

## Parser Integration

Extractors can be configured within parser settings:

```json
{
  "parsers": {
    "legal_documents": {
      "type": "PDFParser",
      "config": {
        "extract_metadata": true,
        "extractors": [
          {
            "type": "entities",
            "config": {
              "entity_types": ["PERSON", "ORG", "DATE", "MONEY", "LAW"]
            }
          },
          {
            "type": "yake", 
            "config": {
              "max_keywords": 20
            }
          }
        ]
      }
    }
  }
}
```

## CLI Commands

```bash
# List available extractors
uv run python cli.py extractors list

# Test extractor on file
uv run python cli.py extractors test --extractor yake --file sample.txt

# Extract from document with specific extractors
uv run python cli.py ingest sample.csv --extractors rake,entities,statistics

# Extract with config override
uv run python cli.py ingest sample.pdf --extractor-config "yake:max_keywords=15"
```

## Output Metadata Structure

Each extractor adds metadata to documents:

```python
{
  "extractors": {
    "rake_keywords": [
      {"phrase": "machine learning", "score": 9.5},
      {"phrase": "data analysis", "score": 7.2}
    ],
    "entities": {
      "PERSON": [{"text": "John Smith", "start": 10, "end": 20, "confidence": 0.9}],
      "ORG": [{"text": "OpenAI", "start": 50, "end": 56, "confidence": 0.95}]
    },
    "statistics": {
      "basic": {"word_count": 1250, "sentence_count": 45},
      "readability": {"flesch_reading_ease": 65.2, "reading_time_minutes": 6.25}
    }
  },
  // Simplified access
  "rake_keywords": ["machine learning", "data analysis"],
  "entities_person": ["John Smith"],
  "word_count": 1250,
  "reading_time_minutes": 6.25
}
```

---

## TODO: Additional Extractors to Implement

### Priority 1: Text Analysis
- [ ] **Sentiment Extractor** (TextBlob/VADER) - Detailed sentiment analysis
- [ ] **Language Detector** (langdetect) - Automatic language identification
- [ ] **Topic Classifier** (sklearn/custom) - Document topic classification
- [ ] **PII Detector** (presidio/regex) - Personally identifiable information detection
- [ ] **Duplicate Detector** (fuzzy matching) - Near-duplicate content identification

### Priority 2: Document Structure
- [ ] **Table Extractor** (camelot/tabula) - Extract tables from PDFs
- [ ] **Header/Section Extractor** - Document structure analysis
- [ ] **List Extractor** - Bullet points and numbered lists
- [ ] **Citation Extractor** - Academic/legal citations
- [ ] **Code Block Extractor** - Programming code detection

### Priority 3: Domain-Specific
- [ ] **Legal Document Extractor** - Clauses, parties, obligations
- [ ] **Financial Data Extractor** - Numbers, percentages, financial terms
- [ ] **Medical Text Extractor** (scispacy) - Medical entities and terminology
- [ ] **Technical Document Extractor** - API references, specifications
- [ ] **Email Extractor** - Email-specific metadata (headers, signatures)

### Priority 4: Advanced Analysis
- [ ] **Readability Enhancer** - Extended readability metrics
- [ ] **Writing Style Analyzer** - Formal vs informal, complexity analysis
- [ ] **Factual Statement Extractor** - Claims vs opinions identification
- [ ] **Question Extractor** - Extract questions from text
- [ ] **Action Item Extractor** - TODOs and action items

### Priority 5: Media and Special Formats
- [ ] **Image Metadata Extractor** (EXIF) - Photo metadata
- [ ] **Audio Transcript Analyzer** - Speech-to-text analysis
- [ ] **URL/Link Analyzer** - Web link analysis and validation
- [ ] **Social Media Extractor** - Hashtags, mentions, social patterns
- [ ] **Log File Analyzer** - System logs, error patterns

### Implementation Guidelines

When implementing new extractors:

1. **Follow the BaseExtractor interface**
2. **Provide fallback methods** when possible
3. **Include comprehensive configuration options**
4. **Add proper error handling and logging**
5. **Include unit tests in `tests/test_extractors.py`**
6. **Update CLI commands** for new extractors
7. **Add configuration examples** 
8. **Document dependencies** and installation requirements

### Dependencies Management

Keep extractors modular with optional dependencies:

```python
def get_dependencies(self) -> List[str]:
    """Return optional dependencies."""
    return ["optional-package"]  # Only for enhanced features

def validate_dependencies(self) -> bool:
    """Always return True if fallbacks exist.""" 
    return True  # Has regex fallback
```

This ensures the RAG system remains functional even when specific extractor dependencies are not available.