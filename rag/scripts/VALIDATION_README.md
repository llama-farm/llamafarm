# Smart Schema Validation System

## Overview

The Smart Schema Validator provides intelligent validation, diagnostics, and suggestions for RAG configuration files. It validates against the JSON Schema definitions and provides helpful error messages and suggestions.

## Features

- **Comprehensive Validation**: Validates against JSON Schema draft-07 specifications
- **Smart Suggestions**: Provides "Did you mean?" suggestions for typos
- **Detailed Diagnostics**: Shows exact paths to problematic fields
- **Example Generation**: Provides correct usage examples for missing fields
- **Fuzzy Matching**: Automatically suggests corrections for common mistakes
- **Component Awareness**: Understands all component types and their configurations
- **Default Suggestions**: Recommends defaults from component schemas

## Usage

### Command Line Usage

```bash
# Basic validation
python scripts/smart_schema_validator.py config_file.yaml

# Verbose output with details
python scripts/smart_schema_validator.py config_file.yaml --verbose

# Save report to file
python scripts/smart_schema_validator.py config_file.yaml --output report.txt

# Auto-fix common issues (coming soon)
python scripts/smart_schema_validator.py config_file.yaml --fix
```

### Programmatic Usage

```python
from scripts.smart_schema_validator import SmartSchemaValidator

# Create validator
validator = SmartSchemaValidator(verbose=True)

# Validate a configuration file
is_valid, issues = validator.validate_file("config.yaml")

# Get detailed report
report = validator.generate_report("config.yaml")
print(report)

# Get improvement suggestions
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)
suggestions = validator.suggest_improvements(config)
```

## Validation Levels

The validator reports issues at three levels:

### ERROR (Red)
Critical issues that prevent the configuration from working:
- Missing required fields
- Invalid component types
- Type mismatches
- Out-of-range values

### WARNING (Yellow)
Issues that may cause unexpected behavior:
- Unknown configuration fields
- Deprecated settings
- Non-standard naming conventions
- Suboptimal configurations

### INFO (Blue)
Suggestions for improvement:
- Performance optimizations
- Best practice recommendations
- Alternative approaches

## Component Validation

### Parsers
Validates parser configurations including:
- CSVParser: Field mappings, delimiters, encoding
- PDFParser: Page handling, metadata extraction, OCR settings
- MarkdownParser: Heading extraction, code blocks, chunking
- DocxParser: Comment extraction, table handling
- ExcelParser: Sheet selection, formula extraction
- HTMLParser: Tag removal, link extraction
- PlainTextParser: Chunk size, overlap, splitting method

### Extractors
Validates extractor configurations including:
- KeywordExtractor: Algorithm selection, keyword limits
- EntityExtractor: Entity types, confidence thresholds
- DateTimeExtractor: Date formats, timezone handling
- HeadingExtractor: Level limits, hierarchy options
- LinkExtractor: URL validation, email extraction
- PatternExtractor: Regex patterns, custom patterns
- StatisticsExtractor: Metrics selection
- SummaryExtractor: Algorithm, sentence count
- TableExtractor: Output format, header extraction

### Embedders
Validates embedder configurations including:
- OllamaEmbedder: Model names, dimensions, batch size
- OpenAIEmbedder: API keys, model selection, retry settings
- HuggingFaceEmbedder: Model names, device selection
- SentenceTransformerEmbedder: Pooling strategies, normalization

### Vector Stores
Validates vector store configurations including:
- ChromaStore: Collection names, persistence, distance functions
- FAISSStore: Index types, dimensions, GPU settings
- PineconeStore: API keys, environments, index names
- QdrantStore: Hosts, ports, collection settings, vector sizes

### Retrieval Strategies
Validates retrieval strategy configurations including:
- BasicSimilarityStrategy: Top-k, distance metrics, thresholds
- MetadataFilteredStrategy: Filters, fallback settings
- MultiQueryStrategy: Query counts, aggregation methods
- RerankedStrategy: Initial/final k, reranking factors
- HybridUniversalStrategy: Sub-strategies, combination methods

## Example Error Messages

### Missing Required Field
```
[ERROR] at 'strategies[0].name': Required field 'name' is missing
  💡 Suggestion: Add the 'name' field to the strategy
  📝 Example:
    name: "my_strategy"
```

### Unknown Component Type
```
[ERROR] at 'strategies[0].components.parser.type': Unknown parser type: 'PDFParserr'
  💡 Suggestion: Did you mean 'PDFParser'?
```

### Invalid Configuration Value
```
[ERROR] at 'strategies[0].components.embedder.config.dimension': Value 10000 exceeds maximum 4096
  💡 Suggestion: Use a value <= 4096
```

### Unknown Configuration Field
```
[WARNING] at 'strategies[0].components.parser.config.unknown_field': Unknown configuration field 'unknown_field'
  💡 Suggestion: Valid fields: combine_pages, extract_metadata, extract_images, page_separator, min_text_length
```

## Improvement Suggestions

The validator provides optimization suggestions:

```
IMPROVEMENT SUGGESTIONS:
💡 Consider using HybridUniversalStrategy for better retrieval results
💡 Add extractors to enrich document metadata (e.g., KeywordExtractor, EntityExtractor)
💡 For production, consider OpenAIEmbedder for better quality (requires API key)
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Validate Configurations

on:
  push:
    paths:
      - 'configs/*.yaml'
      - 'strategies/*.yaml'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
          
      - name: Install dependencies
        run: |
          pip install pyyaml jsonschema
          
      - name: Validate configurations
        run: |
          for config in configs/*.yaml strategies/*.yaml; do
            echo "Validating $config"
            python scripts/smart_schema_validator.py "$config"
          done
```

### Pre-commit Hook

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: validate-configs
        name: Validate Configuration Files
        entry: python scripts/smart_schema_validator.py
        language: python
        files: '\.(yaml|yml)$'
        additional_dependencies: [pyyaml, jsonschema]
```

## Common Issues and Solutions

### Issue: Strategy name doesn't follow convention
**Problem**: Name contains spaces or uppercase letters
**Solution**: Use lowercase with underscores: `my_strategy` not `My Strategy`

### Issue: Missing component configuration
**Problem**: Component type specified but no config provided
**Solution**: Add config section or use defaults from examples

### Issue: Incompatible component combinations
**Problem**: Embedder dimension doesn't match vector store requirements
**Solution**: Ensure dimensions are consistent across components

### Issue: Invalid enum values
**Problem**: Using incorrect string values for enums
**Solution**: Check allowed values in error message and use exact match

## Best Practices

1. **Always validate before deployment**: Run validation on all configuration changes
2. **Use the verbose flag during development**: Get detailed diagnostic information
3. **Check improvement suggestions**: Optimize configurations for production
4. **Save validation reports**: Keep audit trail of configuration changes
5. **Integrate with CI/CD**: Automatically validate on commits
6. **Use component defaults**: Start with recommended defaults and customize as needed

## Troubleshooting

### Validator not finding schemas
Ensure you're running from the correct directory:
```bash
cd /path/to/rag
python scripts/smart_schema_validator.py config.yaml
```

### Import errors
Install required dependencies:
```bash
pip install pyyaml jsonschema difflib
```

### Schema version mismatch
Update schemas using consolidation script:
```bash
python scripts/consolidate_schemas.py
```

## Future Enhancements

- **Auto-fix capability**: Automatically correct common issues
- **Interactive mode**: Step-by-step configuration wizard
- **Performance profiling**: Estimate performance impact of configurations
- **A/B testing support**: Compare multiple configurations
- **Visual validation**: Web interface for configuration validation
- **Schema migration**: Automatic migration between schema versions

## Support

For issues or questions about the validator:
1. Check this documentation
2. Review example configurations in `config_samples/`
3. Run with `--verbose` for detailed diagnostics
4. Check component schemas in `schemas/consolidated.yaml`
5. Report issues with full error output and configuration file