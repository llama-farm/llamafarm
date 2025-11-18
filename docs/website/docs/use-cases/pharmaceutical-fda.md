# Pharmaceutical & Therapeutics: FDA Document Analysis

## Overview

Pharmaceutical and therapeutics companies undergoing FDA approval face a critical challenge: tracking hundreds of questions and answers across multiple rounds of FDA correspondence. LlamaFarm provides an automated solution to identify unanswered questions, validate existing answers, and maintain compliance throughout the approval process.

This guide provides complete step-by-step instructions with code examples and configuration. All steps are self-contained and can be followed using only this documentation.

:::tip Video Walkthrough Available
A full video demonstration of this use case is available as a supplement to this guide: [FDA Document Analysis with LlamaFarm](https://loom.com/share/19b0f86d7e074025b12ca675c2257f25)
:::

## Business Problem

During the FDA approval process, companies must:

- Track regulatory questions across multiple document types (Complete Response Letters, Information Requests, Meeting Minutes)
- Ensure all FDA questions have been adequately answered
- Validate answers against historical correspondence (COFUS database)
- Maintain confidence levels for answer authenticity
- Avoid missing critical unanswered questions that could delay approval

Manual review is time-consuming, error-prone, and doesn't scale as document volumes increase.

## Solution Architecture

LlamaFarm's agent-based approach automates this workflow:

```
FDA Documents → Vector Database → Agent Analysis → Question Extraction →
Answer Validation → Confidence Scoring → Summary Report
```

### Key Components

1. **RAG-Enabled Document Store**: All FDA correspondence ingested into vector database
2. **Document Analysis Agent**: Recursively processes chunks to extract questions
3. **Answer Validation Agent**: Cross-references extracted questions against COFUS database
4. **Batch Orchestrator**: Manages processing of large document sets
5. **Confidence Scoring**: Provides reliability metrics for each answer

## Standard Operating Procedure (SOP)

### Prerequisites

- LlamaFarm installed and configured
- FDA documents in supported formats (PDF, Word, etc.)
- Access to COFUS or equivalent answer database

### Step 1: Ingest FDA Documents into Vector Database

Create a dataset and ingest all FDA correspondence:

```bash
# Create dataset for FDA documents
lf datasets create fda_correspondence -s universal_processor -b fda_db

# Upload documents (supports glob patterns)
lf datasets upload fda_correspondence ./fda_documents/*.pdf

# Process into vector database
lf datasets process fda_correspondence
```

**Best Practice**: Organize documents by submission cycle (e.g., `cycle_1/`, `cycle_2/`) for easier tracking.

### Step 2: Start Recursive Script for Document Analysis

Configure and run the FDA document analyzer agent:

```bash
# Run the FDA document analyzer
lf agents run fda_document_analyzer --input-file ./config/fda_input.json
```

The agent will:
- Break documents into manageable chunks
- Process each chunk independently
- Track progress across the entire corpus

**Configuration Example** (`fda_input.json`):

```json
{
  "database": "fda_db",
  "document_types": ["complete_response", "information_request", "meeting_minutes"],
  "chunk_size": 4000,
  "chunk_overlap": 200
}
```

### Step 3: Extract Questions from Documents

The agent sends document chunks to the LLM with specialized prompts to identify regulatory questions:

**System Prompt Example**:
```
You are analyzing FDA correspondence. Extract all regulatory questions
from the provided text. Focus on substantive questions about:
- Clinical data requirements
- Safety/efficacy concerns
- Manufacturing/quality controls
- Labeling requirements

Exclude administrative questions (meeting scheduling, contact info, etc.)

Return questions in this format:
{
  "question": "...",
  "category": "clinical|safety|manufacturing|labeling",
  "document_section": "..."
}
```

### Step 4: Validate Answers Against COFUS Database

For each extracted question, the agent:

1. Queries the COFUS database using RAG
2. Retrieves relevant passages
3. Validates if the question has been adequately answered
4. Assesses answer authenticity

```bash
# Query specific question against COFUS
lf rag query --database cofus_db \
  --score-threshold 0.7 \
  "Has clinical endpoint XYZ been addressed?"
```

### Step 5: Save Results and Confidence Scores

The agent generates structured output with confidence metrics:

```json
{
  "question_id": "Q_001",
  "question": "What additional clinical data is required for endpoint validation?",
  "status": "answered",
  "confidence_score": 0.95,
  "answer_source": "COFUS Letter 2024-03-15",
  "answer_summary": "Two additional Phase 3 studies required...",
  "validation_method": "semantic_match"
}
```

**Confidence Threshold**: Focus on scores ≥ 90% for reliable answers. Questions with lower scores may require manual review.

### Step 6: Review Summary of Findings

Generate an executive summary report:

```bash
lf agents run fda_summary_generator --input-file ./results/analysis_results.json
```

**Sample Summary Output**:

```
FDA Document Analysis Summary
=============================
Total Questions Identified: 47
Answered Questions: 42 (89%)
Unanswered Questions: 5 (11%)
Average Confidence Score: 0.93

High Priority Unanswered Questions:
1. [Clinical] What is the required duration for long-term safety follow-up?
2. [Manufacturing] Has the API impurity profile been fully characterized?
3. [Labeling] Are pediatric use restrictions required in Section 8?

Next Actions:
- Review 3 low-confidence answers (0.70-0.85 range)
- Prepare responses for 5 unanswered questions
- Submit supplemental information package
```

### Step 7: Utilize Batch Orchestrator for Processing

For large document sets, use the batch orchestrator:

```bash
# Process multiple documents in parallel
lf agents run batch_orchestrator \
  --agent fda_document_analyzer \
  --input-dir ./fda_documents/ \
  --concurrency 5 \
  --output-dir ./results/
```

**Monitoring**:
```bash
# Check processing status
lf agents status batch_orchestrator

# View logs
lf agents logs batch_orchestrator --tail 100
```

### Step 8: Adjust Agents and System Prompts as Needed

Customize agents based on your specific regulatory focus:

**Example: Prioritize Safety Questions**

Edit `llamafarm.yaml`:

```yaml
agents:
  fda_document_analyzer:
    system_prompt: |
      You are analyzing FDA correspondence with EXTRA FOCUS on safety concerns.
      Prioritize questions related to:
      - Adverse events
      - Safety signal monitoring
      - Risk mitigation strategies

      Mark safety questions with HIGH priority.

    parameters:
      temperature: 0.2  # Lower for consistency
      top_k: 5
      score_threshold: 0.85
```

Test configurations on a subset before scaling:

```bash
# Test on 10 documents first
lf agents run fda_document_analyzer \
  --input-file ./test_config.json \
  --limit 10 \
  --dry-run
```

## Cautionary Notes

⚠️ **Document Formatting**: Ensure documents are properly formatted before ingestion. Scanned PDFs may require OCR preprocessing.

⚠️ **Process Interruption**: The recursive process saves checkpoints. If stopped, it can resume from the last checkpoint, but plan for uninterrupted runs of 2-4 hours for large document sets.

⚠️ **Confidence Thresholds**: Don't rely solely on high-confidence scores. Critical questions should undergo manual review regardless of score.

⚠️ **Model Selection**: Use more powerful models (e.g., GPT-4, Claude) for regulatory analysis. Smaller models may miss nuanced questions.

## Tips for Efficiency

💡 **Overnight Processing**: Start the analysis at the end of the workday and let agents run independently for several hours. Review results the next morning.

💡 **Staged Rollout**: Test on a subset (10-20 documents) first to validate prompts and configuration before processing the full corpus.

💡 **Model Switching**: Use fast models for initial question extraction, then switch to powerful models for answer validation:

```bash
# Fast model for extraction
lf chat --model fast --agent question_extractor

# Powerful model for validation
lf chat --model powerful --agent answer_validator
```

💡 **Incremental Processing**: Process documents as they arrive rather than batch-processing at the end:

```bash
# Add to existing dataset
lf datasets upload fda_correspondence ./new_documents/*.pdf
lf datasets process fda_correspondence
```

## Results & ROI

Organizations using this workflow report:

- **Time Savings**: 80-90% reduction in manual document review time
- **Accuracy**: 95%+ question identification rate (when using appropriate models)
- **Risk Mitigation**: Earlier identification of unanswered questions
- **Audit Trail**: Complete tracking of all questions and answers for regulatory inspections

## Getting Started

1. **Review the example**: Check out `examples/fda_rag/` in the LlamaFarm repository
2. **Start small**: Begin with one submission cycle (5-10 documents)
3. **Iterate**: Refine prompts and configuration based on results
4. **Scale**: Expand to full document corpus once validated

**Optional**: [Watch the full video walkthrough](https://loom.com/share/19b0f86d7e074025b12ca675c2257f25) for a visual demonstration of these steps.

## Additional Resources

- [RAG Configuration Guide](../rag/index.md)
- [Agent Development Guide](../extending/index.md)
- [CLI Reference](../cli/index.md)

## Questions?

If you're implementing this workflow and need assistance, please reach out through our [GitHub Discussions](https://github.com/llama-farm/llamafarm/discussions).
