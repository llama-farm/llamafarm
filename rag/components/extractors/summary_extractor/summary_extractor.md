# Summary Extractor

**Framework:** Pure Python (extractive summarization)

**When to use:** Generate document summaries without LLMs using sentence ranking.

**Schema fields:**
- `summary_sentences`: Number of sentences in summary
- `min_sentence_length`: Minimum sentence length
- `max_sentence_length`: Maximum sentence length
- `include_key_phrases`: Extract key phrases
- `include_statistics`: Include numeric data

**Best practices:**
- Adjust sentences based on document length
- Filter short/long sentences for quality
- Include key phrases for context
- Extract statistics for factual summaries