---
title: Examples
sidebar_position: 10
---

# Examples

LlamaFarm ships with end-to-end examples that demonstrate ingestion pipelines, RAG queries, and chat workflows. Each example lives under `examples/` in the repo with scripts you can run locally.

## FDA Letters Assistant (`examples/fda_rag`)

- **Documents**: FDA Complete Response letters (PDF).
- **Strategy**: `pdf_ingest` with heading/entity extractors.
- **Dataset script**: `examples/fda_rag/run_all.sh` (creates dataset, uploads files, processes, runs queries).
- **Highlights**:
  - Demonstrates large PDF ingestion.
  - Shows how to query for specific regulatory requirements.
  - Includes baseline chats with and without RAG.

Run the script or execute commands manually:

```bash
cd examples/fda_rag
./run_all.sh
# or step-by-step
lf datasets create -s pdf_ingest -b main_db fda_letters
lf datasets upload fda_letters ./files/*.pdf
lf datasets process fda_letters
lf rag query --database main_db "What did the 2024 letters request?"
lf chat --database main_db "Summarize deficiencies from 2024 letters"
```

## Raleigh UDO Planning Helper (`examples/gov_rag`)

- **Documents**: Raleigh Unified Development Ordinance (large PDF).
- **Strategy**: Custom PDF parser with longer chunking.
- **Dataset script**: `examples/gov_rag/run_all.sh` (noting ingestion can take several minutes).
- **Highlights**:
  - Illustrates long-running ingestion and duplicate detection.
  - Shows how to query for zoning/transition requirements.
  - Communicates to users that processing may take extra time.

Usage:

```bash
cd examples/gov_rag
./run_all.sh
# or manually
lf datasets create -s udo_pdf_processor -b raleigh_udo_db raleigh_udo_dataset
lf datasets upload raleigh_udo_dataset ./files/UDOSupplement31.pdf
lf datasets process raleigh_udo_dataset
sleep 10  # allow worker to finish
lf rag query --database raleigh_udo_db "Which section covers neighborhood transitions?"
lf chat --database raleigh_udo_db "Summarize parking requirements with citations"
```

> **Note:** The dataset processing command may time out while the worker finishes parsing. If you see a timeout, re-run the command or check worker logs—processing continues in the background.

## Build Your Own Example

1. Copy one of the example folders.
2. Update `llamafarm-example.yaml` with your runtime and RAG strategy.
3. Replace `files/` with your documents.
4. Modify the scripts to reference your dataset name and queries.
5. Document expected results in a README for easy sharing.

Share your example via PR or a discussion thread—we’re collecting real-world workflows to expand the gallery.
