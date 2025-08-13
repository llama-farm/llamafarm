---
title: Architecture
sidebar_label: Architecture
slug: /rag/architecture
toc_min_heading_level: 2
toc_max_heading_level: 3
---

The system is built around five core components:

## 1. Parsers

Convert various data formats into a universal `Document` format.

- CSVParser: Generic CSV parser with configurable fields
- CustomerSupportCSVParser: Specialized for support ticket data
- PDFParser: Extract text, metadata, and structure from PDFs

## 2. Embedders

Generate vector embeddings from text content.

- OllamaEmbedder: Uses local Ollama models

## 3. Vector stores

Store and search document embeddings.

- ChromaStore: ChromaDB integration with persistence

## 4. Universal retrieval strategies

Database-agnostic strategies that optimize for your store.

- BasicSimilarityStrategy
- MetadataFilteredStrategy
- MultiQueryStrategy
- RerankedStrategy
- HybridUniversalStrategy

## 5. Pipeline

Chains components together for end-to-end processing.
