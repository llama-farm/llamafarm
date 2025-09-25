#!/usr/bin/env python3
"""
Update all component schema.yaml and defaults.yaml files to match JSON Schema format
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any

# Base path for components
COMPONENTS_PATH = Path(__file__).parent.parent / "components"


def create_json_schema_header(
    component_type: str, component_name: str, title: str, description: str
) -> Dict[str, Any]:
    """Create standard JSON Schema header"""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": f"components/{component_type}/{component_name}/schema.yaml",
        "title": title,
        "description": description,
        "type": "object",
        "additionalProperties": False,
    }


# ============================================================================
# EXTRACTORS
# ============================================================================


def update_keyword_extractor():
    """Update Keyword Extractor"""
    schema_path = COMPONENTS_PATH / "extractors" / "keyword_extractor" / "schema.yaml"
    defaults_path = (
        COMPONENTS_PATH / "extractors" / "keyword_extractor" / "defaults.yaml"
    )

    schema = create_json_schema_header(
        "extractors",
        "keyword_extractor",
        "Keyword Extractor Configuration",
        "Extract keywords from text using various algorithms",
    )
    schema["properties"] = {
        "algorithm": {
            "type": "string",
            "enum": ["rake", "yake", "tfidf", "textrank"],
            "default": "rake",
            "description": "Extraction algorithm",
        },
        "max_keywords": {
            "type": "integer",
            "default": 10,
            "minimum": 1,
            "maximum": 100,
            "description": "Maximum keywords",
        },
        "min_keyword_length": {
            "type": "integer",
            "default": 3,
            "minimum": 1,
            "description": "Minimum keyword length",
        },
        "include_scores": {
            "type": "boolean",
            "default": True,
            "description": "Include relevance scores",
        },
        "language": {
            "type": "string",
            "default": "english",
            "description": "Language for stop words",
        },
    }

    with open(schema_path, "w") as f:
        f.write("# Keyword Extractor Component Schema\n")
        f.write("# JSON Schema draft-07 format\n")
        yaml.dump(schema, f, default_flow_style=False, sort_keys=False)

    defaults = {
        "general_purpose": {
            "name": "General Purpose",
            "description": "Standard keyword extraction",
            "config": {
                "algorithm": "rake",
                "max_keywords": 10,
                "min_keyword_length": 3,
                "include_scores": True,
                "language": "english",
            },
            "recommended_for": ["Documents", "Articles", "Reports"],
        }
    }

    with open(defaults_path, "w") as f:
        f.write("# Keyword Extractor Default Configurations\n\n")
        yaml.dump(defaults, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Updated Keyword Extractor")


def update_entity_extractor():
    """Update Entity Extractor"""
    schema_path = COMPONENTS_PATH / "extractors" / "entity_extractor" / "schema.yaml"
    defaults_path = (
        COMPONENTS_PATH / "extractors" / "entity_extractor" / "defaults.yaml"
    )

    schema = create_json_schema_header(
        "extractors",
        "entity_extractor",
        "Entity Extractor Configuration",
        "Extract named entities using NER",
    )
    schema["properties"] = {
        "model": {
            "type": "string",
            "default": "en_core_web_sm",
            "description": "NER model name",
        },
        "entity_types": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "PERSON",
                    "ORG",
                    "GPE",
                    "DATE",
                    "TIME",
                    "MONEY",
                    "EMAIL",
                    "PHONE",
                    "URL",
                    "LAW",
                ],
            },
            "default": [
                "PERSON",
                "ORG",
                "GPE",
                "DATE",
                "TIME",
                "MONEY",
                "EMAIL",
                "PHONE",
                "URL",
            ],
            "description": "Entity types to extract",
        },
        "use_fallback": {
            "type": "boolean",
            "default": True,
            "description": "Use regex fallback",
        },
        "min_entity_length": {
            "type": "integer",
            "default": 2,
            "minimum": 1,
            "description": "Minimum entity length",
        },
        "merge_entities": {
            "type": "boolean",
            "default": True,
            "description": "Merge adjacent entities",
        },
        "confidence_threshold": {
            "type": "number",
            "default": 0.7,
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Minimum confidence score",
        },
    }

    with open(schema_path, "w") as f:
        f.write("# Entity Extractor Component Schema\n")
        f.write("# JSON Schema draft-07 format\n")
        yaml.dump(schema, f, default_flow_style=False, sort_keys=False)

    defaults = {
        "general_purpose": {
            "name": "General Purpose",
            "description": "Standard entity extraction",
            "config": {
                "model": "en_core_web_sm",
                "entity_types": [
                    "PERSON",
                    "ORG",
                    "GPE",
                    "DATE",
                    "TIME",
                    "MONEY",
                    "EMAIL",
                    "PHONE",
                    "URL",
                ],
                "use_fallback": True,
                "min_entity_length": 2,
                "merge_entities": True,
                "confidence_threshold": 0.7,
            },
            "recommended_for": ["Documents", "Articles", "Reports"],
        }
    }

    with open(defaults_path, "w") as f:
        f.write("# Entity Extractor Default Configurations\n\n")
        yaml.dump(defaults, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Updated Entity Extractor")


# ============================================================================
# EMBEDDERS
# ============================================================================


def update_ollama_embedder():
    """Update Ollama Embedder"""
    schema_path = COMPONENTS_PATH / "embedders" / "ollama_embedder" / "schema.yaml"
    defaults_path = COMPONENTS_PATH / "embedders" / "ollama_embedder" / "defaults.yaml"

    schema = create_json_schema_header(
        "embedders",
        "ollama_embedder",
        "Ollama Embedder Configuration",
        "Generate embeddings using Ollama models",
    )
    schema["properties"] = {
        "model": {
            "type": "string",
            "default": "nomic-embed-text",
            "enum": ["nomic-embed-text", "mxbai-embed-large"],
            "description": "Ollama model name",
        },
        "base_url": {
            "type": "string",
            "format": "uri",
            "default": "http://localhost:11434",
            "description": "Ollama API endpoint",
        },
        "dimension": {
            "type": "integer",
            "default": 768,
            "minimum": 128,
            "maximum": 4096,
            "description": "Embedding dimension",
        },
        "batch_size": {
            "type": "integer",
            "default": 16,
            "minimum": 1,
            "maximum": 128,
            "description": "Batch processing size",
        },
        "timeout": {
            "type": "integer",
            "default": 60,
            "minimum": 10,
            "description": "Request timeout (seconds)",
        },
        "auto_pull": {
            "type": "boolean",
            "default": True,
            "description": "Auto-pull missing models",
        },
    }

    with open(schema_path, "w") as f:
        f.write("# Ollama Embedder Component Schema\n")
        f.write("# JSON Schema draft-07 format\n")
        yaml.dump(schema, f, default_flow_style=False, sort_keys=False)

    defaults = {
        "general_purpose": {
            "name": "General Purpose",
            "description": "Standard Ollama embeddings",
            "config": {
                "model": "nomic-embed-text",
                "base_url": "http://localhost:11434",
                "dimension": 768,
                "batch_size": 16,
                "timeout": 60,
                "auto_pull": True,
            },
            "recommended_for": ["Local deployment", "Privacy-focused", "Development"],
        }
    }

    with open(defaults_path, "w") as f:
        f.write("# Ollama Embedder Default Configurations\n\n")
        yaml.dump(defaults, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Updated Ollama Embedder")


def update_openai_embedder():
    """Update OpenAI Embedder"""
    schema_path = COMPONENTS_PATH / "embedders" / "openai_embedder" / "schema.yaml"
    defaults_path = COMPONENTS_PATH / "embedders" / "openai_embedder" / "defaults.yaml"

    schema = create_json_schema_header(
        "embedders",
        "openai_embedder",
        "OpenAI Embedder Configuration",
        "Generate embeddings using OpenAI API",
    )
    schema["properties"] = {
        "api_key": {"type": "string", "description": "OpenAI API key"},
        "model": {
            "type": "string",
            "default": "text-embedding-3-small",
            "enum": [
                "text-embedding-3-small",
                "text-embedding-3-large",
                "text-embedding-ada-002",
            ],
            "description": "OpenAI model",
        },
        "dimension": {
            "type": ["integer", "null"],
            "minimum": 256,
            "maximum": 3072,
            "description": "Override dimension",
        },
        "batch_size": {
            "type": "integer",
            "default": 100,
            "minimum": 1,
            "maximum": 2048,
            "description": "Batch size",
        },
        "max_retries": {
            "type": "integer",
            "default": 3,
            "minimum": 0,
            "maximum": 10,
            "description": "Max retry attempts",
        },
    }

    schema["required"] = ["api_key"]

    with open(schema_path, "w") as f:
        f.write("# OpenAI Embedder Component Schema\n")
        f.write("# JSON Schema draft-07 format\n")
        yaml.dump(schema, f, default_flow_style=False, sort_keys=False)

    defaults = {
        "small_model": {
            "name": "Small Model",
            "description": "Fast and cost-effective",
            "config": {
                "api_key": "${OPENAI_API_KEY}",
                "model": "text-embedding-3-small",
                "dimension": None,
                "batch_size": 100,
                "max_retries": 3,
            },
            "recommended_for": ["Production", "High throughput", "Cost optimization"],
        },
        "large_model": {
            "name": "Large Model",
            "description": "Higher quality embeddings",
            "config": {
                "api_key": "${OPENAI_API_KEY}",
                "model": "text-embedding-3-large",
                "dimension": None,
                "batch_size": 50,
                "max_retries": 3,
            },
            "recommended_for": ["High accuracy", "Research", "Quality focus"],
        },
    }

    with open(defaults_path, "w") as f:
        f.write("# OpenAI Embedder Default Configurations\n\n")
        yaml.dump(defaults, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Updated OpenAI Embedder")


# ============================================================================
# STORES
# ============================================================================


def update_chroma_store():
    """Update Chroma Store"""
    schema_path = COMPONENTS_PATH / "stores" / "chroma_store" / "schema.yaml"
    defaults_path = COMPONENTS_PATH / "stores" / "chroma_store" / "defaults.yaml"

    schema = create_json_schema_header(
        "stores",
        "chroma_store",
        "Chroma Store Configuration",
        "ChromaDB vector database configuration",
    )
    schema["properties"] = {
        "collection_name": {
            "type": "string",
            "default": "documents",
            "pattern": "^[a-zA-Z0-9_-]+$",
            "description": "Collection name",
        },
        "host": {
            "type": ["string", "null"],
            "default": None,
            "description": "Server host",
        },
        "port": {
            "type": "integer",
            "default": 8000,
            "minimum": 1,
            "maximum": 65535,
            "description": "Server port",
        },
        "distance_function": {
            "type": "string",
            "enum": ["cosine", "l2", "ip"],
            "default": "cosine",
            "description": "Distance metric",
        },
        "embedding_function": {
            "type": ["string", "null"],
            "default": None,
            "description": "Built-in embedding function",
        },
    }

    with open(schema_path, "w") as f:
        f.write("# Chroma Store Component Schema\n")
        f.write("# JSON Schema draft-07 format\n")
        yaml.dump(schema, f, default_flow_style=False, sort_keys=False)

    defaults = {
        "local_persistent": {
            "name": "Local Persistent",
            "description": "Local persistent storage",
            "config": {
                "collection_name": "documents",
                "host": None,
                "port": 8000,
                "distance_function": "cosine",
                "embedding_function": None,
            },
            "recommended_for": ["Development", "Small scale", "Local deployment"],
        },
        "client_server": {
            "name": "Client-Server",
            "description": "Connect to Chroma server",
            "config": {
                "collection_name": "documents",
                "host": "localhost",
                "port": 8000,
                "distance_function": "cosine",
                "embedding_function": None,
            },
            "recommended_for": ["Production", "Distributed", "Scalable"],
        },
    }

    with open(defaults_path, "w") as f:
        f.write("# Chroma Store Default Configurations\n\n")
        yaml.dump(defaults, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Updated Chroma Store")


def update_faiss_store():
    """Update FAISS Store"""
    schema_path = COMPONENTS_PATH / "stores" / "faiss_store" / "schema.yaml"
    defaults_path = COMPONENTS_PATH / "stores" / "faiss_store" / "defaults.yaml"

    schema = create_json_schema_header(
        "stores",
        "faiss_store",
        "FAISS Store Configuration",
        "FAISS vector index configuration",
    )
    schema["properties"] = {
        "dimension": {
            "type": "integer",
            "minimum": 1,
            "maximum": 4096,
            "description": "Vector dimension",
        },
        "index_type": {
            "type": "string",
            "enum": ["Flat", "IVF", "HNSW", "LSH"],
            "default": "Flat",
            "description": "Index type",
        },
        "metric": {
            "type": "string",
            "enum": ["L2", "IP", "Cosine"],
            "default": "L2",
            "description": "Distance metric",
        },
        "nlist": {
            "type": "integer",
            "default": 100,
            "minimum": 1,
            "description": "Number of clusters (IVF)",
        },
        "nprobe": {
            "type": "integer",
            "default": 10,
            "minimum": 1,
            "description": "Clusters to search (IVF)",
        },
        "use_gpu": {
            "type": "boolean",
            "default": False,
            "description": "Enable GPU acceleration",
        },
    }

    schema["required"] = ["dimension"]

    with open(schema_path, "w") as f:
        f.write("# FAISS Store Component Schema\n")
        f.write("# JSON Schema draft-07 format\n")
        yaml.dump(schema, f, default_flow_style=False, sort_keys=False)

    defaults = {
        "flat_index": {
            "name": "Flat Index",
            "description": "Exact search, best for small datasets",
            "config": {
                "dimension": 768,
                "index_type": "Flat",
                "metric": "L2",
                "nlist": 100,
                "nprobe": 10,
                "use_gpu": False,
            },
            "recommended_for": ["Small datasets", "Exact search", "High accuracy"],
        },
        "ivf_index": {
            "name": "IVF Index",
            "description": "Approximate search for larger datasets",
            "config": {
                "dimension": 768,
                "index_type": "IVF",
                "metric": "Cosine",
                "nlist": 1000,
                "nprobe": 50,
                "use_gpu": False,
            },
            "recommended_for": ["Large datasets", "Fast search", "Trade-off accuracy"],
        },
    }

    with open(defaults_path, "w") as f:
        f.write("# FAISS Store Default Configurations\n\n")
        yaml.dump(defaults, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Updated FAISS Store")


# ============================================================================
# RETRIEVERS
# ============================================================================


def update_basic_similarity_retriever():
    """Update Basic Similarity Retriever"""
    schema_path = COMPONENTS_PATH / "retrievers" / "basic_similarity" / "schema.yaml"
    defaults_path = (
        COMPONENTS_PATH / "retrievers" / "basic_similarity" / "defaults.yaml"
    )

    schema = create_json_schema_header(
        "retrievers",
        "basic_similarity",
        "Basic Similarity Retriever Configuration",
        "Simple vector similarity search",
    )
    schema["properties"] = {
        "top_k": {
            "type": "integer",
            "default": 10,
            "minimum": 1,
            "maximum": 1000,
            "description": "Number of results",
        },
        "distance_metric": {
            "type": "string",
            "default": "cosine",
            "enum": ["cosine", "euclidean", "manhattan", "dot"],
            "description": "Distance metric",
        },
        "score_threshold": {
            "type": ["number", "null"],
            "default": None,
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Minimum similarity score",
        },
    }

    with open(schema_path, "w") as f:
        f.write("# Basic Similarity Retriever Component Schema\n")
        f.write("# JSON Schema draft-07 format\n")
        yaml.dump(schema, f, default_flow_style=False, sort_keys=False)

    defaults = {
        "general_purpose": {
            "name": "General Purpose",
            "description": "Standard similarity search",
            "config": {
                "top_k": 10,
                "distance_metric": "cosine",
                "score_threshold": None,
            },
            "recommended_for": ["Simple queries", "Fast retrieval", "Baseline"],
        },
        "high_precision": {
            "name": "High Precision",
            "description": "Higher threshold for better precision",
            "config": {"top_k": 5, "distance_metric": "cosine", "score_threshold": 0.7},
            "recommended_for": ["Precise matching", "Quality over quantity"],
        },
    }

    with open(defaults_path, "w") as f:
        f.write("# Basic Similarity Retriever Default Configurations\n\n")
        yaml.dump(defaults, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Updated Basic Similarity Retriever")


def update_remaining_extractors():
    """Update remaining extractors with minimal config"""
    extractors = [
        ("datetime_extractor", "DateTime Extractor", "Extract dates and times"),
        ("heading_extractor", "Heading Extractor", "Extract document headings"),
        ("link_extractor", "Link Extractor", "Extract URLs and links"),
        ("path_extractor", "Path Extractor", "Extract file paths"),
        ("pattern_extractor", "Pattern Extractor", "Extract regex patterns"),
        ("statistics_extractor", "Statistics Extractor", "Extract text statistics"),
        ("summary_extractor", "Summary Extractor", "Generate summaries"),
        ("table_extractor", "Table Extractor", "Extract tables"),
    ]

    for name, title, description in extractors:
        schema_path = COMPONENTS_PATH / "extractors" / name / "schema.yaml"
        defaults_path = COMPONENTS_PATH / "extractors" / name / "defaults.yaml"

        schema = create_json_schema_header(
            "extractors", name, f"{title} Configuration", description
        )
        schema["properties"] = {
            "enabled": {
                "type": "boolean",
                "default": True,
                "description": "Enable this extractor",
            }
        }

        with open(schema_path, "w") as f:
            f.write(f"# {title} Component Schema\n")
            f.write("# JSON Schema draft-07 format\n")
            yaml.dump(schema, f, default_flow_style=False, sort_keys=False)

        defaults = {
            "general_purpose": {
                "name": "General Purpose",
                "description": f"Standard {name.replace('_', ' ')}",
                "config": {"enabled": True},
                "recommended_for": ["General use"],
            }
        }

        with open(defaults_path, "w") as f:
            f.write(f"# {title} Default Configurations\n\n")
            yaml.dump(defaults, f, default_flow_style=False, sort_keys=False)

        print(f"✓ Updated {title}")


def update_remaining_embedders():
    """Update remaining embedders"""
    embedders = [
        (
            "huggingface_embedder",
            "HuggingFace Embedder",
            "HuggingFace model embeddings",
        ),
        (
            "sentence_transformer_embedder",
            "Sentence Transformer",
            "Sentence transformer embeddings",
        ),
    ]

    for name, title, description in embedders:
        schema_path = COMPONENTS_PATH / "embedders" / name / "schema.yaml"
        defaults_path = COMPONENTS_PATH / "embedders" / name / "defaults.yaml"

        schema = create_json_schema_header(
            "embedders", name, f"{title} Configuration", description
        )
        schema["properties"] = {
            "model_name": {
                "type": "string",
                "default": "sentence-transformers/all-MiniLM-L6-v2",
                "description": "Model name",
            },
            "device": {
                "type": "string",
                "default": "cpu",
                "enum": ["cpu", "cuda", "mps"],
                "description": "Computation device",
            },
            "batch_size": {
                "type": "integer",
                "default": 32,
                "minimum": 1,
                "maximum": 256,
                "description": "Batch size",
            },
        }

        with open(schema_path, "w") as f:
            f.write(f"# {title} Component Schema\n")
            f.write("# JSON Schema draft-07 format\n")
            yaml.dump(schema, f, default_flow_style=False, sort_keys=False)

        defaults = {
            "general_purpose": {
                "name": "General Purpose",
                "description": f"Standard configuration",
                "config": {
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "device": "cpu",
                    "batch_size": 32,
                },
                "recommended_for": ["General use", "CPU inference"],
            }
        }

        with open(defaults_path, "w") as f:
            f.write(f"# {title} Default Configurations\n\n")
            yaml.dump(defaults, f, default_flow_style=False, sort_keys=False)

        print(f"✓ Updated {title}")


def update_remaining_stores():
    """Update remaining stores"""
    stores = [
        ("pinecone_store", "Pinecone Store", "Pinecone vector database"),
        ("qdrant_store", "Qdrant Store", "Qdrant vector database"),
    ]

    for name, title, description in stores:
        schema_path = COMPONENTS_PATH / "stores" / name / "schema.yaml"
        defaults_path = COMPONENTS_PATH / "stores" / name / "defaults.yaml"

        schema = create_json_schema_header(
            "stores", name, f"{title} Configuration", description
        )

        if name == "pinecone_store":
            schema["properties"] = {
                "api_key": {"type": "string", "description": "Pinecone API key"},
                "environment": {
                    "type": "string",
                    "default": "us-east-1-aws",
                    "description": "Pinecone environment",
                },
                "index_name": {
                    "type": "string",
                    "pattern": "^[a-z0-9-]+$",
                    "description": "Index name",
                },
                "dimension": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20000,
                    "description": "Vector dimension",
                },
                "metric": {
                    "type": "string",
                    "enum": ["euclidean", "cosine", "dotproduct"],
                    "default": "cosine",
                    "description": "Distance metric",
                },
            }
            schema["required"] = ["api_key", "index_name", "dimension"]
        else:  # qdrant
            schema["properties"] = {
                "host": {
                    "type": "string",
                    "default": "localhost",
                    "description": "Server host",
                },
                "port": {
                    "type": "integer",
                    "default": 6333,
                    "minimum": 1,
                    "maximum": 65535,
                    "description": "Server port",
                },
                "collection_name": {
                    "type": "string",
                    "default": "documents",
                    "pattern": "^[a-zA-Z0-9_-]+$",
                    "description": "Collection name",
                },
                "vector_size": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 65536,
                    "description": "Vector dimension",
                },
                "distance": {
                    "type": "string",
                    "enum": ["Cosine", "Euclid", "Dot"],
                    "default": "Cosine",
                    "description": "Distance metric",
                },
            }
            schema["required"] = ["vector_size"]

        with open(schema_path, "w") as f:
            f.write(f"# {title} Component Schema\n")
            f.write("# JSON Schema draft-07 format\n")
            yaml.dump(schema, f, default_flow_style=False, sort_keys=False)

        defaults = {
            "general_purpose": {
                "name": "General Purpose",
                "description": f"Standard {title.lower()} configuration",
                "config": schema["properties"].copy(),
                "recommended_for": ["Production", "Scalable", "Cloud"],
            }
        }

        # Set example values for defaults
        if name == "pinecone_store":
            defaults["general_purpose"]["config"]["api_key"] = "${PINECONE_API_KEY}"
            defaults["general_purpose"]["config"]["index_name"] = "documents"
            defaults["general_purpose"]["config"]["dimension"] = 768
        else:
            defaults["general_purpose"]["config"]["vector_size"] = 768

        with open(defaults_path, "w") as f:
            f.write(f"# {title} Default Configurations\n\n")
            yaml.dump(defaults, f, default_flow_style=False, sort_keys=False)

        print(f"✓ Updated {title}")


def update_remaining_retrievers():
    """Update remaining retrievers"""
    retrievers = [
        ("metadata_filtered", "Metadata Filtered", "Filter by metadata"),
        ("multi_query", "Multi Query", "Multiple query variations"),
        ("reranked", "Reranked", "Re-rank results"),
        ("hybrid_universal", "Hybrid Universal", "Combine multiple strategies"),
    ]

    for name, title, description in retrievers:
        schema_path = COMPONENTS_PATH / "retrievers" / name / "schema.yaml"
        defaults_path = COMPONENTS_PATH / "retrievers" / name / "defaults.yaml"

        schema = create_json_schema_header(
            "retrievers", name, f"{title} Retriever Configuration", description
        )
        schema["properties"] = {
            "top_k": {
                "type": "integer",
                "default": 10,
                "minimum": 1,
                "maximum": 1000,
                "description": "Number of results",
            }
        }

        with open(schema_path, "w") as f:
            f.write(f"# {title} Retriever Component Schema\n")
            f.write("# JSON Schema draft-07 format\n")
            yaml.dump(schema, f, default_flow_style=False, sort_keys=False)

        defaults = {
            "general_purpose": {
                "name": "General Purpose",
                "description": f"Standard {title.lower()} retrieval",
                "config": {"top_k": 10},
                "recommended_for": ["General use"],
            }
        }

        with open(defaults_path, "w") as f:
            f.write(f"# {title} Retriever Default Configurations\n\n")
            yaml.dump(defaults, f, default_flow_style=False, sort_keys=False)

        print(f"✓ Updated {title} Retriever")


def main():
    """Update all component schemas"""
    print("Updating ALL component schemas to JSON Schema format...\n")

    print("=== Updating Extractors ===")
    update_keyword_extractor()
    update_entity_extractor()
    update_remaining_extractors()

    print("\n=== Updating Embedders ===")
    update_ollama_embedder()
    update_openai_embedder()
    update_remaining_embedders()

    print("\n=== Updating Stores ===")
    update_chroma_store()
    update_faiss_store()
    update_remaining_stores()

    print("\n=== Updating Retrievers ===")
    update_basic_similarity_retriever()
    update_remaining_retrievers()

    print("\n✅ All components updated successfully!")


if __name__ == "__main__":
    main()
