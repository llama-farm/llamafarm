"""
Metadata configuration and schemas for RAG system.
"""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class MetadataLevel(str, Enum):
    """Metadata detail levels."""
    MINIMAL = "minimal"      # Only essential fields
    STANDARD = "standard"    # Recommended fields
    COMPREHENSIVE = "comprehensive"  # All available fields


class CoreMetadataConfig(BaseModel):
    """Core metadata fields that should be present in all documents."""
    
    # Document identification
    generate_id: bool = Field(True, description="Auto-generate document IDs")
    id_method: str = Field("content_hash", description="Method for ID generation")
    
    # Content hashing
    generate_hash: bool = Field(True, description="Generate content hash")
    hash_algorithm: str = Field("sha256", description="Hash algorithm to use")
    
    # Temporal tracking
    add_timestamps: bool = Field(True, description="Add processing timestamps")
    timestamp_format: str = Field("iso", description="Timestamp format")
    
    # Content statistics
    calculate_stats: bool = Field(True, description="Calculate content statistics")
    stats_fields: List[str] = Field(
        default=["word_count", "char_count", "line_count"],
        description="Statistics to calculate"
    )


class MetadataSchema(BaseModel):
    """Complete metadata schema configuration."""
    
    # Metadata level
    level: MetadataLevel = Field(
        MetadataLevel.STANDARD,
        description="Level of metadata detail"
    )
    
    # Core configuration
    core: CoreMetadataConfig = Field(
        default_factory=CoreMetadataConfig,
        description="Core metadata configuration"
    )
    
    # Required fields
    required_fields: List[str] = Field(
        default=["id", "source", "processing_timestamp"],
        description="Fields that must be present"
    )
    
    # Optional fields to include
    optional_fields: List[str] = Field(
        default=[
            "title", "summary", "category", "tags",
            "language", "author", "created_date"
        ],
        description="Optional fields to include"
    )
    
    # Custom fields for specific domains
    custom_fields: Dict[str, Any] = Field(
        default={},
        description="Domain-specific custom fields"
    )
    
    # Field validation rules
    validation_rules: Dict[str, Dict[str, Any]] = Field(
        default={},
        description="Validation rules for fields"
    )
    
    # ChromaDB compatibility
    chroma_compatible: bool = Field(
        True,
        description="Ensure ChromaDB compatibility"
    )


class DocumentMetadata(BaseModel):
    """Standard document metadata model."""
    
    # Required fields
    id: str = Field(..., description="Unique document identifier")
    source: str = Field(..., description="Document source path or URL")
    processing_timestamp: str = Field(..., description="When document was processed")
    
    # Recommended fields
    content_hash: Optional[str] = Field(None, description="Content hash for deduplication")
    source_type: Optional[str] = Field(None, description="Type of source (pdf, html, etc)")
    processing_date: Optional[str] = Field(None, description="Processing date (YYYY-MM-DD)")
    
    # Content description
    title: Optional[str] = Field(None, description="Document title")
    summary: Optional[str] = Field(None, description="Document summary")
    language: Optional[str] = Field(None, description="Language code")
    word_count: Optional[int] = Field(None, description="Number of words")
    char_count: Optional[int] = Field(None, description="Number of characters")
    
    # Organization
    category: Optional[str] = Field(None, description="Document category")
    tags: Optional[List[str]] = Field(default=[], description="Document tags")
    
    # Processing information
    parser_type: Optional[str] = Field(None, description="Parser used")
    chunk_number: Optional[int] = Field(None, description="Chunk number if chunked")
    chunk_size: Optional[int] = Field(None, description="Size of chunk")
    embedding_model: Optional[str] = Field(None, description="Embedding model used")
    
    # Extractor results
    extractors: Optional[Dict[str, Any]] = Field(default={}, description="Extractor outputs")
    
    # Custom fields
    custom: Optional[Dict[str, Any]] = Field(default={}, description="Custom metadata")
    
    @validator('processing_timestamp')
    def validate_timestamp(cls, v):
        """Ensure timestamp is in valid format."""
        if v:
            try:
                # Try parsing ISO format
                datetime.fromisoformat(v.replace('Z', '+00:00'))
            except:
                raise ValueError(f"Invalid timestamp format: {v}")
        return v
    
    @validator('tags')
    def validate_tags(cls, v):
        """Ensure tags are strings."""
        if v:
            return [str(tag).strip() for tag in v if tag]
        return v
    
    def to_chroma_compatible(self) -> Dict[str, Union[str, int, float, bool, None]]:
        """Convert metadata to ChromaDB-compatible format."""
        import json
        
        chroma_metadata = {}
        
        for field_name, field_value in self.dict(exclude_none=True).items():
            if field_name == "extractors" or field_name == "custom":
                # Handle nested dictionaries
                if isinstance(field_value, dict):
                    for key, value in field_value.items():
                        if isinstance(value, (str, int, float, bool)) or value is None:
                            chroma_metadata[f"{field_name}_{key}"] = value
                        else:
                            chroma_metadata[f"{field_name}_{key}_json"] = json.dumps(value)
            elif isinstance(field_value, (str, int, float, bool)) or field_value is None:
                chroma_metadata[field_name] = field_value
            elif isinstance(field_value, list):
                # Convert lists to JSON
                chroma_metadata[f"{field_name}_json"] = json.dumps(field_value)
                chroma_metadata[f"{field_name}_count"] = len(field_value)
            else:
                # Convert other types to string
                chroma_metadata[field_name] = str(field_value)
        
        return chroma_metadata


class MetadataPresets:
    """Predefined metadata configurations for different use cases."""
    
    @staticmethod
    def minimal() -> MetadataSchema:
        """Minimal metadata for basic use cases."""
        return MetadataSchema(
            level=MetadataLevel.MINIMAL,
            required_fields=["id", "source", "processing_timestamp"],
            optional_fields=["title", "word_count"],
            core=CoreMetadataConfig(
                calculate_stats=False,
                stats_fields=["word_count"]
            )
        )
    
    @staticmethod
    def research_papers() -> MetadataSchema:
        """Metadata schema for research papers."""
        return MetadataSchema(
            level=MetadataLevel.COMPREHENSIVE,
            optional_fields=[
                "title", "abstract", "authors", "publication_date",
                "journal", "doi", "keywords", "citations"
            ],
            custom_fields={
                "paper_type": "research",
                "peer_reviewed": True,
                "fields_of_study": []
            },
            validation_rules={
                "doi": {"pattern": r"^10\.\d{4,}/.*"},
                "publication_date": {"format": "YYYY-MM-DD"}
            }
        )
    
    @staticmethod
    def customer_support() -> MetadataSchema:
        """Metadata schema for customer support tickets."""
        return MetadataSchema(
            level=MetadataLevel.STANDARD,
            required_fields=[
                "id", "source", "processing_timestamp",
                "ticket_id", "priority", "status"
            ],
            optional_fields=[
                "customer_id", "agent_id", "category",
                "resolution_time", "satisfaction_score"
            ],
            custom_fields={
                "ticket_type": "support",
                "sla_breach": False,
                "escalated": False
            }
        )
    
    @staticmethod
    def legal_documents() -> MetadataSchema:
        """Metadata schema for legal documents."""
        return MetadataSchema(
            level=MetadataLevel.COMPREHENSIVE,
            required_fields=[
                "id", "source", "processing_timestamp",
                "document_type", "jurisdiction"
            ],
            optional_fields=[
                "case_number", "parties", "effective_date",
                "expiration_date", "confidentiality", "status"
            ],
            custom_fields={
                "requires_signature": True,
                "regulatory_compliance": [],
                "related_documents": []
            }
        )
    
    @staticmethod
    def code_documentation() -> MetadataSchema:
        """Metadata schema for code documentation."""
        return MetadataSchema(
            level=MetadataLevel.STANDARD,
            optional_fields=[
                "version", "api_version", "deprecated",
                "last_reviewed", "code_examples", "related_apis"
            ],
            custom_fields={
                "programming_language": "python",
                "framework": None,
                "breaking_changes": False
            }
        )