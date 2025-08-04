# Vector Database Document Management Strategies

## Table of Contents
1. [Overview](#overview)
2. [Deletion Strategies](#deletion-strategies)
3. [Update Strategies](#update-strategies)
4. [Hash-Based Management](#hash-based-management)
5. [Metadata Best Practices](#metadata-best-practices)
6. [Database-Specific Implementations](#database-specific-implementations)
7. [Common Scenarios & Solutions](#common-scenarios--solutions)
8. [Performance Considerations](#performance-considerations)
9. [Implementation Checklist](#implementation-checklist)

## Overview

Managing documents in vector databases requires careful planning around identification, versioning, and lifecycle management. This guide focuses on practical strategies for deletion, updates, and hash-based approaches.

### Key Principles
- **Immutability**: Prefer versioning over in-place updates
- **Traceability**: Always maintain audit trails
- **Efficiency**: Batch operations when possible
- **Consistency**: Use deterministic hashing for repeatability

## Deletion Strategies

### Strategy Comparison

| Strategy | Use Case | Pros | Cons | Recovery |
|----------|----------|------|------|----------|
| **Hard Delete** | Legal compliance, storage limits | Immediate space recovery | No recovery possible | None |
| **Soft Delete** | Standard operations | Reversible, audit trail | Storage overhead | Simple |
| **Archive & Delete** | Compliance + recovery | Balance of both | Complex workflow | From archive |
| **Cascade Delete** | Related documents | Maintains consistency | Risk of data loss | Difficult |
| **Expire & Purge** | Time-based data | Automated, predictable | Requires scheduling | Time-limited |

### CLI Usage Examples

```bash
# Soft delete documents older than 30 days
uv run python cli.py manage delete --older-than 30 --strategy soft

# Hard delete specific documents (permanent)
uv run python cli.py manage delete --doc-ids doc1 doc2 doc3 --strategy hard

# Archive and delete documents by filename
uv run python cli.py manage delete --filenames "old_contract.pdf" --strategy archive

# Delete by content hash (deduplication)
uv run python cli.py manage delete --content-hashes "sha256:abc123..." --strategy soft

# Delete expired documents
uv run python cli.py manage delete --expired --strategy soft

# Dry run to see what would be deleted
uv run python cli.py manage delete --older-than 30 --dry-run
```

### Deletion Methods by Database

| Database | ID-Based | Filter-Based | Batch Support | Transaction Support |
|----------|----------|--------------|---------------|---------------------|
| Pinecone | ✅ | ✅ | ✅ | ❌ |
| Weaviate | ✅ | ✅ | ✅ | ⚡ |
| Qdrant | ✅ | ✅ | ✅ | ✅ |
| Milvus | ✅ | ✅ | ✅ | ⚡ |
| ChromaDB | ✅ | ✅ | ✅ | ❌ |

✅ Full support | ⚡ Partial support | ❌ No support

## Update Strategies

### Update Approach Comparison

| Approach | Description | Best For | Performance Impact |
|----------|-------------|----------|-------------------|
| **Replace All** | Delete + Reinsert entire document | Small documents, major changes | High |
| **Incremental** | Update only changed chunks | Large documents, minor edits | Medium |
| **Versioning** | Keep all versions, mark active | Audit requirements | Low (query-time filtering) |
| **Copy-on-Write** | Create new version on any change | Historical tracking | Medium |
| **Merge Updates** | Combine old and new data | Partial updates | Low |

### CLI Usage Examples

```bash
# Replace document with versioning (recommended)
uv run python cli.py manage replace document.pdf --target-doc-id doc123 --strategy versioning

# Replace all content (destructive)
uv run python cli.py manage replace document.pdf --target-doc-id doc123 --strategy replace_all

# Incremental update (future feature)
uv run python cli.py manage replace document.pdf --target-doc-id doc123 --strategy incremental
```

### Pros and Cons by Strategy

#### Replace All Strategy
**Pros:**
- Simple implementation
- Guarantees consistency
- No orphaned chunks

**Cons:**
- High computational cost
- Temporary inconsistency
- Loses access history

#### Versioning Strategy
**Pros:**
- Full history retained
- Easy rollback
- Audit compliance

**Cons:**
- Storage overhead
- Query complexity
- Cleanup required

## Hash-Based Management

### Hash Strategy Applications

| Use Case | Hash Type | Purpose | Example |
|----------|-----------|---------|---------|
| **Deduplication** | Content hash | Prevent duplicate storage | SHA256 of normalized text |
| **Change Detection** | Document hash | Identify modifications | MD5 of entire file |
| **Unique IDs** | Composite hash | Deterministic identifiers | Hash of doc_id + chunk_num |
| **Version Control** | Version hash | Track document evolution | Hash of content + timestamp |
| **Integrity Check** | Chunk hash | Verify data integrity | SHA1 of chunk content |

### CLI Usage Examples

```bash
# Find duplicate documents by content hash
uv run python cli.py manage hash --find-duplicates

# Verify document integrity
uv run python cli.py manage hash --verify-integrity

# Regenerate all hashes (after algorithm change)
uv run python cli.py manage hash --rehash
```

### Hash Implementation Matrix

| Feature | Content Hash | Document Hash | Composite Hash | Version Hash |
|---------|--------------|---------------|----------------|--------------|
| Deduplication | ✅ | ⚡ | ❌ | ❌ |
| Change Detection | ⚡ | ✅ | ❌ | ✅ |
| Unique Identification | ⚡ | ❌ | ✅ | ✅ |
| Cross-Reference | ✅ | ✅ | ❌ | ⚡ |
| Storage Efficiency | ✅ | ✅ | ⚡ | ⚡ |

## Metadata Best Practices

### Essential Metadata Fields

| Field | Type | Purpose | Example |
|-------|------|---------|---------|
| `doc_id` | String | Primary document identifier | "contract_2024_001" |
| `chunk_id` | String | Unique chunk identifier | "doc123_chunk_5" |
| `filename` | String | Original source file | "quarterly_report.pdf" |
| `file_hash` | String | File-level hash | "sha256:abcd1234..." |
| `chunk_hash` | String | Content hash | "md5:5678efgh..." |
| `created_at` | ISO 8601 | Creation timestamp | "2024-01-15T10:30:00Z" |
| `updated_at` | ISO 8601 | Last modification | "2024-01-20T14:45:00Z" |
| `version` | Integer | Document version | 3 |
| `is_active` | Boolean | Current version flag | true |
| `parent_doc` | String | Related document | "master_agreement" |

### Configuration Example

```json
{
  "vector_stores": {
    "production": {
      "type": "ChromaStore",
      "config": {
        "collection_name": "production_documents",
        "metadata_config": {
          "enable_versioning": true,
          "enable_soft_delete": true,
          "hash_algorithm": "sha256",
          "retention_policy": {
            "default_ttl_days": 2555,
            "max_versions": 50,
            "cleanup_schedule": "weekly",
            "archive_before_delete": true
          },
          "required_metadata": [
            "doc_id", "chunk_id", "filename", "file_hash", 
            "chunk_hash", "created_at", "updated_at", "version"
          ],
          "audit_config": {
            "enable_audit_log": true,
            "track_access": true,
            "track_modifications": true
          }
        }
      }
    }
  }
}
```

### Comprehensive Metadata Structure
```json
{
  // Identity
  "doc_id": "unique_document_identifier",
  "chunk_id": "chunk_specific_id",
  "chunk_index": 5,
  "total_chunks": 42,
  
  // Source Information
  "filename": "original_document.pdf",
  "filepath": "/documents/2024/contracts/",
  "source_system": "sharepoint",
  "file_size": 2048576,
  
  // Hashes
  "file_hash": "sha256:full_file_hash",
  "chunk_hash": "sha256:chunk_content_hash",
  "metadata_hash": "md5:metadata_only_hash",
  
  // Timestamps
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-20T14:45:00Z",
  "indexed_at": "2024-01-20T14:46:00Z",
  "expires_at": "2026-01-15T10:30:00Z",
  
  // Versioning
  "version": 2,
  "is_active": true,
  "previous_version": "v1_doc_id",
  "change_summary": "Updated section 3.2",
  
  // Content Analysis
  "summary": "Brief content summary",
  "keywords": ["contract", "payment", "terms"],
  "language": "en",
  "reading_time_minutes": 15.5,
  
  // Relationships
  "parent_doc": "master_doc_id",
  "related_docs": ["doc_a", "doc_b"],
  "supersedes": "old_doc_id",
  
  // Permissions & Classification
  "access_level": "confidential",
  "department": "legal",
  "retention_category": "legal_documents",
  "tags": ["contract", "2024", "active"]
}
```

## Database-Specific Implementations

### ChromaDB Implementation

```python
# ChromaDB-specific features
class ChromaDocumentManager(DocumentManager):
    def delete_by_filter(self, metadata_filter: Dict[str, Any]) -> int:
        """ChromaDB supports rich metadata filtering."""
        return self.vector_store.collection.delete(
            where=metadata_filter
        )
    
    def update_metadata_batch(self, updates: List[Dict]) -> int:
        """Batch metadata updates for efficiency."""
        ids = [update["id"] for update in updates]
        metadatas = [update["metadata"] for update in updates]
        
        self.vector_store.collection.update(
            ids=ids,
            metadatas=metadatas
        )
        return len(ids)
```

### Pinecone Implementation

```python
# Pinecone-specific features
class PineconeDocumentManager(DocumentManager):
    def delete_by_namespace(self, namespace: str) -> Dict[str, Any]:
        """Pinecone namespace-based deletion."""
        return self.vector_store.index.delete(
            delete_all=True,
            namespace=namespace
        )
    
    def upsert_with_metadata(self, vectors: List[Dict]) -> int:
        """Pinecone upsert with metadata."""
        return self.vector_store.index.upsert(
            vectors=vectors,
            namespace=self.namespace
        )
```

### Weaviate Implementation

```python
# Weaviate-specific features
class WeaviateDocumentManager(DocumentManager):
    def delete_by_where_filter(self, where_filter: Dict) -> int:
        """Weaviate GraphQL-style filtering."""
        return self.vector_store.client.batch.delete_objects(
            class_name=self.class_name,
            where=where_filter
        )
    
    def batch_update_properties(self, updates: List[Dict]) -> int:
        """Batch property updates."""
        with self.vector_store.client.batch as batch:
            for update in updates:
                batch.add_data_object(update)
        return len(updates)
```

## Common Scenarios & Solutions

### Scenario Solution Matrix

| Scenario | Strategy | Key Metadata | Hash Usage | CLI Command |
|----------|----------|--------------|------------|-------------|
| **Document Replacement** | Versioning | version, updated_at, supersedes | Document hash for change detection | `manage replace` |
| **Partial Update** | Incremental update | chunk_hash, page_number | Chunk-level hashing | `manage replace --strategy incremental` |
| **Multi-version Support** | Versioning | version, is_active, created_at | Version hash for uniqueness | Auto-managed |
| **Deduplication** | Hash-based deletion | content_hash, source_count | Content normalization + hash | `manage hash --find-duplicates` |
| **Expiration** | Soft delete + purge | expires_at, retention_policy | N/A | `manage delete --expired` |
| **Consolidation** | Archive + merge | consolidated_from, parent_doc | Composite hash for new doc | `manage cleanup` |

### CLI Management Examples

```bash
# Get document statistics
uv run python cli.py manage stats --detailed

# Clean up old versions (keep only latest 5)
uv run python cli.py manage cleanup --old-versions 5

# Remove duplicate documents
uv run python cli.py manage cleanup --duplicates

# Clean up expired documents
uv run python cli.py manage cleanup --expired

# Combined cleanup operation
uv run python cli.py manage cleanup --old-versions 10 --expired
```

### Decision Flow

```
Document Change Detected
├── Is it a minor update? (<20% change)
│   └── Use Incremental Update
├── Is history important?
│   └── Use Versioning Strategy
├── Is storage critical?
│   └── Use Replace All Strategy
└── Default: Use Versioning with Cleanup
```

## Performance Considerations

### Operation Performance Comparison

| Operation | Relative Cost | Batch Benefit | Index Impact | CLI Support |
|-----------|--------------|---------------|--------------|-------------|
| Single Delete | Low | High (100x) | Minimal | ✅ |
| Filtered Delete | Medium | Medium (10x) | Moderate | ✅ |
| Mass Delete | High | N/A | Requires reindex | ✅ |
| Metadata Update | Low | High (50x) | None | ⚡ |
| Vector Update | High | Medium (5x) | Moderate | ⚡ |

### Optimization Strategies

| Strategy | When to Use | Performance Gain | CLI Implementation |
|----------|------------|------------------|-------------------|
| **Batch Operations** | >10 items | 10-100x | Built-in batching |
| **Async Processing** | Non-critical updates | 2-5x | Future feature |
| **Scheduled Maintenance** | Cleanup, reindexing | N/A | Cron integration |
| **Sharding by Date** | Time-series data | 5-10x | Multi-collection support |
| **Hash Indexes** | Frequent lookups | 10-50x | Metadata indexing |

## Implementation Checklist

### Pre-Implementation
- [ ] Define document identification strategy
- [ ] Choose hashing algorithm(s) (SHA256 recommended)
- [ ] Design metadata schema
- [ ] Plan versioning approach
- [ ] Set retention policies
- [ ] Configure vector store metadata settings

### Configuration Setup
```bash
# 1. Update your configuration file
cat config_examples/unified_multi_strategy_config.json

# 2. Test the configuration
uv run python cli.py test --test-file samples/small_sample.csv

# 3. Initialize with enhanced metadata
uv run python cli.py ingest samples/small_sample.csv
```

### Metadata Requirements
- [ ] Unique identifiers (doc_id, chunk_id)
- [ ] Timestamps (created_at, updated_at)
- [ ] File information (filename, file_hash)
- [ ] Content hashes (chunk_hash)
- [ ] Version tracking (version, is_active)
- [ ] Relationships (parent_doc, related_docs)

### Operational Procedures
- [ ] Document insertion workflow
- [ ] Update detection mechanism
- [ ] Deletion approval process
- [ ] Backup and recovery plan
- [ ] Monitoring and alerting

### Testing Checklist
```bash
# Test document management features
uv run python cli.py manage stats
uv run python cli.py manage delete --dry-run --older-than 1
uv run python cli.py manage hash --find-duplicates
uv run python cli.py manage cleanup --expired
```

## Best Practices Summary

### Configuration Best Practices

1. **Always Include Core Metadata**
   ```json
   "required_metadata": [
     "doc_id", "chunk_id", "filename", "file_hash", 
     "chunk_hash", "created_at", "updated_at", "version"
   ]
   ```

2. **Use Appropriate Hash Strategies**
   - Content hash for deduplication (SHA256)
   - File hash for change detection (MD5/SHA256)
   - Composite hash for unique IDs

3. **Plan for Scale**
   - Enable batching by default
   - Implement cleanup schedules
   - Monitor index fragmentation

4. **Maintain Data Lineage**
   - Track document evolution
   - Preserve deletion reasons
   - Log all modifications

### CLI Best Practices

```bash
# Always use dry-run first for destructive operations
uv run python cli.py manage delete --older-than 30 --dry-run

# Use soft delete by default
uv run python cli.py manage delete --older-than 30 --strategy soft

# Regular maintenance schedule
uv run python cli.py manage cleanup --old-versions 10 --expired

# Monitor your system
uv run python cli.py manage stats --detailed
```

### Common Pitfalls to Avoid

| Pitfall | Impact | Prevention | CLI Check |
|---------|--------|------------|-----------|
| No unique IDs | Duplicate data | Use deterministic hashing | `manage hash --find-duplicates` |
| Missing timestamps | No audit trail | Always set created_at/updated_at | `manage stats` |
| Ignoring versions | Data loss | Implement versioning strategy | Check `is_active` field |
| No cleanup policy | Storage bloat | Schedule regular maintenance | `manage cleanup` |
| Weak hashing | Collisions | Use SHA256 for critical IDs | `manage hash --verify-integrity` |

## Future Database Support

### Extensibility Framework

The document management system is designed to be easily extended for new vector databases:

```python
# Example: Adding Qdrant support
class QdrantDocumentManager(DocumentManager):
    def delete_by_payload(self, payload_filter: Dict) -> int:
        """Qdrant payload-based filtering."""
        return self.vector_store.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[
                    models.FieldCondition(key=k, match=models.MatchValue(value=v))
                    for k, v in payload_filter.items()
                ])
            )
        )
```

### Database Support Roadmap

| Database | Status | Unique Features | ETA |
|----------|--------|-----------------|-----|
| ChromaDB | ✅ Complete | Rich metadata filtering | Available |
| Pinecone | 🚧 Planned | Namespace-based operations | Q2 2024 |
| Weaviate | 🚧 Planned | GraphQL querying | Q2 2024 |
| Qdrant | 🚧 Planned | Advanced payload filtering | Q3 2024 |
| Milvus | 🚧 Planned | Partition management | Q3 2024 |

## Conclusion

Effective vector database management requires:
- Clear identification strategy using hashes and metadata
- Appropriate deletion and update approaches for your use case
- Comprehensive metadata including dates, files, and hashes
- Regular maintenance and monitoring through CLI tools

The RAG system provides a comprehensive CLI for managing document lifecycles across different vector databases while maintaining consistency and extensibility.

Choose strategies based on your specific requirements for consistency, performance, and compliance.