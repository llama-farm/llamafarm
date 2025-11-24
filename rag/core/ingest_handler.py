"""
Ingest handler for LlamaFarm CLI integration.
Manages the flow from CLI file uploads to blob processing and vector storage.
"""

import importlib
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.blob_processor import BlobProcessor
from core.logging import RAGStructLogger
from core.settings import settings
from core.strategies.handler import SchemaHandler

repo_root = Path(__file__).parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from config import load_config
    from config.datamodel import Database, DataProcessingStrategy
except ImportError as e:
    raise ImportError(
        f"Could not import config module. Make sure you're running from the repo root. Error: {e}"
    ) from e

from observability.event_logger import EventLogger

logger = RAGStructLogger("rag.core.ingest_handler")


class IngestHandler:
    """
    Handles document ingestion from LlamaFarm CLI.
    Coordinates blob processing, embedding, and storage.
    """

    def __init__(
        self,
        config_path: str,
        data_processing_strategy: str,
        database: str,
        dataset_name: Optional[str] = None,
    ):
        """
        Initialize the ingest handler.

        Args:
            config_path: Path to the LlamaFarm configuration file
            data_processing_strategy: Name of the data processing strategy
            database: Name of the database to use
            dataset_name: Optional dataset name for logging
        """
        self.config_path = Path(config_path)
        self.data_processing_strategy = data_processing_strategy
        self.database = database
        self.dataset_name = dataset_name

        # Load config to extract namespace and project
        self.config = load_config(str(self.config_path))
        self.namespace = self.config.namespace
        self.project = self.config.name

        # Initialize schema handler
        self.schema_handler = SchemaHandler(config_path)

        # Get configurations separately
        self.processing_config = self._get_processing_config()
        self.database_config = self._get_database_config()

        # Get project directory from config path
        # Config path format: ~/.llamafarm/projects/{namespace}/{project}/llamafarm.yaml
        project_dir = self.config_path.parent

        # Initialize components
        self.blob_processor = BlobProcessor(self.processing_config)
        self.embedder = self._initialize_embedder(self.database_config)
        self.vector_store = self._initialize_vector_store(
            project_dir, self.database_config
        )

    def _get_processing_config(self) -> DataProcessingStrategy:
        """
        Get the data processing strategy configuration.

        Returns:
            Dictionary with parsers and extractors
        """
        return self.schema_handler.create_processing_config(
            self.data_processing_strategy
        )

    def _get_database_config(self) -> Dict[str, Any]:
        """
        Get the database configuration.

        Returns:
            Dictionary with database settings
        """
        return self.schema_handler.create_database_config(self.database)

    def _initialize_embedder(self, db_config: Dict[str, Any]):
        """
        Initialize the embedder from database configuration.

        Args:
            db_config: Database configuration

        Returns:
            Initialized embedder instance
        """
        # Get embedder configuration from config
        embedder_config = self.schema_handler.get_embedder_config(db_config)
        embedder_type = embedder_config.type.value

        if not embedder_type:
            raise ValueError("No embedder type specified in configuration")

        logger.info(
            f"Initializing embedder: {embedder_type} with config: {embedder_config.config}"
        )

        # Dynamically import the embedder based on type from config
        # Convert type like "OllamaEmbedder" to module path
        embedder_name_lower = embedder_type.replace("Embedder", "_embedder").lower()
        module_path = f"components.embedders.{embedder_name_lower}"

        try:
            # Import the module
            module = importlib.import_module(module_path)
            # Get the class (should match the type name)
            embedder_class = getattr(module, embedder_type)
            # Initialize with config from the config file
            # Pass config as a dictionary, not as kwargs
            return embedder_class(config=embedder_config.config)
        except (ImportError, AttributeError) as e:
            logger.error(
                f"Failed to load embedder {embedder_type} from {module_path}: {e}"
            )
            raise ValueError(f"Cannot initialize embedder {embedder_type}: {e}") from e

    def _initialize_vector_store(self, project_dir: Path, database: Database):
        """
        Initialize the vector store from database configuration.

        Args:
            project_dir: Project directory
            db_config: Database configuration

        Returns:
            Initialized vector store instance
        """
        # Get vector store configuration from config
        vector_store_config = database.config
        vector_store_type = database.type.value

        if not vector_store_type:
            raise ValueError("No vector store type specified in configuration")

        logger.info(
            f"Initializing vector store: {vector_store_type} with config: {vector_store_config}"
        )

        # Dynamically import the store based on type from config
        # Convert type like "ChromaStore" to module path
        store_name_lower = vector_store_type.replace("Store", "_store").lower()
        module_path = f"components.stores.{store_name_lower}"

        try:
            # Import the module
            module = importlib.import_module(module_path)
            # Get the class (should match the type name)
            store_class = getattr(module, vector_store_type)
            # Initialize with config from the config file
            # Use the already resolved config_dict from above

            # Pass config as a dictionary, not as kwargs
            return store_class(config=vector_store_config, project_dir=project_dir)
        except (ImportError, AttributeError) as e:
            logger.error(
                f"Failed to load vector store {vector_store_type} from {module_path}: {e}"
            )
            raise ValueError(
                f"Cannot initialize vector store {vector_store_type}: {e}"
            ) from e

    def ingest_file(self, file_data: bytes, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ingest a single file from the CLI.

        Args:
            file_data: Raw bytes of the file
            metadata: Metadata about the file (filename, content_type, etc.)

        Returns:
            Dictionary with ingestion results
        """
        import time

        start_time = time.time()

        filename = metadata.get("filename", "unknown")
        file_size = len(file_data)
        logger.info(f"Ingesting file: {filename}")

        # Create event logger (config hash computed internally)
        request_id = f"proc_{uuid.uuid4().hex[:12]}"
        event_logger = EventLogger(
            event_type="rag_processing",
            request_id=request_id,
            namespace=self.namespace,
            project=self.project,
            config=self.config,
        )

        # Log processing start
        event_logger.log_event("file_ingestion_start", {
            "filename": filename,
            "size_bytes": file_size,
            "content_type": metadata.get("content_type", "unknown"),
            "dataset_name": self.dataset_name,
            "database": self.database,
            "strategy": self.data_processing_strategy,
        })

        # Print file info
        logger.info(
            f"\n{'=' * 60}\n"
            f"📁 FILE: {filename}\n"
            f"   Size: {file_size:,} bytes ({file_size / 1024:.1f} KB)\n"
            f"   Type: {metadata.get('content_type', 'unknown')}\n"
            f"{'=' * 60}"
        )

        try:
            # Process the blob with the blob processor
            documents = self.blob_processor.process_blob(file_data, metadata)

            if not documents:
                event_logger.fail_event(f"No documents extracted from {filename}")
                return {
                    "status": "error",
                    "message": f"No documents extracted from {filename}",
                    "filename": filename,
                    "document_count": 0,
                }

            # Log file parsed
            # Extract parser names
            parser_names = list(set(
                doc.metadata.get("parser", "unknown")
                for doc in documents
            ))
            event_logger.log_event("file_parsed", {
                "filename": filename,
                "parsers": parser_names,
                "mime_type": metadata.get("content_type", "unknown"),
            })

            # Generate file hash for deduplication
            import hashlib

            file_hash = hashlib.sha256(file_data).hexdigest()

            # Log chunks created
            avg_chunk_size = sum(len(doc.content) for doc in documents) / len(documents) if documents else 0
            event_logger.log_event("chunks_created", {
                "chunk_count": len(documents),
                "avg_chunk_size": int(avg_chunk_size),
                "file_hash": file_hash[:16],
            })

            # Generate embeddings for each document
            embedded_documents = []
            for i, doc in enumerate(documents):
                # Generate a unique ID based on file hash and chunk index
                # This ensures the same file won't be re-embedded
                doc.id = f"{file_hash[:16]}_{i:04d}"

                # Add file hash to metadata for tracking
                doc.metadata["file_hash"] = file_hash
                doc.metadata["chunk_index"] = i
                doc.metadata["total_chunks"] = len(documents)

                # Generate embedding
                embedding = self.embedder.embed([doc.content])  # embed expects a list

                # Set embedding on the document object itself
                if embedding and len(embedding) > 0:
                    doc.embeddings = embedding[0]  # Get first embedding from list
                    if i == 0:  # Print embedding info only once
                        logger.info(
                            f"\n🧠 Embedding with {self.embedder.__class__.__name__}:"
                        )
                        logger.info(f"   └─ Dimensions: {len(doc.embeddings)}")
                embedded_documents.append(doc)

            # Log embeddings generated
            embedding_dim = len(embedded_documents[0].embeddings) if embedded_documents and hasattr(embedded_documents[0], 'embeddings') else 0
            event_logger.log_event("embeddings_generated", {
                "embedding_count": len(embedded_documents),
                "embedder": self.embedder.__class__.__name__,
                "embedding_dimension": embedding_dim,
            })

            # Store documents in vector store with duplicate detection
            # Try batch add first (more efficient)
            stored_count = 0
            skipped_count = 0
            doc_ids = []

            try:
                # Batch add all documents at once
                result = self.vector_store.add_documents(embedded_documents)

                # ChromaStore returns a list of IDs for successfully added documents
                # Empty list means all were duplicates
                if isinstance(result, list):
                    if len(result) > 0:
                        # Some or all documents were stored
                        doc_ids = result
                        stored_count = len(result)
                        skipped_count = len(embedded_documents) - stored_count

                        if stored_count == len(embedded_documents):
                            logger.info(f"Stored all {stored_count} documents")
                            logger.info(
                                f"[STORED] All {stored_count} documents embedded and stored"
                            )
                        else:
                            logger.info(
                                f"Stored {stored_count} documents, skipped {skipped_count} duplicates"
                            )
                            logger.info(
                                f"[PARTIAL] Stored {stored_count} new documents, skipped {skipped_count} duplicates"
                            )
                    else:
                        # All documents were duplicates
                        skipped_count = len(embedded_documents)
                        logger.info(
                            f"All {skipped_count} documents were duplicates - skipped"
                        )
                        logger.info(
                            f"[DUPLICATE] All {skipped_count} documents already in database - skipped"
                        )
                elif result is False:
                    # Database error occurred
                    logger.error("Database error occurred during document storage")
                    raise Exception(
                        "Failed to add documents to vector store - database error"
                    )
                else:
                    # Unexpected return type
                    logger.error(
                        f"Unexpected return type from add_documents: {type(result)}, value: {result}"
                    )
                    raise Exception(
                        f"Unexpected return from vector store: {type(result)}"
                    )

            except Exception as e:
                # If batch add fails, try individual adds for better error handling
                logger.warning(f"Batch add failed: {e}, trying individual adds")

                for doc in embedded_documents:
                    try:
                        result = self.vector_store.add_documents([doc])
                        if isinstance(result, list) and len(result) > 0:
                            doc_ids.extend(result)
                            stored_count += 1
                            logger.info(f"Stored document {doc.id}")
                            logger.info(
                                f"[STORED] Document {doc.id} embedded and stored"
                            )
                        elif isinstance(result, list) and len(result) == 0:
                            # Empty list means duplicate
                            skipped_count += 1
                            logger.info(f"Document {doc.id} is duplicate - skipped")
                            logger.info(
                                f"[DUPLICATE] Document {doc.id} already in database - skipped"
                            )
                        elif result is False:
                            # Database error
                            logger.error(f"Database error storing document {doc.id}")
                            logger.error(
                                f"[ERROR] Database error storing document {doc.id}"
                            )
                            raise Exception(f"Database error storing document {doc.id}")
                        else:
                            # Unexpected return
                            logger.error(
                                f"Unexpected return from add_documents for {doc.id}: {result}"
                            )
                            raise Exception(
                                f"Unexpected return storing document {doc.id}: {type(result)}"
                            )
                    except Exception as doc_e:
                        logger.error(f"Failed to store document {doc.id}: {doc_e}")
                        logger.error(
                            f"[ERROR] Failed to store document {doc.id}: {doc_e}"
                        )
                        # Re-raise to ensure error is not silently ignored
                        raise

            # Log chunks stored
            event_logger.log_event("chunks_stored", {
                "database": self.database,
                "stored_count": stored_count,
                "skipped_count": skipped_count,
                "storage_type": self.vector_store.__class__.__name__,
            })

            # Calculate processing time
            elapsed_time = time.time() - start_time

            # Output storage details
            logger.info(
                f"\n💾 Database Storage ({self.vector_store.__class__.__name__}):"
            )
            if stored_count > 0:
                logger.info(f"   ✅ Stored: {stored_count} new chunks")
            if skipped_count > 0:
                logger.info(f"   ⏭️  Skipped: {skipped_count} duplicate chunks")

            # Summary
            summary_lines = [
                "\n📈 Processing Summary:",
                f"  ⏱️  Total time: {elapsed_time:.2f} seconds",
            ]
            if stored_count > 0:
                summary_lines.append(
                    f"   ✅ Status: SUCCESS - {stored_count}/{len(documents)} chunks stored"
                )
            elif skipped_count > 0:
                summary_lines.append(
                    f"   ⚠️  Status: DUPLICATE - All {skipped_count} chunks already in database"
                )
            else:
                summary_lines.append("   ❌ Status: FAILED")

            summary_lines.append(
                f"Ingestion complete: {stored_count} stored, {skipped_count} skipped from {filename}"
            )

            logger.info("\n".join(summary_lines))

            # Extract parser names safely
            parser_names = []
            for doc in documents:
                parser = doc.metadata.get("parser", "unknown")
                if isinstance(parser, str):
                    parser_names.append(parser)
                else:
                    parser_names.append(str(parser))

            # Determine overall status
            if stored_count == 0 and skipped_count > 0:
                # All chunks were duplicates
                status = "skipped"
                reason = "duplicate"
                logger.warning(
                    f"\n⚠️ FILE ALREADY PROCESSED - All {skipped_count} chunks already exist in database"
                )
            else:
                status = "success"
                reason = None

            # Complete event logging
            event_logger.log_event("processing_complete", {
                "status": status,
                "total_chunks": len(documents),
                "stored_chunks": stored_count,
                "skipped_chunks": skipped_count,
                # Note: total_elapsed_time_ms is automatically added by EventLogger
            })
            event_logger.complete_event()

            return {
                "status": status,
                "filename": filename,
                "document_count": len(documents),
                "stored_count": stored_count,
                "skipped_count": skipped_count,
                "document_ids": doc_ids,
                "parsers_used": list(set(parser_names)),
                "extractors_applied": self._get_applied_extractors(
                    documents[0] if documents else None
                ),
                "embedder": self.embedder.__class__.__name__,
                "chunk_size": documents[0].metadata.get("chunk_size")
                if documents
                else None,
                "reason": reason,
            }

        except Exception as e:
            logger.error(f"Error ingesting file {filename}: {e}")

            # Fail event logging
            event_logger.fail_event(str(e))

            return {
                "status": "error",
                "message": str(e),
                "filename": filename,
                "document_count": 0,
            }

    def _get_applied_extractors(self, document) -> List[str]:
        """
        Get list of extractors that were applied to a document.

        Args:
            document: Document to check

        Returns:
            List of extractor names
        """
        if not document:
            return []

        extractors = []
        for key in document.metadata:
            if key.startswith("extractor_") and document.metadata[key]:
                extractor_name = key.replace("extractor_", "")
                extractors.append(extractor_name)

        return extractors

    def ingest_directory(
        self, directory_path: str, recursive: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest all files from a directory.

        This method is provided for convenience but files are still processed one-by-one.

        Args:
            directory_path: Path to the directory
            recursive: Whether to process subdirectories

        Returns:
            Dictionary with aggregated ingestion results
        """
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            return {
                "status": "error",
                "message": f"Directory not found: {directory_path}",
                "total_files": 0,
                "successful": 0,
                "failed": 0,
            }

        # Get supported extensions from blob processor
        supported_extensions = self.blob_processor.get_supported_extensions()

        # Find all matching files
        pattern = "**/*" if recursive else "*"
        files = [f for f in directory.glob(pattern) if f.is_file()]

        # Filter by supported extensions if any
        if supported_extensions:
            files = [
                f
                for f in files
                if any(f.suffix.lower() == ext.lower() for ext in supported_extensions)
            ]

        results = {
            "status": "success",
            "directory": directory_path,
            "total_files": len(files),
            "successful": 0,
            "failed": 0,
            "file_results": [],
        }

        for file_path in files:
            try:
                # Read file data
                with open(file_path, "rb") as f:
                    file_data = f.read()

                # Create metadata
                metadata = {
                    "filename": file_path.name,
                    "filepath": str(file_path),
                    "content_type": self._guess_content_type(file_path),
                    "size": file_path.stat().st_size,
                }

                # Ingest the file
                result = self.ingest_file(file_data, metadata)

                if result["status"] == "success":
                    results["successful"] += 1
                else:
                    results["failed"] += 1

                results["file_results"].append(result)

            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                results["failed"] += 1
                results["file_results"].append(
                    {"status": "error", "filename": file_path.name, "message": str(e)}
                )

        return results

    def _guess_content_type(self, file_path: Path) -> str:
        """
        Guess content type from file extension.

        Args:
            file_path: Path to the file

        Returns:
            MIME type string
        """
        import mimetypes

        content_type, _ = mimetypes.guess_type(str(file_path))
        return content_type or "application/octet-stream"

    def query(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Query the vector store for relevant documents.

        Args:
            query_text: Query string
            top_k: Number of results to return

        Returns:
            List of relevant documents with scores
        """
        # Generate query embedding
        query_embedding = self.embedder.embed(query_text)

        # Search vector store
        results = self.vector_store.search(query_embedding, top_k=top_k)

        return results
