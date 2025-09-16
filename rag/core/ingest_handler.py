"""
Ingest handler for LlamaFarm CLI integration.
Manages the flow from CLI file uploads to blob processing and vector storage.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import importlib
from rag.core.blob_processor import BlobProcessor
from rag.core.strategies.handler import SchemaHandler

logger = logging.getLogger(__name__)


class IngestHandler:
    """
    Handles document ingestion from LlamaFarm CLI.
    Coordinates blob processing, embedding, and storage.
    """
    
    def __init__(self, config_path: str, data_processing_strategy: str, database: str):
        """
        Initialize the ingest handler.
        
        Args:
            config_path: Path to the LlamaFarm configuration file
            data_processing_strategy: Name of the data processing strategy
            database: Name of the database to use
        """
        self.config_path = Path(config_path)
        self.data_processing_strategy = data_processing_strategy
        self.database = database
        
        # Initialize schema handler
        self.schema_handler = SchemaHandler(config_path)
        
        # Get configurations separately
        self.processing_config = self._get_processing_config()
        self.database_config = self._get_database_config()
        
        # Initialize components
        self.blob_processor = BlobProcessor(self.processing_config)
        self.embedder = self._initialize_embedder(self.database_config)
        self.vector_store = self._initialize_vector_store(self.database_config)
        
    def _get_processing_config(self) -> Dict[str, Any]:
        """
        Get the data processing strategy configuration.
        
        Returns:
            Dictionary with parsers and extractors
        """
        return self.schema_handler.create_processing_config(self.data_processing_strategy)
    
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
        embedder_type = embedder_config.get('type')
        
        if not embedder_type:
            raise ValueError("No embedder type specified in configuration")
        
        logger.info(f"Initializing embedder: {embedder_type} with config: {embedder_config.get('config', {})}")
        
        # Dynamically import the embedder based on type from config
        # Convert type like "OllamaEmbedder" to module path
        embedder_name_lower = embedder_type.replace('Embedder', '_embedder').lower()
        module_path = f"rag.components.embedders.{embedder_name_lower}"
        
        try:
            # Import the module
            module = importlib.import_module(module_path)
            # Get the class (should match the type name)
            embedder_class = getattr(module, embedder_type)
            # Initialize with config from the config file
            # Pass config as a dictionary, not as kwargs
            return embedder_class(config=embedder_config.get('config', {}))
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load embedder {embedder_type} from {module_path}: {e}")
            raise ValueError(f"Cannot initialize embedder {embedder_type}: {e}")
    
    def _initialize_vector_store(self, db_config: Dict[str, Any]):
        """
        Initialize the vector store from database configuration.
        
        Args:
            db_config: Database configuration
            
        Returns:
            Initialized vector store instance
        """
        # Get vector store configuration from config
        vector_store_config = self.schema_handler.get_vector_store_config(db_config)
        vector_store_type = vector_store_config.get('type')
        
        if not vector_store_type:
            raise ValueError("No vector store type specified in configuration")
        
        # Resolve persist_directory if it's relative and we have ChromaStore
        config_dict = vector_store_config.get('config', {}).copy()
        if vector_store_type == 'ChromaStore' and 'persist_directory' in config_dict:
            persist_dir = config_dict['persist_directory']
            if not persist_dir.startswith('/'):
                # Make it relative to the config file's directory (project directory)
                import os
                config_dir = os.path.dirname(self.config_path)
                config_dict['persist_directory'] = os.path.join(config_dir, persist_dir)
                print(f"[IngestHandler] Resolved persist_directory from '{persist_dir}' to: {config_dict['persist_directory']}")
                logger.info(f"Resolved persist_directory to: {config_dict['persist_directory']}")
        
        logger.info(f"Initializing vector store: {vector_store_type} with config: {config_dict}")
        
        # Dynamically import the store based on type from config
        # Convert type like "ChromaStore" to module path
        store_name_lower = vector_store_type.replace('Store', '_store').lower()
        module_path = f"rag.components.stores.{store_name_lower}"
        
        try:
            # Import the module
            module = importlib.import_module(module_path)
            # Get the class (should match the type name)
            store_class = getattr(module, vector_store_type)
            # Initialize with config from the config file
            # Use the already resolved config_dict from above
            
            # Pass config as a dictionary, not as kwargs
            return store_class(config=config_dict)
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load vector store {vector_store_type} from {module_path}: {e}")
            raise ValueError(f"Cannot initialize vector store {vector_store_type}: {e}")
    
    def ingest_file(self, file_data: bytes, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ingest a single file from the CLI.
        
        Args:
            file_data: Raw bytes of the file
            metadata: Metadata about the file (filename, content_type, etc.)
            
        Returns:
            Dictionary with ingestion results
        """
        filename = metadata.get("filename", "unknown")
        logger.info(f"Ingesting file: {filename}")
        
        try:
            # Process the blob with the blob processor
            documents = self.blob_processor.process_blob(file_data, metadata)
            
            if not documents:
                return {
                    "status": "error",
                    "message": f"No documents extracted from {filename}",
                    "filename": filename,
                    "document_count": 0
                }
            
            # Generate file hash for deduplication
            import hashlib
            file_hash = hashlib.sha256(file_data).hexdigest()
            
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
                embedded_documents.append(doc)
            
            # Store documents in vector store
            doc_ids = self.vector_store.add_documents(embedded_documents)
            
            logger.info(f"Successfully ingested {len(documents)} documents from {filename}")
            
            # Extract parser names safely
            parser_names = []
            for doc in documents:
                parser = doc.metadata.get("parser", "unknown")
                if isinstance(parser, str):
                    parser_names.append(parser)
                else:
                    parser_names.append(str(parser))
            
            return {
                "status": "success",
                "filename": filename,
                "document_count": len(documents),
                "document_ids": doc_ids,
                "parsers_used": list(set(parser_names)),
                "extractors_applied": self._get_applied_extractors(documents[0] if documents else None)
            }
            
        except Exception as e:
            logger.error(f"Error ingesting file {filename}: {e}")
            return {
                "status": "error",
                "message": str(e),
                "filename": filename,
                "document_count": 0
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
    
    def ingest_directory(self, directory_path: str, recursive: bool = True) -> Dict[str, Any]:
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
                "failed": 0
            }
        
        # Get supported extensions from blob processor
        supported_extensions = self.blob_processor.get_supported_extensions()
        
        # Find all matching files
        pattern = "**/*" if recursive else "*"
        files = [f for f in directory.glob(pattern) if f.is_file()]
        
        # Filter by supported extensions if any
        if supported_extensions:
            files = [f for f in files if any(f.suffix.lower() == ext.lower() for ext in supported_extensions)]
        
        results = {
            "status": "success",
            "directory": directory_path,
            "total_files": len(files),
            "successful": 0,
            "failed": 0,
            "file_results": []
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
                    "size": file_path.stat().st_size
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
                results["file_results"].append({
                    "status": "error",
                    "filename": file_path.name,
                    "message": str(e)
                })
        
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