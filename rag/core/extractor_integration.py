"""Integration between parsers and extractors."""

from typing import Dict, Any, List, Optional
from pathlib import Path

from core.base import Document, ProcessingResult
from components.extractors.base import ExtractorPipeline, create_pipeline_from_config
from components.extractors import registry

from core.logging import RAGStructLogger
logger = RAGStructLogger("rag.core.extractor_integration")


class ExtractorIntegrator:
    """Integrates extractors with parsers for enhanced document processing."""
    
    def __init__(self, extractor_config: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize extractor integrator.
        
        Args:
            extractor_config: List of extractor configurations
        """
        self.extractor_config = extractor_config or []
        self.pipeline = None
        self._setup_pipeline()
    
    def _setup_pipeline(self) -> None:
        """Setup extractor pipeline from configuration."""
        if not self.extractor_config:
            self.pipeline = None
            return
        
        try:
            self.pipeline = create_pipeline_from_config(self.extractor_config, registry)
            logger.info(f"Created extractor pipeline with {len(self.pipeline.extractors)} extractors")
        except Exception as e:
            logger.error(f"Failed to create extractor pipeline: {e}")
            self.pipeline = None
    
    def has_extractors(self) -> bool:
        """Check if any extractors are configured."""
        return self.pipeline is not None and len(self.pipeline.extractors) > 0
    
    def process_documents(
        self, 
        documents: List[Document], 
        file_path: Optional[Path] = None
    ) -> List[Document]:
        """
        Process documents through extractor pipeline.
        
        Args:
            documents: List of documents to process
            file_path: Optional source file path for context
            
        Returns:
            Enhanced documents with extracted metadata
        """
        if not self.has_extractors():
            return documents
        
        try:
            # Add file path context to documents if available
            if file_path:
                for doc in documents:
                    if "source_file" not in doc.metadata:
                        doc.metadata["source_file"] = str(file_path)
                        doc.metadata["source_filename"] = file_path.name
            
            # Run extractor pipeline
            enhanced_documents = self.pipeline.run(documents)
            
            logger.debug(f"Enhanced {len(enhanced_documents)} documents with extractors")
            return enhanced_documents
            
        except Exception as e:
            logger.error(f"Extractor processing failed: {e}")
            return documents  # Return original documents on failure
    
    def get_extractor_info(self) -> Dict[str, Any]:
        """Get information about configured extractors."""
        if not self.pipeline:
            return {"extractors": [], "total_extractors": 0}
        
        return self.pipeline.get_pipeline_info()


def create_extractor_integrator(config: Dict[str, Any]) -> Optional[ExtractorIntegrator]:
    """
    Create extractor integrator from parser config.
    
    Args:
        config: Parser configuration that may contain extractor settings
        
    Returns:
        ExtractorIntegrator instance or None if no extractors configured
    """
    extractor_config = config.get("extractors", [])
    
    if not extractor_config:
        return None
    
    return ExtractorIntegrator(extractor_config)


def enhance_processing_result(
    result: ProcessingResult, 
    integrator: Optional[ExtractorIntegrator],
    file_path: Optional[Path] = None
) -> ProcessingResult:
    """
    Enhance processing result with extractor metadata.
    
    Args:
        result: Original processing result
        integrator: Extractor integrator instance
        file_path: Source file path
        
    Returns:
        Enhanced processing result
    """
    if not integrator or not integrator.has_extractors():
        return result
    
    try:
        # Process documents through extractors
        enhanced_documents = integrator.process_documents(result.documents, file_path)
        
        # Update result
        result.documents = enhanced_documents
        
        # Add extractor info to metrics
        if "extractors" not in result.metrics:
            result.metrics["extractors"] = {}
        
        result.metrics["extractors"]["pipeline_info"] = integrator.get_extractor_info()
        result.metrics["extractors"]["applied"] = True
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to enhance processing result: {e}")
        return result


def apply_extractors_from_cli_args(
    documents: List[Document], 
    extractor_names: List[str],
    extractor_configs: Optional[Dict[str, Dict[str, Any]]] = None
) -> List[Document]:
    """
    Apply extractors specified from CLI arguments.
    
    Args:
        documents: Documents to process
        extractor_names: Names of extractors to apply
        extractor_configs: Optional configurations for extractors
        
    Returns:
        Enhanced documents
    """
    if not extractor_names:
        return documents
    
    extractors = []
    extractor_configs = extractor_configs or {}
    
    for extractor_name in extractor_names:
        config = extractor_configs.get(extractor_name, {})
        extractor = registry.create(extractor_name, config)
        
        if extractor:
            extractors.append(extractor)
            logger.info(f"Added CLI extractor: {extractor_name}")
        else:
            logger.warning(f"Failed to create CLI extractor: {extractor_name}")
    
    if not extractors:
        logger.warning("No valid extractors created from CLI arguments")
        return documents
    
    # Create and run pipeline
    pipeline = ExtractorPipeline(extractors)
    return pipeline.run(documents)