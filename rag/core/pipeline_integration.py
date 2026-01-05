"""Pipeline Integration - Connects RAG pipeline to UnifiedDatasetStore.

Phase 23: RAG Pipeline Integration

Provides integration between:
- RAG document processing pipeline
- Entity extraction for graph population
- UnifiedDatasetStore for multi-store persistence
"""

import logging
from typing import Any

from core.base import Document, Pipeline, ProcessingResult

logger = logging.getLogger(__name__)


class DatasetIntegratedPipeline(Pipeline):
    """Pipeline that integrates with UnifiedDatasetStore for multi-store persistence.

    This pipeline extends the base Pipeline to automatically:
    1. Extract entities from documents and populate graph store
    2. Store documents in vector store (via existing ChromaDB integration)
    3. Create cross-store links via LinkageTable
    """

    def __init__(
        self,
        name: str = "Dataset Integrated Pipeline",
        dataset_store: Any = None,
        config: dict[str, Any] | None = None,
    ):
        """Initialize the integrated pipeline.

        Args:
            name: Pipeline name
            dataset_store: UnifiedDatasetStore instance
            config: Pipeline configuration
        """
        super().__init__(name)
        self.dataset_store = dataset_store
        self.config = config or {}

        # Entity extraction configuration
        self.extract_entities = self.config.get("extract_entities", True)
        self.entity_types = self.config.get("entity_types", None)  # Use defaults
        self.extract_relationships = self.config.get("extract_relationships", False)

        # Initialize entity extractor if needed
        self._entity_extractor = None
        if self.extract_entities:
            self._init_entity_extractor()

    def _init_entity_extractor(self):
        """Initialize the entity extractor."""
        try:
            from components.extractors.entity_extractor import EntityExtractor

            extractor_config = {
                "entity_types": self.entity_types,
                "use_fallback": True,
                "extract_relationships": self.extract_relationships,
            }

            self._entity_extractor = EntityExtractor(
                name="PipelineEntityExtractor",
                config=extractor_config,
            )
            logger.info("Entity extractor initialized for pipeline integration")

        except ImportError as e:
            logger.warning(f"Could not import EntityExtractor: {e}")
            self._entity_extractor = None

    def process_with_dataset(
        self,
        documents: list[Document],
        store_in_vector: bool = True,
        store_in_graph: bool = True,
    ) -> ProcessingResult:
        """Process documents through pipeline and store in dataset.

        Args:
            documents: Documents to process
            store_in_vector: Whether to store in vector store
            store_in_graph: Whether to extract entities to graph

        Returns:
            ProcessingResult with processed documents and any errors
        """
        errors = []

        # Run base pipeline processing
        if self.components:
            logger.info(f"Processing {len(documents)} documents through pipeline")
            result = self.run(documents=documents)
            documents = result.documents
            errors.extend(result.errors)

        # Extract entities to graph store
        if store_in_graph and self.dataset_store and self._entity_extractor:
            logger.info("Extracting entities to graph store")
            graph_result = self._extract_entities_to_graph(documents)
            if graph_result.get("errors"):
                errors.extend(graph_result["errors"])

        # Store in vector store (via existing store component or direct)
        if store_in_vector and self.dataset_store:
            logger.info("Storing documents in vector store")
            vector_result = self._store_in_vector(documents)
            if vector_result.get("errors"):
                errors.extend(vector_result["errors"])

        return ProcessingResult(documents=documents, errors=errors)

    def _extract_entities_to_graph(self, documents: list[Document]) -> dict[str, Any]:
        """Extract entities from documents and add to graph store.

        Args:
            documents: Documents to extract entities from

        Returns:
            Dictionary with extraction results
        """
        if not self.dataset_store or not self.dataset_store.graph_store:
            return {"entities_extracted": 0, "errors": []}

        total_entities = 0
        total_relationships = 0
        errors = []

        for doc in documents:
            try:
                # Extract entities
                entities = self._entity_extractor.extract_entities(doc)

                # Add to graph store
                for entity in entities:
                    node_id = self.dataset_store.add_node(
                        name=entity.name,
                        node_type=entity.entity_type.lower(),
                        node_id=entity.entity_id,
                        properties={
                            "source_doc": doc.id,
                            "span_start": entity.span_start,
                            "span_end": entity.span_end,
                            "confidence": entity.confidence,
                            "method": entity.method,
                        },
                    )

                    if node_id:
                        # Link to source document
                        self.dataset_store.linkage_table.link(
                            concept_uuid=doc.id,
                            graph_node_id=node_id,
                        )
                        total_entities += 1

                # Extract relationships if enabled
                if self.extract_relationships and len(entities) >= 2:
                    relationships = self._entity_extractor.extract_relationships_llm(
                        doc, entities
                    )

                    for rel in relationships:
                        # Find entity IDs
                        source_entity = next(
                            (e for e in entities if e.name == rel.source_entity), None
                        )
                        target_entity = next(
                            (e for e in entities if e.name == rel.target_entity), None
                        )

                        if source_entity and target_entity:
                            edge_id = self.dataset_store.add_edge(
                                source_id=source_entity.entity_id,
                                target_id=target_entity.entity_id,
                                relationship=rel.relationship_type,
                                properties={"source_doc": doc.id},
                            )
                            if edge_id:
                                total_relationships += 1

            except Exception as e:
                logger.error(f"Error extracting entities from doc {doc.id}: {e}")
                errors.append(
                    {
                        "document_id": doc.id,
                        "error": str(e),
                        "stage": "entity_extraction",
                    }
                )

        logger.info(
            f"Extracted {total_entities} entities and {total_relationships} relationships"
        )

        return {
            "entities_extracted": total_entities,
            "relationships_extracted": total_relationships,
            "errors": errors,
        }

    def _store_in_vector(self, documents: list[Document]) -> dict[str, Any]:
        """Store documents in vector store.

        Args:
            documents: Documents to store

        Returns:
            Dictionary with storage results
        """
        # Vector store integration is handled by the existing pipeline components
        # This method is a hook for future direct integration
        return {"stored": len(documents), "errors": []}


def create_integrated_pipeline(
    project_path: str,
    dataset_name: str,
    dataset_type: str = "knowledge",
    components: list | None = None,
    config: dict[str, Any] | None = None,
) -> DatasetIntegratedPipeline:
    """Create an integrated pipeline with UnifiedDatasetStore.

    Args:
        project_path: Path to project directory
        dataset_name: Name of the dataset
        dataset_type: Type of dataset (knowledge, realtime, etc.)
        components: Pipeline components to use
        config: Pipeline configuration

    Returns:
        Configured DatasetIntegratedPipeline instance
    """
    from core.unified_store import UnifiedDatasetStore

    # Create dataset store
    store = UnifiedDatasetStore(
        dataset_config={"name": dataset_name, "type": dataset_type},
        project_dir=project_path,
    )

    # Create pipeline
    pipeline = DatasetIntegratedPipeline(
        name=f"{dataset_name} Pipeline",
        dataset_store=store,
        config=config,
    )

    # Add components if provided
    if components:
        for component in components:
            pipeline.add_component(component)

    return pipeline


def process_documents_to_dataset(
    documents: list[Document],
    project_path: str,
    dataset_name: str,
    dataset_type: str = "knowledge",
    extract_entities: bool = True,
) -> dict[str, Any]:
    """Convenience function to process documents into a dataset.

    Args:
        documents: Documents to process
        project_path: Path to project directory
        dataset_name: Name of the dataset
        dataset_type: Type of dataset
        extract_entities: Whether to extract entities

    Returns:
        Dictionary with processing results
    """
    config = {
        "extract_entities": extract_entities,
    }

    pipeline = create_integrated_pipeline(
        project_path=project_path,
        dataset_name=dataset_name,
        dataset_type=dataset_type,
        config=config,
    )

    result = pipeline.process_with_dataset(
        documents=documents,
        store_in_graph=extract_entities,
    )

    # Get stats
    stats = pipeline.dataset_store.get_stats()

    # Cleanup
    pipeline.dataset_store.close()

    return {
        "documents_processed": len(result.documents),
        "errors": result.errors,
        "stats": stats,
    }
