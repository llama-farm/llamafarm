
from core.base import Pipeline

class RAGService:
    """
    Service for running RAG (Retrieval-Augmented Generation) pipelines
    based on project and dataset configuration.
    """

    @classmethod
    def _get_parser_from_config(cls, rag_cfg: dict, parser_name: str):
        """
        Get and instantiate a parser from configuration.

        Args:
            rag_cfg (dict): The RAG configuration section
            parser_name (str): Name of the parser to instantiate

        Returns:
            The instantiated parser object
        """
        parser_cfg = rag_cfg.get("parsers", {}).get(parser_name)
        if not parser_cfg:
            raise ValueError(f"Parser '{parser_name}' not found in config.")

        parser_type = parser_cfg["type"]
        parser_kwargs = parser_cfg.get("config", {})

        # Dynamically import parser class
        if parser_type == "CustomerSupportCSVParser":
            from parsers.csv_parser import CustomerSupportCSVParser
            return CustomerSupportCSVParser(**parser_kwargs)
        else:
            raise ValueError(f"Unsupported parser type: {parser_type}")

    @classmethod
    def _get_embedder_from_config(cls, rag_cfg: dict, embedder_name: str):
        """
        Get and instantiate an embedder from configuration.

        Args:
            rag_cfg (dict): The RAG configuration section
            embedder_name (str): Name of the embedder to instantiate

        Returns:
            The instantiated embedder object
        """
        embedder_cfg = rag_cfg.get("embedders", {}).get(embedder_name)
        if not embedder_cfg:
            raise ValueError(f"Embedder '{embedder_name}' not found in config.")

        embedder_type = embedder_cfg["type"]
        embedder_kwargs = embedder_cfg.get("config", {})

        if embedder_type == "OllamaEmbedder":
            from embedders.ollama_embedder import OllamaEmbedder
            return OllamaEmbedder(**embedder_kwargs)
        else:
            raise ValueError(f"Unsupported embedder type: {embedder_type}")

    @classmethod
    def _get_vector_store_from_config(cls, rag_cfg: dict, vector_store_name: str):
        """
        Get and instantiate a vector store from configuration.

        Args:
            rag_cfg (dict): The RAG configuration section
            vector_store_name (str): Name of the vector store to instantiate

        Returns:
            The instantiated vector store object
        """
        store_cfg = rag_cfg.get("vector_stores", {}).get(vector_store_name)
        if not store_cfg:
            raise ValueError(f"Vector store '{vector_store_name}' not found in config.")

        store_type = store_cfg["type"]
        store_kwargs = store_cfg.get("config", {})

        if store_type == "ChromaStore":
            from stores.chroma_store import ChromaStore
            return ChromaStore(**store_kwargs)
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")

    @classmethod
    def ingest_dataset(cls, project_config: dict, dataset_config: dict):
        """
        Ingest data for a dataset using the RAG pipeline defined in the config.

        Args:
            project_config (dict): The loaded project configuration (from config/schema.yaml).
            dataset_config (dict): The dataset configuration (from config["datasets"][...]).
        Returns:
            The result of the pipeline run.
        """
        # Get RAG config section
        rag_cfg = project_config.get("rag", {})
        defaults = rag_cfg.get("defaults", {})
        # Get dataset pipeline components (parser, embedder, vector_store, etc.)
        parser_name = dataset_config.get("parser", defaults.get("parser"))
        embedder_name = dataset_config.get("embedder", defaults.get("embedder"))
        vector_store_name = dataset_config.get("vector_store", defaults.get("vector_store"))

        # Validate defaults exist in config when used
        if not parser_name:
            raise ValueError("No parser specified and no default parser configured")
        if not embedder_name:
            raise ValueError("No embedder specified and no default embedder configured")
        if not vector_store_name:
            raise ValueError("No vector_store specified and no default vector_store configured")
        # retrieval_strategy_name = dataset_config.get("retrieval_strategy", rag_cfg.get("defaults", {}).get("retrieval_strategy", "default"))

        # --- Instantiate components using helper methods ---
        parser = cls._get_parser_from_config(rag_cfg, parser_name)
        embedder = cls._get_embedder_from_config(rag_cfg, embedder_name)
        vector_store = cls._get_vector_store_from_config(rag_cfg, vector_store_name)

        # Optionally, handle retrieval strategy if needed (not used in ingestion)

        # --- Build pipeline ---
        pipeline = Pipeline(f"{project_config.get('name', 'RAG Pipeline')}: {dataset_config.get('name', '')}")
        pipeline.add_component(parser)
        pipeline.add_component(embedder)
        pipeline.add_component(vector_store)

        # --- Run pipeline for each file in dataset ---
        import logging
        results = []
        for file_path in dataset_config.get("files", []):
            try:
                result = pipeline.run(source=file_path)
                results.append(result)
            except Exception as e:
                logging.error(f"Failed to process file '{file_path}': {e}")
                results.append({"file": file_path, "error": str(e)})
        return results

    # --- Retrieval Example Usage ---

    # Example: Quick search using the pipeline's vector store
    def retrieve(self, project_config: dict, dataset_config: dict, query: str, top_k=3, min_score=None, metadata_filter=None, return_raw_documents=False):
        """
        Perform a retrieval/search using the vector store and embedder.

        Args:
            project_config (dict): The loaded project configuration (from config/schema.yaml).
            dataset_config (dict): The dataset configuration (from config["datasets"][...]).
            query (str): The search query.
            top_k (int): Number of top results to return.
            min_score (float, optional): Minimum score threshold.
            metadata_filter (dict, optional): Metadata filter for results.
            return_raw_documents (bool): If True, return raw Document objects.

        Returns:
            List of results or Document objects.
        """
        # Get RAG config section and component names
        rag_cfg = project_config.get("rag", {})
        embedder_name = dataset_config.get("embedder", rag_cfg.get("defaults", {}).get("embedder", "default"))
        vector_store_name = dataset_config.get("vector_store", rag_cfg.get("defaults", {}).get("vector_store", "default"))

        # Instantiate embedder and vector store using helper methods
        embedder = self._get_embedder_from_config(rag_cfg, embedder_name)
        vector_store = self._get_vector_store_from_config(rag_cfg, vector_store_name)

        # Embed the query
        query_embedding = embedder.embed([query])[0]

        # Search in the vector store
        search_kwargs = {
            "query_embedding": query_embedding,
            "top_k": top_k,
        }
        if min_score is not None:
            search_kwargs["min_score"] = min_score
        if metadata_filter is not None:
            search_kwargs["metadata_filter"] = metadata_filter
        if return_raw_documents:
            search_kwargs["return_raw_documents"] = True

        results = vector_store.search(**search_kwargs)
        return results

        # Example usage:
        # rag_service = RAGService()
        # results = rag_service.retrieve(project_config, dataset_config, "password reset", top_k=3)
        # for result in results:
        #     print(f"Score: {result.score:.3f} - {result.content[:100]}...")

        # Note: Removed unused inner function get_collection_info
