import json
from typing import Dict, Any, List
from pathlib import Path

from core.celery import app
from core.logging import FastAPIStructLogger
from services.project_service import ProjectService
from config.datamodel import LlamaFarmConfig

from rag.core.ingest_handler import IngestHandler
from rag.api import DatabaseSearchAPI

logger = FastAPIStructLogger(__name__)

@app.task(bind=True, name='rag.query')
def rag_query_task(self, namespace: str, project: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        logger.info("Processing RAG query task", namespace=namespace, project=project)

        project_obj = ProjectService.get_project(namespace, project)
        project_dir = ProjectService.get_project_dir(namespace, project)

        query = request_data.get('query', '')
        database = request_data.get('database')
        retrieval_strategy = request_data.get('retrieval_strategy')
        top_k = request_data.get('top_k', 5)
        score_threshold = request_data.get('score_threshold')
        metadata_filters = request_data.get('metadata_filters')

        search_api = DatabaseSearchAPI(
            config_path=str(project_dir / "llamafarm.yaml"),
            database=database
        )

        results = search_api.search(
            query=query,
            top_k=top_k,
            min_score=score_threshold,
            metadata_filter=metadata_filters,
            retrieval_strategy=retrieval_strategy
        )

        result_dicts = []
        for result in results:
            result_dicts.append({
                'id': result.id,
                'content': result.content,
                'score': result.score,
                'metadata': result.metadata,
                'source': result.source
            })

        response = {
            'query': query,
            'results': result_dicts,
            'total_results': len(result_dicts),
            'retrieval_strategy_used': retrieval_strategy,
            'database_used': database
        }

        logger.info("RAG query completed successfully", result_count=len(result_dicts))
        return response

    except Exception as e:
        logger.error("RAG query task failed", error=str(e), exc_info=True)
        raise


@app.task(bind=True, name='rag.ingest')
def rag_ingest_task(self, namespace: str, project: str, dataset: str, files: List[str]) -> Dict[str, Any]:
    try:
        logger.info("Processing RAG ingest task", namespace=namespace, project=project, dataset=dataset, file_count=len(files))

        project_obj = ProjectService.get_project(namespace, project)
        project_dir = ProjectService.get_project_dir(namespace, project)

        dataset_config = None
        if project_obj.config.datasets:
            for ds in project_obj.config.datasets:
                if ds.name == dataset:
                    dataset_config = ds
                    break

        if not dataset_config:
            raise ValueError(f"Dataset {dataset} not found")

        handler = IngestHandler(
            config_path=str(project_dir / "llamafarm.yaml"),
            data_processing_strategy=dataset_config.data_processing_strategy,
            database=dataset_config.database,
            dataset_name=dataset
        )

        total_processed = 0
        total_stored = 0
        total_skipped = 0
        file_results = []

        for file_path in files:
            try:
                with open(file_path, 'rb') as f:
                    file_data = f.read()

                file_path_obj = Path(file_path)

                if 'lf_data/raw' in str(file_path):
                    file_hash = file_path_obj.name
                    meta_dir = file_path_obj.parent.parent / 'meta'
                    meta_file = meta_dir / f"{file_hash}.json"

                    if meta_file.exists():
                        with open(meta_file, 'r') as mf:
                            meta_content = json.load(mf)
                            original_filename = meta_content.get('original_file_name', file_hash)
                            mime_type = meta_content.get('mime_type', 'application/octet-stream')
                    else:
                        original_filename = file_hash
                        mime_type = 'application/octet-stream'
                else:
                    original_filename = file_path_obj.name
                    mime_type = 'application/octet-stream'

                metadata = {
                    'filename': original_filename,
                    'filepath': str(file_path_obj),
                    'size': len(file_data),
                    'content_type': mime_type
                }

                result = handler.ingest_file(file_data=file_data, metadata=metadata)

                file_results.append({
                    'filename': original_filename,
                    'status': result.get('status'),
                    'stored_count': result.get('stored_count', 0),
                    'skipped_count': result.get('skipped_count', 0),
                    'error': result.get('message') if result.get('status') == 'error' else None
                })

                total_processed += 1
                total_stored += result.get('stored_count', 0)
                total_skipped += result.get('skipped_count', 0)

                self.update_state(
                    meta={
                        'processed_files': total_processed,
                        'total_files': len(files),
                        'stored_count': total_stored,
                        'skipped_count': total_skipped
                    }
                )

            except Exception as e:
                logger.error(f"Failed to ingest file {file_path}", error=str(e))
                file_results.append({
                    'filename': Path(file_path).name,
                    'status': 'error',
                    'stored_count': 0,
                    'skipped_count': 0,
                    'error': str(e)
                })

        response = {
            'message': f'Successfully processed {total_processed} files',
            'total_processed': total_processed,
            'total_stored': total_stored,
            'total_skipped': total_skipped,
            'file_results': file_results
        }

        logger.info("RAG ingest task completed", **response)
        return response

    except Exception as e:
        logger.error("RAG ingest task failed", error=str(e), exc_info=True)
        raise


@app.task(bind=True, name='rag.health_check')
def rag_health_check_task(self) -> Dict[str, Any]:
    try:
        return {
            'status': 'healthy',
            'message': 'RAG worker is responding',
            'worker_type': 'celery'
        }
    except Exception as e:
        logger.error("RAG health check failed", error=str(e))
        return {
            'status': 'unhealthy',
            'message': f'RAG worker health check failed: {str(e)}',
            'worker_type': 'celery'
        }
