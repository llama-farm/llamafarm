"""
Hugging Face dataset import router.

Provides endpoint to import datasets from Hugging Face Hub into LlamaFarm projects.
"""

from __future__ import annotations

import json
import os
from typing import Literal

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.logging import FastAPIStructLogger
from services.data_service import DataService, MetadataFileContent
from services.dataset_service import DatasetService

logger = FastAPIStructLogger()

router = APIRouter(prefix="/huggingface", tags=["huggingface"])

HF_DATASETS_SERVER = "https://datasets-server.huggingface.co"


class HFDatasetImportRequest(BaseModel):
    """Request to import a HF dataset into a project."""

    namespace: str
    project: str
    dataset: str  # target dataset name in project
    hf_dataset_id: str  # e.g., "squad", "username/dataset-name"
    config: str = "default"
    split: str = "train"
    max_rows: int = Field(default=100, le=100)  # HF datasets-server API limit
    format: Literal["jsonl", "csv"] = "jsonl"
    data_processing_strategy: str
    database: str


class HFDatasetImportResponse(BaseModel):
    """Response from HF dataset import."""

    project: str
    namespace: str
    dataset: str
    file_count: int
    row_count: int
    task_id: str | None = None


def _add_file_from_bytes(
    namespace: str,
    project: str,
    dataset: str,
    file_data: bytes,
    filename: str,
) -> MetadataFileContent:
    """Add a file to a dataset from bytes."""
    import time

    data_dir = DataService.ensure_data_dir(namespace, project, dataset)
    data_hash = DataService.hash_data(file_data)
    resolved_file_name = DataService.append_collision_timestamp(filename)

    # Write metadata
    meta_path = os.path.join(data_dir, "meta", f"{data_hash}.json")
    meta = MetadataFileContent(
        original_file_name=filename,
        resolved_file_name=resolved_file_name,
        timestamp=float(time.time()),
        size=len(file_data),
        mime_type="application/jsonlines"
        if filename.endswith(".jsonl")
        else "text/csv",
        hash=data_hash,
    )
    with open(meta_path, "w") as f:
        f.write(meta.model_dump_json())

    # Write raw file
    raw_path = os.path.join(data_dir, "raw", data_hash)
    with open(raw_path, "wb") as f:
        f.write(file_data)

    # Create index symlink
    index_dir = os.path.join(data_dir, "index", "by_name")
    os.makedirs(index_dir, exist_ok=True)
    symlink_path = os.path.join(index_dir, resolved_file_name)
    if not os.path.exists(symlink_path):
        os.symlink(raw_path, symlink_path)

    return meta


@router.post("/datasets/import", response_model=HFDatasetImportResponse)
async def import_hf_dataset(request: HFDatasetImportRequest) -> HFDatasetImportResponse:
    """
    Import rows from a Hugging Face dataset into a project dataset.

    1. Fetches rows from HF datasets-server API
    2. Converts to JSONL file
    3. Creates dataset if needed
    4. Adds file to dataset
    5. Triggers ingestion
    """
    # Resolve 'default' strategy and database to actual values from project config
    data_processing_strategy = request.data_processing_strategy
    database = request.database

    if data_processing_strategy == "default" or database == "default":
        supported_strategies = DatasetService.get_supported_data_processing_strategies(
            request.namespace, request.project
        )
        supported_databases = DatasetService.get_supported_databases(
            request.namespace, request.project
        )

        if data_processing_strategy == "default":
            if not supported_strategies:
                raise HTTPException(
                    status_code=400,
                    detail="No data processing strategies configured for this project",
                )
            data_processing_strategy = supported_strategies[0]

        if database == "default":
            if not supported_databases:
                raise HTTPException(
                    status_code=400,
                    detail="No databases configured for this project",
                )
            database = supported_databases[0]

    logger.info(
        "Importing HF dataset",
        hf_dataset_id=request.hf_dataset_id,
        namespace=request.namespace,
        project=request.project,
        dataset=request.dataset,
        data_processing_strategy=data_processing_strategy,
        database=database,
    )

    # 1. Get valid configs if using 'default' - HF doesn't accept 'default' as a config name
    config_to_use = request.config
    split_to_use = request.split

    if config_to_use == "default":
        # Fetch actual available configs from HF
        async with httpx.AsyncClient(timeout=30.0) as client:
            splits_url = f"{HF_DATASETS_SERVER}/splits?dataset={request.hf_dataset_id}"
            splits_response = await client.get(splits_url)
            if splits_response.status_code == 200:
                splits_data = splits_response.json()
                available_splits = splits_data.get("splits", [])
                if available_splits:
                    # Use the first available config
                    config_to_use = available_splits[0].get("config", "default")
                    # Prefer 'train' split if available, otherwise use first available
                    available_split_names = [
                        s.get("split")
                        for s in available_splits
                        if s.get("config") == config_to_use
                    ]
                    if "train" in available_split_names:
                        split_to_use = "train"
                    elif available_split_names:
                        split_to_use = available_split_names[0]
                    logger.info(
                        "Resolved HF config/split",
                        original_config=request.config,
                        resolved_config=config_to_use,
                        resolved_split=split_to_use,
                    )

    # 2. Fetch rows from HF datasets-server
    hf_url = (
        f"{HF_DATASETS_SERVER}/rows"
        f"?dataset={request.hf_dataset_id}"
        f"&config={config_to_use}"
        f"&split={split_to_use}"
        f"&offset=0"
        f"&length={request.max_rows}"
    )

    logger.info("Fetching rows from HF", url=hf_url)

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(hf_url)
        if response.status_code == 401:
            raise HTTPException(
                status_code=400,
                detail="Dataset requires authentication and cannot be imported",
            )
        if response.status_code != 200:
            # Try to extract a meaningful error message
            error_detail = f"Failed to fetch dataset from Hugging Face (HTTP {response.status_code})"
            try:
                error_data = response.json()
                if "error" in error_data:
                    hf_error = error_data["error"]
                    if "size limit exceeded" in hf_error.lower():
                        error_detail = "This dataset is too large for the Hugging Face API. Please try a smaller dataset."
                    elif "not found" in hf_error.lower():
                        error_detail = f"Dataset config or split not found: {config_to_use}/{split_to_use}"
                    else:
                        error_detail = f"Hugging Face error: {hf_error[:200]}"
            except Exception:
                pass

            logger.error(
                "HF API error",
                status_code=response.status_code,
                response_text=response.text[:500],
                url=hf_url,
            )
            raise HTTPException(
                status_code=502,
                detail=error_detail,
            )

        data = response.json()

    rows = data.get("rows", [])
    if not rows:
        raise HTTPException(status_code=400, detail="No rows returned from dataset")

    # 2. Extract actual row data (HF returns {row_idx, row, truncated_cells})
    row_data = [r["row"] for r in rows]

    # 3. Convert rows to file format
    safe_name = request.hf_dataset_id.replace("/", "_")
    filename = f"hf_{safe_name}_{request.split}.{request.format}"

    if request.format == "jsonl":
        file_content = "\n".join(json.dumps(row) for row in row_data)
    else:  # csv
        import csv
        import io

        output = io.StringIO()
        if row_data:
            writer = csv.DictWriter(output, fieldnames=row_data[0].keys())
            writer.writeheader()
            writer.writerows(row_data)
        file_content = output.getvalue()

    file_bytes = file_content.encode("utf-8")

    # 4. Create dataset if it doesn't exist
    try:
        DatasetService.create_dataset(
            namespace=request.namespace,
            project=request.project,
            name=request.dataset,
            data_processing_strategy=data_processing_strategy,
            database=database,
        )
        logger.info("Created dataset", dataset=request.dataset)
    except ValueError as e:
        if "already exists" not in str(e).lower():
            raise HTTPException(status_code=400, detail=str(e)) from e
        logger.info("Dataset already exists", dataset=request.dataset)

    # 5. Add file to dataset
    logger.info(
        "Adding file to dataset",
        dataset=request.dataset,
        filename=filename,
        file_size=len(file_bytes),
        row_count=len(row_data),
    )
    meta = _add_file_from_bytes(
        namespace=request.namespace,
        project=request.project,
        dataset=request.dataset,
        file_data=file_bytes,
        filename=filename,
    )
    logger.info(
        "File added successfully",
        dataset=request.dataset,
        file_hash=meta.hash,
    )

    # 6. Start ingestion
    task_id: str | None = None
    try:
        launch = DatasetService.start_dataset_ingestion(
            request.namespace, request.project, request.dataset
        )
        task_id = launch.task_id
        logger.info("Started ingestion", task_id=task_id)
    except Exception as e:
        logger.warning("Failed to start ingestion", error=str(e))
        # Don't fail the import if ingestion fails to start

    return HFDatasetImportResponse(
        project=request.project,
        namespace=request.namespace,
        dataset=request.dataset,
        file_count=1,
        row_count=len(row_data),
        task_id=task_id,
    )
