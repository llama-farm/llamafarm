from __future__ import annotations

import json
import os
from glob import glob
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.logging import FastAPIStructLogger
from config.datamodel import LlamaFarmConfig
from services.project_service import ProjectService
from services.dataset_service import DatasetService
from services.data_service import DataService, MetadataFileContent

logger = FastAPIStructLogger()

router = APIRouter(prefix="/examples", tags=["examples"])


class ExampleSummary(BaseModel):
    id: str
    slug: str | None = None
    title: str
    description: str | None = None
    primaryModel: str | None = None
    tags: list[str] = Field(default_factory=list)
    dataset_count: int | None = None
    data_size_bytes: int | None = None
    data_size_human: str | None = None
    project_size_bytes: int | None = None
    project_size_human: str | None = None
    updated_at: str | None = None


class ExampleDataset(BaseModel):
    """Dataset defined by an example manifest, with computed file sizes."""

    example_id: str
    example_title: str | None = None
    name: str
    strategy: str | None = None
    database: str | None = None
    kind: str | None = None
    ingest: list[str] = Field(default_factory=list)
    size_bytes: int | None = None
    size_human: str | None = None


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _examples_root() -> str:
    return os.path.join(_repo_root(), "examples")


def _scan_manifests() -> list[dict[str, Any]]:
    def _humanize_size(num_bytes: int) -> str:
        units = ["B", "KB", "MB", "GB", "TB"]
        size = float(num_bytes)
        unit_idx = 0
        while size >= 1024 and unit_idx < len(units) - 1:
            size /= 1024.0
            unit_idx += 1
        if unit_idx == 0:
            return f"{int(size)}{units[unit_idx]}"
        return f"{size:.1f}{units[unit_idx]}"

    manifests: list[dict[str, Any]] = []
    root = _examples_root()
    # Primary path: directories with manifest.yaml
    for manifest_path in glob(os.path.join(root, "*", "manifest.yaml")):
        try:
            import yaml

            with open(manifest_path) as f:
                m = yaml.safe_load(f) or {}
            if not m.get("id") or not m.get("title"):
                continue
            base_dir = os.path.dirname(manifest_path)
            ds_list = m.get("datasets") or []

            # Compute data size by summing files matched by ingest globs
            data_bytes = 0
            for ds in ds_list or []:
                for pattern in ds.get("ingest", []) or []:
                    abs_glob = _ensure_path_under_examples(
                        os.path.join(_repo_root(), pattern)
                    )
                    for path in glob(abs_glob):
                        try:
                            data_bytes += os.path.getsize(path)
                        except OSError:
                            pass

            # Project size approximation: manifest + referenced cfg
            proj_bytes = 0
            try:
                proj_bytes += os.path.getsize(manifest_path)
            except OSError:
                pass
            cfg_path = m.get("config", {}).get("yaml_path")
            if cfg_path:
                cfg_abs = _ensure_path_under_examples(
                    os.path.join(_repo_root(), cfg_path)
                )
                try:
                    proj_bytes += os.path.getsize(cfg_abs)
                except OSError:
                    pass

            # Updated at: directory mtime
            try:
                updated_at = None
                dir_mtime = os.path.getmtime(base_dir)
                import datetime as _dt

                updated_at = _dt.datetime.fromtimestamp(dir_mtime).isoformat()
            except OSError:
                updated_at = None

            manifests.append(
                {
                    "id": m.get("id"),
                    "slug": m.get("slug"),
                    "title": m.get("title"),
                    "description": m.get("description"),
                    "tags": m.get("tags", []),
                    "primaryModel": m.get("primaryModel"),
                    "dataset_count": len(ds_list)
                    if isinstance(ds_list, list)
                    else None,
                    "data_size_bytes": data_bytes,
                    "data_size_human": _humanize_size(data_bytes)
                    if data_bytes
                    else None,
                    "project_size_bytes": proj_bytes,
                    "project_size_human": _humanize_size(proj_bytes)
                    if proj_bytes
                    else None,
                    "updated_at": updated_at,
                }
            )
        except Exception as e:
            logger.warning(
                "Failed to read example manifest", path=manifest_path, error=str(e)
            )

    # Fallback: directories without manifest but with files/ or llamafarm.yaml
    for candidate in glob(os.path.join(root, "*")):
        if not os.path.isdir(candidate):
            continue
        manifest_file = os.path.join(candidate, "manifest.yaml")
        if os.path.exists(manifest_file):
            continue  # already handled
        # Identify as example if it has a files directory or a llamafarm config
        has_files = os.path.isdir(os.path.join(candidate, "files"))
        cfg_guess = None
        for name in ["llamafarm.yaml", "llamafarm-example.yaml"]:
            p = os.path.join(candidate, name)
            if os.path.exists(p):
                cfg_guess = p
                break
        if not has_files and not cfg_guess:
            continue

        dir_name = os.path.basename(candidate)
        # Title-case the dir name as a simple title
        title = dir_name.replace("-", " ").replace("_", " ").title()

        # Try to infer primary model from config if present
        inferred_model = None
        if cfg_guess:
            try:
                import yaml as _yaml

                with open(cfg_guess) as f:
                    cfg = _yaml.safe_load(f) or {}
                runtime = cfg.get("runtime") or {}
                inferred_model = runtime.get("model")
            except Exception:
                inferred_model = None

        # Data size sum of files under files/
        data_bytes = 0
        files_dir = os.path.join(candidate, "files")
        if os.path.isdir(files_dir):
            for root_dir, _dirs, files in os.walk(files_dir):
                for fn in files:
                    fp = os.path.join(root_dir, fn)
                    try:
                        data_bytes += os.path.getsize(fp)
                    except OSError:
                        pass

        proj_bytes = 0
        if cfg_guess:
            try:
                proj_bytes += os.path.getsize(cfg_guess)
            except OSError:
                pass

        try:
            import datetime as _dt

            updated_at = _dt.datetime.fromtimestamp(
                os.path.getmtime(candidate)
            ).isoformat()
        except OSError:
            updated_at = None

        manifests.append(
            {
                "id": dir_name,
                "slug": dir_name,
                "title": title,
                "description": None,
                "tags": [],
                "primaryModel": inferred_model,
                "dataset_count": None,
                "data_size_bytes": data_bytes,
                "data_size_human": _humanize_size(data_bytes) if data_bytes else None,
                "project_size_bytes": proj_bytes,
                "project_size_human": _humanize_size(proj_bytes)
                if proj_bytes
                else None,
                "updated_at": updated_at,
            }
        )

    return manifests


# Support both with and without trailing slash to avoid proxy redirect issues
@router.get("/")
@router.get("")
async def list_examples() -> dict[str, list[ExampleSummary]]:
    data = _scan_manifests()
    items = [ExampleSummary(**ex) for ex in data]
    return {"examples": items}


def _load_manifest_by_id(example_id: str) -> dict[str, Any]:
    for manifest_path in glob(os.path.join(_examples_root(), "*", "manifest.yaml")):
        import yaml

        with open(manifest_path) as f:
            m = yaml.safe_load(f) or {}
        if m.get("id") == example_id:
            # Attach base_dir for resolving relative paths
            m["_base_dir"] = os.path.dirname(manifest_path)
            return m
    raise HTTPException(status_code=404, detail=f"Example '{example_id}' not found")


def _humanize_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    unit_idx = 0
    while size >= 1024 and unit_idx < len(units) - 1:
        size /= 1024.0
        unit_idx += 1
    if unit_idx == 0:
        return f"{int(size)}{units[unit_idx]}"
    return f"{size:.1f}{units[unit_idx]}"


def _list_example_datasets_from_manifest(m: dict[str, Any]) -> list[ExampleDataset]:
    """Construct ExampleDataset entries from a loaded manifest dict."""
    example_id = m.get("id")
    example_title = m.get("title")
    datasets: list[ExampleDataset] = []
    for ds in m.get("datasets", []) or []:
        name = ds.get("name")
        if not name:
            continue
        ingest_patterns = list(ds.get("ingest", []) or [])
        # Compute total size by expanding globs under examples dir
        total = 0
        for pattern in ingest_patterns:
            abs_glob = _ensure_path_under_examples(os.path.join(_repo_root(), pattern))
            for path in glob(abs_glob):
                try:
                    total += os.path.getsize(path)
                except OSError as e:
                    logger.warning(
                        "Failed to get size for path", path=path, error=str(e)
                    )

        # Naive kind inference from file extensions in patterns
        kind: str | None = None
        pattern_str = " ".join(ingest_patterns).lower()
        if ".pdf" in pattern_str:
            kind = "pdf"
        elif ".csv" in pattern_str:
            kind = "csv"
        elif ".md" in pattern_str or "markdown" in pattern_str:
            kind = "markdown"
        elif any(
            ext in pattern_str for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
        ):
            kind = "images"
        elif ".json" in pattern_str:
            kind = "json"

        datasets.append(
            ExampleDataset(
                example_id=example_id,
                example_title=example_title,
                name=name,
                strategy=ds.get("strategy"),
                database=ds.get("database"),
                ingest=ingest_patterns,
                size_bytes=total or None,
                size_human=_humanize_size(total) if total else None,
                kind=kind,
            )
        )
    return datasets


@router.get("/{example_id}/datasets")
async def list_example_datasets(example_id: str) -> dict[str, list[ExampleDataset]]:
    """List datasets for a specific example id from its manifest."""
    m = _load_manifest_by_id(example_id)
    return {"datasets": _list_example_datasets_from_manifest(m)}


@router.get("/datasets")
async def list_all_example_datasets() -> dict[str, list[ExampleDataset]]:
    """Flattened list of datasets across all examples."""
    datasets: list[ExampleDataset] = []
    for manifest_path in glob(os.path.join(_examples_root(), "*", "manifest.yaml")):
        try:
            import yaml

            with open(manifest_path) as f:
                m = yaml.safe_load(f) or {}
            if not m.get("id"):
                continue
            datasets.extend(_list_example_datasets_from_manifest(m))
        except Exception as e:
            logger.warning(
                "Failed to read example manifest for datasets",
                path=manifest_path,
                error=str(e),
            )
    return {"datasets": datasets}


class ImportProjectRequest(BaseModel):
    namespace: str
    name: str
    process: bool = True


class ImportProjectResponse(BaseModel):
    project: str
    namespace: str
    datasets: list[str]
    task_ids: list[str]


def _ensure_path_under_examples(path: str) -> str:
    abs_path = os.path.abspath(path)
    ex_root = os.path.abspath(_examples_root()) + os.sep
    if not abs_path.startswith(ex_root):
        raise HTTPException(status_code=400, detail="Path outside examples directory")
    return abs_path


def _add_file_from_path(
    namespace: str, project: str, dataset: str, source_path: str
) -> MetadataFileContent:
    # Replicate DataService.add_data_file for local path
    data_dir = DataService.ensure_data_dir(namespace, project, dataset)
    with open(source_path, "rb") as f:
        file_data = f.read()
    data_hash = DataService.hash_data(file_data)
    original_name = os.path.basename(source_path)
    resolved_file_name = DataService.append_collision_timestamp(original_name)

    # Write metadata
    meta_path = os.path.join(data_dir, "meta", f"{data_hash}.json")
    meta = MetadataFileContent(
        original_file_name=original_name,
        resolved_file_name=resolved_file_name,
        timestamp=float(os.path.getmtime(source_path)),
        size=len(file_data),
        mime_type="application/octet-stream",
        hash=data_hash,
    )
    with open(meta_path, "w") as f:
        f.write(meta.model_dump_json())

    # Write raw
    raw_path = os.path.join(data_dir, "raw", data_hash)
    with open(raw_path, "wb") as f:
        f.write(file_data)

    # Index symlink
    index_dir = os.path.join(data_dir, "index", "by_name")
    os.makedirs(index_dir, exist_ok=True)
    index_path = os.path.join(index_dir, resolved_file_name)
    try:
        os.symlink(raw_path, index_path)
    except FileExistsError:
        pass

    DatasetService.add_file_to_dataset(namespace, project, dataset, meta)
    return meta


def _merge_rag_components(
    base_cfg: LlamaFarmConfig, example_cfg: LlamaFarmConfig
) -> LlamaFarmConfig:
    # Extend databases and strategies by name without duplicates
    base = base_cfg.model_dump(mode="json")
    ex = example_cfg.model_dump(mode="json")
    base_rag = base.get("rag") or {}
    ex_rag = ex.get("rag") or {}

    # Databases
    base_dbs = {d["name"]: d for d in base_rag.get("databases", []) or []}
    for d in ex_rag.get("databases", []) or []:
        base_dbs.setdefault(d.get("name"), d)
    # Strategies
    base_strats = {
        s["name"]: s for s in base_rag.get("data_processing_strategies", []) or []
    }
    for s in ex_rag.get("data_processing_strategies", []) or []:
        base_strats.setdefault(s.get("name"), s)

    base_rag["databases"] = list(base_dbs.values())
    base_rag["data_processing_strategies"] = list(base_strats.values())
    if ex_rag.get("default_embedding_strategy"):
        base_rag.setdefault(
            "default_embedding_strategy", ex_rag.get("default_embedding_strategy")
        )
    if ex_rag.get("default_retrieval_strategy"):
        base_rag.setdefault(
            "default_retrieval_strategy", ex_rag.get("default_retrieval_strategy")
        )
    base["rag"] = base_rag

    return LlamaFarmConfig(**base)


@router.post("/{example_id}/import-project", response_model=ImportProjectResponse)
async def import_project(
    example_id: str, request: ImportProjectRequest
) -> ImportProjectResponse:
    import yaml

    m = _load_manifest_by_id(example_id)
    base_dir = m.get("_base_dir")
    cfg_path = m.get("config", {}).get("yaml_path")
    if not cfg_path:
        raise HTTPException(status_code=400, detail="Manifest missing config.yaml_path")
    cfg_abs = _ensure_path_under_examples(os.path.join(_repo_root(), cfg_path))

    # Create project and write example config with overridden name/namespace
    ProjectService.create_project(request.namespace, request.name, config_template=None)

    with open(cfg_abs) as f:
        cfg_dict = yaml.safe_load(f) or {}
    cfg_dict["name"] = request.name
    cfg_dict["namespace"] = request.namespace
    cfg_model = LlamaFarmConfig(**cfg_dict)
    ProjectService.save_config(request.namespace, request.name, cfg_model)

    # Create datasets and ingest files
    datasets = []
    task_ids: list[str] = []
    for ds in m.get("datasets", []) or []:
        ds_name = ds.get("name")
        strategy = ds.get("strategy")
        database = ds.get("database")
        if not (ds_name and strategy and database):
            continue
        try:
            DatasetService.create_dataset(
                request.namespace, request.name, ds_name, strategy, database
            )
        except ValueError as e:
            # Allow already-existing datasets; surface other errors
            msg = str(e)
            if "already exists" in msg:
                logger.info("Dataset already exists; continuing", dataset=ds_name)
            else:
                logger.warning(
                    "Failed to create dataset during import_project",
                    dataset=ds_name,
                    error=msg,
                )
                raise HTTPException(status_code=400, detail=msg) from e
        datasets.append(ds_name)

        # Expand globs relative to repo root
        for pattern in ds.get("ingest", []) or []:
            abs_glob = _ensure_path_under_examples(os.path.join(_repo_root(), pattern))
            for path in glob(abs_glob):
                _add_file_from_path(request.namespace, request.name, ds_name, path)

        if request.process:
            launch = DatasetService.start_dataset_ingestion(
                request.namespace, request.name, ds_name
            )
            task_ids.append(launch.task_id)

    return ImportProjectResponse(
        project=request.name,
        namespace=request.namespace,
        datasets=datasets,
        task_ids=task_ids,
    )


class ImportDataRequest(BaseModel):
    namespace: str
    project: str
    include_strategies: bool = True
    process: bool = True


class ImportDataResponse(BaseModel):
    project: str
    namespace: str
    datasets: list[str]
    task_ids: list[str]


@router.post("/{example_id}/import-data", response_model=ImportDataResponse)
async def import_data(
    example_id: str, request: ImportDataRequest
) -> ImportDataResponse:
    import yaml

    m = _load_manifest_by_id(example_id)
    cfg_path = m.get("config", {}).get("yaml_path")
    if not cfg_path:
        raise HTTPException(status_code=400, detail="Manifest missing config.yaml_path")

    # Optionally merge strategies/databases
    if request.include_strategies:
        cfg_abs = _ensure_path_under_examples(os.path.join(_repo_root(), cfg_path))
        with open(cfg_abs) as f:
            ex_cfg_dict = yaml.safe_load(f) or {}
        ex_cfg = LlamaFarmConfig(**ex_cfg_dict)
        base_cfg = ProjectService.load_config(request.namespace, request.project)
        merged = _merge_rag_components(base_cfg, ex_cfg)
        ProjectService.update_project(request.namespace, request.project, merged)

    datasets = []
    task_ids: list[str] = []
    for ds in m.get("datasets", []) or []:
        ds_name = ds.get("name")
        strategy = ds.get("strategy")
        database = ds.get("database")
        if not (ds_name and strategy and database):
            continue
        # Create if missing
        try:
            DatasetService.create_dataset(
                request.namespace, request.project, ds_name, strategy, database
            )
        except ValueError as e:
            msg = str(e)
            if "already exists" in msg:
                logger.info("Dataset already exists; continuing", dataset=ds_name)
            else:
                logger.warning(
                    "Failed to create dataset during import_data",
                    dataset=ds_name,
                    error=msg,
                )
                raise HTTPException(status_code=400, detail=msg) from e
        datasets.append(ds_name)

        for pattern in ds.get("ingest", []) or []:
            abs_glob = _ensure_path_under_examples(os.path.join(_repo_root(), pattern))
            for path in glob(abs_glob):
                _add_file_from_path(request.namespace, request.project, ds_name, path)

        if request.process:
            launch = DatasetService.start_dataset_ingestion(
                request.namespace, request.project, ds_name
            )
            task_ids.append(launch.task_id)

    return ImportDataResponse(
        project=request.project,
        namespace=request.namespace,
        datasets=datasets,
        task_ids=task_ids,
    )


class ImportDatasetRequest(BaseModel):
    namespace: str
    project: str
    dataset: str  # dataset name from manifest
    target_dataset: str | None = None  # optional new name in project
    include_strategies: bool = True
    process: bool = True


class ImportDatasetResponse(BaseModel):
    project: str
    namespace: str
    dataset: str
    file_count: int
    task_id: str | None = None


@router.post("/{example_id}/import-dataset", response_model=ImportDatasetResponse)
async def import_dataset(
    example_id: str, request: ImportDatasetRequest
) -> ImportDatasetResponse:
    import yaml

    m = _load_manifest_by_id(example_id)
    cfg_path = m.get("config", {}).get("yaml_path")
    if not cfg_path:
        raise HTTPException(status_code=400, detail="Manifest missing config.yaml_path")

    # Merge strategies/databases if requested
    if request.include_strategies:
        cfg_abs = _ensure_path_under_examples(os.path.join(_repo_root(), cfg_path))
        with open(cfg_abs) as f:
            ex_cfg_dict = yaml.safe_load(f) or {}
        ex_cfg = LlamaFarmConfig(**ex_cfg_dict)
        base_cfg = ProjectService.load_config(request.namespace, request.project)
        merged = _merge_rag_components(base_cfg, ex_cfg)
        ProjectService.update_project(request.namespace, request.project, merged)

    # Find dataset in manifest
    manifest_ds = next(
        (
            ds
            for ds in (m.get("datasets", []) or [])
            if ds.get("name") == request.dataset
        ),
        None,
    )
    if not manifest_ds:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{request.dataset}' not found in example '{example_id}'",
        )

    ds_name = request.target_dataset or manifest_ds.get("name")
    strategy = manifest_ds.get("strategy")
    database = manifest_ds.get("database")
    if not (ds_name and strategy and database):
        raise HTTPException(
            status_code=400,
            detail="Dataset entry incomplete in manifest (name/strategy/database)",
        )

    # Create if missing
    try:
        DatasetService.create_dataset(
            request.namespace, request.project, ds_name, strategy, database
        )
    except ValueError as e:
        msg = str(e)
        if "already exists" in msg:
            logger.info("Dataset already exists; continuing", dataset=ds_name)
        else:
            logger.warning(
                "Failed to create dataset during import_dataset",
                dataset=ds_name,
                error=msg,
            )
            raise HTTPException(status_code=400, detail=msg) from e

    # Add files
    file_count = 0
    for pattern in manifest_ds.get("ingest", []) or []:
        abs_glob = _ensure_path_under_examples(os.path.join(_repo_root(), pattern))
        for path in glob(abs_glob):
            _add_file_from_path(request.namespace, request.project, ds_name, path)
            file_count += 1

    task_id: str | None = None
    if request.process:
        launch = DatasetService.start_dataset_ingestion(
            request.namespace, request.project, ds_name
        )
        task_id = launch.task_id

    return ImportDatasetResponse(
        project=request.project,
        namespace=request.namespace,
        dataset=ds_name,
        file_count=file_count,
        task_id=task_id,
    )
