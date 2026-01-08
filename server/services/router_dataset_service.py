"""
Router Dataset Service - Storage for router training utterances.

Utterances are stored as JSON files in the router storage directory:
    {project_dir}/lf_data/routers/{router_name}/datasets/{route_name}.json

This allows utterances to be:
- Generated and saved for later use
- Loaded during router training
- Appended or overwritten
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from services.router_storage_service import RouterStorageService

logger = logging.getLogger(__name__)


class RouterDatasetService:
    """Service for managing router utterance datasets."""

    @classmethod
    def get_datasets_dir(cls, namespace: str, project_id: str, router_name: str) -> Path:
        """Get the datasets directory for a router."""
        router_dir = RouterStorageService.get_router_dir(namespace, project_id, router_name)
        return router_dir / "datasets"

    @classmethod
    def get_route_dataset_path(
        cls, namespace: str, project_id: str, router_name: str, route_name: str
    ) -> Path:
        """Get the dataset file path for a specific route."""
        datasets_dir = cls.get_datasets_dir(namespace, project_id, router_name)
        # Sanitize route name for filename
        safe_route_name = route_name.replace("/", "_").replace("\\", "_")
        return datasets_dir / f"{safe_route_name}.json"

    @classmethod
    def save_utterances(
        cls,
        namespace: str,
        project_id: str,
        router_name: str,
        route_name: str,
        utterances: list[str],
        mode: str = "append",
        description: str | None = None,
    ) -> dict[str, Any]:
        """Save utterances to a route dataset.

        Args:
            namespace: Project namespace
            project_id: Project ID
            router_name: Router name
            route_name: Route name
            utterances: List of utterance strings
            mode: "append" to add to existing or "overwrite" to replace
            description: Optional description of the dataset

        Returns:
            Dict with saved dataset info
        """
        dataset_path = cls.get_route_dataset_path(
            namespace, project_id, router_name, route_name
        )

        # Ensure directory exists
        dataset_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing data if appending
        existing_utterances: list[str] = []
        if mode == "append" and dataset_path.exists():
            with open(dataset_path) as f:
                data = json.load(f)
                existing_utterances = data.get("utterances", [])

        # Merge utterances (deduplicate)
        all_utterances = existing_utterances + utterances
        seen = set()
        unique_utterances = []
        for u in all_utterances:
            lower = u.lower()
            if lower not in seen:
                seen.add(lower)
                unique_utterances.append(u)

        # Build dataset structure
        dataset = {
            "route_name": route_name,
            "description": description or f"Utterances for route: {route_name}",
            "utterances": unique_utterances,
            "count": len(unique_utterances),
            "updated_at": datetime.now(UTC).isoformat(),
            "metadata": {
                "namespace": namespace,
                "project_id": project_id,
                "router_name": router_name,
            },
        }

        # Save
        with open(dataset_path, "w") as f:
            json.dump(dataset, f, indent=2)

        logger.info(
            f"Saved router dataset: {dataset_path} ({len(unique_utterances)} utterances)"
        )

        return {
            "path": str(dataset_path),
            "route_name": route_name,
            "count": len(unique_utterances),
            "added": len(utterances),
            "mode": mode,
        }

    @classmethod
    def load_utterances(
        cls, namespace: str, project_id: str, router_name: str, route_name: str
    ) -> list[str]:
        """Load utterances from a route dataset.

        Returns empty list if dataset doesn't exist.
        """
        dataset_path = cls.get_route_dataset_path(
            namespace, project_id, router_name, route_name
        )

        if not dataset_path.exists():
            logger.debug(f"Dataset not found: {dataset_path}")
            return []

        with open(dataset_path) as f:
            data = json.load(f)
            return data.get("utterances", [])

    @classmethod
    def list_datasets(
        cls, namespace: str, project_id: str, router_name: str
    ) -> list[dict[str, Any]]:
        """List all route datasets for a router."""
        datasets_dir = cls.get_datasets_dir(namespace, project_id, router_name)

        if not datasets_dir.exists():
            return []

        datasets = []
        for dataset_file in datasets_dir.glob("*.json"):
            try:
                with open(dataset_file) as f:
                    data = json.load(f)
                    datasets.append({
                        "route_name": data.get("route_name"),
                        "count": data.get("count", 0),
                        "description": data.get("description"),
                        "updated_at": data.get("updated_at"),
                        "path": str(dataset_file),
                    })
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in dataset file: {dataset_file}")
                continue

        return datasets

    @classmethod
    def delete_dataset(
        cls, namespace: str, project_id: str, router_name: str, route_name: str
    ) -> bool:
        """Delete a route dataset.

        Returns True if deleted, False if didn't exist.
        """
        dataset_path = cls.get_route_dataset_path(
            namespace, project_id, router_name, route_name
        )

        if not dataset_path.exists():
            return False

        dataset_path.unlink()
        logger.info(f"Deleted router dataset: {dataset_path}")
        return True
