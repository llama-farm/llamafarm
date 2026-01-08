"""
Router Storage Service - Project-specific storage for semantic router models.

Routers are stored in the project data directory:
    {project_dir}/lf_data/routers/{router_name}/
        config.json       - Router configuration (routes, thresholds, etc.)
        embeddings.pt     - Pre-computed route embeddings

This allows routers to be project-specific while keeping the config
as the source of truth in llamafarm.yaml.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from services.project_service import ProjectService

logger = logging.getLogger(__name__)

DATA_DIR_NAME = "lf_data"
ROUTERS_DIR_NAME = "routers"


class RouterStorageService:
    """Service for managing project-specific router storage."""

    @classmethod
    def get_routers_dir(cls, namespace: str, project_id: str) -> Path:
        """Get the routers directory for a project."""
        project_dir = ProjectService.get_project_dir(namespace, project_id)
        routers_dir = Path(project_dir) / DATA_DIR_NAME / ROUTERS_DIR_NAME
        return routers_dir

    @classmethod
    def get_router_dir(cls, namespace: str, project_id: str, router_name: str) -> Path:
        """Get the directory for a specific router."""
        routers_dir = cls.get_routers_dir(namespace, project_id)
        # Validate router name to prevent path traversal
        if "/" in router_name or "\\" in router_name or ".." in router_name:
            raise ValueError(f"Invalid router name: {router_name!r}")
        return routers_dir / router_name

    @classmethod
    def ensure_router_dir(cls, namespace: str, project_id: str, router_name: str) -> Path:
        """Ensure the router directory exists and return its path."""
        router_dir = cls.get_router_dir(namespace, project_id, router_name)
        router_dir.mkdir(parents=True, exist_ok=True)
        return router_dir

    @classmethod
    def save_router_config(
        cls,
        namespace: str,
        project_id: str,
        router_name: str,
        config: dict[str, Any],
    ) -> Path:
        """Save router configuration to the project directory.

        Args:
            namespace: Project namespace
            project_id: Project ID
            router_name: Router model name
            config: Router configuration dict (routes, threshold, etc.)

        Returns:
            Path to the saved config file
        """
        router_dir = cls.ensure_router_dir(namespace, project_id, router_name)
        config_path = router_dir / "config.json"

        # Add metadata
        config["_metadata"] = {
            "saved_at": datetime.now(UTC).isoformat(),
            "namespace": namespace,
            "project_id": project_id,
            "router_name": router_name,
        }

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved router config: {config_path}")
        return config_path

    @classmethod
    def load_router_config(
        cls,
        namespace: str,
        project_id: str,
        router_name: str,
    ) -> dict[str, Any] | None:
        """Load router configuration from the project directory.

        Returns None if the router doesn't exist.
        """
        router_dir = cls.get_router_dir(namespace, project_id, router_name)
        config_path = router_dir / "config.json"

        if not config_path.exists():
            return None

        with open(config_path) as f:
            return json.load(f)

    @classmethod
    def save_embeddings(
        cls,
        namespace: str,
        project_id: str,
        router_name: str,
        embeddings_data: bytes,
    ) -> Path:
        """Save pre-computed embeddings to the project directory.

        Args:
            namespace: Project namespace
            project_id: Project ID
            router_name: Router model name
            embeddings_data: Serialized embeddings (torch.save format)

        Returns:
            Path to the saved embeddings file
        """
        router_dir = cls.ensure_router_dir(namespace, project_id, router_name)
        embeddings_path = router_dir / "embeddings.pt"

        with open(embeddings_path, "wb") as f:
            f.write(embeddings_data)

        logger.info(f"Saved router embeddings: {embeddings_path}")
        return embeddings_path

    @classmethod
    def load_embeddings(
        cls,
        namespace: str,
        project_id: str,
        router_name: str,
    ) -> bytes | None:
        """Load pre-computed embeddings from the project directory.

        Returns None if embeddings don't exist.
        """
        router_dir = cls.get_router_dir(namespace, project_id, router_name)
        embeddings_path = router_dir / "embeddings.pt"

        if not embeddings_path.exists():
            return None

        with open(embeddings_path, "rb") as f:
            return f.read()

    @classmethod
    def list_routers(cls, namespace: str, project_id: str) -> list[dict[str, Any]]:
        """List all routers in a project.

        Returns list of router info dicts with name, config, and metadata.
        """
        routers_dir = cls.get_routers_dir(namespace, project_id)

        if not routers_dir.exists():
            return []

        routers = []
        for router_dir in routers_dir.iterdir():
            if not router_dir.is_dir():
                continue

            config = cls.load_router_config(namespace, project_id, router_dir.name)
            if config:
                routers.append({
                    "name": router_dir.name,
                    "path": str(router_dir),
                    "has_embeddings": (
                        (router_dir / "embeddings.pt").exists()
                        or (router_dir / "embeddings.npz").exists()
                    ),
                    "config": config,
                })

        return routers

    @classmethod
    def delete_router(cls, namespace: str, project_id: str, router_name: str) -> bool:
        """Delete a router and all its data.

        Returns True if deleted, False if didn't exist.
        """
        import shutil

        router_dir = cls.get_router_dir(namespace, project_id, router_name)

        if not router_dir.exists():
            return False

        shutil.rmtree(router_dir)
        logger.info(f"Deleted router: {router_dir}")
        return True

    @classmethod
    def router_exists(cls, namespace: str, project_id: str, router_name: str) -> bool:
        """Check if a router exists in the project."""
        router_dir = cls.get_router_dir(namespace, project_id, router_name)
        return router_dir.exists() and (router_dir / "config.json").exists()
