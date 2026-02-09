"""Addon service implementation."""

import asyncio
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

from core.logging import FastAPIStructLogger

from .registry import get_addon_registry
from .types import AddonInfo, AddonTaskStatus

logger = FastAPIStructLogger()

# Validate addon names: alphanumeric, hyphens, underscores only
ADDON_NAME_PATTERN = re.compile(r"^[a-z0-9_-]+$")


class AddonService:
    """Service for managing addons."""

    def __init__(self):
        self.task_statuses: dict[str, AddonTaskStatus] = {}
        self.task_status_lock = asyncio.Lock()
        self.state_file = Path.home() / ".llamafarm" / "addons.json"

    def _validate_addon_name(self, name: str) -> None:
        """Validate addon name to prevent injection attacks."""
        if not name:
            raise ValueError("Addon name cannot be empty")
        if not ADDON_NAME_PATTERN.match(name):
            raise ValueError(
                f"Invalid addon name: {name}. Must contain only lowercase letters, numbers, hyphens, and underscores."
            )

    def list_addons(self) -> list[AddonInfo]:
        """List all addons with installation status."""
        state = self._load_state()
        registry = get_addon_registry()

        result = []
        for name, addon in registry.items():
            installed_info = state.get("installed_addons", {}).get(name)

            result.append(
                AddonInfo(
                    name=addon["name"],
                    display_name=addon["display_name"],
                    description=addon["description"],
                    component=addon["component"],
                    version=addon["version"],
                    dependencies=addon.get("dependencies", []),
                    installed=installed_info is not None,
                    installed_at=datetime.fromisoformat(installed_info["installed_at"])
                    if installed_info
                    else None,
                )
            )

        return result

    def addon_exists(self, name: str) -> bool:
        """Check if an addon exists in the registry."""
        registry = get_addon_registry()
        return name in registry

    async def install_addon_task(self, task_id: str, addon_name: str, restart: bool):
        """Background task to install an addon."""
        try:
            # Validate addon name before using it
            self._validate_addon_name(addon_name)

            await self._update_task_status_async(
                task_id, "in_progress", 0, "Starting installation..."
            )

            # Find CLI binary
            cli_path = self._find_cli_binary()

            # Run CLI install command
            await self._update_task_status_async(
                task_id, "in_progress", 50, "Installing addon..."
            )
            await asyncio.to_thread(
                subprocess.run,
                [cli_path, "addons", "install", addon_name],
                check=True,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            # Restart service if requested
            if restart:
                await self._update_task_status_async(
                    task_id, "in_progress", 90, "Restarting service..."
                )
                registry = get_addon_registry()
                addon = registry[addon_name]
                component = addon["component"]

                # Validate component name as well
                if not ADDON_NAME_PATTERN.match(component):
                    raise ValueError(f"Invalid component name: {component}")

                await asyncio.to_thread(
                    subprocess.run,
                    [cli_path, "services", "stop", component],
                    check=True,
                    timeout=60,
                )
                await asyncio.to_thread(
                    subprocess.run,
                    [cli_path, "services", "start", component],
                    check=True,
                    timeout=60,
                )

            await self._update_task_status_async(
                task_id, "completed", 100, "Installation complete"
            )

        except ValueError as e:
            logger.error(f"Validation error installing addon {addon_name}: {e}")
            await self._update_task_status_async(
                task_id, "failed", 0, "Validation failed", str(e)
            )
        except subprocess.TimeoutExpired as e:
            logger.error(f"Timeout installing addon {addon_name}: {e}")
            await self._update_task_status_async(
                task_id, "failed", 0, "Installation timeout", str(e)
            )
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            logger.error(f"Failed to install addon {addon_name}: {error_msg}")
            await self._update_task_status_async(
                task_id, "failed", 0, "Installation failed", error_msg
            )
        except Exception as e:
            logger.error(f"Unexpected error installing addon {addon_name}: {e}")
            await self._update_task_status_async(
                task_id, "failed", 0, "Installation failed", str(e)
            )

    def uninstall_addon(self, addon_name: str):
        """Uninstall an addon."""
        self._validate_addon_name(addon_name)
        cli_path = self._find_cli_binary()
        subprocess.run(
            [cli_path, "addons", "uninstall", addon_name],
            check=True,
            timeout=60,
        )

    async def get_task_status_async(self, task_id: str) -> AddonTaskStatus | None:
        """Get the status of a task (thread-safe)."""
        async with self.task_status_lock:
            return self.task_statuses.get(task_id)

    def get_task_status(self, task_id: str) -> AddonTaskStatus | None:
        """Get the status of a task (synchronous version)."""
        return self.task_statuses.get(task_id)

    def _find_cli_binary(self) -> str:
        """Find the CLI binary path."""
        # Check PATH first
        result = subprocess.run(["which", "lf"], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()

        # Check ~/.llamafarm/bin/
        home_bin = Path.home() / ".llamafarm" / "bin" / "lf"
        if home_bin.exists():
            return str(home_bin)

        raise FileNotFoundError("CLI binary 'lf' not found")

    def _load_state(self) -> dict:
        """Load addon state from file."""
        if not self.state_file.exists():
            return {"version": "1", "installed_addons": {}}

        with open(self.state_file) as f:
            return json.load(f)

    async def _update_task_status_async(
        self,
        task_id: str,
        status: str,
        progress: int,
        message: str,
        error: str | None = None,
    ):
        """Update task status (thread-safe async version)."""
        async with self.task_status_lock:
            self.task_statuses[task_id] = AddonTaskStatus(
                status=status, progress=progress, message=message, error=error
            )

    def _update_task_status(
        self,
        task_id: str,
        status: str,
        progress: int,
        message: str,
        error: str | None = None,
    ):
        """Update task status (synchronous version - use async version when possible)."""
        self.task_statuses[task_id] = AddonTaskStatus(
            status=status, progress=progress, message=message, error=error
        )
