"""Addon service implementation.

Architecture note: The server delegates addon install/uninstall to the CLI
binary via subprocess. The CLI handles wheel download, extraction, and
dependency cleanup. Service restart is handled separately to avoid the CLI
stopping the server process that spawned it.

The --no-restart flag tells the CLI to skip its stop/restart cycle. After the
CLI finishes, this service restarts only the affected component (never the
server itself) via a second CLI call.

Task status is persisted to disk so that if the server restarts mid-install,
the client can poll the final status once the server is back up.
"""

import asyncio
import json
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from core.logging import FastAPIStructLogger
from core.settings import settings

from .registry import get_addon_registry, reload_addon_registry
from .types import AddonInfo, AddonTaskStatus

logger = FastAPIStructLogger()

# Validate addon names: alphanumeric, hyphens, underscores only
ADDON_NAME_PATTERN = re.compile(r"^[a-z0-9_-]+$")


class AddonService:
    """Service for managing addons."""

    def __init__(self):
        self.task_statuses: dict[str, AddonTaskStatus] = {}
        self.task_status_lock = asyncio.Lock()
        self.state_file = Path(settings.lf_data_dir) / "addons.json"
        self.tasks_dir = Path(settings.lf_data_dir) / "addon-tasks"
        self.tasks_dir.mkdir(parents=True, exist_ok=True)

        # Recover any persisted task statuses from a previous server lifetime
        self._recover_persisted_tasks()

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
        """Background task to install an addon.

        Uses --no-restart so the CLI never stops/starts services. After the
        CLI finishes, we restart only the affected component (not the server).
        """
        try:
            self._validate_addon_name(addon_name)

            await self._update_task_status_async(
                task_id, "in_progress", 0, "Starting installation..."
            )

            cli_path = self._find_cli_binary()

            # Use --no-restart so the CLI only downloads and extracts.
            # This avoids the CLI stopping the server process that spawned it.
            await self._update_task_status_async(
                task_id, "in_progress", 10, "Downloading and installing addon packages..."
            )

            result = await asyncio.to_thread(
                subprocess.run,
                [cli_path, "addons", "install", "--no-restart", addon_name],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode, result.args, result.stdout, result.stderr
                )

            # Surface CLI output in progress message
            install_output = result.stdout.strip()
            await self._update_task_status_async(
                task_id, "in_progress", 70, f"Addon installed. {self._summarize_output(install_output)}"
            )

            # Restart the affected component (not the server)
            if restart:
                registry = get_addon_registry()
                addon_def = registry.get(addon_name, {})
                component = addon_def.get("component", "")

                if component and component != "server":
                    await self._update_task_status_async(
                        task_id, "in_progress", 80, f"Restarting {component}..."
                    )
                    restart_result = await asyncio.to_thread(
                        subprocess.run,
                        [cli_path, "services", "start", component],
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                    if restart_result.returncode != 0:
                        logger.warning(
                            f"Failed to restart {component}: {restart_result.stderr}"
                        )
                        await self._update_task_status_async(
                            task_id,
                            "completed",
                            100,
                            f"Addon installed but {component} restart failed. "
                            f"Run 'lf services start {component}' manually.",
                        )
                        return
                elif component == "server":
                    logger.warning(
                        "Addon targets server component; skipping automatic restart "
                        "to avoid self-termination. Manual restart required."
                    )

            reload_addon_registry()

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
                task_id, "failed", 0, "Installation timed out after 5 minutes", str(e)
            )
        except subprocess.CalledProcessError as e:
            error_msg = (e.stderr or e.stdout or str(e)).strip()
            logger.error(f"Failed to install addon {addon_name}: {error_msg}")
            await self._update_task_status_async(
                task_id, "failed", 0, "Installation failed", error_msg
            )
        except Exception as e:
            logger.error(f"Unexpected error installing addon {addon_name}: {e}")
            await self._update_task_status_async(
                task_id, "failed", 0, "Installation failed", str(e)
            )

    async def uninstall_addon(self, addon_name: str):
        """Uninstall an addon.

        Uses --no-restart so the CLI never stops/starts services. Restarts the
        affected component afterward (unless it targets the server).
        """
        self._validate_addon_name(addon_name)
        cli_path = self._find_cli_binary()

        # Look up component before uninstall (state is cleared after)
        registry = get_addon_registry()
        addon_def = registry.get(addon_name, {})
        component = addon_def.get("component", "")

        result = await asyncio.to_thread(
            subprocess.run,
            [cli_path, "addons", "uninstall", "--no-restart", addon_name],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            error_msg = (result.stderr or result.stdout or "Unknown error").strip()
            raise RuntimeError(f"Uninstall failed: {error_msg}")

        reload_addon_registry()

        # Restart the affected component (not the server)
        if component and component != "server":
            restart_result = await asyncio.to_thread(
                subprocess.run,
                [cli_path, "services", "start", component],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if restart_result.returncode != 0:
                logger.warning(f"Failed to restart {component} after uninstall: {restart_result.stderr}")

    async def get_task_status(self, task_id: str) -> AddonTaskStatus | None:
        """Get the status of a task (holds lock for consistency with updates)."""
        async with self.task_status_lock:
            return self.task_statuses.get(task_id)

    def _find_cli_binary(self) -> str:
        """Find the CLI binary path.

        Search order:
        1. PATH (works when CLI is installed normally)
        2. LF_DATA_DIR/bin/lf (works in containers/systemd where PATH is minimal)
        """
        cli_path = shutil.which("lf")
        if cli_path:
            return cli_path

        data_dir_bin = Path(settings.lf_data_dir) / "bin" / "lf"
        if data_dir_bin.exists():
            return str(data_dir_bin)

        raise FileNotFoundError(
            f"CLI binary 'lf' not found on PATH or at {data_dir_bin}. "
            f"Ensure the CLI is installed and accessible."
        )

    def _load_state(self) -> dict:
        """Load addon state from file."""
        if not self.state_file.exists():
            return {"version": "1", "installed_addons": {}}

        with open(self.state_file) as f:
            return json.load(f)

    @staticmethod
    def _summarize_output(output: str) -> str:
        """Extract the last meaningful line from CLI output for the progress message."""
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if not lines:
            return ""
        # Return the last line, truncated to a reasonable length
        last = lines[-1]
        if len(last) > 120:
            return last[:117] + "..."
        return last

    async def _update_task_status_async(
        self,
        task_id: str,
        status: str,
        progress: int,
        message: str,
        error: str | None = None,
    ):
        """Update task status in memory and persist to disk."""
        task_status = AddonTaskStatus(
            status=status, progress=progress, message=message, error=error
        )
        async with self.task_status_lock:
            self.task_statuses[task_id] = task_status

        # Persist terminal states to disk so they survive server restarts
        if status in ("completed", "failed"):
            self._persist_task_status(task_id, task_status)

    def _persist_task_status(self, task_id: str, task_status: AddonTaskStatus):
        """Write task status to disk so it survives server restarts."""
        try:
            task_file = self.tasks_dir / f"{task_id}.json"
            data = {
                **task_status.model_dump(),
                "updated_at": datetime.now(UTC).isoformat(),
            }
            task_file.write_text(json.dumps(data))
        except Exception as e:
            logger.warning(f"Failed to persist task status {task_id}: {e}")

    def _recover_persisted_tasks(self):
        """Load terminal task statuses from disk (from a previous server lifetime).

        Prunes task files older than 24 hours to prevent unbounded accumulation.
        """
        max_age = 24 * 60 * 60  # 24 hours in seconds
        now = datetime.now(UTC)
        try:
            for task_file in self.tasks_dir.glob("*.json"):
                try:
                    data = json.loads(task_file.read_text())

                    # Prune old task files
                    updated_at = data.get("updated_at")
                    if updated_at:
                        age = (now - datetime.fromisoformat(updated_at)).total_seconds()
                        if age > max_age:
                            task_file.unlink(missing_ok=True)
                            continue

                    task_id = task_file.stem
                    self.task_statuses[task_id] = AddonTaskStatus(
                        status=data["status"],
                        progress=data["progress"],
                        message=data["message"],
                        error=data.get("error"),
                    )
                except Exception as e:
                    logger.warning(f"Failed to recover task {task_file.name}: {e}")
        except Exception as e:
            logger.warning(f"Failed to recover persisted tasks: {e}")
