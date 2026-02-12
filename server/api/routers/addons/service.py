"""Addon service implementation."""

import asyncio
import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from core.logging import FastAPIStructLogger
from core.settings import settings

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
        self.state_file = Path(settings.lf_data_dir) / "addons.json"

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
                    packages=addon.get("packages", []),
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

    async def _install_dependencies(
        self, task_id: str, addon_name: str, installing_chain: set[str]
    ) -> None:
        """
        Recursively install dependencies for an addon.

        Args:
            task_id: Task ID for status updates
            addon_name: Name of the addon whose dependencies to install
            installing_chain: Set of addon names currently being installed (prevents circular deps)
        """
        registry = get_addon_registry()
        state = self._load_state()

        addon = registry.get(addon_name)
        if not addon:
            return

        dependencies = addon.get("dependencies", [])
        if not dependencies:
            return

        for dep_name in dependencies:
            # Skip if already in installation chain (circular dependency)
            if dep_name in installing_chain:
                logger.warning(f"Circular dependency detected: {dep_name} already in chain")
                continue

            # Skip if already installed
            if dep_name in state.get("installed_addons", {}):
                logger.info(f"Dependency {dep_name} already installed, skipping")
                continue

            # Validate dependency exists
            if dep_name not in registry:
                raise ValueError(f"Dependency not found: {dep_name}")

            logger.info(f"Installing dependency: {dep_name} for {addon_name}")
            await self._update_task_status_async(
                task_id, "in_progress", 10, f"Installing dependency: {dep_name}..."
            )

            # Add to chain before recursing
            installing_chain.add(dep_name)

            # Recursively install dependencies of this dependency
            await self._install_dependencies(task_id, dep_name, installing_chain)

            # Install this dependency
            await self._install_single_addon(task_id, dep_name, restart=False)

            # Remove from chain after installation
            installing_chain.discard(dep_name)

    async def _install_single_addon(
        self, task_id: str, addon_name: str, restart: bool
    ) -> None:
        """Install a single addon without dependencies."""
        registry = get_addon_registry()
        addon = registry[addon_name]
        packages = addon.get("packages", [])

        if not packages:
            # Meta-addon with no packages (only dependencies)
            logger.info(f"Addon {addon_name} has no packages (meta-addon)")
            await self._mark_installed(addon_name)
            return

        # Determine the component directory to install packages into
        component = addon["component"]
        if component == "universal-runtime":
            component_dir = Path(__file__).parent.parent.parent.parent.parent / "runtimes" / "universal"
        else:
            raise ValueError(f"Unsupported component: {component}")

        if not component_dir.exists():
            raise ValueError(f"Component directory not found: {component_dir}")

        # Install packages using uv
        await self._update_task_status_async(
            task_id, "in_progress", 20, f"Installing {len(packages)} package(s) for {addon_name}..."
        )

        for i, package in enumerate(packages):
            progress = 20 + int((i / len(packages)) * 50)
            await self._update_task_status_async(
                task_id, "in_progress", progress, f"Installing {package}..."
            )

            # Clear VIRTUAL_ENV to prevent uv from targeting wrong environment
            env = dict(os.environ)
            env.pop("VIRTUAL_ENV", None)

            result = await asyncio.to_thread(
                subprocess.run,
                ["uv", "add", package],
                cwd=component_dir,
                env=env,
                check=True,
                capture_output=True,
                text=True,
                timeout=180,  # 3 minute timeout per package
            )

            # Log output for debugging
            if result.stdout:
                logger.info(f"uv add {package} stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"uv add {package} stderr: {result.stderr}")

        # Mark addon as installed in state
        await self._mark_installed(addon_name)
        logger.info(f"Successfully installed {addon_name}")

    async def install_addon_task(self, task_id: str, addon_name: str, restart: bool):
        """Background task to install an addon.

        Note: Service restart is handled manually by the user after installation.
        The restart parameter is kept for API compatibility but is not used.
        """
        try:
            # Validate addon name before using it
            self._validate_addon_name(addon_name)

            await self._update_task_status_async(
                task_id, "in_progress", 0, "Starting installation..."
            )

            # Get addon info from registry
            registry = get_addon_registry()
            if addon_name not in registry:
                raise ValueError(f"Addon not found: {addon_name}")

            # Install dependencies first (recursively)
            installing_chain: set[str] = {addon_name}
            await self._install_dependencies(task_id, addon_name, installing_chain)

            # Install the addon itself
            await self._install_single_addon(task_id, addon_name, restart=False)

            await self._update_task_status_async(
                task_id, "completed", 100, "Installation complete! Restart universal-runtime to use the addon."
            )

        except ValueError as e:
            logger.error(f"Validation error installing addon {addon_name}: {e}")
            await self._update_task_status_async(
                task_id, "failed", 0, "Validation failed", str(e)
            )
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            logger.error(f"Failed to install addon {addon_name}: {error_msg}")
            await self._update_task_status_async(
                task_id, "failed", 0, "Installation failed", error_msg
            )
        except subprocess.TimeoutExpired as e:
            logger.error(f"Timeout installing addon {addon_name}: {e}")
            await self._update_task_status_async(
                task_id, "failed", 0, "Installation timeout", str(e)
            )
        except Exception as e:
            logger.error(f"Unexpected error installing addon {addon_name}: {e}")
            await self._update_task_status_async(
                task_id, "failed", 0, "Installation failed", str(e)
            )

    async def uninstall_addon(self, addon_name: str):
        """Uninstall an addon."""
        self._validate_addon_name(addon_name)

        # Get addon info from registry
        registry = get_addon_registry()
        if addon_name not in registry:
            raise ValueError(f"Addon not found: {addon_name}")

        addon = registry[addon_name]
        packages = addon.get("packages", [])

        # Determine the component directory
        component = addon["component"]
        if component == "universal-runtime":
            component_dir = Path(__file__).parent.parent.parent.parent.parent / "runtimes" / "universal"
        else:
            raise ValueError(f"Unsupported component: {component}")

        # Remove packages using uv
        for package in packages:
            # Extract package name from version specifier (e.g., "faster-whisper>=1.0.0" -> "faster-whisper")
            package_name = package.split(">=")[0].split("==")[0].split("<")[0].split(">")[0].strip()

            # Skip URL packages for now
            if package.startswith("http"):
                logger.warning(f"Skipping URL package removal: {package}")
                continue

            try:
                # Clear VIRTUAL_ENV to prevent uv from targeting wrong environment
                env = dict(os.environ)
                env.pop("VIRTUAL_ENV", None)

                result = await asyncio.to_thread(
                    subprocess.run,
                    ["uv", "remove", package_name],
                    cwd=component_dir,
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

                # Log output for debugging
                if result.stdout:
                    logger.info(f"uv remove {package_name} stdout: {result.stdout}")
                if result.stderr:
                    logger.warning(f"uv remove {package_name} stderr: {result.stderr}")

            except subprocess.CalledProcessError as e:
                error_msg = e.stderr if e.stderr else str(e)
                logger.warning(f"Failed to remove package {package_name}: {error_msg}")

        # Mark as uninstalled
        await self._mark_uninstalled(addon_name)

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
        cli_path = shutil.which("lf")
        if cli_path:
            return cli_path

        # Check LF_DATA_DIR/bin/ (respects LF_DATA_DIR env var)
        data_dir_bin = Path(settings.lf_data_dir) / "bin" / "lf"
        if data_dir_bin.exists():
            return str(data_dir_bin)

        raise FileNotFoundError("CLI binary 'lf' not found")

    def _load_state(self) -> dict:
        """Load addon state from file."""
        if not self.state_file.exists():
            return {"version": "1", "installed_addons": {}}

        with open(self.state_file) as f:
            return json.load(f)

    async def _mark_installed(self, addon_name: str):
        """Mark an addon as installed in the state file."""
        # Ensure directory exists
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        state = self._load_state()
        if "installed_addons" not in state:
            state["installed_addons"] = {}

        state["installed_addons"][addon_name] = {
            "installed_at": datetime.now().isoformat()
        }

        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    async def _mark_uninstalled(self, addon_name: str):
        """Mark an addon as uninstalled in the state file."""
        state = self._load_state()
        if "installed_addons" in state and addon_name in state["installed_addons"]:
            del state["installed_addons"][addon_name]

            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)

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
