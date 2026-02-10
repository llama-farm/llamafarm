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

            await asyncio.to_thread(
                subprocess.run,
                ["uv", "add", package],
                cwd=component_dir,
                check=True,
                capture_output=True,
                text=True,
                timeout=180,  # 3 minute timeout per package
            )

        # Mark addon as installed in state
        await self._mark_installed(addon_name)
        logger.info(f"Successfully installed {addon_name}")

    async def install_addon_task(self, task_id: str, addon_name: str, restart: bool):
        """Background task to install an addon."""
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

            addon = registry[addon_name]

            await self._update_task_status_async(
                task_id, "in_progress", 90, "Installation complete"
            )

            # Restart the service if requested
            if restart:
                component = addon["component"]
                logger.info(f"Restarting {component} service...")

                await self._update_task_status_async(
                    task_id, "in_progress", 95, "Restarting service..."
                )

                # Run the restart script in the background (fire and forget)
                # This script will kill the process on port 11540 and restart it
                script_path = Path(__file__).parent.parent.parent.parent / "restart_runtime.sh"

                try:
                    # Use Popen with start_new_session to run in background
                    await asyncio.to_thread(
                        subprocess.Popen,
                        [str(script_path)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                    logger.info(f"Restart script launched for {component}")
                except Exception as e:
                    logger.error(f"Failed to launch restart script: {e}")
                    # Don't fail the installation if restart fails
                    await self._update_task_status_async(
                        task_id, "completed", 100, "Installation complete! Service restart may have failed - please check."
                    )
                    return

            await self._update_task_status_async(
                task_id, "completed", 100, "Installation complete! Service restarting..."
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
                await asyncio.to_thread(
                    subprocess.run,
                    ["uv", "remove", package_name],
                    cwd=component_dir,
                    check=True,
                    timeout=60,
                )
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to remove package {package_name}: {e}")

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
