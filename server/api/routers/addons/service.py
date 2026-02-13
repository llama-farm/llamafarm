"""Addon service implementation.

Architecture note: The server tries CLI-based wheel installation first (fast,
pre-built packages from GitHub releases). When that fails (e.g., development
builds where wheels aren't published), it falls back to installing packages
directly via uv pip install from PyPI.

The --no-restart flag tells the CLI to skip its stop/restart cycle. After the
install finishes, this service restarts only the affected component (never the
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
        self.install_lock = asyncio.Lock()
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

    async def install_addon_task(self, task_id: str, addon_name: str, restart: bool):
        """Background task to install an addon.

        Tries CLI-based wheel installation first (fast, pre-built packages).
        Falls back to direct package installation via uv when CLI fails
        (e.g., development builds where wheels aren't published).
        """
        try:
            self._validate_addon_name(addon_name)

            if self.install_lock.locked():
                await self._update_task_status_async(
                    task_id,
                    "failed",
                    0,
                    "Another install/uninstall operation is in progress",
                )
                return

            async with self.install_lock:
                await self._do_install(task_id, addon_name, restart)

        except ValueError as e:
            logger.error(f"Validation error installing addon {addon_name}: {e}")
            await self._update_task_status_async(
                task_id, "failed", 0, "Validation failed", str(e)
            )
        except Exception as e:
            logger.error(f"Failed to install addon {addon_name}: {e}")
            await self._update_task_status_async(
                task_id, "failed", 0, "Installation failed", str(e)
            )

    async def _do_install(self, task_id: str, addon_name: str, restart: bool):
        """Inner install logic, called under install_lock."""
        await self._update_task_status_async(
            task_id, "in_progress", 0, "Starting installation..."
        )

        # Try CLI-based install first (pre-built wheel bundles from releases)
        cli_success = False
        try:
            cli_path = self._find_cli_binary()
            await self._update_task_status_async(
                task_id, "in_progress", 10, "Downloading addon packages..."
            )

            result = await asyncio.to_thread(
                subprocess.run,
                [cli_path, "addons", "install", "--no-restart", addon_name],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode == 0:
                cli_success = True
                install_output = result.stdout.strip()
                await self._update_task_status_async(
                    task_id,
                    "in_progress",
                    70,
                    f"Addon installed. {self._summarize_output(install_output)}",
                )
            else:
                error_msg = (result.stderr or result.stdout or "").strip()
                logger.warning(
                    f"CLI install failed for {addon_name}, "
                    f"falling back to direct install: {error_msg}"
                )
        except FileNotFoundError:
            logger.warning("CLI binary not found, falling back to direct install")
        except subprocess.TimeoutExpired:
            logger.warning("CLI install timed out, falling back to direct install")

        # Fall back to direct package installation via uv
        if not cli_success:
            await self._install_directly(task_id, addon_name)

        # Restart the affected component (not the server)
        restart_set_terminal = False
        if restart:
            restart_set_terminal = await self._restart_component(
                task_id, addon_name
            )

        reload_addon_registry()

        if not restart_set_terminal:
            await self._update_task_status_async(
                task_id, "completed", 100, "Installation complete"
            )

    async def uninstall_addon(self, addon_name: str):
        """Uninstall an addon.

        Checks install method from state to determine cleanup strategy:
        - "direct": removes packages via uv pip uninstall
        - "cli" (default): delegates to CLI binary
        """
        self._validate_addon_name(addon_name)

        async with self.install_lock:
            await self._do_uninstall(addon_name)

    async def _do_uninstall(self, addon_name: str):
        """Inner uninstall logic, called under install_lock."""
        registry = get_addon_registry()
        addon_def = registry.get(addon_name, {})
        component = addon_def.get("component", "")

        # Check install method from state
        state = self._load_state()
        installed_info = state.get("installed_addons", {}).get(addon_name)
        install_method = (
            installed_info.get("install_method", "cli") if installed_info else "cli"
        )

        if install_method == "direct":
            await self._uninstall_directly(addon_name, addon_def)
        else:
            cli_path = self._find_cli_binary()
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
            try:
                cli_path = self._find_cli_binary()
                restart_result = await asyncio.to_thread(
                    subprocess.run,
                    [cli_path, "services", "start", component],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if restart_result.returncode != 0:
                    logger.warning(
                        f"Failed to restart {component} after uninstall: "
                        f"{restart_result.stderr}"
                    )
            except FileNotFoundError:
                logger.warning(
                    f"CLI not found for restart after uninstall of {addon_name}"
                )

    async def get_task_status(self, task_id: str) -> AddonTaskStatus | None:
        """Get the status of a task (holds lock for consistency with updates)."""
        async with self.task_status_lock:
            return self.task_statuses.get(task_id)

    # ── Direct install/uninstall (uv pip fallback) ──────────────────────

    async def _install_directly(self, task_id: str, addon_name: str):
        """Install addon packages directly via uv pip install.

        Fallback when CLI wheel download is unavailable (e.g., development builds
        where pre-built wheels haven't been published to GitHub releases).
        Resolves the full dependency chain and installs all packages at once.
        """
        registry = get_addon_registry()
        install_order = self._resolve_dependencies(addon_name, registry)

        # Skip already-installed addons
        state = self._load_state()
        installed = state.get("installed_addons", {})
        to_install = [n for n in install_order if n not in installed]

        if not to_install:
            return

        # Collect packages from all addons to install
        all_packages: list[str] = []
        component = ""
        for name in to_install:
            addon_def = registry[name]
            all_packages.extend(addon_def.get("packages", []))
            if not component:
                component = addon_def.get("component", "")

        if all_packages:
            # Validate all addons target the same component
            components = {
                registry[n].get("component", "") for n in to_install
            } - {""}
            if len(components) > 1:
                raise ValueError(
                    f"Cross-component addon dependencies not supported. "
                    f"Addons span components: {', '.join(sorted(components))}"
                )

            project_dir = self._get_component_project_dir(component)

            await self._update_task_status_async(
                task_id,
                "in_progress",
                20,
                f"Installing {len(all_packages)} packages from PyPI...",
            )

            uv_path = shutil.which("uv")
            if not uv_path:
                raise FileNotFoundError("uv not found in PATH")

            result = await asyncio.to_thread(
                subprocess.run,
                [uv_path, "pip", "install"] + all_packages,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(project_dir),
            )

            if result.returncode != 0:
                error_detail = (
                    result.stderr or result.stdout or "Unknown error"
                ).strip()
                raise RuntimeError(f"Package installation failed: {error_detail}")

            await self._update_task_status_async(
                task_id, "in_progress", 60, "Packages installed, updating state..."
            )

        # Mark all addons in the chain as installed
        for name in to_install:
            self._save_addon_state(name, registry[name], install_method="direct")

        await self._update_task_status_async(
            task_id, "in_progress", 70, "Packages installed successfully"
        )

    async def _uninstall_directly(self, addon_name: str, addon_def: dict):
        """Uninstall packages installed via direct uv pip install."""
        packages = addon_def.get("packages", [])
        component = addon_def.get("component", "")

        if packages and component:
            project_dir = self._get_component_project_dir(component)
            uv_path = shutil.which("uv")
            if uv_path:
                # Strip version specifiers for uninstall
                pkg_names = self._extract_package_names(packages)

                if pkg_names:
                    result = await asyncio.to_thread(
                        subprocess.run,
                        [uv_path, "pip", "uninstall"] + pkg_names,
                        capture_output=True,
                        text=True,
                        timeout=120,
                        cwd=str(project_dir),
                    )
                    if result.returncode != 0:
                        logger.warning(
                            f"Failed to uninstall packages for {addon_name}: "
                            f"{result.stderr}"
                        )
            else:
                logger.warning(
                    f"uv not found in PATH; cannot uninstall packages for "
                    f"{addon_name}. Packages may remain installed."
                )

        # Always remove from state
        self._remove_addon_state(addon_name)

    # ── Dependency resolution ───────────────────────────────────────────

    def _resolve_dependencies(
        self, addon_name: str, registry: dict
    ) -> list[str]:
        """Resolve addon dependencies in topological order (dependencies first)."""
        visited: set[str] = set()
        in_stack: set[str] = set()
        order: list[str] = []

        def visit(name: str):
            if name in visited:
                return
            if name in in_stack:
                raise ValueError(f"Circular dependency detected involving '{name}'")
            in_stack.add(name)
            addon_def = registry.get(name)
            if not addon_def:
                raise ValueError(f"Addon '{name}' not found in registry")
            for dep in addon_def.get("dependencies", []):
                visit(dep)
            in_stack.discard(name)
            visited.add(name)
            order.append(name)

        visit(addon_name)
        return order

    # ── Component restart ───────────────────────────────────────────────

    async def _restart_component(self, task_id: str, addon_name: str) -> bool:
        """Restart the component affected by an addon install.

        Returns True if it set a terminal task status (caller should not overwrite).
        """
        registry = get_addon_registry()
        addon_def = registry.get(addon_name, {})
        component = addon_def.get("component", "")

        if component == "server":
            logger.warning(
                "Addon targets server component; skipping automatic restart "
                "to avoid self-termination. Manual restart required."
            )
            return False

        if not component:
            return False

        await self._update_task_status_async(
            task_id, "in_progress", 80, f"Restarting {component}..."
        )

        try:
            cli_path = self._find_cli_binary()
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
                return True
        except FileNotFoundError:
            logger.warning("CLI not found for restart, manual restart required")
            await self._update_task_status_async(
                task_id,
                "completed",
                100,
                f"Addon installed but automatic restart unavailable. "
                f"Restart {component} manually.",
            )
            return True

        return False

    # ── Path resolution ─────────────────────────────────────────────────

    def _find_cli_binary(self) -> str:
        """Find the CLI binary path.

        Search order:
        1. dist/lf in repo root (development builds, can find addon registry)
        2. LF_DATA_DIR/bin/lf (installed binary)
        3. PATH (system-wide install)
        """
        # 1. Check repo dist/ directory (dev builds resolve registry via relative path)
        repo_root = Path(__file__).parent.parent.parent.parent.parent
        dist_bin = repo_root / "dist" / "lf"
        if dist_bin.exists():
            return str(dist_bin)

        # 2. Check LF_DATA_DIR/bin/ (installed binary)
        data_dir_bin = Path(settings.lf_data_dir) / "bin" / "lf"
        if data_dir_bin.exists():
            return str(data_dir_bin)

        # 3. Fall back to PATH
        cli_path = shutil.which("lf")
        if cli_path:
            return cli_path

        raise FileNotFoundError(
            f"CLI binary 'lf' not found in dist/, {data_dir_bin}, or PATH. "
            f"Run 'nx build cli' to build, or ensure lf is installed."
        )

    def _get_component_project_dir(self, component: str) -> Path:
        """Map component name to its project directory."""
        repo_root = Path(__file__).parent.parent.parent.parent.parent
        component_dirs = {
            "universal-runtime": repo_root / "runtimes" / "universal",
            "server": repo_root / "server",
            "rag": repo_root / "rag",
        }
        project_dir = component_dirs.get(component)
        if not project_dir or not project_dir.exists():
            raise FileNotFoundError(
                f"Project directory for component '{component}' not found"
            )
        return project_dir

    # ── State management ────────────────────────────────────────────────

    def _load_state(self) -> dict:
        """Load addon state from file."""
        if not self.state_file.exists():
            return {"version": "1", "installed_addons": {}}

        with open(self.state_file) as f:
            return json.load(f)

    def _save_addon_state(
        self, addon_name: str, addon_def: dict, install_method: str = "cli"
    ):
        """Add addon to installed state in addons.json."""
        state = self._load_state()
        state.setdefault("installed_addons", {})[addon_name] = {
            "name": addon_name,
            "version": addon_def.get("version", "unknown"),
            "component": addon_def.get("component", ""),
            "installed_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "install_method": install_method,
        }
        self.state_file.write_text(json.dumps(state, indent=2))

    def _remove_addon_state(self, addon_name: str):
        """Remove addon from installed state in addons.json."""
        state = self._load_state()
        state.get("installed_addons", {}).pop(addon_name, None)
        self.state_file.write_text(json.dumps(state, indent=2))

    # ── Helpers ─────────────────────────────────────────────────────────

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

    @staticmethod
    def _extract_package_names(packages: list[str]) -> list[str]:
        """Extract bare package names from versioned specifiers for uninstall.

        Handles both version-specifier packages (e.g., 'faster-whisper>=1.0.0')
        and URL-based packages (e.g., 'https://.../en_core_web_sm-3.8.0-...whl').
        """
        names: list[str] = []
        for pkg in packages:
            if pkg.startswith(("http://", "https://")):
                # Extract name from wheel URL filename
                wheel_name = pkg.split("/")[-1]
                pkg_name = wheel_name.split("-")[0]
                names.append(pkg_name)
            else:
                # Strip version specifiers
                pkg_name = re.split(r"[><=!~\[]", pkg)[0].strip()
                if pkg_name:
                    names.append(pkg_name)
        return names

    # ── Task status persistence ─────────────────────────────────────────

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
