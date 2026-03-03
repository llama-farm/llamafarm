"""Bundle service — downloads components and creates distributable archives."""

import asyncio
import json
import os
import re
import shutil
import tarfile
import tempfile
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from pathlib import Path

import httpx

from core.logging import FastAPIStructLogger
from core.settings import settings
from core.version import version as current_version

from .types import (
    ACCELERATOR_PLATFORM_RULES,
    ARCH_TO_GOARCH,
    INVALID_COMBOS,
    PLATFORM_TO_GOOS,
    PLATFORM_TO_PYAPP_OS,
    SIZE_ESTIMATES,
    VALID_ACCELERATORS,
    VALID_ADDONS,
    VALID_ARCHITECTURES,
    VALID_PLATFORMS,
    BundleManifest,
    BundleRequest,
    BundleSummary,
)

logger = FastAPIStructLogger()

REPO_OWNER = os.getenv("LF_ADDON_REPO_OWNER", "llama-farm")
REPO_NAME = os.getenv("LF_ADDON_REPO_NAME", "llamafarm")
DRY_RUN = os.getenv("LF_BUNDLE_DRY_RUN", "").lower() in ("1", "true", "yes")

_latest_release_cache: dict[str, str | None] = {}

_VERSION_RE = re.compile(r'^v?\d+\.\d+\.\d+')

# Safe name pattern: alphanumeric, dots, hyphens, underscores only.
# Used to validate any user-provided value that touches filesystem paths.
_SAFE_NAME_RE = re.compile(r'^[a-zA-Z0-9._-]+$')


def _is_valid_version(ver: str) -> bool:
    """Check if a version string looks like a semver release."""
    return bool(_VERSION_RE.match(ver))


async def _get_latest_release_tag() -> str | None:
    """Fetch the latest release tag from GitHub. Cached for the process lifetime."""
    if "tag" in _latest_release_cache:
        return _latest_release_cache["tag"]
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/releases/latest",
                headers={"Accept": "application/vnd.github+json"},
            )
            resp.raise_for_status()
            tag = resp.json().get("tag_name")
            _latest_release_cache["tag"] = tag
            return tag
    except Exception as exc:
        logger.warning(f"Failed to fetch latest release: {exc}")
        _latest_release_cache["tag"] = None
        return None


async def get_latest_version() -> str:
    """Return the version to use for bundling (current or latest release)."""
    ver = current_version
    if not _is_valid_version(ver):
        tag = await _get_latest_release_tag()
        return tag or "dev"
    return ver


def _bundles_dir() -> Path:
    return Path(settings.lf_data_dir).resolve() / "bundles"


def _addon_platform_string(platform: str, arch: str) -> str:
    """Get the addon wheel archive platform string."""
    if platform == "darwin":
        return f"macos-{arch}"
    return f"{platform}-{arch}"


def validate_request(req: BundleRequest) -> str | None:
    """Return an error message if the request is invalid, else None."""
    if req.platform not in VALID_PLATFORMS:
        return f"Invalid platform '{req.platform}'"
    if req.arch not in VALID_ARCHITECTURES:
        return f"Invalid arch '{req.arch}'"
    if req.accelerator not in VALID_ACCELERATORS:
        return f"Invalid accelerator '{req.accelerator}'"
    if (req.platform, req.arch) in INVALID_COMBOS:
        return f"Unsupported combo: {req.platform}/{req.arch}"
    allowed = ACCELERATOR_PLATFORM_RULES.get(req.accelerator)
    if allowed and req.platform not in allowed:
        return (
            f"Accelerator '{req.accelerator}' not available on {req.platform}"
        )
    for addon in req.addons:
        if addon not in VALID_ADDONS:
            return f"Invalid addon '{addon}'"
    if req.version and not _SAFE_NAME_RE.match(req.version):
        return f"Invalid version string"
    return None


def estimate_size(req: BundleRequest) -> dict[str, int]:
    """Return per-component size estimates."""
    components: dict[str, int] = {
        "cli": SIZE_ESTIMATES["cli"],
        "server": SIZE_ESTIMATES["server"],
        "rag": SIZE_ESTIMATES["rag"],
        "runtime": SIZE_ESTIMATES["runtime"],
    }
    if req.accelerator != "cpu":
        key = f"torch_{req.accelerator}"
        components["torch"] = SIZE_ESTIMATES.get(key, 800_000_000)
    for addon in req.addons:
        key = f"addon_{addon}"
        components[addon] = SIZE_ESTIMATES.get(key, 200_000_000)
    return components


async def _download_asset(
    client: httpx.AsyncClient,
    version: str,
    asset_name: str,
    dest: Path,
) -> int:
    """Download a GitHub release asset. Returns bytes written."""
    if DRY_RUN:
        # Simulate download with a small placeholder file
        await asyncio.sleep(0.5)  # simulate network delay
        dest.write_bytes(b"dry-run-placeholder")
        size = SIZE_ESTIMATES.get(asset_name.split("-")[0], 50_000_000)
        return size

    url = (
        f"https://github.com/{REPO_OWNER}/{REPO_NAME}"
        f"/releases/download/{version}/{asset_name}"
    )
    async with client.stream("GET", url, follow_redirects=True) as resp:
        resp.raise_for_status()
        total = 0
        with open(dest, "wb") as f:
            async for chunk in resp.aiter_bytes(chunk_size=65536):
                f.write(chunk)
                total += len(chunk)
    return total


def _run_bundle_sync(
    req: BundleRequest,
    ver: str,
    queue: "asyncio.Queue[str | None]",
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Run the bundle process synchronously in a thread, pushing SSE events to queue."""
    go_os = PLATFORM_TO_GOOS[req.platform]
    go_arch = ARCH_TO_GOARCH[req.arch]
    bundle_id = str(uuid.uuid4())[:8]

    tmp_dir = Path(tempfile.mkdtemp(prefix="llamafarm-bundle-"))
    manifest_data: dict[str, str] = {}
    steps = _build_steps(req)
    total_steps = len(steps)

    def emit(event: str, data: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, _sse(event, data))

    try:
        import httpx as _httpx

        with _httpx.Client(timeout=300) as client:
            for i, step in enumerate(steps):
                step_name = step["name"]
                emit("progress", {
                    "step": step_name, "status": "downloading",
                    "progress": i / total_steps,
                    "stepIndex": i, "totalSteps": total_steps,
                })

                try:
                    size = _download_asset_sync(
                        client, ver, step, go_os, go_arch, req, tmp_dir
                    )
                    manifest_data[step_name] = step.get("asset", step_name)
                except Exception as exc:
                    logger.error(f"Bundle step {step_name} failed: {exc}", exc_info=True)
                    emit("error", {"message": "A bundle step failed. Check server logs for details."})
                    return

                emit("progress", {
                    "step": step_name, "status": "complete",
                    "progress": (i + 1) / total_steps,
                    "size": size, "stepIndex": i, "totalSteps": total_steps,
                })

            # Write manifest
            manifest = BundleManifest(
                id=bundle_id, version=ver,
                platform=req.platform, arch=req.arch,
                accelerator=req.accelerator,
                components=manifest_data, addons=req.addons,
                created_at=datetime.now(UTC).isoformat(),
            )
            (tmp_dir / "manifest.json").write_text(
                json.dumps(manifest.model_dump(), indent=2)
            )

            # Package step
            emit("progress", {
                "step": "packaging", "status": "downloading",
                "progress": total_steps / (total_steps + 1),
                "stepIndex": total_steps, "totalSteps": total_steps + 1,
            })

            filename = (
                f"llamafarm-{ver}-{req.platform}-{req.arch}"
                f"-{req.accelerator}.tar.gz"
            )
            bundles_dir = _bundles_dir()
            bundles_dir.mkdir(parents=True, exist_ok=True)
            bundle_dir = bundles_dir / bundle_id
            bundle_dir.mkdir(parents=True, exist_ok=True)

            archive_path = bundle_dir / filename
            _create_tar_gz(str(archive_path), str(tmp_dir))

            archive_size = archive_path.stat().st_size
            manifest.size = archive_size
            manifest.filename = filename

            (bundle_dir / "manifest.json").write_text(
                json.dumps(manifest.model_dump(), indent=2)
            )

            emit("progress", {
                "step": "packaging", "status": "complete",
                "progress": 1.0,
                "stepIndex": total_steps, "totalSteps": total_steps + 1,
            })

            emit("complete", {
                "id": bundle_id, "filename": filename,
                "size": archive_size, "version": ver,
                "platform": req.platform, "arch": req.arch,
                "accelerator": req.accelerator,
            })
    except Exception as exc:
        logger.error(f"Bundle failed: {exc}", exc_info=True)
        emit("error", {"message": "Bundle failed unexpectedly. Check server logs."})
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        loop.call_soon_threadsafe(queue.put_nowait, None)  # signal done


def _download_asset_sync(
    client: "httpx.Client",
    ver: str,
    step: dict,
    go_os: str,
    go_arch: str,
    req: BundleRequest,
    tmp_dir: Path,
) -> int:
    """Synchronous version of asset download for thread-based bundle."""
    import time as _time

    step_name = step["name"]

    if DRY_RUN:
        _time.sleep(0.3)
        placeholder = tmp_dir / f"{step_name}-placeholder"
        placeholder.write_bytes(b"dry-run")
        return SIZE_ESTIMATES.get(step_name, 50_000_000)

    # Build asset name based on step
    if step_name == "cli":
        name = f"lf-{go_os}-{ARCH_TO_GOARCH[req.arch]}"
        if req.platform == "windows":
            name += ".exe"
    elif step_name == "torch":
        name = f"torch-{req.accelerator}-{req.platform}-{req.arch}.tar.gz"
        dest_dir = tmp_dir / "torch"
        dest_dir.mkdir(exist_ok=True)
        return _download_release_sync(client, ver, name, dest_dir / name)
    elif step_name in ("server", "rag", "runtime"):
        pyapp_os = PLATFORM_TO_PYAPP_OS[req.platform]
        platform_str = f"{pyapp_os}-{req.arch}"
        name = f"llamafarm-{step_name}-{platform_str}"
        if req.platform == "windows":
            name += ".exe"
    else:
        # Addon
        addon_dir = tmp_dir / "addons"
        addon_dir.mkdir(exist_ok=True)
        plat = _addon_platform_string(req.platform, req.arch)
        name = f"{step_name}-wheels-{plat}.tar.gz"
        return _download_release_sync(client, ver, name, addon_dir / name)

    return _download_release_sync(client, ver, name, tmp_dir / name)


def _download_release_sync(
    client: "httpx.Client",
    version: str,
    asset_name: str,
    dest: Path,
) -> int:
    """Download a GitHub release asset synchronously."""
    url = (
        f"https://github.com/{REPO_OWNER}/{REPO_NAME}"
        f"/releases/download/{version}/{asset_name}"
    )
    with client.stream("GET", url, follow_redirects=True) as resp:
        resp.raise_for_status()
        total = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=65536):
                f.write(chunk)
                total += len(chunk)
    return total


async def create_bundle(
    req: BundleRequest,
) -> AsyncGenerator[str, None]:
    """Execute the bundle process, yielding SSE-formatted events.

    Runs the actual work in a background thread to avoid blocking the event loop.
    """
    ver = req.version or current_version
    if not _is_valid_version(ver):
        ver = await _get_latest_release_tag()
        if not ver:
            yield _sse("error", {"message": "Cannot bundle dev version — specify a version or ensure GitHub releases exist"})
            return
    if not ver.startswith("v"):
        ver = f"v{ver}"

    queue: asyncio.Queue[str | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    # Run bundle work in a thread so the event loop stays free
    loop.run_in_executor(None, _run_bundle_sync, req, ver, queue, loop)

    # Yield SSE events as they arrive from the thread
    while True:
        event = await queue.get()
        if event is None:
            break
        yield event


def list_bundles() -> list[BundleSummary]:
    """List all completed bundles."""
    bundles_dir = _bundles_dir()
    if not bundles_dir.exists():
        return []

    results = []
    for entry in sorted(bundles_dir.iterdir()):
        manifest_file = entry / "manifest.json"
        if not manifest_file.exists():
            continue
        try:
            data = json.loads(manifest_file.read_text())
            results.append(BundleSummary(**data))
        except Exception:
            logger.warning(f"Skipping invalid bundle manifest: {manifest_file}")
            continue

    return sorted(results, key=lambda b: b.created_at, reverse=True)


_SAFE_ID_RE = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9\-]{0,63}$')
_SAFE_FILENAME_RE = re.compile(r'^[a-zA-Z0-9._\-]+$')


def get_bundle_path(bundle_id: str) -> Path | None:
    """Get the archive path for a bundle.

    Sanitisation is inlined so that static analysers (CodeQL) can verify
    each filesystem access is guarded by a ``startswith`` containment check.
    """
    if not _SAFE_ID_RE.fullmatch(bundle_id):
        logger.warning(f"Invalid bundle_id rejected: {bundle_id!r}")
        return None

    safe_root = os.path.realpath(_bundles_dir())
    real_dir = os.path.realpath(os.path.join(safe_root, bundle_id))
    if not real_dir.startswith(safe_root + os.sep):
        logger.warning(f"Path traversal blocked for bundle_id: {bundle_id!r}")
        return None

    if not os.path.isdir(real_dir):
        return None

    manifest_path = os.path.join(real_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        return None

    try:
        data = json.loads(Path(manifest_path).read_text())
    except Exception:
        logger.exception(f"Failed to read manifest for bundle {bundle_id!r}")
        return None

    filename = data.get("filename", "")
    if not filename or not _SAFE_FILENAME_RE.fullmatch(filename):
        return None

    real_archive = os.path.realpath(os.path.join(real_dir, filename))
    if not real_archive.startswith(real_dir + os.sep):
        return None

    if os.path.isfile(real_archive):
        return Path(real_archive)
    return None


def delete_bundle(bundle_id: str) -> bool:
    """Delete a bundle directory. Returns True if deleted."""
    if not _SAFE_ID_RE.fullmatch(bundle_id):
        return False

    safe_root = os.path.realpath(_bundles_dir())
    real_dir = os.path.realpath(os.path.join(safe_root, bundle_id))
    if not real_dir.startswith(safe_root + os.sep):
        return False

    if not os.path.isdir(real_dir):
        return False
    shutil.rmtree(real_dir, ignore_errors=True)
    return True


# --- Internal helpers ---


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _download_cli(
    client: httpx.AsyncClient,
    ver: str,
    go_os: str,
    go_arch: str,
    req: BundleRequest,
    tmp_dir: Path,
) -> int:
    name = f"lf-{go_os}-{go_arch}"
    if req.platform == "windows":
        name += ".exe"
    return await _download_asset(client, ver, name, tmp_dir / name)


def _make_pyapp_downloader(component: str):
    async def _download(
        client: httpx.AsyncClient,
        ver: str,
        go_os: str,
        go_arch: str,
        req: BundleRequest,
        tmp_dir: Path,
    ) -> int:
        pyapp_os = PLATFORM_TO_PYAPP_OS[req.platform]
        platform_str = f"{pyapp_os}-{req.arch}"
        name = f"llamafarm-{component}-{platform_str}"
        if req.platform == "windows":
            name += ".exe"
        return await _download_asset(client, ver, name, tmp_dir / name)

    return _download


async def _download_torch(
    client: httpx.AsyncClient,
    ver: str,
    go_os: str,
    go_arch: str,
    req: BundleRequest,
    tmp_dir: Path,
) -> int:
    torch_dir = tmp_dir / "torch"
    torch_dir.mkdir(exist_ok=True)
    name = f"torch-{req.accelerator}-{req.platform}-{req.arch}.tar.gz"
    return await _download_asset(client, ver, name, torch_dir / name)


def _make_addon_downloader(addon: str):
    async def _download(
        client: httpx.AsyncClient,
        ver: str,
        go_os: str,
        go_arch: str,
        req: BundleRequest,
        tmp_dir: Path,
    ) -> int:
        addon_dir = tmp_dir / "addons"
        addon_dir.mkdir(exist_ok=True)
        plat = _addon_platform_string(req.platform, req.arch)
        name = f"{addon}-wheels-{plat}.tar.gz"
        return await _download_asset(client, ver, name, addon_dir / name)

    return _download


def _create_tar_gz(output_path: str, source_dir: str) -> None:
    """Create a tar.gz archive of source_dir."""
    with tarfile.open(output_path, "w:gz") as tar:
        for entry in Path(source_dir).iterdir():
            tar.add(str(entry), arcname=entry.name)


def _build_steps(req: BundleRequest) -> list[dict]:
    steps: list[dict] = [
        {"name": "cli"},
        {"name": "server"},
        {"name": "rag"},
        {"name": "runtime"},
    ]
    if req.accelerator != "cpu":
        steps.append({"name": "torch"})
    for addon in req.addons:
        steps.append({"name": addon})
    return steps
