"""Bundle service — downloads components and creates distributable archives."""

import asyncio
import json
import os
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
    SIZE_ESTIMATES,
    VALID_ACCELERATORS,
    VALID_ARCHITECTURES,
    VALID_PLATFORMS,
    BundleManifest,
    BundleRequest,
    BundleSummary,
)

logger = FastAPIStructLogger()

REPO_OWNER = os.getenv("LF_ADDON_REPO_OWNER", "llama-farm")
REPO_NAME = os.getenv("LF_ADDON_REPO_NAME", "llamafarm")


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


async def create_bundle(
    req: BundleRequest,
) -> AsyncGenerator[str, None]:
    """Execute the bundle process, yielding SSE-formatted events."""
    ver = req.version or current_version
    if ver == "dev":
        yield _sse("error", {"message": "Cannot bundle dev version"})
        return
    if not ver.startswith("v"):
        ver = f"v{ver}"

    go_os = PLATFORM_TO_GOOS[req.platform]
    go_arch = ARCH_TO_GOARCH[req.arch]
    bundle_id = str(uuid.uuid4())[:8]

    tmp_dir = Path(tempfile.mkdtemp(prefix="llamafarm-bundle-"))
    manifest_data: dict[str, str] = {}
    steps = _build_steps(req)
    total_steps = len(steps)

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            for i, step in enumerate(steps):
                step_name = step["name"]
                yield _sse(
                    "progress",
                    {
                        "step": step_name,
                        "status": "downloading",
                        "progress": i / total_steps,
                        "stepIndex": i,
                        "totalSteps": total_steps,
                    },
                )

                try:
                    size = await step["fn"](
                        client, ver, go_os, go_arch, req, tmp_dir
                    )
                    manifest_data[step_name] = step.get("asset", step_name)
                except Exception as exc:
                    logger.error(
                        f"Bundle step {step_name} failed: {exc}"
                    )
                    yield _sse(
                        "error",
                        {"message": f"Failed at {step_name}: {exc}"},
                    )
                    return

                yield _sse(
                    "progress",
                    {
                        "step": step_name,
                        "status": "complete",
                        "progress": (i + 1) / total_steps,
                        "size": size,
                        "stepIndex": i,
                        "totalSteps": total_steps,
                    },
                )

            # Write manifest
            manifest = BundleManifest(
                id=bundle_id,
                version=ver,
                platform=req.platform,
                arch=req.arch,
                accelerator=req.accelerator,
                components=manifest_data,
                addons=req.addons,
                created_at=datetime.now(UTC).isoformat(),
            )

            manifest_path = tmp_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest.model_dump(), indent=2)
            )

            # Package step
            yield _sse(
                "progress",
                {
                    "step": "packaging",
                    "status": "downloading",
                    "progress": (total_steps) / (total_steps + 1),
                    "stepIndex": total_steps,
                    "totalSteps": total_steps + 1,
                },
            )

            filename = (
                f"llamafarm-{ver}-{req.platform}-{req.arch}"
                f"-{req.accelerator}.tar.gz"
            )
            bundles_dir = _bundles_dir()
            bundles_dir.mkdir(parents=True, exist_ok=True)
            bundle_dir = bundles_dir / bundle_id
            bundle_dir.mkdir(parents=True, exist_ok=True)

            archive_path = bundle_dir / filename
            await asyncio.to_thread(
                _create_tar_gz, str(archive_path), str(tmp_dir)
            )

            archive_size = archive_path.stat().st_size
            manifest.size = archive_size
            manifest.filename = filename

            # Write final manifest to bundle dir
            (bundle_dir / "manifest.json").write_text(
                json.dumps(manifest.model_dump(), indent=2)
            )

            yield _sse(
                "progress",
                {
                    "step": "packaging",
                    "status": "complete",
                    "progress": 1.0,
                    "stepIndex": total_steps,
                    "totalSteps": total_steps + 1,
                },
            )

            yield _sse(
                "complete",
                {
                    "id": bundle_id,
                    "filename": filename,
                    "size": archive_size,
                    "version": ver,
                    "platform": req.platform,
                    "arch": req.arch,
                    "accelerator": req.accelerator,
                },
            )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


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
            continue

    return sorted(results, key=lambda b: b.created_at, reverse=True)


def get_bundle_path(bundle_id: str) -> Path | None:
    """Get the archive path for a bundle."""
    bundle_dir = _bundles_dir() / bundle_id
    manifest_file = bundle_dir / "manifest.json"
    if not manifest_file.exists():
        return None
    try:
        data = json.loads(manifest_file.read_text())
        filename = data.get("filename", "")
        archive = bundle_dir / filename
        if archive.exists():
            return archive
    except Exception:
        pass
    return None


def delete_bundle(bundle_id: str) -> bool:
    """Delete a bundle directory. Returns True if deleted."""
    bundle_dir = _bundles_dir() / bundle_id
    if not bundle_dir.exists():
        return False
    shutil.rmtree(bundle_dir, ignore_errors=True)
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
        platform_str = f"{go_os}-{req.arch}"
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
        {"name": "cli", "fn": _download_cli},
        {"name": "server", "fn": _make_pyapp_downloader("server")},
        {"name": "rag", "fn": _make_pyapp_downloader("rag")},
        {"name": "runtime", "fn": _make_pyapp_downloader("runtime")},
    ]
    if req.accelerator != "cpu":
        steps.append({"name": "torch", "fn": _download_torch})
    for addon in req.addons:
        steps.append(
            {"name": addon, "fn": _make_addon_downloader(addon)}
        )
    return steps
