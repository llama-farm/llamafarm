from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Optional

import httpx

GITHUB_LATEST_RELEASE_URL = (
    "https://api.github.com/repos/llama-farm/llamafarm/releases/latest"
)
CACHE_TTL = timedelta(hours=6)


@dataclass
class CLIRelease:
    tag_name: str
    html_url: str
    published_at: Optional[datetime]
    name: Optional[str]
    body: Optional[str]
    from_cache: bool = False


# Thread-safe cache using asyncio.Lock
_cache_lock = asyncio.Lock()
_cached_release: Optional[CLIRelease] = None
_cached_at: Optional[datetime] = None


async def get_latest_cli_release(force_refresh: bool = False) -> CLIRelease:
    """Fetch the latest published CLI release from GitHub with async-safe caching."""
    global _cached_at, _cached_release

    async with _cache_lock:
        now = datetime.now(UTC)
        if (
            not force_refresh
            and _cached_release
            and _cached_at
            and now - _cached_at < CACHE_TTL
        ):
            _cached_release.from_cache = True
            return _cached_release

        try:
            release = await _fetch_latest_release_from_github()
        except httpx.HTTPError as exc:
            if _cached_release:
                _cached_release.from_cache = True
                return _cached_release
            raise exc

        release.from_cache = False
        _cached_release = release
        _cached_at = now
        return release


async def _fetch_latest_release_from_github(token: str | None = None) -> CLIRelease:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "llamafarm-server",
    }

    if token:
        headers["Authorization"] = f"Bearer {token}"

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(GITHUB_LATEST_RELEASE_URL, headers=headers)
        response.raise_for_status()

    payload = response.json()
    if payload.get("draft") or payload.get("prerelease"):
        raise RuntimeError("latest release is marked as draft or prerelease")

    published_at = payload.get("published_at")
    published_dt: datetime | None = None
    if published_at:
        try:
            if published_at.endswith("Z"):
                published_at = published_at.replace("Z", "+00:00")
            published_dt = datetime.fromisoformat(published_at)
        except ValueError:
            published_dt = None

    return CLIRelease(
        tag_name=payload.get("tag_name", ""),
        html_url=payload.get("html_url", ""),
        published_at=published_dt,
        name=payload.get("name"),
        body=payload.get("body"),
    )
