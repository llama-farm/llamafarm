from datetime import UTC

import httpx
from fastapi import APIRouter, HTTPException

from services.upgrade_service import get_latest_cli_release

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/version-check")
async def cli_upgrade():
    try:
        release = await get_latest_cli_release()
    except RuntimeError as exc:  # draft/prerelease surfaced
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503, detail="Unable to reach GitHub releases"
        ) from exc

    published_at = None
    if release.published_at:
        published_at = release.published_at.astimezone(UTC).isoformat()

    return {
        "latest_version": release.tag_name,
        "name": release.name,
        "release_notes": release.body,
        "release_url": release.html_url,
        "published_at": published_at,
        "from_cache": release.from_cache,
        "install": {
            "mac_linux": (
                "curl -fsSL https://raw.githubusercontent.com/"
                "llama-farm/llamafarm/main/install.sh | bash"
            ),
            "windows": "winget install LlamaFarm.CLI",
        },
    }
