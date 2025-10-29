import asyncio, fnmatch, os
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, Optional

from huggingface_hub import HfApi, hf_hub_download
from tqdm.auto import tqdm


# ---------- helpers
def _match(
    path: str, allow: Optional[Iterable[str]], ignore: Optional[Iterable[str]]
) -> bool:
    if allow and not any(fnmatch.fnmatch(path, pat) for pat in allow):
        return False
    if ignore and any(fnmatch.fnmatch(path, pat) for pat in ignore):
        return False
    return True


# ---------- per-file tqdm that forwards byte updates
def make_file_tqdm(on_update: Callable[[Dict], None], position: int = 1):
    class FileTQDM(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("position", position)  # keep overall bar on position=0
            super().__init__(*args, **kwargs)
            try:
                on_update(
                    {
                        "event": "file_start",
                        "file": self.desc,
                        "total": int(self.total) if self.total else None,
                    }
                )
            except Exception as e:
                on_update({"event": "file_error", "file": self.desc, "error": str(e)})

        def update(self, n=1):
            r = super().update(n)
            try:
                on_update(
                    {
                        "event": "file_progress",
                        "file": self.desc,
                        "n": int(self.n),
                        "total": int(self.total) if self.total else None,
                    }
                )
            except Exception as e:
                on_update({"event": "file_error", "file": self.desc, "error": str(e)})
            return r

        def close(self):
            try:
                on_update(
                    {
                        "event": "file_end",
                        "file": self.desc,
                        "n": int(self.n),
                        "total": int(self.total) if self.total else None,
                    }
                )
            except Exception as e:
                super().close()
                on_update({"event": "file_error", "file": self.desc, "error": str(e)})
                return

    return FileTQDM


# ---------- ASYNC: yields events while running sync code in a thread
async def iter_snapshot_download_events(
    repo_id: str,
    revision: str = "main",
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    allow_patterns: Optional[Iterable[str]] = None,
    ignore_patterns: Optional[Iterable[str]] = None,
):
    """
    Async generator that yields the same event dicts you’d get via on_update in the sync version.
    Keeps default console tqdm bars visible in server logs/terminal.
    """
    queue: asyncio.Queue = asyncio.Queue()

    def push(evt: Dict):  # bridge sync -> async
        try:
            asyncio.get_running_loop().call_soon_threadsafe(queue.put_nowait, evt)
        except RuntimeError:
            # if no running loop (e.g., inside worker thread), capture loop first in caller if needed
            pass

    def work():
        snapshot_download_with_per_file_progress(
            repo_id=repo_id,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            on_update=push,
        )

    task = asyncio.to_thread(work)
    consumer = asyncio.create_task(task)
    done = False
    while not done:
        evt = await queue.get()
        yield evt
        done = evt.get("event") == "done"
    await consumer
