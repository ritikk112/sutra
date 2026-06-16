from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Callable, Optional

from sutra.core.artifact.atomic_writer import READY_SENTINEL

# Poll cadence + debounce.  Polling (vs inotify) keeps this dependency-free
# and works on every filesystem an artifact directory might live on
# (rsync targets, network mounts, Dropbox-style sync folders).
DEFAULT_POLL_SECONDS = 2.0


class ArtifactWatcher:
    """
    Watches `.ready` sentinels — NEVER the data files — under an artifacts
    root, and fires a callback per changed artifact directory.

    Debounce by design: the AtomicArtifactWriter removes `.ready` before
    touching data files and rewrites it only after the bundle commit, so
    a sentinel mtime change IS the commit boundary.  Mid-write states have
    no sentinel and therefore never fire.

    The callback runs on the watcher thread; it must swallow its own
    errors (the registry-swap callback logs-and-keeps-old on torn loads).
    """

    def __init__(
        self,
        root: Path | str,
        on_ready_change: Callable[[Path], None],
        poll_seconds: float = DEFAULT_POLL_SECONDS,
    ) -> None:
        self._root = Path(root)
        self._callback = on_ready_change
        self._poll = poll_seconds
        self._mtimes: dict[Path, float] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Prime with current state so startup doesn't re-fire every repo.
        for sentinel in self._root.glob(f"*/{READY_SENTINEL}"):
            self._mtimes[sentinel.parent] = sentinel.stat().st_mtime

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="sutra-artifact-watcher", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._poll * 2)

    def check_once(self) -> list[Path]:
        """One poll pass — also the test seam.  Returns dirs that fired."""
        fired: list[Path] = []
        for sentinel in self._root.glob(f"*/{READY_SENTINEL}"):
            artifact_dir = sentinel.parent
            try:
                mtime = sentinel.stat().st_mtime
            except FileNotFoundError:
                continue   # raced a commit; next pass catches the rewrite
            previous = self._mtimes.get(artifact_dir)
            if previous is None or mtime > previous:
                self._mtimes[artifact_dir] = mtime
                fired.append(artifact_dir)
                try:
                    self._callback(artifact_dir)
                except Exception as exc:   # noqa: BLE001 — watcher must survive
                    print(
                        f"[sutra-mcp] reload callback failed for "
                        f"{artifact_dir.name}: {exc}",
                        file=sys.stderr,
                    )
        return fired

    def _run(self) -> None:
        while not self._stop.wait(self._poll):
            self.check_once()
