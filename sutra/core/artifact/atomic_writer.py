from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

# The files that constitute one artifact generation.  They commit together
# because the consumer loads them together.
ARTIFACT_FILES = ("graph.json", "embeddings.npy", "embeddings_index.json")

READY_SENTINEL = ".ready"


class AtomicArtifactWriter:
    """
    Bundle-commit for artifact directories (design-decision component).

    Protocol per commit:
      1. Remove `.ready` — watchers stop treating the directory as fresh.
      2. Retain the previous generation: each current file is renamed to
         `<name>.prev` (one generation kept, older .prev overwritten).
      3. Each new file is written to `<name>.tmp`, fsync'd, then
         os.replace'd into place (atomic per file on POSIX).
      4. fsync the directory entry.
      5. Write `.ready` LAST (content: a generation marker).  The watcher
         watches ONLY this sentinel.

    Crash-consistency story: a crash mid-commit leaves no `.ready`, so
    hot-reload never fires on a torn directory; a manual load of a mixed
    generation is caught by ArtifactLoader's torn-artifact cross-checks
    (row count ↔ index length ↔ embedding_id agreement) and rejected,
    keeping the previous in-memory snapshot serving.
    """

    def commit(
        self,
        artifact_dir: Path | str,
        write_files: Callable[[Path], None],
        generation: str = "",
    ) -> None:
        """
        Run one bundle commit.

        `write_files(staging_dir)` must create ARTIFACT_FILES inside the
        provided staging directory; this writer then promotes them as a
        unit and stamps the sentinel.
        """
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        ready = artifact_dir / READY_SENTINEL
        if ready.exists():
            ready.unlink()

        staging = artifact_dir / ".staging"
        staging.mkdir(exist_ok=True)
        try:
            write_files(staging)

            missing = [f for f in ARTIFACT_FILES if not (staging / f).exists()]
            if missing:
                raise FileNotFoundError(
                    f"write_files() did not produce: {missing} — refusing a "
                    f"partial artifact commit."
                )

            # Retain one previous generation, then promote the new files.
            for name in ARTIFACT_FILES:
                current = artifact_dir / name
                if current.exists():
                    os.replace(current, artifact_dir / f"{name}.prev")
            for name in ARTIFACT_FILES:
                src = staging / name
                with open(src, "rb+") as fh:
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(src, artifact_dir / name)

            # fsync the directory so the renames are durable.
            dir_fd = os.open(artifact_dir, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)

            ready.write_text(generation or "ready", encoding="utf-8")
        finally:
            # Clean the staging area whatever happened.
            for leftover in staging.glob("*"):
                leftover.unlink()
            staging.rmdir()
