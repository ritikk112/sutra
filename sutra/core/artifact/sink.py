from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class ArtifactSink(Protocol):
    """
    Commits an artifact bundle atomically.

    `write_files(staging_dir)` must create the three artifact files inside
    the provided staging directory; the sink promotes them as a unit and
    stamps the `.ready` sentinel last.  AtomicArtifactWriter is the
    production implementation.
    """

    def commit(
        self,
        artifact_dir: Path,
        write_files: Callable[[Path], None],
        generation: str = "",
    ) -> None: ...
