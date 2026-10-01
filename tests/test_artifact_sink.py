from pathlib import Path

from sutra.core.artifact import ArtifactSink
from sutra.core.artifact.atomic_writer import (
    ARTIFACT_FILES,
    READY_SENTINEL,
    AtomicArtifactWriter,
)


def test_atomic_writer_satisfies_sink_protocol():
    assert isinstance(AtomicArtifactWriter(), ArtifactSink)


def test_arbitrary_object_does_not_satisfy_protocol():
    assert not isinstance(object(), ArtifactSink)


def _write_bundle(staging: Path, tag: str) -> None:
    for name in ARTIFACT_FILES:
        (staging / name).write_bytes(f"{name}:{tag}".encode())


def test_commit_promotes_files_and_stamps_ready_on_this_os(tmp_path):
    # Runs the real commit path on whatever OS the suite is on.  Windows has no
    # directory fsync (os.open on a directory raises PermissionError), which
    # used to abort the commit *after* the files were promoted but *before*
    # `.ready` was written — a bundle `sutra serve` would silently ignore.
    out = tmp_path / "repo"
    AtomicArtifactWriter().commit(out, lambda s: _write_bundle(s, "gen1"), generation="gen1")

    assert (out / READY_SENTINEL).read_text(encoding="utf-8") == "gen1"
    for name in ARTIFACT_FILES:
        assert (out / name).read_bytes() == f"{name}:gen1".encode()
    assert not (out / ".staging").exists()

    # A second generation keeps one .prev and re-stamps the sentinel.
    AtomicArtifactWriter().commit(out, lambda s: _write_bundle(s, "gen2"), generation="gen2")
    assert (out / READY_SENTINEL).read_text(encoding="utf-8") == "gen2"
    for name in ARTIFACT_FILES:
        assert (out / name).read_bytes() == f"{name}:gen2".encode()
        assert (out / f"{name}.prev").read_bytes() == f"{name}:gen1".encode()
