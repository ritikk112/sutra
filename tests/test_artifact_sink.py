from sutra.core.artifact import ArtifactSink
from sutra.core.artifact.atomic_writer import AtomicArtifactWriter


def test_atomic_writer_satisfies_sink_protocol():
    assert isinstance(AtomicArtifactWriter(), ArtifactSink)


def test_arbitrary_object_does_not_satisfy_protocol():
    assert not isinstance(object(), ArtifactSink)
