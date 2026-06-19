from pathlib import Path

from sutra.core.artifact.atomic_writer import READY_SENTINEL
from sutra.core.embedder.fixture import FixtureEmbedder
from sutra.core.extractor.adapters.python import PythonAdapter
from sutra.core.indexer import Indexer
from sutra.core.output.json_graph_exporter import JsonGraphExporter

_FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_python_repo"


def test_index_publishes_ready_sentinel_last(tmp_path):
    out = tmp_path / "repo"
    Indexer(
        adapters={"python": PythonAdapter()},
        exporter=JsonGraphExporter(),
        embedder=FixtureEmbedder(),
    ).index(root=_FIXTURE_REPO, repo_url="https://github.com/test/svc", output_dir=out)

    assert (out / "graph.json").exists()
    assert (out / "embeddings.npy").exists()
    assert (out / "embeddings_index.json").exists()
    assert (out / READY_SENTINEL).exists()
    assert not (out / ".staging").exists()
