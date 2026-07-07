from pathlib import Path
from sutra.core.embedder.chunk_builder import build_chunks
from sutra.core.extractor.base import FunctionSymbol, Location, Visibility


def _fn(id_, name, is_local):
    return FunctionSymbol(
        id=id_, name=name, qualified_name=name, file_path="f.py",
        location=Location(1, 2, 0, 5), body_hash="sha256:x", language="python",
        visibility=Visibility.PUBLIC, is_exported=True, signature=f"def {name}()",
        is_local=is_local,
    )


def test_local_symbols_excluded_from_chunks(tmp_path):
    (tmp_path / "f.py").write_text("def top():\n    def inner():\n        pass\n")
    syms = [
        _fn("sutra python r f.py top().", "top", False),
        _fn("sutra python r f.py top().inner().", "inner", True),
    ]
    chunks, monikers = build_chunks(syms, tmp_path)
    assert "sutra python r f.py top()." in monikers
    assert "sutra python r f.py top().inner()." not in monikers
