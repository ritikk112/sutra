"""
Tests for Priority 9 — Embedder module.

Covers:
  - FixtureEmbedder: shape, dtype, determinism, dimensions property
  - chunk_builder.build_chunks(): format, truncation, sorting, error handling
  - factory.from_config(): provider dispatch, ConfigError on missing env var,
    unknown provider, missing config file
  - Shape invariant: len(chunks) == len(monikers) == vectors.shape[0]

No mocking — real symbol instances, real source files written to tmp_path,
real FixtureEmbedder calls.  OpenAI and local (sentence-transformers) tests
are in test_real_embedder.py, skipped unless the relevant env/package is present.
"""
from __future__ import annotations

import textwrap
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from sutra.core.embedder.chunk_builder import (
    _MAX_CHARS,
    _MAX_CLASS_METHODS,
    _MAX_FUNC_LINES,
    build_chunks,
)
from sutra.core.embedder.factory import ConfigError, from_config
from sutra.core.embedder.fixture import DEFAULT_FIXTURE_DIMS, FixtureEmbedder
from sutra.core.extractor.base import (
    ClassSymbol,
    FunctionSymbol,
    Location,
    MethodSymbol,
    ModuleSymbol,
    Parameter,
    VariableSymbol,
    Visibility,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _loc(
    line_start: int = 1,
    line_end: int = 5,
    byte_start: int = 0,
    byte_end: int = 100,
) -> Location:
    return Location(
        line_start=line_start, line_end=line_end,
        byte_start=byte_start, byte_end=byte_end,
    )


def _make_func(
    name: str = "my_func",
    qualified_name: str = "pkg.my_func",
    file_path: str = "pkg/a.py",
    byte_start: int = 0,
    byte_end: int = 50,
    docstring: str | None = None,
    signature: str = "def my_func() -> None",
) -> FunctionSymbol:
    return FunctionSymbol(
        id=f"sutra python repo {file_path} {name}().",
        name=name,
        qualified_name=qualified_name,
        file_path=file_path,
        location=_loc(byte_start=byte_start, byte_end=byte_end),
        body_hash="sha256:abc",
        language="python",
        visibility=Visibility.PUBLIC,
        is_exported=True,
        signature=signature,
        parameters=[],
        return_type="None",
        docstring=docstring,
        decorators=[],
        is_async=False,
        complexity=1,
    )


def _make_method(
    name: str = "my_method",
    class_id: str = "sutra python repo pkg/a.py MyClass#",
    file_path: str = "pkg/a.py",
    byte_start: int = 0,
    byte_end: int = 50,
    signature: str = "def my_method(self) -> None",
) -> MethodSymbol:
    return MethodSymbol(
        id=f"sutra python repo {file_path} MyClass#{name}().",
        name=name,
        qualified_name=f"pkg.MyClass.{name}",
        file_path=file_path,
        location=_loc(byte_start=byte_start, byte_end=byte_end),
        body_hash="sha256:def",
        language="python",
        visibility=Visibility.PUBLIC,
        is_exported=True,
        signature=signature,
        parameters=[Parameter(name="self")],
        return_type="None",
        docstring=None,
        decorators=[],
        is_async=False,
        complexity=1,
        enclosing_class_id=class_id,
        is_static=False,
        is_constructor=False,
    )


def _make_class(
    name: str = "MyClass",
    file_path: str = "pkg/a.py",
    base_classes: list[str] | None = None,
    docstring: str | None = None,
) -> ClassSymbol:
    return ClassSymbol(
        id=f"sutra python repo {file_path} {name}#",
        name=name,
        qualified_name=f"pkg.{name}",
        file_path=file_path,
        location=_loc(),
        body_hash="sha256:ghi",
        language="python",
        visibility=Visibility.PUBLIC,
        is_exported=True,
        base_classes=base_classes or [],
        docstring=docstring,
        decorators=[],
        is_abstract=False,
    )


def _make_source_file(tmp_path: Path, file_path: str, content: str) -> Path:
    """Write content to tmp_path / file_path, creating parent dirs as needed."""
    abs_path = tmp_path / file_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_text(content, encoding="utf-8")
    return abs_path


# ---------------------------------------------------------------------------
# FixtureEmbedder
# ---------------------------------------------------------------------------

class TestFixtureEmbedder:
    def test_default_dimensions(self) -> None:
        assert FixtureEmbedder().dimensions == DEFAULT_FIXTURE_DIMS

    def test_custom_dimensions(self) -> None:
        assert FixtureEmbedder(dims=128).dimensions == 128

    def test_embed_empty_returns_correct_shape(self) -> None:
        result = FixtureEmbedder().embed([])
        assert result.shape == (0, DEFAULT_FIXTURE_DIMS)
        assert result.dtype == np.float32

    def test_embed_single_chunk(self) -> None:
        result = FixtureEmbedder().embed(["hello world"])
        assert result.shape == (1, DEFAULT_FIXTURE_DIMS)
        assert result.dtype == np.float32

    def test_embed_multiple_chunks(self) -> None:
        chunks = ["chunk one", "chunk two", "chunk three"]
        result = FixtureEmbedder().embed(chunks)
        assert result.shape == (3, DEFAULT_FIXTURE_DIMS)
        assert result.dtype == np.float32

    def test_deterministic_same_chunk(self) -> None:
        chunk = "Function: pkg.foo\nFile: pkg/a.py\nBody:\ndef foo(): pass"
        v1 = FixtureEmbedder().embed([chunk])
        v2 = FixtureEmbedder().embed([chunk])
        np.testing.assert_array_equal(v1, v2)

    def test_different_chunks_produce_different_vectors(self) -> None:
        v1 = FixtureEmbedder().embed(["chunk one"])
        v2 = FixtureEmbedder().embed(["chunk two"])
        # Very unlikely to be equal with sha256-seeded RNG
        assert not np.allclose(v1, v2)

    def test_custom_dims_propagated(self) -> None:
        result = FixtureEmbedder(dims=64).embed(["test"])
        assert result.shape == (1, 64)

    def test_returns_float32(self) -> None:
        result = FixtureEmbedder().embed(["test"])
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# chunk_builder — function/method chunks
# ---------------------------------------------------------------------------

class TestChunkBuilderFunctionChunks:
    def test_function_chunk_format(self, tmp_path: Path) -> None:
        src = "def my_func() -> None:\n    pass\n"
        _make_source_file(tmp_path, "pkg/a.py", src)

        func = _make_func(
            byte_start=0,
            byte_end=len(src.encode()),
            signature="def my_func() -> None",
        )
        chunks, monikers = build_chunks([func], tmp_path)

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.startswith("Function: pkg.my_func")
        assert "File: pkg/a.py" in chunk
        assert "Signature: def my_func() -> None" in chunk
        assert "Body:" in chunk
        assert "def my_func() -> None:" in chunk

    def test_function_chunk_with_docstring(self, tmp_path: Path) -> None:
        src = "def foo() -> None:\n    '''A docstring.'''\n    pass\n"
        _make_source_file(tmp_path, "pkg/a.py", src)

        func = _make_func(
            byte_start=0,
            byte_end=len(src.encode()),
            docstring="A docstring.",
        )
        chunks, _ = build_chunks([func], tmp_path)
        assert "Docstring: A docstring." in chunks[0]

    def test_method_chunk_uses_method_prefix(self, tmp_path: Path) -> None:
        src = "    def my_method(self) -> None:\n        pass\n"
        _make_source_file(tmp_path, "pkg/a.py", src)

        method = _make_method(byte_start=0, byte_end=len(src.encode()))
        chunks, _ = build_chunks([method], tmp_path)
        assert chunks[0].startswith("Method: pkg.MyClass.my_method")

    def test_function_body_truncated_at_max_lines(self, tmp_path: Path) -> None:
        # Source: 1 "def" line + (_MAX_FUNC_LINES + 20) body lines = total_src_lines
        # After truncating at _MAX_FUNC_LINES: remaining = total_src_lines - _MAX_FUNC_LINES
        extra = 20
        body_lines_count = _MAX_FUNC_LINES + extra
        lines = ["def big_func() -> None:"] + [f"    x = {i}" for i in range(body_lines_count)]
        src = "\n".join(lines) + "\n"
        total_src_lines = len(lines)  # 1 + body_lines_count
        expected_remaining = total_src_lines - _MAX_FUNC_LINES
        _make_source_file(tmp_path, "pkg/a.py", src)

        func = _make_func(byte_start=0, byte_end=len(src.encode()))
        chunks, _ = build_chunks([func], tmp_path)

        chunk = chunks[0]
        # Truncation marker must appear
        assert f"... ({expected_remaining} more lines)" in chunk
        # The truncated body must have exactly _MAX_FUNC_LINES lines in the Body section
        body_section = chunk.split("Body:\n", 1)[1]
        body_section_lines = body_section.split("\n")
        # Last line is the "... (N more lines)" marker
        assert body_section_lines[-1] == f"... ({expected_remaining} more lines)"
        # Before the marker: _MAX_FUNC_LINES lines
        assert len(body_section_lines) - 1 == _MAX_FUNC_LINES

    def test_hard_char_cap(self, tmp_path: Path) -> None:
        # Single-line function body that exceeds _MAX_CHARS
        long_line = "x = " + "a" * _MAX_CHARS
        src = f"def huge() -> None:\n    {long_line}\n"
        _make_source_file(tmp_path, "pkg/a.py", src)

        func = _make_func(byte_start=0, byte_end=len(src.encode()))
        chunks, _ = build_chunks([func], tmp_path)
        assert len(chunks[0]) == _MAX_CHARS

    def test_file_not_found_raises_clear_error(self, tmp_path: Path) -> None:
        func = _make_func(file_path="does/not/exist.py")
        with pytest.raises(FileNotFoundError, match="does/not/exist.py"):
            build_chunks([func], tmp_path)


# ---------------------------------------------------------------------------
# chunk_builder — class chunks
# ---------------------------------------------------------------------------

class TestChunkBuilderClassChunks:
    def test_class_chunk_format(self, tmp_path: Path) -> None:
        cls = _make_class(docstring="Does stuff.")
        chunks, monikers = build_chunks([cls], tmp_path)

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.startswith("Class: pkg.MyClass")
        assert "File: pkg/a.py" in chunk
        assert "Docstring: Does stuff." in chunk

    def test_class_chunk_with_base_classes(self, tmp_path: Path) -> None:
        cls = _make_class(base_classes=["Base", "Mixin"])
        chunks, _ = build_chunks([cls], tmp_path)
        assert "Extends: Base, Mixin" in chunks[0]

    def test_class_chunk_without_base_classes(self, tmp_path: Path) -> None:
        cls = _make_class()
        chunks, _ = build_chunks([cls], tmp_path)
        assert "Extends" not in chunks[0]

    def test_class_chunk_methods_listed(self, tmp_path: Path) -> None:
        # Methods are embeddable, so they need a source file on disk.
        src = "    def alpha(self): pass\n    def beta(self): pass\n"
        _make_source_file(tmp_path, "pkg/a.py", src)

        cls = _make_class()
        method_a = _make_method(name="alpha", class_id=cls.id,
                                byte_start=0, byte_end=25)
        method_b = _make_method(name="beta", class_id=cls.id,
                                byte_start=25, byte_end=len(src.encode()))
        chunks, monikers = build_chunks([cls, method_a, method_b], tmp_path)

        cls_chunk = next(c for c, m in zip(chunks, monikers) if "Class:" in c)
        assert "alpha" in cls_chunk
        assert "beta" in cls_chunk

    def test_class_methods_truncated_at_max(self, tmp_path: Path) -> None:
        cls = _make_class()
        # Build 200 MethodSymbol instances — all pointing to cls but not in symbols list
        # chunk_builder collects method names from symbols; build them all
        many_methods = [
            MethodSymbol(
                id=f"sutra python repo pkg/a.py MyClass#m{i}().",
                name=f"m{i}",
                qualified_name=f"pkg.MyClass.m{i}",
                file_path="pkg/a.py",
                location=_loc(),
                body_hash=f"sha256:{i:03d}",
                language="python",
                visibility=Visibility.PUBLIC,
                is_exported=True,
                signature=f"def m{i}(self)",
                parameters=[],
                return_type=None,
                docstring=None,
                decorators=[],
                is_async=False,
                complexity=1,
                enclosing_class_id=cls.id,
                is_static=False,
                is_constructor=False,
            )
            for i in range(200)
        ]

        # build_chunks only embeds cls (ClassSymbol); methods are collected for the Methods: line
        # But methods also need source if they're embeddable — pass only cls to avoid file reads
        chunks, monikers = build_chunks([cls], tmp_path)
        # class_methods for cls will be empty because methods not in symbols list
        # To properly test truncation, we need to pass all symbols
        # chunk_builder collects class_methods from ALL symbols in the list
        src = "\n".join(f"    def m{i}(self): pass" for i in range(200)) + "\n"
        _make_source_file(tmp_path, "pkg/a.py", src)

        chunks, monikers = build_chunks([cls] + many_methods, tmp_path)
        cls_chunk = next(c for c, m in zip(chunks, monikers) if "Class:" in c)

        assert f"... ({200 - _MAX_CLASS_METHODS} more)" in cls_chunk
        # Exactly _MAX_CLASS_METHODS method names before the truncation suffix
        methods_line = next(
            line for line in cls_chunk.splitlines() if line.startswith("Methods:")
        )
        # Count method names (everything before the "... (N more)" part)
        before_ellipsis = methods_line.split(", ... (")[0]
        listed_names = [n.strip() for n in before_ellipsis.replace("Methods: ", "").split(",")]
        assert len(listed_names) == _MAX_CLASS_METHODS

    def test_class_no_docstring_no_docstring_line(self, tmp_path: Path) -> None:
        cls = _make_class(docstring=None)
        chunks, _ = build_chunks([cls], tmp_path)
        assert "Docstring" not in chunks[0]

    def test_class_hard_char_cap(self, tmp_path: Path) -> None:
        # Docstring that causes chunk to exceed _MAX_CHARS
        long_doc = "x" * (_MAX_CHARS + 5000)
        cls = _make_class(docstring=long_doc)
        chunks, _ = build_chunks([cls], tmp_path)
        assert len(chunks[0]) == _MAX_CHARS


# ---------------------------------------------------------------------------
# chunk_builder — filtering and ordering
# ---------------------------------------------------------------------------

class TestChunkBuilderFiltering:
    def test_variable_symbols_not_included(self, tmp_path: Path) -> None:
        var = VariableSymbol(
            id="sutra python repo pkg/a.py X.",
            name="X",
            qualified_name="pkg.X",
            file_path="pkg/a.py",
            location=_loc(),
            body_hash="sha256:var",
            language="python",
            visibility=Visibility.PUBLIC,
            is_exported=True,
        )
        chunks, monikers = build_chunks([var], tmp_path)
        assert chunks == []
        assert monikers == []

    def test_module_symbols_not_included(self, tmp_path: Path) -> None:
        mod = ModuleSymbol(
            id="sutra python repo pkg/a.py pkg/a/",
            name="a",
            qualified_name="pkg.a",
            file_path="pkg/a.py",
            location=_loc(),
            body_hash="sha256:mod",
            language="python",
            visibility=Visibility.PUBLIC,
            is_exported=True,
        )
        chunks, monikers = build_chunks([mod], tmp_path)
        assert chunks == []
        assert monikers == []

    def test_output_sorted_by_moniker(self, tmp_path: Path) -> None:
        src = "def z() -> None: pass\ndef a() -> None: pass\n"
        _make_source_file(tmp_path, "pkg/a.py", src)

        func_z = _make_func(name="z", qualified_name="pkg.z",
                            byte_start=0, byte_end=22)
        func_a = _make_func(name="a", qualified_name="pkg.a",
                            byte_start=22, byte_end=44)
        # func_z has id with "z" > "a" in sort order
        # Override IDs to guarantee z > a
        func_z_id = "sutra python repo pkg/a.py z()."
        func_a_id = "sutra python repo pkg/a.py a()."

        # Manually set ids using dataclass mutation
        func_z.id = func_z_id
        func_a.id = func_a_id

        chunks, monikers = build_chunks([func_z, func_a], tmp_path)
        # Output must be sorted by id (a < z)
        assert monikers == sorted(monikers)
        assert monikers[0] == func_a_id

    def test_len_invariant(self, tmp_path: Path) -> None:
        src = "def f() -> None: pass\n"
        _make_source_file(tmp_path, "pkg/a.py", src)
        func = _make_func(byte_start=0, byte_end=len(src.encode()))
        cls = _make_class()
        chunks, monikers = build_chunks([func, cls], tmp_path)
        assert len(chunks) == len(monikers)

    def test_empty_symbols_list(self, tmp_path: Path) -> None:
        chunks, monikers = build_chunks([], tmp_path)
        assert chunks == []
        assert monikers == []


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestFactory:
    def test_none_config_returns_fixture_embedder(self) -> None:
        embedder = from_config(None)
        assert isinstance(embedder, FixtureEmbedder)

    def test_nonexistent_config_path_returns_fixture_embedder(self, tmp_path: Path) -> None:
        embedder = from_config(tmp_path / "does_not_exist.yaml")
        assert isinstance(embedder, FixtureEmbedder)

    def test_fixture_provider_returns_fixture_embedder(self, tmp_path: Path) -> None:
        config = tmp_path / "sutra.yaml"
        config.write_text("embedder:\n  provider: fixture\n  dimensions: 256\n", encoding="utf-8")
        embedder = from_config(config)
        assert isinstance(embedder, FixtureEmbedder)
        assert embedder.dimensions == 256

    def test_missing_provider_defaults_to_fixture(self, tmp_path: Path) -> None:
        config = tmp_path / "sutra.yaml"
        config.write_text("embedder: {}\n", encoding="utf-8")
        embedder = from_config(config)
        assert isinstance(embedder, FixtureEmbedder)

    def test_empty_config_file_returns_fixture_embedder(self, tmp_path: Path) -> None:
        config = tmp_path / "sutra.yaml"
        config.write_text("", encoding="utf-8")
        embedder = from_config(config)
        assert isinstance(embedder, FixtureEmbedder)

    def test_unknown_provider_raises_config_error(self, tmp_path: Path) -> None:
        config = tmp_path / "sutra.yaml"
        config.write_text("embedder:\n  provider: magic\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="magic"):
            from_config(config)

    def test_openai_missing_env_var_raises_config_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = tmp_path / "sutra.yaml"
        config.write_text(
            "embedder:\n  provider: openai\n  api_key_env: OPENAI_API_KEY\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
            from_config(config)

    def test_openai_empty_env_var_raises_config_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "")
        config = tmp_path / "sutra.yaml"
        config.write_text(
            "embedder:\n  provider: openai\n  api_key_env: OPENAI_API_KEY\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
            from_config(config)

    def test_openai_with_fake_key_returns_openai_embedder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-key-factory-test")
        config = tmp_path / "sutra.yaml"
        config.write_text(
            "embedder:\n  provider: openai\n  dimensions: 1536\n  batch_size: 50\n",
            encoding="utf-8",
        )
        from sutra.core.embedder.openai import OpenAIEmbedder
        embedder = from_config(config)
        assert isinstance(embedder, OpenAIEmbedder)
        assert embedder.dimensions == 1536

    def test_custom_api_key_env_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MY_CUSTOM_KEY", "sk-custom-key")
        config = tmp_path / "sutra.yaml"
        config.write_text(
            "embedder:\n  provider: openai\n  api_key_env: MY_CUSTOM_KEY\n",
            encoding="utf-8",
        )
        from sutra.core.embedder.openai import OpenAIEmbedder
        embedder = from_config(config)
        assert isinstance(embedder, OpenAIEmbedder)

    def test_custom_api_key_missing_custom_env_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MY_KEY", raising=False)
        config = tmp_path / "sutra.yaml"
        config.write_text(
            "embedder:\n  provider: openai\n  api_key_env: MY_KEY\n",
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match="MY_KEY"):
            from_config(config)


# ---------------------------------------------------------------------------
# Shape invariant
# ---------------------------------------------------------------------------

class TestShapeInvariant:
    def test_fixture_embedder_shape_invariant(self, tmp_path: Path) -> None:
        src = "def f() -> None: pass\n"
        _make_source_file(tmp_path, "pkg/a.py", src)
        func = _make_func(byte_start=0, byte_end=len(src.encode()))
        cls = _make_class()

        chunks, monikers = build_chunks([func, cls], tmp_path)
        vectors = FixtureEmbedder().embed(chunks)

        assert len(chunks) == len(monikers) == vectors.shape[0]
        assert vectors.dtype == np.float32

    def test_empty_symbols_produces_zero_rows(self, tmp_path: Path) -> None:
        chunks, monikers = build_chunks([], tmp_path)
        vectors = FixtureEmbedder().embed(chunks)
        assert vectors.shape[0] == 0
        assert len(monikers) == 0
