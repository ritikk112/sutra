from pathlib import Path
from sutra.core.embedder.chunk_builder import build_chunks
from sutra.core.extractor.base import (
    FunctionSymbol,
    Location,
    ModuleSymbol,
    VariableSymbol,
    Visibility,
)


def _fn(id_, name, is_local):
    return FunctionSymbol(
        id=id_, name=name, qualified_name=name, file_path="f.py",
        location=Location(1, 2, 0, 5), body_hash="sha256:x", language="python",
        visibility=Visibility.PUBLIC, is_exported=True, signature=f"def {name}()",
        is_local=is_local,
    )


def _var(id_, name, byte_start, byte_end, *, is_constant=False, is_local=False,
         type_annotation=None, file_path="f.py"):
    return VariableSymbol(
        id=id_, name=name, qualified_name=name, file_path=file_path,
        location=Location(1, 1, byte_start, byte_end), body_hash="sha256:x",
        language="python", visibility=Visibility.PUBLIC, is_exported=True,
        is_local=is_local, is_constant=is_constant, type_annotation=type_annotation,
    )


def _mod(id_, name, *, docstring=None, file_path="f.py"):
    return ModuleSymbol(
        id=id_, name=name, qualified_name=name, file_path=file_path,
        location=Location(1, 1, 0, 0), body_hash="sha256:x", language="python",
        visibility=Visibility.PUBLIC, is_exported=True, docstring=docstring,
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


def test_variable_chunk_includes_assignment_source(tmp_path):
    src = 'HOOKS = {"response": [], "pre_send": []}\n'
    (tmp_path / "f.py").write_text(src)
    sym = _var(
        "sutra python r f.py HOOKS.", "HOOKS", 0, len(src) - 1, is_constant=True,
    )
    chunks, monikers = build_chunks([sym], tmp_path)
    assert monikers == ["sutra python r f.py HOOKS."]
    chunk = chunks[0]
    assert chunk.startswith("Variable: HOOKS")
    assert "File: f.py" in chunk
    assert "constant" in chunk
    assert '"response"' in chunk   # the assignment RHS is part of the chunk


def test_variable_chunk_includes_type_annotation(tmp_path):
    src = "timeout: float = 3.5\n"
    (tmp_path / "f.py").write_text(src)
    sym = _var(
        "sutra python r f.py timeout.", "timeout", 0, len(src) - 1,
        type_annotation="float",
    )
    chunks, _ = build_chunks([sym], tmp_path)
    assert "Type: float" in chunks[0]


def test_variable_value_is_truncated(tmp_path):
    lines = [f'    "key{i}": {i},' for i in range(200)]
    src = "BIG = {\n" + "\n".join(lines) + "\n}\n"
    (tmp_path / "f.py").write_text(src)
    sym = _var("sutra python r f.py BIG.", "BIG", 0, len(src) - 1)
    chunks, _ = build_chunks([sym], tmp_path)
    assert "key0" in chunks[0]
    assert "key199" not in chunks[0]
    assert "more lines" in chunks[0]


def test_local_variable_excluded(tmp_path):
    (tmp_path / "f.py").write_text("x = 1\n")
    sym = _var("sutra python r f.py f().x.", "x", 0, 5, is_local=True)
    chunks, monikers = build_chunks([sym], tmp_path)
    assert monikers == []


def test_module_chunk_lists_docstring_and_members(tmp_path):
    (tmp_path / "f.py").write_text('"""Alias helpers."""\n\ndef to_camel():\n    pass\n')
    mod = _mod("sutra python r f.py /", "f", docstring="Alias helpers.")
    fn = _fn("sutra python r f.py to_camel().", "to_camel", False)
    var = _var("sutra python r f.py LIMIT.", "LIMIT", 0, 5, is_constant=True)
    chunks, monikers = build_chunks([mod, fn, var], tmp_path)
    assert "sutra python r f.py /" in monikers
    mod_chunk = chunks[monikers.index("sutra python r f.py /")]
    assert mod_chunk.startswith("Module: f")
    assert "Alias helpers." in mod_chunk
    assert "to_camel" in mod_chunk   # members listed
    assert "LIMIT" in mod_chunk


def test_module_member_list_is_capped(tmp_path):
    (tmp_path / "f.py").write_text("pass\n")
    mod = _mod("sutra python r f.py /", "f")
    fns = [
        _fn(f"sutra python r f.py fn{i:03d}().", f"fn{i:03d}", False)
        for i in range(80)
    ]
    chunks, monikers = build_chunks([mod] + fns, tmp_path)
    mod_chunk = chunks[monikers.index("sutra python r f.py /")]
    assert "fn000" in mod_chunk
    assert "fn079" not in mod_chunk
    assert "more" in mod_chunk
