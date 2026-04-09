from __future__ import annotations

from pathlib import Path

from sutra.core.extractor.base import (
    ClassSymbol,
    FunctionSymbol,
    MethodSymbol,
    Symbol,
)

# Symbol types that receive embeddings.
# VariableSymbol and ModuleSymbol are excluded — no meaningful body chunk.
_EMBEDDABLE = (FunctionSymbol, ClassSymbol, MethodSymbol)

# Truncation limits
_MAX_FUNC_LINES = 150        # max lines of function/method body included in chunk
_MAX_CLASS_METHODS = 30      # max method names listed in class chunk
_MAX_CHARS = 30_000          # hard character cap on any chunk (pathological single-line files)


def build_chunks(
    symbols: list[Symbol],
    root: Path,
) -> tuple[list[str], list[str]]:
    """
    Build embedding text chunks for all embeddable symbols.

    Returns (chunks, monikers) where:
    - chunks[i] is the text to embed for monikers[i]
    - Both lists are in ascending sym.id order (stable, matches exporter sort)
    - len(chunks) == len(monikers) is guaranteed

    Source files are read from disk once per unique file_path, using a local
    dict cache scoped to this invocation.  The cache is released when the
    function returns, preventing long-lived memory retention.

    Raises FileNotFoundError with a clear message if a source file is missing.
    """
    # Sort embeddable symbols by id — matches exporter's stable ordering so
    # embedding_id integers in graph.json are deterministic across runs.
    embeddable = sorted(
        [s for s in symbols if isinstance(s, _EMBEDDABLE)],
        key=lambda s: s.id,
    )

    # Build class_id → method_names mapping for class chunk "Methods:" line.
    # Only methods that point to a class in our embeddable set are tracked.
    class_ids = {s.id for s in embeddable if isinstance(s, ClassSymbol)}
    class_methods: dict[str, list[str]] = {cid: [] for cid in class_ids}
    for sym in symbols:
        if isinstance(sym, MethodSymbol) and sym.enclosing_class_id in class_methods:
            class_methods[sym.enclosing_class_id].append(sym.name)

    # Local file cache — scoped to this call, GC'd on return.
    file_cache: dict[str, bytes] = {}

    chunks: list[str] = []
    monikers: list[str] = []

    for sym in embeddable:
        if isinstance(sym, ClassSymbol):
            chunk = _build_class_chunk(sym, class_methods.get(sym.id, []))
        else:
            # FunctionSymbol or MethodSymbol — needs source bytes
            if sym.file_path not in file_cache:
                src_path = root / sym.file_path
                if not src_path.exists():
                    raise FileNotFoundError(
                        f"Source file not found while building embedding chunk "
                        f"for symbol {sym.id!r}: {src_path}"
                    )
                file_cache[sym.file_path] = src_path.read_bytes()
            chunk = _build_function_chunk(sym, file_cache[sym.file_path])

        chunks.append(chunk)
        monikers.append(sym.id)

    assert len(chunks) == len(monikers)
    return chunks, monikers


def _build_function_chunk(sym: FunctionSymbol, src_bytes: bytes) -> str:
    kind = "Method" if isinstance(sym, MethodSymbol) else "Function"

    # Extract full symbol text from byte range.
    # This includes the signature, but we also show it separately in the
    # "Signature:" field for explicitness.  Duplication is acceptable in
    # Phase 1 — the chunk is for embedding, not human reading.
    sym_bytes = src_bytes[sym.location.byte_start:sym.location.byte_end]
    body_text = sym_bytes.decode("utf-8", errors="replace")

    # Truncate by lines
    lines = body_text.splitlines()
    if len(lines) > _MAX_FUNC_LINES:
        remaining = len(lines) - _MAX_FUNC_LINES
        body_text = "\n".join(lines[:_MAX_FUNC_LINES]) + f"\n... ({remaining} more lines)"
    else:
        body_text = "\n".join(lines)

    parts = [
        f"{kind}: {sym.qualified_name}",
        f"File: {sym.file_path}",
        f"Signature: {sym.signature}",
    ]
    if getattr(sym, "docstring", None):
        parts.append(f"Docstring: {sym.docstring}")
    parts.append(f"Body:\n{body_text}")

    chunk = "\n".join(parts)

    # Hard character cap — catches pathological single-line files that bypass
    # line-based truncation (minified JS, generated code, long string literals).
    if len(chunk) > _MAX_CHARS:
        chunk = chunk[:_MAX_CHARS]

    return chunk


def _build_class_chunk(sym: ClassSymbol, method_names: list[str]) -> str:
    parts = [
        f"Class: {sym.qualified_name}",
        f"File: {sym.file_path}",
    ]
    if sym.base_classes:
        parts.append(f"Extends: {', '.join(sym.base_classes)}")
    if sym.docstring:
        parts.append(f"Docstring: {sym.docstring}")

    if method_names:
        if len(method_names) > _MAX_CLASS_METHODS:
            remaining = len(method_names) - _MAX_CLASS_METHODS
            display = method_names[:_MAX_CLASS_METHODS]
            parts.append(f"Methods: {', '.join(display)}, ... ({remaining} more)")
        else:
            parts.append(f"Methods: {', '.join(method_names)}")

    chunk = "\n".join(parts)

    # Hard character cap
    if len(chunk) > _MAX_CHARS:
        chunk = chunk[:_MAX_CHARS]

    return chunk
