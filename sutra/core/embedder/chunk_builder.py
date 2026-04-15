from __future__ import annotations

from pathlib import Path

from sutra.core.extractor.base import (
    ClassSymbol,
    FunctionSymbol,
    MethodSymbol,
    RelationKind,
    Relationship,
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
    relationships: list[Relationship] | None = None,
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

    # Build class_id → MethodSymbol list mapping for class chunk "Methods:" block.
    # Only methods that point to a class in our embeddable set are tracked.
    class_ids = {s.id for s in embeddable if isinstance(s, ClassSymbol)}
    class_method_syms: dict[str, list[MethodSymbol]] = {cid: [] for cid in class_ids}
    for sym in symbols:
        if isinstance(sym, MethodSymbol) and sym.enclosing_class_id in class_method_syms:
            class_method_syms[sym.enclosing_class_id].append(sym)

    # Build source_id → sorted unique call targets from CALLS relationships.
    calls_by_source: dict[str, list[str]] = {}
    if relationships:
        for rel in relationships:
            if rel.kind == RelationKind.CALLS:
                target = rel.target_name
                if target is None and rel.target_id:
                    # Resolved relationship — extract bare name from the moniker
                    target = rel.target_id.split()[-1].rstrip("().")
                if target:
                    calls_by_source.setdefault(rel.source_id, []).append(target)
        for source_id in calls_by_source:
            calls_by_source[source_id] = sorted(set(calls_by_source[source_id]))

    # Local file cache — scoped to this call, GC'd on return.
    file_cache: dict[str, bytes] = {}

    chunks: list[str] = []
    monikers: list[str] = []

    for sym in embeddable:
        if isinstance(sym, ClassSymbol):
            chunk = _build_class_chunk(sym, class_method_syms.get(sym.id, []))
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
            chunk = _build_function_chunk(
                sym,
                file_cache[sym.file_path],
                calls=calls_by_source.get(sym.id),
            )

        chunks.append(chunk)
        monikers.append(sym.id)

    assert len(chunks) == len(monikers)
    return chunks, monikers


def _detect_delegation(body_lines: list[str]) -> str | None:
    """Return the delegation target name if this is a thin wrapper, else None."""
    meaningful = [
        line.strip() for line in body_lines
        if line.strip() and not line.strip().startswith('#')
    ]
    # Strip try/except wrapper — pattern: try: return await X(...) except: raise
    if meaningful and meaningful[0] == 'try:':
        skip = {'try:', 'except Exception as e:', 'except:', 'raise'}
        meaningful = [line for line in meaningful if line not in skip]
    if len(meaningful) == 1 and meaningful[0].startswith('return'):
        call = meaningful[0].removeprefix('return').strip().removeprefix('await').strip()
        if '(' in call:
            return call[:call.index('(')]
    return None


def _method_sig(m: MethodSymbol) -> str:
    """Render a concise method signature: [async ]name(param: type, ...) [-> return]."""
    prefix = "async " if m.is_async else ""
    params = [p for p in m.parameters if p.name not in ("self", "cls")]
    param_strs: list[str] = []
    for p in params:
        if p.type_annotation:
            param_strs.append(f"{p.name}: {p.type_annotation}")
        else:
            param_strs.append(p.name)
    sig = f"{prefix}{m.name}({', '.join(param_strs)})"
    if m.return_type:
        sig += f" -> {m.return_type}"
    return sig


def _build_function_chunk(
    sym: FunctionSymbol,
    src_bytes: bytes,
    calls: list[str] | None = None,
) -> str:
    kind = "Method" if isinstance(sym, MethodSymbol) else "Function"

    # Extract full symbol text from byte range.
    sym_bytes = src_bytes[sym.location.byte_start:sym.location.byte_end]
    raw_text = sym_bytes.decode("utf-8", errors="replace")
    all_lines = raw_text.splitlines()

    # --- Fix 1: Strip signature lines from the body ---
    # The def/async def signature ends at the first line whose stripped form
    # ends with ':' (handles multi-line signatures wrapped in parentheses).
    sig_end = len(all_lines)  # fallback: treat everything as signature
    for idx, line in enumerate(all_lines):
        if line.rstrip().endswith(':') and not line.rstrip().endswith('\\'):
            sig_end = idx + 1
            break
    body_lines = all_lines[sig_end:]

    # Strip leading docstring block from body (it already appears in Docstring: field).
    if body_lines:
        first = body_lines[0].strip()
        if first.startswith('"""') or first.startswith("'''"):
            quote = '"""' if first.startswith('"""') else "'''"
            # Single-line: """..."""
            if first.endswith(quote) and len(first) > 3:
                body_lines = body_lines[1:]
            else:
                # Multi-line: scan forward to closing quotes
                j = 1
                while j < len(body_lines):
                    if quote in body_lines[j]:
                        body_lines = body_lines[j + 1:]
                        break
                    j += 1
                else:
                    body_lines = body_lines[j:]

    # --- Fix 6: Detect delegation-only functions ---
    delegation_target = _detect_delegation(body_lines)

    # --- Fix 4: Build behavioral tags ---
    tags: list[str] = []
    if sym.is_async:
        tags.append("async")
    if isinstance(sym, MethodSymbol):
        if sym.is_static:
            tags.append("static")
        if sym.is_constructor:
            tags.append("constructor")
    for dec in sym.decorators:
        tags.append(dec.lstrip("@"))
    if sym.complexity is not None and sym.complexity > 3:
        tags.append(f"complexity={sym.complexity}")
    if delegation_target:
        tags.append(f"delegates_to={delegation_target}")

    # --- Assemble chunk ---
    parts = [
        f"{kind}: {sym.qualified_name}",
        f"File: {sym.file_path}",
    ]
    if tags:
        parts.append(f"Tags: {', '.join(tags)}")

    parts.append(f"Signature: {sym.signature}")

    # --- Fix 3: Typed parameters and return type ---
    typed_params = [
        p for p in sym.parameters
        if p.name not in ("self", "cls") and p.type_annotation
    ]
    if typed_params:
        param_strs = [f"{p.name}: {p.type_annotation}" for p in typed_params]
        parts.append(f"Parameters: {', '.join(param_strs)}")
    if sym.return_type:
        parts.append(f"Returns: {sym.return_type}")

    if getattr(sym, "docstring", None):
        parts.append(f"Docstring: {sym.docstring}")

    # --- Fix 2: Calls from relationships ---
    if calls:
        cap = 20
        if len(calls) > cap:
            parts.append(f"Calls: {', '.join(calls[:cap])}, ... ({len(calls) - cap} more)")
        else:
            parts.append(f"Calls: {', '.join(calls)}")

    # --- Fix 1 cont.: Implementation body (deduplicated) ---
    if delegation_target:
        # Fix 6: thin wrapper — summarise rather than repeat the callee's body
        parts.append(f"Implementation: delegates to {delegation_target}")
    else:
        if len(body_lines) > _MAX_FUNC_LINES:
            remaining = len(body_lines) - _MAX_FUNC_LINES
            impl_text = "\n".join(body_lines[:_MAX_FUNC_LINES]) + f"\n... ({remaining} more lines)"
        else:
            impl_text = "\n".join(body_lines)
        parts.append(f"Implementation:\n{impl_text}")

    chunk = "\n".join(parts)

    # Hard character cap — catches pathological single-line files that bypass
    # line-based truncation (minified JS, generated code, long string literals).
    if len(chunk) > _MAX_CHARS:
        chunk = chunk[:_MAX_CHARS]

    return chunk


def _build_class_chunk(sym: ClassSymbol, methods: list[MethodSymbol]) -> str:
    parts = [
        f"Class: {sym.qualified_name}",
        f"File: {sym.file_path}",
    ]
    if sym.base_classes:
        parts.append(f"Extends: {', '.join(sym.base_classes)}")
    if sym.docstring:
        parts.append(f"Docstring: {sym.docstring}")

    if methods:
        if len(methods) > _MAX_CLASS_METHODS:
            display = methods[:_MAX_CLASS_METHODS]
            remaining = len(methods) - _MAX_CLASS_METHODS
            method_lines = "\n".join(f"  {_method_sig(m)}" for m in display)
            parts.append(f"Methods:\n{method_lines}\n  ... ({remaining} more)")
        else:
            method_lines = "\n".join(f"  {_method_sig(m)}" for m in methods)
            parts.append(f"Methods:\n{method_lines}")

    chunk = "\n".join(parts)

    # Hard character cap
    if len(chunk) > _MAX_CHARS:
        chunk = chunk[:_MAX_CHARS]

    return chunk
