from __future__ import annotations

from typing import Any, Optional, Union

from sutra.core.artifact.loader import ArtifactSnapshot
from sutra.core.retrieval.types import SearchResult

# The spec'd default (SUTRA_PHASE2.md P18).  568M params — on CPU expect
# seconds per 50-candidate query; opt-in per query for exactly that reason.
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# Cap on Calls: entries in synthesized rerank text (mirrors chunk_builder).
_MAX_CALLS = 20


def build_rerank_text(moniker: str, snapshot: ArtifactSnapshot) -> str:
    """
    Synthesize the cross-encoder document for one candidate, FROM THE
    ARTIFACT ONLY.

    Deliberate deviation from the original P18 note ("recompute via
    chunk_builder at query time"): chunk_builder needs source bytes from
    disk, and the consumer owns nothing but the artifact directory
    (zero-infra is a locked decision).  This builds the same header-style
    text minus the Implementation body: kind, qualified name, file,
    signature, params/returns, docstring, and a Calls: line rebuilt from
    the artifact's CALLS relationships.
    """
    sym = snapshot.symbols.get(moniker)
    if sym is None:
        return moniker  # unknown candidate — score it on its name alone

    kind = sym.get("kind", "symbol")
    parts = [
        f"{kind.capitalize()}: {sym.get('qualified_name') or sym.get('name', '')}",
        f"File: {sym.get('file_path', '')}",
    ]

    if sym.get("signature"):
        parts.append(f"Signature: {sym['signature']}")

    params = [
        f"{p['name']}: {p['type_annotation']}"
        for p in (sym.get("parameters") or [])
        if p.get("name") not in ("self", "cls") and p.get("type_annotation")
    ]
    if params:
        parts.append(f"Parameters: {', '.join(params)}")
    if sym.get("return_type"):
        parts.append(f"Returns: {sym['return_type']}")
    if sym.get("base_classes"):
        parts.append(f"Extends: {', '.join(sym['base_classes'])}")
    if sym.get("docstring"):
        parts.append(f"Docstring: {sym['docstring']}")

    calls = sorted({
        rel.get("target_name") or (rel.get("target_id") or "").split()[-1].rstrip("().")
        for rel in snapshot.relationships
        if rel.get("source_id") == moniker and rel.get("kind") == "calls"
    } - {""})
    if calls:
        shown = calls[:_MAX_CALLS]
        suffix = f", ... ({len(calls) - _MAX_CALLS} more)" if len(calls) > _MAX_CALLS else ""
        parts.append(f"Calls: {', '.join(shown)}{suffix}")

    return "\n".join(parts)


def load_rerank_model(model_name: str = DEFAULT_RERANK_MODEL) -> Any:
    """
    Load a CrossEncoder.  sentence-transformers is an OPTIONAL dependency
    (requirements-ml.txt) — a clear remediation hint on ImportError, never
    a silent fallback.
    """
    try:
        from sentence_transformers import CrossEncoder  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "The reranker needs sentence-transformers, which is not installed "
            "(it is an optional dependency).  Install CPU torch first, then:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "  pip install -r requirements-ml.txt"
        ) from exc
    return CrossEncoder(model_name)


def rerank(
    query: str,
    candidates: list[SearchResult],
    snapshot: ArtifactSnapshot,
    model: Union[Any, str] = DEFAULT_RERANK_MODEL,
    top_k: Optional[int] = None,
) -> list[SearchResult]:
    """
    Re-order `candidates` by cross-encoder relevance to `query`.

    A plain function, NOT an ABC — per the design-pattern decisions a
    second reranker promotes this to an interface; one impl does not.

    `model` is either a loaded CrossEncoder (preferred — load once, reuse
    across queries) or a model name to load on the fly.  Returns new
    SearchResults sorted by rerank score with provenance extended by
    {"rerank": score}; prior provenance (vector/bm25/rrf) is preserved.
    Documents are synthesized per candidate from the artifact and cached
    per call — one batch predict per query.
    """
    if not candidates:
        return []

    if isinstance(model, str):
        model = load_rerank_model(model)

    docs = [build_rerank_text(r.moniker, snapshot) for r in candidates]
    scores = model.predict([(query, doc) for doc in docs])

    rescored = [
        SearchResult(
            moniker=r.moniker,
            score=float(s),
            provenance={**r.provenance, "rerank": float(s)},
        )
        for r, s in zip(candidates, scores)
    ]
    rescored.sort(key=lambda r: r.score, reverse=True)
    return rescored[:top_k] if top_k is not None else rescored
