from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# The kind vocabulary as serialized in graph.json symbols.
KINDS = frozenset({"function", "method", "class", "variable", "module"})


@dataclass(frozen=True, slots=True)
class ParsedQuery:
    """
    The analyzed query — produced ONCE by the QueryAnalyzer (P16), shared
    by all channels.  No channel re-parses or re-embeds the query.

    Fields
    ------
    text : str
        The raw query string.
    embedding : np.ndarray | None
        Query embedding.  Optional: lexical-only callers (BM25 acceptance
        runs, moniker lookups) never pay for an embedding they don't use.
    kind_hint : frozenset[str] | None
        Symbol kinds the query is asking for, derived conservatively from
        explicit kind nouns ("function", "model", …) or behavioral verbs.
        None = no confident hint = never filter.
    verbs : frozenset[str]
        Behavioral verbs found in the query (lemma-lite normalized),
        e.g. {"save", "upload"}.  Used for intent detection and carried
        for future channels.
    entities : tuple[str, ...]
        Code-identifier-looking tokens in query order ("PostgresDALWrapper",
        "find_one") — exact/moniker lookup fodder.
    """

    text: str
    embedding: Optional[np.ndarray] = None
    kind_hint: Optional[frozenset[str]] = None
    verbs: frozenset[str] = frozenset()
    entities: tuple[str, ...] = ()
