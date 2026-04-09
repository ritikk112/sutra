"""
AGE + pgvector schema constants for Sutra.

No logic lives here — only labels, defaults, and deliberate design notes.

Node label design (Phase 1):
    All symbol kinds share a single 'Symbol' vertex label.  'kind' is stored
    as a property ("function", "class", "method", "variable", "module").

    Trade-off accepted: kind-specific Cypher queries must filter by property
    (MATCH (n:Symbol {kind: 'function'})) instead of by label (MATCH (n:Function)).
    Label indexes cannot be used for kind filtering, which is slightly slower.
    Acceptable for Phase 1 repo sizes.  If per-kind label indexes become a
    bottleneck, split labels in a future migration.

Edge labels:
    One edge label per RelationKind, uppercased (AGE convention).
    BELONGS_TO is a meta-edge linking every Symbol to its Repository node —
    not a RelationKind, but used internally for repo-scoped queries and
    replace-mode delete.

Repository node:
    A 'Repository' vertex (label = 'Repository') is created for each indexed
    repo.  Every Symbol gets a BELONGS_TO edge to its Repository.  This makes
    repos queryable as units ("give me all symbols from repo X") and gives
    the incremental pipeline (Priority 11) somewhere to record last-indexed
    commit SHA.
"""

# -----------------------------------------------------------------------
# AGE defaults
# -----------------------------------------------------------------------

DEFAULT_GRAPH_NAME = "sutra_graph"

# Vertex labels
SYMBOL_LABEL = "Symbol"
REPO_LABEL = "Repository"

# Meta-edge label (not in RelationKind; used only by writer internals)
BELONGS_TO_LABEL = "BELONGS_TO"

# Edge labels — uppercase of RelationKind values
EDGE_LABELS = frozenset({
    "CALLS",
    "EXTENDS",
    "IMPLEMENTS",
    "IMPORTS",
    "CONTAINS",
    "REFERENCES",
    "RETURNS_TYPE",
    "PARAMETER_TYPE",
})

# -----------------------------------------------------------------------
# pgvector defaults
# -----------------------------------------------------------------------

DEFAULT_TABLE_NAME = "sutra_embeddings"
