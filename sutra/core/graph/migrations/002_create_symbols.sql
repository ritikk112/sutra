-- Symbols: one row per extracted code symbol.
--
-- Replaces the AGE Symbol vertex with a `kind` property.  Single table
-- holding all symbol kinds (function, class, method, variable, module) —
-- matches Phase 1's single-label design (see graph/schema.py rationale).
--
-- Moniker is the primary key (no surrogate ID).  AGE forced a node ID;
-- plain SQL doesn't need one and the moniker is already a stable identity
-- that survives renames via body_hash matching.
--
-- repo_name FK with ON DELETE CASCADE matches AGE's DETACH DELETE on the
-- BELONGS_TO edge: drop a repository row, lose all its symbols.
--
-- Per-kind fields are nullable columns rather than JSONB.  Trade-off: a
-- wider table, but every field stays inspectable and indexable.  metadata
-- is JSONB for genuinely-unmodelled escape-hatch data only.
--
-- Indexes:
--   * (repo_name)        — repo-scoped queries, the common case
--   * (file_path)        — incremental updater computes per-file diffs
--   * (kind)             — kind-filter for retrieval (P16)
--   * (qualified_name)   — Go method→struct cross-file linking
--
-- No FK on relationships→symbols (see 003_create_relationships.sql).

CREATE TABLE IF NOT EXISTS sutra_symbols (
    moniker          TEXT PRIMARY KEY,
    repo_name        TEXT NOT NULL REFERENCES sutra_repositories(name) ON DELETE CASCADE,
    kind             TEXT NOT NULL,
    name             TEXT NOT NULL,
    qualified_name   TEXT NOT NULL,
    file_path        TEXT NOT NULL,
    language         TEXT NOT NULL,
    visibility       TEXT NOT NULL,
    is_exported      BOOLEAN NOT NULL,
    body_hash        TEXT NOT NULL,
    line_start       INTEGER NOT NULL,
    line_end         INTEGER NOT NULL,

    -- Function/Method shared
    signature        TEXT,
    return_type      TEXT,
    is_async         BOOLEAN,
    complexity       INTEGER,
    docstring        TEXT,

    -- Method only
    is_static        BOOLEAN,
    is_constructor   BOOLEAN,
    receiver_kind    TEXT,

    -- Class only
    is_abstract      BOOLEAN,
    base_classes     TEXT,

    -- Variable only
    type_annotation  TEXT,
    is_constant      BOOLEAN,

    indexed_at       TIMESTAMPTZ NOT NULL,
    metadata         JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_sutra_symbols_repo
    ON sutra_symbols(repo_name);
CREATE INDEX IF NOT EXISTS idx_sutra_symbols_file
    ON sutra_symbols(file_path);
CREATE INDEX IF NOT EXISTS idx_sutra_symbols_kind
    ON sutra_symbols(kind);
CREATE INDEX IF NOT EXISTS idx_sutra_symbols_qname
    ON sutra_symbols(qualified_name);
