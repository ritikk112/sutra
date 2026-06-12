-- Relationships: directed edges between symbols.
--
-- Replaces AGE edges (CALLS, EXTENDS, IMPLEMENTS, …).  Composite primary key
-- on (source_id, target_id, kind) gives free deduplication on insert.
--
-- NO FK to sutra_symbols.  Two reasons:
--
--   1. Even resolved relationships may point at monikers that don't exist
--      as symbols (e.g. calls into third-party packages — Phase 1 emits
--      these with is_resolved=True when the call form is unambiguous, but
--      the target's package isn't indexed).  Matches AGE's behaviour: AGE
--      silently dropped MERGE-edge calls where the endpoint MATCH found
--      nothing.  In SQL we keep the row; consumers JOIN with sutra_symbols
--      and unresolvable rows fall out at query time.
--
--   2. ON DELETE CASCADE on a FK would force every symbol delete to scan
--      the relationships table.  We model deletion explicitly in
--      SqlGraphWriter.delete_symbols() — delete the symbol AND all
--      relationships where it appears as source OR target — matching
--      AGE's DETACH DELETE semantics with a single application-level
--      contract.
--
-- is_resolved is stored on the row.  Phase 1 only persists resolved
-- relationships (matches AGEWriter._write_relationships, which skips
-- unresolved).  Keeping the column means the heuristic resolver (P20-lite)
-- can later flip rows from is_resolved=False → True without requiring DDL.

CREATE TABLE IF NOT EXISTS sutra_relationships (
    source_id    TEXT NOT NULL,
    target_id    TEXT NOT NULL,
    kind         TEXT NOT NULL,
    is_resolved  BOOLEAN NOT NULL DEFAULT TRUE,
    metadata     JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (source_id, target_id, kind)
);

CREATE INDEX IF NOT EXISTS idx_sutra_rels_src
    ON sutra_relationships(source_id);
CREATE INDEX IF NOT EXISTS idx_sutra_rels_dst
    ON sutra_relationships(target_id);
CREATE INDEX IF NOT EXISTS idx_sutra_rels_kind
    ON sutra_relationships(kind);
