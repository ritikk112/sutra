-- Repositories: one row per indexed repo.
--
-- Replaces the AGE Repository vertex.  Stores the canonical URL and the
-- last-indexed commit SHA.  Symbols carry repo_name as a foreign key with
-- ON DELETE CASCADE so dropping a repository row removes all its symbols
-- (matches AGE's DETACH DELETE behaviour).
--
-- Primary key is `name`, not `url`, because:
--   * `name` is what monikers carry (sutra <lang> <repo_name> <file> <descriptor>),
--   * `name` is what callers use to look up state (repo_name_from_url(url)),
--   * `url` is canonical metadata for display, not a join key.
--
-- last_commit_sha is nullable: the row is created on the first index run,
-- but a partial run may complete before commit_sha is patched (the LAST
-- step in the incremental updater).  Nullable preserves the recovery
-- invariant — re-running an interrupted update sees old_sha=NULL and
-- treats it as a first run, which is correct.

CREATE TABLE IF NOT EXISTS sutra_repositories (
    name             TEXT PRIMARY KEY,
    url              TEXT NOT NULL,
    last_commit_sha  TEXT,
    indexed_at       TIMESTAMPTZ
);
