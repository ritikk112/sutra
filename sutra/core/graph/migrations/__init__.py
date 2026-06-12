"""SQL migrations for the Sutra graph store.

Migration files are plain `.sql` and applied in order by SqlGraphWriter.setup().
Each migration is wrapped in CREATE ... IF NOT EXISTS so re-running setup() is
idempotent — the same files apply cleanly to a fresh database and to one that
has already been set up.
"""
