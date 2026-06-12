"""
Constants for the Sutra graph store.

After PR 2 (AGE → plain SQL), the only remaining constant here is the
default pgvector table name.  The AGE-specific constants (graph name,
node labels, edge labels, BELONGS_TO meta-edge) are gone — graph
schema now lives in the migration files under graph/migrations/.

Kept as a module-level constant rather than inlined into pgvector_store
so callers can override it explicitly (e.g. integration tests using a
distinct table name to avoid collisions with the production data).
"""

# pgvector default table name — symbol embeddings keyed by moniker.
DEFAULT_TABLE_NAME = "sutra_embeddings"
