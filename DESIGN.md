# SUTRA — IMPLEMENTATION CONTEXT

## What Sutra Is
A backend service that takes a public Git repository URL and produces two artifacts:
1. **JSON graph file** with documented schema (symbols, relationships, files, metadata)
2. **Binary embeddings file** linked to graph nodes by ID

These get loaded into a shared MCP server that team members query as a RAG over indexed repos. Sutra is a successor to a previous "Master Repo" project. The defining principle is **pure code-based analysis with zero LLM enrichment in the indexing pipeline** — the graph structure plus embeddings should be rich enough that a downstream LLM can understand the repo without us pre-summarizing anything.

## Tech Stack (Locked In)
- **Tree-sitter** (used directly, not via ast-grep) for parsing 40+ languages via `.scm` query files. One query file per language. Incremental parsing supports diff-based re-indexing.
- **PostgreSQL + Apache AGE** for graph storage (openCypher queries, single database for graph + relational data)
- **pgvector** for embedding storage in the same Postgres instance
- **Configurable embedder** via factory pattern: local sentence-transformers OR OpenAI API, selected via YAML config
- **Git diff** for incremental updates on commits
- **SCIP-style monikers** for stable symbol IDs that survive renames

## Phase 1 Scope
- Languages: **Python, TypeScript, Go** only (test rigorously on real repos)
- Tree-sitter only — **no LSP** (Phase 2 work)
- JSON graph export with versioned schema
- Binary embeddings file
- Configurable embedder
- Incremental updates via git diff
- SCIP-style monikers

**Skip in v1:** markdown output (MCP makes it redundant), LSP cross-file resolution, languages beyond the initial 3, package metadata detection in monikers (use simplified scheme)

## Data Model

**Discriminated symbol types** (not a flat dataclass):
- `SymbolBase` — common fields: id (moniker), name, qualified_name, file_path, location (line_start/end, byte_start/end), body_hash, language, visibility, is_exported
- `FunctionSymbol(SymbolBase)` — parameters, return_type, docstring, decorators, is_async, complexity
- `ClassSymbol(SymbolBase)` — base_classes, docstring, decorators, is_abstract
- `MethodSymbol(FunctionSymbol)` — enclosing_class_id, is_static, is_constructor
- `VariableSymbol(SymbolBase)` — type_annotation, is_constant
- `ModuleSymbol(SymbolBase)` — docstring
- `Symbol = Union[FunctionSymbol | ClassSymbol | MethodSymbol | VariableSymbol | ModuleSymbol]`

**Relationships are first-class objects** (separate from symbols):
```python
@dataclass
class Relationship:
    source_id: str
    target_id: Optional[str]      # None if unresolved
    target_name: Optional[str]    # used when unresolved (Phase 1 default for cross-file)
    kind: RelationKind            # CALLS | EXTENDS | IMPLEMENTS | IMPORTS | CONTAINS | REFERENCES | RETURNS_TYPE | PARAMETER_TYPE
    location: Optional[Location]
    is_resolved: bool
    metadata: dict
```

**Pipeline output:**
```python
@dataclass
class FileExtraction:
    file: File
    symbols: list[Symbol]
    relationships: list[Relationship]

@dataclass
class IndexResult:
    repository: Repository
    files: list[File]
    symbols: list[Symbol]
    relationships: list[Relationship]
    indexed_at: datetime
    commit_hash: str
    languages: dict[str, int]
```

## SCIP-Style Moniker Format
```
sutra <language> <repo_name> <relative_file_path> <descriptor_path>
```

Descriptor suffixes carry semantic meaning:
- `/` — namespace/module separator
- `#` — class or type
- `.` — function, method, or term
- `()` — callable

Examples:
- `sutra python my-app src/services/user.py UserService#create_user().`
- `sutra typescript frontend src/components/Button.tsx Button#`
- `sutra go backend internal/auth/jwt.go ValidateToken().`

**Rename detection:** When re-indexing, compare body_hashes between disappearing and appearing monikers in the same commit. If body_hash matches → it's a rename, preserve the connection. File renames detected via git diff rename markers (`R100 old_path new_path`).

## Embedding Strategy

**Critical principle:** Graph nodes store **metadata only** — no function bodies in the database (would just duplicate the source code). Bodies stay in source files. But bodies ARE included in the embedding input so vectors match against actual behavior.

**Chunk format for functions/methods:**
```
Function: <qualified_name>
File: <file_path>
Signature: <signature>
Docstring: <docstring if present>
Body:
<truncated body text>
```

**Chunk format for classes:**
```
Class: <qualified_name>
File: <file_path>
Extends: <base_classes>
Docstring: <docstring>
Methods: <list of method names>
```
(Don't embed method bodies inside class chunks — methods get their own embeddings.)

**Token budgets:** 2000-4000 tokens per chunk for OpenAI text-embedding-3-small (8K limit), 400-500 for sentence-transformers (512 limit). Long bodies: truncate to first N lines.

## JSON Graph Export Schema
```json
{
  "sutra_version": "1.0",
  "schema_version": "1.0",
  "repository": {
    "url": "...", "name": "...", "indexed_at": "ISO8601",
    "commit_hash": "...", "languages": {"python": 142}
  },
  "symbols": [
    {
      "id": "scip-style-moniker",
      "kind": "function",
      "name": "...", "qualified_name": "...",
      "file_path": "...", "location": {...},
      "signature": "...", "parameters": [...],
      "return_type": "...", "docstring": "...",
      "decorators": [], "is_async": true, "is_exported": true,
      "visibility": "public", "body_hash": "sha256:...",
      "language": "typescript",
      "embedding_id": "uuid"
    }
  ],
  "relationships": [
    {
      "source_id": "...", "target_id": "...", "target_name": "...",
      "kind": "calls", "location": {"line": 45, "column": 21},
      "is_resolved": true, "metadata": {}
    }
  ],
  "files": [{"path": "...", "language": "...", "size_bytes": ..., "hash": "..."}],
  "embeddings": {
    "format": "pgvector_export",
    "dimensions": 1536,
    "model": "text-embedding-3-small",
    "vectors_file": "embeddings.bin"
  }
}
```

## Module Structure
```
sutra/
├── core/
│   ├── extractor/
│   │   ├── base.py                  # Symbol/Relationship/IndexResult dataclasses
│   │   ├── tree_sitter_runner.py    # generic tree-sitter orchestration
│   │   ├── moniker.py               # SCIP-style ID generation
│   │   └── adapters/
│   │       ├── python.py            # loads python.scm, walks captures → Symbols
│   │       ├── typescript.py
│   │       └── go.py
│   ├── queries/
│   │   ├── python.scm               # tree-sitter S-expression queries
│   │   ├── typescript.scm
│   │   └── go.scm
│   ├── resolver/
│   │   ├── base.py                  # abstract Resolver interface
│   │   └── noop_resolver.py         # Phase 1 default (passes through)
│   │   # Phase 2: lsp_resolver.py added here, no other changes needed
│   ├── embedder/
│   │   ├── base.py                  # abstract Embedder interface
│   │   ├── chunk_builder.py         # builds embedding input from symbol + body
│   │   ├── local.py                 # sentence-transformers
│   │   ├── openai.py                # OpenAI API
│   │   └── factory.py               # reads config, returns implementation
│   ├── graph/
│   │   ├── schema.py                # AGE node/edge schema definitions
│   │   ├── postgres_age.py          # AGE writer (Cypher via SQL)
│   │   └── pgvector_store.py        # embedding writer
│   ├── differ/
│   │   ├── git_diff.py              # parse git diff output
│   │   ├── symbol_differ.py         # compare body_hashes for added/modified/deleted
│   │   └── rename_detector.py       # detect file renames from git
│   └── output/
│       ├── base.py                  # OutputGenerator interface
│       └── json_graph_exporter.py   # v1 JSON export
│       # Phase 2: markdown_generator.py
├── mcp/                             # Phase 2 placeholder
├── config/
│   └── sutra.yaml                   # embedder config, languages, paths
└── pipelines/
    ├── full_index.py                # initial indexing entry point
    └── incremental_update.py        # diff-based update entry point
```

## Pipeline Flow

**Full indexing:**
1. Clone repo to temp dir
2. Walk file tree, detect languages by extension and project markers
3. For each file: run tree-sitter adapter → extract symbols + relationships → assign monikers
4. Pass through resolver (noop in Phase 1)
5. Build embedding chunks via chunk_builder
6. Generate embeddings via configured embedder
7. Write nodes and edges to AGE graph
8. Write embeddings to pgvector
9. Export as JSON graph + binary embeddings file
10. Return artifacts

**Incremental update:**
1. `git fetch && git diff <last_indexed_commit>..HEAD --name-only` (with rename detection flags)
2. For each changed file: re-extract with tree-sitter
3. Compare new symbol body_hashes against stored:
   - New monikers → INSERT
   - Same moniker, different body_hash → UPDATE + regenerate embedding
   - Disappeared monikers → check rename detection, then DELETE
4. Apply graph deltas
5. Update `repository.last_indexed_commit`

## Configuration File Format
```yaml
embedder:
  provider: openai          # or "local"
  model: text-embedding-3-small
  api_key: ${OPENAI_API_KEY}
  batch_size: 100
  # local-only options
  device: cpu

languages:
  - python
  - typescript
  - go

storage:
  postgres_url: ${POSTGRES_URL}
  
indexing:
  max_repo_size_mb: 500
  exclude_patterns:
    - "node_modules/**"
    - ".git/**"
    - "**/__pycache__/**"
```

## Key Design Decisions & Rationale

1. **Tree-sitter direct, not ast-grep:** Sutra does systematic enumeration (walk every node, extract everything), not pattern search. Tree-sitter's native query API + tree walking is the right fit. ast-grep is optimized for find-and-rewrite workflows.

2. **Discriminated symbol types over flat class:** Functions, classes, and variables are genuinely different — flat class with nullable fields creates messy code. Discriminated union enables clean pattern matching and prevents accidental field access errors.

3. **Relationships separate from symbols:** Maps directly to graph DB. Allows Phase 2 LSP resolver to upgrade unresolved → resolved without touching symbol code. Relationships can carry their own metadata (call site location, etc.).

4. **No function bodies in DB, but bodies in embeddings:** Storing bodies duplicates the source code. Embeddings encode behavior semantically without storing the original. Source code stays in source files.

5. **SCIP-style monikers in Phase 1:** Stable IDs that survive renames within files. Hierarchical structure with semantic suffixes. Globally unique enough for cross-repo references in the future.

6. **PostgreSQL + AGE + pgvector single-database approach:** One database, ACID transactions, hybrid retrieval (graph traversal + vector similarity) in a single SQL query. Operational tradeoff: AGE is in incubation, requires specific Postgres versions, has Cypher quirks.

7. **Plug-and-play resolver/embedder/output:** All three follow the same pattern — abstract base class, factory or registry, swappable implementations. Adding LSP, new embedders, or markdown output is purely additive.

## Known Tradeoffs Accepted
- Phase 1 cross-file references will be unresolved (target_name only). pgvector semantic search bridges some of the gap.
- Apache AGE has operational warts (incubation status, Postgres version constraints, Cypher quirks).
- Tree-sitter grammar quality varies per language — initial 3 languages chosen for grammar maturity.
- File renames via `git mv` invalidate monikers in the renamed file (handled by git rename detection in differ).
- Long function bodies are truncated for embedding — first N lines typically capture intent.

## Implementation Priorities for First PR
1. Core dataclasses (Symbol hierarchy, Relationship, IndexResult) in `extractor/base.py`
2. Moniker generation in `extractor/moniker.py`
3. Tree-sitter runner + Python adapter as the first vertical slice
4. `python.scm` query file
5. JSON graph exporter (no embedder yet, write fixture vectors)
6. End-to-end test on a small Python repo
7. Then: TypeScript and Go adapters
8. Then: embedder (start with OpenAI, local as second)
9. Then: AGE writer + pgvector writer
10. Then: incremental update pipeline

---

This is the full design. The new chat agent should pull project memory entries 1-7 for the persistent state and use this compacted context as the conversation reference. Ready to start building.