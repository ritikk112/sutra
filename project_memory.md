# Sutra — Project Memory

These are the persistent design decisions and constraints for the Sutra project. Treat these as authoritative — they were locked in during the design phase and should not be changed without explicit discussion.

## Project Identity

Sutra is a successor to a previous "Master Repo" project. It indexes public git repositories and outputs:
1. A JSON graph file with documented schema (symbols, relationships, files, metadata)
2. A separate binary embeddings file linked to graph nodes by ID

The artifacts are designed to be loaded into a shared MCP server that team members query as a RAG over indexed repositories. The current use case is an internal team tool, not a public service.

## Core Principles

**Pure code-based analysis, no LLM enrichment in the indexing pipeline.** The graph structure plus embeddings should be rich enough that a downstream LLM can understand the repository without us pre-summarizing anything. LLM calls during indexing would be cost-prohibitive at scale.

**Graph nodes store metadata only — no function bodies in the database.** Storing bodies would duplicate the source code. Bodies stay in source files. However, function bodies ARE included in the embedding input so vectors match against actual behavior, not just signatures.

**Plug-and-play architecture for future extensions.** The resolver, embedder, and output modules all follow the same pattern: abstract base class, factory or registry, swappable implementations. Adding LSP resolution, new embedders, or markdown output should be purely additive with no changes to existing code.

## Tech Stack (Locked In)

- **Tree-sitter** used directly (not via ast-grep) for parsing 40+ languages via `.scm` query files. One query file per language. Incremental parsing supports diff-based re-indexing.
- **PostgreSQL + Apache AGE** for graph storage with openCypher queries. Single database for graph and relational data.
- **pgvector** for embedding storage in the same Postgres instance.
- **Configurable embedder** via factory pattern: local sentence-transformers OR OpenAI API, selected via YAML config.
- **Git diff** for incremental updates on commits.
- **SCIP-style monikers** for stable symbol IDs that survive renames.

## Phase 1 Scope

Languages: Python, TypeScript, Go only (test rigorously on real repos before adding more).
Tree-sitter only — no LSP (Phase 2 work).
JSON graph export with versioned schema.
Binary embeddings file.
Configurable embedder.
Incremental updates via git diff.
SCIP-style monikers with simplified scheme (no package metadata detection in v1).

**Explicitly skipped in v1:** markdown output (MCP makes it redundant), LSP cross-file resolution, languages beyond the initial 3, package metadata detection in monikers, MCP server itself.

## Data Model

Discriminated symbol types sharing a common base, with relationships as first-class objects separate from symbols:

- `SymbolBase` — common fields shared by all symbol types
- `FunctionSymbol(SymbolBase)` — parameters, return_type, docstring, decorators, is_async, complexity
- `ClassSymbol(SymbolBase)` — base_classes, docstring, decorators, is_abstract
- `MethodSymbol(FunctionSymbol)` — enclosing_class_id, is_static, is_constructor
- `VariableSymbol(SymbolBase)` — type_annotation, is_constant
- `ModuleSymbol(SymbolBase)` — docstring
- `Symbol = Union[FunctionSymbol | ClassSymbol | MethodSymbol | VariableSymbol | ModuleSymbol]`

Relationships are separate first-class objects with `source_id`, `target_id` (resolved) or `target_name` (unresolved), `kind` enum, `location`, and `is_resolved` boolean. Phase 1 stores cross-file calls as unresolved; Phase 2 LSP resolver will upgrade them without changing other code.

## SCIP-Style Moniker Format

Format: `sutra <language> <repo_name> <relative_file_path> <descriptor_path>`

Descriptor suffixes carry semantic meaning:
- `/` — namespace/module separator
- `#` — class or type
- `.` — function, method, or term
- `()` — callable

Example: `sutra python my-app src/services/user.py UserService#create_user().`

Rename detection: compare body_hashes between disappearing and appearing monikers in the same commit. If body_hash matches → it's a rename. File renames detected via git diff rename markers (`R100 old_path new_path`).

## Embedding Strategy

Chunk format for functions and methods:
Function: <qualified_name>
File: <file_path>
Signature: <signature>
Docstring: <docstring if present>
Body:
<truncated body text>

Token budgets: 2000-4000 tokens per chunk for OpenAI text-embedding-3-small (8K limit), 400-500 for sentence-transformers (512 limit). Long bodies are truncated to first N lines.

Class chunks include name, file, base classes, docstring, and method names — but NOT method bodies (methods get their own embeddings).

## Module Structure
sutra/
├── core/
│   ├── extractor/          (base.py, tree_sitter_runner.py, moniker.py, adapters/)
│   ├── queries/            (python.scm, typescript.scm, go.scm)
│   ├── resolver/           (base.py, noop_resolver.py — LSP plug-and-play seam)
│   ├── embedder/           (base.py, chunk_builder.py, local.py, openai.py, factory.py)
│   ├── graph/              (schema.py, postgres_age.py, pgvector_store.py)
│   ├── differ/             (git_diff.py, symbol_differ.py, rename_detector.py)
│   └── output/             (base.py, json_graph_exporter.py)
├── mcp/                    (Phase 2 placeholder)
├── config/                 (sutra.yaml)
└── pipelines/              (full_index.py, incremental_update.py)

## Implementation Priorities (Build Order)

1. Core dataclasses (Symbol hierarchy, Relationship, IndexResult) in `extractor/base.py`
2. Moniker generation in `extractor/moniker.py`
3. Tree-sitter runner + Python adapter as the first vertical slice
4. `python.scm` query file
5. JSON graph exporter (no embedder yet — use fixture vectors for testing)
6. End-to-end test on a small Python repo
7. TypeScript adapter
8. Go adapter
9. Embedder (start with OpenAI, add local as second)
10. AGE writer + pgvector writer
11. Incremental update pipeline

## Known Tradeoffs Accepted

- Phase 1 cross-file references will be unresolved (target_name only). Pgvector semantic search bridges some of the gap.
- Apache AGE has operational warts: incubation status, Postgres version constraints, Cypher quirks.
- Tree-sitter grammar quality varies per language — initial 3 languages chosen for grammar maturity.
- File renames via `git mv` invalidate monikers in the renamed file (handled by git rename detection in differ).
- Long function bodies are truncated for embedding — first N lines typically capture intent.