"""
Full indexing CLI entry point.

Usage
-----
    python -m pipelines.full_index \\
        --root /path/to/repo \\
        --repo-url https://github.com/org/repo \\
        --output-dir /path/to/output \\
        [--config config/sutra.yaml] \\
        [--pg-url postgresql://user:pass@host:port/db] \\
        [--replace]

    --replace  DETACH DELETE existing symbols before writing (re-index).
               Without this flag, symbols are upserted (additive).

Environment variables:
    SUTRA_PG_URL   — PostgreSQL connection string
    OPENAI_API_KEY — required when embedder.provider = openai in config

Exit codes:
    0 — success
    1 — error (printed to stderr)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from pipelines._common import build_dependencies
from sutra.core.gitignore_filter import GitignoreFilter
from sutra.core.indexer import Indexer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sutra full index — index an entire repository."
    )
    parser.add_argument("--root", required=True, type=Path, help="Repository root path")
    parser.add_argument("--repo-url", required=True, help="Canonical remote URL")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/sutra.yaml"),
        help="Path to sutra.yaml (default: config/sutra.yaml)",
    )
    parser.add_argument("--pg-url", default=None, help="PostgreSQL connection string")
    parser.add_argument(
        "--replace", action="store_true",
        help="Delete existing symbols for this repo before writing (re-index mode).",
    )
    parser.add_argument(
        "--recreate-embeddings-table",
        action="store_true",
        dest="recreate_embeddings_table",
        help=(
            "DROP and recreate the sutra_embeddings table before indexing. "
            "WARNING: destroys all existing embeddings. Use when switching "
            "embedder providers that change vector dimensions (e.g. local→openai)."
        ),
    )
    parser.add_argument(
        "--resolver",
        choices=["heuristic", "lsp", "none"],
        default="heuristic",
        help=(
            "CALLS resolution strategy. 'heuristic' (default) resolves "
            "intra-repo calls by local/import/unique rules (P20-lite); "
            "'lsp' chains the heuristic with pyright type inference for "
            "the ambiguous residue — Python files only, needs pyright "
            "installed (P20-full); 'none' preserves Phase 1 behavior."
        ),
    )
    args = parser.parse_args(argv)

    pg_url = args.pg_url or os.environ.get("SUTRA_PG_URL")
    deps = build_dependencies(
        config_path=args.config,
        pg_url=pg_url,
        recreate_embeddings=args.recreate_embeddings_table,
    )

    try:
        gitignore_filter = GitignoreFilter(args.root)

        resolver = None
        if args.resolver == "heuristic":
            from sutra.core.resolver import HeuristicResolver
            resolver = HeuristicResolver()
        elif args.resolver == "lsp":
            from sutra.core.resolver import HeuristicResolver
            from sutra.core.resolver.lsp_resolver import ChainResolver, LspResolver
            # Cheap rules first; pyright only sees the ambiguous residue.
            resolver = ChainResolver(
                HeuristicResolver(), LspResolver(root=args.root)
            )

        indexer = Indexer(
            adapters=deps.adapters,
            exporter=deps.exporter,
            embedder=deps.embedder,
            graph_writer=deps.graph_writer,
            pgvector_store=deps.pgvector_store,
            gitignore_filter=gitignore_filter,
            resolver=resolver,
        )

        result = indexer.index(
            args.root, args.repo_url, args.output_dir, replace=args.replace
        )

        print(
            f"Full index complete: {len(result.symbols)} symbols, "
            f"{len(result.files)} files, commit {result.commit_hash}"
        )
        usage = deps.embedder.usage_stats()
        if usage:
            total = usage.get("total_tokens", 0)
            cost = usage.get("estimated_cost_usd", 0.0)
            model = usage.get("model", "unknown")
            rate = usage.get("usd_per_1m_tokens", 0.0)
            print(
                "Embedding usage: "
                f"model={model}, total_tokens={total}, "
                f"estimated_cost_usd=${float(cost):.8f} "
                f"(rate=${float(rate):.4f}/1M tokens)"
            )
        else:
            print(
                "Embedding usage unavailable: provider does not expose token usage "
                "(e.g. fixture/local embedder)."
            )
        if result.failed_files:
            print(f"Failed files ({len(result.failed_files)}):", file=sys.stderr)
            for path, err in result.failed_files:
                print(f"  {path}: {err}", file=sys.stderr)

        return 0
    finally:
        deps.close()


if __name__ == "__main__":
    sys.exit(main())
