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
    parser.add_argument("--config", type=Path, default=None, help="Path to sutra.yaml")
    parser.add_argument("--pg-url", default=None, help="PostgreSQL connection string")
    parser.add_argument(
        "--replace", action="store_true",
        help="DETACH DELETE existing symbols before writing (re-index mode).",
    )
    args = parser.parse_args(argv)

    pg_url = args.pg_url or os.environ.get("SUTRA_PG_URL")
    deps = build_dependencies(config_path=args.config, pg_url=pg_url)

    try:
        gitignore_filter = GitignoreFilter(args.root)
        indexer = Indexer(
            adapters=deps.adapters,
            exporter=deps.exporter,
            embedder=deps.embedder,
            age_writer=deps.age_writer,
            pgvector_store=deps.pgvector_store,
            gitignore_filter=gitignore_filter,
        )

        # If replace=True and age_writer is present, pass replace to write_repository.
        # Indexer.index() does not expose replace directly; set it on the writer.
        if args.replace and deps.age_writer is not None:
            _orig = deps.age_writer.write_repository

            def _replace_write(result, replace=False):  # type: ignore[override]
                return _orig(result, replace=True)
            deps.age_writer.write_repository = _replace_write  # type: ignore[method-assign]

        result = indexer.index(args.root, args.repo_url, args.output_dir)

        print(
            f"Full index complete: {len(result.symbols)} symbols, "
            f"{len(result.files)} files, commit {result.commit_hash}"
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
