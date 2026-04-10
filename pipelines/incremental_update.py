"""
Incremental update CLI entry point.

Usage
-----
    python -m pipelines.incremental_update \\
        --root /path/to/repo \\
        --repo-url https://github.com/org/repo \\
        --output-dir /path/to/output \\
        [--config config/sutra.yaml] \\
        [--pg-url postgresql://user:pass@host:port/db]

Environment variables (override CLI flags):
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
from sutra.core.incremental_updater import IncrementalUpdater


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sutra incremental update — re-index only changed files."
    )
    parser.add_argument("--root", required=True, type=Path, help="Repository root path")
    parser.add_argument("--repo-url", required=True, help="Canonical remote URL")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--config", type=Path, default=None, help="Path to sutra.yaml")
    parser.add_argument("--pg-url", default=None, help="PostgreSQL connection string")
    args = parser.parse_args(argv)

    pg_url = args.pg_url or os.environ.get("SUTRA_PG_URL")

    deps = build_dependencies(config_path=args.config, pg_url=pg_url)
    try:
        if deps.age_writer is None or deps.age_reader is None or deps.pgvector_store is None:
            print(
                "Error: --pg-url or SUTRA_PG_URL is required for incremental update.",
                file=sys.stderr,
            )
            return 1

        gitignore_filter = GitignoreFilter(args.root)
        updater = IncrementalUpdater(
            adapters=deps.adapters,
            embedder=deps.embedder,
            age_writer=deps.age_writer,
            age_reader=deps.age_reader,
            pgvector_store=deps.pgvector_store,
            gitignore_filter=gitignore_filter,
            exporter=deps.exporter,
        )

        result = updater.update(args.root, args.repo_url, args.output_dir)

        if result.fell_back_to_full_index:
            print(
                f"Full re-index: {result.added} symbols indexed "
                f"({result.old_commit_sha} → {result.new_commit_sha})"
            )
        else:
            print(
                f"Incremental update complete: "
                f"+{result.added} added, ~{result.updated} updated, "
                f"-{result.deleted} deleted, ={result.skipped} skipped "
                f"({result.old_commit_sha} → {result.new_commit_sha})"
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
