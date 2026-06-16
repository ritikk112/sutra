from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

# Query categories — fixed vocabulary so aggregates are comparable across
# datasets and priorities.  See SUTRA_PHASE2.md P12.
CATEGORIES = frozenset({
    "behavioral",      # "which function saves the listing in db"
    "entity",          # "the Listing model"
    "paraphrase",      # "auth flow" → authentication code
    "exact-name",      # "create_user"
    "known-failure",   # documented Phase 1 failure cases
})


@dataclass(frozen=True, slots=True)
class EvalCase:
    """
    One eval query.

    `expected` is an ANY-OF set: the query is a hit if any of these
    monikers appears in the top-k (multiple symbols can legitimately
    answer one query).  `must_include` is an ALL-OF set: every listed
    moniker should be present in the top-k for full credit — used for
    queries with several required answers.  `kind_filter` is carried for
    P16; the harness passes it through when the retriever supports it.
    """

    query: str
    category: str
    repo: str
    expected: tuple[str, ...]
    must_include: tuple[str, ...] = ()
    kind_filter: Optional[str] = None


def load_dataset(path: Path | str) -> list[EvalCase]:
    """
    Load one YAML dataset file.

    Format:
        repo: booth
        cases:
          - query: "which function saves the listing in db"
            category: behavioral
            expected: ["sutra python booth ..."]
            must_include: []          # optional
            kind_filter: function     # optional
    """
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    repo = data.get("repo")
    if not repo:
        raise ValueError(f"Dataset {path} is missing the top-level 'repo' key.")

    raw_cases = data.get("cases") or []
    if not raw_cases:
        raise ValueError(f"Dataset {path} has no cases.")

    cases: list[EvalCase] = []
    for i, raw in enumerate(raw_cases):
        query = (raw.get("query") or "").strip()
        if not query:
            raise ValueError(f"Dataset {path} case #{i}: empty query.")

        category = raw.get("category", "")
        if category not in CATEGORIES:
            raise ValueError(
                f"Dataset {path} case #{i} ({query!r}): unknown category "
                f"{category!r}. Valid: {sorted(CATEGORIES)}"
            )

        expected = tuple(raw.get("expected") or ())
        if not expected:
            raise ValueError(
                f"Dataset {path} case #{i} ({query!r}): 'expected' must list "
                f"at least one moniker."
            )

        cases.append(EvalCase(
            query=query,
            category=category,
            repo=repo,
            expected=expected,
            must_include=tuple(raw.get("must_include") or ()),
            kind_filter=raw.get("kind_filter"),
        ))

    return cases


def load_datasets(directory: Path | str) -> list[EvalCase]:
    """Load every *.yaml dataset under `directory` (sorted, deterministic)."""
    directory = Path(directory)
    files = sorted(directory.glob("*.yaml"))
    if not files:
        raise FileNotFoundError(f"No *.yaml datasets found in {directory}")
    cases: list[EvalCase] = []
    for f in files:
        cases.extend(load_dataset(f))
    return cases
