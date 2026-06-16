from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import yaml

from sutra.core.embedder.base import Embedder
from sutra.core.retrieval.query import KINDS, ParsedQuery

_DEFAULT_LEXICON = Path(__file__).parent / "lexicons" / "query_lexicon.yaml"

# A token "looks like code" when it carries identifier morphology the user
# typed deliberately: an underscore, an interior capital, or a dotted path.
_IDENTIFIER_RE = re.compile(
    r"[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+"      # snake_case
    r"|[a-z]+[A-Z][A-Za-z0-9]*"                # camelCase
    r"|[A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]*)+"   # PascalCase (≥2 words)
    r"|[A-Za-z][A-Za-z0-9]*(?:\.[A-Za-z][A-Za-z0-9_]*)+"  # dotted.path
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def _lemma_lite(word: str) -> str:
    """
    Tiny suffix-stripping lemmatizer for verb matching — deliberately dumb,
    deterministic, and dependency-free.  saves→save, saving→save (well,
    sav→ handled via lexicon lookup of both forms), attached→attach.
    """
    w = word.lower()
    for suffix, replacement in (
        ("ies", "y"),    # retries → retry
        ("sses", "ss"),  # processes → process
        ("ing", ""),     # saving → sav (second lookup tries +e: save)
        ("ed", ""),      # attached → attach
        ("es", ""),      # fetches → fetch
        ("s", ""),       # saves → save
    ):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[: len(w) - len(suffix)] + replacement
    return w


class QueryAnalyzer:
    """
    THE query analyzer — one instance, shared by every channel and the
    pipeline.  Pure-Python keyword/regex over a checked-in YAML lexicon;
    no LLM, no network.

    parse() produces the complete ParsedQuery: embedding (computed once,
    optional), conservative kind_hint, behavioral verbs, and code-entity
    tokens.  Channels never re-parse.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        lexicon_path: Path | str = _DEFAULT_LEXICON,
    ) -> None:
        self._embedder = embedder

        data = yaml.safe_load(Path(lexicon_path).read_text(encoding="utf-8"))

        # noun (lowercase) → frozenset of kinds; group name → kinds for ambiguity checks
        self._noun_kinds: dict[str, frozenset[str]] = {}
        self._noun_group: dict[str, str] = {}
        for group, spec in (data.get("kind_nouns") or {}).items():
            kinds = frozenset(spec["kinds"])
            unknown = kinds - KINDS
            if unknown:
                raise ValueError(
                    f"Lexicon group {group!r} names unknown kinds: {sorted(unknown)}"
                )
            for noun in spec["nouns"]:
                self._noun_kinds[noun.lower()] = kinds
                self._noun_group[noun.lower()] = group

        self._verbs = frozenset(
            v.lower() for v in (data.get("behavioral_verbs") or [])
        )

        # noun → words that, when immediately preceding it, void its
        # kind-noun reading ("image type" = data property, not a class ask).
        self._blockers: dict[str, frozenset[str]] = {
            noun.lower(): frozenset(w.lower() for w in words)
            for noun, words in (data.get("noun_collocation_blockers") or {}).items()
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, text: str, embed: bool = True) -> ParsedQuery:
        """
        Analyze `text`.  When `embed=True` and an embedder was supplied,
        the query embedding is computed here — exactly once per query.
        """
        words = [m.group(0) for m in _WORD_RE.finditer(text)]
        lowered = [w.lower() for w in words]

        kind_hint = self._kind_hint(lowered)
        verbs = self._extract_verbs(lowered)

        # Verb-derived fallback: behavioral verbs imply a callable, but only
        # when no explicit kind noun contradicts/none present (conservative).
        if kind_hint is None and verbs and not self._has_any_kind_noun(lowered):
            kind_hint = frozenset({"function", "method"})

        embedding = None
        if embed and self._embedder is not None:
            embedding = self._embedder.embed([text])[0]

        return ParsedQuery(
            text=text,
            embedding=embedding,
            kind_hint=kind_hint,
            verbs=verbs,
            entities=self._extract_entities(text),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _active_nouns(self, lowered: list[str]) -> list[str]:
        """Kind nouns in the query, minus collocation-blocked occurrences."""
        active: list[str] = []
        for i, w in enumerate(lowered):
            if w not in self._noun_kinds:
                continue
            blockers = self._blockers.get(w)
            if blockers and i > 0 and lowered[i - 1] in blockers:
                continue   # "image type", "content type", … — data phrase
            active.append(w)
        return active

    def _has_any_kind_noun(self, lowered: list[str]) -> bool:
        return bool(self._active_nouns(lowered))

    def _kind_hint(self, lowered: list[str]) -> Optional[frozenset[str]]:
        """Explicit kind nouns → hint.  Nouns from >1 group → ambiguous → None."""
        nouns = self._active_nouns(lowered)
        groups = {self._noun_group[w] for w in nouns}
        if len(groups) != 1:
            return None
        return self._noun_kinds[nouns[0]]

    def _extract_verbs(self, lowered: list[str]) -> frozenset[str]:
        found: set[str] = set()
        for w in lowered:
            if w in self._verbs:
                found.add(w)
                continue
            lemma = _lemma_lite(w)
            if lemma in self._verbs:
                found.add(lemma)
            elif lemma + "e" in self._verbs:   # saving → sav → save
                found.add(lemma + "e")
        return frozenset(found)

    @staticmethod
    def _extract_entities(text: str) -> tuple[str, ...]:
        seen: set[str] = set()
        out: list[str] = []
        for m in _IDENTIFIER_RE.finditer(text):
            token = m.group(0)
            if token not in seen:
                seen.add(token)
                out.append(token)
        return tuple(out)
