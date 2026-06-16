from __future__ import annotations

import re

# Splits an identifier into its word parts:
#   camelCase / PascalCase boundaries, snake_case, kebab-case, digits.
#   "getUserIdFromToken" → get, User, Id, From, Token
#   "HTTPSConnection"    → HTTPS, Connection   (acronym-aware)
_IDENT_PART = re.compile(
    r"[A-Z]+(?=[A-Z][a-z])"   # acronym run before a capitalized word: HTTPS|Connection
    r"|[A-Z]?[a-z]+"          # capitalized or lowercase word
    r"|[A-Z]+"                # trailing acronym run
    r"|\d+"                   # digit run
)

# Splits raw text into identifier-ish tokens (word chars), dropping punctuation.
_WORD = re.compile(r"[A-Za-z0-9_]+")


def split_identifier(identifier: str) -> list[str]:
    """Split one identifier into lowercase word parts."""
    parts: list[str] = []
    for chunk in identifier.split("_"):
        parts.extend(m.group(0).lower() for m in _IDENT_PART.finditer(chunk))
    return parts


def tokenize(text: str) -> list[str]:
    """
    Code-aware tokenizer shared by the BM25 corpus and BM25 queries.

    Every word-token is emitted lowercase; identifiers additionally emit
    their split parts so "getUserIdFromToken" matches the query "user id
    from token" AND the exact-name query "getUserIdFromToken".
    """
    tokens: list[str] = []
    for m in _WORD.finditer(text):
        word = m.group(0)
        lower = word.lower()
        tokens.append(lower)
        parts = split_identifier(word)
        # Only add parts when the identifier actually split into pieces —
        # plain lowercase words would otherwise be double-counted.
        if len(parts) > 1:
            tokens.extend(parts)
    return tokens
