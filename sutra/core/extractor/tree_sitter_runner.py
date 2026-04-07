from __future__ import annotations

from tree_sitter import Language, Node, Parser, Query, QueryCursor, Tree


def parse(source_bytes: bytes, language: Language) -> Tree:
    """Parse source bytes with the given tree-sitter Language. Always returns a Tree."""
    return Parser(language).parse(source_bytes)


def run_query(node: Node, language: Language, query_scm: str) -> dict[str, list[Node]]:
    """
    Execute a tree-sitter S-expression query against a node.

    Returns a dict mapping capture name → list of matching Nodes.
    Capture names with dots (e.g. @class.def) are preserved as-is in the keys.

    API: tree-sitter 0.25.x uses Query() + QueryCursor().captures().
    """
    query = Query(language, query_scm)
    return QueryCursor(query).captures(node)
