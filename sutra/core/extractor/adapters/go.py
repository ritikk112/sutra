"""
Go language adapter for Sutra — Priority 8.

Verified against tree-sitter-go 0.25.0. See RESEARCH.md for node type reference.

Design decisions locked in this module:
  - Untyped package-level vars: extracted with type_annotation="<inferred>"
  - Untyped consts: extracted with type_annotation=None
  - Pointer vs value receiver: stored as receiver_kind on MethodSymbol
  - Cross-file method→struct links: enclosing_class_id=None; resolved by
    Indexer._resolve_go_methods() post-aggregation pass
  - Doc comment: consecutive // siblings; blank line = byte-range gap >1 newline;
    //go:... directive lines are skipped (not documentation)
  - Build tags: //go:build ignore → empty FileExtraction
  - init() disambiguation: init@{line_start} in moniker when a file has multiple
  - Struct embedding → EXTENDS with metadata.embedding_kind="struct"
  - Interface type constraints (binary/union types) → not EXTENDS (only pure
    type_identifier type_elem children generate EXTENDS)
  - import "C" (cgo) → regular IMPORTS, target_name="C", no crash
  - blank import (_) and dot import (.) → skipped
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional

import tree_sitter_go as tsgo
from tree_sitter import Language, Parser

from sutra.core.extractor.base import (
    ClassSymbol,
    File,
    FileExtraction,
    FunctionSymbol,
    Location,
    MethodSymbol,
    ModuleSymbol,
    Parameter,
    RelationKind,
    Relationship,
    VariableSymbol,
    Visibility,
)
from sutra.core.extractor.moniker import MonikerBuilder

# //go:build ignore at the start of a file → skip entirely
_BUILD_IGNORE_RE = re.compile(r"^//go:build\s+ignore\b", re.MULTILINE)

# Cyclomatic complexity branch node types for Go
_BRANCH_NODES = frozenset({
    "if_statement",
    "for_statement",
    "expression_switch_statement",
    "type_switch_statement",
    "select_statement",
    "case_clause",
    "communication_case",
})


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _node_text(node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _node_loc(node) -> Location:
    return Location(
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        byte_start=node.start_byte,
        byte_end=node.end_byte,
        column_start=node.start_point[1],
        column_end=node.end_point[1],
    )


def _go_visibility(name: str) -> tuple[Visibility, bool]:
    """Go name-based visibility: uppercase first letter = public/exported."""
    if name and name[0].isupper():
        return Visibility.PUBLIC, True
    return Visibility.INTERNAL, False


def _doc_comment(node, siblings: list, source_bytes: bytes) -> Optional[str]:
    """
    Collect consecutive // comment nodes immediately before `node` in `siblings`.

    Blank line detection: if the byte gap between a comment's end and the next
    sibling's start contains more than one newline, a blank line exists and the
    comment chain is broken at that point.

    Lines starting with //go: (build directives, generate directives, etc.) are
    silently skipped — they are not documentation.
    """
    idx = next((i for i, s in enumerate(siblings) if s.id == node.id), None)
    if idx is None:
        return None

    comment_nodes: list = []
    i = idx - 1
    while i >= 0:
        sib = siblings[i]
        if sib.type != "comment":
            break
        # Check byte gap between end of this comment and start of next item
        gap = source_bytes[sib.end_byte:siblings[i + 1].start_byte]
        if gap.count(b"\n") > 1:
            break  # blank line separates comment block from declaration
        comment_nodes.insert(0, sib)
        i -= 1

    if not comment_nodes:
        return None

    lines = []
    for cn in comment_nodes:
        text = _node_text(cn, source_bytes)
        if text.startswith("//go:"):
            continue  # directive, not documentation
        if text.startswith("// "):
            lines.append(text[3:].rstrip())
        elif text.startswith("//"):
            lines.append(text[2:].rstrip())

    lines = [ln for ln in lines if ln.strip()]
    return "\n".join(lines) if lines else None


def _extract_params(param_list_node, source_bytes: bytes) -> list[Parameter]:
    """Extract Parameters from a parameter_list node."""
    params = []
    for child in param_list_node.children:
        if child.type == "parameter_declaration":
            name_node = child.child_by_field_name("name")
            type_node = child.child_by_field_name("type")
            name = _node_text(name_node, source_bytes) if name_node else ""
            type_text = _node_text(type_node, source_bytes) if type_node else None
            params.append(Parameter(name=name, type_annotation=type_text))
        elif child.type == "variadic_parameter_declaration":
            name_node = child.child_by_field_name("name")
            type_node = child.child_by_field_name("type")
            name = _node_text(name_node, source_bytes) if name_node else ""
            type_text = _node_text(type_node, source_bytes) if type_node else None
            params.append(Parameter(name=name, type_annotation=type_text, is_variadic=True))
    return params


def _return_type_text(func_node, source_bytes: bytes) -> Optional[str]:
    """Return type from result field of function_declaration or method_declaration."""
    result = func_node.child_by_field_name("result")
    if result is None:
        return None
    return _node_text(result, source_bytes)


def _signature_text(func_node, source_bytes: bytes) -> str:
    """Everything from 'func' to end of result clause (before body block)."""
    body = func_node.child_by_field_name("body")
    if body:
        sig = source_bytes[func_node.start_byte:body.start_byte]
    else:
        sig = source_bytes[func_node.start_byte:func_node.end_byte]
    return sig.decode("utf-8", errors="replace").strip()


def _complexity(body_node, source_bytes: bytes) -> int:
    """McCabe cyclomatic complexity for a Go function/method body."""
    count = 1
    stack = list(body_node.children)
    while stack:
        n = stack.pop()
        if n.type == "func_literal":
            continue  # separate scope — don't descend
        if n.type in _BRANCH_NODES:
            count += 1
        elif n.type == "binary_expression":
            op = n.child_by_field_name("operator")
            if op and _node_text(op, source_bytes) in ("&&", "||"):
                count += 1
        stack.extend(n.children)
    return count


def _extract_calls(body_node, source_id: str, source_bytes: bytes) -> list[Relationship]:
    """Walk a Go function/method body for call_expression, respecting func_literal scope."""
    rels = []
    stack = list(body_node.children)
    while stack:
        n = stack.pop()
        if n.type == "func_literal":
            continue  # scope boundary — inner calls belong to the literal, not outer func
        if n.type == "call_expression":
            func_field = n.child_by_field_name("function")
            if func_field is not None:
                if func_field.type == "identifier":
                    rels.append(Relationship(
                        source_id=source_id,
                        kind=RelationKind.CALLS,
                        is_resolved=False,
                        target_id=None,
                        target_name=_node_text(func_field, source_bytes),
                        location=_node_loc(n),
                        metadata={"call_form": "direct"},
                    ))
                elif func_field.type == "selector_expression":
                    obj = func_field.child_by_field_name("operand")
                    field = func_field.child_by_field_name("field")
                    if obj and field:
                        rels.append(Relationship(
                            source_id=source_id,
                            kind=RelationKind.CALLS,
                            is_resolved=False,
                            target_id=None,
                            target_name=_node_text(field, source_bytes),
                            location=_node_loc(n),
                            metadata={
                                "call_form": "method",
                                "receiver": _node_text(obj, source_bytes),
                            },
                        ))
        stack.extend(n.children)
    return rels


def _collect_specs(decl_node, spec_type: str) -> list:
    """Collect all spec nodes (const_spec / var_spec) from a declaration node."""
    specs = []
    list_type = f"{spec_type}_list"
    for child in decl_node.children:
        if child.type == spec_type:
            specs.append(child)
        elif child.type == list_type:
            specs.extend(c for c in child.children if c.type == spec_type)
    return specs


def _first_named_child(node, child_type: str):
    """Return first direct child with the given node type, or None."""
    return next((c for c in node.children if c.type == child_type), None)


def _type_identifier_text(type_node, source_bytes: bytes) -> Optional[str]:
    """
    Extract the base type_identifier from a type node, handling pointer_type and
    generic_type wrappers. Returns None if the type is not a simple named type.

    NOTE: In tree-sitter-go 0.25.0, pointer_type has no named "type" field;
    the inner type is a direct (unnamed) child — use child iteration, not
    child_by_field_name.
    """
    if type_node.type == "type_identifier":
        return _node_text(type_node, source_bytes)
    if type_node.type == "pointer_type":
        # Inner node is a direct child (not a named field in this grammar version)
        inner = _first_named_child(type_node, "type_identifier")
        if inner:
            return _node_text(inner, source_bytes)
        # Nested pointer or generic: recurse on any named child
        for c in type_node.children:
            if c.is_named:
                return _type_identifier_text(c, source_bytes)
    if type_node.type == "generic_type":
        inner = _first_named_child(type_node, "type_identifier")
        if inner:
            return _node_text(inner, source_bytes)
    return None


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class GoAdapter:
    """
    Tree-sitter Go adapter for Sutra.

    Handles .go files. _test.go files are excluded by the Indexer before
    this adapter is called.
    """

    def __init__(self) -> None:
        lang = Language(tsgo.language())
        self._parser = Parser(lang)

    def extract(
        self, file_path: str, source_bytes: bytes, repo_name: str
    ) -> FileExtraction:
        source_text = source_bytes.decode("utf-8", errors="replace")

        # //go:build ignore → file is excluded from normal compilation; skip it
        if _BUILD_IGNORE_RE.search(source_text):
            return FileExtraction(
                file=File(
                    path=file_path,
                    language="go",
                    size_bytes=len(source_bytes),
                    hash=_sha256(source_bytes),
                ),
                symbols=[],
                relationships=[],
            )

        tree = self._parser.parse(source_bytes)
        root = tree.root_node

        package_name = self._get_package_name(root, source_bytes)
        if not package_name:
            package_name = Path(file_path).stem

        mb = MonikerBuilder(language="go", repo_name=repo_name)
        top_level = list(root.children)

        symbols = []
        relationships = []

        # Module symbol
        module_path = file_path[:-3] if file_path.endswith(".go") else file_path
        mod_id = mb.for_module(file_path, module_path)
        mod_sym = ModuleSymbol(
            id=mod_id,
            name=Path(file_path).stem,
            qualified_name=package_name,
            file_path=file_path,
            location=_node_loc(root),
            body_hash=_sha256(source_bytes),
            language="go",
            visibility=Visibility.PUBLIC,
            is_exported=True,
            docstring=None,
        )
        symbols.append(mod_sym)

        # First pass: index type names defined in this file for same-file
        # method→struct linking (cross-file linking done by Indexer post-pass).
        in_file_classes: dict[str, str] = {}  # type_name → class_id
        for node in top_level:
            if node.type == "type_declaration":
                for spec in (
                    c for c in node.children
                    if c.type in ("type_spec", "type_alias")
                ):
                    name_node = spec.child_by_field_name("name")
                    type_node = spec.child_by_field_name("type")
                    if name_node and type_node and type_node.type in (
                        "struct_type", "interface_type"
                    ):
                        tname = _node_text(name_node, source_bytes)
                        in_file_classes[tname] = mb.for_class(file_path, tname)

        # top_level_ids: symbols that go directly under the module CONTAINS
        # (methods are excluded — they go under their receiver struct)
        top_level_ids: list[str] = []

        # Second pass: process all declarations
        for node in top_level:
            if node.type == "import_declaration":
                relationships.extend(
                    self._process_imports(node, mod_id, source_bytes)
                )

            elif node.type == "function_declaration":
                sym, rels = self._process_function(
                    node, file_path, package_name, mb, source_bytes, top_level
                )
                if sym is not None:
                    symbols.append(sym)
                    top_level_ids.append(sym.id)
                    relationships.extend(rels)

            elif node.type == "method_declaration":
                sym, rels = self._process_method(
                    node, file_path, package_name, mb, source_bytes,
                    top_level, in_file_classes,
                )
                if sym is not None:
                    symbols.append(sym)
                    # Methods are not in top_level_ids — they belong under their struct
                    relationships.extend(rels)

            elif node.type == "type_declaration":
                new_syms, new_rels = self._process_type_declaration(
                    node, file_path, package_name, mb, source_bytes, top_level
                )
                symbols.extend(new_syms)
                # Interface methods (MethodSymbol) are children of the interface,
                # NOT direct children of the module — exclude them from top_level_ids
                top_level_ids.extend(
                    s.id for s in new_syms if not isinstance(s, MethodSymbol)
                )
                relationships.extend(new_rels)

            elif node.type in ("const_declaration", "var_declaration"):
                new_syms = self._process_const_var(
                    node, file_path, package_name, mb, source_bytes, top_level
                )
                symbols.extend(new_syms)
                top_level_ids.extend(s.id for s in new_syms)

        # Module CONTAINS all top-level symbols (not methods)
        for sym_id in top_level_ids:
            relationships.append(Relationship(
                source_id=mod_id,
                kind=RelationKind.CONTAINS,
                is_resolved=True,
                target_id=sym_id,
                target_name=None,
            ))

        return FileExtraction(
            file=File(
                path=file_path,
                language="go",
                size_bytes=len(source_bytes),
                hash=_sha256(source_bytes),
            ),
            symbols=symbols,
            relationships=relationships,
        )

    # ------------------------------------------------------------------
    # Package name
    # ------------------------------------------------------------------

    def _get_package_name(self, root, source_bytes: bytes) -> Optional[str]:
        for child in root.children:
            if child.type == "package_clause":
                for c in child.children:
                    if c.type == "package_identifier":
                        return _node_text(c, source_bytes)
        return None

    # ------------------------------------------------------------------
    # Imports
    # ------------------------------------------------------------------

    def _process_imports(
        self, node, source_id: str, source_bytes: bytes
    ) -> list[Relationship]:
        rels = []
        specs = []
        for child in node.children:
            if child.type == "import_spec":
                specs.append(child)
            elif child.type == "import_spec_list":
                specs.extend(c for c in child.children if c.type == "import_spec")

        for spec in specs:
            path_node = next(
                (c for c in spec.children if c.type == "interpreted_string_literal"),
                None,
            )
            if path_node is None:
                continue

            import_path = _node_text(path_node, source_bytes).strip('"')
            target_name = import_path.rsplit("/", 1)[-1]  # last path segment

            # In tree-sitter-go 0.25.0:
            #   aliased import: alias is `package_identifier` (not `identifier`)
            #   blank import:   alias is `blank_identifier` (not `identifier`)
            #   dot import:     alias is `identifier` with text "."
            alias_node = next(
                (
                    c for c in spec.children
                    if c.type in ("package_identifier", "identifier", "blank_identifier")
                ),
                None,
            )
            alias = _node_text(alias_node, source_bytes) if alias_node else None

            # Blank import (_ "pkg") and dot import (. "pkg") → skip
            if alias in ("_", ".") or (alias_node and alias_node.type == "blank_identifier"):
                continue

            rels.append(Relationship(
                source_id=source_id,
                kind=RelationKind.IMPORTS,
                is_resolved=False,
                target_id=None,
                target_name=alias if alias else target_name,
                location=_node_loc(path_node),
                metadata={"import_source": import_path, "alias": alias},
            ))
        return rels

    # ------------------------------------------------------------------
    # Functions
    # ------------------------------------------------------------------

    def _process_function(
        self,
        node,
        file_path: str,
        package_name: str,
        mb: MonikerBuilder,
        source_bytes: bytes,
        siblings: list,
    ) -> tuple[Optional[FunctionSymbol], list[Relationship]]:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None, []

        name = _node_text(name_node, source_bytes)

        # init() disambiguation — multiple init() per file is valid Go
        if name == "init":
            line_start = node.start_point[0] + 1
            moniker_name = f"init@{line_start}"
        else:
            moniker_name = name

        visibility, is_exported = _go_visibility(name)
        func_id = mb.for_function(file_path, moniker_name)

        params_node = node.child_by_field_name("parameters")
        params = _extract_params(params_node, source_bytes) if params_node else []

        body_node = node.child_by_field_name("body")
        body_bytes = (
            source_bytes[body_node.start_byte:body_node.end_byte]
            if body_node else b""
        )

        sym = FunctionSymbol(
            id=func_id,
            name=name,
            qualified_name=f"{package_name}.{moniker_name}",
            file_path=file_path,
            location=_node_loc(node),
            body_hash=_sha256(body_bytes),
            language="go",
            visibility=visibility,
            is_exported=is_exported,
            signature=_signature_text(node, source_bytes),
            parameters=params,
            return_type=_return_type_text(node, source_bytes),
            docstring=_doc_comment(node, siblings, source_bytes),
            decorators=[],
            is_async=False,
            complexity=_complexity(body_node, source_bytes) if body_node else None,
        )

        rels = []
        if body_node:
            rels.extend(_extract_calls(body_node, func_id, source_bytes))

        return sym, rels

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def _process_method(
        self,
        node,
        file_path: str,
        package_name: str,
        mb: MonikerBuilder,
        source_bytes: bytes,
        siblings: list,
        in_file_classes: dict[str, str],
    ) -> tuple[Optional[MethodSymbol], list[Relationship]]:
        """
        Process a method_declaration node.

        If the receiver type is defined in this file, emit a resolved CONTAINS
        relationship and set enclosing_class_id.  Otherwise, leave
        enclosing_class_id=None and emit no CONTAINS — the Indexer post-pass
        (_resolve_go_methods) will link cross-file methods after all files are
        processed.
        """
        receiver_list = node.child_by_field_name("receiver")
        name_node = node.child_by_field_name("name")
        if not receiver_list or not name_node:
            return None, []

        method_name = _node_text(name_node, source_bytes)
        visibility, is_exported = _go_visibility(method_name)

        receiver_type_name, receiver_kind = self._parse_receiver(
            receiver_list, source_bytes
        )
        if not receiver_type_name:
            return None, []

        method_id = mb.for_method(file_path, receiver_type_name, method_name)

        params_node = node.child_by_field_name("parameters")
        params = _extract_params(params_node, source_bytes) if params_node else []

        body_node = node.child_by_field_name("body")
        body_bytes = (
            source_bytes[body_node.start_byte:body_node.end_byte]
            if body_node else b""
        )

        # Same-file lookup; None means cross-file (post-pass will resolve)
        class_id = in_file_classes.get(receiver_type_name)

        sym = MethodSymbol(
            id=method_id,
            name=method_name,
            qualified_name=f"{package_name}.{receiver_type_name}.{method_name}",
            file_path=file_path,
            location=_node_loc(node),
            body_hash=_sha256(body_bytes),
            language="go",
            visibility=visibility,
            is_exported=is_exported,
            signature=_signature_text(node, source_bytes),
            parameters=params,
            return_type=_return_type_text(node, source_bytes),
            docstring=_doc_comment(node, siblings, source_bytes),
            decorators=[],
            is_async=False,
            complexity=_complexity(body_node, source_bytes) if body_node else None,
            enclosing_class_id=class_id,
            is_static=False,
            is_constructor=False,
            receiver_kind=receiver_kind,
        )

        rels = []
        if body_node:
            rels.extend(_extract_calls(body_node, method_id, source_bytes))

        # CONTAINS from struct → method (same-file only)
        if class_id is not None:
            rels.append(Relationship(
                source_id=class_id,
                kind=RelationKind.CONTAINS,
                is_resolved=True,
                target_id=method_id,
                target_name=None,
            ))

        return sym, rels

    def _parse_receiver(
        self, receiver_list, source_bytes: bytes
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Parse receiver parameter_list.

        Returns (type_name, receiver_kind) where receiver_kind is "pointer" or
        "value".  Returns (None, None) if the receiver cannot be parsed.

        Handles plain, pointer, and generic receivers:
          (c Config)     → ("Config", "value")
          (c *Config)    → ("Config", "pointer")
          (c *Config[T]) → ("Config", "pointer")  — generic, strip type params
        """
        for child in receiver_list.children:
            if child.type == "parameter_declaration":
                type_node = child.child_by_field_name("type")
                if type_node is None:
                    continue
                if type_node.type == "pointer_type":
                    # _type_identifier_text handles pointer_type via direct child
                    # iteration (child_by_field_name("type") returns None in 0.25.0)
                    name = _type_identifier_text(type_node, source_bytes)
                    if name:
                        return name, "pointer"
                else:
                    name = _type_identifier_text(type_node, source_bytes)
                    if name:
                        return name, "value"
        return None, None

    # ------------------------------------------------------------------
    # Type declarations (struct, interface, type alias)
    # ------------------------------------------------------------------

    def _process_type_declaration(
        self,
        node,
        file_path: str,
        package_name: str,
        mb: MonikerBuilder,
        source_bytes: bytes,
        siblings: list,
    ) -> tuple[list, list[Relationship]]:
        symbols = []
        relationships = []

        # Doc comment is on the declaration node, shared across all specs in a group
        doc = _doc_comment(node, siblings, source_bytes)

        # type_declaration contains either type_spec (type definition) or
        # type_alias (type X = T, verified tree-sitter-go 0.25.0) children.
        for spec in (
            c for c in node.children
            if c.type in ("type_spec", "type_alias")
        ):
            name_node = spec.child_by_field_name("name")
            type_node = spec.child_by_field_name("type")
            if not name_node or not type_node:
                continue

            type_name = _node_text(name_node, source_bytes)
            visibility, is_exported = _go_visibility(type_name)

            if type_node.type == "struct_type":
                class_id = mb.for_class(file_path, type_name)
                embedded = self._extract_struct_embeddings(type_node, source_bytes)

                class_sym = ClassSymbol(
                    id=class_id,
                    name=type_name,
                    qualified_name=f"{package_name}.{type_name}",
                    file_path=file_path,
                    location=_node_loc(spec),
                    body_hash=_sha256(
                        source_bytes[spec.start_byte:spec.end_byte]
                    ),
                    language="go",
                    visibility=visibility,
                    is_exported=is_exported,
                    base_classes=embedded[:],
                    docstring=doc,
                    decorators=[],
                    is_abstract=False,
                )
                symbols.append(class_sym)

                for emb_name in embedded:
                    relationships.append(Relationship(
                        source_id=class_id,
                        kind=RelationKind.EXTENDS,
                        is_resolved=False,
                        target_id=None,
                        target_name=emb_name,
                        metadata={"embedding_kind": "struct"},
                    ))

            elif type_node.type == "interface_type":
                class_id = mb.for_class(file_path, type_name)
                method_syms, iface_rels = self._extract_interface_body(
                    type_node, class_id, type_name, file_path, package_name, mb,
                    source_bytes,
                )

                # Collect embedded interface names for base_classes
                embedded_ifaces = [
                    r.target_name
                    for r in iface_rels
                    if r.kind == RelationKind.EXTENDS and r.target_name
                ]

                class_sym = ClassSymbol(
                    id=class_id,
                    name=type_name,
                    qualified_name=f"{package_name}.{type_name}",
                    file_path=file_path,
                    location=_node_loc(spec),
                    body_hash=_sha256(
                        source_bytes[spec.start_byte:spec.end_byte]
                    ),
                    language="go",
                    visibility=visibility,
                    is_exported=is_exported,
                    base_classes=embedded_ifaces,
                    docstring=doc,
                    decorators=[],
                    is_abstract=True,
                )
                symbols.append(class_sym)
                symbols.extend(method_syms)
                relationships.extend(iface_rels)

            else:
                # Type alias or other named type → VariableSymbol
                var_id = mb.for_variable(file_path, type_name)
                symbols.append(VariableSymbol(
                    id=var_id,
                    name=type_name,
                    qualified_name=f"{package_name}.{type_name}",
                    file_path=file_path,
                    location=_node_loc(spec),
                    body_hash=_sha256(
                        source_bytes[spec.start_byte:spec.end_byte]
                    ),
                    language="go",
                    visibility=visibility,
                    is_exported=is_exported,
                    type_annotation=_node_text(type_node, source_bytes),
                    is_constant=True,  # type aliases are immutable bindings
                ))

        return symbols, relationships

    def _extract_struct_embeddings(
        self, struct_type_node, source_bytes: bytes
    ) -> list[str]:
        """
        Return type names of embedded fields in a struct.

        In tree-sitter-go 0.25.0, embedded fields are `field_declaration` nodes
        where `child_by_field_name("name")` is None.  The `type` field points
        directly to the type_identifier (even for pointer receivers — the `*`
        is a sibling token, not a pointer_type wrapper at the field level).
        """
        embedded = []
        for child in struct_type_node.children:
            if child.type == "field_declaration_list":
                for field in child.children:
                    if field.type == "field_declaration":
                        # Embedded field: no explicit name
                        if field.child_by_field_name("name") is None:
                            type_node = field.child_by_field_name("type")
                            if type_node and type_node.type == "type_identifier":
                                embedded.append(_node_text(type_node, source_bytes))
        return embedded

    def _extract_interface_body(
        self,
        iface_type_node,
        class_id: str,
        type_name: str,
        file_path: str,
        package_name: str,
        mb: MonikerBuilder,
        source_bytes: bytes,
    ) -> tuple[list[MethodSymbol], list[Relationship]]:
        """
        Extract method signatures and embedded interface references from an
        interface_type node.

        method_elem → MethodSymbol (complexity=None, no body)
        type_elem with type_identifier child → EXTENDS
        type_elem with binary/union type → type constraint, skip (not EXTENDS)
        """
        methods: list[MethodSymbol] = []
        rels: list[Relationship] = []

        for child in iface_type_node.children:
            if child.type == "method_elem":
                name_node = next(
                    (c for c in child.children if c.type == "field_identifier"),
                    None,
                )
                if not name_node:
                    continue

                mname = _node_text(name_node, source_bytes)
                visibility, is_exported = _go_visibility(mname)
                method_id = mb.for_method(file_path, type_name, mname)

                param_lists = [
                    c for c in child.children if c.type == "parameter_list"
                ]
                params = (
                    _extract_params(param_lists[0], source_bytes)
                    if param_lists else []
                )
                return_type = (
                    _node_text(param_lists[1], source_bytes)
                    if len(param_lists) >= 2
                    else None
                )
                # Single non-parameter_list return type (e.g. plain type_identifier)
                if return_type is None:
                    result = child.child_by_field_name("result")
                    if result:
                        return_type = _node_text(result, source_bytes)

                methods.append(MethodSymbol(
                    id=method_id,
                    name=mname,
                    qualified_name=f"{package_name}.{type_name}.{mname}",
                    file_path=file_path,
                    location=_node_loc(child),
                    body_hash=_sha256(
                        source_bytes[child.start_byte:child.end_byte]
                    ),
                    language="go",
                    visibility=visibility,
                    is_exported=is_exported,
                    signature=_node_text(child, source_bytes).strip(),
                    parameters=params,
                    return_type=return_type,
                    docstring=None,
                    decorators=[],
                    is_async=False,
                    complexity=None,  # no body
                    enclosing_class_id=class_id,
                    is_static=False,
                    is_constructor=False,
                ))
                rels.append(Relationship(
                    source_id=class_id,
                    kind=RelationKind.CONTAINS,
                    is_resolved=True,
                    target_id=method_id,
                    target_name=None,
                ))

            elif child.type == "type_elem":
                # Only emit EXTENDS for pure embedded interfaces: a single
                # type_identifier child, no other named siblings.
                # Type constraints (int | float64) have multiple named children
                # (both "int" and "float64" are type_identifiers) — skip those.
                named_children = [c for c in child.children if c.is_named]
                if (
                    len(named_children) == 1
                    and named_children[0].type == "type_identifier"
                ):
                    rels.append(Relationship(
                        source_id=class_id,
                        kind=RelationKind.EXTENDS,
                        is_resolved=False,
                        target_id=None,
                        target_name=_node_text(named_children[0], source_bytes),
                        metadata={"embedding_kind": "interface"},
                    ))

        return methods, rels

    # ------------------------------------------------------------------
    # Const and var declarations
    # ------------------------------------------------------------------

    def _process_const_var(
        self,
        node,
        file_path: str,
        package_name: str,
        mb: MonikerBuilder,
        source_bytes: bytes,
        siblings: list,
    ) -> list[VariableSymbol]:
        """
        Process const_declaration or var_declaration.

        Consts: always extracted; type_annotation=None for untyped consts.
        Vars: always extracted; type_annotation="<inferred>" for untyped vars.

        Multiple names in one spec (const X, Y = 1, 2) → one VariableSymbol
        per name, all sharing the same type annotation.
        """
        is_const = node.type == "const_declaration"
        spec_type = "const_spec" if is_const else "var_spec"
        syms: list[VariableSymbol] = []

        for spec in _collect_specs(node, spec_type):
            name_field = spec.child_by_field_name("name")
            if name_field is None:
                continue

            # name field may be a single identifier or an identifier_list
            if name_field.type == "identifier":
                names = [_node_text(name_field, source_bytes)]
            elif name_field.type == "identifier_list":
                names = [
                    _node_text(c, source_bytes)
                    for c in name_field.children
                    if c.type == "identifier"
                ]
            else:
                names = [_node_text(name_field, source_bytes)]

            type_node = spec.child_by_field_name("type")
            if type_node:
                type_text: Optional[str] = _node_text(type_node, source_bytes)
            elif is_const:
                type_text = None  # untyped const (iota, etc.)
            else:
                type_text = "<inferred>"  # untyped var — type inferred from RHS

            for var_name in names:
                visibility, is_exported = _go_visibility(var_name)
                var_id = mb.for_variable(file_path, var_name)
                syms.append(VariableSymbol(
                    id=var_id,
                    name=var_name,
                    qualified_name=f"{package_name}.{var_name}",
                    file_path=file_path,
                    location=_node_loc(spec),
                    body_hash=_sha256(
                        source_bytes[spec.start_byte:spec.end_byte]
                    ),
                    language="go",
                    visibility=visibility,
                    is_exported=is_exported,
                    type_annotation=type_text,
                    is_constant=is_const,
                ))

        return syms
