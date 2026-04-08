from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import tree_sitter_typescript as tsts
from tree_sitter import Language, Node, Tree

from ..base import (
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
    Symbol,
    VariableSymbol,
    Visibility,
)
from ..moniker import MonikerBuilder
from ..tree_sitter_runner import parse, run_query

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_QUERIES_DIR = Path(__file__).parent.parent.parent / "queries"
_TS_LANGUAGE = Language(tsts.language_typescript())
_TSX_LANGUAGE = Language(tsts.language_tsx())

# Cyclomatic complexity branch node types for TypeScript.
_BRANCH_NODES = frozenset({
    "if_statement",
    "else_clause",
    "for_statement",
    "for_in_statement",
    "while_statement",
    "do_statement",
    "catch_clause",
    "switch_case",          # each case arm
    "conditional_expression",  # ternary
    "binary_expression",    # && / || short-circuit (node type for logical ops)
})

# Top-level node types that declare a named symbol.
# export_statement wraps these — we unwrap transparently.
_CLASS_NODES = frozenset({
    "class_declaration",
    "abstract_class_declaration",
})
_INTERFACE_NODE = "interface_declaration"
_ENUM_NODE = "enum_declaration"
_TYPE_ALIAS_NODE = "type_alias_declaration"
_FUNCTION_NODE = "function_declaration"
_LEXICAL_NODE = "lexical_declaration"   # const / let

# Method-like node types inside class/interface bodies
_METHOD_NODES = frozenset({
    "method_definition",        # concrete method in a class
    "abstract_method_signature",  # abstract method in abstract class
    "method_signature",         # method in an interface (no body)
})


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _txt(node: Node) -> str:
    return node.text.decode("utf-8", errors="replace") if node.text else ""


def _slice(source: bytes, node: Node) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _location(node: Node) -> Location:
    return Location(
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        byte_start=node.start_byte,
        byte_end=node.end_byte,
        column_start=node.start_point[1],
        column_end=node.end_point[1],
    )


def _module_path(file_path: str) -> str:
    p = Path(file_path)
    return str(p.with_suffix("")).replace("\\", "/")


def _qualified_name(file_path: str) -> str:
    return _module_path(file_path).replace("/", ".")


def _strip_jsdoc(text: str) -> str:
    """
    Strip /** ... */ delimiters and leading * from each line.
    Returns clean docstring text.
    """
    text = text.strip()
    if text.startswith("/**"):
        text = text[3:]
    if text.endswith("*/"):
        text = text[:-2]
    lines = []
    for line in text.splitlines():
        stripped = line.strip().lstrip("*").strip()
        lines.append(stripped)
    return "\n".join(lines).strip()


def _jsdoc_before(node: Node, source: bytes) -> Optional[str]:
    """
    Return the JSDoc comment immediately before `node`, or None.
    'Immediately before' means the previous non-whitespace sibling is a
    comment node whose text starts with '/**'.
    """
    parent = node.parent
    if parent is None:
        return None
    prev = None
    for child in parent.children:
        if child == node:
            break
        if child.type not in ("comment", "{", "}", ";", "\n"):
            prev = None  # non-trivial non-comment node resets the window
        elif child.type == "comment":
            prev = child
    if prev is None:
        return None
    text = _txt(prev)
    if text.startswith("/**"):
        return _strip_jsdoc(text)
    return None


def _module_jsdoc(root: Node, source: bytes) -> Optional[str]:
    """
    Module-level JSDoc: first non-empty child of program that is a comment
    starting with '/**'.
    """
    for child in root.children:
        if child.type == "comment":
            text = _txt(child)
            if text.startswith("/**"):
                return _strip_jsdoc(text)
            return None  # non-JSDoc comment — not a module docstring
        if child.type not in (";", "\n"):
            break
    return None


# ---------------------------------------------------------------------------
# Export unwrapping
# ---------------------------------------------------------------------------

def _unwrap_export(node: Node) -> tuple[Node, bool]:
    """
    If `node` is an export_statement, return (inner_declaration, True).
    Otherwise return (node, False).

    Anonymous exports (`export default function() {}`, `export default class {}`)
    have no name and are skipped by callers — they return (None, True).
    Also handles re-export forms (export_clause, *, namespace_export) by
    returning (None, True) — callers that only want declarations ignore these.
    """
    if node.type != "export_statement":
        return node, False

    is_default = any(c.type == "default" for c in node.children)

    for child in node.children:
        if child.type in ("export", "default", ";", "from", "string",
                          "export_clause", "namespace_export"):
            continue
        if child.type == "*":
            continue
        # Found the inner declaration
        if child.type in (*_CLASS_NODES, _INTERFACE_NODE, _ENUM_NODE,
                          _TYPE_ALIAS_NODE, _FUNCTION_NODE, _LEXICAL_NODE):
            return child, True
        # Anonymous: function_expression or bare 'class' token
        if child.type in ("function_expression", "class"):
            return None, True   # anonymous — skip
        # Re-export with 'from' clause — no new symbol declared
        return None, True

    return None, True


# ---------------------------------------------------------------------------
# Visibility helpers
# ---------------------------------------------------------------------------

def _visibility_from_modifier(node: Node) -> Visibility:
    """
    Read the accessibility_modifier child of a method or field node.
    Falls back to PUBLIC if no modifier is present.
    """
    for child in node.children:
        if child.type == "accessibility_modifier":
            text = _txt(child)
            if text == "private":
                return Visibility.PRIVATE
            if text == "protected":
                return Visibility.PROTECTED
            return Visibility.PUBLIC
    return Visibility.PUBLIC


def _is_exported_name(name: str) -> bool:
    """Convention-based: private names start with _ (no keyword in TS for this)."""
    return not name.startswith("_")


# ---------------------------------------------------------------------------
# Signature extraction
# ---------------------------------------------------------------------------

def _function_signature(node: Node, source: bytes) -> str:
    """
    Extract signature up to (but not including) the statement_block body.
    Strips trailing whitespace. Works for function_declaration, method_definition,
    abstract_method_signature, method_signature, arrow_function.
    """
    body = node.child_by_field_name("body")
    if body is None:
        # No body (method_signature, abstract_method_signature) — full text is the signature
        return _slice(source, node).strip()
    sig = source[node.start_byte:body.start_byte].decode("utf-8", errors="replace")
    return sig.strip()


# ---------------------------------------------------------------------------
# Parameter extraction
# ---------------------------------------------------------------------------

def _extract_parameters(params_node: Node, source: bytes) -> list[Parameter]:
    if params_node is None:
        return []
    result = []
    for child in params_node.children:
        if child.type in ("(", ")", ","):
            continue
        p = _parse_param(child, source)
        if p is not None:
            result.append(p)
    return result


def _parse_param(node: Node, source: bytes) -> Optional[Parameter]:
    t = node.type

    if t == "required_parameter":
        name_child = node.children[0] if node.children else None
        type_node = node.child_by_field_name("type")
        is_variadic = name_child is not None and name_child.type == "rest_pattern"
        name = ""
        if name_child is not None:
            raw = _txt(name_child)
            name = raw.lstrip(".")  # strip leading '...' from rest_pattern text
        return Parameter(
            name=name,
            type_annotation=_slice(source, type_node).lstrip(":").strip() if type_node else None,
            is_variadic=is_variadic,
        )

    if t == "optional_parameter":
        name_child = None
        for c in node.children:
            if c.type == "identifier":
                name_child = c
                break
        type_node = node.child_by_field_name("type")
        return Parameter(
            name=_txt(name_child) if name_child else "",
            type_annotation=_slice(source, type_node).lstrip(":").strip() if type_node else None,
        )

    # assignment_pattern: param = default (no type annotation)
    if t == "assignment_pattern":
        left = node.child_by_field_name("left")
        return Parameter(
            name=_txt(left) if left else "",
            default_value=_slice(source, node.child_by_field_name("right"))
            if node.child_by_field_name("right") else None,
        )

    return None


# ---------------------------------------------------------------------------
# Return type extraction
# ---------------------------------------------------------------------------

def _extract_return_type(node: Node, source: bytes) -> Optional[str]:
    type_node = node.child_by_field_name("return_type")
    if type_node is None:
        # For method_definition, the return type_annotation is a direct child
        for child in node.children:
            if child.type == "type_annotation":
                return _slice(source, child).lstrip(":").strip()
        return None
    return _slice(source, type_node).lstrip(":").strip()


# ---------------------------------------------------------------------------
# Complexity
# ---------------------------------------------------------------------------

def _compute_complexity(body: Node) -> int:
    count = 0
    stack = list(body.children)
    while stack:
        n = stack.pop()
        if n.type in ("function_declaration", "arrow_function", "function_expression"):
            continue  # scope boundary
        if n.type in _BRANCH_NODES:
            # binary_expression only counts when it's a logical operator
            if n.type == "binary_expression":
                op = n.child_by_field_name("operator")
                if op and _txt(op) in ("&&", "||", "??"):
                    count += 1
            else:
                count += 1
        stack.extend(n.children)
    return 1 + count


# ---------------------------------------------------------------------------
# CALLS extraction
# ---------------------------------------------------------------------------

def _extract_calls(body: Node, source: bytes) -> list[tuple[str, dict, Location]]:
    results = []
    stack = list(body.children)
    while stack:
        n = stack.pop()
        if n.type in ("function_declaration", "arrow_function", "function_expression"):
            continue  # scope boundary
        if n.type == "call_expression":
            parsed = _parse_call(n, source)
            if parsed:
                target, meta = parsed
                results.append((target, meta, _location(n)))
        stack.extend(n.children)
    return results


def _parse_call(node: Node, source: bytes) -> Optional[tuple[str, dict]]:
    func = node.child_by_field_name("function")
    if func is None:
        return None
    if func.type == "identifier":
        return (_txt(func), {"call_form": "direct"})
    if func.type == "member_expression":
        obj = func.child_by_field_name("object")
        prop = func.child_by_field_name("property")
        if obj and prop:
            return (_txt(prop), {"call_form": "method", "receiver": _txt(obj)})
    return None


# ---------------------------------------------------------------------------
# Heritage parsing
# ---------------------------------------------------------------------------

def _class_base_classes(node: Node) -> list[str]:
    """Extract base class from extends_clause (class_heritage child)."""
    result = []
    for child in node.children:
        if child.type == "class_heritage":
            for hc in child.children:
                if hc.type == "extends_clause":
                    for ec in hc.children:
                        if ec.type in ("identifier", "type_identifier"):
                            result.append(_txt(ec))
    return result


def _class_implements(node: Node) -> list[str]:
    """Extract implemented interfaces from implements_clause."""
    result = []
    for child in node.children:
        if child.type == "class_heritage":
            for hc in child.children:
                if hc.type == "implements_clause":
                    for ic in hc.children:
                        if ic.type in ("type_identifier", "identifier",
                                       "generic_type"):
                            # For generic_type (IFoo<T>), use full text
                            result.append(_txt(ic))
    return result


def _interface_extends(node: Node) -> list[str]:
    """Extract base interfaces from extends_type_clause (interface heritage)."""
    result = []
    for child in node.children:
        if child.type == "extends_type_clause":
            for ec in child.children:
                if ec.type in ("type_identifier", "identifier", "generic_type"):
                    result.append(_txt(ec))
    return result


# ---------------------------------------------------------------------------
# Re-export relationship builders
# ---------------------------------------------------------------------------

def _build_reexport_rels(
    node: Node, module_id: str, source: bytes
) -> list[Relationship]:
    """
    Handle re-export forms that don't declare a new symbol:
      export { foo } from './bar'       → export_clause
      export * from './baz'             → bare *
      export * as ns from './qux'       → namespace_export
    Each produces an IMPORTS relationship from this module to the source module.
    """
    # Find the source module string
    source_str = None
    for child in node.children:
        if child.type == "string":
            raw = _txt(child)
            source_str = raw.strip("\"'")
            break
    if source_str is None:
        return []

    results = []

    has_export_clause = any(c.type == "export_clause" for c in node.children)
    has_star = any(c.type == "*" for c in node.children)
    has_ns_export = any(c.type == "namespace_export" for c in node.children)

    if has_export_clause:
        for child in node.children:
            if child.type == "export_clause":
                for spec in child.children:
                    if spec.type == "export_specifier":
                        # first identifier child is the name being re-exported
                        name_node = None
                        for sc in spec.children:
                            if sc.type == "identifier":
                                name_node = sc
                                break
                        if name_node:
                            results.append(Relationship(
                                source_id=module_id,
                                kind=RelationKind.IMPORTS,
                                is_resolved=False,
                                target_name=_txt(name_node),
                                location=_location(spec),
                                metadata={"import_source": source_str, "alias": None,
                                          "re_export": True},
                            ))
    elif has_ns_export:
        # export * as ns from '...'
        for child in node.children:
            if child.type == "namespace_export":
                alias_node = None
                for sc in child.children:
                    if sc.type == "identifier":
                        alias_node = sc
                if alias_node:
                    results.append(Relationship(
                        source_id=module_id,
                        kind=RelationKind.IMPORTS,
                        is_resolved=False,
                        target_name="*",
                        location=_location(node),
                        metadata={"import_source": source_str,
                                  "alias": _txt(alias_node), "re_export": True},
                    ))
    elif has_star:
        # export * from '...'
        results.append(Relationship(
            source_id=module_id,
            kind=RelationKind.IMPORTS,
            is_resolved=False,
            target_name="*",
            location=_location(node),
            metadata={"import_source": source_str, "alias": None, "re_export": True},
        ))

    return results


# ---------------------------------------------------------------------------
# Import relationship builders
# ---------------------------------------------------------------------------

def _build_import_rels(
    node: Node, module_id: str, source: bytes
) -> list[Relationship]:
    """
    import { Foo, Bar as B } from './foo'
    import path from 'path'
    import * as fs from 'fs'
    """
    source_str = None
    for child in node.children:
        if child.type == "string":
            raw = _txt(child)
            source_str = raw.strip("\"'")
            break
    if source_str is None:
        return []

    results = []
    clause = None
    for child in node.children:
        if child.type == "import_clause":
            clause = child
            break

    if clause is None:
        return []

    for child in clause.children:
        if child.type == "identifier":
            # default import: import path from 'x'
            results.append(Relationship(
                source_id=module_id,
                kind=RelationKind.IMPORTS,
                is_resolved=False,
                target_name=_txt(child),
                location=_location(child),
                metadata={"import_source": source_str, "alias": None,
                          "import_form": "default"},
            ))
        elif child.type == "named_imports":
            for spec in child.children:
                if spec.type == "import_specifier":
                    # May be 'Foo' or 'Foo as Bar'
                    names = [c for c in spec.children if c.type == "identifier"]
                    if names:
                        original = _txt(names[0])
                        alias = _txt(names[1]) if len(names) > 1 else None
                        results.append(Relationship(
                            source_id=module_id,
                            kind=RelationKind.IMPORTS,
                            is_resolved=False,
                            target_name=original,
                            location=_location(spec),
                            metadata={"import_source": source_str, "alias": alias,
                                      "import_form": "named"},
                        ))
        elif child.type == "namespace_import":
            # import * as fs from 'fs'
            alias_node = None
            for sc in child.children:
                if sc.type == "identifier":
                    alias_node = sc
            results.append(Relationship(
                source_id=module_id,
                kind=RelationKind.IMPORTS,
                is_resolved=False,
                target_name="*",
                location=_location(child),
                metadata={"import_source": source_str,
                          "alias": _txt(alias_node) if alias_node else None,
                          "import_form": "namespace"},
            ))

    return results


# ---------------------------------------------------------------------------
# Class body walker (shared by class, abstract class, interface)
# ---------------------------------------------------------------------------

def _walk_class_body(
    body_node: Node,
    source: bytes,
    file_path: str,
    builder: MonikerBuilder,
    mod_qname: str,
    cls_sym: ClassSymbol,
) -> tuple[list[MethodSymbol], list[Relationship]]:
    """
    Walk class_body or interface_body children.
    Accumulates decorators and JSDoc comments, flushes them to the next method node.
    Returns (method_symbols, relationships).
    """
    methods: list[MethodSymbol] = []
    rels: list[Relationship] = []
    pending_decorators: list[str] = []
    pending_jsdoc: Optional[str] = None

    for child in body_node.children:
        if child.type == "decorator":
            pending_decorators.append(_txt(child))
            continue
        if child.type == "comment":
            text = _txt(child)
            if text.startswith("/**"):
                pending_jsdoc = _strip_jsdoc(text)
            # Non-JSDoc comment: don't reset pending_jsdoc
            continue
        if child.type in _METHOD_NODES:
            method = _build_method(
                child, source, file_path, builder, mod_qname, cls_sym,
                pending_decorators, pending_jsdoc
            )
            pending_decorators = []
            pending_jsdoc = None
            if method is None:
                continue
            methods.append(method)
            rels.append(Relationship(
                source_id=cls_sym.id,
                kind=RelationKind.CONTAINS,
                is_resolved=True,
                target_id=method.id,
            ))
            body = child.child_by_field_name("body")
            if body:
                for target, meta, loc in _extract_calls(body, source):
                    rels.append(Relationship(
                        source_id=method.id,
                        kind=RelationKind.CALLS,
                        is_resolved=False,
                        target_name=target,
                        location=loc,
                        metadata=meta,
                    ))
        else:
            # Any non-method, non-decorator, non-comment node resets the window
            if child.type not in ("{", "}", ";"):
                pending_decorators = []
                pending_jsdoc = None

    return methods, rels


def _build_method(
    node: Node,
    source: bytes,
    file_path: str,
    builder: MonikerBuilder,
    mod_qname: str,
    cls_sym: ClassSymbol,
    decorators: list[str],
    docstring: Optional[str],
) -> Optional[MethodSymbol]:
    name_node = None
    for child in node.children:
        if child.type == "property_identifier":
            name_node = child
            break
    if name_node is None:
        return None
    name = _txt(name_node)

    visibility = _visibility_from_modifier(node)
    is_async = any(c.type == "async" for c in node.children)
    is_static = any(c.type == "static" for c in node.children)
    is_abstract = node.type == "abstract_method_signature"

    params_node = node.child_by_field_name("parameters")
    body = node.child_by_field_name("body")
    body_bytes = source[body.start_byte:body.end_byte] if body else b""

    return MethodSymbol(
        id=builder.for_method(file_path, cls_sym.name, name),
        name=name,
        qualified_name=f"{mod_qname}.{cls_sym.name}.{name}",
        file_path=file_path,
        location=_location(node),
        body_hash=_sha256(body_bytes),
        language="typescript",
        visibility=visibility,
        is_exported=cls_sym.is_exported,  # method visibility follows class export
        signature=_function_signature(node, source),
        parameters=_extract_parameters(params_node, source) if params_node else [],
        return_type=_extract_return_type(node, source),
        docstring=docstring,
        decorators=decorators,
        is_async=is_async,
        complexity=_compute_complexity(body) if body else None,
        enclosing_class_id=cls_sym.id,
        is_static=is_static,
        is_constructor=(name == "constructor"),
    )


# ---------------------------------------------------------------------------
# Top-level symbol builders
# ---------------------------------------------------------------------------

def _build_class_symbol(
    node: Node,
    source: bytes,
    file_path: str,
    builder: MonikerBuilder,
    mod_qname: str,
    is_exported: bool,
    docstring: Optional[str],
) -> Optional[ClassSymbol]:
    """Handles class_declaration and abstract_class_declaration."""
    name_node = None
    for child in node.children:
        if child.type == "type_identifier":
            name_node = child
            break
    if name_node is None:
        return None  # anonymous class

    name = _txt(name_node)
    is_abstract = node.type == "abstract_class_declaration"
    base_classes = _class_base_classes(node)
    body = node.child_by_field_name("body")
    body_bytes = source[body.start_byte:body.end_byte] if body else b""

    return ClassSymbol(
        id=builder.for_class(file_path, name),
        name=name,
        qualified_name=f"{mod_qname}.{name}",
        file_path=file_path,
        location=_location(node),
        body_hash=_sha256(body_bytes),
        language="typescript",
        visibility=Visibility.PUBLIC,
        is_exported=is_exported,
        base_classes=base_classes,
        docstring=docstring,
        decorators=[],
        is_abstract=is_abstract,
    )


def _build_interface_symbol(
    node: Node,
    source: bytes,
    file_path: str,
    builder: MonikerBuilder,
    mod_qname: str,
    is_exported: bool,
    docstring: Optional[str],
) -> Optional[ClassSymbol]:
    """Interface → ClassSymbol with is_abstract=True."""
    name_node = None
    for child in node.children:
        if child.type == "type_identifier":
            name_node = child
            break
    if name_node is None:
        return None

    name = _txt(name_node)
    base_ifaces = _interface_extends(node)
    body = None
    for child in node.children:
        if child.type == "interface_body":
            body = child
            break
    body_bytes = source[body.start_byte:body.end_byte] if body else b""

    return ClassSymbol(
        id=builder.for_class(file_path, name),
        name=name,
        qualified_name=f"{mod_qname}.{name}",
        file_path=file_path,
        location=_location(node),
        body_hash=_sha256(body_bytes),
        language="typescript",
        visibility=Visibility.PUBLIC,
        is_exported=is_exported,
        base_classes=base_ifaces,   # interface extends stored in base_classes
        docstring=docstring,
        decorators=[],
        is_abstract=True,   # interfaces are always abstract by definition
    )


def _build_enum_symbol(
    node: Node,
    source: bytes,
    file_path: str,
    builder: MonikerBuilder,
    mod_qname: str,
    is_exported: bool,
    docstring: Optional[str],
) -> Optional[ClassSymbol]:
    """
    Enum → ClassSymbol.
    Deliberate Phase 1 choice: no EnumSymbol discriminated type to avoid
    rippling schema changes. Flagged here so it's easy to revisit.
    """
    name_node = None
    for child in node.children:
        if child.type == "identifier":
            name_node = child
            break
    if name_node is None:
        return None

    name = _txt(name_node)
    body = None
    for child in node.children:
        if child.type == "enum_body":
            body = child
            break
    body_bytes = source[body.start_byte:body.end_byte] if body else b""

    return ClassSymbol(
        id=builder.for_class(file_path, name),
        name=name,
        qualified_name=f"{mod_qname}.{name}",
        file_path=file_path,
        location=_location(node),
        body_hash=_sha256(body_bytes),
        language="typescript",
        visibility=Visibility.PUBLIC,
        is_exported=is_exported,
        base_classes=[],
        docstring=docstring,
        decorators=[],
        is_abstract=False,
    )


def _build_type_alias_symbol(
    node: Node,
    source: bytes,
    file_path: str,
    builder: MonikerBuilder,
    mod_qname: str,
    is_exported: bool,
    docstring: Optional[str],
) -> Optional[VariableSymbol]:
    """type Alias = ... → VariableSymbol (is_constant=True)."""
    name_node = None
    for child in node.children:
        if child.type == "type_identifier":
            name_node = child
            break
    if name_node is None:
        return None

    name = _txt(name_node)
    # Capture the full RHS as type_annotation (no truncation — let embedder decide)
    type_annotation = _slice(source, node).strip()

    return VariableSymbol(
        id=builder.for_variable(file_path, name),
        name=name,
        qualified_name=f"{mod_qname}.{name}",
        file_path=file_path,
        location=_location(node),
        body_hash=_sha256(source[node.start_byte:node.end_byte]),
        language="typescript",
        visibility=Visibility.PUBLIC,
        is_exported=is_exported,
        type_annotation=type_annotation,
        is_constant=True,
    )


def _build_function_symbol(
    node: Node,
    source: bytes,
    file_path: str,
    builder: MonikerBuilder,
    mod_qname: str,
    is_exported: bool,
    docstring: Optional[str],
    name_override: Optional[str] = None,
) -> Optional[FunctionSymbol]:
    """Handles function_declaration and arrow_function (name_override for arrow)."""
    if name_override is not None:
        name = name_override
    else:
        name_node = None
        for child in node.children:
            if child.type == "identifier":
                name_node = child
                break
        if name_node is None:
            return None
        name = _txt(name_node)

    is_async = any(c.type == "async" for c in node.children)
    params_node = node.child_by_field_name("parameters")
    body = node.child_by_field_name("body")
    body_bytes = source[body.start_byte:body.end_byte] if body else b""

    return FunctionSymbol(
        id=builder.for_function(file_path, name),
        name=name,
        qualified_name=f"{mod_qname}.{name}",
        file_path=file_path,
        location=_location(node),
        body_hash=_sha256(body_bytes),
        language="typescript",
        visibility=Visibility.PUBLIC,
        is_exported=is_exported,
        signature=_function_signature(node, source),
        parameters=_extract_parameters(params_node, source) if params_node else [],
        return_type=_extract_return_type(node, source),
        docstring=docstring,
        decorators=[],
        is_async=is_async,
        complexity=_compute_complexity(body) if body else 1,
    )


def _build_variable_symbol(
    declarator: Node,
    source: bytes,
    file_path: str,
    builder: MonikerBuilder,
    mod_qname: str,
    is_exported: bool,
    is_const: bool,
    docstring: Optional[str],
) -> Optional[VariableSymbol]:
    """Build from a variable_declarator that has a type_annotation child."""
    name_node = None
    type_node = None
    for child in declarator.children:
        if child.type == "identifier" and name_node is None:
            name_node = child
        if child.type == "type_annotation":
            type_node = child
    if name_node is None or type_node is None:
        return None
    name = _txt(name_node)

    return VariableSymbol(
        id=builder.for_variable(file_path, name),
        name=name,
        qualified_name=f"{mod_qname}.{name}",
        file_path=file_path,
        location=_location(declarator),
        body_hash=_sha256(source[declarator.start_byte:declarator.end_byte]),
        language="typescript",
        visibility=Visibility.PUBLIC,
        is_exported=is_exported,
        type_annotation=_slice(source, type_node).lstrip(":").strip(),
        is_constant=is_const,
    )


# ---------------------------------------------------------------------------
# TypeScriptAdapter
# ---------------------------------------------------------------------------

class TypeScriptAdapter:
    """
    Extracts symbols and relationships from a TypeScript (.ts / .tsx) source
    file using tree-sitter.

    Usage:
        adapter_ts  = TypeScriptAdapter()          # .ts files
        adapter_tsx = TypeScriptAdapter(tsx=True)  # .tsx files
        result = adapter_ts.extract("src/foo.ts", source_bytes, "my-repo")

    Symbol mapping:
        function_declaration        → FunctionSymbol
        arrow_function in const x = → FunctionSymbol
        class_declaration           → ClassSymbol
        abstract_class_declaration  → ClassSymbol (is_abstract=True)
        interface_declaration       → ClassSymbol (is_abstract=True)
        enum_declaration            → ClassSymbol (deliberate Phase 1 choice)
        type_alias_declaration      → VariableSymbol (is_constant=True)
        lexical_declaration with type annotation  → VariableSymbol
        method_definition           → MethodSymbol
        abstract_method_signature   → MethodSymbol (complexity=None, no body)
        method_signature            → MethodSymbol (complexity=None, no body)

    Skipped in Phase 1:
        Anonymous export defaults (export default function() {})
        Namespace/module declarations (internal_module node)
        Untyped const/let variables (no type annotation)
    """

    def __init__(self, tsx: bool = False) -> None:
        self._language = _TSX_LANGUAGE if tsx else _TS_LANGUAGE
        self._query_scm = (_QUERIES_DIR / "typescript.scm").read_text(encoding="utf-8")

    def extract(
        self, file_path: str, source_bytes: bytes, repo_name: str
    ) -> FileExtraction:
        tree = parse(source_bytes, self._language)
        builder = MonikerBuilder(language="typescript", repo_name=repo_name)
        file_hash = _sha256(source_bytes)
        mod_path = _module_path(file_path)
        mod_qname = _qualified_name(file_path)

        file_record = File(
            path=file_path,
            language="typescript",
            size_bytes=len(source_bytes),
            hash=file_hash,
        )

        module_id = builder.for_module(file_path, mod_path)
        module_sym = ModuleSymbol(
            id=module_id,
            name=Path(file_path).stem,
            qualified_name=mod_qname,
            file_path=file_path,
            location=_location(tree.root_node),
            body_hash=file_hash,
            language="typescript",
            visibility=Visibility.PUBLIC,
            is_exported=True,
            docstring=_module_jsdoc(tree.root_node, source_bytes),
        )

        symbols: list[Symbol] = [module_sym]
        relationships: list[Relationship] = []

        self._walk_program(
            tree.root_node, source_bytes, file_path, builder,
            mod_qname, module_id, symbols, relationships
        )

        return FileExtraction(
            file=file_record,
            symbols=symbols,
            relationships=relationships,
        )

    # ------------------------------------------------------------------
    # Program walker
    # ------------------------------------------------------------------

    def _walk_program(
        self,
        root: Node,
        source: bytes,
        file_path: str,
        builder: MonikerBuilder,
        mod_qname: str,
        module_id: str,
        symbols: list[Symbol],
        relationships: list[Relationship],
    ) -> None:
        for child in root.children:
            # Imports
            if child.type == "import_statement":
                relationships.extend(_build_import_rels(child, module_id, source))
                continue

            # Re-exports (export_statement with a 'from' clause and no new declaration)
            if child.type == "export_statement":
                has_from = any(c.type == "from" for c in child.children)
                if has_from:
                    relationships.extend(_build_reexport_rels(child, module_id, source))
                    continue

            # Unwrap export_statement to get the inner declaration
            inner, is_exported = _unwrap_export(child)
            if inner is None:
                continue  # anonymous default export or re-export — skip

            # Namespaces (internal_module inside expression_statement)
            if child.type == "expression_statement":
                for ec in child.children:
                    if ec.type == "internal_module":
                        # Phase 1: skip namespace content, document the gap
                        pass
                continue

            docstring = _jsdoc_before(inner, source)

            # Classes (including abstract)
            if inner.type in _CLASS_NODES:
                cls = _build_class_symbol(
                    inner, source, file_path, builder, mod_qname, is_exported, docstring
                )
                if cls is None:
                    continue
                symbols.append(cls)
                relationships.append(Relationship(
                    source_id=module_id,
                    kind=RelationKind.CONTAINS,
                    is_resolved=True,
                    target_id=cls.id,
                ))
                for base in cls.base_classes:
                    relationships.append(Relationship(
                        source_id=cls.id,
                        kind=RelationKind.EXTENDS,
                        is_resolved=False,
                        target_name=base,
                        metadata={"import_source": None},
                    ))
                for iface in _class_implements(inner):
                    relationships.append(Relationship(
                        source_id=cls.id,
                        kind=RelationKind.IMPLEMENTS,
                        is_resolved=False,
                        target_name=iface,
                        metadata={"import_source": None},
                    ))
                # Walk class body for methods
                body_node = inner.child_by_field_name("body")
                if body_node:
                    methods, method_rels = _walk_class_body(
                        body_node, source, file_path, builder, mod_qname, cls
                    )
                    symbols.extend(methods)
                    relationships.extend(method_rels)

            # Interfaces
            elif inner.type == _INTERFACE_NODE:
                iface = _build_interface_symbol(
                    inner, source, file_path, builder, mod_qname, is_exported, docstring
                )
                if iface is None:
                    continue
                symbols.append(iface)
                relationships.append(Relationship(
                    source_id=module_id,
                    kind=RelationKind.CONTAINS,
                    is_resolved=True,
                    target_id=iface.id,
                ))
                for base in iface.base_classes:
                    relationships.append(Relationship(
                        source_id=iface.id,
                        kind=RelationKind.EXTENDS,
                        is_resolved=False,
                        target_name=base,
                        metadata={"import_source": None},
                    ))
                # Walk interface body for method signatures
                for ic in inner.children:
                    if ic.type == "interface_body":
                        methods, method_rels = _walk_class_body(
                            ic, source, file_path, builder, mod_qname, iface
                        )
                        symbols.extend(methods)
                        relationships.extend(method_rels)

            # Enums
            elif inner.type == _ENUM_NODE:
                enum_sym = _build_enum_symbol(
                    inner, source, file_path, builder, mod_qname, is_exported, docstring
                )
                if enum_sym is None:
                    continue
                symbols.append(enum_sym)
                relationships.append(Relationship(
                    source_id=module_id,
                    kind=RelationKind.CONTAINS,
                    is_resolved=True,
                    target_id=enum_sym.id,
                ))

            # Type aliases
            elif inner.type == _TYPE_ALIAS_NODE:
                alias_sym = _build_type_alias_symbol(
                    inner, source, file_path, builder, mod_qname, is_exported, docstring
                )
                if alias_sym is None:
                    continue
                symbols.append(alias_sym)
                relationships.append(Relationship(
                    source_id=module_id,
                    kind=RelationKind.CONTAINS,
                    is_resolved=True,
                    target_id=alias_sym.id,
                ))

            # Functions
            elif inner.type == _FUNCTION_NODE:
                fn = _build_function_symbol(
                    inner, source, file_path, builder, mod_qname, is_exported, docstring
                )
                if fn is None:
                    continue
                symbols.append(fn)
                relationships.append(Relationship(
                    source_id=module_id,
                    kind=RelationKind.CONTAINS,
                    is_resolved=True,
                    target_id=fn.id,
                ))
                body = inner.child_by_field_name("body")
                if body:
                    for target, meta, loc in _extract_calls(body, source):
                        relationships.append(Relationship(
                            source_id=fn.id,
                            kind=RelationKind.CALLS,
                            is_resolved=False,
                            target_name=target,
                            location=loc,
                            metadata=meta,
                        ))

            # const/let declarations
            elif inner.type == _LEXICAL_NODE:
                is_const = any(c.type == "const" for c in inner.children)
                for decl_child in inner.children:
                    if decl_child.type != "variable_declarator":
                        continue
                    # Check if value is an arrow function or function expression
                    value_node = None
                    for vc in decl_child.children:
                        if vc.type in ("arrow_function", "function_expression"):
                            value_node = vc
                            break
                    if value_node is not None:
                        # Only extract if the arrow/function has a return type
                        # annotation. Untyped arrows (const cb = (x) => x) carry
                        # no structural signal and are skipped — consistent with
                        # the untyped-variable rule.
                        has_return_type = any(
                            c.type == "type_annotation" for c in value_node.children
                        )
                        if not has_return_type:
                            continue
                        # Only handle plain identifier names — skip destructuring
                        name_node = None
                        for nc in decl_child.children:
                            if nc.type == "identifier":
                                name_node = nc
                                break
                        if name_node is None:
                            continue
                        fn = _build_function_symbol(
                            value_node, source, file_path, builder, mod_qname,
                            is_exported, docstring,
                            name_override=_txt(name_node),
                        )
                        if fn is None:
                            continue
                        symbols.append(fn)
                        relationships.append(Relationship(
                            source_id=module_id,
                            kind=RelationKind.CONTAINS,
                            is_resolved=True,
                            target_id=fn.id,
                        ))
                        body = value_node.child_by_field_name("body")
                        if body and body.type == "statement_block":
                            for target, meta, loc in _extract_calls(body, source):
                                relationships.append(Relationship(
                                    source_id=fn.id,
                                    kind=RelationKind.CALLS,
                                    is_resolved=False,
                                    target_name=target,
                                    location=loc,
                                    metadata=meta,
                                ))
                    else:
                        # Not a function — check for typed variable
                        var = _build_variable_symbol(
                            decl_child, source, file_path, builder, mod_qname,
                            is_exported, is_const, docstring
                        )
                        if var is None:
                            continue
                        symbols.append(var)
                        relationships.append(Relationship(
                            source_id=module_id,
                            kind=RelationKind.CONTAINS,
                            is_resolved=True,
                            target_id=var.id,
                        ))
