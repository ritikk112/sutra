from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import tree_sitter_python as tspython
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
_PY_LANGUAGE = Language(tspython.language())

# Cyclomatic complexity branch node types — verified against tree-sitter-python 0.25.0
_BRANCH_NODES = frozenset({
    "if_statement",
    "elif_clause",
    "for_statement",
    "while_statement",
    "except_clause",
    "with_statement",
    "boolean_operator",       # and / or — node type confirmed
    "conditional_expression", # ternary: x if cond else y
    "case_clause",            # match/case (Python 3.10+)
})


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _txt(node: Node) -> str:
    """Decode node bytes to str, replacing invalid UTF-8."""
    return node.text.decode("utf-8", errors="replace") if node.text else ""


def _slice(source: bytes, node: Node) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _location(node: Node) -> Location:
    # tree-sitter start_point/end_point are 0-indexed rows; we store 1-indexed lines
    return Location(
        line_start=node.start_point[0] + 1,
        line_end=node.end_point[0] + 1,
        byte_start=node.start_byte,
        byte_end=node.end_byte,
        column_start=node.start_point[1],
        column_end=node.end_point[1],
    )


def _visibility(name: str) -> Visibility:
    if name.startswith("__") and name.endswith("__"):
        return Visibility.PUBLIC   # dunder methods are public
    if name.startswith("_"):
        return Visibility.PRIVATE
    return Visibility.PUBLIC


def _is_exported(name: str) -> bool:
    is_dunder = name.startswith("__") and name.endswith("__")
    return not name.startswith("_") or is_dunder


def _module_path(file_path: str) -> str:
    """Strip file extension; keep directory separators as slashes.
    e.g. src/services/user.py → src/services/user"""
    p = Path(file_path)
    return str(p.with_suffix("")).replace("\\", "/")


def _qualified_name(file_path: str) -> str:
    """Python-style qualified module name.
    e.g. src/services/user.py → src.services.user"""
    return _module_path(file_path).replace("/", ".")


# ---------------------------------------------------------------------------
# Ancestor-walk helpers
# ---------------------------------------------------------------------------

def _find_enclosing_class(node: Node) -> Optional[Node]:
    """
    Walk up the ancestor chain from a function_definition node.

    Returns the nearest enclosing class_definition, or None if:
    - A function_definition is encountered first (nested function case)
    - The module root is reached with no class encountered
    """
    current = node.parent
    while current is not None:
        if current.type == "class_definition":
            return current
        if current.type == "function_definition":
            return None  # nested inside another function — not a method
        current = current.parent
    return None


def _find_enclosing_function(node: Node) -> Optional[Node]:
    """
    Walk up ancestors. Returns the nearest enclosing function_definition,
    or None if module root is reached first. Used to detect nested functions.
    """
    current = node.parent
    while current is not None:
        if current.type == "function_definition":
            return current
        if current.type == "module":
            return None
        current = current.parent
    return None


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_decorators(node: Node, source: bytes) -> list[str]:
    """
    In tree-sitter Python, a decorated function/class is wrapped in a
    decorated_definition node: [decorator..., function_definition].
    Decorators are children of decorated_definition, NOT siblings of
    the function_definition. Check node.parent.
    """
    parent = node.parent
    if parent is None or parent.type != "decorated_definition":
        return []
    return [
        _slice(source, c).strip()
        for c in parent.children
        if c.type == "decorator"
    ]


def _extract_signature(node: Node, source: bytes) -> str:
    """
    Raw source text from function start through parameter list and return
    type, but excluding the body block. Strips trailing colon.

    node.start_byte is at 'def' (or 'async' for async functions).
    """
    body = node.child_by_field_name("body")
    if body is None:
        return _slice(source, node).strip()
    sig = source[node.start_byte:body.start_byte].decode("utf-8", errors="replace")
    return sig.strip().rstrip(":").strip()


def _extract_parameters(node: Node, source: bytes) -> list[Parameter]:
    """
    Parse the parameters node of a function_definition into typed Parameter objects.

    Verified parameter child types in tree-sitter-python 0.25.0:
      identifier               — plain param (self, cls, or untyped)
      typed_parameter          — param: type
      default_parameter        — param=default
      typed_default_parameter  — param: type = default
      list_splat_pattern       — *args
      dictionary_splat_pattern — **kwargs
    """
    params_node = node.child_by_field_name("parameters")
    if params_node is None:
        return []
    result = []
    for child in params_node.children:
        if child.type in ("(", ")", ",", "*", "/"):
            continue
        p = _parse_single_param(child, source)
        if p is not None:
            result.append(p)
    return result


def _parse_single_param(node: Node, source: bytes) -> Optional[Parameter]:
    t = node.type

    if t == "identifier":
        return Parameter(name=_txt(node))

    if t == "typed_parameter":
        # Children: identifier, ":", type_node [and possibly more]
        name_node = None
        type_node = None
        for c in node.children:
            if c.type == "identifier" and name_node is None:
                name_node = c
            elif c.type not in (":", ","):
                type_node = c
        return Parameter(
            name=_txt(name_node) if name_node else "",
            type_annotation=_slice(source, type_node) if type_node else None,
        )

    if t == "default_parameter":
        # param=value — fields: name (identifier), value (expression)
        name_node = node.child_by_field_name("name")
        value_node = node.child_by_field_name("value")
        return Parameter(
            name=_txt(name_node) if name_node else "",
            default_value=_slice(source, value_node) if value_node else None,
        )

    if t == "typed_default_parameter":
        # param: type = value
        # Children: identifier, ":", type, "=", value
        name_node = None
        type_node = None
        value_node = None
        found_eq = False
        for c in node.children:
            if c.type == "identifier" and name_node is None:
                name_node = c
            elif c.type == "=":
                found_eq = True
            elif c.type == ":" :
                pass
            elif not found_eq and name_node is not None:
                type_node = c
            elif found_eq:
                value_node = c
        return Parameter(
            name=_txt(name_node) if name_node else "",
            type_annotation=_slice(source, type_node) if type_node else None,
            default_value=_slice(source, value_node) if value_node else None,
        )

    if t == "list_splat_pattern":
        # *args — strip the leading "*"
        raw = _txt(node).lstrip("*")
        return Parameter(name=raw, is_variadic=True)

    if t == "dictionary_splat_pattern":
        # **kwargs — strip the leading "**"
        raw = _txt(node).lstrip("*")
        return Parameter(name=raw, is_keyword_variadic=True)

    # Positional-only separator "/" or keyword-only "*" — skip
    return None


def _extract_return_type(node: Node, source: bytes) -> Optional[str]:
    ret = node.child_by_field_name("return_type")
    if ret is None:
        return None
    # The return type node text includes the "->"; strip it
    return _slice(source, ret).lstrip("->").strip()


def _extract_docstring(body_node: Node, source: bytes) -> Optional[str]:
    """
    First statement in a block that is an expression_statement containing
    only a string node. Must be the very first non-trivial statement.
    """
    for child in body_node.children:
        if child.type in ("newline", "comment", "indent", "dedent"):
            continue
        if child.type == "expression_statement":
            string_children = [c for c in child.children if c.type == "string"]
            non_trivial = [c for c in child.children if c.type not in ("newline",)]
            if len(non_trivial) == 1 and string_children:
                return _strip_quotes(_slice(source, string_children[0]))
        break  # first real statement is not a docstring
    return None


def _extract_module_docstring(root: Node, source: bytes) -> Optional[str]:
    """
    Module docstring: the very first expression_statement whose only child
    is a string node. A line like BANNER = "..." has an assignment child
    (not a bare string), so it is correctly excluded.
    """
    for child in root.children:
        if child.type in ("comment", "newline"):
            continue
        if child.type == "expression_statement":
            string_children = [c for c in child.children if c.type == "string"]
            non_trivial = [c for c in child.children if c.type != "newline"]
            if len(non_trivial) == 1 and string_children:
                return _strip_quotes(_slice(source, string_children[0]))
        break
    return None


def _strip_quotes(s: str) -> str:
    for q in ('"""', "'''", '"', "'"):
        if s.startswith(q) and s.endswith(q) and len(s) >= 2 * len(q):
            return s[len(q):-len(q)]
    return s


def _compute_complexity(body: Node) -> int:
    """
    Cyclomatic complexity: 1 + count of branch nodes in the function body.
    Does NOT descend into nested function_definitions (scope boundary).
    """
    count = 0
    stack = list(body.children)
    while stack:
        n = stack.pop()
        if n.type == "function_definition":
            continue  # scope boundary — inner functions counted separately
        if n.type in _BRANCH_NODES:
            count += 1
        stack.extend(n.children)
    return 1 + count


def _extract_calls(
    body: Node, source: bytes
) -> list[tuple[str, dict, Location]]:
    """
    Walk the function body and collect all call expressions.
    Does NOT descend into nested function_definitions.

    Returns list of (target_name, metadata, call_site_location).
    metadata keys: call_form ("direct" | "method"), receiver (method calls only).
    """
    results: list[tuple[str, dict, Location]] = []
    stack = list(body.children)
    while stack:
        n = stack.pop()
        if n.type == "function_definition":
            continue  # scope boundary
        if n.type == "call":
            parsed = _parse_call_node(n, source)
            if parsed:
                target, meta = parsed
                results.append((target, meta, _location(n)))
        stack.extend(n.children)
    return results


def _parse_call_node(
    node: Node, source: bytes
) -> Optional[tuple[str, dict]]:
    func = node.child_by_field_name("function")
    if func is None:
        return None
    if func.type == "identifier":
        return (_txt(func), {"call_form": "direct"})
    if func.type == "attribute":
        obj = func.child_by_field_name("object")
        attr = func.child_by_field_name("attribute")
        if obj and attr:
            return (
                _txt(attr),
                {"call_form": "method", "receiver": _txt(obj)},
            )
    return None


def _extract_base_classes(class_node: Node) -> list[str]:
    """
    Parse base classes from a class_definition's argument_list.
    Handles plain identifiers (Base) and dotted names (module.Base).
    """
    result = []
    for child in class_node.children:
        if child.type == "argument_list":
            for arg in child.children:
                if arg.type in (",", "(", ")"):
                    continue
                if arg.type in ("identifier", "attribute"):
                    result.append(_txt(arg))
    return result


def _is_abstract(base_classes: list[str], decorators: list[str]) -> bool:
    return (
        any("ABC" in b or "ABCMeta" in b for b in base_classes)
        or any("ABCMeta" in d for d in decorators)
    )


# ---------------------------------------------------------------------------
# PythonAdapter
# ---------------------------------------------------------------------------

class PythonAdapter:
    """
    Extracts symbols and relationships from a Python source file using
    tree-sitter and the python.scm query file.

    Usage:
        adapter = PythonAdapter()
        result = adapter.extract("src/foo.py", source_bytes, "my-repo")
    """

    def __init__(self) -> None:
        self._language = _PY_LANGUAGE
        self._query_scm = (_QUERIES_DIR / "python.scm").read_text(encoding="utf-8")

    def extract(
        self, file_path: str, source_bytes: bytes, repo_name: str
    ) -> FileExtraction:
        """
        Parse source_bytes and produce a FileExtraction.

        On syntax errors: tree-sitter always produces a tree with ERROR nodes.
        Extraction continues on whatever is parseable — partial results are
        preferable to a hard failure. ERRORs produce fewer captured symbols.
        """
        tree = parse(source_bytes, self._language)
        captures = run_query(tree.root_node, self._language, self._query_scm)

        builder = MonikerBuilder(language="python", repo_name=repo_name)
        file_hash = _sha256(source_bytes)
        mod_path = _module_path(file_path)
        mod_qname = _qualified_name(file_path)

        file_record = File(
            path=file_path,
            language="python",
            size_bytes=len(source_bytes),
            hash=file_hash,
        )

        # ModuleSymbol — one per file
        module_id = builder.for_module(file_path, mod_path)
        module_sym = ModuleSymbol(
            id=module_id,
            name=Path(file_path).stem,
            qualified_name=mod_qname,
            file_path=file_path,
            location=_location(tree.root_node),
            body_hash=file_hash,  # hash of entire file for fast change detection
            language="python",
            visibility=Visibility.PUBLIC,
            is_exported=True,
            docstring=_extract_module_docstring(tree.root_node, source_bytes),
        )

        symbols: list[Symbol] = [module_sym]
        relationships: list[Relationship] = []

        # Process classes first so class_node.id → ClassSymbol is populated
        # before we process methods.
        class_node_to_sym: dict[int, ClassSymbol] = {}
        for class_node in captures.get("class.def", []):
            cls = self._build_class(class_node, file_path, source_bytes, builder, mod_qname)
            if cls is None:
                continue
            symbols.append(cls)
            class_node_to_sym[class_node.id] = cls

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
                    metadata={
                        "import_source": None,
                        "call_form": "attribute" if "." in base else "direct",
                    },
                ))

        # Process all function_definition nodes — distinguish methods, functions, nested
        for func_node in captures.get("func.def", []):
            enclosing_class = _find_enclosing_class(func_node)

            if enclosing_class is not None:
                # It's a method
                cls_sym = class_node_to_sym.get(enclosing_class.id)
                if cls_sym is None:
                    continue  # method inside a nested/untracked class

                method = self._build_method(
                    func_node, file_path, source_bytes, builder, mod_qname, cls_sym
                )
                if method is None:
                    continue
                symbols.append(method)

                relationships.append(Relationship(
                    source_id=cls_sym.id,
                    kind=RelationKind.CONTAINS,
                    is_resolved=True,
                    target_id=method.id,
                ))
                body = func_node.child_by_field_name("body")
                if body:
                    for target, meta, loc in _extract_calls(body, source_bytes):
                        relationships.append(Relationship(
                            source_id=method.id,
                            kind=RelationKind.CALLS,
                            is_resolved=False,
                            target_name=target,
                            location=loc,
                            metadata=meta,
                        ))
            else:
                # Not inside a class — but may be nested inside another function
                if _find_enclosing_function(func_node) is not None:
                    continue  # nested function — skip (Phase 1)

                fn = self._build_function(
                    func_node, file_path, source_bytes, builder, mod_qname
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
                body = func_node.child_by_field_name("body")
                if body:
                    for target, meta, loc in _extract_calls(body, source_bytes):
                        relationships.append(Relationship(
                            source_id=fn.id,
                            kind=RelationKind.CALLS,
                            is_resolved=False,
                            target_name=target,
                            location=loc,
                            metadata=meta,
                        ))

        # Annotated module-level variables
        # In tree-sitter-python 0.25.0, these are expression_statement → assignment
        # where assignment has a "type" field (annotation). NOT an annotated_assignment
        # node — that node type does not exist in this grammar version.
        for child in tree.root_node.children:
            if child.type == "expression_statement":
                for sub in child.children:
                    if sub.type == "assignment":
                        type_node = sub.child_by_field_name("type")
                        if type_node:
                            var = self._build_variable(
                                sub, file_path, source_bytes, builder, mod_qname
                            )
                            if var:
                                symbols.append(var)
                                relationships.append(Relationship(
                                    source_id=module_id,
                                    kind=RelationKind.CONTAINS,
                                    is_resolved=True,
                                    target_id=var.id,
                                ))

        # Import relationships
        for imp in captures.get("import.from", []):
            relationships.extend(
                self._build_from_import_rels(imp, module_id, source_bytes)
            )
        for imp in captures.get("import.plain", []):
            relationships.extend(
                self._build_plain_import_rels(imp, module_id, source_bytes)
            )

        return FileExtraction(
            file=file_record,
            symbols=symbols,
            relationships=relationships,
        )

    # ------------------------------------------------------------------
    # Symbol builders
    # ------------------------------------------------------------------

    def _build_class(
        self,
        node: Node,
        file_path: str,
        source: bytes,
        builder: MonikerBuilder,
        mod_qname: str,
    ) -> Optional[ClassSymbol]:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return None
        name = _txt(name_node)
        decorators = _extract_decorators(node, source)
        base_classes = _extract_base_classes(node)
        body = node.child_by_field_name("body")
        body_bytes = source[body.start_byte:body.end_byte] if body else b""

        return ClassSymbol(
            id=builder.for_class(file_path, name),
            name=name,
            qualified_name=f"{mod_qname}.{name}",
            file_path=file_path,
            location=_location(node),
            body_hash=_sha256(body_bytes),
            language="python",
            visibility=_visibility(name),
            is_exported=_is_exported(name),
            base_classes=base_classes,
            docstring=_extract_docstring(body, source) if body else None,
            decorators=decorators,
            is_abstract=_is_abstract(base_classes, decorators),
        )

    def _build_function(
        self,
        node: Node,
        file_path: str,
        source: bytes,
        builder: MonikerBuilder,
        mod_qname: str,
    ) -> Optional[FunctionSymbol]:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return None
        name = _txt(name_node)
        body = node.child_by_field_name("body")
        body_bytes = source[body.start_byte:body.end_byte] if body else b""
        sig = _extract_signature(node, source)
        is_async = bool(node.children) and node.children[0].type == "async"

        return FunctionSymbol(
            id=builder.for_function(file_path, name),
            name=name,
            qualified_name=f"{mod_qname}.{name}",
            file_path=file_path,
            location=_location(node),
            body_hash=_sha256(body_bytes),
            language="python",
            visibility=_visibility(name),
            is_exported=_is_exported(name),
            signature=sig,
            parameters=_extract_parameters(node, source),
            return_type=_extract_return_type(node, source),
            docstring=_extract_docstring(body, source) if body else None,
            decorators=_extract_decorators(node, source),
            is_async=is_async,
            complexity=_compute_complexity(body) if body else 1,
        )

    def _build_method(
        self,
        node: Node,
        file_path: str,
        source: bytes,
        builder: MonikerBuilder,
        mod_qname: str,
        cls_sym: ClassSymbol,
    ) -> Optional[MethodSymbol]:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return None
        name = _txt(name_node)
        decorators = _extract_decorators(node, source)
        body = node.child_by_field_name("body")
        body_bytes = source[body.start_byte:body.end_byte] if body else b""
        sig = _extract_signature(node, source)
        is_async = bool(node.children) and node.children[0].type == "async"
        is_static = any("staticmethod" in d for d in decorators)
        is_classmethod = any("classmethod" in d for d in decorators)

        return MethodSymbol(
            id=builder.for_method(file_path, cls_sym.name, name),
            name=name,
            qualified_name=f"{mod_qname}.{cls_sym.name}.{name}",
            file_path=file_path,
            location=_location(node),
            body_hash=_sha256(body_bytes),
            language="python",
            visibility=_visibility(name),
            is_exported=_is_exported(name),
            signature=sig,
            parameters=_extract_parameters(node, source),
            return_type=_extract_return_type(node, source),
            docstring=_extract_docstring(body, source) if body else None,
            decorators=decorators,
            is_async=is_async,
            complexity=_compute_complexity(body) if body else 1,
            enclosing_class_id=cls_sym.id,
            is_static=is_static,
            is_constructor=name == "__init__",
        )

    def _build_variable(
        self,
        assignment_node: Node,
        file_path: str,
        source: bytes,
        builder: MonikerBuilder,
        mod_qname: str,
    ) -> Optional[VariableSymbol]:
        """
        Build a VariableSymbol from a module-level annotated assignment node.
        assignment_node is the 'assignment' node (child of expression_statement).
        Fields: left (name), type (annotation), right (value, optional).
        """
        name_node = assignment_node.child_by_field_name("left")
        type_node = assignment_node.child_by_field_name("type")
        if name_node is None or type_node is None:
            return None
        name = _txt(name_node)
        if not name.isidentifier():
            return None

        return VariableSymbol(
            id=builder.for_variable(file_path, name),
            name=name,
            qualified_name=f"{mod_qname}.{name}",
            file_path=file_path,
            location=_location(assignment_node),
            body_hash=_sha256(source[assignment_node.start_byte:assignment_node.end_byte]),
            language="python",
            visibility=_visibility(name),
            is_exported=_is_exported(name),
            type_annotation=_slice(source, type_node),
            is_constant=name.isupper(),
        )

    # ------------------------------------------------------------------
    # Relationship builders
    # ------------------------------------------------------------------

    def _build_from_import_rels(
        self, node: Node, source_id: str, source: bytes
    ) -> list[Relationship]:
        """
        from x.y.z import a, b as c  →  one IMPORTS relationship per name.
        target_name = original name (not alias).
        metadata["alias"] = alias if renamed, else None.
        metadata["import_source"] = the module path ("x.y.z").
        """
        mod_node = node.child_by_field_name("module_name")
        import_source = _txt(mod_node) if mod_node else None
        result = []

        for child in node.children:
            if child.type == "dotted_name" and child != mod_node:
                # plain: from x import foo
                result.append(Relationship(
                    source_id=source_id,
                    kind=RelationKind.IMPORTS,
                    is_resolved=False,
                    target_name=_txt(child),
                    location=_location(child),
                    metadata={"import_source": import_source, "alias": None},
                ))
            elif child.type == "aliased_import":
                # from x import foo as bar
                name_node = child.child_by_field_name("name")
                alias_node = child.child_by_field_name("alias")
                if name_node:
                    result.append(Relationship(
                        source_id=source_id,
                        kind=RelationKind.IMPORTS,
                        is_resolved=False,
                        target_name=_txt(name_node),
                        location=_location(child),
                        metadata={
                            "import_source": import_source,
                            "alias": _txt(alias_node) if alias_node else None,
                        },
                    ))
            elif child.type == "wildcard_import":
                # from x import *
                result.append(Relationship(
                    source_id=source_id,
                    kind=RelationKind.IMPORTS,
                    is_resolved=False,
                    target_name="*",
                    location=_location(child),
                    metadata={"import_source": import_source, "alias": None},
                ))
        return result

    def _build_plain_import_rels(
        self, node: Node, source_id: str, source: bytes
    ) -> list[Relationship]:
        """
        import os, import numpy as np  →  one IMPORTS relationship per module.
        """
        result = []
        for child in node.children:
            if child.type == "dotted_name":
                result.append(Relationship(
                    source_id=source_id,
                    kind=RelationKind.IMPORTS,
                    is_resolved=False,
                    target_name=_txt(child),
                    location=_location(child),
                    metadata={"import_source": None, "alias": None, "call_form": "module"},
                ))
            elif child.type == "aliased_import":
                name_node = child.child_by_field_name("name")
                alias_node = child.child_by_field_name("alias")
                if name_node:
                    result.append(Relationship(
                        source_id=source_id,
                        kind=RelationKind.IMPORTS,
                        is_resolved=False,
                        target_name=_txt(name_node),
                        location=_location(child),
                        metadata={
                            "import_source": None,
                            "alias": _txt(alias_node) if alias_node else None,
                            "call_form": "module",
                        },
                    ))
        return result
