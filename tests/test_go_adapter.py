"""
Priority 8 — Go adapter test suite.

All tests use real tree-sitter parsing on real Go source snippets.
No mocks.  Every edge case from the P8 feedback is covered explicitly.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional

import pytest

from sutra.core.extractor.adapters.go import GoAdapter
from sutra.core.extractor.base import (
    ClassSymbol,
    FileExtraction,
    FunctionSymbol,
    MethodSymbol,
    ModuleSymbol,
    RelationKind,
    VariableSymbol,
    Visibility,
)
from sutra.core.extractor.moniker import is_valid_moniker

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

REPO = "myrepo"
FILE = "pkg/server/server.go"


def extract(src: str, file: str = FILE, repo: str = REPO) -> FileExtraction:
    adapter = GoAdapter()
    return adapter.extract(file, src.encode(), repo)


def sym_by_name(result: FileExtraction, name: str):
    return next((s for s in result.symbols if s.name == name), None)


def syms_by_kind(result: FileExtraction, kind):
    return [s for s in result.symbols if isinstance(s, kind)]


def rels_by_kind(result: FileExtraction, kind: RelationKind):
    return [r for r in result.relationships if r.kind == kind]


# ---------------------------------------------------------------------------
# Module symbol
# ---------------------------------------------------------------------------

class TestModuleSymbol:
    def test_module_symbol_always_present(self):
        result = extract("package main\n")
        mods = syms_by_kind(result, ModuleSymbol)
        assert len(mods) == 1

    def test_module_qualified_name_is_package_name(self):
        result = extract("package server\n")
        mod = syms_by_kind(result, ModuleSymbol)[0]
        assert mod.qualified_name == "server"

    def test_module_name_is_file_stem(self):
        result = extract("package server\n")
        mod = syms_by_kind(result, ModuleSymbol)[0]
        assert mod.name == "server"

    def test_module_moniker_ends_with_slash(self):
        result = extract("package server\n")
        mod = syms_by_kind(result, ModuleSymbol)[0]
        assert mod.id.endswith("/")
        assert is_valid_moniker(mod.id)

    def test_module_body_hash_is_file_hash(self):
        src = "package main\n"
        result = extract(src)
        mod = syms_by_kind(result, ModuleSymbol)[0]
        import hashlib
        expected = "sha256:" + hashlib.sha256(src.encode()).hexdigest()
        assert mod.body_hash == expected


# ---------------------------------------------------------------------------
# Function declarations
# ---------------------------------------------------------------------------

class TestFunctionDeclaration:
    def test_exported_function(self):
        src = "package main\nfunc Serve(addr string) error { return nil }\n"
        result = extract(src)
        sym = sym_by_name(result, "Serve")
        assert sym is not None
        assert isinstance(sym, FunctionSymbol)
        assert sym.visibility == Visibility.PUBLIC
        assert sym.is_exported is True

    def test_unexported_function(self):
        src = "package main\nfunc parseAddr(s string) string { return s }\n"
        result = extract(src)
        sym = sym_by_name(result, "parseAddr")
        assert sym is not None
        assert sym.visibility == Visibility.INTERNAL
        assert sym.is_exported is False

    def test_function_parameters(self):
        src = "package main\nfunc Add(a int, b int) int { return a + b }\n"
        result = extract(src)
        sym = sym_by_name(result, "Add")
        params = {p.name: p.type_annotation for p in sym.parameters}
        assert params.get("a") == "int"
        assert params.get("b") == "int"

    def test_variadic_parameter(self):
        src = "package main\nfunc Sum(ns ...int) int { return 0 }\n"
        result = extract(src)
        sym = sym_by_name(result, "Sum")
        assert len(sym.parameters) == 1
        assert sym.parameters[0].is_variadic is True
        assert sym.parameters[0].name == "ns"

    def test_return_type_single(self):
        src = "package main\nfunc GetName() string { return \"x\" }\n"
        result = extract(src)
        sym = sym_by_name(result, "GetName")
        assert sym.return_type == "string"

    def test_return_type_multiple(self):
        src = "package main\nfunc Open(path string) (*File, error) { return nil, nil }\n"
        result = extract(src)
        sym = sym_by_name(result, "Open")
        assert sym.return_type is not None
        assert "error" in sym.return_type

    def test_no_return_type(self):
        src = "package main\nfunc Log(msg string) { }\n"
        result = extract(src)
        sym = sym_by_name(result, "Log")
        assert sym.return_type is None

    def test_signature_text(self):
        src = "package main\nfunc Add(a int, b int) int { return a + b }\n"
        result = extract(src)
        sym = sym_by_name(result, "Add")
        assert "func Add" in sym.signature
        assert "int" in sym.signature
        # body brace should not be part of signature
        assert "return" not in sym.signature

    def test_function_moniker_valid(self):
        src = "package main\nfunc Serve() {}\n"
        result = extract(src)
        sym = sym_by_name(result, "Serve")
        assert is_valid_moniker(sym.id)
        assert sym.id.endswith("().")

    def test_function_qualified_name(self):
        src = "package server\nfunc Serve() {}\n"
        result = extract(src, file="pkg/server/server.go")
        sym = sym_by_name(result, "Serve")
        assert sym.qualified_name == "server.Serve"

    def test_module_contains_function(self):
        src = "package main\nfunc Serve() {}\n"
        result = extract(src)
        contains = rels_by_kind(result, RelationKind.CONTAINS)
        mod = syms_by_kind(result, ModuleSymbol)[0]
        func_sym = sym_by_name(result, "Serve")
        assert any(
            r.source_id == mod.id and r.target_id == func_sym.id
            for r in contains
        )


# ---------------------------------------------------------------------------
# init() disambiguation
# ---------------------------------------------------------------------------

class TestInitDisambiguation:
    def test_single_init_has_line_number_in_moniker(self):
        src = "package main\nfunc init() { }\n"
        result = extract(src)
        funcs = [s for s in result.symbols if s.name == "init"]
        assert len(funcs) == 1
        assert "init@" in funcs[0].id

    def test_two_init_functions_get_unique_monikers(self):
        src = (
            "package main\n"
            "func init() { }\n"
            "func init() { }\n"
        )
        result = extract(src)
        init_syms = [s for s in result.symbols if s.name == "init"]
        assert len(init_syms) == 2
        assert init_syms[0].id != init_syms[1].id
        for sym in init_syms:
            assert is_valid_moniker(sym.id)
            assert "init@" in sym.id

    def test_two_init_no_duplicate_monikers(self):
        src = (
            "package main\n"
            "func init() { }\n"
            "\n"
            "func init() { }\n"
        )
        result = extract(src)
        all_ids = [s.id for s in result.symbols]
        assert len(all_ids) == len(set(all_ids)), "Duplicate monikers found"


# ---------------------------------------------------------------------------
# Method declarations
# ---------------------------------------------------------------------------

class TestMethodDeclaration:
    def test_value_receiver(self):
        src = "package main\ntype Config struct{}\nfunc (c Config) Host() string { return \"\" }\n"
        result = extract(src)
        sym = sym_by_name(result, "Host")
        assert isinstance(sym, MethodSymbol)
        assert sym.receiver_kind == "value"

    def test_pointer_receiver(self):
        src = "package main\ntype Config struct{}\nfunc (c *Config) SetHost(h string) { }\n"
        result = extract(src)
        sym = sym_by_name(result, "SetHost")
        assert isinstance(sym, MethodSymbol)
        assert sym.receiver_kind == "pointer"

    def test_method_moniker_contains_class_name(self):
        src = "package main\ntype Config struct{}\nfunc (c *Config) SetHost(h string) { }\n"
        result = extract(src)
        sym = sym_by_name(result, "SetHost")
        assert "Config#SetHost" in sym.id
        assert is_valid_moniker(sym.id)

    def test_method_qualified_name(self):
        src = "package server\ntype Config struct{}\nfunc (c *Config) SetHost(h string) { }\n"
        result = extract(src, file="pkg/server/server.go")
        sym = sym_by_name(result, "SetHost")
        assert sym.qualified_name == "server.Config.SetHost"

    def test_same_file_method_has_enclosing_class_id(self):
        src = "package main\ntype Config struct{}\nfunc (c *Config) SetHost(h string) { }\n"
        result = extract(src)
        method = sym_by_name(result, "SetHost")
        cls = sym_by_name(result, "Config")
        assert method.enclosing_class_id == cls.id

    def test_cross_file_method_has_none_enclosing_class_id(self):
        """Receiver type defined in another file → enclosing_class_id=None."""
        src = "package main\nfunc (c *Config) SetHost(h string) { }\n"
        result = extract(src)
        method = sym_by_name(result, "SetHost")
        assert isinstance(method, MethodSymbol)
        assert method.enclosing_class_id is None

    def test_cross_file_method_no_contains_relationship(self):
        """No CONTAINS emitted for cross-file methods — indexer post-pass handles it."""
        src = "package main\nfunc (c *Config) SetHost(h string) { }\n"
        result = extract(src)
        method = sym_by_name(result, "SetHost")
        contains = rels_by_kind(result, RelationKind.CONTAINS)
        assert not any(r.target_id == method.id for r in contains)

    def test_pointer_and_value_receiver_on_same_struct(self):
        """Both pointer and value receiver methods → distinct monikers, distinct receiver_kind."""
        src = (
            "package main\n"
            "type T struct{}\n"
            "func (t T) Get() string { return \"\" }\n"
            "func (t *T) Set(v string) { }\n"
        )
        result = extract(src)
        get_sym = sym_by_name(result, "Get")
        set_sym = sym_by_name(result, "Set")
        assert get_sym.receiver_kind == "value"
        assert set_sym.receiver_kind == "pointer"
        assert get_sym.id != set_sym.id

    def test_same_file_contains_relationship(self):
        src = "package main\ntype Config struct{}\nfunc (c *Config) SetHost(h string) { }\n"
        result = extract(src)
        method = sym_by_name(result, "SetHost")
        cls = sym_by_name(result, "Config")
        contains = rels_by_kind(result, RelationKind.CONTAINS)
        assert any(
            r.source_id == cls.id and r.target_id == method.id
            for r in contains
        )

    def test_method_not_in_module_contains(self):
        """Methods should NOT appear as direct children of the module."""
        src = "package main\ntype Config struct{}\nfunc (c *Config) SetHost(h string) { }\n"
        result = extract(src)
        mod = syms_by_kind(result, ModuleSymbol)[0]
        method = sym_by_name(result, "SetHost")
        contains = rels_by_kind(result, RelationKind.CONTAINS)
        assert not any(
            r.source_id == mod.id and r.target_id == method.id
            for r in contains
        )

    def test_method_is_not_constructor(self):
        src = "package main\ntype T struct{}\nfunc (t *T) Init() { }\n"
        result = extract(src)
        sym = sym_by_name(result, "Init")
        assert isinstance(sym, MethodSymbol)
        assert sym.is_constructor is False


# ---------------------------------------------------------------------------
# Struct extraction
# ---------------------------------------------------------------------------

class TestStructExtraction:
    def test_basic_struct(self):
        src = "package main\ntype Server struct { host string }\n"
        result = extract(src)
        sym = sym_by_name(result, "Server")
        assert isinstance(sym, ClassSymbol)
        assert sym.is_abstract is False

    def test_exported_struct_visibility(self):
        src = "package main\ntype Server struct{}\n"
        result = extract(src)
        sym = sym_by_name(result, "Server")
        assert sym.visibility == Visibility.PUBLIC
        assert sym.is_exported is True

    def test_unexported_struct(self):
        # Use file="main.go" to avoid stem collision: FILE="pkg/server/server.go"
        # would make ModuleSymbol.name="server" which conflicts with the struct name.
        src = "package main\ntype server struct{}\n"
        result = extract(src, file="main.go")
        sym = sym_by_name(result, "server")
        assert isinstance(sym, ClassSymbol)
        assert sym.visibility == Visibility.INTERNAL
        assert sym.is_exported is False

    def test_struct_moniker(self):
        src = "package main\ntype Config struct{}\n"
        result = extract(src)
        sym = sym_by_name(result, "Config")
        assert sym.id.endswith("Config#")
        assert is_valid_moniker(sym.id)

    def test_struct_qualified_name(self):
        src = "package server\ntype Config struct{}\n"
        result = extract(src, file="pkg/server/server.go")
        sym = sym_by_name(result, "Config")
        assert sym.qualified_name == "server.Config"

    def test_struct_embedded_field_extends(self):
        """Embedded struct field → EXTENDS with embedding_kind=struct."""
        src = "package main\ntype Base struct{}\ntype Child struct { Base }\n"
        result = extract(src)
        child = sym_by_name(result, "Child")
        extends = rels_by_kind(result, RelationKind.EXTENDS)
        struct_extends = [r for r in extends if r.metadata.get("embedding_kind") == "struct"]
        assert len(struct_extends) == 1
        assert struct_extends[0].source_id == child.id
        assert struct_extends[0].target_name == "Base"

    def test_struct_embedded_base_in_base_classes(self):
        src = "package main\ntype Base struct{}\ntype Child struct { Base }\n"
        result = extract(src)
        child = sym_by_name(result, "Child")
        assert "Base" in child.base_classes

    def test_struct_pointer_embedded_field(self):
        """Embedded *Base (pointer) → EXTENDS with type_name Base (pointer stripped)."""
        src = "package main\ntype Base struct{}\ntype Child struct { *Base }\n"
        result = extract(src)
        extends = rels_by_kind(result, RelationKind.EXTENDS)
        struct_extends = [r for r in extends if r.metadata.get("embedding_kind") == "struct"]
        assert any(r.target_name == "Base" for r in struct_extends)

    def test_struct_doc_comment(self):
        src = (
            "package main\n"
            "// Config holds server configuration.\n"
            "type Config struct{}\n"
        )
        result = extract(src)
        sym = sym_by_name(result, "Config")
        assert sym.docstring == "Config holds server configuration."

    def test_module_contains_struct(self):
        src = "package main\ntype Config struct{}\n"
        result = extract(src)
        mod = syms_by_kind(result, ModuleSymbol)[0]
        cls = sym_by_name(result, "Config")
        contains = rels_by_kind(result, RelationKind.CONTAINS)
        assert any(r.source_id == mod.id and r.target_id == cls.id for r in contains)


# ---------------------------------------------------------------------------
# Interface extraction
# ---------------------------------------------------------------------------

class TestInterfaceExtraction:
    def test_basic_interface(self):
        src = "package main\ntype Reader interface { Read(p []byte) (int, error) }\n"
        result = extract(src)
        sym = sym_by_name(result, "Reader")
        assert isinstance(sym, ClassSymbol)
        assert sym.is_abstract is True

    def test_interface_method_extracted(self):
        src = "package main\ntype Doer interface { Do() error }\n"
        result = extract(src)
        method = sym_by_name(result, "Do")
        assert isinstance(method, MethodSymbol)
        assert method.complexity is None  # no body
        assert method.enclosing_class_id is not None

    def test_interface_method_enclosing_class(self):
        src = "package main\ntype Doer interface { Do() error }\n"
        result = extract(src)
        iface = sym_by_name(result, "Doer")
        method = sym_by_name(result, "Do")
        assert method.enclosing_class_id == iface.id

    def test_interface_contains_method(self):
        src = "package main\ntype Doer interface { Do() error }\n"
        result = extract(src)
        iface = sym_by_name(result, "Doer")
        method = sym_by_name(result, "Do")
        contains = rels_by_kind(result, RelationKind.CONTAINS)
        assert any(r.source_id == iface.id and r.target_id == method.id for r in contains)

    def test_embedded_interface_extends(self):
        """Embedded interface → EXTENDS with embedding_kind=interface."""
        src = (
            "package main\n"
            "type Reader interface { Read(p []byte) (int, error) }\n"
            "type ReadWriter interface {\n"
            "    Reader\n"
            "    Write(p []byte) (int, error)\n"
            "}\n"
        )
        result = extract(src)
        rw = sym_by_name(result, "ReadWriter")
        extends = rels_by_kind(result, RelationKind.EXTENDS)
        iface_extends = [r for r in extends if r.metadata.get("embedding_kind") == "interface"]
        assert any(
            r.source_id == rw.id and r.target_name == "Reader"
            for r in iface_extends
        )

    def test_embedded_interface_in_base_classes(self):
        src = (
            "package main\n"
            "type Stringer interface { String() string }\n"
            "type Named interface { Stringer; Name() string }\n"
        )
        result = extract(src)
        named = sym_by_name(result, "Named")
        assert "Stringer" in named.base_classes

    def test_type_constraint_no_extends(self):
        """interface { int | float64 } → no EXTENDS relationship generated."""
        src = (
            "package main\n"
            "type Number interface {\n"
            "    int | float64\n"
            "}\n"
        )
        result = extract(src)
        extends = rels_by_kind(result, RelationKind.EXTENDS)
        # Should not generate any EXTENDS for constraint interfaces
        assert len(extends) == 0

    def test_interface_moniker_valid(self):
        src = "package main\ntype Doer interface { Do() error }\n"
        result = extract(src)
        iface = sym_by_name(result, "Doer")
        assert is_valid_moniker(iface.id)
        assert iface.id.endswith("Doer#")


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

class TestTypeAlias:
    def test_type_alias_is_variable_symbol(self):
        src = "package main\ntype MyInt int\n"
        result = extract(src)
        sym = sym_by_name(result, "MyInt")
        assert isinstance(sym, VariableSymbol)
        assert sym.is_constant is True

    def test_type_alias_annotation(self):
        src = "package main\ntype MyError = error\n"
        result = extract(src)
        sym = sym_by_name(result, "MyError")
        assert sym is not None
        assert sym.type_annotation is not None

    def test_type_alias_moniker(self):
        src = "package main\ntype MyInt int\n"
        result = extract(src)
        sym = sym_by_name(result, "MyInt")
        assert sym.id.endswith("MyInt.")
        assert is_valid_moniker(sym.id)


# ---------------------------------------------------------------------------
# Const declarations
# ---------------------------------------------------------------------------

class TestConstDeclaration:
    def test_typed_const(self):
        src = "package main\nconst MaxRetries int = 3\n"
        result = extract(src)
        sym = sym_by_name(result, "MaxRetries")
        assert isinstance(sym, VariableSymbol)
        assert sym.is_constant is True
        assert sym.type_annotation == "int"

    def test_untyped_const(self):
        """Untyped const (iota) → is_constant=True, type_annotation=None."""
        src = "package main\nconst Pi = 3.14\n"
        result = extract(src)
        sym = sym_by_name(result, "Pi")
        assert isinstance(sym, VariableSymbol)
        assert sym.is_constant is True
        assert sym.type_annotation is None

    def test_grouped_const(self):
        src = (
            "package main\n"
            "const (\n"
            "    Small int = iota\n"
            "    Medium\n"
            "    Large\n"
            ")\n"
        )
        result = extract(src)
        names = {s.name for s in syms_by_kind(result, VariableSymbol)}
        assert "Small" in names

    def test_const_visibility_exported(self):
        src = "package main\nconst MaxRetries = 3\n"
        result = extract(src)
        sym = sym_by_name(result, "MaxRetries")
        assert sym.visibility == Visibility.PUBLIC

    def test_const_visibility_unexported(self):
        src = "package main\nconst maxRetries = 3\n"
        result = extract(src)
        sym = sym_by_name(result, "maxRetries")
        assert sym.visibility == Visibility.INTERNAL

    def test_const_moniker(self):
        src = "package main\nconst Max int = 10\n"
        result = extract(src)
        sym = sym_by_name(result, "Max")
        assert sym.id.endswith("Max.")
        assert is_valid_moniker(sym.id)


# ---------------------------------------------------------------------------
# Var declarations
# ---------------------------------------------------------------------------

class TestVarDeclaration:
    def test_typed_var(self):
        src = "package main\nvar DefaultPort int = 8080\n"
        result = extract(src)
        sym = sym_by_name(result, "DefaultPort")
        assert isinstance(sym, VariableSymbol)
        assert sym.is_constant is False
        assert sym.type_annotation == "int"

    def test_untyped_var_inferred(self):
        """Untyped package-level var → type_annotation='<inferred>' (Go inference idiom)."""
        src = "package main\nimport \"net/http\"\nvar DefaultClient = &http.Client{}\n"
        result = extract(src)
        sym = sym_by_name(result, "DefaultClient")
        assert isinstance(sym, VariableSymbol)
        assert sym.type_annotation == "<inferred>"
        assert sym.is_constant is False

    def test_untyped_exported_var_still_extracted(self):
        """Package-level exported var with inferred type must not be silently dropped."""
        src = "package main\nvar DefaultMux = NewMux()\n"
        result = extract(src)
        sym = sym_by_name(result, "DefaultMux")
        assert sym is not None

    def test_var_moniker(self):
        src = "package main\nvar Port int = 8080\n"
        result = extract(src)
        sym = sym_by_name(result, "Port")
        assert is_valid_moniker(sym.id)
        assert sym.id.endswith("Port.")


# ---------------------------------------------------------------------------
# Import handling
# ---------------------------------------------------------------------------

class TestImports:
    def test_single_import(self):
        src = 'package main\nimport "fmt"\n'
        result = extract(src)
        imports = rels_by_kind(result, RelationKind.IMPORTS)
        assert any(r.target_name == "fmt" for r in imports)

    def test_grouped_imports(self):
        src = 'package main\nimport (\n    "fmt"\n    "os"\n)\n'
        result = extract(src)
        imports = rels_by_kind(result, RelationKind.IMPORTS)
        names = {r.target_name for r in imports}
        assert "fmt" in names
        assert "os" in names

    def test_aliased_import(self):
        src = 'package main\nimport mrand "math/rand"\n'
        result = extract(src)
        imports = rels_by_kind(result, RelationKind.IMPORTS)
        assert any(r.target_name == "mrand" for r in imports)
        aliased = next(r for r in imports if r.target_name == "mrand")
        assert aliased.metadata["import_source"] == "math/rand"
        assert aliased.metadata["alias"] == "mrand"

    def test_blank_import_skipped(self):
        """Blank imports (_ "pkg") have no target name and are skipped."""
        src = 'package main\nimport _ "embed"\n'
        result = extract(src)
        imports = rels_by_kind(result, RelationKind.IMPORTS)
        assert len(imports) == 0

    def test_cgo_import(self):
        """import "C" (cgo pseudo-package) → regular IMPORTS, no crash."""
        src = 'package main\nimport "C"\n'
        result = extract(src)
        imports = rels_by_kind(result, RelationKind.IMPORTS)
        assert any(r.target_name == "C" for r in imports)

    def test_import_source_metadata(self):
        src = 'package main\nimport "net/http"\n'
        result = extract(src)
        imports = rels_by_kind(result, RelationKind.IMPORTS)
        http_import = next(r for r in imports if r.target_name == "http")
        assert http_import.metadata["import_source"] == "net/http"

    def test_import_last_segment_as_name(self):
        src = 'package main\nimport "encoding/json"\n'
        result = extract(src)
        imports = rels_by_kind(result, RelationKind.IMPORTS)
        assert any(r.target_name == "json" for r in imports)


# ---------------------------------------------------------------------------
# Doc comments
# ---------------------------------------------------------------------------

class TestDocComments:
    def test_doc_comment_attached(self):
        src = (
            "package main\n"
            "// Serve starts the HTTP server.\n"
            "func Serve() {}\n"
        )
        result = extract(src)
        sym = sym_by_name(result, "Serve")
        assert sym.docstring == "Serve starts the HTTP server."

    def test_blank_line_breaks_doc_comment(self):
        src = (
            "package main\n"
            "// This is a stray comment.\n"
            "\n"
            "func Serve() {}\n"
        )
        result = extract(src)
        sym = sym_by_name(result, "Serve")
        assert sym.docstring is None

    def test_go_generate_not_in_docstring(self):
        """//go:generate directives must not appear in the docstring of a struct."""
        # Use a struct (ClassSymbol) which has a docstring field.
        # VariableSymbol (type Color int) has no docstring field.
        src = (
            "package main\n"
            "//go:generate stringer -type=Color\n"
            "// Color represents a display color.\n"
            "type Color struct{}\n"
        )
        result = extract(src)
        sym = sym_by_name(result, "Color")
        assert isinstance(sym, ClassSymbol)
        # The go:generate directive line must not appear in the docstring
        if sym.docstring:
            assert "go:generate" not in sym.docstring

    def test_go_directive_only_not_a_docstring(self):
        """A //go:build line alone preceding a declaration is not a docstring."""
        src = (
            "package main\n"
            "//go:build linux\n"
            "func LinuxOnly() {}\n"
        )
        result = extract(src)
        sym = sym_by_name(result, "LinuxOnly")
        assert sym.docstring is None

    def test_multiline_doc_comment(self):
        src = (
            "package main\n"
            "// Serve starts the HTTP server.\n"
            "// It binds to the given address.\n"
            "func Serve(addr string) {}\n"
        )
        result = extract(src)
        sym = sym_by_name(result, "Serve")
        assert sym.docstring is not None
        assert "Serve starts" in sym.docstring
        assert "binds" in sym.docstring


# ---------------------------------------------------------------------------
# Build tags
# ---------------------------------------------------------------------------

class TestBuildTags:
    def test_build_ignore_returns_empty(self):
        src = "//go:build ignore\npackage main\nfunc Serve() {}\n"
        result = extract(src)
        assert result.symbols == []
        assert result.relationships == []

    def test_build_ignore_empty_file_still_has_file_record(self):
        src = "//go:build ignore\npackage main\n"
        result = extract(src)
        assert result.file is not None
        assert result.file.language == "go"

    def test_non_ignore_build_tag_indexed_normally(self):
        """//go:build linux → NOT skipped. Only //go:build ignore is excluded."""
        src = "//go:build linux\npackage main\nfunc Run() {}\n"
        result = extract(src)
        sym = sym_by_name(result, "Run")
        assert sym is not None


# ---------------------------------------------------------------------------
# Call expressions
# ---------------------------------------------------------------------------

class TestCallExpressions:
    def test_direct_call(self):
        src = (
            "package main\n"
            "func helper() {}\n"
            "func main() { helper() }\n"
        )
        result = extract(src)
        main_sym = sym_by_name(result, "main")
        calls = [
            r for r in rels_by_kind(result, RelationKind.CALLS)
            if r.source_id == main_sym.id
        ]
        assert any(r.target_name == "helper" and r.metadata["call_form"] == "direct" for r in calls)

    def test_method_call(self):
        src = (
            "package main\n"
            "import \"fmt\"\n"
            "func main() { fmt.Println(\"hi\") }\n"
        )
        result = extract(src)
        main_sym = sym_by_name(result, "main")
        calls = [
            r for r in rels_by_kind(result, RelationKind.CALLS)
            if r.source_id == main_sym.id
        ]
        assert any(
            r.target_name == "Println"
            and r.metadata["call_form"] == "method"
            and r.metadata["receiver"] == "fmt"
            for r in calls
        )

    def test_calls_inside_func_literal_not_attributed_to_outer(self):
        """Calls inside func literals must not be attributed to the enclosing function."""
        src = (
            "package main\n"
            "func main() {\n"
            "    f := func() { inner() }\n"
            "    f()\n"
            "}\n"
            "func inner() {}\n"
        )
        result = extract(src)
        main_sym = sym_by_name(result, "main")
        calls = [
            r for r in rels_by_kind(result, RelationKind.CALLS)
            if r.source_id == main_sym.id
        ]
        # "f" should be called (it's called directly in main), but "inner" should NOT
        # be in main's calls (it's called inside the func literal)
        inner_calls = [r for r in calls if r.target_name == "inner"]
        assert len(inner_calls) == 0

    def test_call_location_present(self):
        src = "package main\nfunc helper() {}\nfunc main() { helper() }\n"
        result = extract(src)
        calls = rels_by_kind(result, RelationKind.CALLS)
        assert all(r.location is not None for r in calls)


# ---------------------------------------------------------------------------
# Complexity
# ---------------------------------------------------------------------------

class TestComplexity:
    def test_simple_function_complexity_one(self):
        src = "package main\nfunc Noop() { }\n"
        result = extract(src)
        sym = sym_by_name(result, "Noop")
        assert sym.complexity == 1

    def test_if_adds_complexity(self):
        src = (
            "package main\n"
            "func Check(x int) bool {\n"
            "    if x > 0 { return true }\n"
            "    return false\n"
            "}\n"
        )
        result = extract(src)
        sym = sym_by_name(result, "Check")
        assert sym.complexity == 2

    def test_for_adds_complexity(self):
        src = (
            "package main\n"
            "func Loop(n int) {\n"
            "    for i := 0; i < n; i++ { }\n"
            "}\n"
        )
        result = extract(src)
        sym = sym_by_name(result, "Loop")
        assert sym.complexity == 2

    def test_boolean_operator_adds_complexity(self):
        src = (
            "package main\n"
            "func Both(a, b bool) bool {\n"
            "    return a && b\n"
            "}\n"
        )
        result = extract(src)
        sym = sym_by_name(result, "Both")
        assert sym.complexity == 2

    def test_interface_method_complexity_none(self):
        src = "package main\ntype Doer interface { Do() error }\n"
        result = extract(src)
        method = sym_by_name(result, "Do")
        assert method.complexity is None


# ---------------------------------------------------------------------------
# Moniker and visibility invariants
# ---------------------------------------------------------------------------

class TestMonikerInvariants:
    def test_all_monikers_valid(self):
        src = (
            "package server\n"
            'import "fmt"\n'
            "const MaxConn int = 100\n"
            "var DefaultPort = 8080\n"
            "type Config struct { host string }\n"
            "type Handler interface { Handle() }\n"
            "func NewConfig() *Config { return nil }\n"
            "func (c *Config) SetHost(h string) { fmt.Println(h) }\n"
        )
        result = extract(src, file="pkg/server/server.go")
        for sym in result.symbols:
            assert is_valid_moniker(sym.id), f"Invalid moniker: {sym.id!r}"

    def test_no_duplicate_monikers(self):
        src = (
            "package main\n"
            "type A struct{}\n"
            "type B struct{}\n"
            "func (a *A) Get() {}\n"
            "func (b *B) Get() {}\n"
            "func Top() {}\n"
        )
        result = extract(src)
        ids = [s.id for s in result.symbols]
        assert len(ids) == len(set(ids)), f"Duplicate monikers: {ids}"

    def test_all_symbols_have_go_language(self):
        src = "package main\ntype T struct{}\nfunc F() {}\nconst C = 1\n"
        result = extract(src)
        for sym in result.symbols:
            assert sym.language == "go"


# ---------------------------------------------------------------------------
# Cross-file method linking (indexer post-pass)
# ---------------------------------------------------------------------------

class TestCrossFileMethodLinking:
    """
    Tests for Indexer._resolve_go_methods().
    These tests simulate what the indexer does after aggregating all files.
    """

    def test_unresolved_method_gets_linked_by_post_pass(self):
        """
        types.go defines Config struct.  config.go defines Config.SetHost().
        After the post-pass, SetHost.enclosing_class_id should point to Config.
        """
        from sutra.core.extractor.adapters.go import GoAdapter
        from sutra.core.indexer import Indexer
        from sutra.core.output.json_graph_exporter import JsonGraphExporter

        types_src = b"package server\ntype Config struct{}\n"
        config_src = b"package server\nfunc (c *Config) SetHost(h string) { }\n"

        adapter = GoAdapter()
        types_ext = adapter.extract("pkg/server/types.go", types_src, "myrepo")
        config_ext = adapter.extract("pkg/server/config.go", config_src, "myrepo")

        # Simulate indexer aggregation
        all_symbols = types_ext.symbols + config_ext.symbols
        all_rels = types_ext.relationships + config_ext.relationships

        # Run post-pass
        from sutra.core.embedder.fixture import FixtureEmbedder
        indexer = Indexer(adapters={}, exporter=JsonGraphExporter(), embedder=FixtureEmbedder())
        indexer._resolve_go_methods(all_symbols, all_rels)

        # Find SetHost
        set_host = next(s for s in all_symbols if s.name == "SetHost")
        assert set_host.enclosing_class_id is not None, (
            "Cross-file method was not linked to its struct"
        )

        # Find Config
        config = next(s for s in all_symbols if s.name == "Config")
        assert set_host.enclosing_class_id == config.id

    def test_post_pass_adds_contains_relationship(self):
        """After the post-pass, a CONTAINS relationship is added for cross-file methods."""
        from sutra.core.extractor.adapters.go import GoAdapter
        from sutra.core.indexer import Indexer
        from sutra.core.output.json_graph_exporter import JsonGraphExporter

        types_src = b"package server\ntype Config struct{}\n"
        config_src = b"package server\nfunc (c *Config) SetHost(h string) { }\n"

        adapter = GoAdapter()
        types_ext = adapter.extract("pkg/server/types.go", types_src, "myrepo")
        config_ext = adapter.extract("pkg/server/config.go", config_src, "myrepo")

        all_symbols = types_ext.symbols + config_ext.symbols
        all_rels = types_ext.relationships + config_ext.relationships

        from sutra.core.embedder.fixture import FixtureEmbedder
        indexer = Indexer(adapters={}, exporter=JsonGraphExporter(), embedder=FixtureEmbedder())
        indexer._resolve_go_methods(all_symbols, all_rels)

        config = next(s for s in all_symbols if s.name == "Config")
        set_host = next(s for s in all_symbols if s.name == "SetHost")
        contains = [
            r for r in all_rels
            if r.kind == RelationKind.CONTAINS
            and r.source_id == config.id
            and r.target_id == set_host.id
        ]
        assert len(contains) == 1
        assert contains[0].is_resolved is True


# ---------------------------------------------------------------------------
# Indexer integration: _test.go files skipped
# ---------------------------------------------------------------------------

class TestIndexerGoIntegration:
    def test_test_go_files_skipped(self, tmp_path):
        """_test.go files must not be indexed."""
        from sutra.core.extractor.adapters.go import GoAdapter
        from sutra.core.indexer import Indexer
        from sutra.core.output.json_graph_exporter import JsonGraphExporter

        (tmp_path / "main.go").write_bytes(b"package main\nfunc Main() {}\n")
        (tmp_path / "main_test.go").write_bytes(b"package main\nfunc TestMain() {}\n")

        from sutra.core.embedder.fixture import FixtureEmbedder
        indexer = Indexer(
            adapters={"go": GoAdapter()},
            exporter=JsonGraphExporter(),
            embedder=FixtureEmbedder(),
        )
        result = indexer.index(root=tmp_path, repo_url="https://github.com/test/r", output_dir=tmp_path / "out")

        names = {s.name for s in result.symbols}
        assert "Main" in names
        assert "TestMain" not in names
