"""
Tests for TypeScriptAdapter (Priority 7).

Strategy: parse real TypeScript source strings — no mocks, no fake nodes.
Each test class targets a specific adapter code path.
"""
from __future__ import annotations

import pytest

from sutra.core.extractor.adapters.typescript import TypeScriptAdapter
from sutra.core.extractor.base import (
    ClassSymbol,
    FunctionSymbol,
    MethodSymbol,
    ModuleSymbol,
    RelationKind,
    VariableSymbol,
    Visibility,
)
from sutra.core.extractor.moniker import is_valid_moniker

REPO = "myrepo"
FILE = "src/user.ts"
TSX_FILE = "src/comp.tsx"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract(source: str, file: str = FILE) -> object:
    tsx = file.endswith(".tsx")
    adapter = TypeScriptAdapter(tsx=tsx)
    return adapter.extract(file, source.encode(), REPO)


def sym_by_name(result, name: str):
    return next((s for s in result.symbols if s.name == name), None)


def rels_of_kind(result, kind: RelationKind):
    return [r for r in result.relationships if r.kind == kind]


# ---------------------------------------------------------------------------
# Module symbol
# ---------------------------------------------------------------------------

class TestModuleSymbol:
    def test_always_present(self):
        r = extract("const x: number = 1;")
        mod = next(s for s in r.symbols if isinstance(s, ModuleSymbol))
        assert mod.name == "user"
        assert mod.language == "typescript"
        assert is_valid_moniker(mod.id)

    def test_jsdoc_module_docstring(self):
        r = extract('/** Module doc. */\nconst x: number = 1;')
        mod = next(s for s in r.symbols if isinstance(s, ModuleSymbol))
        assert mod.docstring == "Module doc."

    def test_no_jsdoc_no_docstring(self):
        r = extract("// regular comment\nconst x: number = 1;")
        mod = next(s for s in r.symbols if isinstance(s, ModuleSymbol))
        assert mod.docstring is None

    def test_non_jsdoc_block_comment_not_docstring(self):
        r = extract("/* not jsdoc */\nconst x: number = 1;")
        mod = next(s for s in r.symbols if isinstance(s, ModuleSymbol))
        assert mod.docstring is None


# ---------------------------------------------------------------------------
# Function declarations
# ---------------------------------------------------------------------------

class TestFunctionDeclaration:
    def test_basic_function(self):
        r = extract("function greet(name: string): string { return name; }")
        fn = sym_by_name(r, "greet")
        assert isinstance(fn, FunctionSymbol)
        assert fn.return_type == "string"
        assert len(fn.parameters) == 1
        assert fn.parameters[0].name == "name"
        assert fn.parameters[0].type_annotation == "string"
        assert fn.is_async is False
        assert fn.is_exported is False
        assert is_valid_moniker(fn.id)

    def test_exported_function(self):
        r = extract("export function add(a: number, b: number): number { return a + b; }")
        fn = sym_by_name(r, "add")
        assert fn.is_exported is True

    def test_async_function(self):
        r = extract("async function fetch(url: string): Promise<string> { return url; }")
        fn = sym_by_name(r, "fetch")
        assert fn.is_async is True
        assert fn.return_type == "Promise<string>"

    def test_generic_function_signature_includes_type_params(self):
        r = extract("function identity<T>(x: T): T { return x; }")
        fn = sym_by_name(r, "identity")
        assert "<T>" in fn.signature

    def test_optional_parameter(self):
        r = extract("function greet(name: string, age?: number): void {}")
        fn = sym_by_name(r, "greet")
        assert len(fn.parameters) == 2
        assert fn.parameters[1].name == "age"
        assert fn.parameters[1].type_annotation == "number"

    def test_rest_parameter(self):
        r = extract("function sum(...args: number[]): number { return 0; }")
        fn = sym_by_name(r, "sum")
        assert len(fn.parameters) == 1
        assert fn.parameters[0].is_variadic is True
        assert fn.parameters[0].name == "args"

    def test_jsdoc_attached(self):
        r = extract('/** Adds two numbers. */\nfunction add(a: number, b: number): number { return a + b; }')
        fn = sym_by_name(r, "add")
        assert fn.docstring == "Adds two numbers."

    def test_contains_relationship(self):
        r = extract("function greet(): void {}")
        fn = sym_by_name(r, "greet")
        contains = rels_of_kind(r, RelationKind.CONTAINS)
        target_ids = [rel.target_id for rel in contains]
        assert fn.id in target_ids

    def test_calls_relationship(self):
        r = extract("function foo(): void { bar(); }\nfunction bar(): void {}")
        foo = sym_by_name(r, "foo")
        calls = [rel for rel in rels_of_kind(r, RelationKind.CALLS)
                 if rel.source_id == foo.id]
        assert any(c.target_name == "bar" for c in calls)

    def test_method_calls_relationship(self):
        r = extract("function foo(): void { obj.method(); }")
        calls = rels_of_kind(r, RelationKind.CALLS)
        method_calls = [c for c in calls if c.metadata.get("call_form") == "method"]
        assert any(c.target_name == "method" for c in method_calls)

    def test_complexity_if_branch(self):
        r = extract("function foo(x: number): number { if (x > 0) { return x; } return 0; }")
        fn = sym_by_name(r, "foo")
        assert fn.complexity == 2  # 1 base + 1 if

    def test_complexity_nested_scope_not_counted(self):
        r = extract("""
function outer(): void {
  function inner(): void {
    if (true) {}
  }
}
""")
        fn = sym_by_name(r, "outer")
        assert fn.complexity == 1  # inner function's if not counted


# ---------------------------------------------------------------------------
# Arrow functions
# ---------------------------------------------------------------------------

class TestArrowFunctions:
    def test_simple_arrow_is_function_symbol(self):
        r = extract("const greet = (name: string): string => name;")
        fn = sym_by_name(r, "greet")
        assert isinstance(fn, FunctionSymbol)
        assert fn.parameters[0].name == "name"
        assert fn.return_type == "string"

    def test_async_arrow(self):
        r = extract("const load = async (url: string): Promise<void> => { console.log(url); };")
        fn = sym_by_name(r, "load")
        assert isinstance(fn, FunctionSymbol)
        assert fn.is_async is True

    def test_exported_arrow(self):
        r = extract("export const handler = (x: number): void => {};")
        fn = sym_by_name(r, "handler")
        assert fn.is_exported is True

    def test_untyped_arrow_not_extracted(self):
        r = extract("const cb = (x) => x;")
        # No type annotation on parameter, no return type annotation
        # But the variable itself has no type annotation either
        names = [s.name for s in r.symbols if not isinstance(s, ModuleSymbol)]
        assert "cb" not in names

    def test_function_expression_as_const(self):
        r = extract("const fn = function(x: number): string { return String(x); };")
        sym = sym_by_name(r, "fn")
        assert isinstance(sym, FunctionSymbol)

    def test_destructured_arrow_skipped(self):
        r = extract("const { a, b } = someObj;")
        names = [s.name for s in r.symbols if not isinstance(s, ModuleSymbol)]
        assert not names  # no symbols extracted (untyped destructuring)


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------

class TestClassDeclaration:
    def test_basic_class(self):
        r = extract("class Foo {}")
        cls = sym_by_name(r, "Foo")
        assert isinstance(cls, ClassSymbol)
        assert cls.is_abstract is False
        assert cls.base_classes == []
        assert cls.is_exported is False
        assert is_valid_moniker(cls.id)

    def test_exported_class(self):
        r = extract("export class Bar {}")
        cls = sym_by_name(r, "Bar")
        assert cls.is_exported is True

    def test_class_extends(self):
        r = extract("class Dog extends Animal {}")
        cls = sym_by_name(r, "Dog")
        assert cls.base_classes == ["Animal"]
        extends = rels_of_kind(r, RelationKind.EXTENDS)
        assert any(e.target_name == "Animal" for e in extends)

    def test_class_implements(self):
        r = extract("class Service implements IService {}")
        cls = sym_by_name(r, "Service")
        implements = rels_of_kind(r, RelationKind.IMPLEMENTS)
        assert any(i.target_name == "IService" for i in implements)
        assert any(i.source_id == cls.id for i in implements)

    def test_class_extends_and_implements(self):
        r = extract("class Repo extends Base implements IRepo, ISearchable {}")
        extends = rels_of_kind(r, RelationKind.EXTENDS)
        implements = rels_of_kind(r, RelationKind.IMPLEMENTS)
        assert any(e.target_name == "Base" for e in extends)
        impl_targets = {i.target_name for i in implements}
        assert "IRepo" in impl_targets
        assert "ISearchable" in impl_targets

    def test_class_jsdoc(self):
        r = extract('/** A user class. */\nclass User {}')
        cls = sym_by_name(r, "User")
        assert cls.docstring == "A user class."

    def test_abstract_class(self):
        r = extract("abstract class Shape {}")
        cls = sym_by_name(r, "Shape")
        assert cls.is_abstract is True

    def test_generic_class_name_extracted(self):
        r = extract("class Box<T> { getValue(): T { return this.value; } private value: T; }")
        cls = sym_by_name(r, "Box")
        assert isinstance(cls, ClassSymbol)
        assert cls.name == "Box"

    def test_export_default_class_with_name(self):
        r = extract("export default class Named {}")
        cls = sym_by_name(r, "Named")
        assert isinstance(cls, ClassSymbol)
        assert cls.is_exported is True

    def test_anonymous_export_default_class_skipped(self):
        r = extract("export default class {}")
        names = [s.name for s in r.symbols if not isinstance(s, ModuleSymbol)]
        assert not names


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

class TestMethods:
    def test_basic_method(self):
        r = extract("class Svc { run(x: number): void {} }")
        method = sym_by_name(r, "run")
        assert isinstance(method, MethodSymbol)
        assert method.enclosing_class_id.endswith("Svc#")
        assert is_valid_moniker(method.id)

    def test_constructor_detected(self):
        r = extract("class Svc { constructor(private db: string) {} }")
        ctor = sym_by_name(r, "constructor")
        assert isinstance(ctor, MethodSymbol)
        assert ctor.is_constructor is True

    def test_async_method(self):
        r = extract("class Svc { async fetch(): Promise<void> {} }")
        method = sym_by_name(r, "fetch")
        assert method.is_async is True

    def test_static_method(self):
        r = extract("class Svc { static create(): Svc { return new Svc(); } }")
        method = sym_by_name(r, "create")
        assert method.is_static is True

    def test_private_method_visibility(self):
        r = extract("class Svc { private helper(): void {} }")
        method = sym_by_name(r, "helper")
        assert method.visibility == Visibility.PRIVATE

    def test_protected_method_visibility(self):
        r = extract("class Svc { protected init(): void {} }")
        method = sym_by_name(r, "init")
        assert method.visibility == Visibility.PROTECTED

    def test_decorator_on_method(self):
        r = extract("""
class Ctrl {
  @Get("/users")
  list(): void {}
}
""")
        method = sym_by_name(r, "list")
        assert any("Get" in d for d in method.decorators)

    def test_multiple_decorators_on_method(self):
        r = extract("""
class Ctrl {
  @Auth()
  @Log()
  handle(): void {}
}
""")
        method = sym_by_name(r, "handle")
        assert len(method.decorators) == 2

    def test_jsdoc_on_method(self):
        r = extract("""
class Svc {
  /** Runs the service. */
  run(): void {}
}
""")
        method = sym_by_name(r, "run")
        assert method.docstring == "Runs the service."

    def test_abstract_method(self):
        r = extract("abstract class Shape { abstract area(): number; }")
        method = sym_by_name(r, "area")
        assert isinstance(method, MethodSymbol)
        assert method.complexity is None  # no body

    def test_method_calls_extracted(self):
        r = extract("class Svc { run(): void { this.helper(); } helper(): void {} }")
        run = sym_by_name(r, "run")
        calls = [rel for rel in rels_of_kind(r, RelationKind.CALLS)
                 if rel.source_id == run.id]
        assert any(c.target_name == "helper" for c in calls)

    def test_class_contains_method(self):
        r = extract("class Svc { run(): void {} }")
        cls = sym_by_name(r, "Svc")
        method = sym_by_name(r, "run")
        contains = rels_of_kind(r, RelationKind.CONTAINS)
        assert any(c.source_id == cls.id and c.target_id == method.id for c in contains)


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------

class TestInterfaces:
    def test_interface_is_class_symbol_abstract(self):
        r = extract("interface IUser { id: number; }")
        iface = sym_by_name(r, "IUser")
        assert isinstance(iface, ClassSymbol)
        assert iface.is_abstract is True
        assert is_valid_moniker(iface.id)

    def test_interface_method_signature_extracted(self):
        r = extract("interface IRepo { find(id: number): string; }")
        method = sym_by_name(r, "find")
        assert isinstance(method, MethodSymbol)
        assert method.complexity is None  # no body
        assert method.parameters[0].name == "id"

    def test_interface_extends(self):
        r = extract("interface IDerived extends IBase { extra(): void; }")
        iface = sym_by_name(r, "IDerived")
        assert "IBase" in iface.base_classes
        extends = rels_of_kind(r, RelationKind.EXTENDS)
        assert any(e.target_name == "IBase" for e in extends)

    def test_exported_interface(self):
        r = extract("export interface IUser {}")
        iface = sym_by_name(r, "IUser")
        assert iface.is_exported is True


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TestEnums:
    def test_enum_is_class_symbol(self):
        r = extract('enum Direction { Up = "UP", Down = "DOWN" }')
        enum = sym_by_name(r, "Direction")
        assert isinstance(enum, ClassSymbol)
        assert enum.is_abstract is False
        assert enum.base_classes == []
        assert is_valid_moniker(enum.id)

    def test_exported_enum(self):
        r = extract('export enum Status { Active, Inactive }')
        enum = sym_by_name(r, "Status")
        assert enum.is_exported is True

    def test_enum_contains_relationship(self):
        r = extract('enum Color { Red }')
        enum = sym_by_name(r, "Color")
        contains = rels_of_kind(r, RelationKind.CONTAINS)
        assert any(c.target_id == enum.id for c in contains)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

class TestTypeAliases:
    def test_type_alias_is_variable_symbol_constant(self):
        r = extract("type UserId = string;")
        alias = sym_by_name(r, "UserId")
        assert isinstance(alias, VariableSymbol)
        assert alias.is_constant is True
        assert is_valid_moniker(alias.id)

    def test_type_alias_union_full_text_stored(self):
        r = extract("type Result = string | number | null;")
        alias = sym_by_name(r, "Result")
        assert "string" in alias.type_annotation
        assert "number" in alias.type_annotation

    def test_exported_type_alias(self):
        r = extract("export type ID = string;")
        alias = sym_by_name(r, "ID")
        assert alias.is_exported is True


# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

class TestVariables:
    def test_typed_const_extracted(self):
        r = extract("const MAX: number = 3;")
        var = sym_by_name(r, "MAX")
        assert isinstance(var, VariableSymbol)
        assert var.type_annotation == "number"
        assert var.is_constant is True
        assert is_valid_moniker(var.id)

    def test_typed_let_not_constant(self):
        r = extract("let count: number = 0;")
        var = sym_by_name(r, "count")
        assert isinstance(var, VariableSymbol)
        assert var.is_constant is False

    def test_untyped_const_skipped(self):
        r = extract("const x = 5;")
        names = [s.name for s in r.symbols if not isinstance(s, ModuleSymbol)]
        assert "x" not in names

    def test_typed_const_and_untyped_side_by_side(self):
        r = extract("const x: number = 5;\nconst y = 10;")
        names = [s.name for s in r.symbols if not isinstance(s, ModuleSymbol)]
        assert "x" in names
        assert "y" not in names


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

class TestImports:
    def test_named_imports(self):
        r = extract('import { Foo, Bar } from "./lib";')
        imports = rels_of_kind(r, RelationKind.IMPORTS)
        targets = {i.target_name for i in imports}
        assert "Foo" in targets
        assert "Bar" in targets

    def test_named_import_with_alias(self):
        r = extract('import { Foo as F } from "./lib";')
        imports = rels_of_kind(r, RelationKind.IMPORTS)
        assert any(i.target_name == "Foo" and i.metadata.get("alias") == "F"
                   for i in imports)

    def test_default_import(self):
        r = extract('import path from "path";')
        imports = rels_of_kind(r, RelationKind.IMPORTS)
        assert any(i.target_name == "path" for i in imports)

    def test_namespace_import(self):
        r = extract('import * as fs from "fs";')
        imports = rels_of_kind(r, RelationKind.IMPORTS)
        assert any(i.target_name == "*" and i.metadata.get("alias") == "fs"
                   for i in imports)

    def test_import_source_recorded(self):
        r = extract('import { Service } from "./services/user";')
        imports = rels_of_kind(r, RelationKind.IMPORTS)
        assert any(i.metadata.get("import_source") == "./services/user" for i in imports)


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------

class TestReExports:
    def test_named_reexport_is_import_rel(self):
        r = extract('export { foo } from "./bar";')
        imports = rels_of_kind(r, RelationKind.IMPORTS)
        assert any(i.target_name == "foo" and i.metadata.get("re_export") is True
                   for i in imports)

    def test_star_reexport(self):
        r = extract('export * from "./baz";')
        imports = rels_of_kind(r, RelationKind.IMPORTS)
        assert any(i.target_name == "*" and i.metadata.get("import_source") == "./baz"
                   for i in imports)

    def test_namespace_reexport(self):
        r = extract('export * as utils from "./utils";')
        imports = rels_of_kind(r, RelationKind.IMPORTS)
        assert any(i.metadata.get("alias") == "utils" for i in imports)


# ---------------------------------------------------------------------------
# Skipped constructs
# ---------------------------------------------------------------------------

class TestSkippedConstructs:
    def test_anonymous_export_default_function_skipped(self):
        r = extract("export default function() { return 1; }")
        fns = [s for s in r.symbols if isinstance(s, FunctionSymbol)]
        assert not fns

    def test_anonymous_export_default_class_skipped(self):
        r = extract("export default class {}")
        classes = [s for s in r.symbols if isinstance(s, ClassSymbol)]
        assert not classes

    def test_namespace_content_skipped(self):
        r = extract("""
namespace MyNS {
  export function nsFunc(): void {}
}
""")
        # namespace content (nsFunc) is skipped in Phase 1
        names = [s.name for s in r.symbols if not isinstance(s, ModuleSymbol)]
        assert "nsFunc" not in names


# ---------------------------------------------------------------------------
# TSX support
# ---------------------------------------------------------------------------

class TestTSX:
    def test_tsx_adapter_parses_jsx_without_crash(self):
        src = """
import React from "react";

interface Props {
  name: string;
}

export function Greeting({ name }: Props): JSX.Element {
  return <div>{name}</div>;
}
"""
        r = extract(src, file=TSX_FILE)
        # Must not raise; function should still be extracted
        # (JSX expression is just the return value)
        fn = sym_by_name(r, "Greeting")
        assert fn is not None
        assert isinstance(fn, FunctionSymbol)

    def test_tsx_interface_extracted(self):
        src = 'interface Props { name: string; }\nexport function C(p: Props): JSX.Element { return <div/>; }'
        r = extract(src, file=TSX_FILE)
        iface = sym_by_name(r, "Props")
        assert isinstance(iface, ClassSymbol)
        assert iface.is_abstract is True


# ---------------------------------------------------------------------------
# Monikers and contract
# ---------------------------------------------------------------------------

class TestMonikerContract:
    def test_all_monikers_valid(self):
        src = """
import { A } from "./a";
const MAX: number = 1;
type ID = string;
enum Color { Red }
interface IFoo { run(): void; }
class Bar extends A implements IFoo {
  constructor() {}
  run(): void {}
}
export function top(): void {}
export const fn = (): string => "x";
"""
        r = extract(src)
        for sym in r.symbols:
            assert is_valid_moniker(sym.id), f"Invalid moniker: {sym.id!r}"

    def test_no_duplicate_monikers(self):
        src = """
class Svc {
  run(): void {}
  stop(): void {}
}
function run(): void {}
"""
        r = extract(src)
        ids = [s.id for s in r.symbols]
        assert len(ids) == len(set(ids)), "Duplicate monikers found"

    def test_full_integration(self):
        """Realistic module — every symbol type present, all relationships populated."""
        src = """
/** Auth service module. */

import { hash } from "bcrypt";
import type { Request } from "express";

const SALT_ROUNDS: number = 10;
type Token = string;

export interface IAuthService {
  login(email: string, password: string): Promise<Token>;
}

export abstract class BaseService {
  abstract init(): void;
}

export enum AuthError {
  InvalidCredentials = "INVALID_CREDENTIALS",
  Expired = "EXPIRED",
}

/** Handles authentication. */
export class AuthService extends BaseService implements IAuthService {
  constructor(private secret: string) {
    super();
  }

  async login(email: string, password: string): Promise<Token> {
    const h = hash(password, SALT_ROUNDS);
    return this.sign(email);
  }

  private sign(email: string): Token {
    return email;
  }

  init(): void {}
}

/** Signs a token. */
export function signToken(payload: string): Token {
  return payload;
}

export const validateToken = (token: Token): boolean => {
  return token.length > 0;
};
"""
        r = extract(src)
        names = {s.name for s in r.symbols}
        assert "AuthService" in names
        assert "BaseService" in names
        assert "IAuthService" in names
        assert "AuthError" in names
        assert "signToken" in names
        assert "validateToken" in names
        assert "SALT_ROUNDS" in names
        assert "Token" in names

        # EXTENDS
        extends = rels_of_kind(r, RelationKind.EXTENDS)
        assert any(e.target_name == "BaseService" for e in extends)

        # IMPLEMENTS
        implements = rels_of_kind(r, RelationKind.IMPLEMENTS)
        assert any(i.target_name == "IAuthService" for i in implements)

        # CALLS inside login
        auth_svc = sym_by_name(r, "AuthService")
        login = next(s for s in r.symbols
                     if isinstance(s, MethodSymbol) and s.name == "login"
                     and s.enclosing_class_id == auth_svc.id)
        calls = [rel for rel in rels_of_kind(r, RelationKind.CALLS)
                 if rel.source_id == login.id]
        call_targets = {c.target_name for c in calls}
        assert "hash" in call_targets
        assert "sign" in call_targets

        for sym in r.symbols:
            assert is_valid_moniker(sym.id)
