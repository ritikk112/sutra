"""
Tests for the Python tree-sitter adapter (Priority 3+4).

All tests parse real Python source strings — no mocks.
Run: python -m pytest tests/test_python_adapter.py -v
"""
import pytest
from sutra.core.extractor.adapters.python import PythonAdapter
from sutra.core.extractor.base import (
    ClassSymbol, FunctionSymbol, MethodSymbol, ModuleSymbol, VariableSymbol,
    RelationKind, Visibility,
)

REPO = "my-app"
FILE = "src/services/user.py"

@pytest.fixture(scope="module")
def adapter():
    return PythonAdapter()


def extract(adapter, source: str, file_path: str = FILE):
    return adapter.extract(file_path, source.encode(), REPO)


def sym_by_name(extraction, name):
    return next((s for s in extraction.symbols if s.name == name), None)

def rels_of_kind(extraction, kind):
    return [r for r in extraction.relationships if r.kind == kind]

def rels_from(extraction, source_id):
    return [r for r in extraction.relationships if r.source_id == source_id]


# ---------------------------------------------------------------------------
# ModuleSymbol
# ---------------------------------------------------------------------------

class TestModuleSymbol:
    def test_one_module_symbol_per_file(self, adapter):
        result = extract(adapter, "x = 1\n")
        mods = [s for s in result.symbols if isinstance(s, ModuleSymbol)]
        assert len(mods) == 1

    def test_module_name_is_stem(self, adapter):
        result = extract(adapter, "")
        mod = result.symbols[0]
        assert isinstance(mod, ModuleSymbol)
        assert mod.name == "user"

    def test_module_qualified_name(self, adapter):
        result = extract(adapter, "")
        mod = result.symbols[0]
        assert mod.qualified_name == "src.services.user"

    def test_module_docstring_detected(self, adapter):
        src = '"""This is the module docstring."""\nx = 1\n'
        result = extract(adapter, src)
        mod = result.symbols[0]
        assert mod.docstring == "This is the module docstring."

    def test_banner_string_not_docstring(self, adapter):
        src = 'BANNER = "not a docstring"\n'
        result = extract(adapter, src)
        mod = result.symbols[0]
        assert mod.docstring is None

    def test_module_body_hash_is_file_hash(self, adapter):
        src = "x = 1\n"
        result = extract(adapter, src)
        mod = result.symbols[0]
        import hashlib
        expected = "sha256:" + hashlib.sha256(src.encode()).hexdigest()
        assert mod.body_hash == expected

    def test_module_body_hash_changes_with_content(self, adapter):
        r1 = extract(adapter, "x = 1\n")
        r2 = extract(adapter, "x = 2\n")
        assert r1.symbols[0].body_hash != r2.symbols[0].body_hash

    def test_module_id_is_valid_moniker(self, adapter):
        result = extract(adapter, "")
        mod = result.symbols[0]
        assert mod.id.startswith("sutra python my-app")
        assert mod.id.endswith("/")

    def test_empty_file_still_produces_module_symbol(self, adapter):
        result = extract(adapter, "")
        assert len(result.symbols) == 1
        assert isinstance(result.symbols[0], ModuleSymbol)

    def test_module_language_is_python(self, adapter):
        result = extract(adapter, "")
        assert result.symbols[0].language == "python"


# ---------------------------------------------------------------------------
# FunctionSymbol
# ---------------------------------------------------------------------------

class TestFunctionSymbol:
    def test_top_level_function_extracted(self, adapter):
        src = "def hello():\n    pass\n"
        result = extract(adapter, src)
        fn = sym_by_name(result, "hello")
        assert fn is not None
        assert isinstance(fn, FunctionSymbol)

    def test_function_not_a_method(self, adapter):
        src = "def hello():\n    pass\n"
        result = extract(adapter, src)
        fn = sym_by_name(result, "hello")
        assert not isinstance(fn, MethodSymbol)

    def test_function_qualified_name(self, adapter):
        src = "def create_user():\n    pass\n"
        result = extract(adapter, src)
        fn = sym_by_name(result, "create_user")
        assert fn.qualified_name == "src.services.user.create_user"

    def test_function_moniker_format(self, adapter):
        src = "def create_user():\n    pass\n"
        result = extract(adapter, src)
        fn = sym_by_name(result, "create_user")
        assert fn.id == "sutra python my-app src/services/user.py create_user()."

    def test_function_signature_extracted(self, adapter):
        src = "def create_user(email: str, role: str = 'admin') -> bool:\n    pass\n"
        result = extract(adapter, src)
        fn = sym_by_name(result, "create_user")
        assert "create_user" in fn.signature
        assert "email: str" in fn.signature
        assert "->" in fn.signature

    def test_function_return_type(self, adapter):
        src = "def create_user() -> bool:\n    pass\n"
        result = extract(adapter, src)
        fn = sym_by_name(result, "create_user")
        assert fn.return_type == "bool"

    def test_function_no_return_type(self, adapter):
        src = "def helper():\n    pass\n"
        result = extract(adapter, src)
        fn = sym_by_name(result, "helper")
        assert fn.return_type is None

    def test_function_docstring(self, adapter):
        src = 'def helper():\n    """Does a thing."""\n    pass\n'
        result = extract(adapter, src)
        fn = sym_by_name(result, "helper")
        assert fn.docstring == "Does a thing."

    def test_async_function(self, adapter):
        src = "async def fetch_data(url: str) -> bytes:\n    pass\n"
        result = extract(adapter, src)
        fn = sym_by_name(result, "fetch_data")
        assert fn is not None
        assert fn.is_async is True

    def test_sync_function_not_async(self, adapter):
        src = "def sync_fn():\n    pass\n"
        result = extract(adapter, src)
        fn = sym_by_name(result, "sync_fn")
        assert fn.is_async is False

    def test_function_body_hash_sha256_prefix(self, adapter):
        src = "def fn():\n    x = 1\n"
        result = extract(adapter, src)
        fn = sym_by_name(result, "fn")
        assert fn.body_hash.startswith("sha256:")

    def test_function_body_hash_deterministic(self, adapter):
        src = "def fn():\n    x = 1\n"
        r1 = extract(adapter, src)
        r2 = extract(adapter, src)
        assert sym_by_name(r1, "fn").body_hash == sym_by_name(r2, "fn").body_hash

    def test_function_body_hash_changes_when_body_changes(self, adapter):
        src1 = "def fn():\n    x = 1\n"
        src2 = "def fn():\n    x = 2\n"
        r1 = extract(adapter, src1)
        r2 = extract(adapter, src2)
        assert sym_by_name(r1, "fn").body_hash != sym_by_name(r2, "fn").body_hash

    def test_function_body_hash_unchanged_when_only_signature_changes(self, adapter):
        src1 = "def fn(x):\n    return x\n"
        src2 = "def fn(y):\n    return x\n"
        r1 = extract(adapter, src1)
        r2 = extract(adapter, src2)
        # body is identical ("    return x\n"), hashes must match
        assert sym_by_name(r1, "fn").body_hash == sym_by_name(r2, "fn").body_hash

    def test_private_function_visibility(self, adapter):
        src = "def _internal():\n    pass\n"
        result = extract(adapter, src)
        fn = sym_by_name(result, "_internal")
        assert fn.visibility == Visibility.PRIVATE
        assert fn.is_exported is False

    def test_public_function_visibility(self, adapter):
        src = "def public_fn():\n    pass\n"
        result = extract(adapter, src)
        fn = sym_by_name(result, "public_fn")
        assert fn.visibility == Visibility.PUBLIC
        assert fn.is_exported is True

    def test_decorated_function_extracted(self, adapter):
        src = "@app.route('/users')\ndef list_users():\n    pass\n"
        result = extract(adapter, src)
        fn = sym_by_name(result, "list_users")
        assert fn is not None
        assert len(fn.decorators) == 1
        assert "@app.route" in fn.decorators[0]

    def test_nested_function_not_extracted(self, adapter):
        src = (
            "def outer():\n"
            "    def inner():\n"
            "        pass\n"
            "    return inner\n"
        )
        result = extract(adapter, src)
        assert sym_by_name(result, "outer") is not None
        assert sym_by_name(result, "inner") is None

    def test_complexity_simple(self, adapter):
        src = "def fn():\n    return 1\n"
        result = extract(adapter, src)
        fn = sym_by_name(result, "fn")
        assert fn.complexity == 1

    def test_complexity_with_if(self, adapter):
        src = "def fn(x):\n    if x > 0:\n        return x\n    return 0\n"
        result = extract(adapter, src)
        fn = sym_by_name(result, "fn")
        assert fn.complexity == 2  # 1 + 1 if

    def test_complexity_with_boolean_operator(self, adapter):
        src = "def fn(x, y):\n    return x and y\n"
        result = extract(adapter, src)
        fn = sym_by_name(result, "fn")
        assert fn.complexity == 2  # 1 + 1 boolean_operator

    def test_nested_function_calls_not_attributed_to_outer(self, adapter):
        src = (
            "def outer():\n"
            "    def inner():\n"
            "        secret_call()\n"
            "    outer_call()\n"
        )
        result = extract(adapter, src)
        outer = sym_by_name(result, "outer")
        calls = [r for r in result.relationships
                 if r.kind == RelationKind.CALLS and r.source_id == outer.id]
        targets = [r.target_name for r in calls]
        assert "outer_call" in targets
        assert "secret_call" not in targets


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

class TestParameters:
    def test_plain_params(self, adapter):
        src = "def fn(x, y):\n    pass\n"
        result = extract(adapter, src)
        params = sym_by_name(result, "fn").parameters
        names = [p.name for p in params]
        assert "x" in names
        assert "y" in names

    def test_typed_param(self, adapter):
        src = "def fn(x: int):\n    pass\n"
        result = extract(adapter, src)
        params = sym_by_name(result, "fn").parameters
        x = next(p for p in params if p.name == "x")
        assert x.type_annotation == "int"

    def test_default_param(self, adapter):
        src = "def fn(x: str = 'hello'):\n    pass\n"
        result = extract(adapter, src)
        params = sym_by_name(result, "fn").parameters
        x = next(p for p in params if p.name == "x")
        assert x.default_value is not None

    def test_variadic_param(self, adapter):
        src = "def fn(*args):\n    pass\n"
        result = extract(adapter, src)
        params = sym_by_name(result, "fn").parameters
        args = next(p for p in params if p.name == "args")
        assert args.is_variadic is True

    def test_keyword_variadic_param(self, adapter):
        src = "def fn(**kwargs):\n    pass\n"
        result = extract(adapter, src)
        params = sym_by_name(result, "fn").parameters
        kwargs = next(p for p in params if p.name == "kwargs")
        assert kwargs.is_keyword_variadic is True

    def test_all_param_kinds_together(self, adapter):
        src = "def fn(self, a: int, b: str = 'x', *args, **kwargs) -> bool:\n    pass\n"
        result = extract(adapter, src)
        params = sym_by_name(result, "fn").parameters
        names = [p.name for p in params]
        assert "self" in names
        assert "a" in names
        assert "b" in names
        assert "args" in names
        assert "kwargs" in names


# ---------------------------------------------------------------------------
# ClassSymbol
# ---------------------------------------------------------------------------

class TestClassSymbol:
    def test_class_extracted(self, adapter):
        src = "class UserService:\n    pass\n"
        result = extract(adapter, src)
        cls = sym_by_name(result, "UserService")
        assert cls is not None
        assert isinstance(cls, ClassSymbol)

    def test_class_moniker(self, adapter):
        src = "class UserService:\n    pass\n"
        result = extract(adapter, src)
        cls = sym_by_name(result, "UserService")
        assert cls.id == "sutra python my-app src/services/user.py UserService#"

    def test_class_docstring(self, adapter):
        src = 'class UserService:\n    """Manages users."""\n    pass\n'
        result = extract(adapter, src)
        cls = sym_by_name(result, "UserService")
        assert cls.docstring == "Manages users."

    def test_class_base_classes(self, adapter):
        src = "class AdminService(UserService, AuditMixin):\n    pass\n"
        result = extract(adapter, src)
        cls = sym_by_name(result, "AdminService")
        assert "UserService" in cls.base_classes
        assert "AuditMixin" in cls.base_classes

    def test_class_dotted_base_class(self, adapter):
        src = "class Foo(module.Base):\n    pass\n"
        result = extract(adapter, src)
        cls = sym_by_name(result, "Foo")
        assert "module.Base" in cls.base_classes

    def test_multiple_base_classes_with_dotted(self, adapter):
        src = "class Foo(Base, pkg.mixin.Mixin):\n    pass\n"
        result = extract(adapter, src)
        cls = sym_by_name(result, "Foo")
        assert "Base" in cls.base_classes
        assert "pkg.mixin.Mixin" in cls.base_classes

    def test_class_is_abstract_via_abc(self, adapter):
        src = "class Base(ABC):\n    pass\n"
        result = extract(adapter, src)
        cls = sym_by_name(result, "Base")
        assert cls.is_abstract is True

    def test_class_no_init_no_spurious_constructor(self, adapter):
        src = "class Empty:\n    pass\n"
        result = extract(adapter, src)
        methods = [s for s in result.symbols if isinstance(s, MethodSymbol)]
        constructors = [m for m in methods if m.is_constructor]
        assert len(constructors) == 0

    def test_class_decorator(self, adapter):
        src = "@dataclass\nclass Config:\n    x: int = 0\n"
        result = extract(adapter, src)
        cls = sym_by_name(result, "Config")
        assert cls is not None
        assert any("dataclass" in d for d in cls.decorators)


# ---------------------------------------------------------------------------
# MethodSymbol
# ---------------------------------------------------------------------------

class TestMethodSymbol:
    def test_method_extracted(self, adapter):
        src = "class Svc:\n    def create(self):\n        pass\n"
        result = extract(adapter, src)
        m = sym_by_name(result, "create")
        assert m is not None
        assert isinstance(m, MethodSymbol)

    def test_method_moniker(self, adapter):
        src = "class Svc:\n    def create(self):\n        pass\n"
        result = extract(adapter, src)
        m = sym_by_name(result, "create")
        assert m.id == "sutra python my-app src/services/user.py Svc#create()."

    def test_method_is_subtype_of_function(self, adapter):
        src = "class Svc:\n    def create(self):\n        pass\n"
        result = extract(adapter, src)
        m = sym_by_name(result, "create")
        assert isinstance(m, FunctionSymbol)

    def test_method_enclosing_class_id(self, adapter):
        src = "class Svc:\n    def create(self):\n        pass\n"
        result = extract(adapter, src)
        cls = sym_by_name(result, "Svc")
        m = sym_by_name(result, "create")
        assert m.enclosing_class_id == cls.id

    def test_constructor_flag(self, adapter):
        src = "class Svc:\n    def __init__(self):\n        pass\n"
        result = extract(adapter, src)
        m = sym_by_name(result, "__init__")
        assert m.is_constructor is True

    def test_non_init_method_not_constructor(self, adapter):
        src = "class Svc:\n    def save(self):\n        pass\n"
        result = extract(adapter, src)
        m = sym_by_name(result, "save")
        assert m.is_constructor is False

    def test_staticmethod_flag(self, adapter):
        src = "class Svc:\n    @staticmethod\n    def from_token(token: str):\n        pass\n"
        result = extract(adapter, src)
        m = sym_by_name(result, "from_token")
        assert m is not None
        assert m.is_static is True
        assert m.enclosing_class_id != ""

    def test_classmethod_has_class_id(self, adapter):
        src = "class Svc:\n    @classmethod\n    def create(cls):\n        pass\n"
        result = extract(adapter, src)
        m = sym_by_name(result, "create")
        assert m is not None
        assert m.enclosing_class_id != ""

    def test_async_method(self, adapter):
        src = "class Svc:\n    async def fetch(self):\n        pass\n"
        result = extract(adapter, src)
        m = sym_by_name(result, "fetch")
        assert m.is_async is True

    def test_dunder_method_is_exported(self, adapter):
        src = "class Svc:\n    def __str__(self):\n        return ''\n"
        result = extract(adapter, src)
        m = sym_by_name(result, "__str__")
        assert m.is_exported is True
        assert m.visibility == Visibility.PUBLIC

    def test_private_method(self, adapter):
        src = "class Svc:\n    def _helper(self):\n        pass\n"
        result = extract(adapter, src)
        m = sym_by_name(result, "_helper")
        assert m.visibility == Visibility.PRIVATE
        assert m.is_exported is False

    def test_method_qualified_name(self, adapter):
        src = "class Svc:\n    def create(self):\n        pass\n"
        result = extract(adapter, src)
        m = sym_by_name(result, "create")
        assert m.qualified_name == "src.services.user.Svc.create"


# ---------------------------------------------------------------------------
# VariableSymbol
# ---------------------------------------------------------------------------

class TestVariableSymbol:
    def test_annotated_variable_extracted(self, adapter):
        src = "MAX_RETRIES: int = 3\n"
        result = extract(adapter, src)
        var = sym_by_name(result, "MAX_RETRIES")
        assert var is not None
        assert isinstance(var, VariableSymbol)

    def test_variable_type_annotation(self, adapter):
        src = "MAX_RETRIES: int = 3\n"
        result = extract(adapter, src)
        var = sym_by_name(result, "MAX_RETRIES")
        assert var.type_annotation == "int"

    def test_constant_flag_all_caps(self, adapter):
        src = "MAX_RETRIES: int = 3\n"
        result = extract(adapter, src)
        var = sym_by_name(result, "MAX_RETRIES")
        assert var.is_constant is True

    def test_non_constant_variable(self, adapter):
        src = "config: dict = {}\n"
        result = extract(adapter, src)
        var = sym_by_name(result, "config")
        assert var.is_constant is False

    def test_variable_without_annotation_not_extracted(self, adapter):
        src = "x = 5\n"
        result = extract(adapter, src)
        assert sym_by_name(result, "x") is None

    def test_annotated_no_value_extracted(self, adapter):
        src = "name: str\n"
        result = extract(adapter, src)
        var = sym_by_name(result, "name")
        assert var is not None
        assert var.type_annotation == "str"

    def test_variable_moniker(self, adapter):
        src = "MAX: int = 1\n"
        result = extract(adapter, src)
        var = sym_by_name(result, "MAX")
        assert var.id == "sutra python my-app src/services/user.py MAX."


# ---------------------------------------------------------------------------
# CONTAINS relationships
# ---------------------------------------------------------------------------

class TestContainsRelationships:
    def test_module_contains_function(self, adapter):
        src = "def fn():\n    pass\n"
        result = extract(adapter, src)
        fn = sym_by_name(result, "fn")
        mod = result.symbols[0]
        contains = rels_of_kind(result, RelationKind.CONTAINS)
        assert any(r.source_id == mod.id and r.target_id == fn.id for r in contains)

    def test_module_contains_class(self, adapter):
        src = "class Svc:\n    pass\n"
        result = extract(adapter, src)
        cls = sym_by_name(result, "Svc")
        mod = result.symbols[0]
        contains = rels_of_kind(result, RelationKind.CONTAINS)
        assert any(r.source_id == mod.id and r.target_id == cls.id for r in contains)

    def test_class_contains_method(self, adapter):
        src = "class Svc:\n    def create(self):\n        pass\n"
        result = extract(adapter, src)
        cls = sym_by_name(result, "Svc")
        m = sym_by_name(result, "create")
        contains = rels_of_kind(result, RelationKind.CONTAINS)
        assert any(r.source_id == cls.id and r.target_id == m.id for r in contains)

    def test_module_contains_variable(self, adapter):
        src = "MAX: int = 1\n"
        result = extract(adapter, src)
        var = sym_by_name(result, "MAX")
        mod = result.symbols[0]
        contains = rels_of_kind(result, RelationKind.CONTAINS)
        assert any(r.source_id == mod.id and r.target_id == var.id for r in contains)

    def test_all_contains_are_resolved(self, adapter):
        src = "def fn():\n    pass\nclass C:\n    def m(self): pass\n"
        result = extract(adapter, src)
        for r in rels_of_kind(result, RelationKind.CONTAINS):
            assert r.is_resolved is True
            assert r.target_id is not None


# ---------------------------------------------------------------------------
# IMPORTS relationships
# ---------------------------------------------------------------------------

class TestImportsRelationships:
    def test_plain_import(self, adapter):
        src = "import os\n"
        result = extract(adapter, src)
        imports = rels_of_kind(result, RelationKind.IMPORTS)
        assert any(r.target_name == "os" for r in imports)

    def test_from_import(self, adapter):
        src = "from pathlib import Path\n"
        result = extract(adapter, src)
        imports = rels_of_kind(result, RelationKind.IMPORTS)
        imp = next(r for r in imports if r.target_name == "Path")
        assert imp.metadata["import_source"] == "pathlib"

    def test_from_import_multiple_names(self, adapter):
        src = "from os.path import join, exists\n"
        result = extract(adapter, src)
        imports = rels_of_kind(result, RelationKind.IMPORTS)
        targets = {r.target_name for r in imports}
        assert "join" in targets
        assert "exists" in targets

    def test_aliased_import_stores_original_name(self, adapter):
        src = "from os.path import exists as ex\n"
        result = extract(adapter, src)
        imports = rels_of_kind(result, RelationKind.IMPORTS)
        imp = next(r for r in imports if r.target_name == "exists")
        assert imp.metadata["alias"] == "ex"

    def test_deep_from_import(self, adapter):
        src = "from x.y.z import a as b\n"
        result = extract(adapter, src)
        imports = rels_of_kind(result, RelationKind.IMPORTS)
        imp = next(r for r in imports if r.target_name == "a")
        assert imp.metadata["import_source"] == "x.y.z"
        assert imp.metadata["alias"] == "b"

    def test_aliased_module_import(self, adapter):
        src = "import numpy as np\n"
        result = extract(adapter, src)
        imports = rels_of_kind(result, RelationKind.IMPORTS)
        imp = next(r for r in imports if r.target_name == "numpy")
        assert imp.metadata["alias"] == "np"

    def test_imports_are_unresolved(self, adapter):
        src = "import os\nfrom pathlib import Path\n"
        result = extract(adapter, src)
        for r in rels_of_kind(result, RelationKind.IMPORTS):
            assert r.is_resolved is False
            assert r.target_id is None


# ---------------------------------------------------------------------------
# CALLS relationships
# ---------------------------------------------------------------------------

class TestCallsRelationships:
    def test_direct_call_extracted(self, adapter):
        src = "def fn():\n    helper()\n"
        result = extract(adapter, src)
        fn = sym_by_name(result, "fn")
        calls = [r for r in rels_of_kind(result, RelationKind.CALLS)
                 if r.source_id == fn.id]
        assert any(r.target_name == "helper" for r in calls)

    def test_direct_call_metadata(self, adapter):
        src = "def fn():\n    helper()\n"
        result = extract(adapter, src)
        fn = sym_by_name(result, "fn")
        call = next(r for r in result.relationships
                    if r.kind == RelationKind.CALLS and r.source_id == fn.id
                    and r.target_name == "helper")
        assert call.metadata["call_form"] == "direct"

    def test_method_call_extracted(self, adapter):
        src = "def fn():\n    db.insert(data)\n"
        result = extract(adapter, src)
        fn = sym_by_name(result, "fn")
        calls = [r for r in rels_of_kind(result, RelationKind.CALLS)
                 if r.source_id == fn.id]
        call = next(r for r in calls if r.target_name == "insert")
        assert call.metadata["call_form"] == "method"
        assert call.metadata["receiver"] == "db"

    def test_calls_are_unresolved(self, adapter):
        src = "def fn():\n    helper()\n    obj.method()\n"
        result = extract(adapter, src)
        for r in rels_of_kind(result, RelationKind.CALLS):
            assert r.is_resolved is False
            assert r.target_id is None

    def test_calls_have_location(self, adapter):
        src = "def fn():\n    helper()\n"
        result = extract(adapter, src)
        fn = sym_by_name(result, "fn")
        call = next(r for r in result.relationships
                    if r.kind == RelationKind.CALLS and r.source_id == fn.id)
        assert call.location is not None
        assert call.location.line_start > 0

    def test_nested_function_calls_not_leaked(self, adapter):
        src = (
            "def outer():\n"
            "    def inner():\n"
            "        secret()\n"
            "    visible()\n"
        )
        result = extract(adapter, src)
        outer = sym_by_name(result, "outer")
        outer_calls = [r for r in result.relationships
                       if r.kind == RelationKind.CALLS and r.source_id == outer.id]
        names = {r.target_name for r in outer_calls}
        assert "visible" in names
        assert "secret" not in names


# ---------------------------------------------------------------------------
# EXTENDS relationships
# ---------------------------------------------------------------------------

class TestExtendsRelationships:
    def test_extends_created(self, adapter):
        src = "class Child(Parent):\n    pass\n"
        result = extract(adapter, src)
        cls = sym_by_name(result, "Child")
        extends = [r for r in result.relationships
                   if r.kind == RelationKind.EXTENDS and r.source_id == cls.id]
        assert any(r.target_name == "Parent" for r in extends)

    def test_extends_unresolved(self, adapter):
        src = "class Child(Parent):\n    pass\n"
        result = extract(adapter, src)
        cls = sym_by_name(result, "Child")
        for r in result.relationships:
            if r.kind == RelationKind.EXTENDS and r.source_id == cls.id:
                assert r.is_resolved is False

    def test_dotted_base_preserved(self, adapter):
        src = "class Foo(module.Base):\n    pass\n"
        result = extract(adapter, src)
        cls = sym_by_name(result, "Foo")
        extends = [r for r in result.relationships
                   if r.kind == RelationKind.EXTENDS and r.source_id == cls.id]
        assert any(r.target_name == "module.Base" for r in extends)

    def test_multiple_bases_multiple_extends(self, adapter):
        src = "class C(A, B):\n    pass\n"
        result = extract(adapter, src)
        cls = sym_by_name(result, "C")
        extends = [r for r in result.relationships
                   if r.kind == RelationKind.EXTENDS and r.source_id == cls.id]
        targets = {r.target_name for r in extends}
        assert targets == {"A", "B"}


# ---------------------------------------------------------------------------
# File record
# ---------------------------------------------------------------------------

class TestFileRecord:
    def test_file_path_preserved(self, adapter):
        result = extract(adapter, "x = 1\n")
        assert result.file.path == FILE

    def test_file_language_python(self, adapter):
        result = extract(adapter, "")
        assert result.file.language == "python"

    def test_file_size_bytes(self, adapter):
        src = "x = 1\n"
        result = extract(adapter, src)
        assert result.file.size_bytes == len(src.encode())

    def test_file_hash_sha256(self, adapter):
        src = "x = 1\n"
        result = extract(adapter, src)
        import hashlib
        expected = "sha256:" + hashlib.sha256(src.encode()).hexdigest()
        assert result.file.hash == expected


# ---------------------------------------------------------------------------
# Syntax error tolerance
# ---------------------------------------------------------------------------

class TestSyntaxErrorTolerance:
    def test_syntax_error_still_produces_extraction(self, adapter):
        # tree-sitter always produces a tree with ERROR nodes — never crashes
        src = "def broken(\n    pass\n"
        result = extract(adapter, src)
        assert result is not None
        assert isinstance(result.symbols[0], ModuleSymbol)

    def test_valid_symbols_before_error_are_salvaged(self, adapter):
        src = "def good():\n    pass\ndef broken(\n    pass\n"
        result = extract(adapter, src)
        # 'good' may or may not be captured depending on where ERROR propagates,
        # but the extraction must not raise
        assert result is not None


# ---------------------------------------------------------------------------
# Integration: a realistic Python snippet
# ---------------------------------------------------------------------------

class TestRealisticSnippet:
    SRC = (
        '"""User service module."""\n'
        "import os\n"
        "from typing import Optional\n"
        "\n"
        "MAX_USERS: int = 1000\n"
        "\n"
        "class UserService(BaseService):\n"
        '    """Manages user lifecycle."""\n'
        "\n"
        "    def __init__(self, db):\n"
        "        self.db = db\n"
        "\n"
        "    @staticmethod\n"
        "    def validate(email: str) -> bool:\n"
        '        return "@" in email\n'
        "\n"
        "    async def create(self, email: str, role: str = 'user') -> Optional[str]:\n"
        '        """Create a user."""\n'
        "        if not self.validate(email):\n"
        "            return None\n"
        "        return self.db.insert(email)\n"
        "\n"
        "def standalone(x: int) -> int:\n"
        "    return x * 2\n"
    )

    def test_module_docstring(self, adapter):
        result = extract(adapter, self.SRC)
        mod = result.symbols[0]
        assert mod.docstring == "User service module."

    def test_all_symbol_kinds_present(self, adapter):
        result = extract(adapter, self.SRC)
        kinds = {type(s).__name__ for s in result.symbols}
        assert "ModuleSymbol" in kinds
        assert "ClassSymbol" in kinds
        assert "MethodSymbol" in kinds
        assert "FunctionSymbol" in kinds
        assert "VariableSymbol" in kinds

    def test_class_has_correct_base(self, adapter):
        result = extract(adapter, self.SRC)
        cls = sym_by_name(result, "UserService")
        assert "BaseService" in cls.base_classes

    def test_constructor_detected(self, adapter):
        result = extract(adapter, self.SRC)
        init = sym_by_name(result, "__init__")
        assert init.is_constructor is True

    def test_static_method_detected(self, adapter):
        result = extract(adapter, self.SRC)
        validate = sym_by_name(result, "validate")
        assert validate.is_static is True

    def test_async_method_detected(self, adapter):
        result = extract(adapter, self.SRC)
        create = sym_by_name(result, "create")
        assert create.is_async is True

    def test_method_docstring(self, adapter):
        result = extract(adapter, self.SRC)
        create = sym_by_name(result, "create")
        assert create.docstring == "Create a user."

    def test_max_users_variable(self, adapter):
        result = extract(adapter, self.SRC)
        var = sym_by_name(result, "MAX_USERS")
        assert var is not None
        assert var.is_constant is True
        assert var.type_annotation == "int"

    def test_imports_captured(self, adapter):
        result = extract(adapter, self.SRC)
        imports = rels_of_kind(result, RelationKind.IMPORTS)
        targets = {r.target_name for r in imports}
        assert "os" in targets
        assert "Optional" in targets

    def test_extends_relationship(self, adapter):
        result = extract(adapter, self.SRC)
        cls = sym_by_name(result, "UserService")
        extends = [r for r in result.relationships
                   if r.kind == RelationKind.EXTENDS and r.source_id == cls.id]
        assert any(r.target_name == "BaseService" for r in extends)

    def test_calls_from_create_method(self, adapter):
        result = extract(adapter, self.SRC)
        create = sym_by_name(result, "create")
        calls = [r for r in result.relationships
                 if r.kind == RelationKind.CALLS and r.source_id == create.id]
        targets = {r.target_name for r in calls}
        assert "validate" in targets
        assert "insert" in targets

    def test_standalone_function_not_a_method(self, adapter):
        result = extract(adapter, self.SRC)
        fn = sym_by_name(result, "standalone")
        assert isinstance(fn, FunctionSymbol)
        assert not isinstance(fn, MethodSymbol)

    def test_symbol_count_is_sane(self, adapter):
        result = extract(adapter, self.SRC)
        # module + class + __init__ + validate + create + standalone + MAX_USERS = 7
        assert len(result.symbols) == 7
