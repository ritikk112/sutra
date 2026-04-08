"""
Tests for sutra/core/extractor/base.py

Run from the repo root with:
    python -m pytest tests/test_base.py -v
"""
import pytest
from datetime import datetime

from sutra.core.extractor.base import (
    Location, Parameter,
    Visibility, RelationKind,
    SymbolBase, FunctionSymbol, ClassSymbol, MethodSymbol, VariableSymbol, ModuleSymbol,
    Relationship, File, Repository, FileExtraction, IndexResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_location() -> Location:
    return Location(line_start=1, line_end=5, byte_start=0, byte_end=120)


def make_base_fields(name: str = "my_func") -> dict:
    return dict(
        id=f"sutra python my-app src/foo.py {name}().",
        name=name,
        qualified_name=f"src.foo.{name}",
        file_path="src/foo.py",
        location=make_location(),
        body_hash="sha256:abc123",
        language="python",
        visibility=Visibility.PUBLIC,
        is_exported=True,
    )


# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------

class TestLocation:
    def test_required_fields(self):
        loc = Location(line_start=1, line_end=10, byte_start=0, byte_end=200)
        assert loc.line_start == 1
        assert loc.line_end == 10
        assert loc.byte_start == 0
        assert loc.byte_end == 200

    def test_column_defaults_to_zero(self):
        loc = Location(line_start=1, line_end=1, byte_start=0, byte_end=10)
        assert loc.column_start == 0
        assert loc.column_end == 0

    def test_explicit_columns(self):
        loc = Location(line_start=3, line_end=3, byte_start=50, byte_end=80, column_start=4, column_end=34)
        assert loc.column_start == 4
        assert loc.column_end == 34


# ---------------------------------------------------------------------------
# Parameter
# ---------------------------------------------------------------------------

class TestParameter:
    def test_name_only(self):
        p = Parameter(name="email")
        assert p.name == "email"
        assert p.type_annotation is None
        assert p.default_value is None
        assert p.is_variadic is False
        assert p.is_keyword_variadic is False

    def test_typed_parameter(self):
        p = Parameter(name="role", type_annotation="Role", default_value="Role.USER")
        assert p.type_annotation == "Role"
        assert p.default_value == "Role.USER"

    def test_variadic(self):
        p = Parameter(name="args", is_variadic=True)
        assert p.is_variadic is True
        assert p.is_keyword_variadic is False

    def test_keyword_variadic(self):
        p = Parameter(name="kwargs", is_keyword_variadic=True)
        assert p.is_keyword_variadic is True
        assert p.is_variadic is False


# ---------------------------------------------------------------------------
# Visibility enum
# ---------------------------------------------------------------------------

class TestVisibility:
    def test_string_values(self):
        assert Visibility.PUBLIC == "public"
        assert Visibility.PRIVATE == "private"
        assert Visibility.PROTECTED == "protected"
        assert Visibility.INTERNAL == "internal"

    def test_is_str_subclass(self):
        assert isinstance(Visibility.PUBLIC, str)


# ---------------------------------------------------------------------------
# RelationKind enum
# ---------------------------------------------------------------------------

class TestRelationKind:
    def test_all_kinds_present(self):
        kinds = {k.value for k in RelationKind}
        assert kinds == {
            "calls", "extends", "implements", "imports",
            "contains", "references", "returns_type", "parameter_type",
        }

    def test_is_str_subclass(self):
        assert isinstance(RelationKind.CALLS, str)


# ---------------------------------------------------------------------------
# FunctionSymbol
# ---------------------------------------------------------------------------

class TestFunctionSymbol:
    def test_minimal_construction(self):
        fn = FunctionSymbol(**make_base_fields("helper"))
        assert fn.name == "helper"
        assert fn.signature == ""
        assert fn.parameters == []
        assert fn.return_type is None
        assert fn.docstring is None
        assert fn.decorators == []
        assert fn.is_async is False
        assert fn.complexity is None

    def test_full_construction(self):
        params = [Parameter("email", "str"), Parameter("role", "Role", "Role.USER")]
        fn = FunctionSymbol(
            **make_base_fields("create_user"),
            signature="async def create_user(email: str, role: Role = Role.USER) -> User",
            parameters=params,
            return_type="User",
            docstring="Creates a user.",
            decorators=["@app.post"],
            is_async=True,
            complexity=3,
        )
        assert fn.is_async is True
        assert fn.complexity == 3
        assert len(fn.parameters) == 2
        assert fn.return_type == "User"

    def test_not_a_method(self):
        fn = FunctionSymbol(**make_base_fields("standalone"))
        assert not isinstance(fn, MethodSymbol)
        assert isinstance(fn, FunctionSymbol)


# ---------------------------------------------------------------------------
# ClassSymbol
# ---------------------------------------------------------------------------

class TestClassSymbol:
    def test_minimal_construction(self):
        cls = ClassSymbol(**make_base_fields("UserService"))
        assert cls.base_classes == []
        assert cls.decorators == []
        assert cls.is_abstract is False
        assert cls.docstring is None

    def test_with_inheritance(self):
        cls = ClassSymbol(
            **make_base_fields("AdminService"),
            base_classes=["UserService", "AuditMixin"],
            is_abstract=True,
        )
        assert "UserService" in cls.base_classes
        assert cls.is_abstract is True


# ---------------------------------------------------------------------------
# MethodSymbol
# ---------------------------------------------------------------------------

class TestMethodSymbol:
    def test_is_subtype_of_function(self):
        method = MethodSymbol(
            **make_base_fields("create_user"),
            enclosing_class_id="sutra python my-app src/foo.py UserService#",
        )
        assert isinstance(method, FunctionSymbol)
        assert isinstance(method, MethodSymbol)
        assert not isinstance(method, ClassSymbol)

    def test_constructor_flag(self):
        method = MethodSymbol(
            **make_base_fields("__init__"),
            enclosing_class_id="sutra python my-app src/foo.py UserService#",
            is_constructor=True,
        )
        assert method.is_constructor is True

    def test_static_method(self):
        method = MethodSymbol(
            **make_base_fields("from_token"),
            enclosing_class_id="sutra python my-app src/foo.py UserService#",
            is_static=True,
        )
        assert method.is_static is True

    def test_default_enclosing_class_id(self):
        method = MethodSymbol(**make_base_fields("do_thing"))
        assert method.enclosing_class_id is None


# ---------------------------------------------------------------------------
# VariableSymbol
# ---------------------------------------------------------------------------

class TestVariableSymbol:
    def test_constant(self):
        var = VariableSymbol(
            **make_base_fields("MAX_RETRIES"),
            type_annotation="int",
            is_constant=True,
        )
        assert var.is_constant is True
        assert var.type_annotation == "int"

    def test_unannotated(self):
        var = VariableSymbol(**make_base_fields("config"))
        assert var.type_annotation is None
        assert var.is_constant is False


# ---------------------------------------------------------------------------
# ModuleSymbol
# ---------------------------------------------------------------------------

class TestModuleSymbol:
    def test_one_per_file(self):
        mod = ModuleSymbol(
            **make_base_fields("user"),
            docstring="User service module.",
        )
        assert mod.docstring == "User service module."

    def test_no_docstring(self):
        mod = ModuleSymbol(**make_base_fields("utils"))
        assert mod.docstring is None


# ---------------------------------------------------------------------------
# isinstance dispatch order (MethodSymbol before FunctionSymbol)
# ---------------------------------------------------------------------------

class TestSymbolDispatch:
    def test_method_caught_before_function(self):
        method = MethodSymbol(
            **make_base_fields("save"),
            enclosing_class_id="sutra python my-app src/foo.py Model#",
        )
        # Correct dispatch order: check MethodSymbol first
        if isinstance(method, MethodSymbol):
            kind = "method"
        elif isinstance(method, FunctionSymbol):
            kind = "function"
        else:
            kind = "other"
        assert kind == "method"

    def test_function_not_caught_as_method(self):
        fn = FunctionSymbol(**make_base_fields("standalone"))
        assert not isinstance(fn, MethodSymbol)
        assert isinstance(fn, FunctionSymbol)


# ---------------------------------------------------------------------------
# Relationship
# ---------------------------------------------------------------------------

class TestRelationship:
    def test_resolved(self):
        rel = Relationship(
            source_id="sutra python my-app src/a.py handler().",
            kind=RelationKind.CALLS,
            is_resolved=True,
            target_id="sutra python my-app src/b.py create_user().",
            target_name="create_user",
        )
        assert rel.is_resolved is True
        assert rel.target_id is not None

    def test_unresolved_cross_file(self):
        rel = Relationship(
            source_id="sutra python my-app src/a.py handler().",
            kind=RelationKind.CALLS,
            is_resolved=False,
            target_name="create_user",
            metadata={"import_source": "services.user", "call_form": "direct"},
        )
        assert rel.target_id is None
        assert rel.target_name == "create_user"
        assert rel.metadata["call_form"] == "direct"

    def test_method_call_metadata(self):
        rel = Relationship(
            source_id="sutra python my-app src/a.py handler().",
            kind=RelationKind.CALLS,
            is_resolved=False,
            target_name="insert",
            metadata={"call_form": "method", "receiver": "db"},
        )
        assert rel.metadata["receiver"] == "db"

    def test_contains_relationship(self):
        rel = Relationship(
            source_id="sutra python my-app src/foo.py user/",
            kind=RelationKind.CONTAINS,
            is_resolved=True,
            target_id="sutra python my-app src/foo.py UserService#",
        )
        assert rel.kind == RelationKind.CONTAINS

    def test_default_metadata_is_empty_dict(self):
        rel = Relationship(
            source_id="a",
            kind=RelationKind.IMPORTS,
            is_resolved=False,
            target_name="os",
        )
        assert rel.metadata == {}

    def test_metadata_not_shared_between_instances(self):
        r1 = Relationship(source_id="a", kind=RelationKind.CALLS, is_resolved=False)
        r2 = Relationship(source_id="b", kind=RelationKind.CALLS, is_resolved=False)
        r1.metadata["key"] = "value"
        assert "key" not in r2.metadata


# ---------------------------------------------------------------------------
# Pipeline container types
# ---------------------------------------------------------------------------

class TestFile:
    def test_construction(self):
        f = File(path="src/foo.py", language="python", size_bytes=1024, hash="sha256:abc")
        assert f.language == "python"
        assert f.size_bytes == 1024


class TestRepository:
    def test_construction(self):
        repo = Repository(url="https://github.com/org/my-app", name="my-app")
        assert repo.name == "my-app"


class TestFileExtraction:
    def test_construction(self):
        file = File("src/foo.py", "python", 512, "sha256:aaa")
        fn = FunctionSymbol(**make_base_fields("foo"))
        rel = Relationship(
            source_id=fn.id,
            kind=RelationKind.CALLS,
            is_resolved=False,
            target_name="bar",
        )
        extraction = FileExtraction(file=file, symbols=[fn], relationships=[rel])
        assert len(extraction.symbols) == 1
        assert len(extraction.relationships) == 1


class TestIndexResult:
    def test_construction_and_language_counts(self):
        repo = Repository("https://github.com/org/app", "app")
        file = File("src/foo.py", "python", 512, "sha256:aaa")
        fn = FunctionSymbol(**make_base_fields("foo"))
        result = IndexResult(
            repository=repo,
            files=[file],
            symbols=[fn],
            relationships=[],
            indexed_at=datetime.utcnow(),
            commit_hash="deadbeef",
            languages={"python": 1},
        )
        assert result.languages["python"] == 1
        assert result.commit_hash == "deadbeef"
        assert len(result.symbols) == 1
