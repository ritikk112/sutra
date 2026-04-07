"""
Tests for sutra/core/extractor/moniker.py

Run from repo root:
    python -m pytest tests/test_moniker.py -v
"""
import pytest

from sutra.core.extractor.moniker import (
    MonikerBuilder,
    MonikerComponents,
    repo_name_from_url,
    parse_moniker,
    descriptor_kind,
    SCHEME,
)


# ---------------------------------------------------------------------------
# repo_name_from_url
# ---------------------------------------------------------------------------

class TestRepoNameFromUrl:
    def test_https_with_dot_git(self):
        assert repo_name_from_url("https://github.com/org/my-app.git") == "my-app"

    def test_https_without_dot_git(self):
        assert repo_name_from_url("https://github.com/org/my-app") == "my-app"

    def test_ssh_shorthand(self):
        assert repo_name_from_url("git@github.com:org/my-app.git") == "my-app"

    def test_ssh_shorthand_no_dot_git(self):
        assert repo_name_from_url("git@github.com:org/my-app") == "my-app"

    def test_ssh_scheme(self):
        assert repo_name_from_url("ssh://git@github.com/org/my-app.git") == "my-app"

    def test_http(self):
        assert repo_name_from_url("http://github.com/org/my-app") == "my-app"

    def test_trailing_slash_stripped(self):
        assert repo_name_from_url("https://github.com/org/my-app/") == "my-app"

    def test_repo_name_with_dots(self):
        # repo names can legitimately contain dots
        assert repo_name_from_url("https://github.com/org/my.app") == "my.app"

    def test_repo_name_with_underscores(self):
        assert repo_name_from_url("https://github.com/org/my_app.git") == "my_app"

    def test_gitlab_url(self):
        assert repo_name_from_url("https://gitlab.com/group/subgroup/my-app.git") == "my-app"

    def test_self_hosted(self):
        assert repo_name_from_url("https://git.internal.io/team/sutra.git") == "sutra"


# ---------------------------------------------------------------------------
# MonikerBuilder — individual symbol kinds
# ---------------------------------------------------------------------------

class TestMonikerBuilderFunction:
    def setup_method(self):
        self.builder = MonikerBuilder(language="python", repo_name="my-app")

    def test_top_level_function(self):
        m = self.builder.for_function("src/services/user.py", "create_user")
        assert m == "sutra python my-app src/services/user.py create_user()."

    def test_descriptor_suffix_is_callable(self):
        m = self.builder.for_function("src/utils.py", "helper")
        assert m.endswith("helper().")

    def test_deeply_nested_path(self):
        m = self.builder.for_function("a/b/c/d/foo.py", "bar")
        assert m == "sutra python my-app a/b/c/d/foo.py bar()."

    def test_scheme_prefix(self):
        m = self.builder.for_function("src/foo.py", "fn")
        assert m.startswith(f"{SCHEME} python my-app")

    def test_dunder_function(self):
        # __init__ as a top-level factory function (not a method)
        m = self.builder.for_function("src/foo.py", "__init__")
        assert m.endswith("__init__().")


class TestMonikerBuilderMethod:
    def setup_method(self):
        self.builder = MonikerBuilder(language="python", repo_name="my-app")

    def test_standard_method(self):
        m = self.builder.for_method("src/services/user.py", "UserService", "create_user")
        assert m == "sutra python my-app src/services/user.py UserService#create_user()."

    def test_constructor(self):
        m = self.builder.for_method("src/foo.py", "UserService", "__init__")
        assert m == "sutra python my-app src/foo.py UserService#__init__()."

    def test_static_method_same_format(self):
        # is_static is metadata on MethodSymbol, not encoded in the moniker
        m = self.builder.for_method("src/foo.py", "UserService", "from_token")
        assert m.endswith("UserService#from_token().")

    def test_class_hash_separator_present(self):
        m = self.builder.for_method("src/foo.py", "MyClass", "my_method")
        assert "MyClass#my_method" in m

    def test_different_from_function(self):
        fn = self.builder.for_function("src/foo.py", "my_method")
        method = self.builder.for_method("src/foo.py", "MyClass", "my_method")
        assert fn != method


class TestMonikerBuilderClass:
    def setup_method(self):
        self.builder = MonikerBuilder(language="python", repo_name="my-app")

    def test_class_suffix(self):
        m = self.builder.for_class("src/services/user.py", "UserService")
        assert m == "sutra python my-app src/services/user.py UserService#"

    def test_ends_with_hash(self):
        m = self.builder.for_class("src/foo.py", "MyClass")
        assert m.endswith("MyClass#")

    def test_abstract_class_same_format(self):
        # is_abstract is metadata on ClassSymbol, not encoded in the moniker
        m = self.builder.for_class("src/foo.py", "BaseService")
        assert m.endswith("BaseService#")


class TestMonikerBuilderVariable:
    def setup_method(self):
        self.builder = MonikerBuilder(language="python", repo_name="my-app")

    def test_constant(self):
        m = self.builder.for_variable("src/config.py", "MAX_RETRIES")
        assert m == "sutra python my-app src/config.py MAX_RETRIES."

    def test_ends_with_dot(self):
        m = self.builder.for_variable("src/foo.py", "config")
        assert m.endswith("config.")

    def test_different_from_function(self):
        fn = self.builder.for_function("src/foo.py", "config")
        var = self.builder.for_variable("src/foo.py", "config")
        assert fn != var
        assert fn.endswith("().")
        assert var.endswith(".")
        assert not var.endswith("().")


class TestMonikerBuilderModule:
    def setup_method(self):
        self.builder = MonikerBuilder(language="python", repo_name="my-app")

    def test_module_trailing_slash(self):
        m = self.builder.for_module("src/services/user.py", "services/user")
        assert m == "sutra python my-app src/services/user.py services/user/"

    def test_file_path_preserved_separately(self):
        m = self.builder.for_module("src/services/user.py", "services/user")
        parts = m.split(" ")
        assert parts[3] == "src/services/user.py"
        assert parts[4] == "services/user/"

    def test_root_level_module(self):
        m = self.builder.for_module("main.py", "main")
        assert m == "sutra python my-app main.py main/"

    def test_deeply_nested_module(self):
        m = self.builder.for_module("src/a/b/c/mod.py", "a/b/c/mod")
        assert m.endswith("a/b/c/mod/")


# ---------------------------------------------------------------------------
# Language variants
# ---------------------------------------------------------------------------

class TestMonikerBuilderLanguages:
    def test_typescript(self):
        builder = MonikerBuilder(language="typescript", repo_name="frontend")
        m = builder.for_function("src/components/Button.tsx", "render")
        assert m == "sutra typescript frontend src/components/Button.tsx render()."

    def test_go(self):
        builder = MonikerBuilder(language="go", repo_name="backend")
        m = builder.for_function("internal/auth/jwt.go", "ValidateToken")
        assert m == "sutra go backend internal/auth/jwt.go ValidateToken()."

    def test_go_method(self):
        builder = MonikerBuilder(language="go", repo_name="backend")
        m = builder.for_method("internal/auth/jwt.go", "JWTService", "Validate")
        assert m == "sutra go backend internal/auth/jwt.go JWTService#Validate()."


# ---------------------------------------------------------------------------
# parse_moniker
# ---------------------------------------------------------------------------

class TestParseMoniker:
    def test_parse_function(self):
        raw = "sutra python my-app src/services/user.py create_user()."
        c = parse_moniker(raw)
        assert c.scheme == "sutra"
        assert c.language == "python"
        assert c.repo_name == "my-app"
        assert c.file_path == "src/services/user.py"
        assert c.descriptor == "create_user()."

    def test_parse_method(self):
        raw = "sutra python my-app src/services/user.py UserService#create_user()."
        c = parse_moniker(raw)
        assert c.descriptor == "UserService#create_user()."
        assert c.file_path == "src/services/user.py"

    def test_parse_class(self):
        raw = "sutra python my-app src/services/user.py UserService#"
        c = parse_moniker(raw)
        assert c.descriptor == "UserService#"

    def test_parse_variable(self):
        raw = "sutra python my-app src/config.py MAX_RETRIES."
        c = parse_moniker(raw)
        assert c.descriptor == "MAX_RETRIES."

    def test_parse_module(self):
        raw = "sutra python my-app src/services/user.py services/user/"
        c = parse_moniker(raw)
        assert c.descriptor == "services/user/"

    def test_round_trip_function(self):
        builder = MonikerBuilder(language="python", repo_name="my-app")
        original = builder.for_function("src/foo.py", "helper")
        parsed = parse_moniker(original)
        assert parsed.scheme == "sutra"
        assert parsed.language == "python"
        assert parsed.repo_name == "my-app"
        assert parsed.file_path == "src/foo.py"
        assert parsed.descriptor == "helper()."

    def test_round_trip_method(self):
        builder = MonikerBuilder(language="python", repo_name="my-app")
        original = builder.for_method("src/foo.py", "MyClass", "my_method")
        parsed = parse_moniker(original)
        assert parsed.descriptor == "MyClass#my_method()."

    def test_round_trip_class(self):
        builder = MonikerBuilder(language="go", repo_name="backend")
        original = builder.for_class("internal/auth.go", "JWTService")
        parsed = parse_moniker(original)
        assert parsed.language == "go"
        assert parsed.descriptor == "JWTService#"

    def test_round_trip_module(self):
        builder = MonikerBuilder(language="typescript", repo_name="frontend")
        original = builder.for_module("src/utils/format.ts", "utils/format")
        parsed = parse_moniker(original)
        assert parsed.descriptor == "utils/format/"

    def test_invalid_too_few_fields(self):
        with pytest.raises(ValueError, match="5 space-separated fields"):
            parse_moniker("sutra python my-app src/foo.py")

    def test_invalid_wrong_scheme(self):
        with pytest.raises(ValueError, match="scheme"):
            parse_moniker("scip python my-app src/foo.py func().")

    def test_invalid_empty_string(self):
        with pytest.raises(ValueError):
            parse_moniker("")


# ---------------------------------------------------------------------------
# descriptor_kind
# ---------------------------------------------------------------------------

class TestDescriptorKind:
    def test_top_level_function(self):
        assert descriptor_kind("helper_func().") == "function"

    def test_method(self):
        assert descriptor_kind("UserService#create_user().") == "method"

    def test_class(self):
        assert descriptor_kind("UserService#") == "class"

    def test_variable(self):
        assert descriptor_kind("MAX_RETRIES.") == "variable"

    def test_module(self):
        assert descriptor_kind("services/user/") == "module"

    def test_variable_not_confused_with_callable(self):
        # "config." ends with "." but NOT "()." — must be variable
        assert descriptor_kind("config.") == "variable"

    def test_callable_trailing_dot_not_confused_with_variable(self):
        # "fn()." ends with "." but the full suffix "()." takes precedence
        assert descriptor_kind("fn().") == "function"

    def test_nested_method_descriptor(self):
        assert descriptor_kind("Outer#Inner#method().") == "method"

    def test_unrecognised_suffix_raises(self):
        with pytest.raises(ValueError, match="Unrecognised descriptor suffix"):
            descriptor_kind("something_weird")

    def test_all_kinds_via_builder(self):
        builder = MonikerBuilder(language="python", repo_name="repo")

        fn = parse_moniker(builder.for_function("f.py", "fn")).descriptor
        method = parse_moniker(builder.for_method("f.py", "C", "m")).descriptor
        cls = parse_moniker(builder.for_class("f.py", "C")).descriptor
        var = parse_moniker(builder.for_variable("f.py", "V")).descriptor
        mod = parse_moniker(builder.for_module("f.py", "mod")).descriptor

        assert descriptor_kind(fn) == "function"
        assert descriptor_kind(method) == "method"
        assert descriptor_kind(cls) == "class"
        assert descriptor_kind(var) == "variable"
        assert descriptor_kind(mod) == "module"
