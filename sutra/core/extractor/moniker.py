from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Moniker format:
#   sutra <language> <repo_name> <file_path> <descriptor>
#
# Descriptor suffixes (atomic, not stacked):
#   ().   callable term  — functions and methods
#   #     class/type
#   .     non-callable term — variables, constants
#   /     namespace/module
# ---------------------------------------------------------------------------

SCHEME = "sutra"


@dataclass(frozen=True)
class MonikerComponents:
    """Parsed representation of a SCIP-style Sutra moniker."""
    scheme: str        # always "sutra"
    language: str      # "python" | "typescript" | "go"
    repo_name: str
    file_path: str     # repo-relative path, e.g. "src/services/user.py"
    descriptor: str    # e.g. "UserService#create_user()."


# ---------------------------------------------------------------------------
# URL → repo name
# ---------------------------------------------------------------------------

def repo_name_from_url(url: str) -> str:
    """
    Derive a clean repo name from any common Git URL format.

    Supports:
      https://github.com/org/my-app.git  →  my-app
      https://github.com/org/my-app      →  my-app
      git@github.com:org/my-app.git      →  my-app
      ssh://git@github.com/org/my-app    →  my-app
    """
    url = url.strip().rstrip("/")

    # SSH shorthand: git@github.com:org/repo.git
    if ":" in url and not url.startswith(("http://", "https://", "ssh://")):
        url = url.split(":", 1)[1]

    # Strip scheme for everything else
    for prefix in ("ssh://", "https://", "http://"):
        if url.startswith(prefix):
            url = url[len(prefix):]
            break

    # Strip leading host (everything up to the first /)
    if "/" in url:
        url = url.split("/", 1)[1]   # org/repo[.git]
    if "/" in url:
        url = url.rsplit("/", 1)[1]  # repo[.git]

    # Strip .git suffix
    if url.endswith(".git"):
        url = url[:-4]

    return url


def repo_dir_slug(repo_name: str) -> str:
    """
    Filesystem-safe directory name for a canonical `owner/repo` identity.

    The artifact *directory* cannot contain '/', but the moniker `repo_name`
    can.  Both derive from `repo_name_from_url`, so folder and in-graph
    identity never drift.

      team-a/api          -> team-a__api
      group/subgroup/svc  -> group__subgroup__svc
    """
    return repo_name.replace("/", "__")


# ---------------------------------------------------------------------------
# MonikerBuilder
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MonikerBuilder:
    """
    Builds SCIP-style monikers for a specific language + repository.

    Instantiate once per repository and reuse across all files.
    The adapter provides `file_path` (repo-relative) and symbol names.
    """
    language: str
    repo_name: str

    def _build(self, file_path: str, descriptor: str) -> str:
        return f"{SCHEME} {self.language} {self.repo_name} {file_path} {descriptor}"

    def for_function(self, file_path: str, func_name: str) -> str:
        """Top-level function: name()."""
        return self._build(file_path, f"{func_name}().")

    def for_method(
        self,
        file_path: str,
        class_name: str,
        method_name: str,
        *,
        enclosing: tuple[str, ...] = (),
    ) -> str:
        """Instance/class/static method: ClassName#method_name().

        `enclosing` is the chain of outer class names (outermost first) when
        the method's class is nested, e.g. enclosing=("Outer",) class_name="Inner"
        → "Outer#Inner#method_name()."  Defaults to () for the common
        top-level-class case, keeping the descriptor "ClassName#method_name()."
        """
        prefix = "".join(f"{c}#" for c in enclosing)
        return self._build(file_path, f"{prefix}{class_name}#{method_name}().")

    def for_class(
        self,
        file_path: str,
        class_name: str,
        *,
        enclosing: tuple[str, ...] = (),
    ) -> str:
        """Class or interface: ClassName#

        `enclosing` is the chain of outer class names (outermost first) when
        the class is nested, e.g. enclosing=("Outer",) class_name="Config"
        → "Outer#Config#".  Defaults to () for top-level classes, keeping the
        descriptor "ClassName#".  Descriptor stacking mirrors for_method and
        SCIP descriptor semantics, and is what makes same-named nested classes
        (e.g. Pydantic's `class Config`) produce distinct monikers.
        """
        prefix = "".join(f"{c}#" for c in enclosing)
        return self._build(file_path, f"{prefix}{class_name}#")

    def for_variable(self, file_path: str, var_name: str) -> str:
        """Module-level or class-level variable/constant: name."""
        return self._build(file_path, f"{var_name}.")

    def for_module(self, file_path: str, module_path: str) -> str:
        """
        File-level module: module_path/

        `module_path` is the import-style path provided by the adapter
        (source root stripped, extension removed), e.g. "services/user".
        The builder appends the trailing slash.
        """
        return self._build(file_path, f"{module_path}/")


# ---------------------------------------------------------------------------
# Parse a moniker back into components
# ---------------------------------------------------------------------------

def parse_moniker(moniker: str) -> MonikerComponents:
    """
    Split a Sutra moniker into its five components.

    Format: sutra <language> <repo_name> <file_path> <descriptor>

    Raises ValueError if the moniker does not start with "sutra" or has
    fewer than 5 space-separated tokens.
    """
    # The moniker has exactly 5 space-separated fields. file_path and
    # descriptor cannot contain spaces by construction, so a simple split works.
    parts = moniker.split(" ", 4)
    if len(parts) != 5:
        raise ValueError(
            f"Invalid moniker: expected 5 space-separated fields, got {len(parts)}: {moniker!r}"
        )
    scheme, language, repo_name, file_path, descriptor = parts
    if scheme != SCHEME:
        raise ValueError(f"Invalid moniker scheme: expected {SCHEME!r}, got {scheme!r}")
    return MonikerComponents(
        scheme=scheme,
        language=language,
        repo_name=repo_name,
        file_path=file_path,
        descriptor=descriptor,
    )


# ---------------------------------------------------------------------------
# Descriptor introspection
# ---------------------------------------------------------------------------

def is_valid_moniker(s: str) -> bool:
    """
    Return True if s is a well-formed Sutra moniker.

    A valid moniker has the form:
        sutra <language> <repo_name> <file_path> <descriptor>
    where descriptor ends with one of the four canonical suffixes:
        ().   callable (function or method)
        #     class/type
        /     module/namespace
        .     variable (any descriptor ending in . that is NOT ().)
    """
    try:
        parsed = parse_moniker(s)
        descriptor_kind(parsed.descriptor)
        return True
    except ValueError:
        return False


def descriptor_kind(descriptor: str) -> str:
    """
    Return the symbol kind implied by the descriptor's suffix.

    Returns one of: "function", "method", "class", "variable", "module".
    "function" and "method" are both callables — the caller must use context
    (e.g. whether the descriptor contains '#') to distinguish them.

    Raises ValueError for an unrecognised suffix.
    """
    if descriptor.endswith("()."):
        # callable — method if '#' is present, function otherwise
        if "#" in descriptor:
            return "method"
        return "function"
    if descriptor.endswith("#"):
        return "class"
    if descriptor.endswith("/"):
        return "module"
    if descriptor.endswith("."):
        return "variable"
    raise ValueError(f"Unrecognised descriptor suffix: {descriptor!r}")
