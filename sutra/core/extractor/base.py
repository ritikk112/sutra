from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Union


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Visibility(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    PROTECTED = "protected"
    INTERNAL = "internal"   # Go package-private; no Python/TS equivalent


class RelationKind(str, Enum):
    CALLS = "calls"
    EXTENDS = "extends"
    IMPLEMENTS = "implements"
    IMPORTS = "imports"
    CONTAINS = "contains"
    REFERENCES = "references"
    RETURNS_TYPE = "returns_type"
    PARAMETER_TYPE = "parameter_type"


# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------

@dataclass
class Location:
    line_start: int
    line_end: int
    byte_start: int
    byte_end: int
    column_start: int = 0
    column_end: int = 0


@dataclass
class Parameter:
    """A single parameter in a function or method signature."""
    name: str
    type_annotation: Optional[str] = None
    default_value: Optional[str] = None    # raw source text of the default expression
    is_variadic: bool = False              # *args in Python, ...rest in TS, variadic in Go
    is_keyword_variadic: bool = False      # **kwargs in Python only


# ---------------------------------------------------------------------------
# Symbol hierarchy
# ---------------------------------------------------------------------------

@dataclass
class SymbolBase:
    """Fields shared by every symbol type."""
    id: str                      # SCIP-style moniker — globally unique identifier
    name: str                    # unqualified name as it appears in source
    qualified_name: str          # dotted/slash-separated full name (e.g. services.user.UserService)
    file_path: str               # repo-relative path
    location: Location
    body_hash: str               # "sha256:<hex>" of raw body bytes; for ModuleSymbol, hash of entire file
    language: str                # "python" | "typescript" | "go"
    visibility: Visibility
    is_exported: bool


@dataclass
class FunctionSymbol(SymbolBase):
    """A module-level function (not a method)."""
    signature: str = ""                             # raw extracted source text of the signature line
    parameters: list[Parameter] = field(default_factory=list)
    return_type: Optional[str] = None              # raw type annotation text
    docstring: Optional[str] = None
    decorators: list[str] = field(default_factory=list)
    is_async: bool = False
    complexity: Optional[int] = None               # cyclomatic complexity; None if not yet computed


@dataclass
class ClassSymbol(SymbolBase):
    """A class or interface declaration."""
    base_classes: list[str] = field(default_factory=list)
    docstring: Optional[str] = None
    decorators: list[str] = field(default_factory=list)
    is_abstract: bool = False


@dataclass
class MethodSymbol(FunctionSymbol):
    """A method belonging to a class. Inherits all FunctionSymbol fields."""
    enclosing_class_id: Optional[str] = None  # moniker of containing ClassSymbol; None when cross-file (Go)
    is_static: bool = False
    is_constructor: bool = False    # __init__ / constructor / New
    receiver_kind: Optional[str] = None  # "pointer" | "value" | None; Go only


@dataclass
class VariableSymbol(SymbolBase):
    """A module-level or class-level variable."""
    type_annotation: Optional[str] = None
    is_constant: bool = False       # ALL_CAPS convention, const keyword, etc.


@dataclass
class ModuleSymbol(SymbolBase):
    """Represents the file itself as an importable unit. One per source file."""
    docstring: Optional[str] = None
    # body_hash on ModuleSymbol is the hash of the entire file — used for fast change detection


# The canonical union type for any symbol.
# Use isinstance() to dispatch: isinstance(sym, MethodSymbol) before FunctionSymbol
# because MethodSymbol is a subtype of FunctionSymbol.
Symbol = Union[FunctionSymbol, ClassSymbol, MethodSymbol, VariableSymbol, ModuleSymbol]


# ---------------------------------------------------------------------------
# Relationship (first-class object, separate from symbols)
# ---------------------------------------------------------------------------

@dataclass
class Relationship:
    """
    A directed edge between two symbols.

    Phase 1: cross-file references are unresolved — target_id is None,
    target_name holds the literal name from source, metadata holds
    import_source / call_form hints for Phase 2 LSP resolution.
    """
    source_id: str
    kind: RelationKind
    is_resolved: bool
    target_id: Optional[str] = None        # set when resolved to a known moniker
    target_name: Optional[str] = None      # literal name from source (always set for unresolved)
    location: Optional[Location] = None    # call/import site location
    metadata: dict = field(default_factory=dict)
    # metadata keys used in Phase 1:
    #   "import_source": str   — the module path being imported from
    #   "call_form": str       — "direct" | "method" | "attribute"
    #   "receiver": str        — object name for method calls (db in db.insert())


# ---------------------------------------------------------------------------
# Pipeline container types
# ---------------------------------------------------------------------------

@dataclass
class File:
    path: str           # repo-relative path
    language: str       # detected language
    size_bytes: int
    hash: str           # sha256 of full file contents


@dataclass
class Repository:
    url: str
    name: str           # derived from URL (last path segment, no .git)


@dataclass
class FileExtraction:
    """Output of running the tree-sitter adapter on a single file."""
    file: File
    symbols: list[Symbol]
    relationships: list[Relationship]


@dataclass
class IndexResult:
    """Aggregated output of a full indexing run."""
    repository: Repository
    files: list[File]
    symbols: list[Symbol]
    relationships: list[Relationship]
    indexed_at: datetime
    commit_hash: str
    languages: dict[str, int]   # {"python": 42, "typescript": 18, "go": 7}
    failed_files: list[tuple[str, str]] = field(default_factory=list)
    # Each entry: (repo_relative_path, error_message) — files the indexer
    # could not read or that the adapter raised an exception on.
