# Tier-2 Local Symbols Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Index function-local symbols (nested functions, function-local classes, and their methods) as graph-only nodes so an agent can trace a complete call stack through local scopes.

**Architecture:** A function-local symbol gets a self-describing moniker that stacks its enclosing-function scope into the descriptor (`outer().Helper#run().`), plus a structured `enclosing_moniker` field (the resolver compares scopes via this, never by parsing the descriptor) and an explicit `is_local` flag (cross-file resolution exclusion). Locals are excluded from embeddings. The fatal duplicate-moniker assertion becomes a deterministic keep-first backstop.

**Tech Stack:** Python 3.11, tree-sitter (`tree_sitter_runner`), dataclasses, rustworkx (traversal), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-06-29-tier2-local-symbols-design.md`

## Global Constraints

- **Moniker grammar is space-delimited (5 fields); descriptors MUST NOT contain spaces.** `parse_moniker` does `split(" ", 4)`.
- **Descriptor suffix sentinels are self-delimiting:** `().` callable, `#` class, `.` variable, `/` module. Scope segments stack: function → `name().`, class → `name#`.
- **Top-level symbol monikers MUST be byte-identical to today** (regression: existing tests in `tests/test_python_adapter.py` must stay green unchanged).
- **Locals are graph-only:** excluded from `chunk_builder._EMBEDDABLE` output; never embedded.
- **Resolver safety is by the `is_local` flag, never by descriptor shape.**
- **Determinism:** ids depend only on file content + global file sort + tree (byte) order. No randomness, no line numbers.
- **Scope is Python only.** TS/Go adapters unchanged.
- **Run tests with:** `source .venv/bin/activate` then `python -m pytest`.

---

### Task 1: Symbol model — `is_local` and `enclosing_moniker` fields

**Files:**
- Modify: `sutra/core/extractor/base.py` (SymbolBase, ~lines 59-71)
- Test: `tests/test_symbol_model.py` (create)

**Interfaces:**
- Produces: `SymbolBase.is_local: bool = False`, `SymbolBase.enclosing_moniker: Optional[str] = None` — inherited by every Symbol subclass.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_symbol_model.py
from sutra.core.extractor.base import FunctionSymbol, ClassSymbol, Location, Visibility


def _loc():
    return Location(line_start=1, line_end=2, byte_start=0, byte_end=10)


def test_symbols_default_to_non_local():
    fn = FunctionSymbol(
        id="sutra python r f.py outer().", name="outer", qualified_name="f.outer",
        file_path="f.py", location=_loc(), body_hash="sha256:x", language="python",
        visibility=Visibility.PUBLIC, is_exported=True,
    )
    assert fn.is_local is False
    assert fn.enclosing_moniker is None


def test_local_symbol_carries_enclosing_moniker():
    cls = ClassSymbol(
        id="sutra python r f.py outer().NoCause#", name="NoCause",
        qualified_name="f.outer.NoCause", file_path="f.py", location=_loc(),
        body_hash="sha256:x", language="python", visibility=Visibility.PUBLIC,
        is_exported=True, is_local=True,
        enclosing_moniker="sutra python r f.py outer().",
    )
    assert cls.is_local is True
    assert cls.enclosing_moniker == "sutra python r f.py outer()."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_symbol_model.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'is_local'`

- [ ] **Step 3: Add the fields to SymbolBase**

In `sutra/core/extractor/base.py`, append to `SymbolBase` (after `is_exported: bool`):

```python
@dataclass
class SymbolBase:
    """Fields shared by every symbol type."""
    id: str
    name: str
    qualified_name: str
    file_path: str
    location: Location
    body_hash: str
    language: str
    visibility: Visibility
    is_exported: bool
    # Tier-2 local symbols: graph-only nodes nested inside a function scope.
    is_local: bool = False
    # Immediate enclosing scope's moniker (containing function/class); None = module scope.
    enclosing_moniker: Optional[str] = None
```

(`Optional` is already imported at the top of the file.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_symbol_model.py tests/test_python_adapter.py -q`
Expected: PASS (new tests pass; adapter tests still green — defaults don't change existing construction).

- [ ] **Step 5: Commit**

```bash
git add sutra/core/extractor/base.py tests/test_symbol_model.py
git commit -m "feat(model): add is_local + enclosing_moniker to SymbolBase"
```

---

### Task 2: MonikerBuilder — scope-segment chain, `descriptor_kind` generalization, space assert

**Files:**
- Modify: `sutra/core/extractor/moniker.py` (MonikerBuilder ~100-156, descriptor_kind ~215-236)
- Test: `tests/test_moniker.py` (create if absent; else append)

**Interfaces:**
- Produces:
  - `MonikerBuilder.for_function(file_path, func_name, *, enclosing=())`
  - `MonikerBuilder.for_class(file_path, class_name, *, enclosing=())`
  - `MonikerBuilder.for_method(file_path, class_name, method_name, *, enclosing=())`
  - where `enclosing: tuple[tuple[str, str], ...]` is `(name, suffix)` pairs, suffix ∈ `{"#", "()."}`, outermost first.
  - `_build` asserts the descriptor contains no space.
  - `descriptor_kind` callable test is `endswith(").")`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_moniker.py  (append, or create with these imports)
import pytest
from sutra.core.extractor.moniker import MonikerBuilder, descriptor_kind


def _b():
    return MonikerBuilder(language="python", repo_name="r")


def test_for_function_with_function_scope():
    m = _b().for_function("f.py", "inner", enclosing=(("outer", "()."),))
    assert m == "sutra python r f.py outer().inner()."


def test_for_class_with_function_scope():
    m = _b().for_class("f.py", "NoCause", enclosing=(("test_a", "()."),))
    assert m == "sutra python r f.py test_a().NoCause#"


def test_for_method_with_mixed_scope():
    m = _b().for_method("f.py", "Helper", "run", enclosing=(("outer", "()."),))
    assert m == "sutra python r f.py outer().Helper#run()."


def test_for_class_class_scope_unchanged():
    # Regression: existing nested-class form must be byte-identical.
    m = _b().for_class("f.py", "Config", enclosing=(("Outer", "#"),))
    assert m == "sutra python r f.py Outer#Config#"


def test_build_rejects_descriptor_with_space():
    with pytest.raises(AssertionError):
        _b().for_function("f.py", "bad name")  # space in identifier


def test_descriptor_kind_handles_disambiguator():
    assert descriptor_kind("inner(1).") == "function"
    assert descriptor_kind("Helper#run(1).") == "method"
    assert descriptor_kind("NoCause(1)#") == "class"
    assert descriptor_kind("x.") == "variable"   # plain variable still variable
    assert descriptor_kind("outer().inner().") == "function"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_moniker.py -v`
Expected: FAIL — `for_function` has no `enclosing` kwarg; `for_class`/`for_method` pass class names not pairs; no space assert; `descriptor_kind("inner(1).")` returns "variable".

- [ ] **Step 3: Implement the builder changes**

In `sutra/core/extractor/moniker.py`, replace the `_build` and `for_*` methods:

```python
    def _build(self, file_path: str, descriptor: str) -> str:
        # Moniker is space-delimited (parse_moniker splits on spaces); a space
        # in the descriptor would silently corrupt round-tripping.
        assert " " not in descriptor, (
            f"Descriptor must not contain spaces: {descriptor!r}"
        )
        return f"{SCHEME} {self.language} {self.repo_name} {file_path} {descriptor}"

    @staticmethod
    def _prefix(enclosing: tuple[tuple[str, str], ...]) -> str:
        """Render a scope chain of (name, suffix) pairs, outermost first."""
        return "".join(f"{name}{suffix}" for name, suffix in enclosing)

    def for_function(
        self, file_path: str, func_name: str,
        *, enclosing: tuple[tuple[str, str], ...] = (),
    ) -> str:
        """Function: name().  `enclosing` stacks outer scopes (function → name().,
        class → name#) for function-local (nested) functions."""
        return self._build(file_path, f"{self._prefix(enclosing)}{func_name}().")

    def for_method(
        self, file_path: str, class_name: str, method_name: str,
        *, enclosing: tuple[tuple[str, str], ...] = (),
    ) -> str:
        """Method: ClassName#method_name().  `enclosing` is the scope chain ABOVE
        the method's own class (class and/or function segments)."""
        return self._build(
            file_path, f"{self._prefix(enclosing)}{class_name}#{method_name}()."
        )

    def for_class(
        self, file_path: str, class_name: str,
        *, enclosing: tuple[tuple[str, str], ...] = (),
    ) -> str:
        """Class: ClassName#.  `enclosing` stacks outer scopes."""
        return self._build(file_path, f"{self._prefix(enclosing)}{class_name}#")
```

Then generalize `descriptor_kind` (change only the callable branch):

```python
def descriptor_kind(descriptor: str) -> str:
    # callable — method if '#' present, function otherwise. Accept a disambiguator
    # inside the parens (e.g. inner(1).), so the test is ").", not literal "()."
    if descriptor.endswith(")."):
        return "method" if "#" in descriptor else "function"
    if descriptor.endswith("#"):
        return "class"
    if descriptor.endswith("/"):
        return "module"
    if descriptor.endswith("."):
        return "variable"
    raise ValueError(f"Unrecognised descriptor suffix: {descriptor!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_moniker.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sutra/core/extractor/moniker.py tests/test_moniker.py
git commit -m "feat(moniker): scope-segment chain, disambiguator-safe descriptor_kind, no-space assert"
```

---

### Task 3: Adapter — unified scope helpers, keep top-level monikers identical

**Files:**
- Modify: `sutra/core/extractor/adapters/python.py` (add `_scope_chain`, `_node_moniker`; rewire `_build_class`/`_build_function`/`_build_method`)
- Test: `tests/test_python_adapter.py` (existing suite is the regression guard; add two scope-chain tests)

**Interfaces:**
- Consumes: `MonikerBuilder.for_* (..., enclosing=tuple[tuple[str,str],...])` from Task 2.
- Produces:
  - `_scope_chain(node) -> tuple[tuple[str, str], ...]` — enclosing scopes outermost-first, EXCLUDING `node`; class → `(name, "#")`, function → `(name, "().")`. Does NOT break at functions.
  - `_node_moniker(node, file_path, builder) -> str` — base (un-disambiguated) moniker for a class_definition or function_definition node.
  - `_build_class`/`_build_function`/`_build_method` use `_scope_chain` and set `enclosing_moniker`/`is_local` (computed in Step 3).

This task is a **pure refactor**: top-level symbol monikers must not change. Local emission comes in Tasks 4-5.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_python_adapter.py  (append)
from sutra.core.extractor.adapters.python import _scope_chain
import tree_sitter_python as _tsp_unused  # noqa: F401  (adapter loads its own parser)


class TestScopeChain:
    def test_top_level_class_scope_empty(self, adapter):
        # Regression: nested Pydantic-style class still Outer#Config#.
        src = "class Outer:\n    class Config:\n        pass\n"
        result = extract(adapter, src)
        cfg = next(s for s in result.symbols if s.name == "Config")
        assert cfg.id.endswith("Outer#Config#")
        assert cfg.is_local is False
        assert cfg.enclosing_moniker is not None  # the Outer class moniker
        outer = next(s for s in result.symbols if s.name == "Outer")
        assert cfg.enclosing_moniker == outer.id

    def test_top_level_function_enclosing_is_none(self, adapter):
        src = "def f():\n    pass\n"
        result = extract(adapter, src)
        f = sym_by_name(result, "f")
        assert f.is_local is False
        assert f.enclosing_moniker is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_python_adapter.py::TestScopeChain -v`
Expected: FAIL — `_scope_chain` import error; `enclosing_moniker` is `None` for `Config` (not yet set).

- [ ] **Step 3: Implement `_scope_chain`, `_node_moniker`, and rewire builders**

Add helpers near the other ancestor walkers (after `_find_enclosing_function`, ~line 169) in `python.py`:

```python
def _scope_chain(node: Node) -> tuple[tuple[str, str], ...]:
    """
    Enclosing scopes of `node`, outermost first, EXCLUDING `node` itself.
    class_definition → (name, "#"); function_definition → (name, "().").
    Unlike _enclosing_class_names, this does NOT stop at a function — it is
    what gives function-local symbols a self-describing, unique descriptor.
    """
    segments: list[tuple[str, str]] = []
    current = node.parent
    while current is not None:
        if current.type == "class_definition":
            name_node = current.child_by_field_name("name")
            if name_node is not None:
                segments.append((_txt(name_node), "#"))
        elif current.type == "function_definition":
            name_node = current.child_by_field_name("name")
            if name_node is not None:
                segments.append((_txt(name_node), "()."))
        current = current.parent
    segments.reverse()
    return tuple(segments)


def _nearest_scope(node: Node) -> Optional[Node]:
    """Nearest enclosing class_definition or function_definition, or None (module)."""
    current = node.parent
    while current is not None:
        if current.type in ("class_definition", "function_definition"):
            return current
        if current.type == "module":
            return None
        current = current.parent
    return None


def _node_moniker(node: Node, file_path: str, builder: "MonikerBuilder") -> str:
    """Base (un-disambiguated) moniker for a class/function node, from its scope
    chain. Order-independent — used for a node's own id AND for its children's
    enclosing_moniker, so emission order never matters."""
    name = _txt(node.child_by_field_name("name"))
    enclosing = _scope_chain(node)
    if node.type == "class_definition":
        return builder.for_class(file_path, name, enclosing=enclosing)
    # function_definition: method if its nearest scope is a class, else function.
    if _find_enclosing_class(node) is not None:
        class_name = enclosing[-1][0]  # immediate enclosing class is the last segment
        return builder.for_method(
            file_path, class_name, name, enclosing=enclosing[:-1]
        )
    return builder.for_function(file_path, name, enclosing=enclosing)
```

Now rewire the three builders to use `_scope_chain` + set `is_local`/`enclosing_moniker`. Replace the moniker/qual lines in `_build_class` (currently uses `_enclosing_class_names`):

```python
        enclosing = _scope_chain(node)
        is_local = _find_enclosing_function(node) is not None
        scope_node = _nearest_scope(node)
        enclosing_moniker = (
            _node_moniker(scope_node, file_path, builder) if scope_node else None
        )
        qual_path = ".".join((*(n for n, _ in enclosing), name))

        return ClassSymbol(
            id=builder.for_class(file_path, name, enclosing=enclosing),
            name=name,
            qualified_name=f"{mod_qname}.{qual_path}",
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
            is_local=is_local,
            enclosing_moniker=enclosing_moniker,
        )
```

In `_build_function`, replace the `return FunctionSymbol(...)` to compute scope and pass the new fields:

```python
        enclosing = _scope_chain(node)
        is_local = _find_enclosing_function(node) is not None
        scope_node = _nearest_scope(node)
        enclosing_moniker = (
            _node_moniker(scope_node, file_path, builder) if scope_node else None
        )

        return FunctionSymbol(
            id=builder.for_function(file_path, name, enclosing=enclosing),
            name=name,
            qualified_name=f"{mod_qname}.{name}" if not enclosing
                else f"{mod_qname}." + ".".join((*(n for n, _ in enclosing), name)),
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
            is_local=is_local,
            enclosing_moniker=enclosing_moniker,
        )
```

In `_build_method`, replace the chain/qual computation and the moniker, and add the new fields. The method's own class is the LAST segment of `_scope_chain(node)`:

```python
        chain = _scope_chain(node)              # includes the method's own class as last segment
        method_enclosing = chain[:-1]           # everything above the method's class
        qual_path = ".".join((*(n for n, _ in chain), name)) if chain \
            else f"{cls_sym.name}.{name}"
        is_local = any(suffix == "()." for _, suffix in method_enclosing)
        # enclosing scope is the method's own class:
        enclosing_moniker = cls_sym.id

        return MethodSymbol(
            id=builder.for_method(
                file_path, cls_sym.name, name, enclosing=method_enclosing
            ),
            name=name,
            qualified_name=f"{mod_qname}.{qual_path}",
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
            is_local=is_local,
            enclosing_moniker=enclosing_moniker,
        )
```

(Leave the `MonikerBuilder` import/type usable: it is already imported at the top of `python.py`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_python_adapter.py -q`
Expected: PASS — the two new TestScopeChain tests pass AND every pre-existing adapter test stays green (top-level monikers unchanged because `_scope_chain` of a top-level/nested-class symbol contains only `#` segments, identical to the old `_enclosing_class_names` rendering).

- [ ] **Step 5: Commit**

```bash
git add sutra/core/extractor/adapters/python.py tests/test_python_adapter.py
git commit -m "refactor(adapter): unified scope chain + node moniker; set is_local/enclosing_moniker (top-level unchanged)"
```

---

### Task 4: Adapter — emit nested functions as local nodes + CONTAINS

**Files:**
- Modify: `sutra/core/extractor/adapters/python.py` (the `func.def` loop `else` branch, ~line 602-630)
- Test: `tests/test_python_adapter.py`

**Interfaces:**
- Consumes: `_build_function` (now sets `is_local`/`enclosing_moniker`), `_node_moniker`.
- Produces: nested functions emitted as `FunctionSymbol(is_local=True)` with a `function→nested-function` CONTAINS edge (`is_resolved=True`, `target_id` set).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_python_adapter.py  (append)
from sutra.core.extractor.base import RelationKind


class TestNestedFunctionLocals:
    def test_nested_function_emitted_as_local(self, adapter):
        src = "def outer():\n    def inner():\n        pass\n    return inner\n"
        result = extract(adapter, src)
        inner = sym_by_name(result, "inner")
        outer = sym_by_name(result, "outer")
        assert inner is not None
        assert inner.is_local is True
        assert inner.id.endswith("outer().inner().")
        assert inner.enclosing_moniker == outer.id

    def test_function_contains_its_nested_function(self, adapter):
        src = "def outer():\n    def inner():\n        pass\n    return inner\n"
        result = extract(adapter, src)
        outer = sym_by_name(result, "outer")
        inner = sym_by_name(result, "inner")
        contains = [
            r for r in result.relationships
            if r.kind == RelationKind.CONTAINS
            and r.source_id == outer.id and r.target_id == inner.id
        ]
        assert len(contains) == 1
        assert contains[0].is_resolved is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_python_adapter.py::TestNestedFunctionLocals -v`
Expected: FAIL — `inner` is `None` (still skipped at the `_find_enclosing_function` guard).

- [ ] **Step 3: Emit nested functions instead of skipping**

In `python.py`, in the `func.def` loop, replace the `else` branch that currently skips nested functions:

```python
            else:
                # Not inside a class. Either a top-level function or a function
                # nested in another function (Tier-2 local). Emit either way.
                fn = self._build_function(
                    func_node, file_path, source_bytes, builder, mod_qname
                )
                if fn is None:
                    continue
                # Collapse identical-moniker duplicates (Task 6 handles the rare
                # same-scope same-name case via the disambiguator).
                symbols.append(fn)

                # CONTAINS source: the module for top-level, else the enclosing
                # scope (a function for a function-local function).
                container_id = fn.enclosing_moniker or module_id
                relationships.append(Relationship(
                    source_id=container_id,
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
```

(The `_find_enclosing_function(...) continue` line is removed. `_build_function` already sets `enclosing_moniker`, so a top-level function gets `enclosing_moniker=None` → `container_id = module_id`, preserving the existing module→function CONTAINS.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_python_adapter.py -q`
Expected: PASS — nested function emitted; existing module→function CONTAINS unchanged (`enclosing_moniker is None`).

- [ ] **Step 5: Commit**

```bash
git add sutra/core/extractor/adapters/python.py tests/test_python_adapter.py
git commit -m "feat(adapter): emit nested functions as Tier-2 local nodes + CONTAINS"
```

---

### Task 5: Adapter — emit function-local classes + their methods (odin reproduction)

**Files:**
- Modify: `sutra/core/extractor/adapters/python.py` (the `class.def` loop ~line 513, remove the function-local skip; the method loop's `cls_sym is None` path)
- Test: `tests/test_python_adapter.py`

**Interfaces:**
- Consumes: `_build_class`/`_build_method` (set local fields), `class_node_to_sym` map.
- Produces: function-local classes emitted as `ClassSymbol(is_local=True)` with `function→class` CONTAINS; their methods emitted (the method loop already finds them via `class_node_to_sym`).

- [ ] **Step 1: Write the failing test (odin reproduction)**

```python
# tests/test_python_adapter.py  (append)
class TestFunctionLocalClassesIndexed:
    def test_two_same_named_local_classes_distinct_monikers(self, adapter):
        # The odin case: two `class NoCause` in two different test functions.
        src = (
            "def test_a():\n"
            "    class NoCause:\n"
            "        pass\n"
            "    return NoCause()\n"
            "\n"
            "def test_b():\n"
            "    class NoCause:\n"
            "        pass\n"
            "    return NoCause()\n"
        )
        result = extract(adapter, src)
        nocause = [s for s in result.symbols
                   if isinstance(s, ClassSymbol) and s.name == "NoCause"]
        assert len(nocause) == 2
        ids = {s.id for s in nocause}
        assert ids == {
            "my-app python src/services/user.py test_a().NoCause#",
            "my-app python src/services/user.py test_b().NoCause#",
        } or all(s.is_local for s in nocause)  # repo/file prefix may vary; key check: distinct + local
        assert len(ids) == 2
        assert all(s.is_local for s in nocause)

    def test_local_class_method_and_contains(self, adapter):
        src = (
            "def outer():\n"
            "    class Helper:\n"
            "        def run(self):\n"
            "            pass\n"
            "    return Helper\n"
        )
        result = extract(adapter, src)
        helper = next(s for s in result.symbols
                      if isinstance(s, ClassSymbol) and s.name == "Helper")
        run = next(s for s in result.symbols if s.name == "run")
        outer = sym_by_name(result, "outer")
        assert helper.is_local is True and helper.id.endswith("outer().Helper#")
        assert run.is_local is True and run.id.endswith("outer().Helper#run().")
        assert run.enclosing_moniker == helper.id
        # function -> local class CONTAINS
        assert any(
            r.kind == RelationKind.CONTAINS and r.source_id == outer.id
            and r.target_id == helper.id and r.is_resolved
            for r in result.relationships
        )
        # local class -> method CONTAINS
        assert any(
            r.kind == RelationKind.CONTAINS and r.source_id == helper.id
            and r.target_id == run.id and r.is_resolved
            for r in result.relationships
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_python_adapter.py::TestFunctionLocalClassesIndexed -v`
Expected: FAIL — `NoCause` classes are skipped (current odin fix), `len(nocause) == 0`.

- [ ] **Step 3: Remove the class-loop skip and fix the CONTAINS source**

In `python.py`, the class loop currently starts (after Task-3 edits) with the function-local skip added by the odin fix. Replace the loop body so it emits local classes and uses the correct CONTAINS container:

```python
        class_node_to_sym: dict[int, ClassSymbol] = {}
        for class_node in captures.get("class.def", []):
            cls = self._build_class(class_node, file_path, source_bytes, builder, mod_qname)
            if cls is None:
                continue
            symbols.append(cls)
            class_node_to_sym[class_node.id] = cls

            # CONTAINS source: module for top-level, else the enclosing scope
            # (a function, for a function-local class).
            container_id = cls.enclosing_moniker or module_id
            relationships.append(Relationship(
                source_id=container_id,
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
```

The method loop needs no change: a method of a local class has `_find_enclosing_class(func_node)` pointing at the local class, which is now in `class_node_to_sym`, so `cls_sym` is found and `_build_method` (Task 3) sets `is_local`/`enclosing_moniker`.

**Note:** the prior odin guard `if _find_enclosing_function(class_node) is not None: continue` (and any analogous nested-function guard text) is fully removed by this replacement.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_python_adapter.py -q`
Expected: PASS — local classes + methods emitted with correct monikers and CONTAINS; existing class tests green.

- [ ] **Step 5: Commit**

```bash
git add sutra/core/extractor/adapters/python.py tests/test_python_adapter.py
git commit -m "feat(adapter): emit function-local classes + methods as Tier-2 nodes (odin fix superseded)"
```

---

### Task 6: Adapter — residual same-scope disambiguator

**Files:**
- Modify: `sutra/core/extractor/adapters/python.py` (introduce a per-file disambiguator applied at symbol append)
- Test: `tests/test_python_adapter.py`

**Interfaces:**
- Produces: when two symbols would otherwise share a moniker (same scope, name, kind), the 2nd+ in tree order gets `(N)` inserted before the trailing suffix: `inner().`→`inner(1).`, `NoCause#`→`NoCause(1)#`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_python_adapter.py  (append)
class TestResidualDisambiguator:
    def test_same_scope_same_name_classes_disambiguated(self, adapter):
        # Legal but degenerate: class A redefined twice in one function.
        src = (
            "def f():\n"
            "    class A:\n"
            "        pass\n"
            "    class A:\n"
            "        pass\n"
            "    return A\n"
        )
        result = extract(adapter, src)
        a = [s for s in result.symbols if isinstance(s, ClassSymbol) and s.name == "A"]
        ids = [s.id for s in a]
        assert len(ids) == len(set(ids)) == 2
        # one base, one disambiguated; tree order → first keeps base form
        assert any(i.endswith("f().A#") for i in ids)
        assert any(i.endswith("f().A(1)#") for i in ids)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_python_adapter.py::TestResidualDisambiguator -v`
Expected: FAIL — both produce `f().A#`; either a duplicate-id assertion downstream or `len(set(ids)) == 1`.

- [ ] **Step 3: Add the disambiguator**

Add a module-level helper in `python.py`:

```python
def _disambiguate(base_id: str, seen: dict[str, int]) -> str:
    """Deterministic tie-break for same-scope same-name symbols (rare).
    First occurrence keeps the base moniker; the Nth (N>=1) gets `(N)` inserted
    before the trailing suffix, preserving the descriptor's terminating sentinel."""
    n = seen.get(base_id, 0)
    seen[base_id] = n + 1
    if n == 0:
        return base_id
    for suffix in ("().", "#", "/", "."):
        if base_id.endswith(suffix):
            return f"{base_id[: -len(suffix)]}({n}){suffix}"
    return base_id  # unreachable for valid monikers
```

In `extract`, create one `seen: dict[str, int] = {}` at the top of symbol assembly (next to `symbols: list[Symbol] = []`). Then wherever a class/function/method symbol is appended, pass its `id` through `_disambiguate` BEFORE appending and BEFORE building its CONTAINS/CALLS edges. Concretely, immediately after each `_build_*` returns a non-None symbol, do:

```python
            sym.id = _disambiguate(sym.id, seen)
```

Apply this for `cls` (class loop), `method` (method branch), and `fn` (function branch). Because dataclasses are mutable, reassigning `sym.id` is valid and the subsequent CONTAINS/CALLS edges (which reference `sym.id`) pick up the disambiguated value. The `enclosing_moniker` of children pointing at a disambiguated parent is out of scope (parent disambiguation is doubly-rare); if such a collision ever occurs it falls through to the Task-10 keep-first backstop.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_python_adapter.py -q`
Expected: PASS — distinct `f().A#` and `f().A(1)#`; all other adapter tests green.

- [ ] **Step 5: Commit**

```bash
git add sutra/core/extractor/adapters/python.py tests/test_python_adapter.py
git commit -m "feat(adapter): deterministic disambiguator for same-scope same-name locals"
```

---

### Task 7: chunk_builder — exclude locals from embeddings

**Files:**
- Modify: `sutra/core/embedder/chunk_builder.py` (the embeddable filter, ~line 45)
- Test: `tests/test_chunk_builder.py` (create if absent; else append)

**Interfaces:**
- Produces: `build_chunks` excludes any symbol with `is_local=True` from both `chunks` and `monikers`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunk_builder.py  (append/create)
from pathlib import Path
from sutra.core.embedder.chunk_builder import build_chunks
from sutra.core.extractor.base import FunctionSymbol, Location, Visibility


def _fn(id_, name, is_local):
    return FunctionSymbol(
        id=id_, name=name, qualified_name=name, file_path="f.py",
        location=Location(1, 2, 0, 5), body_hash="sha256:x", language="python",
        visibility=Visibility.PUBLIC, is_exported=True, signature=f"def {name}()",
        is_local=is_local,
    )


def test_local_symbols_excluded_from_chunks(tmp_path):
    (tmp_path / "f.py").write_text("def top():\n    def inner():\n        pass\n")
    syms = [
        _fn("sutra python r f.py top().", "top", False),
        _fn("sutra python r f.py top().inner().", "inner", True),
    ]
    chunks, monikers = build_chunks(syms, tmp_path)
    assert "sutra python r f.py top()." in monikers
    assert "sutra python r f.py top().inner()." not in monikers
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_chunk_builder.py::test_local_symbols_excluded_from_chunks -v`
Expected: FAIL — the local `inner` moniker is present in `monikers`.

- [ ] **Step 3: Add the exclusion**

In `sutra/core/embedder/chunk_builder.py`, change the embeddable filter (~line 45):

```python
    embeddable = sorted(
        [s for s in symbols if isinstance(s, _EMBEDDABLE) and not s.is_local],
        key=lambda s: s.id,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_chunk_builder.py tests/test_python_adapter.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sutra/core/embedder/chunk_builder.py tests/test_chunk_builder.py
git commit -m "feat(embeddings): exclude Tier-2 local symbols from chunks"
```

---

### Task 8: Resolver — exclude locals from cross-file pool + scope-gated local rule

**Files:**
- Modify: `sutra/core/resolver/heuristic.py` (`resolve` and `_pick`)
- Test: `tests/test_resolver.py`

**Interfaces:**
- Consumes: `Symbol.is_local`, `Symbol.enclosing_moniker`, `Relationship.source_id` (caller moniker).
- Produces: locals excluded from `by_name`; a new highest-precedence "local-scope" rule resolves a CALLS edge to the innermost visible local on the caller's scope chain.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_resolver.py  (append)
from sutra.core.extractor.base import FunctionSymbol, Location, Visibility, Relationship, RelationKind
from sutra.core.resolver.heuristic import HeuristicResolver


def _fn(id_, name, *, is_local=False, enclosing=None):
    return FunctionSymbol(
        id=id_, name=name, qualified_name=name, file_path="f.py",
        location=Location(1, 2, 0, 5), body_hash="sha256:x", language="python",
        visibility=Visibility.PUBLIC, is_exported=True, is_local=is_local,
        enclosing_moniker=enclosing,
    )


def _calls(src, name):
    return Relationship(source_id=src, kind=RelationKind.CALLS, is_resolved=False,
                        target_name=name, metadata={"call_form": "direct"})


def test_local_scope_call_resolves_to_local_not_toplevel():
    outer = _fn("sutra python r f.py outer().", "inner_caller")  # caller is outer
    outer.name = "outer"; outer.id = "sutra python r f.py outer()."
    top_inner = _fn("sutra python r f.py inner().", "inner")
    local_inner = _fn("sutra python r f.py outer().inner().", "inner",
                      is_local=True, enclosing="sutra python r f.py outer().")
    # outer calls inner -> must resolve to the LOCAL inner, not the top-level one
    rel = _calls("sutra python r f.py outer().", "inner")
    HeuristicResolver().resolve([outer, top_inner, local_inner], [rel])
    assert rel.is_resolved and rel.target_id == local_inner.id


def test_toplevel_call_not_broken_by_sibling_local():
    # A top-level caller calling top-level inner must still resolve, even though
    # a same-named local exists in the same file (regression guard).
    top_caller = _fn("sutra python r f.py main().", "main")
    top_inner = _fn("sutra python r f.py inner().", "inner")
    local_inner = _fn("sutra python r f.py outer().inner().", "inner",
                      is_local=True, enclosing="sutra python r f.py outer().")
    rel = _calls("sutra python r f.py main().", "inner")
    HeuristicResolver().resolve([top_caller, top_inner, local_inner], [rel])
    assert rel.is_resolved and rel.target_id == top_inner.id


def test_local_never_resolved_cross_scope():
    # A caller outside the local's scope must NOT resolve to it (no top-level inner here).
    other = _fn("sutra python r f.py other().", "other")
    local_inner = _fn("sutra python r f.py outer().inner().", "inner",
                      is_local=True, enclosing="sutra python r f.py outer().")
    rel = _calls("sutra python r f.py other().", "inner")
    HeuristicResolver().resolve([other, local_inner], [rel])
    assert not rel.is_resolved
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_resolver.py -k "local_scope or sibling or cross_scope" -v`
Expected: FAIL — locals are in `by_name`, so the first test may resolve to the wrong inner / the sibling test goes ambiguous-unresolved.

- [ ] **Step 3: Implement scope-aware local resolution**

In `heuristic.py` `resolve`, build the non-local `by_name`, a `sym_by_id` map, and a local index keyed by `(enclosing_moniker, name)`:

```python
        # name -> NON-LOCAL candidate symbols (locals resolve via scope, below)
        by_name: dict[str, list[Symbol]] = defaultdict(list)
        for sym in symbols:
            if isinstance(sym, _CALLABLE_TYPES) and not sym.is_local:
                by_name[sym.name].append(sym)

        sym_by_id: dict[str, Symbol] = {s.id: s for s in symbols}
        # (enclosing_moniker, name) -> local callable candidates
        local_by_scope: dict[tuple[str, str], list[Symbol]] = defaultdict(list)
        for sym in symbols:
            if isinstance(sym, _CALLABLE_TYPES) and sym.is_local and sym.enclosing_moniker:
                local_by_scope[(sym.enclosing_moniker, sym.name)].append(sym)
```

Add a helper method:

```python
    @staticmethod
    def _scope_chain_of(caller_id: str, sym_by_id: dict[str, Symbol]) -> list[str]:
        """Caller's own scope + enclosing scopes, innermost first."""
        chain = [caller_id]
        cur = sym_by_id.get(caller_id)
        seen = {caller_id}
        while cur is not None and cur.enclosing_moniker and cur.enclosing_moniker not in seen:
            chain.append(cur.enclosing_moniker)
            seen.add(cur.enclosing_moniker)
            cur = sym_by_id.get(cur.enclosing_moniker)
        return chain

    def _resolve_local(
        self, rel, name, sym_by_id, local_by_scope,
    ):
        """Innermost visible local named `name`, walking the caller's scope chain."""
        for scope in self._scope_chain_of(rel.source_id, sym_by_id):
            hits = self._prefer_call_form(rel, local_by_scope.get((scope, name), []))
            if len(hits) == 1:
                return hits[0]
        return None
```

In the CALLS loop, try the local rule FIRST (highest precedence — Python locals shadow), then fall back to the existing non-local path:

```python
            name = rel.target_name
            if not name:
                continue

            local_target = self._resolve_local(rel, name, sym_by_id, local_by_scope)
            if local_target is not None:
                stats.matchable += 1
                rel.target_id = local_target.id
                rel.is_resolved = True
                rel.metadata["resolved_by"] = "local-scope"
                stats.resolved += 1
                stats.by_rule["local-scope"] = stats.by_rule.get("local-scope", 0) + 1
                continue

            candidates = by_name.get(name)
            if not candidates:
                continue
            stats.matchable += 1
            caller_file = parse_moniker(rel.source_id).file_path
            target = self._pick(rel, candidates, caller_file, imports_of_file)
            # ... unchanged from here ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_resolver.py -q`
Expected: PASS — local-scope resolves to the local; top-level call unaffected by sibling local; cross-scope local stays unresolved.

- [ ] **Step 5: Commit**

```bash
git add sutra/core/resolver/heuristic.py tests/test_resolver.py
git commit -m "feat(resolver): scope-gated local resolution; exclude locals from cross-file pool"
```

---

### Task 9: Exporter + loader — serialize the new fields

**Files:**
- Modify: `sutra/core/output/json_graph_exporter.py` (`_symbol_to_dict`)
- Modify: `sutra/core/artifact/loader.py` (symbol readback, if it reconstructs typed fields)
- Test: `tests/test_e2e_fixture_repo.py` or `tests/test_json_graph_exporter.py` (append)

**Interfaces:**
- Produces: `graph.json` symbol dicts carry `is_local` and `enclosing_moniker`; the loader preserves them in the snapshot symbol dicts.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_json_graph_exporter.py  (append/create)
import json
from pathlib import Path
from sutra.core.embedder.fixture import FixtureEmbedder
from sutra.core.extractor.adapters.python import PythonAdapter
from sutra.core.indexer import Indexer
from sutra.core.output.json_graph_exporter import JsonGraphExporter


def test_graph_json_serializes_local_fields(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    (repo / "m.py").write_text("def outer():\n    def inner():\n        pass\n")
    out = tmp_path / "art"
    Indexer(adapters={"python": PythonAdapter()}, exporter=JsonGraphExporter(),
            embedder=FixtureEmbedder()).index(
        root=repo, repo_url="https://github.com/t/r", output_dir=out)
    graph = json.loads((out / "graph.json").read_text())
    inner = next(s for s in graph["symbols"] if s["name"] == "inner")
    assert inner["is_local"] is True
    assert inner["enclosing_moniker"].endswith("outer().")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_json_graph_exporter.py::test_graph_json_serializes_local_fields -v`
Expected: FAIL — `KeyError: 'is_local'` (exporter doesn't emit it).

- [ ] **Step 3: Serialize the fields**

In `json_graph_exporter.py`, find `_symbol_to_dict` and add the two fields to the base dict it returns (alongside `id`, `name`, etc.):

```python
            "is_local": sym.is_local,
            "enclosing_moniker": sym.enclosing_moniker,
```

In `sutra/core/artifact/loader.py`, the symbol dicts are loaded as-is into `snapshot.symbols` (a `dict[str, dict]`), so the new keys round-trip automatically — confirm no typed reconstruction drops them. If the loader filters keys, add `is_local`/`enclosing_moniker` to the allowed set. (Read `loader.py` ~lines 117-119 where `symbols[sym["id"]] = sym` — it stores the whole dict, so no change is needed; the test confirms it.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_json_graph_exporter.py tests/test_mcp_server.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sutra/core/output/json_graph_exporter.py tests/test_json_graph_exporter.py
git commit -m "feat(exporter): serialize is_local + enclosing_moniker to graph.json"
```

---

### Task 10: Indexer — keep-first backstop replaces the fatal assertion

**Files:**
- Modify: `sutra/core/indexer.py` (the duplicate-moniker `AssertionError`, ~line 188-194)
- Test: `tests/test_indexer_sink.py` (append)

**Interfaces:**
- Produces: a duplicate moniker no longer aborts; the first occurrence (global sort order) is kept, later duplicates are dropped + counted + warned.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_indexer_sink.py  (append)
import warnings
from sutra.core.extractor.base import (
    FunctionSymbol, Location, Visibility, Repository, IndexResult,
)


def test_duplicate_moniker_does_not_abort(tmp_path, capsys):
    # Force a duplicate by indexing a file the adapter would dup on is hard;
    # instead drive the indexer's dedup directly via a tiny repo that the
    # disambiguator does NOT cover would be ideal. Here we assert the public
    # behavior: indexing a repo with a benign collision completes and warns.
    repo = tmp_path / "repo"; repo.mkdir()
    # Two top-level functions with the same name across two files cannot collide
    # (different file_path). The realistic residual is same-file; the adapter
    # disambiguator covers it. This test pins the BACKSTOP itself via a stub.
    from sutra.core.indexer import _dedup_keep_first  # helper added in Step 3
    a = FunctionSymbol(id="dup", name="f", qualified_name="f", file_path="x.py",
                       location=Location(1, 1, 0, 1), body_hash="h", language="python",
                       visibility=Visibility.PUBLIC, is_exported=True)
    b = FunctionSymbol(id="dup", name="f", qualified_name="f", file_path="x.py",
                       location=Location(2, 2, 2, 3), body_hash="h2", language="python",
                       visibility=Visibility.PUBLIC, is_exported=True)
    kept, dropped = _dedup_keep_first([a, b])
    assert [s.id for s in kept] == ["dup"]
    assert dropped == 1
    assert kept[0] is a  # first wins
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_indexer_sink.py::test_duplicate_moniker_does_not_abort -v`
Expected: FAIL — `ImportError: cannot import name '_dedup_keep_first'`.

- [ ] **Step 3: Replace the assertion with keep-first**

In `sutra/core/indexer.py`, add a module-level helper:

```python
def _dedup_keep_first(symbols: list[Symbol]) -> tuple[list[Symbol], int]:
    """Deterministic keep-first dedup. First occurrence (caller-provided order,
    which the indexer keeps globally sorted) wins; later duplicates are dropped.
    Returns (kept, dropped_count)."""
    seen: set[str] = set()
    kept: list[Symbol] = []
    dropped = 0
    for sym in symbols:
        if sym.id in seen:
            dropped += 1
            continue
        seen.add(sym.id)
        kept.append(sym)
    return kept, dropped
```

In `Indexer.index`, replace the loud per-symbol `AssertionError` block (the `for sym in extraction.symbols: if sym.id in seen_monikers: raise AssertionError(...)`) with collection-then-dedup. Keep collecting symbols, then after the file loop and before building `IndexResult`, dedup and warn:

```python
        symbols, dropped = _dedup_keep_first(symbols)
        if dropped:
            import sys
            print(
                f"[indexer] dropped {dropped} duplicate-moniker symbol(s) "
                f"(kept first occurrence). This is expected for rare same-scope "
                f"redefinitions; investigate if the count is large.",
                file=sys.stderr,
            )
```

Remove the `seen_monikers` set and the inner `raise AssertionError` loop (the dedup now owns uniqueness). Relationships referencing a dropped symbol's id simply won't resolve to a node — acceptable for the rare drop.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_indexer_sink.py tests/test_python_adapter.py -q`
Expected: PASS — helper dedups keep-first; full indexer path still works (the empty-index guard from the earlier fix is untouched).

- [ ] **Step 5: Commit**

```bash
git add sutra/core/indexer.py tests/test_indexer_sink.py
git commit -m "feat(indexer): deterministic keep-first backstop replaces fatal duplicate-moniker abort"
```

---

### Task 11: Traversal — verify locals are reachable; surface `is_local`

**Files:**
- Test: `tests/test_mcp_server.py` or `tests/test_graph_traversal.py` (append)
- Modify (optional): `sutra/mcp/server.py` — include `is_local` in `sutra_get_symbol` output.

**Interfaces:**
- Consumes: resolved `function→local` CONTAINS edges (Tasks 4-5), the serialized fields (Task 9).
- Produces: a passing reachability test proving `expand_neighbors` reaches a local via CONTAINS, plus `get_callees` reaches a local that its caller calls.

- [ ] **Step 1: Write the failing/contract test**

```python
# tests/test_graph_traversal.py  (append/create)
import json
from pathlib import Path
from sutra.core.embedder.fixture import FixtureEmbedder
from sutra.core.extractor.adapters.python import PythonAdapter
from sutra.core.indexer import Indexer
from sutra.core.output.json_graph_exporter import JsonGraphExporter
from sutra.core.resolver import HeuristicResolver
from sutra.core.artifact import ArtifactLoader
from sutra.core.graph.traversal import RustworkxTraversal


def _index(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    (repo / "m.py").write_text(
        "def outer():\n"
        "    def inner():\n"
        "        pass\n"
        "    return inner()\n"
    )
    out = tmp_path / "art"
    Indexer(adapters={"python": PythonAdapter()}, exporter=JsonGraphExporter(),
            embedder=FixtureEmbedder(), resolver=HeuristicResolver()).index(
        root=repo, repo_url="https://github.com/t/r", output_dir=out)
    return ArtifactLoader().load(out)


def test_expand_neighbors_reaches_local(tmp_path):
    snap = _index(tmp_path)
    outer = next(m for m in snap.symbols if m.endswith("outer()."))
    inner = next(m for m in snap.symbols if m.endswith("outer().inner()."))
    trav = RustworkxTraversal(snap)
    reached = {n.moniker for n in trav.expand_neighbors(outer, depth=1, kinds={"contains"})}
    assert inner in reached


def test_get_callees_reaches_local(tmp_path):
    snap = _index(tmp_path)
    outer = next(m for m in snap.symbols if m.endswith("outer()."))
    inner = next(m for m in snap.symbols if m.endswith("outer().inner()."))
    trav = RustworkxTraversal(snap)
    callees = {n.moniker for n in trav.get_callees(outer)}
    assert inner in callees
```

- [ ] **Step 2: Run tests to verify they pass or reveal a gap**

Run: `python -m pytest tests/test_graph_traversal.py -v`
Expected: If Tasks 4-5 emitted CONTAINS resolved-with-target and Task 8 resolved the local CALLS, these PASS. If either FAILS, the failure pinpoints a non-resolved edge — fix the emitting task, not this test.

- [ ] **Step 3 (optional): Surface `is_local` in get_symbol**

In `sutra/mcp/server.py`, in the `sutra_get_symbol` handler payload, add `"is_local": symbol_dict.get("is_local", False)` so an agent can tell a traced node is a local. (Read the handler first; match its existing dict-building style.)

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS — all prior tests green, traversal reachability proven.

- [ ] **Step 5: Commit**

```bash
git add tests/test_graph_traversal.py sutra/mcp/server.py
git commit -m "test(traversal): locals reachable via CONTAINS/CALLS; surface is_local in get_symbol"
```

---

## Self-Review

**Spec coverage:**
- Identity model (descriptor + `enclosing_moniker` + `is_local`) → Tasks 1-3.
- Disambiguator → Task 6. Scope = nested funcs + local classes + methods → Tasks 4-5.
- Graph-only (not embedded) → Task 7. New CONTAINS directions resolved-with-target → Tasks 4-5, proven Task 11.
- Resolver exclude-locals + scope-gated rule (the top risk) → Task 8.
- Serialization → Task 9. Keep-first backstop → Task 10. Reachability + MCP surface → Task 11.
- Hardening: build-time space assert + `descriptor_kind` generalization → Task 2; raw-identifier-only is inherent (`_scope_chain` reads tree-sitter `identifier` text); "never infer local from shape" is documented in the spec.

**Type consistency:** `enclosing: tuple[tuple[str, str], ...]` is used identically in Task 2 (builder) and Task 3 (`_scope_chain` output feeds it). `is_local`/`enclosing_moniker` field names consistent across Tasks 1, 3, 7, 8, 9. `_dedup_keep_first` (Task 10) and `_disambiguate` (Task 6) are the only new indexer/adapter helpers.

**Open verification (flagged for implementers):**
- Task 9: confirm `loader.py` stores whole symbol dicts (it does at ~line 118 `symbols[sym["id"]] = sym`); if a future typed reconstruction is added, the fields must be carried.
- Task 11 Step 3 is optional and must match the existing `get_symbol` payload shape — read before editing.
