export const meta = {
  name: 'fresh-ab-main',
  description: 'Fresh sutra-vs-control A/B: 20 tickets x 2 arms x 3 trials at budget 6, blind judged',
  phases: [
    { title: 'Solve', detail: '20 tickets x 2 arms x 3 trials = 120 haiku solvers' },
    { title: 'Judge', detail: '1 blind sonnet judge per ticket over 6 shuffled answers' },
  ],
}

const TICKETS = [
 {
  "id": "BR1",
  "repo": "fastapi",
  "title": "Add a tracing-context argument to solve_dependencies()",
  "text": "We want request-scoped dependency resolution to carry a tracing context object through, so we're adding a required `trace_ctx` parameter to `solve_dependencies()` in fastapi/dependencies/utils.py. Before I touch the signature, walk me through every place in the codebase that actually calls `solve_dependencies` today (not just anything with a similar name) and what each call site would need to pass for the new argument.",
  "gold": "solve_dependencies is an async module-level function defined at fastapi/dependencies/utils.py:586. It has 4 genuine call sites: (1) fastapi/routing.py:481, inside the inner `app()` coroutine returned by get_request_handler, for the normal HTTP request-handling path; (2) fastapi/routing.py:783, inside the inner `app()` coroutine returned by get_websocket_app, for websocket handling; (3) fastapi/routing.py:2234, inside `_FrontendRouteGroup._solve_dependencies` (an async contextmanager method on a *different*, similarly-named symbol) which itself awaits the real solve_dependencies function; (4) fastapi/dependencies/utils.py:640, a recursive call inside solve_dependencies itself while resolving each sub-dependant. The two other grep hits for the substring 'solve_dependencies(' are decoys: fastapi/routing.py:2193 (`self._solve_dependencies(...)`, a call to the *different* method `_FrontendRouteGroup._solve_dependencies`, not to the target function) and fastapi/routing.py:2216 (`async def _solve_dependencies(...)`, the definition line of that different method).",
  "gold_markers": [
   "solve_dependencies",
   "get_request_handler",
   "get_websocket_app",
   "_FrontendRouteGroup"
  ],
  "marker_hit_counts": {
   "solve_dependencies": 7,
   "get_request_handler": 3,
   "get_websocket_app": 2,
   "_FrontendRouteGroup": 6
  },
  "caller_set": [
   "fastapi/routing.py:481 (inner app() coroutine in get_request_handler)",
   "fastapi/routing.py:783 (inner app() coroutine in get_websocket_app)",
   "fastapi/routing.py:2234 (_FrontendRouteGroup._solve_dependencies method body)",
   "fastapi/dependencies/utils.py:640 (recursive self-call inside solve_dependencies)"
  ],
  "decoys": [
   "fastapi/routing.py:2193 (self._solve_dependencies(...) \u2014 calls the differently-named method _FrontendRouteGroup._solve_dependencies, not the target function)",
   "fastapi/routing.py:2216 (async def _solve_dependencies(...) \u2014 definition of that different method, matches the substring but is not a call)"
  ],
  "primary_paths": [
   "fastapi/dependencies/utils.py",
   "fastapi/routing.py"
  ],
  "why_this_is_hard": "The target function solve_dependencies and an unrelated method _solve_dependencies (a bound method on _FrontendRouteGroup that internally calls solve_dependencies) share a substring, so a naive grep for 'solve_dependencies(' returns 7 hits mixing calls to two different symbols plus one definition line. An agent must read each hit's surrounding code to tell 'self._solve_dependencies(' (calls the wrapper method) apart from bare 'solve_dependencies(' (calls the target), and must also notice the recursive call inside the function's own body and the two independent call sites (HTTP vs websocket) hidden inside factory-returned closures in routing.py.",
  "verified_how": "Ran `grep -rn \"solve_dependencies(\" --include=\"*.py\" .` from the fastapi repo root (7 hits total, whole-repo including tests/docs_src returned none extra), then read fastapi/dependencies/utils.py:580-650 and fastapi/routing.py:375-410, 764-800, 2103-2245 to confirm which hits are calls to the module-level solve_dependencies function versus calls/definition of the unrelated _FrontendRouteGroup._solve_dependencies method.",
  "measured_marker_hits": {
   "solve_dependencies": 10,
   "get_request_handler": 3,
   "get_websocket_app": 2,
   "_FrontendRouteGroup": 6
  }
 },
 {
  "id": "BR2",
  "repo": "requests",
  "title": "Change select_proxy() to return a small result object instead of a bare string",
  "text": "I want to change `select_proxy()` in requests/utils.py so it returns a small object (proxy URL plus which config layer it came from) instead of a bare string or None, since we keep needing to know where a proxy setting came from for debugging. Enumerate every call site in the codebase so I know what needs to be updated to unpack the new return value.",
  "gold": "select_proxy is defined at src/requests/utils.py:885. It has 4 genuine call sites: three inside the HTTPAdapter class in src/requests/adapters.py \u2014 get_connection_with_tls_context (line 483), get_connection (line 535), and request_url (line 583), each assigning `proxy = select_proxy(...)` \u2014 plus one direct call in the test suite, tests/test_utils.py:565, which asserts `select_proxy(url, proxies) == expected`.",
  "gold_markers": [
   "select_proxy",
   "get_connection_with_tls_context",
   "prepend_scheme_if_needed"
  ],
  "marker_hit_counts": {
   "select_proxy": 7,
   "get_connection_with_tls_context": 4,
   "prepend_scheme_if_needed": 5
  },
  "caller_set": [
   "src/requests/adapters.py:483 (HTTPAdapter.get_connection_with_tls_context)",
   "src/requests/adapters.py:535 (HTTPAdapter.get_connection)",
   "src/requests/adapters.py:583 (HTTPAdapter.request_url)",
   "tests/test_utils.py:565 (direct test call)"
  ],
  "decoys": [
   "src/requests/adapters.py:58 (import line: `select_proxy,` in the from .utils import block \u2014 no call)",
   "tests/test_utils.py:36 (import line in the test module \u2014 no call)"
  ],
  "primary_paths": [
   "src/requests/utils.py",
   "src/requests/adapters.py"
  ],
  "why_this_is_hard": "select_proxy's name shows up at both its two import sites (adapters.py and the test module) as well as its three real call sites inside HTTPAdapter and one real call inside a test \u2014 a naive grep returns 7 hits and the agent has to distinguish 'imported but only referenced elsewhere in the file' from 'actually invoked on this line', across three different HTTPAdapter methods in one file plus a separate test file.",
  "verified_how": "Ran `grep -rn \"select_proxy\" --include=\"*.py\" .` from the requests repo root (7 hits), then read src/requests/adapters.py around lines 450-590 to confirm the three call sites sit in three distinct HTTPAdapter methods, and confirmed line 58 is only the import. Also checked tests/test_utils.py:30-40 and :560-566 to confirm the test both imports and directly calls the function.",
  "measured_marker_hits": {
   "select_proxy": 7,
   "get_connection_with_tls_context": 5,
   "prepend_scheme_if_needed": 8
  }
 },
 {
  "id": "BR3",
  "repo": "pydantic",
  "title": "Add a structured-reason argument to set_model_mocks()",
  "text": "Right now set_model_mocks() in pydantic/_internal/_mock_val_ser.py takes a free-text `undefined_name` string to build its error message. I want to replace that with a required `reason` object carrying the offending name plus a machine-readable code, instead of a plain string. Walk me through every call site in the codebase so I know what each one needs to construct and pass.",
  "gold": "set_model_mocks is defined at pydantic/_internal/_mock_val_ser.py:151. It has 4 genuine call sites, all inside pydantic/_internal/_model_construction.py: line 249, inside ModelMetaclass.__new__, when config_wrapper.defer_build is set; and three inside complete_model_class \u2014 line 611 (NameError handler around rebuild_model_fields), line 632 (PydanticUndefinedAnnotation handler around gen_schema.generate_schema), and line 640 (InvalidSchemaError handler around gen_schema.clean_schema).",
  "gold_markers": [
   "set_model_mocks",
   "set_dataclass_mocks",
   "complete_model_class"
  ],
  "marker_hit_counts": {
   "set_model_mocks": 6,
   "set_dataclass_mocks": 5,
   "complete_model_class": 6
  },
  "caller_set": [
   "pydantic/_internal/_model_construction.py:249 (ModelMetaclass.__new__, defer_build branch)",
   "pydantic/_internal/_model_construction.py:611 (complete_model_class, NameError handler)",
   "pydantic/_internal/_model_construction.py:632 (complete_model_class, PydanticUndefinedAnnotation handler)",
   "pydantic/_internal/_model_construction.py:640 (complete_model_class, InvalidSchemaError handler)"
  ],
  "decoys": [
   "pydantic/_internal/_model_construction.py:38 (import line: `from ._mock_val_ser import set_model_mocks` \u2014 no call)",
   "pydantic/_internal/_mock_val_ser.py:192 (def set_dataclass_mocks(...) \u2014 a sibling function for dataclasses with an almost-identical name/signature/purpose, easy to mistake for the target)",
   "pydantic/_internal/_dataclasses.py:131,169,177 (three calls to that sibling set_dataclass_mocks, not to set_model_mocks)"
  ],
  "primary_paths": [
   "pydantic/_internal/_mock_val_ser.py",
   "pydantic/_internal/_model_construction.py"
  ],
  "why_this_is_hard": "set_model_mocks lives next to an almost-twin function, set_dataclass_mocks, in the same file with the same (cls, undefined_name) signature and the same call pattern (three exception handlers plus a defer_build branch) mirrored in a different module (_dataclasses.py for dataclasses vs _model_construction.py for BaseModel). An agent has to actually read each call site to confirm it's operating on a `type[BaseModel]` / model-construction code path and not accidentally fold in the dataclass sibling's call sites, which look structurally identical.",
  "verified_how": "Ran `grep -rn \"set_model_mocks\" --include=\"*.py\" .` (6 hits: 1 def, 1 import, 4 calls) and `grep -rn \"set_dataclass_mocks\" --include=\"*.py\" .` (5 hits) from the pydantic repo root, then read pydantic/_internal/_mock_val_ser.py:100-230 and pydantic/_internal/_model_construction.py:200-260,560-645 to confirm which of the four set_model_mocks calls sit in ModelMetaclass.__new__ vs complete_model_class, and pydantic/_internal/_dataclasses.py:120-180 to confirm its three calls target the sibling function instead.",
  "measured_marker_hits": {
   "set_model_mocks": 6,
   "set_dataclass_mocks": 5,
   "complete_model_class": 6
  }
 },
 {
  "id": "BR4",
  "repo": "flask",
  "title": "Require an explicit max_age on send_from_directory()",
  "text": "I want to change send_from_directory() in flask/helpers.py so `max_age` becomes a required positional-or-keyword argument instead of being pulled implicitly out of **kwargs, since we've had callers silently forget to set caching behavior. Walk me through every place in the codebase that actually calls Flask's send_from_directory (not Werkzeug's own function of the same name) so I know what each needs to pass.",
  "gold": "Flask's send_from_directory is defined at src/flask/helpers.py:543. It has 3 genuine call sites: src/flask/app.py:411, inside Flask.send_static_file; src/flask/blueprints.py:100, inside Blueprint.send_static_file (an explicitly-documented duplicate of the Flask method); and tests/test_helpers.py:89, a direct call in test_send_from_directory. Both call sites in app.py and blueprints.py first compute `max_age = self.get_send_file_max_age(filename)` and pass it through as a keyword.",
  "gold_markers": [
   "send_from_directory",
   "send_static_file",
   "get_send_file_max_age"
  ],
  "marker_hit_counts": {
   "send_from_directory": 11,
   "send_static_file": 9,
   "get_send_file_max_age": 6
  },
  "caller_set": [
   "src/flask/app.py:411 (Flask.send_static_file)",
   "src/flask/blueprints.py:100 (Blueprint.send_static_file)",
   "tests/test_helpers.py:89 (test_send_from_directory)"
  ],
  "decoys": [
   "src/flask/helpers.py:582 (werkzeug.utils.send_from_directory(...) \u2014 a different function from the werkzeug package, called inside Flask's own send_from_directory as its implementation, not a call to itself)",
   "src/flask/helpers.py:437 (docstring cross-reference `:func:`send_from_directory`` inside send_file's docstring, not code)",
   "src/flask/helpers.py:554 (docstring code example inside send_from_directory's own docstring, not a real call)"
  ],
  "primary_paths": [
   "src/flask/helpers.py",
   "src/flask/app.py",
   "src/flask/blueprints.py"
  ],
  "why_this_is_hard": "Flask's send_from_directory both contains a docstring code sample that itself reads `send_from_directory(...)` and delegates its real work to `werkzeug.utils.send_from_directory`, a same-named function from a different package. A grep for the bare name pulls in that unrelated werkzeug call plus two docstring mentions alongside the three genuine call sites, which are themselves split across two nearly-identical duplicated methods (Flask.send_static_file and Blueprint.send_static_file, explicitly commented in the source as duplicates of each other) plus one test.",
  "verified_how": "Ran `grep -rn \"send_from_directory\" --include=\"*.py\" .` from the flask repo root (11 hits), then read src/flask/helpers.py:420-585 to see the werkzeug delegation and both docstring mentions, and src/flask/app.py:390-412 and src/flask/blueprints.py:85-101 to confirm the two send_static_file methods (explicitly noted in their own docstrings as duplicates) both call flask's send_from_directory, plus tests/test_helpers.py:84-90 for the direct test call.",
  "measured_marker_hits": {
   "send_from_directory": 24,
   "send_static_file": 11,
   "get_send_file_max_age": 14
  }
 },
 {
  "id": "BR5",
  "repo": "sutra",
  "title": "Rename get_callers on the graph traversal interface",
  "text": "I want to rename `get_callers` to `fetch_callers` on the graph traversal interface (sutra/core/graph/traversal.py) \u2014 it's a common word so I don't fully trust a quick search to have found everything. Walk me through every real call site in the codebase, as opposed to unrelated things that just happen to share the name, so I can rename them all consistently.",
  "gold": "get_callers is declared as an abstract method on GraphTraversal at sutra/core/graph/traversal.py:44 and implemented by RustworkxTraversal at traversal.py:87. It has 4 genuine call sites: sutra/mcp/server.py:324, inside the sutra_get_symbol tool's run() closure (building payload['callers']); sutra/mcp/server.py:344, inside the sutra_get_callers tool's own run() closure; tests/test_mcp_loader.py:334 (`callers = t.get_callers(generate_id)`); and tests/test_mcp_loader.py:353 (an assertion that get_callers on an unresolvable moniker returns []).",
  "gold_markers": [
   "get_callers",
   "GraphTraversal",
   "RustworkxTraversal",
   "sutra_get_callers"
  ],
  "marker_hit_counts": {
   "get_callers": 7,
   "GraphTraversal": 10,
   "RustworkxTraversal": 6,
   "sutra_get_callers": 4
  },
  "caller_set": [
   "sutra/mcp/server.py:324 (sutra_get_symbol tool's run() closure)",
   "sutra/mcp/server.py:344 (sutra_get_callers tool's run() closure)",
   "tests/test_mcp_loader.py:334",
   "tests/test_mcp_loader.py:353"
  ],
  "decoys": [
   "sutra/mcp/server.py:339 (`def sutra_get_callers(moniker: str) -> list[dict[str, Any]]:` \u2014 the MCP tool function's own definition, a different symbol whose name contains 'get_callers(' as a substring but is not a call to the traversal method)",
   "sutra/core/graph/traversal.py:44 (abstract method declaration on GraphTraversal \u2014 not a call)",
   "sutra/core/graph/traversal.py:87 (concrete override declaration on RustworkxTraversal \u2014 not a call)"
  ],
  "primary_paths": [
   "sutra/core/graph/traversal.py",
   "sutra/mcp/server.py",
   "tests/test_mcp_loader.py"
  ],
  "why_this_is_hard": "The MCP tool exposed to the outside world is itself named sutra_get_callers, whose definition line contains the exact substring 'get_callers(' \u2014 a decoy that looks exactly like an invocation until you notice it starts with 'def'. Inside that same tool's body it then genuinely calls unit.traversal.get_callers(...), so the same short window of code holds both a false positive and a true positive under the same name, and a third real call site is buried inside a different tool (sutra_get_symbol) that also builds a 'callers' field.",
  "verified_how": "Ran `grep -rn \"get_callers(\" --include=\"*.py\" . | grep -v .venv` from the sutra repo root (7 hits), then read sutra/mcp/server.py:308-365 to see both tool closures and confirm which lines are definitions vs calls, sutra/core/graph/traversal.py:31-95 to confirm the abstract/concrete declarations, and tests/test_mcp_loader.py:325-355 for the two direct test calls.",
  "measured_marker_hits": {
   "get_callers": 23,
   "GraphTraversal": 11,
   "RustworkxTraversal": 16,
   "sutra_get_callers": 12
  }
 },
 {
  "id": "XR1",
  "repo": "pydantic",
  "repos_required": [
   "pydantic",
   "flask",
   "sutra"
  ],
  "title": "Design review: how should each service resolve its own config defaults?",
  "text": "We're writing an internal style guide for how our Python services should manage configuration: a library-level default, an optional file on disk, and a caller override, with a defined precedence between them. Before we invent our own pattern, I want to survey how the tools we already depend on solve this internally, since a few of them clearly do it already (not runtime app config like a Django settings module \u2014 I mean the layer inside each library that decides 'what value actually applies here' when a default exists, a config source may or may not be present, and a caller might pass something explicit). Can someone trace, for pydantic, flask, and our own sutra indexer, exactly where in the source this resolution happens, what it's called internally, and what happens when nothing is supplied at all (does it error, or fall back silently)? I want file-level pointers, not just 'pydantic has ConfigDict'.",
  "gold": "Each project has its own internal defaults-resolution step, named and structured differently, and each behaves differently on 'nothing supplied': (1) pydantic \u2014 pydantic/_internal/_config.py defines class ConfigWrapper, whose classmethod for_model(bases, namespace, config_wrapper_class) walks a model's base classes and its own `model_config`/class-based Config, merging them (child overrides parent) into one resolved config dict before schema generation; there is always a hard-coded fallback (an empty/default ConfigDict) so this never errors on 'nothing supplied'. (2) flask \u2014 flask/src/flask/config.py defines Config(dict), populated via from_object/from_pyfile/from_envvar/from_prefixed_env; there is no single 'merge with a default' step \u2014 Flask seeds it once in App.__init__ from Flask.default_config and callers mutate/overwrite the same dict afterward, so 'nothing supplied' just means the hard-coded default_config values stand. (3) sutra \u2014 sutra/sutra/core/embedder/factory.py defines from_config(config_path) which loads config/sutra.yaml (if present) and delegates to from_dict(config); from_dict reads config.get('embedder') or {} and provider = embedder_cfg.get('provider', 'fixture') \u2014 if config_path is None or the file doesn't exist, from_config returns FixtureEmbedder() directly, i.e. sutra's fallback is a concrete default object, not just a default value.",
  "gold_markers": [
   "for_model",
   "from_pyfile",
   "embedder/factory.py"
  ],
  "primary_paths": [
   "pydantic/pydantic/_internal/_config.py",
   "flask/src/flask/config.py",
   "sutra/sutra/core/embedder/factory.py"
  ],
  "why_this_is_hard": "The three mechanisms look superficially similar ('config with defaults') but are structurally different: pydantic's is an inheritance-merge over class hierarchies computed once per model at schema-build time, flask's is a mutable dict seeded once and never re-merged, and sutra's is a factory function that returns a wholly different concrete object (not just a dict of values) when config is absent. An agent has to actually read for_model's merge logic, Config's from_* loaders, and from_config's None/missing-file branch to state the difference correctly rather than assuming they're all 'defaults dict + override dict'.",
  "verified_how": "Read pydantic/_internal/_config.py (ConfigWrapper.for_model, lines ~62-190); read flask/src/flask/config.py (Config class, ConfigAttribute, from_pyfile) and confirmed no merge-on-read step exists; read sutra/sutra/core/embedder/factory.py (from_config/from_dict, FixtureEmbedder fallback). Grep counts (repo source, excluding .venv/__pycache__): ConfigWrapper=44 in pydantic, 'def for_model'=1, 'from_pyfile'=5 in flask, 'def from_config'=1 and FixtureEmbedder=14 in sutra.",
  "measured_marker_hits": {
   "for_model": 4,
   "from_pyfile": 30,
   "embedder/factory.py": 6
  },
  "gold_markers_original": [
   "ConfigWrapper",
   "for_model",
   "from_pyfile",
   "from_config",
   "FixtureEmbedder"
  ],
  "marker_note": "markers tightened: originals exceeded 60 repo-wide hits and would credit localization on incidental mentions, biasing toward the arm with more verbose tool output"
 },
 {
  "id": "XR2",
  "repo": "requests",
  "repos_required": [
   "requests",
   "flask",
   "pydantic"
  ],
  "title": "What's the actual extension point in each of requests/flask/pydantic for running custom code at a fixed point in their processing?",
  "text": "I'm putting together onboarding notes on 'how to hook into X at the right moment' for the three libraries our services build on most: requests, flask, and pydantic. Each of them clearly lets a caller (or a third-party package) run extra code at a specific, well-defined point \u2014 requests lets you react when a response comes back, flask lets you run code before/around a request, and pydantic lets a third-party package observe validation as it happens. But I don't actually know the internal name or shape of any of these three mechanisms, and I suspect they're not structurally the same thing at all (one might be a simple callback list, another might be a full protocol class an external package implements). Can you pin down, for each of the three, where the registration happens, where the dispatch/invocation happens, and what a caller has to supply?",
  "gold": "All three are extension points but structurally distinct. (1) requests \u2014 requests/src/requests/hooks.py defines the fixed set of valid hook events (HOOKS) and dispatch_hook(key, hooks, hook_data, **kwargs), which looks up hooks.get(key) and calls each callable in the list, replacing hook_data with the hook's return value if non-None; hooks are just a dict of event-name -> list of plain callables, merged per-request with merge_hooks() in requests/src/requests/sessions.py. (2) flask \u2014 flask/src/flask/app.py maintains before_request_funcs / teardown_request_funcs dicts (keyed by blueprint name) populated via the @before_request/@teardown_request decorators (registered through Scaffold in flask/src/flask/sansio/scaffold.py); Flask.preprocess_request iterates and calls each registered function in order before the view runs, and a non-None return short-circuits the request \u2014 a callback-list pattern like requests', but keyed per-blueprint and short-circuiting. (3) pydantic \u2014 pydantic/plugin/__init__.py defines PydanticPluginProtocol, a Protocol a third-party package implements with a new_schema_validator method; pydantic/plugin/_loader.py discovers plugins via Python entry points, and pydantic/plugin/_schema_validator.py wraps every compiled validator so plugin-provided event handlers (BaseValidateHandlerProtocol) fire on validate_python/validate_json \u2014 a full class-based protocol invoked by the validator itself, not a list of ad hoc callables like the other two.",
  "gold_markers": [
   "dispatch_hook",
   "before_request_funcs",
   "PydanticPluginProtocol",
   "new_schema_validator"
  ],
  "primary_paths": [
   "requests/src/requests/hooks.py",
   "flask/src/flask/app.py",
   "pydantic/pydantic/plugin/_schema_validator.py"
  ],
  "why_this_is_hard": "Surface-level all three are 'hooks', but the actual shapes diverge sharply: requests is a flat dict-of-lists dispatched by string key, flask is per-blueprint callback lists with short-circuit semantics, and pydantic is a discoverable, class-based Protocol wired in at schema-compile time rather than per-call. Answering correctly requires reading dispatch_hook's return-value-replacement behavior, preprocess_request's short-circuit check, and how _schema_validator.py actually invokes the plugin's handler methods \u2014 none of which is guessable from the word 'hook' alone.",
  "verified_how": "Read requests/src/requests/hooks.py (dispatch_hook) and sessions.py (merge_hooks); read flask/src/flask/app.py (before_request_funcs, preprocess_request) and sansio/scaffold.py (before_request decorator); read pydantic/plugin/__init__.py (PydanticPluginProtocol) and _schema_validator.py (new_schema_validator usage). Grep counts (repo source): dispatch_hook=3 in requests, before_request_funcs=7 in flask, PydanticPluginProtocol=7 and new_schema_validator=3 in pydantic.",
  "measured_marker_hits": {
   "dispatch_hook": 4,
   "before_request_funcs": 7,
   "PydanticPluginProtocol": 15,
   "new_schema_validator": 19
  }
 },
 {
  "id": "XR3",
  "repo": "flask",
  "repos_required": [
   "fastapi",
   "flask",
   "requests"
  ],
  "title": "Where does each of fastapi/flask/requests actually turn a raised exception into something the caller sees?",
  "text": "We keep getting bitten by exceptions that don't look the way we expect by the time they reach the client \u2014 sometimes as a JSON body, sometimes as one of our own exception types instead of whatever the underlying library raised. I want to understand the actual mechanism (not the public 'register an error handler' docs page) for three libraries: fastapi, flask, and requests. Specifically: when something goes wrong deep inside, where's the piece of code that decides what the outward-facing representation of that failure is, and does each library have exactly one such place, or several?",
  "gold": "Three different mechanisms, and none of them is a single universal place. (1) fastapi \u2014 fastapi/exception_handlers.py defines http_exception_handler (turns a Starlette HTTPException into a JSONResponse with {'detail': exc.detail}) and request_validation_exception_handler (turns a RequestValidationError into a 422 JSONResponse); these are registered as defaults in fastapi/applications.py via self.exception_handlers.setdefault(HTTPException, http_exception_handler) and .setdefault(RequestValidationError, request_validation_exception_handler) \u2014 a type-keyed dict of async handler functions, overridable per exception type. (2) flask \u2014 flask/src/flask/app.py has two separate methods: handle_http_exception (only for werkzeug HTTPException subclasses, looks up a registered handler via _find_error_handler and falls back to returning the exception itself as the response) and handle_user_exception (for arbitrary exceptions, delegates to handle_http_exception if it's an HTTPException, otherwise looks up a handler registered via @errorhandler in sansio/scaffold.py or re-raises); so flask has an explicit two-tier split that fastapi doesn't. (3) requests \u2014 has no handler-registration mechanism at all; requests/src/requests/adapters.py's HTTPAdapter.send() wraps the actual network call in a series of except clauses that catch urllib3-internal exceptions (MaxRetryError, ProtocolError, _SSLError, ClosedPoolError, etc.) and re-raise them as requests' own exception types (ConnectionError, SSLError, ProxyError, RetryError) with `request=request` attached \u2014 translation happens inline at the call site, there's no dispatch table or override point.",
  "gold_markers": [
   "http_exception_handler",
   "handle_user_exception",
   "handle_http_exception",
   "MaxRetryError"
  ],
  "primary_paths": [
   "fastapi/fastapi/exception_handlers.py",
   "fastapi/fastapi/applications.py",
   "flask/src/flask/app.py",
   "requests/src/requests/adapters.py"
  ],
  "why_this_is_hard": "The three have genuinely different shapes: fastapi's is a type-keyed, overridable dict of handler functions; flask's is two cooperating methods (handle_http_exception vs handle_user_exception) with different fallback behavior depending on exception type; requests has no registration mechanism whatsoever, just inline except/raise translation buried inside HTTPAdapter.send(). An agent has to read applications.py to see how fastapi wires its defaults, read both flask methods to see the HTTPException-vs-other split, and read adapters.py's except chain to realize requests doesn't have a 'handler' concept at all \u2014 a wrong answer would just say 'they all have error handlers you register'.",
  "verified_how": "Read fastapi/exception_handlers.py and applications.py (setdefault registration around line 1003); read flask/src/flask/app.py (handle_http_exception line 833, handle_user_exception line 868) and sansio/scaffold.py (errorhandler line 606); read requests/adapters.py send() except chain (MaxRetryError -> RetryError/ProxyError/SSLError/ConnectionError). Grep counts (repo source): http_exception_handler=3 in fastapi, handle_http_exception=4 and handle_user_exception=3 in flask, MaxRetryError=2 in requests.",
  "measured_marker_hits": {
   "http_exception_handler": 9,
   "handle_user_exception": 3,
   "handle_http_exception": 4,
   "MaxRetryError": 2
  }
 },
 {
  "id": "XR4",
  "repo": "fastapi",
  "repos_required": [
   "fastapi",
   "flask",
   "sutra"
  ],
  "title": "Do fastapi, flask, and our sutra exporter all auto-convert things like datetimes to JSON, or do some of them require the caller to do it by hand?",
  "text": "We had a bug where a datetime value crashed json.dumps() in one service but serialized fine in another, and I want to know why before we standardize a pattern. Looking at fastapi, flask, and sutra's own graph-export code specifically: when a response/output needs to become JSON and it contains something json.dumps can't handle natively (a datetime, a UUID, a dataclass, etc.), does each of these have a general mechanism that walks the value and converts unsupported types automatically, or does the calling code have to convert those fields itself before serializing? I want to know exactly which of these three actually 'just works' for an arbitrary nested object versus which ones will still blow up unless you convert the field yourself.",
  "gold": "fastapi and flask both have a general, type-dispatching conversion mechanism; sutra does not. (1) fastapi \u2014 fastapi/encoders.py's jsonable_encoder() recursively walks any value (dict, list, BaseModel, dataclass, Enum, date/datetime, UUID, Decimal, Path, etc.) and converts it to a JSON-safe structure; it is applied automatically to response return values before serialization. (2) flask \u2014 flask/src/flask/json/provider.py defines a module-level _default(o) function (handling date, Decimal, uuid.UUID, dataclasses, and objects with an __html__ method) that is wired in as DefaultJSONProvider.default = staticmethod(_default) and passed to json.dumps(..., default=...) inside DefaultJSONProvider.dumps(), so any of those types anywhere in the object graph are converted automatically. (3) sutra \u2014 sutra/sutra/core/output/json_graph_exporter.py has no such general mechanism at all: JsonGraphExporter builds its output dict by calling datetime.now(timezone.utc).isoformat() explicitly at the one call site that needs it (the 'generated_at' field), then passes the dict straight to json.dumps(); if a new datetime-typed field were added anywhere else in the exported symbol/relationship structures, it would raise TypeError unless someone added another explicit .isoformat() call \u2014 there is no default= callback or recursive walk.",
  "gold_markers": [
   "fastapi/encoders.py",
   "def jsonable_encoder",
   "DefaultJSONProvider",
   "json_graph_exporter.py"
  ],
  "primary_paths": [
   "fastapi/fastapi/encoders.py",
   "flask/src/flask/json/provider.py",
   "sutra/sutra/core/output/json_graph_exporter.py"
  ],
  "why_this_is_hard": "It's tempting to assume 'JSON serialization' is the same solved problem everywhere; the real answer requires noticing that sutra's exporter has zero generic conversion logic (just one hard-coded .isoformat() call at one field) while fastapi and flask both have genuine type-registries/recursive walkers, and even between those two the wiring differs (recursive encoder function vs. a json.dumps(default=...) callback attached to a provider class). Confirming sutra's gap requires reading the whole exporter file and noticing the absence of any dispatch mechanism, not just finding one grep hit for 'isoformat'.",
  "verified_how": "Read fastapi/encoders.py (jsonable_encoder signature and type dispatch); read flask/src/flask/json/provider.py (_default function line 108, DefaultJSONProvider.default line 138); read sutra/sutra/core/output/json_graph_exporter.py in full and confirmed the single inline isoformat() call at the generated_at field with no generic encoder. Grep counts (repo source): jsonable_encoder=24 in fastapi, DefaultJSONProvider=5 in flask, isoformat=1 in sutra/core/output (4 repo-wide, all inline call sites, none a registered default callback).",
  "repaired": true,
  "measured_marker_hits": {
   "fastapi/encoders.py": 3,
   "def jsonable_encoder": 1,
   "DefaultJSONProvider": 9,
   "json_graph_exporter.py": 12
  },
  "gold_markers_original": [
   "jsonable_encoder",
   "DefaultJSONProvider",
   "JsonGraphExporter",
   "generated_at"
  ],
  "marker_note": "markers tightened: originals exceeded 60 repo-wide hits and would credit localization on incidental mentions, biasing toward the arm with more verbose tool output"
 },
 {
  "id": "XR5",
  "repo": "requests",
  "repos_required": [
   "pydantic",
   "fastapi",
   "requests",
   "flask"
  ],
  "title": "How do pydantic, fastapi, requests, and flask actually warn callers about deprecated functionality \u2014 do they all use plain DeprecationWarning?",
  "text": "We're deciding how our own libraries should signal 'this still works but is going away' to callers, and I want to base it on what our dependencies actually do rather than guessing. For pydantic, fastapi, requests, and flask specifically: does each one raise the stdlib DeprecationWarning directly, or does it define its own warning class? If it defines its own, what's it called, where does it live, and does it carry any extra structure (e.g. version info) beyond a message string? I've heard stdlib DeprecationWarning is silently ignored by default in a lot of contexts, so I'm especially curious which of these libraries worked around that and how.",
  "gold": "Four different approaches. (1) pydantic \u2014 pydantic/warnings.py defines PydanticDeprecationWarning(DeprecationWarning), which carries structured `since: tuple[int, int]` and `expected_removal` version fields on the instance, plus a family of per-version subclasses (PydanticDeprecatedSince20, PydanticDeprecatedSince26, etc.) so call sites can express exactly when a feature was deprecated. (2) fastapi \u2014 fastapi/exceptions.py defines FastAPIDeprecationWarning(UserWarning) \u2014 deliberately NOT a DeprecationWarning subclass, with an explicit docstring reason: 'A custom deprecation warning as DeprecationWarning is ignored'; it's raised via warnings.warn(..., category=FastAPIDeprecationWarning) from fastapi/params.py, fastapi/responses.py, and fastapi/utils.py. (3) requests \u2014 requests/src/requests/exceptions.py defines RequestsWarning(Warning) as a base class and FileModeWarning(RequestsWarning, DeprecationWarning) for one specific case (file opened in text mode), but most other deprecation sites in requests (utils.py, adapters.py, auth.py) just call warnings.warn(..., DeprecationWarning) directly with the plain stdlib class, not RequestsWarning \u2014 so requests is inconsistent, unlike the other three. (4) flask \u2014 has no custom warning class anywhere; every deprecation site (e.g. flask/src/flask/app.py's should_ignore_error, flask/src/flask/ctx.py's RequestContext access) calls warnings.warn(message, DeprecationWarning, stacklevel=...) directly with the stdlib class.",
  "gold_markers": [
   "PydanticDeprecationWarning",
   "FastAPIDeprecationWarning",
   "RequestsWarning",
   "FileModeWarning"
  ],
  "primary_paths": [
   "pydantic/pydantic/warnings.py",
   "fastapi/fastapi/exceptions.py",
   "requests/src/requests/exceptions.py",
   "flask/src/flask/app.py"
  ],
  "why_this_is_hard": "The interesting finding is a genuine divergence, not a shared pattern: pydantic has a rich, version-aware warning hierarchy; fastapi deliberately avoids subclassing DeprecationWarning at all (and says why, in a docstring an agent has to actually open); requests has a custom base class but uses it inconsistently, falling back to plain DeprecationWarning at several call sites; and flask has no custom class whatsoever. Getting this right requires reading fastapi/exceptions.py's docstring rationale, checking multiple requests call sites (not just exceptions.py) to notice the inconsistency, and confirming flask truly has zero custom warning classes by checking several files, not stopping after finding the first plain DeprecationWarning.",
  "verified_how": "Read pydantic/warnings.py (PydanticDeprecationWarning class and since/expected_removal fields); read fastapi/exceptions.py (FastAPIDeprecationWarning(UserWarning) at line 252 with its docstring) and confirmed usage in params.py/responses.py/utils.py; read requests/exceptions.py (RequestsWarning line 153, FileModeWarning line 157) and grepped utils.py/adapters.py/auth.py to confirm plain DeprecationWarning is used elsewhere too; read flask/app.py line 1009 and flask/ctx.py line 535, confirmed both use plain DeprecationWarning with no custom subclass anywhere in flask/src/flask. Grep counts (repo source): PydanticDeprecationWarning=17 in pydantic, FastAPIDeprecationWarning=13 in fastapi, RequestsWarning=3 and FileModeWarning=6 in requests.",
  "measured_marker_hits": {
   "PydanticDeprecationWarning": 26,
   "FastAPIDeprecationWarning": 45,
   "RequestsWarning": 3,
   "FileModeWarning": 6
  }
 },
 {
  "id": "GF1",
  "repo": "flask",
  "grep_favorable_kind": "verbatim_identifier",
  "title": "Document what SecureCookieSessionInterface actually does",
  "text": "For our security review write-up I need a short explanation of Flask's SecureCookieSessionInterface class: what it does with the session data, how it signs/serializes it, and where in the codebase it's defined and wired up as the default.",
  "gold": "SecureCookieSessionInterface is defined in src/flask/sessions.py (subclass of SessionInterface). It stores the whole session dict client-side in a signed cookie: get_signing_serializer() builds an itsdangerous URLSafeTimedSerializer using the app's secret_key (plus any SECRET_KEY_FALLBACKS), the class attribute salt = 'cookie-session', key_derivation = 'hmac', digest_method = staticmethod(_lazy_sha1), and serializer = session_json_serializer; open_session()/save_session() use that serializer to decode/encode the cookie payload (session_class = SecureCookieSession). It is instantiated as Flask's default in src/flask/app.py, where `session_interface: SessionInterface = SecureCookieSessionInterface()` is set as the class attribute Flask uses unless an app overrides it.",
  "gold_markers": [
   "SecureCookieSessionInterface",
   "get_signing_serializer",
   "cookie-session"
  ],
  "marker_hit_counts": {
   "SecureCookieSessionInterface": 7,
   "get_signing_serializer": 3,
   "cookie-session": 1
  },
  "primary_paths": [
   "src/flask/sessions.py",
   "src/flask/app.py"
  ],
  "why_this_is_hard": "Not hard to locate (that's the point of this class) but a symbol-level index that only stores signatures/docstrings may not surface the class-attribute-driven signing behavior (salt, key_derivation, digest_method) that's scattered as bare assignments rather than in the docstring, and may miss the app.py wiring entirely since that's just an attribute default, not a call.",
  "verified_how": "grep -rn SecureCookieSessionInterface across flask/*.py (7 hits, defined in sessions.py:284, imported+instantiated in app.py:46/253); read sessions.py lines 284-330 for get_signing_serializer/salt/key_derivation/digest_method/serializer; confirmed app.py:253 sets session_interface = SecureCookieSessionInterface().",
  "repaired": true,
  "measured_marker_hits": {
   "SecureCookieSessionInterface": 8,
   "get_signing_serializer": 3,
   "cookie-session": 1
  }
 },
 {
  "id": "GF2",
  "repo": "fastapi",
  "grep_favorable_kind": "string_literal",
  "title": "Track down where \"SSE 'id' must not contain null characters\" comes from",
  "text": "A background worker crashed with `ValueError: SSE \\'id\\' must not contain null characters`. I need to know exactly what code raises this and what input condition triggers it, so I can add a guard before we construct the event.",
  "gold": "Raised in fastapi/sse.py by the helper `_check_id_valid(v)` (around line 47-49): if the string contains a null byte ('\\0' in v) it raises `ValueError(\"SSE 'id' must not contain null characters\")` (and otherwise still enforces single-line via `_check_single_line`). This function is wired as `AfterValidator(_check_id_valid)` on the `id: str | None` field of the `ServerSentEvent` Pydantic model (also in fastapi/sse.py), so it fires the moment a caller constructs `ServerSentEvent(id=...)` with a null character in the id, before the event is ever streamed via `EventSourceResponse`.",
  "gold_markers": [
   "_check_id_valid",
   "must not contain null characters"
  ],
  "marker_hit_counts": {
   "_check_id_valid": 1,
   "ServerSentEvent": 4,
   "must not contain null characters": 1
  },
  "primary_paths": [
   "fastapi/sse.py"
  ],
  "why_this_is_hard": "Grep on the exact message finds the raise instantly, but understanding *when* it fires requires connecting the private validator function to the `AfterValidator(...)` annotation on the `id` field two dozen lines away inside the ServerSentEvent model definition -- a wiring pattern (decorator-as-annotation-argument) that call-graph extraction tends to miss since it's not a normal function call.",
  "verified_how": "grep -rn \"must not contain null characters\" . -> single hit fastapi/sse.py:48; grep -n _check_id_valid fastapi/sse.py -> defined line 47, referenced in AfterValidator(_check_id_valid) on the id field; confirmed only 1 file defines it and ServerSentEvent is constructed in many places outside sse.py (tests, docs_src tutorials); the question concerns the id-field validator, not construction sites.",
  "repaired": true,
  "measured_marker_hits": {
   "_check_id_valid": 2,
   "must not contain null characters": 1
  },
  "gold_markers_original": [
   "_check_id_valid",
   "ServerSentEvent",
   "must not contain null characters"
  ],
  "marker_note": "markers tightened: originals exceeded 60 repo-wide hits and would credit localization on incidental mentions, biasing toward the arm with more verbose tool output"
 },
 {
  "id": "GF3",
  "repo": "requests",
  "grep_favorable_kind": "non_code_file",
  "title": "Confirm requests' declared urllib3 constraint and build backend before we vendor it",
  "text": "Before we vendor `requests` into our monorepo I need two packaging facts confirmed: the exact version range it declares for its `urllib3` dependency, and which build backend is configured to package the project.",
  "gold": "In pyproject.toml: under [project].dependencies the constraint is `\"urllib3>=1.26,<3\"` (alongside charset_normalizer>=2,<4, idna>=2.5,<4, certifi>=2023.5.7). The build backend is declared in the [build-system] table: `requires = [\"setuptools>=61.0\"]` and `build-backend = \"setuptools.build_meta\"`.",
  "gold_markers": [
   "urllib3>=1.26,<3",
   "setuptools.build_meta",
   "build-system"
  ],
  "marker_hit_counts": {
   "urllib3>=1.26,<3": 1,
   "setuptools.build_meta": 1,
   "build-system": 1
  },
  "primary_paths": [
   "pyproject.toml"
  ],
  "why_this_is_hard": "Trivially answered by opening one config file with grep/cat, but a symbol-oriented code index has nothing to extract from pyproject.toml at all -- there are no functions/classes in it -- so it's a clean test of whether the index falls back sanely to plain file search versus grep having no such gap.",
  "verified_how": "cat pyproject.toml lines 1-25; grep -n 'urllib3' pyproject.toml -> line 21 exact string 'urllib3>=1.26,<3'; grep -n 'build-backend' pyproject.toml -> line 3 'setuptools.build_meta'; grep -rn 'urllib3>=1.26,<3' . -> only 1 hit in the whole repo.",
  "measured_marker_hits": {
   "urllib3>=1.26,<3": 1,
   "setuptools.build_meta": 1,
   "build-system": 1
  }
 },
 {
  "id": "GF4",
  "repo": "fastapi",
  "grep_favorable_kind": "module_constant",
  "title": "Find what controls the SSE keep-alive ping interval",
  "text": "Our SSE endpoints (built with EventSourceResponse) periodically send a `: ping` comment line to idle clients to keep proxies from timing the connection out. What value controls how often that ping fires, and where in the code is it actually applied to the stream?",
  "gold": "The interval is the module-level constant `_PING_INTERVAL: float = 15.0` defined in fastapi/sse.py, alongside `KEEPALIVE_COMMENT = b': ping\\n\\n'`. Both are imported into fastapi/routing.py, where the SSE streaming loop wraps each wait for the next event in `with anyio.fail_after(_PING_INTERVAL):` -- when that timeout fires (i.e. no real event arrived within 15 seconds), the loop sends `KEEPALIVE_COMMENT` on `send_keepalive` instead. Changing `_PING_INTERVAL` changes how often every SSE response emits keep-alive pings.",
  "gold_markers": [
   "_PING_INTERVAL",
   "KEEPALIVE_COMMENT",
   "send_keepalive"
  ],
  "marker_hit_counts": {
   "_PING_INTERVAL": 5,
   "KEEPALIVE_COMMENT": 4,
   "send_keepalive": 4
  },
  "primary_paths": [
   "fastapi/sse.py",
   "fastapi/routing.py"
  ],
  "why_this_is_hard": "It's a bare float assignment with no docstring and no function signature, so symbol extraction that targets functions/classes is prone to skip it entirely; also its actual effect only becomes clear by following the import into routing.py's anyio.fail_after(...) call, a second file a value-only search wouldn't necessarily surface.",
  "verified_how": "grep -n '_PING_INTERVAL\\|KEEPALIVE_COMMENT' fastapi/sse.py -> defined lines 237/241; grep -rn '_PING_INTERVAL\\|KEEPALIVE_COMMENT' --include=*.py . -> used in fastapi/routing.py lines 78-79 (import), 596 (anyio.fail_after(_PING_INTERVAL)), 600 (send_keepalive.send(KEEPALIVE_COMMENT)); confirmed via tests/test_sse.py which monkeypatches fastapi.routing._PING_INTERVAL.",
  "measured_marker_hits": {
   "_PING_INTERVAL": 5,
   "KEEPALIVE_COMMENT": 4,
   "send_keepalive": 4
  }
 },
 {
  "id": "GF5",
  "repo": "sutra",
  "grep_favorable_kind": "index_staleness",
  "title": "Where does the new 'median cache-write tokens' line in the A/B report come from?",
  "text": "The battle-test efficiency report (benchmarks/battle_test/analyze_ab_transcripts.py) now prints a 'median cache-write tokens' line per arm that wasn't there before. Where does that number actually get computed from the raw transcript, and which field of the usage record does it read?",
  "gold": "In benchmarks/battle_test/analyze_ab_transcripts.py's analyze_file(), a per-transcript counter `cache_write_tokens = 0` is accumulated line by line via `cache_write_tokens += usage.get(\"cache_creation_input_tokens\", 0) or 0` (reading the same `usage` dict already used for output_tokens), then returned in the result dict as `\"cache_write_tokens\": cache_write_tokens`. In main(), the summary loop prints it with `print(f\"  median cache-write tokens: {med([r['cache_write_tokens'] for r in rs])}\")`. So it is the per-file sum of each transcript record's `usage.cache_creation_input_tokens`, then medianed across trials per arm.",
  "gold_markers": [
   "cache_write_tokens",
   "cache_creation_input_tokens",
   "median cache-write tokens"
  ],
  "marker_hit_counts": {
   "cache_write_tokens": 4,
   "cache_creation_input_tokens": 2,
   "median cache-write tokens": 1
  },
  "primary_paths": [
   "benchmarks/battle_test/analyze_ab_transcripts.py"
  ],
  "why_this_is_hard": "This mechanism was added in the 3 commits after the snapshot at 2a1f368 (confirmed via git diff 2a1f368..HEAD), so any index built from that older snapshot has no symbol/reference for cache_write_tokens at all and would either hallucinate or fail silently, while grep against the live working tree finds it immediately.",
  "verified_how": "git -C sutra diff 2a1f368..HEAD -- '*.py' shows cache_write_tokens/cache_creation_input_tokens added to analyze_ab_transcripts.py (absent before); git show 2a1f368:benchmarks/battle_test/analyze_ab_transcripts.py | grep cache_write_tokens -> no match, confirming absence at the snapshot commit; grep -n cache_write_tokens benchmarks/battle_test/analyze_ab_transcripts.py at HEAD shows the accumulation (line 56/65) and print (line 117).",
  "measured_marker_hits": {
   "cache_write_tokens": 28,
   "cache_creation_input_tokens": 2,
   "median cache-write tokens": 1
  }
 },
 {
  "id": "NT1",
  "repo": "flask",
  "title": "Session changes to a list value silently not saved",
  "text": "We store a small list of recently viewed item ids in the session and update it inside request handlers with `session['recent'].append(item_id)`. Locally it looked fine, but during a longer manual test we noticed the browser's session cookie never actually changes across requests \u2014 the `Set-Cookie` header is simply missing from the response, even though the value in memory during the request clearly has the new item appended. If we instead do `session['recent'] = session['recent'] + [item_id]` (reassigning the whole key), the cookie updates every time. We're using the default cookie-based session backend, nothing custom configured.",
  "gold": "SecureCookieSession (flask/sessions.py) wraps a werkzeug CallbackDict whose on_update callback sets session.modified = True only when the top-level mapping itself is mutated (__setitem__, __delitem__, etc.). Calling a method on a nested mutable object already stored in the session (e.g. list.append) does not go through the dict's own mutation methods, so on_update never fires and session.modified stays False. At the end of the request, Flask.process_response() (flask/app.py) calls self.session_interface.save_session(...), and SecureCookieSessionInterface.should_set_cookie() returns `session.modified or (permanent and SESSION_REFRESH_EACH_REQUEST)` \u2014 with modified False and refresh-each-request off by default, the cookie write is skipped entirely, so the mutated list is silently dropped.",
  "gold_markers": [
   "on_update",
   "should_set_cookie",
   "SecureCookieSession",
   "save_session"
  ],
  "marker_hit_counts": {
   "on_update": 2,
   "should_set_cookie": 2,
   "SecureCookieSession": 9,
   "save_session": 5
  },
  "primary_paths": [
   "src/flask/sessions.py",
   "src/flask/app.py"
  ],
  "repos_required": [
   "flask"
  ],
  "why_this_is_hard": "The symptom (missing Set-Cookie header) shows up in app.py's response pipeline, but the actual cause is a modification-tracking contract documented only as a code comment inside sessions.py's SecureCookieSession class \u2014 an agent has to connect should_set_cookie()'s modified check back to how modified gets set (or fails to get set) by the CallbackDict on_update hook, then realize in-place mutation of a nested value bypasses that hook.",
  "verified_how": "Read src/flask/sessions.py (SecureCookieSession, on_update, should_set_cookie) and src/flask/app.py (process_response calling session_interface.save_session); confirmed CallbackDict's on_update only fires on the dict's own MutableMapping methods, and grepped hit counts for all markers in src/.",
  "measured_marker_hits": {
   "on_update": 2,
   "should_set_cookie": 2,
   "SecureCookieSession": 13,
   "save_session": 6
  }
 },
 {
  "id": "NT2",
  "repo": "requests",
  "title": "Authorization header disappears after redirect to same host, different port",
  "text": "We're using a requests.Session and set the auth header directly: `session.headers['Authorization'] = f'Bearer {token}'`. Against our staging environment, the first request goes to `https://api.internal.example:8443/v1/...` and gets redirected (301) to `https://api.internal.example/v1/...` \u2014 same hostname, same scheme, just no explicit port. The redirected request comes back 401. Capturing traffic with a proxy shows the Authorization header is present on the first request and completely absent from the second one, even though we never touch headers between requests and allow_redirects is left at its default.",
  "gold": "Session.resolve_redirects() (src/requests/sessions.py) calls rebuild_auth(prepared_request, resp) on every redirect hop. rebuild_auth deletes the Authorization header whenever should_strip_auth(original_url, new_url) returns True. should_strip_auth first checks hostname equality, then carves out only the http(default port)->https(default port) upgrade case as safe; for any other case it falls through to `return changed_port or changed_scheme`. Going from an explicit non-default port (8443) to the implicit default port on the same host counts as a changed_port, so should_strip_auth returns True and the Authorization header is stripped even though the hostname never changed.",
  "gold_markers": [
   "should_strip_auth",
   "rebuild_auth",
   "DEFAULT_PORTS"
  ],
  "marker_hit_counts": {
   "should_strip_auth": 2,
   "rebuild_auth": 2,
   "DEFAULT_PORTS": 3
  },
  "primary_paths": [
   "src/requests/sessions.py",
   "src/requests/utils.py"
  ],
  "repos_required": [
   "requests"
  ],
  "why_this_is_hard": "An agent has to trace resolve_redirects -> rebuild_auth -> should_strip_auth across several nearby methods in sessions.py, and to understand *why* a same-host redirect strips auth it also has to check requests/utils.py for DEFAULT_PORTS to see that only the explicit port (8443) vs. implicit default port comparison triggers the changed_port branch \u2014 a same-host redirect is not automatically treated as safe.",
  "verified_how": "Read src/requests/sessions.py lines ~150-330 (should_strip_auth, resolve_redirects, rebuild_auth) and confirmed DEFAULT_PORTS is defined in src/requests/utils.py; traced the changed_port/changed_scheme fallthrough logic and grepped marker hit counts in src/.",
  "measured_marker_hits": {
   "should_strip_auth": 16,
   "rebuild_auth": 2,
   "DEFAULT_PORTS": 3
  }
 },
 {
  "id": "NT3",
  "repo": "fastapi",
  "title": "Dependency function only executes once even though it's wired in twice",
  "text": "We have a dependency `get_request_context()` that appends a line to an in-memory list every time it runs (used for a lightweight audit trail). One endpoint depends on it directly, and also depends on a second dependency, `get_current_user()`, which itself depends on `get_request_context()` to look up context. We expected two entries in the audit list per request \u2014 once for each place it's referenced \u2014 but only ever see one. This isn't an external cache: restarting the server fresh and hitting the endpoint exactly once still only produces a single entry.",
  "gold": "solve_dependencies() in fastapi/dependencies/utils.py maintains a dependency_cache dict scoped to a single request/dependency-tree resolution. For each sub-dependency it computes a cache key via _get_cache_key(dependant, uses_scopes_cache) and, when the Depends() was declared with use_cache=True (the default for Depends()), reuses whatever value is already in dependency_cache under that key instead of calling the function again. Because the same dependency_cache dict is threaded through recursive solve_dependencies() calls for every sub-dependency in the tree (invoked from fastapi/routing.py's request handling), a dependency referenced twice in one request's dependency graph is only actually invoked once, and the second reference just reads the cached result.",
  "gold_markers": [
   "use_cache",
   "dependency_cache",
   "_get_cache_key"
  ],
  "marker_hit_counts": {
   "use_cache": 12,
   "dependency_cache": 10,
   "_get_cache_key": 6
  },
  "primary_paths": [
   "fastapi/dependencies/utils.py",
   "fastapi/routing.py"
  ],
  "repos_required": [
   "fastapi"
  ],
  "why_this_is_hard": "The caching is per-request rather than global/process-wide, which rules out the obvious 'external cache' explanation the reporter already checked; finding it requires reading how routing.py invokes solve_dependencies() and then following the recursive dependency_cache dict and _get_cache_key logic inside dependencies/utils.py to see that use_cache defaults to True and applies within a single request's whole dependency tree, not just to one Depends() call site.",
  "verified_how": "Read fastapi/dependencies/utils.py (solve_dependencies, dependency_cache, sub_dependant_cache_key, _get_cache_key, use_cache default) and confirmed routing.py calls solve_dependencies for request handling; grepped marker hit counts under fastapi/.",
  "measured_marker_hits": {
   "use_cache": 53,
   "dependency_cache": 11,
   "_get_cache_key": 10
  }
 },
 {
  "id": "NT4",
  "repo": "sutra",
  "title": "Incremental re-index doesn't pick up signature/decorator changes",
  "text": "We updated a Python function's type hints and added a `@retry` decorator to it, then ran an incremental update over the repo. The function body text itself was untouched. After the run, pulling that function's stored record still shows the old signature and no decorator, as if the edit never happened. Deleting the repo's index and doing a full re-index from scratch picks up the new signature and decorator immediately, so extraction itself clearly handles it correctly \u2014 something about the incremental path specifically is skipping this symbol.",
  "gold": "In sutra/core/extractor/adapters/python.py, _build_function computes a symbol's body_hash from `source[body.start_byte:body.end_byte]` \u2014 i.e. only the function's body node, which does not include the def line, its type annotations, or its decorators. sutra/core/incremental_updater.py's _process_file compares old vs. new body_hash per moniker to split common symbols into changed_monikers vs unchanged_monikers, and only entries in changed_monikers are added to symbols_to_write and actually persisted (see the 'Write added + changed symbols' step). Since a decorator or signature edit alone doesn't change the body bytes, the symbol lands in unchanged_monikers, its freshly-extracted signature/decorators are computed but then discarded, and the stored record is never overwritten.",
  "gold_markers": [
   "changed_monikers",
   "_build_function",
   "body_bytes",
   "symbols_to_write"
  ],
  "marker_hit_counts": {
   "changed_monikers": 6,
   "_build_function": 7,
   "body_bytes": 20,
   "symbols_to_write": 4
  },
  "primary_paths": [
   "sutra/core/extractor/adapters/python.py",
   "sutra/core/incremental_updater.py"
  ],
  "repos_required": [
   "sutra"
  ],
  "why_this_is_hard": "The reporter already ruled out extraction (full re-index works), which points toward the incremental-diff logic, but the actual cause is upstream of that diff: it's a scoping choice in how body_hash is computed per symbol kind in the adapter. An agent must read _build_function in python.py to see body_hash is derived from only the body sub-node, then separately read _process_file in incremental_updater.py to see that body_hash equality is the sole gate deciding which freshly-extracted symbols get written vs silently discarded.",
  "verified_how": "Read sutra/core/extractor/adapters/python.py (_build_function, body_bytes/body_hash computation) and sutra/core/incremental_updater.py (_process_file: changed_monikers/unchanged_monikers split, symbols_to_write, the 'unchanged skipped entirely' comment); grepped marker hit counts across the repo's .py files.",
  "measured_marker_hits": {
   "changed_monikers": 6,
   "_build_function": 13,
   "body_bytes": 23,
   "symbols_to_write": 4
  }
 },
 {
  "id": "NT5",
  "repo": "pydantic",
  "title": "Combining two config mixins picks the wrong one's settings",
  "text": "We have two small base models used purely as config mixins: one sets `model_config = ConfigDict(extra='forbid')`, the other sets `model_config = ConfigDict(extra='allow')`. Our final model is declared as `class Foo(StrictMixin, LooseMixin): ...`, listing the strict one first because that's normally how attribute resolution with multiple inheritance works in Python. Instead, constructing `Foo` with an unexpected keyword argument doesn't raise a validation error \u2014 it's silently accepted, as if only the second mixin's config mattered.",
  "gold": "ModelMetaclass.__new__ (pydantic/_internal/_model_construction.py) builds a model's effective config via ConfigWrapper.for_model(bases, namespace, kwargs) (pydantic/_internal/_config.py). for_model iterates `for base in bases: config_new.update(base.model_config)` in the same left-to-right order the bases were declared in, then layers the class's own namespace/kwargs config on top. Because plain dict.update() lets each later base overwrite the same key from an earlier one, the *last*-listed base's config values win for any overlapping ConfigDict key \u2014 the opposite of normal Python MRO attribute lookup, where the first-listed base takes precedence. So with `class Foo(StrictMixin, LooseMixin)`, LooseMixin's extra='allow' overwrites StrictMixin's extra='forbid' in config_new even though StrictMixin is listed first.",
  "gold_markers": [
   "for_model",
   "config_new",
   "ModelMetaclass"
  ],
  "marker_hit_counts": {
   "for_model": 4,
   "config_new": 5,
   "ModelMetaclass": 23
  },
  "primary_paths": [
   "pydantic/_internal/_config.py",
   "pydantic/_internal/_model_construction.py"
  ],
  "repos_required": [
   "pydantic"
  ],
  "why_this_is_hard": "The reporter's mental model (first-listed base wins, matching normal Python MRO) is reasonable and matches how pydantic itself describes config priority in general terms, so the bug is counter-intuitive; pinning it down requires reading ModelMetaclass in _model_construction.py to see where class config gets assembled, then reading the actual base-iteration loop inside ConfigWrapper.for_model in _config.py closely enough to notice a plain dict.update() per base silently reverses precedence among multiple bases specifically.",
  "verified_how": "Read pydantic/_internal/_config.py (ConfigWrapper.for_model, the `for base in bases: config_new.update(...)` loop) and pydantic/_internal/_model_construction.py (ModelMetaclass.__new__ calling ConfigWrapper.for_model(bases, namespace, kwargs)); confirmed the update-order/last-base-wins behavior by static reading of the loop, and grepped marker hit counts under pydantic/.",
  "measured_marker_hits": {
   "for_model": 4,
   "config_new": 5,
   "ModelMetaclass": 34
  }
 }
]
const BUDGET = 6   // fixed by the calibration pilot; never tuned per class or ticket
const TRIALS = [1, 2, 3]
const BASE = '/Users/ritikshukla/Desktop/claude-dir/'
const INDEXED = {
  fastapi: 'fastapi/fastapi', pydantic: 'pydantic/pydantic', flask: 'pallets/flask',
  requests: 'psf/requests', sutra: 'ritikk112/sutra',
}

function repoList(t) { return t.repos_required && t.repos_required.length ? t.repos_required : [t.repo] }

function toolsBlock(t, arm) {
  if (arm === 'control') return '- Bash, Grep, Glob, Read\nUse only these tools.'
  const names = repoList(t).map(r => '"' + INDEXED[r] + '"').join(', ')
  return '- The sutra code-index MCP tools, which serve a pre-built index of ' +
    (repoList(t).length > 1 ? 'these repos (indexed as ' : 'this repo (indexed as ') + names + '): ' +
    'mcp__sutra__sutra_search, mcp__sutra__sutra_get_symbol, mcp__sutra__sutra_get_callers, ' +
    'mcp__sutra__sutra_get_callees, mcp__sutra__sutra_expand_neighbors, mcp__sutra__sutra_list_repos\n' +
    '- Bash, Grep, Glob, Read'
}

function solverPrompt(t, arm, cap, trial) {
  const paths = repoList(t).map(r => '- ' + r + ': ' + BASE + r).join('\n')
  let extra = ''
  if (repoList(t).includes('sutra')) {
    extra = '\nThis repository\'s markdown files contain notes that give away answers; base your answer on the source code only.'
  }
  return t.text + '\n\nRepo checkout(s):\n' + paths +
    '\n\n[trial marker: ' + t.id + '-' + arm + '-t' + trial + ']' +
    '\n\nTools available to you:\n' + toolsBlock(t, arm) +
    '\n\nYou have a hard budget of ' + cap + ' tool calls for this task. EVERY call to ANY tool ' +
    'counts against this budget, including code-index/MCP tools, Bash, Grep, Glob and Read. ' +
    'Count them as you go. Once you have used ' + cap + ' calls you must stop investigating ' +
    'immediately and answer with whatever you have established so far, even if incomplete. ' +
    'Exceeding the budget invalidates your answer. Loading tool definitions does not count ' +
    'against this budget.' +
    '\n\nAnswer the ticket. Name the specific mechanism in the source that explains it: the ' +
    'files, the functions or classes involved, and how they interact to produce the described ' +
    'behaviour. Be concrete and cite paths. Your final message is your answer — write it for an ' +
    'engineer who has not read this code.' + extra
}

const JUDGE_SCHEMA = {
  type: 'object',
  properties: {
    grades: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          answer_id: { type: 'string' },
          verdict: { type: 'string', enum: ['correct', 'partial', 'wrong'] },
          false_localization: { type: 'boolean' },
          claimed_call_sites: { type: 'integer', description: 'BR only: how many call sites the answer claims' },
          correct_call_sites: { type: 'integer', description: 'BR only: how many claimed sites are in the verified caller set' },
          missed_call_sites: { type: 'integer', description: 'BR only: verified sites the answer omitted' },
          decoys_claimed: { type: 'integer', description: 'BR only: decoys the answer wrongly claimed as call sites' },
          evidence: { type: 'string' },
        },
        required: ['answer_id', 'verdict', 'false_localization', 'evidence'],
      },
    },
    gold_dispute: { type: 'string' },
  },
  required: ['grades'],
}

phase('Solve')
log('Main run: ' + TICKETS.length + ' tickets x 2 arms x ' + TRIALS.length + ' trials at budget ' + BUDGET)

const results = await pipeline(
  TICKETS,
  async (t, _orig, idx) => {
    const jobs = []
    TRIALS.forEach(trial => {
      // arms interleaved rather than grouped, so neither arm systematically runs
      // in a busier slice of the machine
      ;['sutra', 'control'].forEach(arm => {
        jobs.push(() => agent(solverPrompt(t, arm, BUDGET, trial), {
          label: t.id + ':' + arm + ':t' + trial, phase: 'Solve', model: 'haiku',
        }).then(ans => ({ ticket: t.id, arm, cap: BUDGET, trial, answer: ans || '' })))
      })
    })
    const answers = await parallel(jobs)
    return { t, idx, answers: answers.filter(Boolean) }
  },
  async ({ t, idx, answers }) => {
    // deterministic rotation so no fixed position always holds the same arm
    const rot = idx % answers.length
    const shuffled = answers.map((_, i) => answers[(i + rot) % answers.length])
    const labels = 'ABCDEF'
    const block = shuffled.map((a, i) =>
      '### ANSWER ' + labels[i] + '\n' + (a.answer || '(no answer produced)')).join('\n\n')
    const grades = await agent(
      'You are grading answers to one engineering question. You will not be told how any ' +
      'answer was produced, and you must not speculate about it.\n\n' +
      '## The question that was asked\n' + t.text + '\n\n' +
      '## The verified correct answer (gold)\n' + t.gold + '\n\n' +
      'Relevant files: ' + (t.primary_paths || []).join(', ') + '\n' +
      (t.caller_set ? '\nVerified complete call site list:\n- ' + t.caller_set.join('\n- ') +
        '\nThese hits look similar but are NOT genuine call sites:\n- ' + (t.decoys || []).join('\n- ') + '\n' : '') +
      '\n## The answers to grade\n' + block + '\n\n' +
      '## How to grade\n' +
      'For each answer A-F give a verdict:\n' +
      '- correct: identifies the actual mechanism, right files/symbols, and the explanation of ' +
      'how it produces the behaviour is accurate.\n' +
      '- partial: finds part of the mechanism, or the right area with a materially wrong or ' +
      'missing explanation.\n' +
      '- wrong: wrong mechanism, or confidently asserts something the code does not do.\n\n' +
      'Also set false_localization=true when an answer names a specific wrong symbol or file ' +
      'CONFIDENTLY. Score this separately from wrong, because a confident wrong pointer sends ' +
      'an engineer to read the wrong file, whereas "I could not determine this" merely fails.\n\n' +
      'Rules you must follow:\n' +
      '1. Grade against the gold, not against the other answers. Do not curve. All six may be ' +
      'correct; all six may be wrong. A ticket where every answer is correct is a real and ' +
      'publishable result, not a sign you graded too leniently.\n' +
      '2. Longer is not better. An answer stating the right mechanism in two sentences beats one ' +
      'burying it in three paragraphs of adjacent detail.\n' +
      '3. Do not reward naming many plausible files. Precision counts.\n' +
      '4. If the gold itself looks wrong against the described code, say so in gold_dispute ' +
      'rather than forcing six wrong verdicts.\n\n' +
      (t.caller_set ? 'This is a call-site question: for each answer also fill claimed_call_sites, ' +
        'correct_call_sites, missed_call_sites and decoys_claimed against the verified list above. ' +
        'A decoy claimed as a real call site is a precision failure and should weigh against the verdict.\n\n' : '') +
      'Return the answer_id as the letter (A-F).',
      { label: 'judge:' + t.id, phase: 'Judge', schema: JUDGE_SCHEMA, model: 'sonnet' }
    )
    // map letters back to (arm, cap) - the judge never saw this mapping
    const key = {}
    shuffled.forEach((a, i) => { key[labels[i]] = { arm: a.arm, cap: a.cap } })
    const graded = (grades && grades.grades ? grades.grades : []).map(g => ({
      ticket: t.id, ...key[g.answer_id], verdict: g.verdict,
      false_localization: g.false_localization,
      claimed_call_sites: g.claimed_call_sites, correct_call_sites: g.correct_call_sites,
      missed_call_sites: g.missed_call_sites, decoys_claimed: g.decoys_claimed,
      evidence: g.evidence,
    }))
    return { ticket: t.id, graded, gold_dispute: grades && grades.gold_dispute }
  }
)

return { results: results.filter(Boolean) }
