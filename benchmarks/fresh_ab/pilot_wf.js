export const meta = {
  name: 'fresh-ab-pilot',
  description: 'Call-budget calibration pilot: find a cap where neither arm floors nor ceilings',
  phases: [
    { title: 'Solve', detail: '3 tickets x 3 caps x 2 arms = 18 solvers' },
    { title: 'Judge', detail: '1 blind judge per ticket over 6 shuffled answers' },
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
const CAPS = [3, 6, 12]
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
    '\n\nYou have a hard budget of ' + cap + ' tool calls for this task. Once you have used ' +
    cap + ' calls you must stop investigating and answer with whatever you have established ' +
    'so far. Budget your calls deliberately. Loading tool definitions does not count against ' +
    'this budget.' +
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
log('Pilot: 3 tickets x caps ' + CAPS.join('/') + ' x 2 arms')

const results = await pipeline(
  TICKETS,
  async (t, _orig, idx) => {
    const jobs = []
    CAPS.forEach((cap, ci) => {
      ;['sutra', 'control'].forEach(arm => {
        jobs.push(() => agent(solverPrompt(t, arm, cap, ci + 1), {
          label: t.id + ':' + arm + ':cap' + cap, phase: 'Solve', model: 'haiku',
        }).then(ans => ({ ticket: t.id, arm, cap, answer: ans || '' })))
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
      'Return the answer_id as the letter (A-F).',
      { label: 'judge:' + t.id, phase: 'Judge', schema: JUDGE_SCHEMA, model: 'sonnet' }
    )
    // map letters back to (arm, cap) - the judge never saw this mapping
    const key = {}
    shuffled.forEach((a, i) => { key[labels[i]] = { arm: a.arm, cap: a.cap } })
    const graded = (grades && grades.grades ? grades.grades : []).map(g => ({
      ticket: t.id, ...key[g.answer_id], verdict: g.verdict,
      false_localization: g.false_localization, evidence: g.evidence,
    }))
    return { ticket: t.id, graded, gold_dispute: grades && grades.gold_dispute }
  }
)

return { results: results.filter(Boolean) }
