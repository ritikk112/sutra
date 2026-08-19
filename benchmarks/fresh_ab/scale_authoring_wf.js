export const meta = {
  name: 'scale-ab-authoring',
  description: 'Author corpus-size + cross-repo tickets stratified by name ambiguity, and verify every gold',
  phases: [
    { title: 'Author', detail: '4 repo authors (blast radius) + 1 cross-repo author' },
    { title: 'Verify', detail: 'independent verifier re-derives every caller set' },
  ],
}

const SEL = {
 "flask": {
  "ambiguous": [
   {
    "name": "dispatch_request",
    "definitions": 4,
    "def_sites": [
     "src/flask/app.py:969",
     "src/flask/views.py:30",
     "src/flask/views.py:78",
     "src/flask/views.py:182"
    ],
    "call_hits": 3,
    "sample_hits": [
     "src/flask/app.py:995",
     "src/flask/app.py:1019",
     "src/flask/app.py:1600"
    ]
   },
   {
    "name": "wrapper",
    "definitions": 4,
    "def_sites": [
     "src/flask/app.py:87",
     "src/flask/app.py:99",
     "src/flask/ctx.py:201",
     "src/flask/sansio/blueprints.py:240"
    ],
    "call_hits": 7,
    "sample_hits": [
     "src/flask/app.py:93",
     "src/flask/app.py:107",
     "src/flask/cli.py:402",
     "src/flask/ctx.py:206",
     "src/flask/helpers.py:124",
     "src/flask/sansio/blueprints.py:244",
     "src/flask/sansio/scaffold.py:49"
    ]
   }
  ],
  "unique": [
   {
    "name": "init_db",
    "definitions": 1,
    "def_sites": [
     "examples/tutorial/flaskr/db.py:33"
    ],
    "call_hits": 7,
    "sample_hits": [
     "docs/appcontext.rst:66",
     "docs/patterns/sqlalchemy.rst:49",
     "docs/patterns/sqlalchemy.rst:91",
     "docs/patterns/sqlite3.rst:147",
     "docs/tutorial/database.rst:133",
     "examples/tutorial/flaskr/db.py:44",
     "src/flask/app.py:1496"
    ]
   },
   {
    "name": "from_prefixed_env",
    "definitions": 1,
    "def_sites": [
     "src/flask/config.py:126"
    ],
    "call_hits": 5,
    "sample_hits": [
     "CHANGES.rst:483",
     "docs/config.rst:597",
     "docs/lifecycle.rst:25",
     "docs/patterns/celery.rst:116",
     "examples/celery/src/task_app/__init__.py:16"
    ]
   }
  ],
  "size": 497,
  "prefix": "FL"
 },
 "celery": {
  "ambiguous": [
   {
    "name": "convert",
    "definitions": 10,
    "def_sites": [
     "celery/bin/worker.py:27",
     "celery/bin/worker.py:44",
     "celery/bin/worker.py:71",
     "celery/bin/worker.py:80",
     "celery/bin/base.py:235",
     "celery/bin/base.py:244"
    ],
    "call_hits": 4,
    "sample_hits": [
     "celery/app/utils.py:213",
     "celery/app/utils.py:217",
     "celery/bin/base.py:320",
     "celery/bin/worker.py:50"
    ]
   },
   {
    "name": "register_with_event_loop",
    "definitions": 8,
    "def_sites": [
     "celery/concurrency/asynpool.py:344",
     "celery/concurrency/asynpool.py:522",
     "celery/concurrency/prefork.py:135",
     "celery/concurrency/base.py:94",
     "celery/worker/worker.py:214",
     "celery/worker/components.py:177"
    ],
    "call_hits": 5,
    "sample_hits": [
     "celery/concurrency/asynpool.py:524",
     "celery/worker/components.py:178",
     "celery/worker/consumer/consumer.py:557",
     "celery/worker/loops.py:62",
     "celery/worker/loops.py:63"
    ]
   }
  ],
  "unique": [
   {
    "name": "in_sighandler",
    "definitions": 1,
    "def_sites": [
     "celery/utils/log.py:63"
    ],
    "call_hits": 8,
    "sample_hits": [
     "celery/apps/worker.py:302",
     "celery/apps/worker.py:480",
     "celery/apps/worker.py:497",
     "celery/apps/worker.py:507",
     "celery/apps/worker.py:520",
     "celery/utils/log.py:39",
     "celery/utils/log.py:65",
     "celery/utils/log.py:69"
    ]
   },
   {
    "name": "loader",
    "definitions": 1,
    "def_sites": [
     "celery/app/base.py:1587"
    ],
    "call_hits": 8,
    "sample_hits": [
     "celery/app/base.py:357",
     "celery/app/base.py:436",
     "celery/apps/beat.py:82",
     "celery/apps/beat.py:129",
     "celery/utils/imports.py:113",
     "docs/history/changelog-1.0.rst:778",
     "docs/history/whatsnew-3.1.rst:363",
     "docs/history/whatsnew-4.0.rst:2130"
    ]
   }
  ],
  "size": 3601,
  "prefix": "CE"
 },
 "django": {
  "ambiguous": [
   {
    "name": "aexists",
    "definitions": 7,
    "def_sites": [
     "django/db/models/query.py:1505",
     "django/contrib/sessions/backends/signed_cookies.py:59",
     "django/contrib/sessions/backends/db.py:65",
     "django/contrib/sessions/backends/cached_db.py:81",
     "django/contrib/sessions/backends/cache.py:120",
     "django/contrib/sessions/backends/file.py:192"
    ],
    "call_hits": 6,
    "sample_hits": [
     "django/contrib/sessions/backends/base.py:207",
     "django/contrib/sessions/backends/cached_db.py:85",
     "django/contrib/sessions/backends/db.py:66",
     "docs/ref/models/querysets.txt:2873",
     "docs/ref/models/querysets.txt:2875",
     "docs/topics/http/sessions.txt:795"
    ]
   },
   {
    "name": "get_new_connection",
    "definitions": 7,
    "def_sites": [
     "django/db/backends/postgresql/base.py:318",
     "django/db/backends/oracle/base.py:301",
     "django/db/backends/sqlite3/base.py:204",
     "django/db/backends/mysql/base.py:254",
     "django/db/backends/base/base.py:215",
     "django/contrib/gis/db/backends/postgis/base.py:125"
    ],
    "call_hits": 4,
    "sample_hits": [
     "django/contrib/gis/db/backends/postgis/base.py:126",
     "django/contrib/gis/db/backends/spatialite/base.py:41",
     "django/db/backends/base/base.py:218",
     "django/db/backends/base/base.py:256"
    ]
   }
  ],
  "unique": [
   {
    "name": "topology_func",
    "definitions": 1,
    "def_sites": [
     "django/contrib/gis/gdal/prototypes/geom.py:32"
    ],
    "call_hits": 8,
    "sample_hits": [
     "django/contrib/gis/gdal/prototypes/geom.py:151",
     "django/contrib/gis/gdal/prototypes/geom.py:152",
     "django/contrib/gis/gdal/prototypes/geom.py:153",
     "django/contrib/gis/gdal/prototypes/geom.py:154",
     "django/contrib/gis/gdal/prototypes/geom.py:155",
     "django/contrib/gis/gdal/prototypes/geom.py:156",
     "django/contrib/gis/gdal/prototypes/geom.py:157",
     "django/contrib/gis/gdal/prototypes/geom.py:158"
    ]
   },
   {
    "name": "user_passes_test",
    "definitions": 1,
    "def_sites": [
     "django/contrib/auth/decorators.py:13"
    ],
    "call_hits": 7,
    "sample_hits": [
     "django/contrib/admin/views/decorators.py:12",
     "django/contrib/auth/decorators.py:81",
     "django/contrib/auth/decorators.py:136",
     "docs/topics/auth/default.txt:695",
     "docs/topics/auth/default.txt:707",
     "docs/topics/auth/default.txt:734",
     "tests/urlpatterns_reverse/views.py:50"
    ]
   }
  ],
  "size": 11010,
  "prefix": "DJ"
 },
 "sqlalchemy": {
  "ambiguous": [
   {
    "name": "_accept_with",
    "definitions": 11,
    "def_sites": [
     "lib/sqlalchemy/orm/events.py:103",
     "lib/sqlalchemy/orm/events.py:265",
     "lib/sqlalchemy/orm/events.py:806",
     "lib/sqlalchemy/orm/events.py:1582",
     "lib/sqlalchemy/orm/events.py:2485",
     "lib/sqlalchemy/orm/events.py:3277"
    ],
    "call_hits": 4,
    "sample_hits": [
     "lib/sqlalchemy/engine/events.py:139",
     "lib/sqlalchemy/event/api.py:30",
     "lib/sqlalchemy/orm/events.py:193",
     "lib/sqlalchemy/orm/events.py:1609"
    ]
   },
   {
    "name": "partitions",
    "definitions": 8,
    "def_sites": [
     "lib/sqlalchemy/engine/result.py:862",
     "lib/sqlalchemy/engine/result.py:1339",
     "lib/sqlalchemy/engine/result.py:1440",
     "lib/sqlalchemy/engine/result.py:1625",
     "lib/sqlalchemy/ext/asyncio/result.py:188",
     "lib/sqlalchemy/ext/asyncio/result.py:562"
    ],
    "call_hits": 5,
    "sample_hits": [
     "doc/build/changelog/migration_14.rst:1358",
     "doc/build/core/connections.rst:743",
     "doc/build/orm/queryguide/api.rst:246",
     "lib/sqlalchemy/ext/asyncio/result.py:198",
     "test/typing/plain_files/engine/engine_result.py:44"
    ]
   }
  ],
  "unique": [
   {
    "name": "CompileState",
    "definitions": 1,
    "def_sites": [
     "lib/sqlalchemy/sql/base.py:727"
    ],
    "call_hits": 8,
    "sample_hits": [
     "lib/sqlalchemy/orm/bulk_persistence.py:665",
     "lib/sqlalchemy/orm/context.py:233",
     "lib/sqlalchemy/orm/context.py:341",
     "lib/sqlalchemy/orm/context.py:386",
     "lib/sqlalchemy/orm/context.py:774",
     "lib/sqlalchemy/orm/context.py:1082",
     "lib/sqlalchemy/orm/context.py:1089",
     "lib/sqlalchemy/sql/base.py:867"
    ]
   },
   {
    "name": "common_parent",
    "definitions": 1,
    "def_sites": [
     "lib/sqlalchemy/orm/mapper.py:3326"
    ],
    "call_hits": 6,
    "sample_hits": [
     "lib/sqlalchemy/orm/context.py:2319",
     "lib/sqlalchemy/orm/context.py:3405",
     "lib/sqlalchemy/orm/relationships.py:1683",
     "lib/sqlalchemy/orm/relationships.py:2259",
     "lib/sqlalchemy/orm/relationships.py:3267",
     "lib/sqlalchemy/orm/util.py:2139"
    ]
   }
  ],
  "size": 12830,
  "prefix": "SA"
 }
}
const BASE = '/Users/ritikshukla/Desktop/claude-dir/'

const COMMON = `
You are authoring tickets for a blind, controlled benchmark. Two arms of coding agents will
each try to solve them; one arm has an extra code-search tool, the other has only
Bash/Grep/Glob/Read. You must NOT know or care which arm wins.

HARD RULES:
1. Write each ticket the way a real engineer would file it: an observed behaviour or a
   concrete refactoring intent. Never mention indexes, search tools, grep, embeddings, or
   this benchmark, and never hint at how to find the answer.
2. Build every gold by READING THE CODE (Grep/Read/Bash). Verify each claim before emitting.
3. Tickets must be answerable from source alone. Ignore any markdown/docs answer keys.
4. Do not put the answer in the ticket text.
`

const BR_TASK = (repo, size, prefix, targets) => `${COMMON}
YOUR REPO: ${repo} at ${BASE}${repo}  (~${size} definitions in non-test source)

Author exactly 4 "blast radius" tickets, ids ${prefix}1..${prefix}4, one per target below.
Each ticket asks what would break if the target symbol changed - renamed, given a new
required parameter, or a changed return type. Frame it as an engineer planning a refactor:
"I need to add a required argument to <X>. Walk me through every place that calls it so I
can update them - which call sites are there and what would each need?"

THE TARGETS (pre-selected; use exactly these, in this order):
${targets.map((t, i) => `  ${prefix}${i + 1}  ${t.name}  [${t.stratum}] defined ${t.definitions}x in this repo, ~${t.call_hits} call-ish hits
       definition sites: ${(t.def_sites || []).join(', ')}`).join('\n')}

For each ticket produce a 'caller_set': the COMPLETE list of genuine call sites of THE
SPECIFIC target definition, each as "path/to/file.py:line". Build it by grepping the name
and READING every hit to decide whether it is a genuine call to THIS symbol, or one of:
a call to a DIFFERENT same-named symbol, an import, a docstring/comment, a string literal,
or the definition line itself. Record those in 'decoys'.

Two of your targets have a name defined ONCE in the repo, and two have a name defined many
times. For the multiply-defined ones, the other definitions ARE the decoys and you must
list them explicitly - identifying which definition the call sites belong to is the entire
difficulty of the ticket, so be precise about which class/module owns the target.

Set 'stratum' to 'unique' or 'ambiguous' exactly as given above, and 'target_name',
'target_definitions', 'corpus_size' (${size}) on every ticket.

If a target turns out to be unusable (e.g. its call sites are untraceable, or it is a
generic decorator idiom with no meaningful architecture), still emit the ticket but set
'unusable_reason' explaining why, so it can be dropped rather than silently weakening the
set.`

const XR_TASK = `${COMMON}
YOUR TASK: 4 cross-repo tickets, ids XL1..XL4, spanning THREE LARGE codebases:
  - django      ${BASE}django       (~11000 definitions)
  - celery      ${BASE}celery       (~3600 definitions)
  - sqlalchemy  ${BASE}sqlalchemy   (~12800 definitions)

These three are a realistic production stack: a web framework, a task queue, and an ORM.
Each ticket asks ONE question whose answer requires finding and COMPARING the corresponding
mechanism in all three. Frame it as an engineer doing a design review, a migration, or
debugging an interaction between the three: "we're standardising how our services handle
<X> - how does each of these do it today, and where do they actually diverge?"

Pick mechanisms that genuinely exist in all three but are implemented differently and NAMED
DIFFERENTLY in each - divergent vocabulary is the crux. If all three call it the same thing
the ticket is too easy. Candidate themes (verify before using, and choose your own if these
do not hold up): connection/resource pooling and lifecycle, retry and backoff, transaction
or task atomicity, configuration and settings resolution, signal/hook/event dispatch,
serialization of task or query payloads, lazy loading and deferred evaluation.

The gold must state the mechanism PER REPO with file paths, and gold_markers must include
at least one distinctive marker from EACH of the three repos. Set 'repos_required' to the
three repo names. Prefer markers that are precise: a marker matching more than ~50 places
in its repo is too broad, so measure with grep and pick a tighter one.`

const SCHEMA = {
  type: 'object',
  properties: {
    tickets: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          repo: { type: 'string' },
          repos_required: { type: 'array', items: { type: 'string' } },
          title: { type: 'string' },
          text: { type: 'string' },
          gold: { type: 'string' },
          gold_markers: { type: 'array', items: { type: 'string' } },
          primary_paths: { type: 'array', items: { type: 'string' } },
          caller_set: { type: 'array', items: { type: 'string' } },
          decoys: { type: 'array', items: { type: 'string' } },
          stratum: { type: 'string' },
          target_name: { type: 'string' },
          target_definitions: { type: 'integer' },
          corpus_size: { type: 'integer' },
          unusable_reason: { type: 'string' },
          why_this_is_hard: { type: 'string' },
          verified_how: { type: 'string' },
        },
        required: ['id', 'repo', 'title', 'text', 'gold', 'gold_markers', 'primary_paths', 'verified_how'],
      },
    },
  },
  required: ['tickets'],
}

phase('Author')
const repos = Object.keys(SEL)
log('Authoring 16 blast-radius tickets across the size ladder + 4 cross-repo tickets')

const jobs = repos.map(r => () => {
  const s = SEL[r]
  const targets = [
    { ...s.unique[0], stratum: 'unique' },
    { ...s.unique[1], stratum: 'unique' },
    { ...s.ambiguous[0], stratum: 'ambiguous' },
    { ...s.ambiguous[1], stratum: 'ambiguous' },
  ]
  return agent(BR_TASK(r, s.size, s.prefix, targets),
    { label: 'author:' + r, phase: 'Author', schema: SCHEMA, model: 'sonnet' })
})
jobs.push(() => agent(XR_TASK, { label: 'author:cross-repo-large', phase: 'Author', schema: SCHEMA, model: 'sonnet' }))

const authored = await parallel(jobs)
const tickets = authored.filter(Boolean).flatMap(a => a.tickets || [])
log('authored ' + tickets.length + ' tickets: ' + tickets.map(t => t.id).join(', '))

phase('Verify')
const V_SCHEMA = {
  type: 'object',
  properties: {
    results: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          verdict: { type: 'string', enum: ['pass', 'repair', 'reject'] },
          markers_ok: { type: 'boolean' },
          paths_ok: { type: 'boolean' },
          gold_accurate: { type: 'boolean' },
          caller_set_ok: { type: 'boolean' },
          caller_set_notes: { type: 'string' },
          missed_call_sites: { type: 'array', items: { type: 'string' } },
          spurious_call_sites: { type: 'array', items: { type: 'string' } },
          broad_markers: { type: 'array', items: { type: 'string' } },
          notes: { type: 'string' },
          suggested_fix: { type: 'string' },
        },
        required: ['id', 'verdict', 'markers_ok', 'paths_ok', 'gold_accurate', 'notes'],
      },
    },
    summary: { type: 'string' },
  },
  required: ['results', 'summary'],
}

const verified = await agent(
  `You are the independent verifier for a benchmark ticket set. You did NOT author these and
you must be skeptical. A broken ticket silently poisons a whole row of results.

Repos: django ${BASE}django, celery ${BASE}celery, sqlalchemy ${BASE}sqlalchemy,
flask ${BASE}flask.

Tickets as JSON:

${'${JSON.stringify(tickets, null, 1)}'}

For EVERY ticket, verify by running commands:
1. Each gold_marker appears in the stated repo. Markers that are FILE BASENAMES must be
   checked with ls/git ls-files, not by grepping file contents - do not report a path as
   missing merely because it does not appear inside a file. Also measure each marker's
   repo-wide hit count and list any exceeding ~50 in 'broad_markers': an over-broad marker
   silently credits an arm for incidental mentions.
2. Each primary_path exists.
3. The stated gold is ACTUALLY WHAT THE CODE DOES. Read it and check. A plausible-sounding
   but wrong gold is the worst failure mode - it makes judges mark correct answers wrong.
4. For blast-radius tickets, INDEPENDENTLY re-derive the caller set: grep the target name,
   read every hit, and decide genuine call / different same-named symbol / import /
   docstring / definition. Report anything the author MISSED and anything SPURIOUS. For
   'ambiguous' tickets be especially careful that every listed call site really belongs to
   the target definition and not to a same-named sibling - that distinction is the whole
   point of those tickets.
5. Flag any ticket carrying 'unusable_reason', and say whether you agree it should be
   dropped.

Verdicts: pass = usable as-is; repair = fixable, say exactly how; reject = fundamentally
broken. Be strict: fixing a ticket now is far cheaper than discovering it was broken after
120 solver runs.`,
  { label: 'verify:scale-golds', phase: 'Verify', schema: V_SCHEMA, model: 'sonnet' }
)

return { tickets, verification: verified }
