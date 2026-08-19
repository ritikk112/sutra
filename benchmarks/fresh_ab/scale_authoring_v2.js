export const meta = {
  name: 'scale-ab-authoring-v2',
  description: 'Author blast-radius + disambiguation tickets across the corpus-size ladder, then verify',
  phases: [
    { title: 'Author', detail: '4 repo authors, 16 tickets' },
    { title: 'Verify', detail: 'independent verifier re-derives every gold' },
  ],
}

const PLAN = {
 "flask": {
  "size": 497,
  "blast_prefix": "FL",
  "disamb_prefix": "FD",
  "blast": [
   {
    "name": "record_once",
    "definitions": 1,
    "def_site": "src/flask/sansio/blueprints.py:233",
    "in_scope_calls": 10,
    "total_calls": 10
   },
   {
    "name": "_method_route",
    "definitions": 1,
    "def_site": "src/flask/sansio/scaffold.py:284",
    "in_scope_calls": 6,
    "total_calls": 7
   }
  ],
  "disamb": [
   {
    "name": "decorator",
    "definitions": 13,
    "distinct_files": 5,
    "def_sites": [
     "src/flask/cli.py:395",
     "src/flask/cli.py:422",
     "src/flask/helpers.py:120",
     "src/flask/sansio/blueprints.py:469",
     "src/flask/sansio/blueprints.py:525",
     "src/flask/sansio/blueprints.py:583",
     "src/flask/sansio/blueprints.py:663",
     "src/flask/sansio/app.py:692"
    ]
   }
  ]
 },
 "celery": {
  "size": 3601,
  "blast_prefix": "CE",
  "disamb_prefix": "CD",
  "blast": [
   {
    "name": "_ensure_retryable",
    "definitions": 1,
    "def_site": "celery/backends/base.py:636",
    "in_scope_calls": 9,
    "total_calls": 9
   },
   {
    "name": "_set_task_join_will_block",
    "definitions": 1,
    "def_site": "celery/_state.py:55",
    "in_scope_calls": 8,
    "total_calls": 10
   }
  ],
  "disamb": [
   {
    "name": "delete",
    "definitions": 16,
    "distinct_files": 15,
    "def_sites": [
     "celery/result.py:969",
     "celery/backends/elasticsearch.py:260",
     "celery/backends/arangodb.py:165",
     "celery/backends/dynamodb.py:538",
     "celery/backends/gcs.py:101",
     "celery/backends/filesystem.py:95",
     "celery/backends/cache.py:72",
     "celery/backends/cache.py:128"
    ]
   },
   {
    "name": "update",
    "definitions": 10,
    "distinct_files": 9,
    "def_sites": [
     "celery/platforms.py:688",
     "celery/result.py:636",
     "celery/beat.py:149",
     "celery/app/task.py:131",
     "celery/app/base.py:229",
     "celery/utils/graph.py:95",
     "celery/utils/collections.py:299",
     "celery/utils/collections.py:544"
    ]
   }
  ]
 },
 "django": {
  "size": 11010,
  "blast_prefix": "DJ",
  "disamb_prefix": "DD",
  "blast": [
   {
    "name": "units_name",
    "definitions": 1,
    "def_site": "django/contrib/gis/db/models/fields.py:132",
    "in_scope_calls": 6,
    "total_calls": 6
   },
   {
    "name": "_sqlite_datetime_parse",
    "definitions": 1,
    "def_site": "django/db/backends/sqlite3/_functions.py:119",
    "in_scope_calls": 6,
    "total_calls": 6
   }
  ],
  "disamb": [
   {
    "name": "as_oracle",
    "definitions": 39,
    "distinct_files": 13,
    "def_sites": [
     "django/db/models/expressions.py:1980",
     "django/db/models/lookups.py:149",
     "django/db/models/aggregates.py:271",
     "django/db/models/aggregates.py:383",
     "django/db/models/functions/mixins.py:35",
     "django/db/models/functions/comparison.py:60",
     "django/db/models/functions/comparison.py:88",
     "django/db/models/functions/comparison.py:169"
    ]
   },
   {
    "name": "as_mysql",
    "definitions": 18,
    "distinct_files": 10,
    "def_sites": [
     "django/db/models/expressions.py:1262",
     "django/db/models/aggregates.py:360",
     "django/db/models/aggregates.py:400",
     "django/db/models/functions/mixins.py:29",
     "django/db/models/functions/comparison.py:38",
     "django/db/models/functions/text.py:9",
     "django/db/models/functions/text.py:47",
     "django/db/models/functions/text.py:102"
    ]
   },
   {
    "name": "output_field",
    "definitions": 14,
    "distinct_files": 11,
    "def_sites": [
     "django/db/models/expressions.py:326",
     "django/db/models/lookups.py:168",
     "django/db/models/sql/query.py:334",
     "django/db/models/sql/where.py:293",
     "django/contrib/postgres/expressions.py:13",
     "django/contrib/gis/db/models/functions.py:118",
     "django/contrib/gis/db/models/functions.py:163",
     "django/contrib/gis/db/models/functions.py:305"
    ]
   }
  ]
 },
 "sqlalchemy": {
  "size": 12830,
  "blast_prefix": "SA",
  "disamb_prefix": "SD",
  "blast": [
   {
    "name": "detect_is_backref",
    "definitions": 1,
    "def_site": "lib/sqlalchemy/orm/util.py:272",
    "in_scope_calls": 7,
    "total_calls": 7
   },
   {
    "name": "inspect_formatargspec",
    "definitions": 1,
    "def_site": "lib/sqlalchemy/util/compat.py:187",
    "in_scope_calls": 7,
    "total_calls": 7
   }
  ],
  "disamb": [
   {
    "name": "fetchmany",
    "definitions": 19,
    "distinct_files": 7,
    "def_sites": [
     "lib/sqlalchemy/connectors/asyncio.py:103",
     "lib/sqlalchemy/connectors/asyncio.py:321",
     "lib/sqlalchemy/connectors/asyncio.py:345",
     "lib/sqlalchemy/engine/interfaces.py:216",
     "lib/sqlalchemy/engine/result.py:947",
     "lib/sqlalchemy/engine/result.py:1362",
     "lib/sqlalchemy/engine/result.py:1466",
     "lib/sqlalchemy/engine/result.py:1665"
    ]
   },
   {
    "name": "python_type",
    "definitions": 16,
    "distinct_files": 4,
    "def_sites": [
     "lib/sqlalchemy/sql/sqltypes.py:320",
     "lib/sqlalchemy/sql/sqltypes.py:413",
     "lib/sqlalchemy/sql/sqltypes.py:539",
     "lib/sqlalchemy/sql/sqltypes.py:907",
     "lib/sqlalchemy/sql/sqltypes.py:932",
     "lib/sqlalchemy/sql/sqltypes.py:977",
     "lib/sqlalchemy/sql/sqltypes.py:1029",
     "lib/sqlalchemy/sql/sqltypes.py:2016"
    ]
   }
  ]
 }
}
const BASE = '/Users/ritikshukla/Desktop/claude-dir/'

const COMMON = `
You are authoring tickets for a blind, controlled benchmark. Two arms of coding agents will
each try to solve them; one has an extra code-search tool, the other only Bash/Grep/Glob/Read.
You must NOT know or care which arm wins.

HARD RULES:
1. File each ticket the way a real engineer would: an observed behaviour or a concrete
   intent. Never mention indexes, search tools, grep, embeddings, or this benchmark, and
   never hint at how to find the answer.
2. Build every gold by READING THE CODE. Verify each claim before you emit it.
3. Do not put the answer in the ticket text.
4. Markers must be precise. Measure each with grep; if one matches more than ~50 places in
   the repo, pick a tighter one. A file path is often the best marker.
`

const TASK = (repo, p) => `${COMMON}
YOUR REPO: ${repo} at ${BASE}${repo}  (~${p.size} definitions in non-test source)

You will author TWO different kinds of ticket.

=== KIND 1: BLAST RADIUS (ids ${p.blast_prefix}1..${p.blast_prefix}${p.blast.length}) ===
Each asks what breaks if the target changes - renamed, given a new required parameter, or a
changed return type. Frame it as an engineer planning a refactor: "I need to add a required
argument to <X>; walk me through every place that calls it so I can update them."

Targets (use exactly these, in order):
${p.blast.map((t, i) => `  ${p.blast_prefix}${i + 1}  ${t.name}   defined once at ${t.def_site}, about ${t.in_scope_calls} call sites`).join('\n')}

Produce a 'caller_set': every genuine call site as "path/to/file.py:line". Build it by
grepping the name and READING each hit to classify it as a genuine call, an import, a
docstring/comment, a string literal, or the definition line. Put the non-calls in 'decoys'.

=== KIND 2: DISAMBIGUATION (ids ${p.disamb_prefix}1..${p.disamb_prefix}${p.disamb.length}) ===
This kind does NOT ask what calls something. It asks WHICH IMPLEMENTATION RUNS.

Each target below is a name defined many times across the repo - sibling implementations of
a shared interface. Write a ticket describing a concrete situation or observed behaviour,
and ask which specific implementation handles it and how it differs from its siblings.
For example: "we see <specific behaviour> when <specific condition>. Several parts of this
codebase implement <name>; which one is actually running in that path, and what does it do
that the others don't?"

Targets:
${p.disamb.map((t, i) => `  ${p.disamb_prefix}${i + 1}  ${t.name}   ${t.definitions} definitions across ${t.distinct_files} files
       e.g. ${(t.def_sites || []).slice(0, 5).join(', ')}`).join('\n')}

Choose ONE specific sibling as the answer and build the ticket's situation so that exactly
that one is correct. The gold must state: which definition (file:line and the owning
class/module), what condition selects it over the siblings, and what it does differently.
Set 'correct_implementation' to "path/to/file.py:line (OwningClass)" and 'sibling_count' to
the number of same-named definitions. List the other siblings in 'decoys'.

The ticket must be answerable from source alone and must NOT name the owning class, since
naming it would give away the disambiguation.

For EVERY ticket set 'task_kind' to 'blast_radius' or 'disambiguation', plus 'target_name',
'target_definitions', and 'corpus_size' (${p.size}).

If a target proves unusable, still emit the ticket but set 'unusable_reason'.`

const SCHEMA = {
  type: 'object',
  properties: {
    tickets: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' }, repo: { type: 'string' },
          task_kind: { type: 'string' }, title: { type: 'string' }, text: { type: 'string' },
          gold: { type: 'string' }, gold_markers: { type: 'array', items: { type: 'string' } },
          primary_paths: { type: 'array', items: { type: 'string' } },
          caller_set: { type: 'array', items: { type: 'string' } },
          decoys: { type: 'array', items: { type: 'string' } },
          correct_implementation: { type: 'string' },
          sibling_count: { type: 'integer' },
          target_name: { type: 'string' }, target_definitions: { type: 'integer' },
          corpus_size: { type: 'integer' }, unusable_reason: { type: 'string' },
          why_this_is_hard: { type: 'string' }, verified_how: { type: 'string' },
        },
        required: ['id', 'repo', 'task_kind', 'title', 'text', 'gold', 'gold_markers',
                   'primary_paths', 'verified_how'],
      },
    },
  },
  required: ['tickets'],
}

phase('Author')
log('Authoring 16 tickets: blast-radius + disambiguation across the size ladder')
const authored = await parallel(Object.keys(PLAN).map(r => () =>
  agent(TASK(r, PLAN[r]), { label: 'author:' + r, phase: 'Author', schema: SCHEMA, model: 'sonnet' })
))
const tickets = authored.filter(Boolean).flatMap(a => a.tickets || [])
log('authored ' + tickets.length + ': ' + tickets.map(t => t.id).join(', '))

phase('Verify')
const V_SCHEMA = {
  type: 'object',
  properties: {
    results: { type: 'array', items: { type: 'object', properties: {
      id: { type: 'string' },
      verdict: { type: 'string', enum: ['pass', 'repair', 'reject'] },
      markers_ok: { type: 'boolean' }, paths_ok: { type: 'boolean' },
      gold_accurate: { type: 'boolean' },
      caller_set_ok: { type: 'boolean' },
      implementation_correct: { type: 'boolean', description: 'disambiguation: is the named implementation really the one that runs' },
      answer_leaks: { type: 'boolean', description: 'does the ticket text give away the answer' },
      out_of_scope_sites: { type: 'integer', description: 'gold call sites in tests/examples/docs' },
      broad_markers: { type: 'array', items: { type: 'string' } },
      notes: { type: 'string' }, suggested_fix: { type: 'string' },
    }, required: ['id', 'verdict', 'markers_ok', 'paths_ok', 'gold_accurate', 'notes'] } },
    summary: { type: 'string' },
  },
  required: ['results', 'summary'],
}

const verified = await agent(
  `You are the independent verifier for a benchmark ticket set. You did NOT author these and
you must be skeptical. A broken ticket silently poisons a whole row of results.

Repos: flask ${BASE}flask, celery ${BASE}celery, django ${BASE}django,
sqlalchemy ${BASE}sqlalchemy.

Tickets as JSON:

${JSON.stringify(tickets, null, 1)}

Verify EVERY ticket by running commands:
1. Each gold_marker really appears. Markers that are FILE PATHS must be checked with
   ls/git ls-files, not by grepping file contents. Measure each marker's repo-wide hit
   count and list any above ~50 in broad_markers.
2. Each primary_path exists.
3. The gold is ACTUALLY WHAT THE CODE DOES. Read it and check. A plausible but wrong gold
   is the worst failure: it makes judges mark correct answers wrong.
4. BLAST RADIUS tickets: independently re-derive the caller set by grepping the name and
   reading every hit. Report missed or spurious sites. Count how many gold call sites sit
   in tests/, examples/ or docs/ and put that in out_of_scope_sites.
5. DISAMBIGUATION tickets: confirm the named implementation is genuinely the one that runs
   under the ticket's stated condition, and that the sibling implementations really are
   plausible alternatives. Set implementation_correct. This is the whole point of these
   tickets, so check it by reading the dispatch path, not by assuming.
6. Set answer_leaks=true if the ticket text names the owning class or otherwise gives away
   the answer.

Verdicts: pass = usable as-is; repair = fixable, say exactly how; reject = fundamentally
broken. Be strict.`,
  { label: 'verify:v2', phase: 'Verify', schema: V_SCHEMA, model: 'sonnet' })

return { tickets, verification: verified }
