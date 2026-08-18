export const meta = {
  name: 'sutra-single-abc-benchmark',
  description: 'Single-repo (sutra) A/B/C (SUTRA_ONLY/GREP_ONLY/BOTH) code-tracing benchmark: solve + blind-grade each cell, n=6 trials',
  phases: [{ title: 'Solve' }, { title: 'Grade' }],
}

// ---- config ----
const REPOS = {
  sutra: { dir: '/tmp/claude-1000/-home-ritik-Desktop-sutra/5d5da943-8e2c-467b-8fbe-07a9b7f411aa/scratchpad/sutra_src',
           sutra_repo: 'sutra',
           desc: 'the Sutra codebase (a code-index / retrieval engine: language extractors -> embedder -> Postgres graph + pgvector store -> an MCP retrieval pipeline; Python package under sutra/, with entrypoints under pipelines/)' },
}

const SOLVE_MODEL = 'claude-sonnet-5'
const GRADE_MODEL = 'claude-sonnet-5'

const TICKETS = {
  sutra: [
    { id:'SL1', text:'Trace how sutra re-scores its initial search candidates with a second-pass reranker model and applies the final top-k cutoff, identifying where candidates below the cutoff are dropped.' },
    { id:'SL2', text:'Trace how sutra builds the stable identifier (the "moniker") it uses to refer to a code symbol across re-indexes, and what disambiguates two symbols that share the same name.' },
    { id:'SL3', text:'Trace how sutra performs an incremental re-index of only the files that changed between two commits: how it computes the changed-file set and how it treats added vs modified vs deleted files.' },
    { id:'SS1', text:'Sutra runs several independent searches for one query — an exact-name lookup, a keyword search, and a vector-similarity search — each returning its own ranked list. Trace how those separate ranked lists get combined into the single ordering the caller finally sees.' },
    { id:'SS2', text:'A single sutra server has many repositories indexed at once. Trace what guarantees that a search asking about repository A can never return a symbol that actually belongs to repository B.' },
    { id:'SS3', text:'Sutra needs to link a function call in one file to where that function is actually defined in a different file. Some calls are ambiguous and can’t be linked by simple heuristics, so during indexing sutra brings in a heavier external program to work them out. Trace when that external program is started and how that step is turned on.' },
  ],
}

// gold "must contain" facts for blind grading (authored offline, verified vs grep + sutra index at 6a1a33a)
const GOLD = {
  'sutra|SL1': [
    'the heavier second pass is a cross-encoder run in rerank() (sutra/core/retrieval/reranker.py) that calls model.predict on (query, document) pairs to produce new scores',
    'after re-scoring, rerank() sorts by the new score descending and returns rescored[:top_k] (reranker.py:124-125) — the top_k slice is the final cutoff that drops lower-ranked candidates',
    'the reranker runs only when the per-query rerank=True flag is set in RetrievalPipeline.search (pipeline.py:101) and receives only the top DEFAULT_RERANK_CANDIDATES=50 of the fused list; with rerank=False the plain results[:top_k] cutoff applies instead',
  ],
  'sutra|SL2': [
    "the moniker is a space-delimited 5-field string 'sutra <language> <repo_name> <file_path> <descriptor>' built by MonikerBuilder._build (sutra/core/extractor/moniker.py); file_path being a field is what keeps same-named symbols in different files distinct",
    'the descriptor disambiguates by kind via a trailing suffix ((). function/method, # class, . variable, / module) plus a scope-chain prefix (from _prefix / _scope_chain) so nested/local vs top-level same-name symbols differ',
    'genuine same-scope same-name collisions are broken by _disambiguate (python.py) inserting (N) before the suffix for the Nth duplicate, with _dedup_keep_first (indexer.py) as a keep-first backstop',
  ],
  'sutra|SL3': [
    'the changed-file set is computed by changed_files() in sutra/core/git_differ.py running `git diff --name-status --diff-filter=ACDM --no-renames <old_sha>..<current_sha>`, where old_sha is the last indexed commit from the state store and current_sha is HEAD',
    'files are classified by git status letter (A/C->added, M->modified, D->deleted); added+modified are unioned and re-extracted/re-embedded while deleted files are processed separately to remove their symbols',
    '(edge/bonus) rename detection is disabled with --no-renames, so a renamed file is treated as a delete of the old path plus an add of the new path',
  ],
  'sutra|SS1': [
    'the three channel result lists (moniker/exact-name, BM25 keyword, vector similarity) are combined by Reciprocal Rank Fusion in rrf_fuse() (sutra/core/retrieval/fusion.py), invoked from RetrievalPipeline.search (pipeline.py:95)',
    'the fused score is the sum over channels of 1/(k + rank) with k=DEFAULT_RRF_K=60, and the rank is assigned by fusion itself (enumerate the channel’s sorted list), not read from the channel’s raw score',
    'raw channel scores are never compared across channels — only ordinal ranks enter the formula — and results are sorted by descending fused score (moniker tie-break)',
  ],
  'sutra|SS2': [
    'repo scope is enforced by selecting a per-repo ServingUnit from the SnapshotRegistry via registry.get(repo) at the MCP search entrypoint (server.py _unit / sutra_search) — there is no shared cross-repo index and no repo filter inside the query',
    'each ServingUnit’s RetrievalPipeline / vector store / channels is built from exactly one repo’s ArtifactSnapshot (build_serving_unit -> ArtifactLoader().load), and RetrievalPipeline.search takes no repo parameter — isolation is physical partitioning, not a query-time WHERE/filter',
    '(distinguishing fact) the vector store’s filter_monikers is a within-snapshot filter (not repo scoping) and sql_reader’s WHERE repo_name=%s is the write/incremental path, not the search read path',
  ],
  'sutra|SS3': [
    'cross-file/ambiguous call resolution via the external language server runs at INDEX time, kicked off from Indexer.index() calling self.resolver.resolve() (sutra/core/indexer.py:244)',
    "the external program is pyright: LspResolver spawns a pyright-langserver --stdio subprocess (subprocess.Popen in _LspClient, sutra/core/resolver/lsp_resolver.py:52), lazily and once per resolve() run (early-out when there is no unresolved work)",
    'it is opt-in via --resolver lsp (default heuristic never starts pyright); lsp mode wires ChainResolver(HeuristicResolver, LspResolver) so the heuristic runs first and the LSP only handles the calls it left unresolved',
  ],
}

const ARM_TOOLS = {
  SUTRA_ONLY: 'TOOLS ALLOWED: ONLY the Sutra MCP tools and the Read tool. First load the Sutra tools with ONE call: ToolSearch query "select:mcp__sutra__sutra_search,mcp__sutra__sutra_get_symbol,mcp__sutra__sutra_get_callers,mcp__sutra__sutra_get_callees,mcp__sutra__sutra_expand_neighbors". Navigate EXCLUSIVELY via those sutra_* tools (always pass repo="{SUTRA_REPO}"); use Read only to confirm a specific file:line. You are STRICTLY FORBIDDEN from using Bash, Grep, or Glob — do not call them for any reason.',
  GREP_ONLY:  'TOOLS ALLOWED: ONLY Bash, Grep, Glob, and Read. You are STRICTLY FORBIDDEN from using or loading ANY tool whose name contains "sutra" — treat them as nonexistent; never call ToolSearch to load them.',
  BOTH:       'TOOLS ALLOWED: any of — the Sutra MCP tools (load via ToolSearch "select:mcp__sutra__sutra_search,mcp__sutra__sutra_get_symbol,mcp__sutra__sutra_get_callers,mcp__sutra__sutra_get_callees,mcp__sutra__sutra_expand_neighbors"; repo="{SUTRA_REPO}"), Bash, Grep, Glob, and Read. Use whichever you prefer.',
}

const ANSWER_SCHEMA = { type:'object', additionalProperties:false,
  properties:{ answer:{ type:'string', description:'A concise root-cause trace answering the ticket, with concrete file:line references. Do NOT mention which tools you used.' } }, required:['answer'] }
const GRADE_SCHEMA = { type:'object', additionalProperties:false,
  properties:{ score:{ type:'integer', minimum:0, maximum:4 }, justification:{ type:'string', description:'one sentence' } }, required:['score','justification'] }

function solvePrompt(cell){
  const r = REPOS[cell.repo]
  const armTools = ARM_TOOLS[cell.arm].replace(/\{SUTRA_REPO\}/g, r.sutra_repo)
  return `You are a senior engineer investigating ${r.desc}. The repo is checked out at the exact indexed commit at: ${r.dir}. Work ONLY within that directory. (Never access any other copy of the repo on disk.)

${armTools}

TICKET: ${cell.ticket}

Investigate and produce a concise root-cause / trace answer backed by concrete file:line references. Be efficient. Return ONLY the answer via the structured output; do not describe your process or which tools you used.`
}

function gradePrompt(cell, candidate){
  const facts = (GOLD[`${cell.repo}|${cell.tid}`]||[]).map((f,i)=>`  ${i+1}. ${f}`).join('\n')
  return `You are blind-grading one candidate answer to a code-tracing ticket against ground truth. You do NOT know how the candidate was produced; judge ONLY correctness versus the ground truth.

TICKET: ${cell.ticket}

GROUND TRUTH — a correct answer must identify these facts:
${facts}

CANDIDATE ANSWER:
"""
${candidate}
"""

Score 0-4:
 0 = wrong or fabricated (names wrong files/mechanism, or invents things)
 1 = partially right but wrong root cause
 2 = right area, but mechanism vague or missing the key fact(s)
 3 = correct: right file(s):line and the correct mechanism (covers the ground-truth facts)
 4 = correct AND surfaces a real, accurate edge case or bug beyond the ground truth
Return score + a one-sentence justification.`
}

// ---- build cells: 6 tickets x 3 arms x 6 trials = 108 ----
const ARMS = ['SUTRA_ONLY','GREP_ONLY','BOTH']
const TRIALS = [1,2,3,4,5,6]
let CELLS = []
for (const repo of Object.keys(TICKETS))
  for (const t of TICKETS[repo])
    for (const arm of ARMS)
      for (const trial of TRIALS)
        CELLS.push({ id:`${repo}|${t.id}|${arm}|t${trial}`, repo, tid:t.id, arm, trial, ticket:t.text })

// parameterization via args
let A = (typeof args !== 'undefined' && args) ? args : {}
if (typeof A === 'string') { try { A = JSON.parse(A) } catch { A = {} } }
if (A.repos) CELLS = CELLS.filter(c => A.repos.includes(c.repo))
if (A.arms)  CELLS = CELLS.filter(c => A.arms.includes(c.arm))
if (A.trials) CELLS = CELLS.filter(c => A.trials.includes(c.trial))
if (A.tickets) CELLS = CELLS.filter(c => A.tickets.includes(c.tid))
if (A.only)  CELLS = CELLS.filter(c => A.only.includes(c.id))
if (A.limit) CELLS = CELLS.slice(0, A.limit)
const MODE = A.mode || 'both'   // 'solve' = solvers only; 'both' = solve+grade

log(`cells=${CELLS.length} mode=${MODE} solve_model=${SOLVE_MODEL}`)

const results = await pipeline(
  CELLS,
  (c) => agent(solvePrompt(c), { label:`solve|${c.id}`, phase:'Solve', schema:ANSWER_SCHEMA, model:SOLVE_MODEL, effort:'medium' })
           .then(r => ({ cell:c.id, answer: r ? r.answer : null })),
  (solveRes, c) => {
    if (MODE === 'solve' || !solveRes || !solveRes.answer) return { cell:c.id, answer: solveRes ? solveRes.answer : null, score:null }
    return agent(gradePrompt(c, solveRes.answer), { label:`grade|${c.id}`, phase:'Grade', schema:GRADE_SCHEMA, model:GRADE_MODEL, effort:'low' })
             .then(g => ({ cell:c.id, answer:solveRes.answer, score: g ? g.score : null, justification: g ? g.justification : null }))
  }
)

return results.filter(Boolean)
